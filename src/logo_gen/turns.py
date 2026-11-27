"""Designer replies run on the server, not in the browser's request.

A turn is an asyncio task that streams the model's reply into the assistant
message row. Text and reasoning are saved to SQLite as they arrive (at most
every SAVE_EVERY seconds, and at the end), so a page that reloads or comes
back later sees what's there so far. Each turn also keeps an in-memory log
of events. A browser follows it over SSE from a cursor (the event id), and
reconnects where it left off with Last-Event-ID. Leaving the page closes
the SSE connection but doesn't touch the task.

Before this, the reply was generated inside the POST request's streaming
response. Navigating away or reloading closed that response, which
cancelled the model call, and nothing of the reply was saved.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from logo_gen import chats, library, paths, prompt_engine
from logo_gen.config import get_settings

SAVE_EVERY = 0.5
KEEP_FINISHED = 120.0  # seconds a finished turn's event log stays in memory for late subscribers


@dataclass
class Turn:
    chat_id: str
    message_id: int
    content: str = ""
    reasoning: str = ""
    status: str = "streaming"
    error: str = ""
    events: list[dict] = field(default_factory=list)
    task: asyncio.Task | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    saved_at: float = 0.0

    def emit(self, event: dict) -> None:
        self.events.append(event | {"id": len(self.events) + 1})
        self.changed.set()
        self.changed = asyncio.Event()

    @property
    def finished(self) -> bool:
        return self.status != "streaming"

    def snapshot(self) -> dict:
        return {
            "type": "snapshot",
            "id": len(self.events),
            "content": self.content,
            "reasoning": self.reasoning,
            "status": self.status,
            "error": self.error,
        }


_turns: dict[int, Turn] = {}


def get(message_id: int) -> Turn | None:
    return _turns.get(message_id)


def running_in(chat_id: str) -> Turn | None:
    for t in _turns.values():
        if t.chat_id == chat_id and not t.finished:
            return t
    return None


def start(chat_id: str, parent_id: int) -> int:
    """Add an empty assistant reply under parent_id and start writing it. Returns its message id."""
    if running_in(chat_id):
        raise TurnBusy()
    history = chats.history_for(parent_id)
    model = prompt_engine.backend_label()
    mid = chats.add_message(chat_id, "assistant", "", parent_id, status="streaming", model=model)
    turn = Turn(chat_id=chat_id, message_id=mid)
    _turns[mid] = turn
    turn.emit({"type": "start", "model": model})
    turn.task = asyncio.get_running_loop().create_task(_run(turn, history))
    return mid


class TurnBusy(Exception):
    message = "The designer is still answering in this chat. Stop it first, or wait for it to finish."


def stop(message_id: int) -> bool:
    turn = _turns.get(message_id)
    if not turn or turn.finished or not turn.task:
        return False
    turn.task.cancel()
    return True


def _save(turn: Turn, force: bool = False) -> None:
    now = time.time()
    if not force and now - turn.saved_at < SAVE_EVERY:
        return
    turn.saved_at = now
    chats.update_message(turn.message_id, content=turn.content, reasoning=turn.reasoning)


async def _run(turn: Turn, history: list[dict]) -> None:
    from logo_gen.server import _error_text  # error wording shared with the API

    try:
        vision = await prompt_engine.chat_supports_images()
        ask_first = chats.asks_first(turn.chat_id)
        # The system prompt carries what this chat's reactions taught the designer.
        messages = prompt_engine.build_history(history, paths.resolve_output, vision, chat_id=turn.chat_id, ask_first=ask_first)
        async for ev in prompt_engine.stream_reply(messages):
            kind = ev.get("type")
            if kind == "token":
                turn.content += ev["content"]
                turn.emit({"type": "token", "content": ev["content"]})
            elif kind == "reasoning":
                text = ev.get("content", "")
                if text:
                    turn.reasoning += text
                    turn.emit({"type": "reasoning", "content": text})
            elif kind == "usage":
                rec = prompt_engine.cost_record(ev, get_settings().chat_model)
                library.add_cost("chat", chat_id=turn.chat_id, message_id=turn.message_id, **rec)
                turn.emit({"type": "usage", "credits": rec["credits"], "byok": rec["byok"]})
            _save(turn)
        prompts, feedback = prompt_engine.parse_reply(turn.content)
        if feedback:  # "more of / less of" phrases the designer pulled from the user's message
            prompt_engine.learn_from_reply(turn.chat_id, turn.content)
        shown = prompt_engine.strip_prompt_block(turn.content) if prompts or feedback else turn.content
        if prompts and not ask_first:
            # Act first: render the directions now, as a server job linked to this reply, so the
            # images show up under it even if nobody is watching.
            from logo_gen.server import auto_round

            try:
                line = await auto_round(turn.chat_id, turn.message_id, prompts, history)
            except Exception as e:  # noqa: BLE001  (the reply itself is fine; say why nothing rendered)
                line = f"Couldn't start rendering: {_error_text(e)} Press Generate to try again."
            shown = f"{shown}\n\n_{line}_"
        turn.status = "done"
        chats.update_message(
            turn.message_id, content=shown, reasoning=turn.reasoning, prompts=prompts, status="done", finished_at=time.time()
        )
        turn.emit({"type": "done", "status": "done", "content": shown, "prompts": prompts})
    except asyncio.CancelledError:
        turn.status = "stopped"
        chats.update_message(
            turn.message_id, content=turn.content, reasoning=turn.reasoning, status="stopped", finished_at=time.time()
        )
        turn.emit({"type": "done", "status": "stopped", "content": turn.content})
    except Exception as e:  # noqa: BLE001
        turn.status = "error"
        turn.error = _error_text(e)
        chats.update_message(
            turn.message_id, content=turn.content, reasoning=turn.reasoning, status="error", error=turn.error,
            finished_at=time.time(),
        )
        turn.emit({"type": "error", "status": "error", "message": turn.error})
    finally:
        chats.touch(turn.chat_id)
        asyncio.get_running_loop().call_later(KEEP_FINISHED, lambda: _turns.pop(turn.message_id, None))


async def follow(message_id: int, after: int | None) -> AsyncIterator[dict]:
    """Events for one reply, starting after event id `after`.

    Without a usable cursor it starts with a snapshot of everything so far.
    For a reply that isn't running in this process (finished long ago, or the
    server restarted) it sends a snapshot from the database and ends.
    """
    turn = _turns.get(message_id)
    if not turn:
        m = chats.get_message(message_id)
        if not m:
            yield {"type": "error", "status": "error", "message": "That message no longer exists."}
            return
        yield {"type": "snapshot", "id": 0, "content": m["content"], "reasoning": m["reasoning"], "status": m["status"],
               "error": m["error"]}
        yield {"type": "end", "status": m["status"]}
        return
    if after is None or after < 0 or after > len(turn.events):
        yield turn.snapshot()
        i = len(turn.events)
    else:
        i = after
    while True:
        while i < len(turn.events):
            yield turn.events[i]
            i += 1
        if turn.finished:
            yield {"type": "end", "status": turn.status}
            return
        waiter = turn.changed
        try:
            await asyncio.wait_for(waiter.wait(), timeout=15)
        except asyncio.TimeoutError:
            yield {"type": "ping"}


def recover() -> None:
    """At startup: replies that were streaming when the server stopped can't continue."""
    for m in chats.running_turns():
        chats.update_message(
            m["id"], status="error", error="The server stopped before this reply finished. Regenerate to try again.",
            finished_at=time.time(),
        )
