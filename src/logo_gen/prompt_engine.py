"""Prompt writing: the chat designer, one-shot directions, and variations."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import AsyncIterator

from logo_gen.clients import claude_cli, local_llm, openrouter
from logo_gen.config import get_settings

from logo_gen.prompting import brief as briefs
from logo_gen.prompting import preferences

_FIELD_LINES = "\n".join(f"- {k}: {v}" for k, v in briefs.FIELDS.items())

BRIEF_RULES = f"""\
You write design briefs, not image prompts. A separate step turns each brief into the \
prompt each image model reads best, so leave out prompt tricks, quality words and \
model names. Each direction is a JSON object with these fields:
{_FIELD_LINES}

What makes a good brief:
- One idea per direction, built from a few concrete shapes. It has to read at 16 to 32 px.
- The subject says what is drawn and how it is built, not the mood ("a leaf made of two \
overlapping circles", not "a dynamic, organic feel"). Keep it under 25 words.
- Describe shapes in plain words. No percentages, measurements or unit counts: image \
models can't follow "corner radius 10 percent" or "gaps of one unit".
- Keep the meaning out of the subject. Image models draw every noun they get, so \
"an L ending in a dot, reading as a ledger line" comes back as a ledger. Put the idea \
in your one-line summary instead.
- text holds the exact wording only for wordmark, lettermark, combination or emblem marks. \
Keep it to the name; no taglines. Use an empty string for symbols.
- 1 to 3 colors with hex values that fit the brand, with enough contrast to work in one color too.
- avoid lists specific things that would hurt this brief, not generic ones."""

DESIGNER_SYSTEM = f"""\
You are a senior brand designer. You help the user find a logo through a short conversation.

Before proposing directions you need: the name, what it is (product or industry), who it \
is for, two or three personality words, the mark type (symbol, lettermark, wordmark, \
combination, emblem), colors to use or avoid, and where it will be used (app icon, \
favicon, website, print). If the user gave most of this, don't ask: fill the gaps with \
sensible defaults and say in one line what you assumed. Otherwise ask at most 4 short \
questions in one message, each with a suggested default, so the user can just say "go".

When you have enough, or the user asks to generate, propose 4 directions (or the number \
they ask for). Make them different ideas: a different subject or construction, and \
different mark types where it makes sense. Not one idea in four colors.

When the user reacts to results, say in one sentence what you will change, then propose \
new directions. If they attach reference images, the image models see them too: say what \
to keep and what to change instead of describing them.

Be brief and concrete. No flattery.

{BRIEF_RULES}

Reply format: a one-line summary per direction, then exactly one fenced JSON block:

```json
{{"directions": [{{"concept": "...", "mark_type": "...", "format": "...", "subject": "...", "composition": "...", \
"shape_language": "...", "style": "...", "palette": [{{"name": "...", "hex": "#RRGGBB"}}], "background": "white", \
"text": "", "avoid": ["..."]}}], "feedback": {{"more": [], "less": []}}}}
```

feedback holds short phrases from the user's own reactions since your last reply, such as \
"more": ["thicker strokes"] or "less": ["gradients"]. Leave both lists empty if there were none. \
A reply that only asks questions has no JSON block, unless the user gave feedback; then \
send a block with only "feedback"."""

DIRECTIONS_SYSTEM = f"""\
You are a senior brand designer. Given a short brief, return distinct logo directions. Each \
one must be a different idea (a different subject or construction), not a recolor. Where \
the brief leaves things open, choose what a good designer would and stay consistent with it.

{BRIEF_RULES}"""

VARIATION_SYSTEM = f"""\
You are a senior brand designer. The user picked a logo and wants variations of it. Keep \
the core idea of the original brief and change one thing per variation: proportions, \
stroke weight, palette, level of detail, or a small twist on the subject. If the user gave \
an instruction, follow it in every variation.

{BRIEF_RULES}"""

BRIEF_SCHEMA = briefs.json_schema()

# ── Parsing ──────────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def direction_from_brief(d: dict) -> dict:
    """A brief from the chat model as a direction: the readable fields the UI shows, plus the brief."""
    b = briefs.from_dict(d)
    return {"concept": b.concept, "style": b.style, "prompt": b.summary(), "brief": b.to_dict()}


def _clean_prompts(items: object) -> list[dict]:
    """Directions from a list of briefs, or of old-style {"prompt": ...} entries."""
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        if briefs.is_brief(it):
            out.append(direction_from_brief(it))
        elif str(it.get("prompt") or "").strip():
            out.append({
                "concept": str(it.get("concept") or "").strip() or "Direction",
                "style": str(it.get("style") or "").strip(),
                "prompt": str(it["prompt"]).strip(),
            })
    return out


def _blocks(text: str) -> list[dict]:
    candidates = _FENCE.findall(text)
    if not candidates:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidates = [text[start : end + 1]]
    out = []
    for raw in reversed(candidates):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out.append(data)
        elif isinstance(data, list):
            out.append({"directions": data})
    return out


def parse_reply(text: str) -> tuple[list[dict], dict]:
    """(directions, feedback) from a reply. Uses the last block that has them."""
    directions: list[dict] = []
    feedback: dict = {}
    for data in _blocks(text):
        if not directions:
            directions = _clean_prompts(data.get("directions") or data.get("prompts"))
        fb = data.get("feedback")
        if not feedback and isinstance(fb, dict):
            feedback = {
                k: [str(x).strip() for x in (fb.get(k) or []) if str(x).strip()][:6]
                for k in ("more", "less")
            }
    return directions, feedback


def parse_prompts(text: str) -> list[dict]:
    return parse_reply(text)[0]


def strip_prompt_block(text: str) -> str:
    """Remove the JSON block from a reply for display."""
    return _FENCE.sub("", text).strip()


def learn_from_reply(chat_id: str, text: str) -> dict:
    """Store the feedback phrases the designer pulled from the user's messages."""
    _, fb = parse_reply(text)
    if fb.get("more") or fb.get("less"):
        preferences.add_notes(chat_id, fb.get("more"), fb.get("less"))
    return fb


# ── Messages ─────────────────────────────────────────────────────────


def _data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(suffix, "png")
    return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def user_content(text: str, images: list[Path] | None) -> str | list[dict]:
    images = [p for p in (images or []) if p.exists()]
    if not images:
        return text
    return [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": _data_uri(p)}} for p in images
    ]


def designer_system(chat_id: str | None = None) -> str:
    learned = preferences.designer_context(chat_id)
    return DESIGNER_SYSTEM + (f"\n\n{learned}" if learned else "")


def build_history(messages: list[dict], resolve, vision: bool, chat_id: str | None = None) -> list[dict]:
    """Turn stored chat messages into LLM messages.

    Assistant turns get their directions re-attached so the model can refer
    to "direction 2" later. The system prompt carries what the user's
    reactions taught us so far.
    """
    out: list[dict] = [{"role": "system", "content": designer_system(chat_id)}]
    for m in messages:
        if m["role"] == "user":
            imgs = [resolve(p) for p in m.get("images") or []] if vision else []
            text = m["content"]
            if m.get("images") and not vision:
                text += "\n\n(The user attached reference images, but this model cannot see images.)"
            out.append({"role": "user", "content": user_content(text, imgs)})
        else:
            text = m["content"]
            if m.get("prompts"):
                ps = m["prompts"]
                block = (
                    {"directions": [p["brief"] for p in ps]}
                    if all(briefs.is_brief(p.get("brief")) for p in ps)
                    else {"prompts": [{k: p.get(k, "") for k in ("concept", "style", "prompt")} for p in ps]}
                )
                text += "\n\n```json\n" + json.dumps(block) + "\n```"
            out.append({"role": "assistant", "content": text})
    return out


# ── Backends ─────────────────────────────────────────────────────────


def backend_label() -> str:
    s = get_settings()
    if s.llm_backend == "claude-cli":
        return "Claude CLI"
    if s.llm_backend == "local":
        return f"local/{s.local_llm_model}"
    return s.chat_model


async def chat_supports_images() -> bool:
    s = get_settings()
    if s.llm_backend != "openrouter":
        return s.llm_backend == "local"  # assume a local vision model; it will error if not
    info = await openrouter.model_info(s.chat_model) or {}
    return "image" in (info.get("input_modalities") or [])


async def stream_reply(messages: list[dict]) -> AsyncIterator[dict]:
    """Yield {"type": "token"|"reasoning"|"usage", ...} from the configured backend."""
    s = get_settings()
    if s.llm_backend == "claude-cli":
        async for tok in claude_cli.chat_stream(messages):
            yield {"type": "token", "content": tok}
    elif s.llm_backend == "local":
        async for tok in local_llm.chat_stream(messages, model=s.local_llm_model, temperature=0.7):
            yield {"type": "token", "content": tok}
    else:
        async for ev in openrouter.chat_stream(
            messages, model=s.chat_model, temperature=0.7, reasoning_effort=s.reasoning_effort
        ):
            yield ev


def cost_record(usage: dict | None, model: str) -> dict:
    """Normalize a usage object into the fields library.add_cost takes."""
    usage = usage or {}
    credits, byok = openrouter.usage_cost(usage)
    return {
        "credits": credits,
        "byok": byok,
        "model": model,
        "generation_id": usage.get("id"),
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
    }


async def _complete(messages: list[dict], structured: bool) -> tuple[str, dict]:
    """Returns (reply, cost record). Local backends cost nothing."""
    s = get_settings()
    if s.llm_backend == "claude-cli":
        return await claude_cli.chat(messages), cost_record(None, "claude-cli")
    if s.llm_backend == "local":
        return await local_llm.chat(messages, model=s.local_llm_model, temperature=0.9), cost_record(None, backend_label())
    text, usage = await openrouter.chat(
        messages,
        model=s.chat_model,
        temperature=0.9,
        reasoning_effort=s.reasoning_effort,
        json_schema=BRIEF_SCHEMA if structured else None,
    )
    return text, cost_record(usage, s.chat_model)


_JSON_TAIL = 'Return only JSON in this form: {"directions": [brief, ...]}, with every field filled in.'


async def directions(
    brief: str, n: int, images: list[Path] | None = None, chat_id: str | None = None
) -> tuple[list[dict], dict]:
    """One-shot: brief -> n directions, each with a design brief."""
    text = f"Brief: {brief}\n\nWrite {n} directions. {_JSON_TAIL}"
    if images:
        text += "\nThe attached images are references; the image model will see them too."
    learned = preferences.designer_context(chat_id)
    if learned:
        text += f"\n\n{learned}"
    vision = await chat_supports_images()
    messages = [
        {"role": "system", "content": DIRECTIONS_SYSTEM},
        {"role": "user", "content": user_content(text, images if vision else None)},
    ]
    reply, cost = await _complete(messages, structured=True)
    prompts = parse_prompts(reply)
    if not prompts:
        raise ValueError("The chat model did not return usable directions. Try again or pick another chat model.")
    return prompts[:n], cost


async def variations(original: dict, n: int, instruction: str = "") -> tuple[list[dict], dict]:
    """Close variations of one direction (its brief, or its prompt for older images)."""
    b = original.get("brief")
    if briefs.is_brief(b):
        text = f"Original brief: {json.dumps(b)}\n"
    else:
        text = (
            f"Original prompt: {original.get('prompt')}\n"
            f"Concept: {original.get('concept') or ''}\nStyle: {original.get('style') or ''}\n"
        )
    if instruction:
        text += f"Instruction: {instruction}\n"
    text += f"\nWrite {n} variations. {_JSON_TAIL}"
    learned = preferences.designer_context(original.get("chat_id"))
    if learned:
        text += f"\n\n{learned}"
    messages = [{"role": "system", "content": VARIATION_SYSTEM}, {"role": "user", "content": text}]
    reply, cost = await _complete(messages, structured=True)
    prompts = parse_prompts(reply)
    if not prompts:
        raise ValueError("The chat model did not return usable directions.")
    return prompts[:n], cost
