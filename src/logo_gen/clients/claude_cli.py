"""Local Claude CLI client — shells out to `claude -p`.

Uses the user's locally-installed and already-authenticated Claude CLI
(from their Claude subscription). No API key, no auth flow — the user
handles `claude login` themselves.

Conversation history is flattened into a single prompt and piped via
stdin (avoids ARG_MAX on Windows). The system prompt is passed via
--system-prompt so Claude behaves as our logo-design assistant rather
than the default Claude Code agent.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys


def _resolve_binary() -> str:
    """Resolve the claude CLI binary path. Raises if not found."""
    path = shutil.which("claude")
    if not path:
        raise RuntimeError(
            "claude CLI not found on PATH. Install Claude Code and run `claude login`."
        )
    return path


def _wrap_for_shim(binary: str, args: list[str]) -> list[str]:
    """On Windows, .cmd/.bat shims need cmd.exe to interpret them."""
    if sys.platform == "win32" and binary.lower().endswith((".cmd", ".bat")):
        return ["cmd.exe", "/c", binary, *args]
    return [binary, *args]


def _extract_text(content) -> str:
    """Extract plain text from either a string or a multimodal content list."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return content or ""


def _flatten_messages(messages: list[dict]) -> tuple[str, str]:
    """Split messages into (system_prompt, stdin_prompt).

    All system messages are concatenated into system_prompt. The remaining
    user/assistant turns are flattened into a single text prompt with the
    last user message as the "current" turn.
    """
    system_parts: list[str] = []
    history: list[tuple[str, str]] = []

    for m in messages:
        role = m.get("role", "")
        text = _extract_text(m.get("content"))
        if role == "system":
            if text:
                system_parts.append(text)
        elif role in ("user", "assistant"):
            history.append((role, text))

    system_prompt = "\n\n".join(system_parts)

    # Find the last user message — that's the "current" turn.
    last_user_idx = None
    for i in range(len(history) - 1, -1, -1):
        if history[i][0] == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return system_prompt, ""

    current = history[last_user_idx][1]
    prior = history[:last_user_idx]

    if not prior:
        return system_prompt, current

    lines = ["Previous conversation:\n"]
    for role, text in prior:
        marker = "User" if role == "user" else "Assistant"
        lines.append(f"{marker}: {text}")
    lines.append("")
    lines.append(f"User: {current}")
    return system_prompt, "\n".join(lines)


def _build_argv(system_prompt: str, stream: bool) -> list[str]:
    binary = _resolve_binary()
    args = ["-p"]
    if stream:
        args += [
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
        ]
    else:
        args += ["--output-format", "text"]
    if system_prompt:
        args += ["--system-prompt", system_prompt]
    # Prevent Claude from trying to use tools — we just want a chat reply.
    args += ["--allowed-tools", ""]
    return _wrap_for_shim(binary, args)


async def _spawn(argv: list[str], stdin_data: str):
    return await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def chat(
    messages: list[dict],
    model: str | None = None,  # accepted for signature parity; ignored
    temperature: float = 0.7,  # accepted for signature parity; ignored
    response_format: dict | None = None,  # accepted for signature parity; ignored
) -> str:
    """One-shot chat via `claude -p`."""
    system_prompt, user_input = _flatten_messages(messages)
    argv = _build_argv(system_prompt, stream=False)

    proc = await _spawn(argv, user_input)
    stdout, stderr = await proc.communicate(input=user_input.encode("utf-8"))
    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI failed (exit {proc.returncode}): "
            f"{stderr.decode('utf-8', errors='replace').strip()}"
        )
    return stdout.decode("utf-8", errors="replace").strip()


async def chat_stream(
    messages: list[dict],
    model: str | None = None,  # accepted for signature parity; ignored
    temperature: float = 0.7,  # accepted for signature parity; ignored
):
    """Stream chat completion tokens via `claude -p --output-format stream-json`."""
    system_prompt, user_input = _flatten_messages(messages)
    argv = _build_argv(system_prompt, stream=True)

    proc = await _spawn(argv, user_input)
    assert proc.stdin is not None and proc.stdout is not None

    try:
        proc.stdin.write(user_input.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
    except (BrokenPipeError, ConnectionResetError):
        pass

    try:
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Partial deltas (--include-partial-messages)
            if event.get("type") == "stream_event":
                inner = event.get("event", {})
                if inner.get("type") == "content_block_delta":
                    delta = inner.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield text
    finally:
        await proc.wait()
        if proc.returncode and proc.returncode != 0:
            stderr_data = b""
            if proc.stderr is not None:
                try:
                    stderr_data = await proc.stderr.read()
                except Exception:
                    pass
            raise RuntimeError(
                f"claude CLI failed (exit {proc.returncode}): "
                f"{stderr_data.decode('utf-8', errors='replace').strip()}"
            )


async def is_available() -> bool:
    """Check whether the claude CLI is installed and runnable."""
    path = shutil.which("claude")
    if not path:
        return False
    try:
        argv = _wrap_for_shim(path, ["--version"])
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=5)
        return proc.returncode == 0
    except (asyncio.TimeoutError, FileNotFoundError, OSError):
        return False
