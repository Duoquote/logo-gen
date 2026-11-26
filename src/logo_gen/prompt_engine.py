"""Prompt writing: the chat designer, one-shot directions, and variations."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import AsyncIterator

from logo_gen.clients import claude_cli, local_llm, openrouter
from logo_gen.config import get_settings

PROMPT_RULES = """\
How to write an image prompt:
- 1 to 3 plain sentences, no keyword lists. Lead with the logo type and the symbol.
- Describe the symbol in concrete shapes ("a fox head built from three triangles", \
not "a dynamic fox").
- Name the style (flat vector, geometric, line mark, negative space, soft gradient, etc.).
- Give 1 to 3 colors by name and hex, e.g. "deep teal #0F766E".
- Say how it sits: centered, generous padding, on a plain white background.
- It is an icon, so no text, letters or wordmarks unless the user asks for them. \
Say "no text" once; do not pile on negative phrases.
- Keep it simple enough to read at 32 px."""

DESIGNER_SYSTEM = f"""\
You are a senior brand designer who also writes prompts for image models. \
You help the user find a logo mark (an icon, not a wordmark) through a short conversation.

How to talk:
- Be brief and concrete. No flattery, no filler.
- If the brief is thin, ask at most 3 short questions in one message (what the thing is, \
the feeling, colors or constraints). If the user wants to skip questions, go straight to directions.
- When you have enough, or the user asks to generate, propose distinct directions. \
Each direction should be a different idea, not the same idea in four colors.
- When the user gives feedback on results, adjust and propose new directions.
- If the user attaches reference images, the image model will see them too. Refer to \
what to keep and what to change rather than describing the image from scratch.

{PROMPT_RULES}

When you propose directions, write a one-line summary of each, then end your reply \
with exactly one fenced JSON block in this form (4 directions unless the user asks \
for a different number):

```json
{{"prompts": [{{"concept": "short name", "style": "style in 2-4 words", "prompt": "the image prompt"}}]}}
```

Do not put a JSON block in replies that only ask questions."""

DIRECTIONS_SYSTEM = f"""\
You are a senior brand designer writing prompts for image models. Given a short brief, \
return distinct logo mark directions. Each one must be a different idea (different \
symbol or construction), not a recolor.

{PROMPT_RULES}"""

VARIATION_SYSTEM = f"""\
You are a senior brand designer. The user picked a logo and wants variations of it. \
Keep the core idea of the original prompt and change one thing per variation: \
proportions, line weight, color palette, level of detail, or a small twist on the symbol. \
If the user gave an instruction, follow it in every variation.

{PROMPT_RULES}"""

PROMPTS_SCHEMA = {
    "name": "logo_prompts",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string"},
                        "style": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["concept", "style", "prompt"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["prompts"],
        "additionalProperties": False,
    },
}

# ── Parsing ──────────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _clean_prompts(items: object) -> list[dict]:
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, dict) and str(it.get("prompt") or "").strip():
            out.append({
                "concept": str(it.get("concept") or "").strip() or "Direction",
                "style": str(it.get("style") or "").strip(),
                "prompt": str(it["prompt"]).strip(),
            })
    return out


def parse_prompts(text: str) -> list[dict]:
    """Find a {"prompts": [...]} object in a reply. Uses the last fenced block."""
    candidates = _FENCE.findall(text)
    if not candidates:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            candidates = [text[start : end + 1]]
    for raw in reversed(candidates):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            prompts = _clean_prompts(data.get("prompts"))
            if prompts:
                return prompts
        if isinstance(data, list):
            prompts = _clean_prompts(data)
            if prompts:
                return prompts
    return []


def strip_prompt_block(text: str) -> str:
    """Remove the JSON block from a reply for display."""
    return _FENCE.sub("", text).strip()


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


def build_history(messages: list[dict], resolve, vision: bool) -> list[dict]:
    """Turn stored chat messages into LLM messages.

    Assistant turns get their prompt directions re-attached so the model can
    refer to "direction 2" later.
    """
    out: list[dict] = [{"role": "system", "content": DESIGNER_SYSTEM}]
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
                text += "\n\n```json\n" + json.dumps({"prompts": m["prompts"]}) + "\n```"
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


async def _complete(messages: list[dict], structured: bool) -> tuple[str, float]:
    s = get_settings()
    if s.llm_backend == "claude-cli":
        return await claude_cli.chat(messages), 0.0
    if s.llm_backend == "local":
        return await local_llm.chat(messages, model=s.local_llm_model, temperature=0.9), 0.0
    text, usage = await openrouter.chat(
        messages,
        model=s.chat_model,
        temperature=0.9,
        reasoning_effort=s.reasoning_effort,
        json_schema=PROMPTS_SCHEMA if structured else None,
    )
    return text, float(usage.get("cost") or 0.0)


_JSON_TAIL = (
    'Return only JSON in this form: {"prompts": [{"concept": "...", "style": "...", "prompt": "..."}]}'
)


async def directions(brief: str, n: int, images: list[Path] | None = None) -> tuple[list[dict], float]:
    """One-shot: brief -> n prompt directions."""
    text = f"Brief: {brief}\n\nWrite {n} directions. {_JSON_TAIL}"
    if images:
        text += "\nThe attached images are references; the image model will see them too."
    vision = await chat_supports_images()
    messages = [
        {"role": "system", "content": DIRECTIONS_SYSTEM},
        {"role": "user", "content": user_content(text, images if vision else None)},
    ]
    reply, cost = await _complete(messages, structured=True)
    prompts = parse_prompts(reply)
    if not prompts:
        raise ValueError("The chat model did not return usable prompts. Try again or pick another chat model.")
    return prompts[:n], cost


async def variations(original: dict, n: int, instruction: str = "") -> tuple[list[dict], float]:
    """Close variations of one prompt."""
    text = (
        f"Original prompt: {original.get('prompt')}\n"
        f"Concept: {original.get('concept') or ''}\nStyle: {original.get('style') or ''}\n"
    )
    if instruction:
        text += f"Instruction: {instruction}\n"
    text += f"\nWrite {n} variations. {_JSON_TAIL}"
    messages = [{"role": "system", "content": VARIATION_SYSTEM}, {"role": "user", "content": text}]
    reply, cost = await _complete(messages, structured=True)
    prompts = parse_prompts(reply)
    if not prompts:
        raise ValueError("The chat model did not return usable prompts.")
    return prompts[:n], cost
