"""What the designer knows about the render setup on each turn, and what it may change.

Every turn reads the setup fresh (Settings, plus the chat's own switches) instead of trusting
anything said earlier in the chat. The snapshot is stored on the reply, so the next turn can
tell the designer what the user changed in between ("switched from X to Y").

The designer never says what is rendering or what it costs: the server works that out from
the actual state when the reply ends (server.reply_round) and shows it under the reply.
"""

from __future__ import annotations

from logo_gen import chats, generator
from logo_gen.clients import openrouter
from logo_gen.config import get_settings
from logo_gen.local_models import manager
from logo_gen.local_models.registry import MODELS
from logo_gen.prompting.profiles import profile_for

TEXT_SKILL = {
    "strong": "renders exact wording well",
    "fair": "short wording only, sometimes misspelled",
    "weak": "often garbles text",
    "none": "can't render text",
}
MAX_PER_DIRECTION = 8


def short(model: str) -> str:
    return model.removeprefix(generator.LOCAL_PREFIX) if generator.is_local(model) else model.split("/", 1)[-1]


def _model_entry(model: str) -> dict:
    s = get_settings()
    p = profile_for(model)
    entry = {"id": model, "name": short(model), "cap": s.cap_for(model), "text": p.text, "profile": p.name}
    if generator.is_local(model):
        spec = MODELS.get(model.removeprefix(generator.LOCAL_PREFIX))
        entry |= {"local": True, "price": 0.0, "name": spec.name if spec else short(model)}
    else:
        info = openrouter.cached_model(model) or {}
        entry |= {"local": False, "price": info.get("price_per_image"), "name": info.get("name") or short(model)}
    return entry


def snapshot(chat_id: str) -> dict:
    """The render setup as it is right now, with this chat's overrides applied."""
    s = get_settings()
    return {
        "models": [_model_entry(m) for m in s.image_models if m],
        "per_direction": max(1, int(s.images_per_prompt or 1)),
        "generate_automatically": chats.generates_automatically(chat_id),
        "ask_first": chats.asks_first(chat_id),
    }


def available_models(limit: int = 40) -> list[str]:
    """Image model ids the user can switch to: installed local models, then the cached OpenRouter catalog."""
    out = [f"{generator.LOCAL_PREFIX}{spec.id}" for spec in MODELS.values() if manager.is_installed(spec)]
    for _, data in list(openrouter._catalog_cache.values())[:1]:
        out += [m["id"] for m in data.get("image", [])]
    seen, uniq = set(), []
    for m in out:
        if m not in seen:
            seen.add(m)
            uniq.append(m)
    return uniq[:limit]


def _names(snap: dict) -> str:
    return ", ".join(m["id"] for m in snap.get("models") or []) or "none"


def changes(prev: dict | None, snap: dict) -> list[str]:
    """What the user changed since the designer's last reply, in plain words."""
    if not prev:
        return []
    out = []
    if [m["id"] for m in prev.get("models") or []] != [m["id"] for m in snap["models"]]:
        out.append(f"switched the image models from {_names(prev)} to {_names(snap)}")
    if prev.get("per_direction") != snap["per_direction"]:
        out.append(f"changed images per direction from {prev.get('per_direction')} to {snap['per_direction']}")
    for key, label in (("generate_automatically", "Generate automatically"), ("ask_first", "Ask questions before generating")):
        if key in prev and prev[key] != snap[key]:
            out.append(f"turned {label} {'on' if snap[key] else 'off'}")
    return out


def _price(m: dict) -> str:
    if m.get("local"):
        return "local, free"
    price = m.get("price")
    return "price unknown" if price is None else f"about ${price:.3f} per image"


def describe(snap: dict, prev: dict | None = None) -> str:
    """The block appended to the designer's system prompt on this turn."""
    lines = [
        "## Current setup",
        "Read fresh for this reply. The user changes it below the message box, so it overrides anything said earlier in the chat.",
    ]
    if snap["models"]:
        lines.append("Image models selected:")
        for m in snap["models"]:
            skill = TEXT_SKILL.get(m["text"], m["text"])
            lines.append(f"- {m['id']} ({m['name']}): {_price(m)}, up to {m['cap']} images per request, text: {skill}.")
    else:
        lines.append("Image models selected: none yet.")
    lines.append(f"Images per direction: {snap['per_direction']}.")
    lines.append(
        "Generate automatically: "
        + ("on, so your directions render as soon as you reply." if snap["generate_automatically"]
           else "off, so the user picks directions and presses Generate.")
    )
    lines.append(f"Ask questions before generating: {'on' if snap['ask_first'] else 'off'}.")
    changed = changes(prev, snap)
    if changed:
        lines.append("Since your last reply, the user " + "; ".join(changed) + ".")
    lines.append(
        "Fit the briefs to these models: when none of them renders text well, prefer symbols over wordmarks "
        "and say so in your defaults line."
    )
    lines.append(
        "Never write lines about rendering, image counts or cost (\"Rendering 16 images now\", \"Ready to render\"). "
        "The app shows the real status under your reply."
    )
    lines.append(
        "If the user asks in words for other image models or another count per direction, add "
        '"render": {"models": ["<model id>"], "per_direction": N} to the JSON block, using ids from the list '
        "below or the ones selected. The app switches the setup to it. Leave it out otherwise."
    )
    avail = available_models()
    if avail:
        lines.append("Image models available: " + ", ".join(avail) + ".")
    return "\n".join(lines)


def previous(history: list[dict]) -> dict | None:
    """The setup stored on the designer's last reply in this branch, if any."""
    for m in reversed(history):
        if m["role"] == "assistant" and m.get("setup"):
            return m["setup"]
    return None


def apply_render(request: dict | None) -> str | None:
    """Switch the setup to what the designer asked for on the user's behalf. Returns a line, or None.

    Only ids the user could pick themselves are accepted, and the count stays between 1 and 8.
    """
    if not isinstance(request, dict):
        return None
    from logo_gen.config import update_settings

    s = get_settings()
    allowed = set(available_models(limit=1000)) | set(s.image_models)
    patch: dict = {}
    models = request.get("models")
    if isinstance(models, list):
        picked = [m for m in models if isinstance(m, str) and m in allowed]
        if picked and picked != s.image_models:
            patch["image_models"] = picked
    per = request.get("per_direction")
    if isinstance(per, (int, float)) and not isinstance(per, bool):
        n = max(1, min(int(per), MAX_PER_DIRECTION))
        if n != s.images_per_prompt:
            patch["images_per_prompt"] = n
    if not patch:
        return None
    update_settings(patch)
    said = []
    if "image_models" in patch:
        said.append(" and ".join(short(m) for m in patch["image_models"]))
    if "images_per_prompt" in patch:
        said.append(f"{patch['images_per_prompt']} per direction")
    return "Switched to " + ", ".join(said) + ", as you asked."
