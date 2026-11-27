"""The design brief: what the designer LLM writes for each direction.

A brief says what the logo is, not how to phrase it for a model. The
renderers in render.py turn one brief into the right prompt for each image
model. Keeping the two apart means the chat model only has to be a good
designer, and the per-model quirks live in data (profiles.py).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

MARK_TYPES = ("pictorial", "abstract", "lettermark", "wordmark", "combination", "emblem", "mascot")
FORMATS = ("logo", "app icon", "favicon")

# What each field means. Used in the designer's system prompt and in the JSON
# schema, so the two can't drift apart.
FIELDS: dict[str, str] = {
    "concept": "short name for the idea, 2 to 4 words",
    "mark_type": "one of: " + ", ".join(MARK_TYPES),
    "format": "one of: " + ", ".join(FORMATS),
    "subject": "what the mark shows, in concrete shapes (\"a fox head built from three triangles\")",
    "composition": "how it sits: centered, padding, symmetry, how many elements",
    "shape_language": "geometry, corners, stroke weight (\"rounded geometric, 2 px monoline\")",
    "style": "rendering style in 2 to 5 words (\"flat vector, two colors\")",
    "palette": "1 to 3 colors, each with a name and a hex value",
    "background": "\"white\", \"transparent\", or a solid color with hex",
    "text": "the exact wording to render, or an empty string for no text",
    "avoid": "up to 4 short things to keep out (\"gradients\", \"thin hairlines\")",
}

_HEX = re.compile(r"#?([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")


@dataclass
class Color:
    name: str = ""
    hex: str = ""

    def label(self) -> str:
        if self.name and self.hex:
            return f"{self.name} {self.hex}"
        return self.name or self.hex


@dataclass
class Brief:
    concept: str = ""
    mark_type: str = "pictorial"
    format: str = "logo"
    subject: str = ""
    composition: str = ""
    shape_language: str = ""
    style: str = ""
    palette: list[Color] = field(default_factory=list)
    background: str = "white"
    text: str = ""
    avoid: list[str] = field(default_factory=list)

    # ── Derived ──────────────────────────────────────────────────────

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())

    @property
    def wants_transparent(self) -> bool:
        return "transparent" in self.background.lower()

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One readable line for the chat and the gallery. Not a model prompt."""
        parts = [f"{self.mark_type.capitalize()} {self.format}: {self.subject}".strip().rstrip(".") + "."]
        look = ", ".join(x for x in (self.shape_language, self.style) if x)
        if look:
            parts.append(look[0].upper() + look[1:] + ".")
        if self.palette:
            parts.append(", ".join(c.label() for c in self.palette) + ".")
        parts.append(f'Text: "{self.text}".' if self.has_text else "No text.")
        return " ".join(parts)


def _hex(value: str) -> str:
    m = _HEX.search(value or "")
    if not m:
        return ""
    h = m.group(1).upper()
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return f"#{h}"


def _str(v: object) -> str:
    return " ".join(str(v).split()) if v is not None else ""


def _pick(value: str, allowed: tuple[str, ...], default: str) -> str:
    v = value.lower().strip()
    for a in allowed:
        if v == a or v.replace("_", " ") == a or v.startswith(a):
            return a
    if v in ("symbol", "icon", "pictogram", "pictorial mark"):
        return "pictorial"
    if v in ("monogram", "letter", "initials"):
        return "lettermark"
    if v in ("logotype", "text"):
        return "wordmark"
    if v in ("badge", "seal", "crest"):
        return "emblem"
    if v in ("app_icon", "appicon", "icon"):
        return "app icon" if "app icon" in allowed else default
    return default


def _palette(raw: object) -> list[Color]:
    out: list[Color] = []
    if isinstance(raw, str):
        raw = [p for p in re.split(r"[;,]", raw) if p.strip()]
    if not isinstance(raw, list):
        return out
    for item in raw[:4]:
        if isinstance(item, dict):
            name, hx = _str(item.get("name")), _hex(_str(item.get("hex")))
        else:
            s = _str(item)
            hx = _hex(s)
            name = _HEX.sub("", s).strip(" :-")
        if name or hx:
            out.append(Color(name=name, hex=hx))
    return out


def from_dict(d: dict) -> Brief:
    """Build a brief from whatever the chat model returned. Forgiving on purpose."""
    avoid = d.get("avoid") or []
    if isinstance(avoid, str):
        avoid = [a for a in re.split(r"[;,]", avoid)]
    b = Brief(
        concept=_str(d.get("concept")) or "Direction",
        mark_type=_pick(_str(d.get("mark_type")), MARK_TYPES, "pictorial"),
        format=_pick(_str(d.get("format")), FORMATS, "logo"),
        subject=_str(d.get("subject")),
        composition=_str(d.get("composition")),
        shape_language=_str(d.get("shape_language")),
        style=_str(d.get("style")),
        palette=_palette(d.get("palette")),
        background=_str(d.get("background")) or "white",
        text=_str(d.get("text")).strip("\"'"),
        avoid=[_str(a) for a in avoid if _str(a)][:6],
    )
    if b.text.lower() in ("none", "no text", "n/a", "-"):
        b.text = ""
    return b


def is_brief(d: object) -> bool:
    return isinstance(d, dict) and bool(_str(d.get("subject")))


def json_schema() -> dict:
    """Strict JSON schema for structured outputs: {"directions": [brief, ...]}."""
    props: dict = {k: {"type": "string", "description": v} for k, v in FIELDS.items()}
    props["mark_type"]["enum"] = list(MARK_TYPES)
    props["format"]["enum"] = list(FORMATS)
    props["palette"] = {
        "type": "array",
        "description": FIELDS["palette"],
        "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "hex": {"type": "string"}},
            "required": ["name", "hex"],
            "additionalProperties": False,
        },
    }
    props["avoid"] = {"type": "array", "description": FIELDS["avoid"], "items": {"type": "string"}}
    return {
        "name": "logo_briefs",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "directions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": props,
                        "required": list(FIELDS),
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["directions"],
            "additionalProperties": False,
        },
    }
