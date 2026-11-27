"""What the designer learns from the user's reactions.

Inputs are feedback events (like, dislike, pick, clear on an image) and short
"more of / less of" phrases the designer pulls out of the user's messages.
Each vote stores the traits of the image it was about (mark type, style
words, shape words, colors, model), taken from the image's design brief at
vote time, so the profile survives the image being deleted.

The profile is recomputed from the stored votes and notes on every read. It
is small, so that's cheap, and it means undoing a vote is just deleting it.

Scopes: ``chat:<id>`` for one chat, and ``global`` for a taste profile across
chats (Settings can turn the global one off).

Integration point for the chat UI: call ``apply_event(FeedbackEvent(...))``
when the user likes, dislikes or picks an image, or POST the same fields to
``/api/preferences/events``. Favorites in the image dialog already count as
likes.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from logo_gen import library
from logo_gen.config import get_settings

Kind = Literal["like", "dislike", "pick", "clear"]
WEIGHTS = {"pick": 2.0, "like": 1.0, "dislike": -1.0}
GLOBAL = "global"

SCHEMA = """
CREATE TABLE IF NOT EXISTS taste_votes (
    scope TEXT NOT NULL,             -- chat:<id> | global
    image_id TEXT NOT NULL,
    kind TEXT NOT NULL,              -- like | dislike | pick
    model TEXT DEFAULT '',
    traits TEXT DEFAULT '{}',        -- {"mark_type": [...], "style": [...], ...} at vote time
    created_at REAL NOT NULL,
    PRIMARY KEY (scope, image_id)
);
CREATE TABLE IF NOT EXISTS taste_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope TEXT NOT NULL,
    kind TEXT NOT NULL,              -- more | less
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS taste_notes_scope ON taste_notes(scope, id);
"""

_ready = False


def _db() -> None:
    global _ready
    if not _ready:
        library.conn().executescript(SCHEMA)
        _ready = True


def chat_scope(chat_id: str) -> str:
    return f"chat:{chat_id}"


@dataclass
class FeedbackEvent:
    """One reaction to one image. This is the shape the chat UI's event endpoint should pass on."""

    kind: Kind
    image_id: str
    chat_id: str | None = None  # defaults to the image's chat
    created_at: float = field(default_factory=time.time)


# ── Traits ───────────────────────────────────────────────────────────

_SPLIT = re.compile(r"[,;/]| and | with ")
_STOP = {"", "a", "an", "the", "logo", "mark", "icon", "style", "colors", "colours", "two colors", "one color"}


def _phrases(s: str) -> list[str]:
    out = []
    for part in _SPLIT.split((s or "").lower()):
        part = " ".join(part.split()).strip(" .")
        if part not in _STOP and len(part) <= 40:
            out.append(part)
    return out


def traits_of(image: dict) -> dict[str, list[str]]:
    """Traits of an image, from its brief when it has one, else from the style line."""
    b = image.get("brief") or {}
    if isinstance(b, str):
        try:
            b = json.loads(b or "{}")
        except json.JSONDecodeError:
            b = {}
    t: dict[str, list[str]] = {}
    if b.get("mark_type"):
        t["mark_type"] = [b["mark_type"]]
    style = _phrases(b.get("style") or image.get("style") or "")
    if style:
        t["style"] = style
    shape = _phrases(b.get("shape_language") or "")
    if shape:
        t["shape"] = shape
    colors = []
    for c in b.get("palette") or []:
        if isinstance(c, dict) and (c.get("hex") or c.get("name")):
            colors.append(" ".join(x for x in (c.get("name"), c.get("hex")) if x))
    if colors:
        t["color"] = colors
    return t


# ── Writing ──────────────────────────────────────────────────────────


def _scopes(chat_id: str | None) -> list[str]:
    scopes = [chat_scope(chat_id)] if chat_id else []
    if get_settings().use_global_taste:
        scopes.append(GLOBAL)
    return scopes


def apply_event(ev: FeedbackEvent) -> dict | None:
    """Record a reaction. Returns the updated chat profile (or the global one for images outside a chat)."""
    _db()
    img = library.get_image(ev.image_id)
    if img is None:
        raise ValueError("Image not found")
    if img.get("parent_id") and img.get("kind") != "generated":
        img = library.get_image(img["parent_id"]) or img  # an upscale or cleanup counts for its original
    chat_id = ev.chat_id or img.get("chat_id")
    for scope in _scopes(chat_id) or [GLOBAL]:
        if ev.kind == "clear":
            library._x("DELETE FROM taste_votes WHERE scope = ? AND image_id = ?", (scope, img["id"]))
            continue
        if ev.kind not in WEIGHTS:
            raise ValueError(f"Unknown feedback kind {ev.kind!r}")
        library._x(
            """INSERT OR REPLACE INTO taste_votes (scope, image_id, kind, model, traits, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (scope, img["id"], ev.kind, img.get("model") or "", json.dumps(traits_of(img)), ev.created_at),
        )
    return profile(chat_scope(chat_id) if chat_id else GLOBAL)


def add_notes(chat_id: str | None, more: list[str] | None = None, less: list[str] | None = None) -> None:
    """Store "more of" / "less of" phrases from the user's own words."""
    _db()
    now = time.time()
    for scope in _scopes(chat_id):
        for kind, items in (("more", more or []), ("less", less or [])):
            for text in items[:6]:
                text = " ".join(str(text).split())[:80]
                if text:
                    library._x(
                        "INSERT INTO taste_notes (scope, kind, text, created_at) VALUES (?, ?, ?, ?)",
                        (scope, kind, text, now),
                    )


def clear(scope: str) -> None:
    _db()
    library._x("DELETE FROM taste_votes WHERE scope = ?", (scope,))
    library._x("DELETE FROM taste_notes WHERE scope = ?", (scope,))


# ── Reading ──────────────────────────────────────────────────────────


def _rank(scores: dict[str, float], positive: bool, n: int = 5) -> list[str]:
    items = [(k, v) for k, v in scores.items() if (v > 0 if positive else v < 0)]
    items.sort(key=lambda kv: -abs(kv[1]))
    return [k for k, _ in items[:n]]


def profile(scope: str) -> dict:
    """The learned profile for a scope, in a form the UI can show as is."""
    _db()
    votes = library._q("SELECT * FROM taste_votes WHERE scope = ? ORDER BY created_at", (scope,))
    notes = library._q("SELECT * FROM taste_notes WHERE scope = ? ORDER BY id DESC LIMIT 60", (scope,))
    scores: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    models: dict[str, dict] = {}
    counts = {"like": 0, "dislike": 0, "pick": 0}
    for v in votes:
        w = WEIGHTS.get(v["kind"], 0.0)
        counts[v["kind"]] = counts.get(v["kind"], 0) + 1
        for cat, values in json.loads(v["traits"] or "{}").items():
            for val in values:
                scores[cat][val] += w
        if v["model"]:
            m = models.setdefault(v["model"], {"model": v["model"], "score": 0.0, "likes": 0, "dislikes": 0})
            m["score"] += w
            m["likes" if w > 0 else "dislikes"] += 1
    more = list(dict.fromkeys(n["text"] for n in notes if n["kind"] == "more"))[:8]
    less = list(dict.fromkeys(n["text"] for n in notes if n["kind"] == "less"))[:8]
    # A later "more of X" cancels an older "less of X" and the other way around.
    newest: dict[str, str] = {}
    for n in notes:  # newest first
        newest.setdefault(n["text"].lower(), n["kind"])
    more = [x for x in more if newest.get(x.lower()) == "more"]
    less = [x for x in less if newest.get(x.lower()) == "less"]
    cats = ("mark_type", "style", "shape", "color")
    return {
        "scope": scope,
        "liked": {c: _rank(scores[c], True) for c in cats if _rank(scores[c], True)},
        "disliked": {c: _rank(scores[c], False) for c in cats if _rank(scores[c], False)},
        "models": sorted(models.values(), key=lambda m: -m["score"]),
        "more": more,
        "less": less,
        "counts": counts | {"notes": len(more) + len(less)},
        "empty": not votes and not more and not less,
    }


LABELS = {"mark_type": "mark types", "style": "styles", "shape": "shapes", "color": "colors"}


def _describe(p: dict, strong_only: bool = False) -> list[str]:
    lines = []
    liked = "; ".join(f"{LABELS[c]} {', '.join(v)}" for c, v in p["liked"].items())
    disliked = "; ".join(f"{LABELS[c]} {', '.join(v)}" for c, v in p["disliked"].items())
    if liked:
        lines.append(f"- Liked: {liked}")
    if disliked:
        lines.append(f"- Disliked: {disliked}")
    if not strong_only:
        good = [m["model"] for m in p["models"] if m["score"] > 0][:3]
        if good:
            lines.append(f"- Image models behind the liked results: {', '.join(good)}")
    if p["more"]:
        lines.append(f"- Asked for: {'; '.join(p['more'])}")
    if p["less"]:
        lines.append(f"- Asked to avoid: {'; '.join(p['less'])}")
    return lines


def designer_context(chat_id: str | None) -> str:
    """A short block for the designer's system prompt, or '' when nothing is known."""
    parts = []
    if chat_id:
        p = profile(chat_scope(chat_id))
        if not p["empty"]:
            parts.append(
                "What this user reacted to in this chat. Build on what they liked, steer away from what "
                "they disliked, and don't bring back rejected ideas unless they ask:\n" + "\n".join(_describe(p))
            )
    if get_settings().use_global_taste:
        g = profile(GLOBAL)
        if not g["empty"] and (g["counts"]["like"] + g["counts"]["pick"] + g["counts"]["dislike"]) >= 3:
            parts.append(
                "Across earlier chats this user tended toward the following. Treat it as a weak hint; "
                "this chat's brief wins:\n" + "\n".join(_describe(g, strong_only=True))
            )
    return "\n\n".join(parts)


def render_hints(chat_id: str | None) -> list[str]:
    """Extra things to avoid in rendered prompts: the user's own "less of" phrases and disliked styles."""
    if not chat_id:
        return []
    p = profile(chat_scope(chat_id))
    out = [x for x in p["less"] if len(x.split()) <= 4]
    out += [x for x in p["disliked"].get("style", []) if x not in p["liked"].get("style", [])][:2]
    return list(dict.fromkeys(out))[:4]
