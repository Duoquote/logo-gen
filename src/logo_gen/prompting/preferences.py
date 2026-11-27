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

Where reactions come from: the designer chat's thumbs and "Use this
direction" write to the ``feedback_events`` table (``POST /api/feedback``).
``sync_feedback()`` reads that table from a cursor kept in ``app_state`` and
applies each new event here. It runs after every ``POST /api/feedback`` and
at startup, so reactions stored while this layer wasn't running still count.
The table stays the source of truth; ``taste_votes`` is derived from it.
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
# feedback_events kinds -> ours. "unrate" clears a thumbs rating.
FROM_FEEDBACK = {"like": "like", "dislike": "dislike", "pick": "pick", "unrate": "clear"}
CURSOR_KEY = "taste.feedback_cursor"

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

_ready: object | None = None  # the connection our tables were created on


def _db() -> None:
    global _ready
    c = library.conn()
    if _ready is not c:
        c.executescript(SCHEMA)
        _ready = c


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
    if ev.kind not in WEIGHTS and ev.kind != "clear":
        raise ValueError(f"Unknown feedback kind {ev.kind!r}")
    for scope in _scopes(chat_id) or [GLOBAL]:
        rows = library._q("SELECT kind FROM taste_votes WHERE scope = ? AND image_id = ?", (scope, img["id"]))
        current = rows[0]["kind"] if rows else None
        # One vote per image. A pick is stronger than a thumbs up, so a like
        # (or clearing the thumbs) after a pick keeps the pick; a dislike wins.
        if current == "pick" and ev.kind in ("like", "clear"):
            continue
        if ev.kind == "clear":
            library._x("DELETE FROM taste_votes WHERE scope = ? AND image_id = ?", (scope, img["id"]))
            continue
        library._x(
            """INSERT OR REPLACE INTO taste_votes (scope, image_id, kind, model, traits, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (scope, img["id"], ev.kind, img.get("model") or "", json.dumps(traits_of(img)), ev.created_at),
        )
    return profile(chat_scope(chat_id) if chat_id else GLOBAL)


def sync_feedback() -> int:
    """Apply feedback_events rows newer than our cursor. Returns how many were applied."""
    from logo_gen import chats

    _db()
    rows = library._q("SELECT value FROM app_state WHERE key = ?", (CURSOR_KEY,))
    cursor = int(rows[0]["value"]) if rows else 0
    applied = 0
    while True:
        events = chats.list_feedback(cursor, None, 500)
        if not events:
            break
        for e in events:
            cursor = e["id"]
            kind = FROM_FEEDBACK.get(e["kind"])
            if not kind or not e.get("image_id"):
                continue
            try:
                apply_event(FeedbackEvent(kind, e["image_id"], e.get("chat_id"), e.get("created_at") or time.time()))  # type: ignore[arg-type]
                applied += 1
            except ValueError:
                pass  # the image was deleted since
        library._x(
            "INSERT INTO app_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (CURSOR_KEY, str(cursor)),
        )
    return applied


def add_notes(chat_id: str | None, more: list[str] | None = None, less: list[str] | None = None) -> None:
    """Store "more of" / "less of" phrases from the user's own words, for this chat only.

    They're about one project ("stacked stones", "not the P lettermark"), so
    they never go into the cross-chat profile.
    """
    _db()
    now = time.time()
    for scope in [chat_scope(chat_id)] if chat_id else []:
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


def _rank(scores: dict[str, float], positive: bool, n: int = 4) -> list[str]:
    items = [(k, v) for k, v in scores.items() if (v > 0 if positive else v < 0)]
    items.sort(key=lambda kv: -abs(kv[1]))
    return [k for k, _ in items[:n]]


def profile(scope: str, exclude_chat: str | None = None) -> dict:
    """The learned profile for a scope, in a form the UI can show as is.

    exclude_chat leaves out votes on that chat's images, so the cross-chat
    profile shown next to a chat's own doesn't count the same reactions twice.
    """
    _db()
    votes = library._q("SELECT * FROM taste_votes WHERE scope = ? ORDER BY created_at", (scope,))
    if exclude_chat:
        own = {r["image_id"] for r in library._q("SELECT image_id FROM taste_votes WHERE scope = ?", (chat_scope(exclude_chat),))}
        votes = [v for v in votes if v["image_id"] not in own]
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


GENERAL = ("mark_type", "style", "shape")  # traits that carry over between projects; colors and notes don't


def _describe(p: dict, strong_only: bool = False) -> list[str]:
    lines = []
    cats = GENERAL if strong_only else tuple(LABELS)
    liked = "; ".join(f"{LABELS[c]} {', '.join(v)}" for c, v in p["liked"].items() if c in cats)
    disliked = "; ".join(f"{LABELS[c]} {', '.join(v)}" for c, v in p["disliked"].items() if c in cats)
    if liked:
        lines.append(f"- Liked: {liked}")
    if disliked:
        lines.append(f"- Disliked: {disliked}")
    if not strong_only:
        good = [m["model"] for m in p["models"] if m["score"] > 0][:3]
        if good:
            lines.append(f"- Image models behind the liked results: {', '.join(good)}")
    if p["more"] and not strong_only:
        lines.append(f"- Asked for: {'; '.join(p['more'])}")
    if p["less"] and not strong_only:
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
        g = profile(GLOBAL, exclude_chat=chat_id)
        general = _describe(g, strong_only=True)
        if general and (g["counts"]["like"] + g["counts"]["pick"] + g["counts"]["dislike"]) >= 3:
            parts.append(
                "Across earlier chats this user tended toward the following. Treat it as a weak hint; "
                "this chat's brief wins:\n" + "\n".join(general)
            )
    return "\n\n".join(parts)


def render_hints(chat_id: str | None) -> list[str]:
    """Extra things to keep out of rendered prompts, from this chat's "less of" phrases and disliked styles.

    Only visual attributes the renderers know how to phrase (gradients, shadows,
    thin lines, texture, ...). Concept-level feedback ("not the P lettermark")
    goes to the designer instead: naming it in an image prompt tends to draw it.
    """
    if not chat_id:
        return []
    from logo_gen.prompting.render import POSITIVE

    p = profile(chat_scope(chat_id))
    candidates = [*p["less"], *[x for x in p["disliked"].get("style", []) if x not in p["liked"].get("style", [])]]
    out = [x for x in candidates if x.lower().removeprefix("no ").strip() in POSITIVE]
    return list(dict.fromkeys(out))[:4]
