"""Designer chats, stored as a tree of messages.

Every message has a parent (None for the first one). Editing a user message
or regenerating a reply adds a sibling instead of overwriting anything, so
old branches stay. The chat remembers which leaf is showing (active_leaf);
the visible conversation is the path from the root to that leaf.

Deleting is soft: rows get a deleted_at stamp and can be restored with the
same stamp, so the UI can offer undo. purge() removes old deleted rows.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from logo_gen.library import _image_dict, _q, _x, with_primary

TITLE_LEN = 60


# ── Chats ────────────────────────────────────────────────────────────


def create(title: str = "") -> dict:
    now = time.time()
    chat_id = uuid.uuid4().hex[:16]
    _x("INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)", (chat_id, title, now, now))
    return get(chat_id)  # type: ignore[return-value]


def get(chat_id: str, include_deleted: bool = False) -> dict | None:
    rows = _q("SELECT * FROM chats WHERE id = ?" + ("" if include_deleted else " AND deleted_at IS NULL"), (chat_id,))
    if not rows:
        return None
    d = dict(rows[0])
    d["pinned"] = bool(d.get("pinned"))
    d["ask_first"] = None if d.get("ask_first") is None else bool(d["ask_first"])
    d["auto_generate"] = None if d.get("auto_generate") is None else bool(d["auto_generate"])
    return d


def asks_first(chat_id: str) -> bool:
    """Whether the designer asks questions before proposing and rendering, for this chat."""
    from logo_gen.config import get_settings

    chat = get(chat_id, include_deleted=True)
    if chat and chat.get("ask_first") is not None:
        return bool(chat["ask_first"])
    return get_settings().ask_before_generating


def generates_automatically(chat_id: str) -> bool:
    """Whether a reply's directions in this chat render right away, without a click on Generate."""
    from logo_gen.config import get_settings

    chat = get(chat_id, include_deleted=True)
    if chat and chat.get("auto_generate") is not None:
        return bool(chat["auto_generate"])
    return get_settings().generate_automatically


def auto_title(text: str) -> str:
    """The first line of the first message, cut at a word boundary."""
    line = " ".join(text.strip().splitlines()[0].split()) if text.strip() else ""
    if len(line) <= TITLE_LEN:
        return line
    cut = line[:TITLE_LEN].rsplit(" ", 1)[0]
    return cut.rstrip(",.;:") + "..."


def list_all(q: str | None = None) -> list[dict]:
    """Chats with at least one message, newest activity first, with spend and a thumbnail."""
    where = "c.deleted_at IS NULL"
    args: list[Any] = []
    if q:
        where += """ AND (c.title LIKE ? OR EXISTS (
            SELECT 1 FROM messages s WHERE s.chat_id = c.id AND s.deleted_at IS NULL AND s.content LIKE ?))"""
        args += [f"%{q}%", f"%{q}%"]
    rows = _q(
        f"""SELECT c.*,
               (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id AND m.deleted_at IS NULL) AS message_count,
               (SELECT COUNT(*) FROM messages m WHERE m.chat_id = c.id AND m.status = 'streaming') AS streaming,
               (SELECT COALESCE(SUM(credits + byok), 0) FROM costs k WHERE k.chat_id = c.id) AS spend
            FROM chats c WHERE {where} ORDER BY c.updated_at DESC""",
        tuple(args),
    )
    chats = [dict(r) | {"pinned": bool(r["pinned"]), "streaming": bool(r["streaming"])} for r in rows]
    chats = [c for c in chats if c["message_count"] > 0]
    thumbs = thumbnails([c["id"] for c in chats])
    for c in chats:
        c["thumbnail"] = thumbs.get(c["id"])
        c["spend"] = round(c["spend"], 6)
    return chats


def thumbnails(chat_ids: list[str]) -> dict[str, str]:
    """The logo that stands for each chat: the one picked, else the last liked, else the newest image.

    A picked or liked image shows its primary version (an upscale or a cleaned copy) if one is set.
    """
    if not chat_ids:
        return {}
    marks = ", ".join("?" for _ in chat_ids)
    out: dict[str, str] = {}
    candidates: dict[str, tuple[int, float, str]] = {}  # chat -> (priority, time, image id)
    for r in _q(
        f"""SELECT chat_id, image_id, kind, created_at FROM feedback_events
            WHERE chat_id IN ({marks}) AND kind IN ('pick', 'like') AND image_id IS NOT NULL ORDER BY id""",
        tuple(chat_ids),
    ):
        prio = 2 if r["kind"] == "pick" else 1
        cur = candidates.get(r["chat_id"])
        if not cur or (prio, r["created_at"]) >= cur[:2]:
            candidates[r["chat_id"]] = (prio, r["created_at"], r["image_id"])
    liked = current_ratings([c[2] for c in candidates.values()])
    for chat_id, (prio, _, image_id) in list(candidates.items()):
        if prio == 1 and liked.get(image_id) != "like":
            candidates.pop(chat_id)  # the like was taken back
    for r in _q(
        f"""SELECT chat_id, id FROM images i WHERE chat_id IN ({marks}) AND kind = 'generated'
            AND created_at = (SELECT MAX(created_at) FROM images j WHERE j.chat_id = i.chat_id AND j.kind = 'generated')""",
        tuple(chat_ids),
    ):
        candidates.setdefault(r["chat_id"], (0, 0, r["id"]))
    ids = [c[2] for c in candidates.values()]
    if not ids:
        return out
    imgs = {
        i["id"]: i
        for i in with_primary([_image_dict(r) for r in _q(f"SELECT * FROM images WHERE id IN ({', '.join('?' for _ in ids)})", tuple(ids))])
    }
    for chat_id, (_, _, image_id) in candidates.items():
        img = imgs.get(image_id)
        if img:
            out[chat_id] = (img.get("primary") or img)["url"]
    return out


def update(chat_id: str, title: str | None = None, pinned: bool | None = None, ask_first: bool | None = None,
           auto_generate: bool | None = None) -> dict | None:
    if ask_first is not None:
        _x("UPDATE chats SET ask_first = ? WHERE id = ?", (int(ask_first), chat_id))
    if auto_generate is not None:
        _x("UPDATE chats SET auto_generate = ? WHERE id = ?", (int(auto_generate), chat_id))
    if title is not None:
        _x("UPDATE chats SET title = ? WHERE id = ?", (title.strip()[:200], chat_id))
    if pinned is not None:
        _x("UPDATE chats SET pinned = ? WHERE id = ?", (int(pinned), chat_id))
    return get(chat_id)


def delete(chat_id: str) -> None:
    _x("UPDATE chats SET deleted_at = ? WHERE id = ?", (time.time(), chat_id))
    if last_opened() == chat_id:
        _x("DELETE FROM app_state WHERE key = 'designer.last_chat'")


def restore(chat_id: str) -> dict | None:
    _x("UPDATE chats SET deleted_at = NULL WHERE id = ?", (chat_id,))
    return get(chat_id)


def purge(older_than: float = 7 * 86400) -> None:
    """Remove chats and messages deleted more than a week ago. Their images stay in the gallery."""
    cutoff = time.time() - older_than
    for r in _q("SELECT id FROM chats WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,)):
        _x("DELETE FROM messages WHERE chat_id = ?", (r["id"],))
        _x("DELETE FROM chats WHERE id = ?", (r["id"],))
    _x("DELETE FROM messages WHERE deleted_at IS NOT NULL AND deleted_at < ?", (cutoff,))


def mark_opened(chat_id: str) -> None:
    _x("UPDATE chats SET last_opened_at = ? WHERE id = ?", (time.time(), chat_id))
    _x("INSERT INTO app_state (key, value) VALUES ('designer.last_chat', ?) "
       "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (chat_id,))


def last_opened() -> str | None:
    rows = _q("SELECT value FROM app_state WHERE key = 'designer.last_chat'")
    if not rows:
        return None
    chat_id = rows[0]["value"]
    return chat_id if get(chat_id) else None


def touch(chat_id: str) -> None:
    _x("UPDATE chats SET updated_at = ? WHERE id = ?", (time.time(), chat_id))


# ── Messages ─────────────────────────────────────────────────────────


def _with_base_prompts(prompts: list[dict]) -> list[dict]:
    """Add each direction's copyable prompt. Worked out on read, so older replies have one too."""
    from logo_gen.prompting import brief as briefs
    from logo_gen.prompting.render import base_prompt

    for p in prompts:
        if isinstance(p, dict) and briefs.is_brief(p.get("brief")):
            p["base_prompt"] = base_prompt(p["brief"])
    return prompts


def _message(r) -> dict:
    return {
        "id": r["id"],
        "chat_id": r["chat_id"],
        "parent_id": r["parent_id"],
        "role": r["role"],
        "content": r["content"],
        "reasoning": r["reasoning"] or "",
        "images": json.loads(r["images"] or "[]"),
        "prompts": _with_base_prompts(json.loads(r["prompts"] or "[]")),
        "status": r["status"] or "done",
        "error": r["error"] or "",
        "model": r["model"] or "",
        "created_at": r["created_at"],
        "finished_at": r["finished_at"],
    }


def get_message(message_id: int) -> dict | None:
    rows = _q("SELECT * FROM messages WHERE id = ? AND deleted_at IS NULL", (message_id,))
    return _message(rows[0]) if rows else None


def add_message(
    chat_id: str,
    role: str,
    content: str,
    parent_id: int | None,
    images: list[str] | None = None,
    prompts: list[dict] | None = None,
    status: str = "done",
    model: str = "",
) -> int:
    now = time.time()
    cur = _x(
        """INSERT INTO messages (chat_id, parent_id, role, content, images, prompts, status, model, created_at, finished_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (chat_id, parent_id, role, content, json.dumps(images or []), json.dumps(prompts or []), status, model, now,
         now if status == "done" else None),
    )
    mid = int(cur.lastrowid or 0)
    _x("UPDATE chats SET updated_at = ?, active_leaf = ? WHERE id = ?", (now, mid, chat_id))
    if role == "user":
        _x("UPDATE chats SET title = ? WHERE id = ? AND (title IS NULL OR title = '')", (auto_title(content), chat_id))
    return mid


def update_message(message_id: int, **fields: Any) -> None:
    allowed = {"content", "reasoning", "prompts", "status", "error", "finished_at", "model"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if "prompts" in fields:
        fields["prompts"] = json.dumps(fields["prompts"] or [])
    if not fields:
        return
    sets = ", ".join(f"{k} = ?" for k in fields)
    _x(f"UPDATE messages SET {sets} WHERE id = ?", (*fields.values(), message_id))


def path(leaf_id: int | None) -> list[dict]:
    """Messages from the root down to leaf_id."""
    out: list[dict] = []
    seen: set[int] = set()
    mid = leaf_id
    while mid and mid not in seen:
        seen.add(mid)
        m = get_message(mid)
        if not m:
            break
        out.append(m)
        mid = m["parent_id"]
    return out[::-1]


def children(chat_id: str, parent_id: int | None) -> list[int]:
    if parent_id is None:
        rows = _q("SELECT id FROM messages WHERE chat_id = ? AND parent_id IS NULL AND deleted_at IS NULL ORDER BY id", (chat_id,))
    else:
        rows = _q("SELECT id FROM messages WHERE chat_id = ? AND parent_id = ? AND deleted_at IS NULL ORDER BY id", (chat_id, parent_id))
    return [r["id"] for r in rows]


def newest_leaf(chat_id: str, message_id: int) -> int:
    """Follow the newest child down from a message to the end of its branch."""
    leaf = message_id
    for _ in range(10_000):
        kids = children(chat_id, leaf)
        if not kids:
            break
        leaf = kids[-1]
    return leaf


def set_active(chat_id: str, message_id: int) -> None:
    m = get_message(message_id)
    if not m or m["chat_id"] != chat_id:
        raise ValueError("That message is not in this chat.")
    _x("UPDATE chats SET active_leaf = ? WHERE id = ?", (newest_leaf(chat_id, message_id), chat_id))


def subtree(chat_id: str, message_id: int) -> list[int]:
    ids, todo = [], [message_id]
    while todo:
        mid = todo.pop()
        ids.append(mid)
        todo += children(chat_id, mid)
    return ids


def delete_from(chat_id: str, message_id: int) -> float:
    """Hide a message and everything after it, in every branch. Returns a stamp for restore_from."""
    m = get_message(message_id)
    if not m or m["chat_id"] != chat_id:
        raise ValueError("That message is not in this chat.")
    stamp = time.time()
    ids = subtree(chat_id, message_id)
    _x(f"UPDATE messages SET deleted_at = ? WHERE id IN ({', '.join('?' for _ in ids)})", (stamp, *ids))
    # Show the closest remaining branch: a sibling if there is one, else the parent.
    siblings = children(chat_id, m["parent_id"])
    if siblings:
        leaf: int | None = newest_leaf(chat_id, siblings[-1])
    else:
        leaf = m["parent_id"]
    _x("UPDATE chats SET active_leaf = ? WHERE id = ?", (leaf, chat_id))
    return stamp


def restore_from(chat_id: str, message_id: int, stamp: float) -> None:
    _x("UPDATE messages SET deleted_at = NULL WHERE chat_id = ? AND deleted_at = ?", (chat_id, stamp))
    if get_message(message_id):
        set_active(chat_id, message_id)


def running_turns() -> list[dict]:
    return [_message(r) for r in _q("SELECT * FROM messages WHERE status = 'streaming'")]


def history_for(leaf_id: int | None) -> list[dict]:
    """The path the model sees: drops replies that failed or never produced text."""
    return [m for m in path(leaf_id) if m["role"] == "user" or (m["content"].strip() and m["status"] != "error")]


# ── Jobs and images under messages ──────────────────────────────────


def link_job(job_id: str, chat_id: str, message_id: int | None, kind: str) -> None:
    _x(
        "INSERT OR REPLACE INTO chat_jobs (job_id, chat_id, message_id, kind, created_at) VALUES (?, ?, ?, ?, ?)",
        (job_id, chat_id, message_id, kind, time.time()),
    )


def job_message(job_id: str | None) -> int | None:
    if not job_id:
        return None
    rows = _q("SELECT message_id FROM chat_jobs WHERE job_id = ?", (job_id,))
    return rows[0]["message_id"] if rows else None


def chat_jobs(chat_id: str) -> dict[str, int | None]:
    return {r["job_id"]: r["message_id"] for r in _q("SELECT job_id, message_id FROM chat_jobs WHERE chat_id = ?", (chat_id,))}


def outputs(chat_id: str, messages: list[dict]) -> dict[int, list[dict]]:
    """Generated images grouped under the assistant message they came from.

    Images from before messages and jobs were linked are matched by their
    prompt text against the directions each message proposed.
    """
    by_job = chat_jobs(chat_id)
    images = with_primary(
        [_image_dict(r) for r in _q("SELECT * FROM images WHERE chat_id = ? AND kind = 'generated' ORDER BY created_at", (chat_id,))]
    )
    ratings = current_ratings([i["id"] for i in images])
    visible = {m["id"] for m in messages}
    by_prompt: dict[str, int] = {}
    for m in messages:
        for p in m["prompts"]:
            by_prompt.setdefault(p.get("prompt", ""), m["id"])
    out: dict[int, list[dict]] = {}
    for img in images:
        img["rating"] = ratings.get(img["id"])
        mid = by_job.get(img.get("job_id") or "")
        if mid is None and img.get("job_id") not in by_job:
            mid = by_prompt.get(img.get("prompt") or "")
        if mid in visible:
            out.setdefault(mid, []).append(img)  # type: ignore[arg-type]
    return out


def costs_by_message(chat_id: str) -> dict[int, dict]:
    rows = _q(
        """SELECT message_id, COALESCE(SUM(credits), 0) AS credits, COALESCE(SUM(byok), 0) AS byok
           FROM costs WHERE chat_id = ? AND message_id IS NOT NULL GROUP BY message_id""",
        (chat_id,),
    )
    return {r["message_id"]: {"credits": round(r["credits"], 6), "byok": round(r["byok"], 6)} for r in rows}


# ── Feedback ────────────────────────────────────────────────────────

FEEDBACK_KINDS = ("like", "dislike", "unrate", "pick")


def add_feedback(kind: str, image_id: str | None = None, chat_id: str | None = None, message_id: int | None = None,
                 data: dict | None = None) -> dict:
    if kind not in FEEDBACK_KINDS:
        raise ValueError(f"Unknown feedback kind {kind!r}.")
    img = None
    if image_id:
        rows = _q("SELECT * FROM images WHERE id = ?", (image_id,))
        if not rows:
            raise ValueError("Unknown image.")
        img = dict(rows[0])
        chat_id = chat_id or img.get("chat_id")
        message_id = message_id or job_message(img.get("job_id"))
    cur = _x(
        """INSERT INTO feedback_events (created_at, kind, image_id, chat_id, message_id, model, prompt, concept, style, data)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (time.time(), kind, image_id, chat_id, message_id, (img or {}).get("model") or "", (img or {}).get("prompt") or "",
         (img or {}).get("concept") or "", (img or {}).get("style") or "", json.dumps(data or {})),
    )
    return feedback_event(int(cur.lastrowid or 0))  # type: ignore[return-value]


def feedback_event(event_id: int) -> dict | None:
    rows = _q("SELECT * FROM feedback_events WHERE id = ?", (event_id,))
    if not rows:
        return None
    d = dict(rows[0])
    d["data"] = json.loads(d["data"] or "{}")
    return d


def list_feedback(since_id: int = 0, chat_id: str | None = None, limit: int = 500) -> list[dict]:
    sql, args = "SELECT * FROM feedback_events WHERE id > ?", [since_id]
    if chat_id:
        sql += " AND chat_id = ?"
        args.append(chat_id)
    sql += " ORDER BY id LIMIT ?"
    args.append(limit)
    out = []
    for r in _q(sql, tuple(args)):
        d = dict(r)
        d["data"] = json.loads(d["data"] or "{}")
        out.append(d)
    return out


def current_ratings(image_ids: list[str]) -> dict[str, str]:
    """like or dislike per image, from the latest rating event (unrate clears it)."""
    if not image_ids:
        return {}
    marks = ", ".join("?" for _ in image_ids)
    rows = _q(
        f"""SELECT image_id, kind FROM feedback_events f WHERE image_id IN ({marks})
            AND kind IN ('like', 'dislike', 'unrate')
            AND id = (SELECT MAX(id) FROM feedback_events g WHERE g.image_id = f.image_id AND g.kind IN ('like', 'dislike', 'unrate'))""",
        tuple(image_ids),
    )
    return {r["image_id"]: r["kind"] for r in rows if r["kind"] != "unrate"}


# ── The whole chat for the UI ───────────────────────────────────────


def view(chat_id: str) -> dict | None:
    chat = get(chat_id)
    if not chat:
        return None
    msgs = path(chat["active_leaf"])
    costs = costs_by_message(chat_id)
    outs = outputs(chat_id, msgs)
    for m in msgs:
        sib = children(chat_id, m["parent_id"])
        m["siblings"] = sib
        m["cost"] = costs.get(m["id"])
        m["outputs"] = outs.get(m["id"], [])
    chat["messages"] = msgs
    return chat
