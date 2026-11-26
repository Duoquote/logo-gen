"""SQLite store for the image library and chat sessions (data/library.sqlite3)."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from logo_gen import paths

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    path TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,              -- generated | upscaled | cleaned
    parent_id TEXT,                  -- source image for derived images
    job_id TEXT,
    chat_id TEXT,
    prompt TEXT DEFAULT '',
    concept TEXT DEFAULT '',
    style TEXT DEFAULT '',
    model TEXT DEFAULT '',
    seed INTEGER,
    width INTEGER,
    height INTEGER,
    media_type TEXT DEFAULT 'image/png',
    cost REAL DEFAULT 0,
    gen_time REAL DEFAULT 0,
    favorite INTEGER DEFAULT 0,
    note TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS images_created ON images(created_at DESC);
CREATE INDEX IF NOT EXISTS images_parent ON images(parent_id);

CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    images TEXT DEFAULT '[]',        -- uploaded reference paths (relative to output/)
    prompts TEXT DEFAULT '[]',       -- prompt directions parsed from the reply
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id);
"""


def conn() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(paths.DB_FILE, check_same_thread=False, isolation_level=None)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.executescript(SCHEMA)
            _conn = c
        return _conn


def _q(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    with _lock:
        return conn().execute(sql, args).fetchall()


def _x(sql: str, args: tuple = ()) -> sqlite3.Cursor:
    with _lock:
        return conn().execute(sql, args)


# ── Images ───────────────────────────────────────────────────────────


def _image_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["favorite"] = bool(d["favorite"])
    d["url"] = f"/api/files/{d['path']}"
    d["vector"] = d["media_type"] == "image/svg+xml"
    return d


def add_image(path: Path, kind: str, **meta: Any) -> dict:
    image_id = uuid.uuid4().hex[:16]
    width = height = None
    media_type = meta.pop("media_type", None) or ("image/svg+xml" if path.suffix == ".svg" else "image/png")
    if media_type != "image/svg+xml":
        try:
            with Image.open(path) as im:
                width, height = im.size
        except OSError:
            pass
    fields = {
        "id": image_id,
        "created_at": meta.pop("created_at", None) or time.time(),
        "path": paths.rel_output(path),
        "kind": kind,
        "width": width,
        "height": height,
        "media_type": media_type,
        **{k: v for k, v in meta.items() if v is not None},
    }
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    _x(f"INSERT INTO images ({cols}) VALUES ({marks})", tuple(fields.values()))
    return get_image(image_id)  # type: ignore[return-value]


def get_image(image_id: str) -> dict | None:
    rows = _q("SELECT * FROM images WHERE id = ?", (image_id,))
    return _image_dict(rows[0]) if rows else None


def list_images(
    kind: str | None = None,
    favorite: bool | None = None,
    q: str | None = None,
    chat_id: str | None = None,
    job_id: str | None = None,
    ids: list[str] | None = None,
    limit: int = 200,
    offset: int = 0,
) -> list[dict]:
    where, args = [], []
    if kind:
        where.append("kind = ?")
        args.append(kind)
    if favorite:
        where.append("favorite = 1")
    if q:
        where.append("(prompt LIKE ? OR concept LIKE ? OR model LIKE ? OR style LIKE ?)")
        args += [f"%{q}%"] * 4
    if chat_id:
        where.append("chat_id = ?")
        args.append(chat_id)
    if job_id:
        where.append("job_id = ?")
        args.append(job_id)
    if ids is not None:
        if not ids:
            return []
        where.append(f"id IN ({', '.join('?' for _ in ids)})")
        args += ids
    sql = "SELECT * FROM images"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    args += [limit, offset]
    return [_image_dict(r) for r in _q(sql, tuple(args))]


def children(image_id: str) -> list[dict]:
    return [_image_dict(r) for r in _q("SELECT * FROM images WHERE parent_id = ? ORDER BY created_at", (image_id,))]


def update_image(image_id: str, **fields: Any) -> dict | None:
    allowed = {k: v for k, v in fields.items() if k in ("favorite", "note")}
    if allowed:
        sets = ", ".join(f"{k} = ?" for k in allowed)
        _x(f"UPDATE images SET {sets} WHERE id = ?", (*[int(v) if k == "favorite" else v for k, v in allowed.items()], image_id))
    return get_image(image_id)


def delete_image(image_id: str) -> bool:
    img = get_image(image_id)
    if not img:
        return False
    for child in children(image_id):
        delete_image(child["id"])
    try:
        paths.resolve_output(img["path"]).unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
    _x("DELETE FROM images WHERE id = ?", (image_id,))
    return True


def stats() -> dict:
    r = _q("SELECT COUNT(*) AS n, COALESCE(SUM(cost), 0) AS cost FROM images")[0]
    return {"images": r["n"], "cost": round(r["cost"], 4)}


def import_existing() -> int:
    """Register images already sitting in output/ (from older versions)."""
    known = {r["path"] for r in _q("SELECT path FROM images")}
    added = 0
    for kind, d in (("generated", paths.GENERATED_DIR), ("upscaled", paths.UPSCALED_DIR), ("cleaned", paths.CLEANED_DIR)):
        if not d.exists():
            continue
        for p in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".svg"):
                continue
            if paths.rel_output(p) in known:
                continue
            add_image(p, kind, created_at=p.stat().st_mtime, model=_guess_model(p.name))
            added += 1
    return added


def _guess_model(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    if "_s" in stem:
        stem = stem.split("_s")[0]
    return stem.replace("_", "/", 1) if "_" in stem else stem


# ── Chats ────────────────────────────────────────────────────────────


def create_chat(title: str = "") -> dict:
    now = time.time()
    chat_id = uuid.uuid4().hex[:16]
    _x("INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)", (chat_id, title, now, now))
    return get_chat(chat_id)  # type: ignore[return-value]


def list_chats() -> list[dict]:
    rows = _q(
        """SELECT c.*, COUNT(m.id) AS message_count FROM chats c
           LEFT JOIN messages m ON m.chat_id = c.id
           GROUP BY c.id ORDER BY c.updated_at DESC"""
    )
    return [dict(r) for r in rows]


def get_chat(chat_id: str) -> dict | None:
    rows = _q("SELECT * FROM chats WHERE id = ?", (chat_id,))
    if not rows:
        return None
    chat = dict(rows[0])
    chat["messages"] = [
        {
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "images": json.loads(m["images"] or "[]"),
            "prompts": json.loads(m["prompts"] or "[]"),
            "created_at": m["created_at"],
        }
        for m in _q("SELECT * FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,))
    ]
    return chat


def add_message(chat_id: str, role: str, content: str, images: list[str] | None = None, prompts: list[dict] | None = None) -> int:
    now = time.time()
    cur = _x(
        "INSERT INTO messages (chat_id, role, content, images, prompts, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (chat_id, role, content, json.dumps(images or []), json.dumps(prompts or []), now),
    )
    _x("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
    if role == "user":
        _x(
            "UPDATE chats SET title = ? WHERE id = ? AND (title IS NULL OR title = '')",
            (" ".join(content.split())[:60], chat_id),
        )
    return int(cur.lastrowid or 0)


def rename_chat(chat_id: str, title: str) -> None:
    _x("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))


def delete_chat(chat_id: str) -> None:
    _x("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    _x("DELETE FROM chats WHERE id = ?", (chat_id,))
