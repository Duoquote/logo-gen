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

-- Every paid or local call, so spend survives restarts. credits is what
-- OpenRouter charged; byok is what an upstream provider billed on the
-- user's own key (OpenRouter reports it as upstream_inference_cost).
CREATE TABLE IF NOT EXISTS costs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL,              -- chat | prompts | image
    model TEXT DEFAULT '',
    chat_id TEXT,
    job_id TEXT,
    generation_id TEXT,
    credits REAL DEFAULT 0,
    byok REAL DEFAULT 0,
    images INTEGER DEFAULT 0,
    local INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS costs_chat ON costs(chat_id);
CREATE INDEX IF NOT EXISTS costs_job ON costs(job_id);

-- Which chat message a generation job belongs to, so its images show under that message.
CREATE TABLE IF NOT EXISTS chat_jobs (
    job_id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    message_id INTEGER,
    kind TEXT DEFAULT '',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_jobs_message ON chat_jobs(message_id);

-- Reactions to images in the designer (like, dislike, unrate, pick). Append-only;
-- the latest like/dislike/unrate per image is its current rating. The prompt
-- layer reads these to learn what the user likes.
CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    kind TEXT NOT NULL,
    image_id TEXT,
    chat_id TEXT,
    message_id INTEGER,
    model TEXT DEFAULT '',
    prompt TEXT DEFAULT '',
    concept TEXT DEFAULT '',
    style TEXT DEFAULT '',
    data TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS feedback_image ON feedback_events(image_id, id);
CREATE INDEX IF NOT EXISTS feedback_chat ON feedback_events(chat_id, id);

-- Small key/value state, such as the designer chat the user had open last.
CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS calibrations (
    model_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    data TEXT NOT NULL
);
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
            _migrate(c)
            _conn = c
        return _conn


def _columns(c: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in c.execute(f"PRAGMA table_info({table})")}


def _migrate(c: sqlite3.Connection) -> None:
    _migrate_chats(c)
    if "message_id" not in _columns(c, "costs"):
        c.execute("ALTER TABLE costs ADD COLUMN message_id INTEGER")
        c.execute("CREATE INDEX IF NOT EXISTS costs_message ON costs(message_id)")
    cols = _columns(c, "images")
    if "primary_id" not in cols:
        # The version of an image the user picked to keep (an upscale or a cleaned copy).
        c.execute("ALTER TABLE images ADD COLUMN primary_id TEXT")
    # Spend from before the costs table existed lived only on images.
    if c.execute("SELECT COUNT(*) FROM costs").fetchone()[0] == 0:
        c.execute(
            """INSERT INTO costs (created_at, kind, model, chat_id, job_id, credits, images, local)
               SELECT created_at, 'image', model, chat_id, job_id, cost, 1, CASE WHEN model LIKE 'local/%' THEN 1 ELSE 0 END
               FROM images WHERE kind = 'generated'"""
        )


def _migrate_chats(c: sqlite3.Connection) -> None:
    """Chats became a tree of messages (edits and regenerations are branches) with server-side turn state."""
    mcols = _columns(c, "messages")
    if "parent_id" not in mcols:
        for col in (
            "parent_id INTEGER",
            "status TEXT DEFAULT 'done'",  # streaming | done | stopped | error
            "reasoning TEXT DEFAULT ''",
            "error TEXT DEFAULT ''",
            "model TEXT DEFAULT ''",
            "finished_at REAL",
            "deleted_at REAL",
        ):
            c.execute(f"ALTER TABLE messages ADD COLUMN {col}")
        # Old chats were linear: each message follows the one before it.
        c.execute(
            """UPDATE messages SET parent_id = (
                   SELECT MAX(p.id) FROM messages p WHERE p.chat_id = messages.chat_id AND p.id < messages.id)"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS messages_parent ON messages(chat_id, parent_id)")
    ccols = _columns(c, "chats")
    if "active_leaf" not in ccols:
        for col in ("pinned INTEGER DEFAULT 0", "deleted_at REAL", "active_leaf INTEGER", "last_opened_at REAL"):
            c.execute(f"ALTER TABLE chats ADD COLUMN {col}")
        c.execute("UPDATE chats SET active_leaf = (SELECT MAX(id) FROM messages m WHERE m.chat_id = chats.id)")


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


def with_primary(images: list[dict]) -> list[dict]:
    """Attach the chosen version (if any) to each image."""
    ids = [i["primary_id"] for i in images if i.get("primary_id")]
    if not ids:
        return images
    rows = {r["id"]: _image_dict(r) for r in _q(f"SELECT * FROM images WHERE id IN ({', '.join('?' for _ in ids)})", tuple(ids))}
    for i in images:
        i["primary"] = rows.get(i.get("primary_id") or "")
    return images


def set_primary(image_id: str, child_id: str | None) -> dict | None:
    if child_id:
        child = get_image(child_id)
        if not child or child["parent_id"] != image_id:
            raise ValueError("That version does not belong to this image.")
    _x("UPDATE images SET primary_id = ? WHERE id = ?", (child_id, image_id))
    return get_image(image_id)


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
    return with_primary([_image_dict(r) for r in _q(sql, tuple(args))])


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
    _x("UPDATE images SET primary_id = NULL WHERE primary_id = ?", (image_id,))
    _x("DELETE FROM images WHERE id = ?", (image_id,))
    return True


def stats() -> dict:
    r = _q("SELECT COUNT(*) AS n FROM images")[0]
    return {"images": r["n"], "cost": spend()["total"]}


# ── Spend ────────────────────────────────────────────────────────────


def add_cost(kind: str, *, credits: float = 0.0, byok: float = 0.0, model: str = "", chat_id: str | None = None,
             job_id: str | None = None, generation_id: str | None = None, images: int = 0, local: bool = False,
             prompt_tokens: int = 0, completion_tokens: int = 0, message_id: int | None = None) -> None:
    _x(
        """INSERT INTO costs (created_at, kind, model, chat_id, job_id, generation_id, credits, byok, images, local,
                              prompt_tokens, completion_tokens, message_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (time.time(), kind, model, chat_id, job_id, generation_id, float(credits or 0), float(byok or 0), images,
         int(local), int(prompt_tokens or 0), int(completion_tokens or 0), message_id),
    )


def spend(chat_id: str | None = None, job_id: str | None = None) -> dict:
    where, args = [], []
    if chat_id:
        where.append("chat_id = ?")
        args.append(chat_id)
    if job_id:
        where.append("job_id = ?")
        args.append(job_id)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    rows = _q(
        f"""SELECT kind, local, COUNT(*) AS calls, COALESCE(SUM(credits), 0) AS credits, COALESCE(SUM(byok), 0) AS byok,
                   COALESCE(SUM(images), 0) AS images FROM costs{clause} GROUP BY kind, local""",
        tuple(args),
    )
    out = {"credits": 0.0, "byok": 0.0, "total": 0.0, "chat_calls": 0, "cloud_images": 0, "local_images": 0, "by_kind": {}}
    for r in rows:
        out["credits"] += r["credits"]
        out["byok"] += r["byok"]
        k = out["by_kind"].setdefault(r["kind"], {"credits": 0.0, "byok": 0.0, "calls": 0})
        k["credits"] += r["credits"]
        k["byok"] += r["byok"]
        k["calls"] += r["calls"]
        if r["kind"] == "image":
            out["local_images" if r["local"] else "cloud_images"] += r["images"]
        else:
            out["chat_calls"] += r["calls"]
    out["total"] = out["credits"] + out["byok"]
    for key in ("credits", "byok", "total"):
        out[key] = round(out[key], 6)
    return out


# ── Calibration ──────────────────────────────────────────────────────


def save_calibration(model_id: str, data: dict) -> None:
    _x("INSERT OR REPLACE INTO calibrations (model_id, created_at, data) VALUES (?, ?, ?)",
       (model_id, time.time(), json.dumps(data)))


def get_calibration(model_id: str) -> dict | None:
    rows = _q("SELECT * FROM calibrations WHERE model_id = ?", (model_id,))
    if not rows:
        return None
    return json.loads(rows[0]["data"]) | {"created_at": rows[0]["created_at"]}


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
