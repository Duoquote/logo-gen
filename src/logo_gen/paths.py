"""Project-level paths. Everything the app writes lives under the project folder."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("LOGO_GEN_HOME") or Path(__file__).resolve().parents[2])

ENV_FILE = ROOT / ".env"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / ".models"
FRONTEND_DIST = ROOT / "frontend" / "dist"

SETTINGS_FILE = DATA_DIR / "settings.json"
DB_FILE = DATA_DIR / "library.sqlite3"

GENERATED_DIR = OUTPUT_DIR / "generated"
UPSCALED_DIR = OUTPUT_DIR / "upscaled"
CLEANED_DIR = OUTPUT_DIR / "cleaned"
UPLOADS_DIR = OUTPUT_DIR / "uploads"


def ensure_dirs() -> None:
    for d in (DATA_DIR, GENERATED_DIR, UPSCALED_DIR, CLEANED_DIR, UPLOADS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def rel_output(path: Path) -> str:
    """Path relative to output/, with forward slashes."""
    return path.resolve().relative_to(OUTPUT_DIR.resolve()).as_posix()


def resolve_output(rel: str) -> Path:
    """Resolve a path relative to output/ and refuse anything that escapes it."""
    p = (OUTPUT_DIR / rel).resolve()
    if not p.is_relative_to(OUTPUT_DIR.resolve()):
        raise ValueError("path outside output folder")
    return p
