"""Settings.

Two stores, both inside the project folder:

- ``.env`` holds secrets (OPENROUTER_KEY, HF_TOKEN). It is gitignored.
  Real environment variables win over the file.
- ``data/settings.json`` holds everything else and is written by the UI.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Literal

from dotenv import dotenv_values, set_key, unset_key
from pydantic import BaseModel, Field

from logo_gen import paths

SECRET_KEYS = ("OPENROUTER_KEY", "HF_TOKEN")

# Older .env files and other tools use these names for the same thing.
_SECRET_ALIASES = {
    "OPENROUTER_KEY": ("OPENROUTER_KEY", "OPENROUTER_API_KEY"),
    "HF_TOKEN": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
}


class Settings(BaseModel):
    # Chat / prompt writing
    llm_backend: Literal["openrouter", "local", "claude-cli"] = "openrouter"
    chat_model: str = "anthropic/claude-haiku-5.5"
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "xhigh"
    local_llm_base_url: str = "http://localhost:11434/v1"
    local_llm_model: str = "qwen3.5:9b"
    local_llm_api_key: str = ""

    # Image generation. OpenRouter ids as-is, local models as "local/<id>".
    image_models: list[str] = Field(default_factory=lambda: ["recraft/recraft-v4.1-flash"])
    images_per_prompt: int = 1
    variations: int = 4
    aspect_ratio: str = "1:1"
    transparent_background: bool = False

    # Learned preferences: also keep a taste profile across chats and use it as a weak hint.
    use_global_taste: bool = True

    # Limits: the most images one model may make in a single request.
    max_images_cloud: int = Field(default=8, ge=1, le=100)
    max_images_local: int = Field(default=16, ge=1, le=200)
    model_caps: dict[str, int] = Field(default_factory=dict)  # per-model override, by model id

    # Throughput
    cloud_concurrency: int = Field(default=4, ge=1, le=16)  # parallel OpenRouter image requests
    local_batch: dict[str, int] = Field(default_factory=dict)  # chosen batch size per local model id

    def cap_for(self, model: str) -> int:
        """Max images this model may make per request."""
        if model in self.model_caps:
            return max(1, int(self.model_caps[model]))
        return self.max_images_local if model.startswith("local/") else self.max_images_cloud


_lock = threading.Lock()
_settings: Settings | None = None


def _migrate(raw: dict) -> dict:
    """Accept the field names the Flask version used."""
    if "llm_model" in raw and "chat_model" not in raw:
        model = raw.pop("llm_model")
        if isinstance(model, str) and model.startswith("local/"):
            raw["llm_backend"] = "local"
            raw["local_llm_model"] = model.removeprefix("local/")
        else:
            raw["chat_model"] = model
    if raw.pop("use_claude_cli", False):
        raw["llm_backend"] = "claude-cli"
    if "images_per_model" in raw and "images_per_prompt" not in raw:
        raw["images_per_prompt"] = raw.pop("images_per_model")
    if "image_models" in raw:
        fixed = []
        for m in raw["image_models"]:
            name = m.split(":")[0] if m.startswith(("flux1-schnell", "zimage-turbo")) else m
            if name in ("flux1-schnell", "zimage-turbo"):
                name = f"local/{name}"
            fixed.append(name)
        raw["image_models"] = fixed
    return raw


def get_settings() -> Settings:
    global _settings
    with _lock:
        if _settings is None:
            raw: dict = {}
            if paths.SETTINGS_FILE.exists():
                try:
                    raw = json.loads(paths.SETTINGS_FILE.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    raw = {}
            _settings = Settings.model_validate(_migrate(raw))
        return _settings


def update_settings(patch: dict) -> Settings:
    global _settings
    current = get_settings()
    merged = current.model_dump() | {k: v for k, v in patch.items() if k in Settings.model_fields}
    new = Settings.model_validate(merged)
    with _lock:
        paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = paths.SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(new.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(paths.SETTINGS_FILE)
        _settings = new
    return new


def reset_settings() -> Settings:
    defaults = Settings().model_dump()
    return update_settings(defaults)


# ── Secrets ──────────────────────────────────────────────────────────


def get_secret(name: str) -> str:
    for alias in _SECRET_ALIASES.get(name, (name,)):
        val = os.environ.get(alias)
        if val:
            return val.strip()
    if paths.ENV_FILE.exists():
        values = dotenv_values(paths.ENV_FILE, encoding="utf-8")
        for alias in _SECRET_ALIASES.get(name, (name,)):
            val = values.get(alias)
            if val:
                return val.strip()
    return ""


def secret_source(name: str) -> Literal["env", "file", "none"]:
    for alias in _SECRET_ALIASES.get(name, (name,)):
        if os.environ.get(alias):
            return "env"
    if paths.ENV_FILE.exists():
        values = dotenv_values(paths.ENV_FILE, encoding="utf-8")
        if any(values.get(a) for a in _SECRET_ALIASES.get(name, (name,))):
            return "file"
    return "none"


def set_secret(name: str, value: str) -> None:
    """Write or clear a secret in the project .env file."""
    if name not in SECRET_KEYS:
        raise ValueError(f"unknown secret {name}")
    value = value.strip()
    if not paths.ENV_FILE.exists():
        paths.ENV_FILE.touch()
    if value:
        set_key(str(paths.ENV_FILE), name, value, quote_mode="never", encoding="utf-8")
    else:
        unset_key(str(paths.ENV_FILE), name, encoding="utf-8")


def mask(value: str) -> str:
    if not value:
        return ""
    return f"ends in {value[-4:]}" if len(value) > 16 else "set"
