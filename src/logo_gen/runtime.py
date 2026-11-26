"""Install local model support (PyTorch, diffusers, ...) on demand.

A cloud-only install never downloads PyTorch. The first time a local model is
installed, this runs `uv sync --extra local --extra <torch build>` into the
project's .venv and records the choice in .tools/local-stack so the launcher
keeps it on later runs.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from logo_gen import paths, system
from logo_gen.spawn import NO_CONSOLE

MARKER = paths.ROOT / ".tools" / "local-stack"


@dataclass
class RuntimeState:
    running: bool = False
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    log: deque = field(default_factory=lambda: deque(maxlen=40))


state = RuntimeState()
_lock = asyncio.Lock()


def uv_path() -> str | None:
    env = os.environ.get("LOGO_GEN_UV")
    if env and Path(env).exists():
        return env
    found = shutil.which("uv")
    if found:
        return found
    local = paths.ROOT / ".tools" / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")
    return str(local) if local.exists() else None


def status() -> dict:
    return {
        "installed": system.local_stack_installed(),
        "variant": system.torch_variant(),
        "torch": system.torch_build() or None,
        "running": state.running,
        "error": state.error,
        "log": list(state.log)[-12:],
        "elapsed": round((state.finished_at or time.time()) - state.started_at, 1) if state.started_at else 0,
    }


async def ensure_installed() -> None:
    """Install the local stack if missing. Safe to call concurrently."""
    if system.local_stack_installed():
        return
    async with _lock:
        if system.local_stack_installed():
            return
        state.running, state.error, state.started_at, state.finished_at = True, "", time.time(), 0.0
        state.log.clear()
        try:
            await _sync()
        except Exception as e:
            state.error = str(e)
            raise
        finally:
            state.running = False
            state.finished_at = time.time()


async def _sync() -> None:
    uv = uv_path()
    if not uv:
        raise RuntimeError("uv was not found. Start logo-gen with ./logo-gen so it can set uv up.")
    variant = system.torch_variant()
    state.log.append(f"Installing local model support with PyTorch ({variant}). This downloads about "
                     f"{'2.5 GB' if variant.startswith('cu') else '300 MB'} once.")
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(Path(sys.prefix))
    proc = await asyncio.create_subprocess_exec(
        uv, "sync", "--inexact", "--extra", "local", "--extra", variant,
        cwd=str(paths.ROOT), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        **NO_CONSOLE,
    )
    assert proc.stdout
    async for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").strip()
        if line:
            state.log.append(line)
    code = await proc.wait()
    if code != 0:
        tail = " ".join(list(state.log)[-3:])
        raise RuntimeError(f"Installing local model support failed: {tail[:400]}")
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(variant, encoding="utf-8")
    importlib.invalidate_caches()
    state.log.append("Done.")
