"""Hardware and environment info. Standard library only, so the launcher can use it too.

Set LOGO_GEN_DEVICE=cpu (or CUDA_VISIBLE_DEVICES to an empty string) to make
the app behave as if there were no GPU.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import platform
import re
import shutil
import subprocess
from importlib import metadata

from logo_gen import paths
from logo_gen.spawn import NO_CONSOLE


def forced_device() -> str | None:
    forced = os.environ.get("LOGO_GEN_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda", "mps"):
        return forced
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return "cpu"
    return None


@functools.cache
def _nvidia_smi() -> tuple[dict, str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}, ""
    try:
        q = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, **NO_CONSOLE,
        ).stdout.strip().splitlines()
        header = subprocess.run([exe], capture_output=True, text=True, timeout=10, **NO_CONSOLE).stdout
    except (OSError, subprocess.SubprocessError):
        return {}, ""
    if not q:
        return {}, ""
    name, mem, driver = [p.strip() for p in q[0].split(",")[:3]]
    try:
        vram_gb = round(float(mem) / 1024, 1)
    except ValueError:
        vram_gb = 0.0
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", header)
    return {"name": name, "vram_gb": vram_gb, "driver": driver}, (m.group(1) if m else "")


def gpu() -> dict:
    """First NVIDIA GPU, or {} when there is none (or CPU is forced)."""
    if forced_device() in ("cpu", "mps"):
        return {}
    return _nvidia_smi()[0]


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def torch_variant() -> str:
    """Which PyTorch build to install here: cu130, cu126 or cpu (macOS gets MPS from the cpu extra)."""
    forced = forced_device()
    if forced == "cpu" or platform.system() == "Darwin":
        return "cpu"
    info, cuda = _nvidia_smi()
    if not info:
        return "cpu"
    try:
        major, minor = (int(x) for x in (cuda.split(".") + ["0"])[:2])
    except ValueError:
        return "cu126"
    if (major, minor) >= (13, 0):
        return "cu130"
    if (major, minor) >= (12, 6):
        return "cu126"
    return "cpu"


def torch_build() -> str:
    """Installed torch version, e.g. 2.14.1+cu130, or '' if not installed."""
    try:
        return metadata.version("torch")
    except metadata.PackageNotFoundError:
        return ""


def local_stack_installed() -> bool:
    importlib.invalidate_caches()
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers", "transformers"))


def device_kind() -> str:
    """Where local models run: cuda, mps or cpu."""
    forced = forced_device()
    if forced:
        return forced
    if gpu() and ("+cu" in torch_build() or not torch_build()):
        return "cuda"
    if is_apple_silicon():
        return "mps"
    return "cpu"


def disk_free_gb() -> float:
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return round(shutil.disk_usage(paths.MODELS_DIR).free / 1e9, 1)


def info() -> dict:
    return {
        "gpu": gpu() or None,
        "device": device_kind(),
        "forced_device": forced_device(),
        "torch": torch_build() or None,
        "torch_variant": torch_variant(),
        "local_stack": local_stack_installed(),
        "disk_free_gb": disk_free_gb(),
        "platform": f"{platform.system()} {platform.machine()}",
        "models_dir": str(paths.MODELS_DIR),
    }
