"""Install, track, cancel and remove local models.

Files go to ``.models/<model id>/`` in the repo layout, so a pipeline loads
straight from that folder. A download runs in a subprocess (worker.py); cancel
kills it and the ``.part`` files stay for the next attempt.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from logo_gen import paths, system
from logo_gen.config import get_secret
from logo_gen.local_models.registry import MODELS, LocalModel

MANIFEST = ".logo-gen.json"
REMOTE_TTL = 600


@dataclass
class InstallState:
    phase: str = "checking"  # checking | runtime | downloading
    downloaded: int = 0
    total: int = 0
    speed: float = 0.0
    message: str = ""
    started_at: float = field(default_factory=time.time)
    proc: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None
    cancelled: bool = False


_states: dict[str, InstallState] = {}
_errors: dict[str, str] = {}
_remote: dict[str, tuple[float, dict]] = {}
_access: dict[str, tuple[float, dict]] = {}


def model_dir(spec: LocalModel) -> Path:
    return paths.MODELS_DIR / spec.id


def _token() -> str | None:
    return get_secret("HF_TOKEN") or None


def _manifest(spec: LocalModel) -> dict | None:
    p = model_dir(spec) / MANIFEST
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def is_installed(spec: LocalModel) -> bool:
    man = _manifest(spec)
    if not man or not man.get("complete"):
        return False
    root = model_dir(spec)
    return all((root / f["path"]).exists() for f in man.get("files", []))


def _matches(path: str, spec: LocalModel) -> bool:
    if not any(fnmatch.fnmatch(path, pat) for pat in spec.allow_patterns):
        return False
    return not any(fnmatch.fnmatch(path, pat) for pat in spec.ignore_patterns)


def _remote_files_sync(spec: LocalModel) -> dict:
    info = HfApi().model_info(spec.repo, files_metadata=True, token=_token())
    files = [
        {"path": s.rfilename, "size": int(s.size or 0)}
        for s in (info.siblings or [])
        if _matches(s.rfilename, spec)
    ]
    return {"revision": info.sha, "files": files, "total": sum(f["size"] for f in files), "gated": info.gated}


async def remote_files(spec: LocalModel, force: bool = False) -> dict:
    hit = _remote.get(spec.id)
    if hit and not force and time.time() - hit[0] < REMOTE_TTL:
        return hit[1]
    data = await asyncio.to_thread(_remote_files_sync, spec)
    _remote[spec.id] = (time.time(), data)
    return data


def _access_sync(spec: LocalModel) -> dict:
    token = _token()
    try:
        HfApi().auth_check(spec.repo, token=token)
        return {"state": "open", "message": ""}
    except GatedRepoError:
        if not token:
            return {
                "state": "needs_token",
                "message": f"This model is gated. Accept its terms at {spec.page_url} "
                           "while logged in, then add a Hugging Face token in Settings.",
            }
        return {
            "state": "gated",
            "message": f"Your Hugging Face token does not have access yet. Open {spec.page_url}, "
                       "accept the terms with the same account, then try again.",
        }
    except RepositoryNotFoundError:
        return {"state": "error", "message": f"{spec.repo} was not found on Hugging Face."}
    except HfHubHTTPError as e:
        return {"state": "unknown", "message": f"Could not check access: {e}"}
    except OSError as e:
        return {"state": "unknown", "message": f"Offline? {e}"}


async def access(spec: LocalModel, force: bool = False) -> dict:
    hit = _access.get(spec.id)
    if hit and not force and time.time() - hit[0] < 300:
        return hit[1]
    data = await asyncio.to_thread(_access_sync, spec)
    _access[spec.id] = (time.time(), data)
    return data


def _bytes_on_disk(spec: LocalModel, files: list[dict]) -> int:
    root = model_dir(spec)
    total = 0
    for f in files:
        target = root / f["path"]
        if target.exists():
            total += target.stat().st_size
        else:
            part = target.with_name(target.name + ".part")
            if part.exists():
                total += part.stat().st_size
    return total


def status(spec: LocalModel) -> dict:
    remote = _remote.get(spec.id, (0, None))[1]
    acc = _access.get(spec.id, (0, None))[1]
    st = _states.get(spec.id)
    installed = is_installed(spec)
    size = remote["total"] if remote else int(spec.size_gb * 1e9)
    gpu = system.gpu()
    device = system.device_kind()

    if installed:
        state = "installed"
    elif st:
        state = st.phase
    elif spec.id in _errors:
        state = "error"
    elif model_dir(spec).exists() and remote and _bytes_on_disk(spec, remote["files"]) > 0:
        state = "partial"
    else:
        state = "not_installed"

    downloaded = 0
    if st:
        downloaded = st.downloaded
    elif state == "partial" and remote:
        downloaded = _bytes_on_disk(spec, remote["files"])

    support = spec.support(device)
    fits = (gpu.get("vram_gb", 0) >= spec.vram_gb) if device == "cuda" else support != "no"
    if support == "no":
        note = "Needs an NVIDIA GPU with CUDA." if device == "cpu" else "Not supported on Apple Silicon; needs an NVIDIA GPU."
    elif support == "slow":
        note = ("Runs on the CPU, but expect a minute or more per image." if device == "cpu"
                else "Runs on Apple Silicon without 4-bit quantization; needs a lot of unified memory and is slow.")
    elif device == "cuda" and not fits:
        note = f"Needs about {spec.vram_gb:g} GB of VRAM; this GPU has {gpu.get('vram_gb', 0):g} GB."
    else:
        note = ""
    return {
        "id": spec.id,
        "ref": f"local/{spec.id}",
        "name": spec.name,
        "repo": spec.repo,
        "page_url": spec.page_url,
        "license": spec.license,
        "license_url": spec.license_url,
        "blurb": spec.blurb,
        "size_bytes": size,
        "vram_gb": spec.vram_gb,
        "resolution": spec.resolution,
        "steps": spec.steps,
        "needs_cuda": spec.needs_cuda,
        "device": device,
        "support": {"cuda": "ok", "mps": spec.mps, "cpu": spec.cpu},
        "runs_here": support if fits or support == "no" else "tight",
        "device_note": note,
        "gated": bool(remote and remote.get("gated")) or (acc or {}).get("state") in ("gated", "needs_token"),
        "access": acc,
        "state": state,
        "downloaded_bytes": downloaded,
        "speed_bps": st.speed if st else 0,
        "message": (st.message if st else "") or _errors.get(spec.id, ""),
        "installed": installed,
    }


async def refresh_remote() -> None:
    """Fetch sizes and access state for every model, ignoring failures."""
    async def one(spec: LocalModel) -> None:
        try:
            await remote_files(spec)
        except Exception:  # noqa: BLE001
            pass
        try:
            await access(spec)
        except Exception:  # noqa: BLE001
            pass
    await asyncio.gather(*(one(s) for s in MODELS.values()))


def list_models() -> list[dict]:
    return [status(s) for s in MODELS.values()]


# ── Install ──────────────────────────────────────────────────────────


async def _run_install(spec: LocalModel, st: InstallState) -> None:
    from logo_gen import runtime

    if spec.support(system.device_kind()) == "no":
        raise RuntimeError(f"{spec.name} cannot run on this machine ({system.device_kind()}).")
    acc = await access(spec, force=True)
    if acc["state"] in ("gated", "needs_token", "error"):
        raise RuntimeError(acc["message"])
    remote = await remote_files(spec, force=True)
    st.total = remote["total"]
    already = _bytes_on_disk(spec, remote["files"])
    need_gb = (st.total - already) / 1e9
    if system.disk_free_gb() < need_gb + 1:
        raise RuntimeError(f"Not enough disk space: {need_gb:.1f} GB needed, {system.disk_free_gb()} GB free.")

    if not system.local_stack_installed():
        st.phase = "runtime"
        st.message = "Installing local model support (PyTorch and diffusers)"
        await runtime.ensure_installed()
        st.message = ""

    root = model_dir(spec)
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    env.pop("HF_TOKEN", None)
    if _token():
        env["HF_TOKEN"] = _token()  # type: ignore[assignment]
    job = {"repo": spec.repo, "revision": remote["revision"], "dest": str(root), "files": remote["files"]}

    st.phase = "downloading"
    st.downloaded = already
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "logo_gen.local_models.worker",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
    )
    st.proc = proc
    assert proc.stdin
    proc.stdin.write(json.dumps(job).encode())
    await proc.stdin.drain()
    proc.stdin.close()

    async def drain(stream) -> bytes:
        return await stream.read()

    out_task = asyncio.create_task(drain(proc.stdout))
    err_task = asyncio.create_task(drain(proc.stderr))
    last_t, last_b = time.time(), already
    while proc.returncode is None:
        try:
            await asyncio.wait_for(proc.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            pass
        st.downloaded = _bytes_on_disk(spec, remote["files"])
        now = time.time()
        if now - last_t >= 1.0:
            st.speed = max(0.0, (st.downloaded - last_b) / (now - last_t))
            last_t, last_b = now, st.downloaded
    await out_task
    err = (await err_task).decode("utf-8", "replace").strip()

    if st.cancelled:
        return
    if proc.returncode != 0:
        last = err.splitlines()[-1] if err else f"exit code {proc.returncode}"
        if "401" in last or "403" in last or "Gated" in last:
            _access.pop(spec.id, None)
        raise RuntimeError(f"Download failed: {last[:300]}")

    (root / MANIFEST).write_text(json.dumps({
        "repo": spec.repo, "revision": remote["revision"], "files": remote["files"],
        "complete": True, "installed_at": time.time(),
    }, indent=2), encoding="utf-8")


def start_install(spec: LocalModel) -> dict:
    if is_installed(spec):
        return status(spec)
    if spec.id in _states:
        return status(spec)
    _errors.pop(spec.id, None)
    st = InstallState()
    _states[spec.id] = st

    async def runner() -> None:
        try:
            await _run_install(spec, st)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            _errors[spec.id] = str(e)
        finally:
            _states.pop(spec.id, None)

    st.task = asyncio.get_running_loop().create_task(runner())
    return status(spec)


async def cancel_install(spec: LocalModel) -> dict:
    st = _states.get(spec.id)
    if st:
        st.cancelled = True
        if st.proc and st.proc.returncode is None:
            st.proc.kill()
            await st.proc.wait()
        if st.task:
            try:
                await asyncio.wait_for(asyncio.shield(st.task), timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                st.task.cancel()
        _states.pop(spec.id, None)
    return status(spec)


async def uninstall(spec: LocalModel) -> dict:
    await cancel_install(spec)
    from logo_gen.local_models import diffusion

    diffusion.unload(spec.id)
    root = model_dir(spec)
    if root.exists():
        await asyncio.to_thread(shutil.rmtree, root, ignore_errors=True)
    _errors.pop(spec.id, None)
    return status(spec)


async def shutdown() -> None:
    for mid in list(_states):
        spec = MODELS.get(mid)
        if spec:
            await cancel_install(spec)
