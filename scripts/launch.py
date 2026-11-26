"""One-command launcher, run by ./logo-gen through `uv run --no-project`.

Standard library only. Steps, each skipped when nothing changed since the
last run (hashes are kept in .tools/state.json):

1. uv sync the project .venv (adds local model support if it was installed)
2. install frontend packages with bun (downloaded into .tools/bun if needed)
3. build the frontend
4. start the server, which prints the URL
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / ".tools"
STATE = TOOLS / "state.json"
FRONTEND = ROOT / "frontend"
IS_WIN = sys.platform == "win32"

sys.path.insert(0, str(ROOT / "src"))
from logo_gen import system  # noqa: E402  (stdlib-only module)


def say(msg: str) -> None:
    print(f"  {msg}", flush=True)


def run(cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> None:
    code = subprocess.call(cmd, cwd=str(cwd), env=env)
    if code != 0:
        sys.exit(f"\nCommand failed ({code}): {' '.join(cmd)}")


def digest(*items: Path) -> str:
    h = hashlib.sha256()
    for item in items:
        files = sorted(p for p in item.rglob("*") if p.is_file()) if item.is_dir() else [item]
        for f in files:
            if not f.exists():
                continue
            h.update(str(f.relative_to(ROOT)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    TOOLS.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def venv_python() -> Path:
    return ROOT / ".venv" / ("Scripts/python.exe" if IS_WIN else "bin/python")


# ── Python ──────────────────────────────────────────────────────────


def python_env(uv: str, state: dict, with_local: bool) -> None:
    marker = TOOLS / "local-stack"
    if with_local and not marker.exists():
        TOOLS.mkdir(exist_ok=True)
        marker.write_text(system.torch_variant(), encoding="utf-8")
    extras: list[str] = []
    if marker.exists():
        extras = ["local", marker.read_text(encoding="utf-8").strip() or system.torch_variant()]
    key = digest(ROOT / "pyproject.toml", ROOT / "uv.lock") + ":" + ",".join(extras)
    if state.get("python") == key and venv_python().exists():
        say("Python packages: up to date")
        return
    label = f" with local model support ({extras[1]})" if extras else " (cloud only; local model support installs on demand)"
    say(f"Python packages: syncing{label}")
    cmd = [uv, "sync"]
    for e in extras:
        cmd += ["--extra", e]
    run(cmd)
    state["python"] = key


# ── Frontend ────────────────────────────────────────────────────────


def _bun_target() -> str:
    machine = platform.machine().lower()
    arch = "aarch64" if machine in ("arm64", "aarch64") else "x64"
    if IS_WIN:
        return f"windows-{arch}"
    if sys.platform == "darwin":
        return f"darwin-{arch}"
    return f"linux-{arch}"


def find_bun() -> str:
    found = shutil.which("bun")
    if found:
        return found
    local = TOOLS / "bun" / ("bun.exe" if IS_WIN else "bun")
    if local.exists():
        return str(local)
    target = _bun_target()
    url = f"https://github.com/oven-sh/bun/releases/latest/download/bun-{target}.zip"
    say(f"bun not found, downloading it into .tools/bun ({target})")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = next(n for n in zf.namelist() if n.endswith("/bun.exe" if IS_WIN else "/bun"))
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(zf.read(name))
    if not IS_WIN:
        local.chmod(0o755)
    return str(local)


def frontend(state: dict, force_build: bool) -> None:
    bun = None
    deps_key = digest(FRONTEND / "package.json", FRONTEND / "bun.lock")
    if state.get("frontend_deps") != deps_key or not (FRONTEND / "node_modules").exists():
        bun = bun or find_bun()
        say("Frontend packages: installing")
        run([bun, "install"], cwd=FRONTEND)
        state["frontend_deps"] = deps_key
    else:
        say("Frontend packages: up to date")

    sources = [FRONTEND / n for n in ("src", "public", "index.html", "vite.config.ts", "tsconfig.json",
                                      "tsconfig.app.json", "tsconfig.node.json", "package.json")]
    build_key = digest(*[s for s in sources if s.exists()])
    if force_build or state.get("frontend_build") != build_key or not (FRONTEND / "dist" / "index.html").exists():
        bun = bun or find_bun()
        say("Frontend: building")
        run([bun, "run", "build"], cwd=FRONTEND)
        state["frontend_build"] = build_key
    else:
        say("Frontend: up to date")


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(prog="logo-gen")
    p.add_argument("--uv", required=True, help=argparse.SUPPRESS)
    p.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to open it to your network")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--with-local", action="store_true", help="Install local model support (PyTorch) now")
    p.add_argument("--dev", action="store_true", help="Run the Vite dev server with hot reload")
    p.add_argument("--rebuild", action="store_true", help="Rebuild the frontend even if nothing changed")
    args = p.parse_args()

    t0 = time.time()
    state = load_state()
    python_env(args.uv, state, args.with_local)
    if not args.dev:
        frontend(state, args.rebuild)
    save_state(state)
    say(f"Ready in {time.time() - t0:.1f}s")

    env = os.environ.copy()
    env["LOGO_GEN_UV"] = args.uv
    env.setdefault("PYTHONUTF8", "1")
    server = [str(venv_python()), "-m", "logo_gen", "--host", args.host, "--port", str(args.port)]

    if not args.dev:
        try:
            sys.exit(subprocess.call(server, cwd=str(ROOT), env=env))
        except KeyboardInterrupt:
            sys.exit(0)

    # Dev: backend with reload on --port, Vite on 5173 proxying /api to it.
    bun = find_bun()
    if not (FRONTEND / "node_modules").exists():
        run([bun, "install"], cwd=FRONTEND)
    env["LOGO_GEN_API_PORT"] = str(args.port)
    back = subprocess.Popen([*server, "--reload", "--quiet-url"], cwd=str(ROOT), env=env)
    try:
        sys.exit(subprocess.call([bun, "run", "dev", "--", "--port", "5173", "--strictPort"], cwd=str(FRONTEND), env=env))
    except KeyboardInterrupt:
        pass
    finally:
        back.terminate()


if __name__ == "__main__":
    main()
