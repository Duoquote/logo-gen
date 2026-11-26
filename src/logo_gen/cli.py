"""Command line entry: `logo-gen [--host H] [--port P] [--reload]`."""

from __future__ import annotations

import argparse
import os
import socket
import sys


def _free_port(host: str, start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise SystemExit(f"No free port between {start} and {start + 49}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="logo-gen", description="Run the logo-gen server")
    parser.add_argument("--host", default=os.environ.get("LOGO_GEN_HOST", "127.0.0.1"),
                        help="Interface to bind. Use 0.0.0.0 to reach it from other machines.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("LOGO_GEN_PORT", "7860")))
    parser.add_argument("--reload", action="store_true", help="Restart on code changes (development)")
    parser.add_argument("--quiet-url", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    # Windows consoles default to a legacy code page; keep output and file IO in UTF-8.
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONUTF8", "1")
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

    import uvicorn

    port = _free_port(args.host, args.port)
    shown = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    if not args.quiet_url:
        print(f"\n  logo-gen is running at http://{shown}:{port}\n", flush=True)
    uvicorn.run(
        "logo_gen.server:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
