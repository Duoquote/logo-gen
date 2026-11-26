"""Download worker, run as a subprocess so it can be stopped at any point.

Reads a JSON job from stdin. Each file is fetched with huggingface_hub's
http_get into ``<dest>/<path>.part`` and renamed when complete. A ``.part``
file left behind by a cancel or crash is resumed with an HTTP Range request.
The HF token comes from the HF_TOKEN environment variable, never argv.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_url
from huggingface_hub.file_download import http_get
from huggingface_hub.utils import build_hf_headers


def main() -> int:
    job = json.loads(sys.stdin.read())
    dest = Path(job["dest"])
    token = os.environ.get("HF_TOKEN") or None
    headers = build_hf_headers(token=token, library_name="logo-gen")
    for f in job["files"]:
        target = dest / f["path"]
        if target.exists() and target.stat().st_size == f["size"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        resume = part.stat().st_size if part.exists() else 0
        if resume > f["size"]:
            part.unlink()
            resume = 0
        url = hf_hub_url(job["repo"], f["path"], revision=job["revision"])
        with part.open("ab") as fh:
            http_get(url, fh, resume_size=resume, headers=dict(headers), expected_size=f["size"],
                     displayed_filename=f["path"])
        if part.stat().st_size != f["size"]:
            print(f"size mismatch for {f['path']}", file=sys.stderr)
            return 2
        os.replace(part, target)
        print(json.dumps({"done": f["path"]}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        print(f"{e.__class__.__name__}: {e}", file=sys.stderr)
        sys.exit(1)
