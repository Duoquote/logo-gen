"""Old prompt pipeline against the new one, on real models, with contact sheets.

    uv run python scripts/eval/run_eval.py --models local/zimage-turbo recraft/recraft-v4.1-flash
    uv run python scripts/eval/run_eval.py --sheets-only

- old: the pre-brief designer prompt (scripts/eval/legacy_prompts.py). One finished
  prompt per direction, the same text for every model, the old local suffix, no
  per-model parameters.
- new: the designer writes briefs (prompt_engine.directions), and each model gets
  its own rendered prompt and parameters (prompting/render.py).

Directions are cached in research/eval/cache/, so a rerun (for example after
changing a profile) re-renders the same briefs instead of asking the chat model
again. Spend goes through the app's own tracking (the costs table), so
`--spend` prints exactly what the app recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import legacy_prompts  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from logo_gen import generator, library, paths, prompt_engine  # noqa: E402
from logo_gen.clients import openrouter  # noqa: E402
from logo_gen.config import get_settings  # noqa: E402
from logo_gen.jobs import Job  # noqa: E402

EVAL = ROOT / "research" / "eval"
CACHE = EVAL / "cache"

BRIEFS = {
    "geo-saas": (
        "Terrastack, a geospatial SaaS that turns satellite and drone imagery into maps for city planners. "
        "Trustworthy, precise, modern. A symbol mark for the app and the website."
    ),
    "coffee-wordmark": (
        "A wordmark for Ember & Oak, a small neighborhood coffee roaster. Warm, a bit retro, hand-crafted. "
        "The name has to be in the logo."
    ),
    "fintech-icon": (
        "App icon for Ledgerly, a fintech app that sets aside tax money for freelancers automatically. "
        "Calm, confident, friendly. It has to work as an iOS and Android home screen icon."
    ),
}


def _load(name: str) -> dict:
    p = CACHE / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(name: str, data: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / name).write_text(json.dumps(data, indent=2), encoding="utf-8")


async def old_directions(text: str, n: int) -> list[dict]:
    s = get_settings()
    messages = [
        {"role": "system", "content": legacy_prompts.DIRECTIONS_SYSTEM},
        {"role": "user", "content": f"Brief: {text}\n\nWrite {n} directions. "
         'Return only JSON in this form: {"prompts": [{"concept": "...", "style": "...", "prompt": "..."}]}'},
    ]
    reply, usage = await openrouter.chat(messages, model=s.chat_model, temperature=0.9,
                                         reasoning_effort=s.reasoning_effort, json_schema=legacy_prompts.PROMPTS_SCHEMA)
    library.add_cost("prompts", **prompt_engine.cost_record(usage, s.chat_model))
    return json.loads(reply)["prompts"][:n]


async def new_directions(text: str, n: int) -> list[dict]:
    prompts, cost = await prompt_engine.directions(text, n)
    library.add_cost("prompts", **cost)
    return prompts


async def directions(brief: str, pipeline: str, n: int, brief_set: str | None = None) -> list[dict]:
    """Cached per brief and brief set. A new pipeline name gets fresh briefs unless --brief-set reuses one."""
    cache = _load("directions.json")
    key = f"{brief}/{brief_set or pipeline}"
    if key not in cache:
        fn = old_directions if pipeline == "old" else new_directions
        cache[key] = await fn(BRIEFS[brief], n)
        _save("directions.json", cache)
    return cache[key]


def for_model(pipeline: str, ds: list[dict], model: str) -> list[dict]:
    if pipeline != "old":
        return ds
    suffix = legacy_prompts.LOCAL_SUFFIX if generator.is_local(model) else ""
    return [{"concept": d["concept"], "style": d["style"], "prompt": d["prompt"] + suffix, "verbatim": True, "params": {}}
            for d in ds]


async def run(models: list[str], briefs: list[str], pipelines: list[str], n: int, brief_set: str | None = None) -> None:
    library.conn()
    results = _load("results.json")
    for brief in briefs:
        for pipeline in pipelines:
            ds = await directions(brief, pipeline, n, brief_set if pipeline != "old" else None)
            for model in models:
                job = Job(id=uuid.uuid4().hex[:12], kind="eval", title=f"eval {brief} {pipeline} {model}")
                await generator.run(job, for_model(pipeline, ds, model), [model], 1)
                imgs = [library.get_image(i) for i in job.image_ids]
                by_concept = {img["concept"]: img for img in imgs if img}
                rendered = {r["direction"]: r for r in job.result.get("rendered", [])}
                rows = []
                for i, d in enumerate(ds):
                    img = by_concept.get(d["concept"])
                    rows.append({
                        "concept": d["concept"],
                        "path": img["path"] if img else None,
                        "prompt": (rendered.get(i) or {}).get("prompt", ""),
                        "params": (rendered.get(i) or {}).get("params", {}),
                        "error": None if img else "; ".join(job.errors)[:300],
                    })
                results.setdefault(brief, {}).setdefault(pipeline, {})[model] = rows
                _save("results.json", results)
                print(f"{brief:16} {pipeline:5} {model:40} {len(job.image_ids)} images, "
                      f"${job.cost + job.byok:.4f}{'  ERR ' + job.errors[0][:120] if job.errors else ''}", flush=True)


# ── Contact sheets ───────────────────────────────────────────────────

CELL = 220
SMALL = (64, 32)
PAD = 12


def _font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _cell(rel: str | None) -> Image.Image:
    """The image at 220 px, with 64 px and 32 px copies under it to judge small sizes."""
    h = CELL + max(SMALL) + 8
    out = Image.new("RGB", (CELL, h), (236, 236, 236))
    if not rel:
        ImageDraw.Draw(out).text((10, CELL // 2), "failed", fill=(180, 0, 0), font=_font(14))
        return out
    path = paths.resolve_output(rel)
    if path.suffix == ".svg":
        ImageDraw.Draw(out).text((10, CELL // 2), "SVG", fill=(0, 0, 0), font=_font(14))
        return out
    with Image.open(path) as im:
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
        out.paste(im.resize((CELL, CELL), Image.LANCZOS), (0, 0))
        x = 4
        for s in SMALL:
            out.paste(im.resize((s, s), Image.LANCZOS), (x, CELL + 4))
            x += s + 8
    return out


LABELS = {
    "old": ("OLD: one prompt for every model", (120, 0, 0)),
    "new": ("NEW, round 1: brief rendered per model", (0, 90, 0)),
    "new2": ("NEW, round 2: same briefs, renderer fixed", (0, 60, 140)),
    "new3": ("NEW, round 3: meaning clauses dropped (Z-Image only)", (90, 0, 120)),
}


def sheet(brief: str, keys: list[str]) -> Path | None:
    """One block of columns per pipeline run, one row per model, with 64 and 32 px copies under each image."""
    results = _load("results.json").get(brief) or {}
    runs = [(k, results.get(k) or {}) for k in keys if results.get(k)]
    models = list(dict.fromkeys(m for _, r in runs for m in r))
    if not models:
        return None
    cols = max(len(v) for _, r in runs for v in r.values())
    label_w, head_h = 170, 54
    cell_h = CELL + max(SMALL) + 8
    block_w = cols * (CELL + PAD)
    W = label_w + len(runs) * (block_w + 2 * PAD)
    H = head_h + 30 + len(models) * (cell_h + 26) + PAD
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = _font(13)
    d.text((PAD, 10), f"{brief}: {BRIEFS[brief][:150]}", fill="black", font=_font(20))
    xs = [label_w + i * (block_w + 2 * PAD) for i in range(len(runs))]
    for x0, (k, _) in zip(xs, runs):
        label, color = LABELS.get(k, (k, (0, 0, 0)))
        d.text((x0, head_h), label, fill=color, font=_font(15))
        if x0 != xs[0]:
            d.line([(x0 - PAD, head_h), (x0 - PAD, H)], fill=(160, 160, 160), width=2)
    y = head_h + 30
    for m in models:
        d.text((PAD, y + CELL // 2 - 10), m.replace("/", "/\n", 1), fill="black", font=f)
        for x0, (_, r) in zip(xs, runs):
            for i, row in enumerate(r.get(m) or []):
                x = x0 + i * (CELL + PAD)
                img.paste(_cell(row.get("path")), (x, y))
                d.text((x, y + cell_h + 2), row["concept"][:30], fill=(90, 90, 90), font=f)
            if not r.get(m):
                d.text((x0, y + CELL // 2), "not run (budget)", fill=(150, 150, 150), font=f)
        y += cell_h + 26
    EVAL.mkdir(parents=True, exist_ok=True)
    out = EVAL / f"{brief}.png"
    img.convert("RGB").save(out, optimize=True)
    return out


def write_prompts_md() -> Path:
    """Every prompt that was sent, next to its image, for reading alongside the sheets."""
    results = _load("results.json")
    lines = ["# Prompts sent in the eval", "", "Generated by scripts/eval/run_eval.py. Images are in the contact sheets.", ""]
    for brief, pipes in results.items():
        lines += [f"## {brief}", "", f"Brief: {BRIEFS[brief]}", ""]
        for pipe, models in pipes.items():
            for model, rows in models.items():
                lines += [f"### {pipe} / {model}", ""]
                for r in rows:
                    params = f" `{json.dumps(r['params'])}`" if r.get("params") else ""
                    lines += [f"- **{r['concept']}**{params}", "", "  " + r["prompt"].replace("\n", "\n  "), ""]
    out = EVAL / "prompts.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--briefs", nargs="*", default=list(BRIEFS))
    ap.add_argument("--pipelines", nargs="*", default=["old", "new"])
    ap.add_argument("-n", type=int, default=3, help="directions per brief")
    ap.add_argument("--sheets-only", action="store_true")
    ap.add_argument("--compare", nargs="*", default=["old", "new", "new2", "new3"], help="pipeline runs to put side by side")
    ap.add_argument("--spend", action="store_true")
    ap.add_argument("--brief-set", help="reuse the cached briefs of this pipeline name (re-render only)")
    a = ap.parse_args()
    if a.models and not a.sheets_only:
        asyncio.run(run(a.models, a.briefs, a.pipelines, a.n, a.brief_set))
    for b in a.briefs:
        p = sheet(b, a.compare)
        if p:
            print("sheet:", p)
    print("prompts:", write_prompts_md())
    if a.spend:
        print(json.dumps(library.spend(), indent=2))


if __name__ == "__main__":
    main()
