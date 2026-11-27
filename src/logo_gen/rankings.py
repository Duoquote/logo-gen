"""Quality, value and popularity rankings for image models.

Quality comes from LMArena's text-to-image leaderboard, which LMArena
publishes as an open dataset on Hugging Face (lmarena-ai/leaderboard-dataset,
CC BY 4.0). We read it through Hugging Face's public datasets-server JSON API,
no key needed. We use the "commercial_design" category, the closest match to
logo work, and fall back to "overall" for models without it.

Popularity comes from OpenRouter's documented Data API (rankings-daily,
modality=image_output, CC BY 4.0). It measures usage on OpenRouter, not
quality, and needs the OpenRouter key the app already has.

Both are cached for a day in data/rankings.json. A dated snapshot of the
leaderboard ships in the repo (rankings_snapshot.json) so it works offline;
refresh it with `uv run python scripts/update_rankings.py`.

Leaderboard names don't match model ids, so ALIASES maps them explicitly.
A model missing from ALIASES is shown as unranked rather than guessed.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from pathlib import Path

import httpx

from logo_gen import paths

SNAPSHOT_FILE = Path(__file__).with_name("rankings_snapshot.json")
CACHE_TTL = 24 * 3600

LMARENA = {
    "name": "LMArena text-to-image leaderboard",
    "url": "https://lmarena.ai/leaderboard/text-to-image",
    "dataset": "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset",
    "license": "CC BY 4.0",
}
LMARENA_ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset=lmarena-ai/leaderboard-dataset"
    "&config=text_to_image&split=latest&offset={offset}&length=100"
)
CATEGORIES = ("commercial_design", "overall")
OPENROUTER_RANKINGS = "https://openrouter.ai/api/v1/datasets/rankings-daily"

# Model id (OpenRouter, or local/<id>) -> LMArena model_name.
# Only exact matches of the same model. Variants the arena doesn't test
# (vector modes, "flash" or "pro" tiers of a different size, chat models
# that wrap an image model) stay unranked on purpose.
ALIASES: dict[str, str] = {
    "openai/gpt-image-2.6-sunburst": "GPT-Image-2.6 Sunburst (Medium)",
    "openai/gpt-image-2.6-flare": "GPT-Image-2.6 Flare (Medium)",
    "openai/gpt-image-2.5-sunburst": "GPT-Image-2.5 Sunburst",
    "openai/gpt-image-2.5-flare": "GPT-Image-2.5 Flare",
    "openai/gpt-image-2": "GPT-Image-2 (Medium)",
    "openai/gpt-image-1": "GPT-Image-1",
    "openai/gpt-image-1-mini": "GPT-Image-1 mini",
    "microsoft/mai-image-2.7": "MAI-Image-2.7",
    "microsoft/mai-image-2.6": "MAI-Image-2.6",
    "microsoft/mai-image-2.5": "MAI-Image-2.5",
    "x-ai/grok-imagine-image-2.0": "Grok Imagine Image 2.0",
    "google/gemini-nano-banana-2.1": "Gemini Nano Banana 2.1",
    "google/gemini-3.1-flash-lite-image": "Gemini 3.1 Flash-Lite Image (nano-banana-2-lite)",
    "google/gemini-3-pro-image": "Gemini 3 Pro Image (nano-banana-pro, 2K)",
    "google/gemini-3-pro-image-preview": "Gemini 3 Pro Image Preview (nano-banana-pro)",
    "google/gemini-2.5-flash-image": "Gemini 2.5 Flash Image Preview (nano-banana)",
    "meta/muse-image": "Muse Image",
    "qwen/qwen-image-3-pro": "Qwen-Image-3.0-Pro",
    "bytedance-seed/seedream-5-0-pro": "Seedream 5.0 Pro",
    "bytedance-seed/seedream-5-0-lite": "Seedream 5.0 Lite",
    "bytedance-seed/seedream-4.5": "Seedream 4.5",
    "black-forest-labs/flux.2-max": "FLUX.2 [max]",
    "black-forest-labs/flux.2-flex": "FLUX.2 [flex]",
    "black-forest-labs/flux.2-pro": "FLUX.2 [pro]",
    "black-forest-labs/flux.2-klein-4b": "FLUX.2 [klein] 4B",
    "krea/krea-2-large": "Krea 2 Large",
    "krea/krea-2-medium": "Krea 2 Medium",
    "recraft/recraft-v4.1-utility-pro": "Recraft V4.1 Utility Pro",
    "recraft/recraft-v4.1-pro": "Recraft V4.1 Pro",
    "recraft/recraft-v4.1-flash": "Recraft V4.1 Flash",
    "recraft/recraft-v4.1-flash-v2": "Recraft V4.1 Flash V2",
    "recraft/recraft-v4.1-flash-v2:free": "Recraft V4.1 Flash V2",
    "recraft/recraft-v4": "Recraft V4",
    "recraft/recraft-v3": "Recraft V3",
    "local/zimage-turbo": "Z-Image-Turbo",
    "local/flux2-klein-4b": "FLUX.2 [klein] 4B",
}

# Best value: Elo a model must gain to be worth twice the price. On an Elo
# scale, 50 points is about a 57% win rate head to head.
ELO_PER_DOUBLING = 50.0
PRICE_FLOOR = 0.001  # free models count as $0.001 so the log stays finite
PRICE_REF = 0.01


def value_score(rating: float, price: float) -> float:
    """Quality for the money: rating minus ELO_PER_DOUBLING per doubling of price above $0.01."""
    return rating - ELO_PER_DOUBLING * math.log2(max(price, PRICE_FLOOR) / PRICE_REF)


def pareto_front(items: dict[str, tuple[float, float]]) -> set[str]:
    """Ids not beaten on both quality and price: nothing else is at least as good and at most as cheap, and better on one."""
    front = set()
    for a, (ra, pa) in items.items():
        dominated = any(
            rb >= ra and pb <= pa and (rb > ra or pb < pa) for b, (rb, pb) in items.items() if b != a
        )
        if not dominated:
            front.add(a)
    return front


# ── Leaderboard data ─────────────────────────────────────────────────


def _slim(rows: list[dict]) -> dict:
    """Keep the categories we use, keyed by model name."""
    out: dict[str, dict] = {}
    date = ""
    for r in rows:
        if r.get("category") not in CATEGORIES:
            continue
        date = max(date, r.get("leaderboard_publish_date") or "")
        e = out.setdefault(r["model_name"], {"organization": r.get("organization") or "", "license": r.get("license") or ""})
        e[r["category"]] = {
            "rating": round(float(r["rating"]), 1),
            "ci": [round(float(r["rating_lower"]), 1), round(float(r["rating_upper"]), 1)],
            "votes": int(r.get("vote_count") or 0),
            "rank": int(r["rank"]),
        }
    return {"source": LMARENA | {"published": date}, "models": out}


async def fetch_lmarena(client: httpx.AsyncClient | None = None) -> dict:
    own = client is None
    client = client or httpx.AsyncClient(timeout=30, headers={"User-Agent": "logo-gen"})
    try:
        rows: list[dict] = []
        for offset in range(0, 5000, 100):
            r = await client.get(LMARENA_ROWS.format(offset=offset))
            r.raise_for_status()
            page = [x["row"] for x in r.json().get("rows", [])]
            rows += page
            if len(page) < 100:
                break
        data = _slim(rows)
        if not data["models"]:
            raise ValueError("The leaderboard dataset came back empty.")
        return data | {"fetched_at": time.time()}
    finally:
        if own:
            await client.aclose()


async def fetch_popularity(key: str) -> dict:
    """Image-output token usage on OpenRouter over the last 30 days, per model."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            OPENROUTER_RANKINGS,
            params={"modality": "image_output", "period": "week"},
            headers={"Authorization": f"Bearer {key}"},
        )
        r.raise_for_status()
        body = r.json()
    totals: dict[str, float] = {}
    for row in body.get("data", []):
        slug = row.get("model_permaslug") or ""
        if slug and slug != "other":
            totals[slug] = totals.get(slug, 0.0) + float(row.get("total_tokens") or 0)
    meta = body.get("meta") or {}
    return {
        "source": {
            "name": "OpenRouter usage rankings (image output)",
            "url": "https://openrouter.ai/rankings",
            "license": "CC BY 4.0",
            "as_of": meta.get("as_of", ""),
            "citation": f"Source: OpenRouter (openrouter.ai/rankings), as of {meta.get('as_of', '')}.",
        },
        "tokens": totals,
        "fetched_at": time.time(),
    }


_DATED = re.compile(r"-\d{8}(?=$|:)")


def popularity_id(slug: str, catalog_ids: set[str]) -> str | None:
    """Map an OpenRouter permaslug (dated variant id) to a catalog id.

    "google/gemini-3.1-flash-image-20260528" -> "google/gemini-3.1-flash-image";
    "meta/muse-image-1.0-eval-20260824" -> "meta/muse-image" (longest catalog id
    followed by "-"). ":free" stays part of the id.
    """
    base = _DATED.sub("", slug)
    if base in catalog_ids:
        return base
    best = None
    for cid in catalog_ids:
        if (":" in cid) != (":" in base):
            continue
        if base.startswith(cid + "-") and (best is None or len(cid) > len(best)):
            best = cid
    return best


# ── Cache ────────────────────────────────────────────────────────────

_lock = asyncio.Lock()


def _cache_file() -> Path:
    return paths.DATA_DIR / "rankings.json"


def _load_cache() -> dict:
    try:
        return json.loads(_cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _cache_file().with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(_cache_file())


def snapshot() -> dict:
    return json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))


async def data(key: str | None, refresh: bool = False) -> tuple[dict, dict | None, list[str]]:
    """(leaderboard, popularity or None, notes). Never raises: falls back to cache, then the snapshot."""
    notes: list[str] = []
    async with _lock:
        cache = _load_cache()
        now = time.time()
        board = cache.get("lmarena")
        if refresh or not board or now - board.get("fetched_at", 0) > CACHE_TTL:
            try:
                board = await fetch_lmarena()
                cache["lmarena"] = board
                _save_cache(cache)
            except Exception as e:  # noqa: BLE001
                notes.append(f"Couldn't refresh the leaderboard ({e.__class__.__name__}), so this uses saved data.")
        if not board:
            board = snapshot() | {"bundled": True}
        pop = cache.get("popularity")
        if key and (refresh or not pop or now - pop.get("fetched_at", 0) > CACHE_TTL):
            try:
                pop = await fetch_popularity(key)
                cache["popularity"] = pop
                _save_cache(cache)
            except Exception as e:  # noqa: BLE001
                notes.append(f"Couldn't refresh OpenRouter usage rankings ({e.__class__.__name__}).")
        return board, pop, notes


# ── Putting it together ──────────────────────────────────────────────


def _rating(entry: dict) -> tuple[dict | None, str]:
    for cat in CATEGORIES:
        if cat in entry:
            return entry[cat], cat
    return None, ""


def rank_models(models: list[dict], board: dict, pop: dict | None) -> dict[str, dict]:
    """Per model id: quality, rank among these models, price, value, Pareto, popularity.

    `models` items need "id" and "price" (USD per 1024px image, 0 for free or
    local, None if unknown).
    """
    out: dict[str, dict] = {}
    for m in models:
        name = ALIASES.get(m["id"])
        entry = board["models"].get(name) if name else None
        r, cat = _rating(entry) if entry else (None, "")
        out[m["id"]] = {
            "ranked": r is not None,
            "leaderboard_name": name if r else None,
            "category": cat or None,
            "rating": r["rating"] if r else None,
            "ci": r["ci"] if r else None,
            "votes": r["votes"] if r else None,
            "arena_rank": r["rank"] if r else None,
            "price": m.get("price"),
            "value": None,
            "pareto": False,
            "rank": None,
            "popularity_rank": None,
        }
    ranked = sorted((i for i, x in out.items() if x["ranked"]), key=lambda i: -out[i]["rating"])
    for n, i in enumerate(ranked, 1):
        out[i]["rank"] = n
    priced = {i: (out[i]["rating"], out[i]["price"]) for i in ranked if out[i]["price"] is not None}
    for i, (rating, price) in priced.items():
        out[i]["value"] = round(value_score(rating, price), 1)
    for i in pareto_front(priced):
        out[i]["pareto"] = True
    if pop:
        ids = {i for i in out if not i.startswith("local/")}
        tokens: dict[str, float] = {}
        for slug, t in pop.get("tokens", {}).items():
            cid = popularity_id(slug, ids)
            if cid:
                tokens[cid] = tokens.get(cid, 0.0) + t
        for n, (i, _) in enumerate(sorted(tokens.items(), key=lambda kv: -kv[1]), 1):
            out[i]["popularity_rank"] = n
    return out
