"""Run prompts through image models and save results to the library."""

from __future__ import annotations

import asyncio
import io
import random
import re
import time
import uuid
from pathlib import Path

from PIL import Image

from logo_gen import library, paths
from logo_gen.clients import openrouter
from logo_gen.config import get_settings
from logo_gen.jobs import Job

LOCAL_PREFIX = "local/"
CLOUD_CONCURRENCY = 4


def is_local(model: str) -> bool:
    return model.startswith(LOCAL_PREFIX)


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[:40]


def _save(data: bytes, media_type: str, model: str, seed: int | None) -> tuple[Path, str]:
    paths.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(model)}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    if media_type == "image/svg+xml":
        p = paths.GENERATED_DIR / f"{stem}.svg"
        p.write_bytes(data)
        return p, media_type
    p = paths.GENERATED_DIR / f"{stem}.png"
    with Image.open(io.BytesIO(data)) as im:
        im.save(p, format="PNG")
    return p, "image/png"


def _friendly(e: BaseException) -> str:
    msg = getattr(e, "message", None) or str(e) or e.__class__.__name__
    if "CUDA out of memory" in msg or "OutOfMemoryError" in e.__class__.__name__:
        return "The GPU ran out of memory. Close other GPU apps or pick a smaller model."
    return msg


async def run(
    job: Job,
    prompts: list[dict],
    models: list[str] | None = None,
    images_per_prompt: int | None = None,
    references: list[Path] | None = None,
    chat_id: str | None = None,
    parent_id: str | None = None,
) -> None:
    s = get_settings()
    models = [m for m in (models or s.image_models) if m]
    if not models:
        raise ValueError("No image models selected. Pick at least one in Settings or the model picker.")
    per_prompt = max(1, min(int(images_per_prompt or s.images_per_prompt), 16))
    refs = [(p.read_bytes(), "image/png" if p.suffix.lower() == ".png" else "image/jpeg") for p in (references or []) if p.exists()]

    # One task per (model, prompt). Cloud tasks ask for n images at once when
    # the model allows it; local tasks batch up to max_batch.
    tasks: list[tuple[str, dict]] = [(m, p) for m in models for p in prompts]
    job.progress(done=0, total=len(tasks) * per_prompt, message=f"Generating {len(tasks) * per_prompt} images")
    cloud_sem = asyncio.Semaphore(CLOUD_CONCURRENCY)
    local_sem = asyncio.Semaphore(1)

    def record(img_path: Path, media_type: str, model: str, prompt: dict, seed: int | None, cost: float, took: float) -> None:
        img = library.add_image(
            img_path, "generated",
            job_id=job.id, chat_id=chat_id, parent_id=parent_id,
            prompt=prompt.get("prompt", ""), concept=prompt.get("concept", ""), style=prompt.get("style", ""),
            model=model, seed=seed, media_type=media_type, cost=cost, gen_time=round(took, 2),
        )
        job.image_ids.append(img["id"])

    async def cloud(model: str, prompt: dict) -> None:
        remaining = per_prompt
        info = await openrouter.model_info(model) or {}
        max_n = max(1, int(info.get("max_n") or 1))
        async with cloud_sem:
            while remaining > 0 and not job.cancelled:
                n = min(remaining, max_n)
                seed = random.randint(1, 2**31 - 1) if info.get("supports_seed") else None
                t0 = time.time()
                try:
                    results, cost = await openrouter.generate_images(
                        prompt["prompt"], model, n=n, seed=seed,
                        aspect_ratio=s.aspect_ratio, transparent=s.transparent_background,
                        references=refs or None,
                    )
                except Exception as e:  # noqa: BLE001
                    job.errors.append(f"{model}: {_friendly(e)}")
                    job.done += remaining
                    return
                job.cost += cost
                took = time.time() - t0
                per_image_cost = cost / max(len(results), 1)
                for r in results[:remaining]:
                    p, mt = await asyncio.to_thread(_save, r.data, r.media_type, model, seed)
                    record(p, mt, model, prompt, seed, per_image_cost, took)
                got = min(len(results), remaining) or n
                remaining -= got
                job.done += got
                job.message = f"{job.done}/{job.total} done"

    async def local(model: str, prompt: dict) -> None:
        from logo_gen.local_models import diffusion
        from logo_gen.local_models.registry import MODELS

        model_id = model.removeprefix(LOCAL_PREFIX)
        spec = MODELS.get(model_id)
        batch = spec.max_batch if spec else 1
        seeds = [random.randint(1, 2**31 - 1) for _ in range(per_prompt)]
        async with local_sem:
            for i in range(0, len(seeds), batch):
                if job.cancelled:
                    return
                chunk = seeds[i : i + batch]
                t0 = time.time()

                def on_step(step: int, total: int) -> None:
                    job.message = f"{spec.name if spec else model_id}: step {step}/{total}"

                def on_status(msg: str) -> None:
                    job.message = msg

                try:
                    images = await asyncio.to_thread(
                        diffusion.generate, model_id, prompt["prompt"], chunk, on_step, on_status, lambda: job.cancelled,
                    )
                except Exception as e:  # noqa: BLE001
                    job.errors.append(f"{model}: {_friendly(e)}")
                    job.done += len(seeds) - i
                    return
                took = (time.time() - t0) / max(len(images), 1)
                for im, seed in zip(images, chunk):
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    p, mt = await asyncio.to_thread(_save, buf.getvalue(), "image/png", model, seed)
                    record(p, mt, model, prompt, seed, 0.0, took)
                job.done += len(chunk)

    await asyncio.gather(*[(local if is_local(m) else cloud)(m, p) for m, p in tasks])
    n = len(job.image_ids)
    job.result.update({"images": n, "errors": len(job.errors)})
    if n == 0 and job.errors and not job.cancelled:
        job.status = "error"
        job.message = job.errors[0]
    else:
        job.message = f"{n} image{'s' if n != 1 else ''}" + (f", {len(job.errors)} failed" if job.errors else "")
