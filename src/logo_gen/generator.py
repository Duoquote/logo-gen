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
from logo_gen.prompting import preferences
from logo_gen.prompting.render import Rendered, render_direction

LOCAL_PREFIX = "local/"


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
        return "The GPU ran out of memory even one image at a time. Close other GPU apps or pick a smaller model."
    return msg


class AdaptiveLimiter:
    """Caps parallel OpenRouter image requests.

    Starts at the configured concurrency, halves it on a rate limit, and
    grows back by one after a run of successes. Shared by all jobs, since
    OpenRouter's limits are per account.
    """

    def __init__(self, limit: int) -> None:
        self.max = limit
        self.limit = limit
        self.active = 0
        self.streak = 0
        self.cond = asyncio.Condition()

    async def __aenter__(self) -> None:
        async with self.cond:
            await self.cond.wait_for(lambda: self.active < self.limit)
            self.active += 1

    async def __aexit__(self, *exc) -> None:
        async with self.cond:
            self.active -= 1
            self.cond.notify_all()

    def backoff(self) -> None:
        self.limit = max(1, self.limit // 2)
        self.streak = 0

    async def success(self) -> None:
        self.streak += 1
        if self.limit < self.max and self.streak >= self.limit:
            self.limit += 1
            self.streak = 0
            async with self.cond:
                self.cond.notify_all()


_limiter: AdaptiveLimiter | None = None


def limiter() -> AdaptiveLimiter:
    global _limiter
    want = get_settings().cloud_concurrency
    if _limiter is None or _limiter.max != want:
        _limiter = AdaptiveLimiter(want)
    return _limiter


def plan(models: list[str], prompts: list[dict], per_prompt: int) -> tuple[dict[str, tuple[list[dict], int]], list[str]]:
    """Apply the per-model caps. Returns {model: (prompts, images per prompt)} and notes on what was cut."""
    s = get_settings()
    out, notes = {}, []
    for m in models:
        cap = s.cap_for(m)
        ps, per = prompts, per_prompt
        if len(ps) * per > cap:
            per = max(1, cap // len(ps))
            if len(ps) * per > cap:
                ps = ps[:cap]
            notes.append(f"{m}: capped at {cap} images per request (Settings, Limits)")
        out[m] = (ps, per)
    return out, notes


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
    job.chat_id = chat_id
    per_prompt = max(1, int(images_per_prompt or s.images_per_prompt))
    work, notes = plan(models, prompts, per_prompt)
    job.notes.extend(notes)
    refs = [
        (p.read_bytes(), "image/png" if p.suffix.lower() == ".png" else "image/jpeg")
        for p in (references or []) if p.exists()
    ]

    total = sum(len(ps) * per for ps, per in work.values())
    job.progress(done=0, total=total, message=f"Generating {total} images")
    local_sem = asyncio.Semaphore(1)

    # Render every direction for every model up front, so the job shows
    # exactly what each model was sent.
    infos = {m: (await openrouter.model_info(m) or {}) for m in work if not is_local(m)}
    try:
        avoid = preferences.render_hints(chat_id)
    except Exception:  # noqa: BLE001  (learned hints are optional)
        avoid = []
    rendered: dict[tuple[str, int], Rendered] = {}
    for m, (ps, _) in work.items():
        for i, p in enumerate(ps):
            r = render_direction(p, m, infos.get(m), s.transparent_background, avoid)
            rendered[(m, i)] = r
            job.result.setdefault("rendered", []).append(r.to_dict() | {"concept": p.get("concept", ""), "direction": i})
            for n in r.notes:
                note = f"{m}: {n}"
                if note not in job.notes and n != "Edited by hand":
                    job.notes.append(note)

    def record(img_path: Path, media_type: str, model: str, prompt: dict, r: Rendered, seed: int | None,
               cost: float, took: float) -> None:
        brief = prompt.get("brief") if isinstance(prompt.get("brief"), dict) else None
        img = library.add_image(
            img_path, "generated",
            job_id=job.id, chat_id=chat_id, parent_id=parent_id,
            prompt=r.prompt, concept=prompt.get("concept", ""), style=prompt.get("style", ""),
            model=model, seed=seed, media_type=media_type, cost=cost, gen_time=round(took, 2),
            brief=brief, negative_prompt=r.negative_prompt, params=r.params, profile=r.profile,
        )
        job.image_ids.append(img["id"])

    async def cloud(model: str, idx: int, prompt: dict, per: int) -> None:
        remaining = per
        info = infos.get(model) or {}
        r = rendered[(model, idx)]
        max_n = max(1, int(info.get("max_n") or 1))
        lim = limiter()
        attempt = 0
        while remaining > 0 and not job.cancelled:
            n = min(remaining, max_n)
            seed = random.randint(1, 2**31 - 1) if info.get("supports_seed") else None
            t0 = time.time()
            try:
                async with lim:
                    results, usage = await openrouter.generate_images(
                        r.prompt, model, n=n, seed=seed,
                        aspect_ratio=s.aspect_ratio, references=refs or None, extra=r.params,
                    )
            except openrouter.OpenRouterError as e:
                delay = openrouter.retry_delay(e, attempt)
                if delay is not None and attempt < 4:
                    attempt += 1
                    lim.backoff()
                    job.message = f"{model}: {e.kind.replace('_', ' ')}, retrying in {int(delay)}s"
                    await asyncio.sleep(delay)
                    continue
                job.errors.append(f"{model}: {_friendly(e)}")
                job.done += remaining
                return
            except Exception as e:  # noqa: BLE001
                job.errors.append(f"{model}: {_friendly(e)}")
                job.done += remaining
                return
            attempt = 0
            await lim.success()
            credits, byok = openrouter.usage_cost(usage)
            got = min(len(results), remaining)
            job.add_cost("image", {"credits": credits, "byok": byok, "model": model}, images=got)
            took = time.time() - t0
            per_image_cost = (credits + byok) / max(len(results), 1)
            for res in results[:remaining]:
                p, mt = await asyncio.to_thread(_save, res.data, res.media_type, model, seed)
                record(p, mt, model, prompt, r, seed, per_image_cost, took)
            got = got or n
            remaining -= got
            job.done += got
            job.message = f"{job.done}/{job.total} done"

    async def local(model: str, ps: list[dict], per: int) -> None:
        """All images for one local model, batched across directions by the saved profile."""
        from logo_gen.local_models import diffusion
        from logo_gen.local_models.registry import MODELS

        model_id = model.removeprefix(LOCAL_PREFIX)
        spec = MODELS.get(model_id)
        batch = max(1, int(s.local_batch.get(model_id) or (spec.max_batch if spec else 1)))
        items = [(p, rendered[(model, i)], random.randint(1, 2**31 - 1)) for i, p in enumerate(ps) for _ in range(per)]
        options = rendered[(model, 0)].params if ps else {}
        name = spec.name if spec else model_id
        async with local_sem:
            for i in range(0, len(items), batch):
                if job.cancelled:
                    return
                chunk = items[i : i + batch]
                t0 = time.time()

                def on_step(step: int, total: int, size: int = len(chunk)) -> None:
                    job.message = f"{name}: batch of {size}, step {step}/{total}"

                def on_status(msg: str) -> None:
                    job.message = msg
                    if "out of memory" in msg.lower() and msg not in job.notes:
                        job.notes.append(msg)

                try:
                    images = await asyncio.to_thread(
                        diffusion.generate, model_id, [r.prompt for _, r, _ in chunk], [sd for _, _, sd in chunk],
                        on_step, on_status, lambda: job.cancelled,
                        [r.negative_prompt for _, r, _ in chunk], options,
                    )
                except Exception as e:  # noqa: BLE001
                    job.errors.append(f"{model}: {_friendly(e)}")
                    job.done += len(items) - i
                    return
                took = (time.time() - t0) / max(len(images), 1)
                for im, (prompt, r, seed) in zip(images, chunk):
                    buf = io.BytesIO()
                    im.save(buf, format="PNG")
                    p, mt = await asyncio.to_thread(_save, buf.getvalue(), "image/png", model, seed)
                    record(p, mt, model, prompt, r, seed, 0.0, took)
                job.local_images += len(images)
                library.add_cost("image", model=model, job_id=job.id, chat_id=chat_id, images=len(images), local=True)
                job.done += len(chunk)

    coros = []
    for m, (ps, per) in work.items():
        if is_local(m):
            coros.append(local(m, ps, per))
        else:
            coros.extend(cloud(m, i, p, per) for i, p in enumerate(ps))
    await asyncio.gather(*coros)
    n = len(job.image_ids)
    job.result.update({"images": n, "errors": len(job.errors)})
    if n == 0 and job.errors and not job.cancelled:
        job.status = "error"
        job.message = job.errors[0]
    else:
        job.message = f"{n} image{'s' if n != 1 else ''}" + (f", {len(job.errors)} failed" if job.errors else "")
