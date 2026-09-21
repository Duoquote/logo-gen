"""Logo generation orchestrator - runs prompts through multiple models/seeds."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from logo_gen.clients import openrouter
from logo_gen.clients.local_diffusion import is_local_model, parse_model_spec
from logo_gen.config import settings


@dataclass
class GeneratedLogo:
    path: Path
    prompt: str
    concept: str
    model: str
    seed: int | None
    generation_time: float


@dataclass
class GenerationResult:
    logos: list[GeneratedLogo] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_time: float = 0.0


# ── Single-image (cloud) ──────────────────────────────────────

async def _generate_cloud_one(
    prompt: str,
    concept: str,
    model: str,
    seed: int | None,
    output_dir: Path,
    reference_images: list[Path] | None,
) -> list[GeneratedLogo]:
    start = time.time()
    try:
        paths = await openrouter.generate_image(
            prompt=prompt,
            model=model,
            seed=seed,
            save_dir=output_dir,
            reference_images=reference_images,
        )
        elapsed = time.time() - start
        return [
            GeneratedLogo(
                path=p,
                prompt=prompt,
                concept=concept,
                model=model,
                seed=seed,
                generation_time=elapsed,
            )
            for p in paths
        ]
    except Exception:
        return [
            GeneratedLogo(
                path=Path("error"),
                prompt=prompt,
                concept=concept,
                model=model,
                seed=seed,
                generation_time=time.time() - start,
            )
        ]


# ── Batched (local) ──────────────────────────────────────────
#
# A local batched task generates len(seeds) images in a single
# pipeline call via num_images_per_prompt. Far faster than serial calls
# because the GPU schedules the work together.

async def _generate_local_batch(
    prompt: str,
    concept: str,
    model: str,
    seeds: list[int],
    output_dir: Path,
) -> list[GeneratedLogo]:
    from logo_gen.clients import local_diffusion

    start = time.time()
    try:
        paths = await local_diffusion.generate_image(
            prompt=prompt,
            model=model,
            seeds=seeds,
            save_dir=output_dir,
        )
        elapsed = time.time() - start
        per_image = elapsed / max(len(paths), 1)
        return [
            GeneratedLogo(
                path=p,
                prompt=prompt,
                concept=concept,
                model=model,
                seed=s,
                generation_time=per_image,
            )
            for p, s in zip(paths, seeds)
        ]
    except Exception:
        elapsed = time.time() - start
        return [
            GeneratedLogo(
                path=Path("error"),
                prompt=prompt,
                concept=concept,
                model=model,
                seed=s,
                generation_time=elapsed,
            )
            for s in seeds
        ]


def _make_seeds(n: int) -> list[int]:
    """Generate diverse seeds."""
    return [random.randint(1, 2**31) for _ in range(n)]


async def generate_logos(
    prompts: list[dict],
    models: list[str] | None = None,
    seeds_per_prompt: int | None = None,
    output_dir: Path | None = None,
    progress_callback=None,
    reference_images: list[Path] | None = None,
) -> GenerationResult:
    """Generate logos from a list of prompt dicts across models and seeds.

    Local models batch via num_images_per_prompt (a single pipeline
    call produces many images). Cloud models stay single-image but run
    concurrently across a 3-wide semaphore. The two pools have separate
    semaphores so a local pipeline can be churning while cloud calls
    run in parallel.

    Tasks are scheduled model-first so the local pipeline (which
    unloads the previous one to free VRAM) doesn't thrash between
    models.
    """
    models = models or settings.image_models
    seeds_per_prompt = seeds_per_prompt or settings.images_per_model
    output_dir = Path(output_dir or settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    result = GenerationResult()

    # Build tasks. Each entry is either:
    #   ("local", prompt, concept, model, [seed, seed, ...])  – one batched call
    #   ("cloud", prompt, concept, model, seed)               – one single call
    local_tasks: list[tuple] = []
    cloud_tasks: list[tuple] = []

    # Iterate models first so all of model A runs before any of model B.
    # Each entry in `models` may include an optional ":N" suffix that
    # overrides batch_cap for local models (e.g. "zimage-turbo:8").
    # Cloud models ignore the suffix — they don't batch.
    for spec in models:
        model, override = parse_model_spec(spec)
        local = is_local_model(model)
        if local:
            from logo_gen.clients import local_diffusion
            default_cap = local_diffusion.get_max_batch(model)
            batch_cap = override if override is not None else default_cap
        else:
            batch_cap = 1

        for prompt_data in prompts:
            prompt_text = prompt_data.get("prompt", "")
            concept = prompt_data.get("concept", "")
            seeds = _make_seeds(seeds_per_prompt)

            if local:
                # Chunk seeds into max_batch-sized groups.
                for i in range(0, len(seeds), batch_cap):
                    chunk = seeds[i : i + batch_cap]
                    local_tasks.append(("local", prompt_text, concept, model, chunk))
            else:
                for seed in seeds:
                    cloud_tasks.append(("cloud", prompt_text, concept, model, seed))

    total_images = sum(len(t[4]) for t in local_tasks) + len(cloud_tasks)
    if progress_callback:
        progress_callback(0, total_images, f"Starting {total_images} images...")

    # Separate semaphores: cloud (3-wide) doesn't queue behind local.
    local_sem = asyncio.Semaphore(1)
    cloud_sem = asyncio.Semaphore(3)
    completed = {"n": 0}

    def _bump(by: int, message: str) -> None:
        completed["n"] += by
        if progress_callback:
            progress_callback(completed["n"], total_images, message)

    async def run_local(args: tuple) -> list[GeneratedLogo]:
        _, prompt_text, concept, model, seeds = args
        async with local_sem:
            if progress_callback:
                progress_callback(
                    completed["n"],
                    total_images,
                    f"{model}: batch of {len(seeds)}",
                )
            logos = await _generate_local_batch(
                prompt_text, concept, model, seeds, output_dir,
            )
            _bump(len(seeds), f"{model}: {completed['n']}/{total_images}")
            return logos

    async def run_cloud(args: tuple) -> list[GeneratedLogo]:
        _, prompt_text, concept, model, seed = args
        async with cloud_sem:
            if progress_callback:
                progress_callback(
                    completed["n"],
                    total_images,
                    f"{model} (seed {seed})",
                )
            logos = await _generate_cloud_one(
                prompt_text, concept, model, seed, output_dir, reference_images,
            )
            _bump(1, f"{model}: {completed['n']}/{total_images}")
            return logos

    coros = [run_local(t) for t in local_tasks] + [run_cloud(t) for t in cloud_tasks]
    results = await asyncio.gather(*coros, return_exceptions=True)

    for r in results:
        if isinstance(r, Exception):
            result.errors.append(str(r))
        elif isinstance(r, list):
            for logo in r:
                if logo.path != Path("error"):
                    result.logos.append(logo)
                else:
                    result.errors.append(f"Failed: {logo.model} seed={logo.seed}")

    result.total_time = time.time() - start

    if progress_callback:
        progress_callback(
            total_images,
            total_images,
            f"Done! {len(result.logos)} logos in {result.total_time:.1f}s",
        )

    return result


async def quick_generate(
    concept: str,
    models: list[str] | None = None,
    n_variations: int = 2,
    output_dir: Path | None = None,
    progress_callback=None,
) -> GenerationResult:
    """Quick generation: enhance prompt with LLM, then generate images.

    Convenience function that handles the full pipeline.
    """
    from logo_gen.prompt_engine import generate_variations

    if progress_callback:
        progress_callback(0, 1, "Enhancing prompt with AI...")

    prompts = await generate_variations(concept, n=4)

    return await generate_logos(
        prompts=prompts,
        models=models,
        seeds_per_prompt=n_variations,
        output_dir=output_dir,
        progress_callback=progress_callback,
    )
