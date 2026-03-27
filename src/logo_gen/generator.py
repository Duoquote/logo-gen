"""Logo generation orchestrator - runs prompts through multiple models/seeds."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from logo_gen.clients import openrouter
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


async def _generate_one(
    prompt: str,
    concept: str,
    model: str,
    seed: int | None,
    output_dir: Path,
) -> list[GeneratedLogo]:
    """Generate a single image and return results."""
    start = time.time()
    try:
        paths = await openrouter.generate_image(
            prompt=prompt,
            model=model,
            seed=seed,
            save_dir=output_dir,
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
    except Exception as e:
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


def _make_seeds(n: int) -> list[int]:
    """Generate diverse seeds."""
    return [random.randint(1, 2**31) for _ in range(n)]


async def generate_logos(
    prompts: list[dict],
    models: list[str] | None = None,
    seeds_per_prompt: int | None = None,
    output_dir: Path | None = None,
    progress_callback=None,
) -> GenerationResult:
    """Generate logos from a list of prompt dicts across models and seeds.

    Args:
        prompts: List of dicts with 'prompt', 'concept', 'style' keys
        models: Image generation models to use (defaults to settings)
        seeds_per_prompt: How many seed variations per prompt/model combo
        output_dir: Where to save images
        progress_callback: Optional callable(current, total, message) for progress updates
    """
    models = models or settings.image_models
    seeds_per_prompt = seeds_per_prompt or settings.images_per_model
    output_dir = Path(output_dir or settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    result = GenerationResult()

    # Build all generation tasks
    tasks = []
    for prompt_data in prompts:
        prompt_text = prompt_data.get("prompt", "")
        concept = prompt_data.get("concept", "")
        seeds = _make_seeds(seeds_per_prompt)

        for model in models:
            for seed in seeds:
                tasks.append((prompt_text, concept, model, seed, output_dir))

    total = len(tasks)
    if progress_callback:
        progress_callback(0, total, f"Starting {total} generation tasks...")

    # Run with limited concurrency to avoid rate limits
    semaphore = asyncio.Semaphore(3)

    async def run_with_sem(idx, args):
        async with semaphore:
            if progress_callback:
                progress_callback(
                    idx, total, f"Generating {idx+1}/{total}: {args[2]} (seed {args[3]})"
                )
            return await _generate_one(*args)

    coros = [run_with_sem(i, t) for i, t in enumerate(tasks)]
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
            total,
            total,
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
