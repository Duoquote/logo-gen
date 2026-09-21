"""Local image generation using diffusers (FLUX.1 Schnell, Z-Image-Turbo).

Provides the same interface as openrouter.generate_image() so the generator
can route to either backend transparently.
"""

from __future__ import annotations

import asyncio
import gc
import os
import re
import time
from pathlib import Path

import torch

from logo_gen.config import settings

# Lazy-loaded pipeline singletons keyed by model name
_pipelines: dict[str, object] = {}
_current_model: str | None = None

# Models directory at project root
MODELS_DIR = Path(__file__).resolve().parent.parent.parent.parent / ".models"

# Supported local models and their HuggingFace repo IDs.
#
# `max_batch` controls how many images can be generated in a single
# pipeline call via num_images_per_prompt. zimage-turbo is small enough
# to batch comfortably; flux1-schnell stays at 1 because it uses far
# more VRAM per inference.
LOCAL_MODELS = {
    "flux1-schnell": {
        "repo": "black-forest-labs/FLUX.1-schnell",
        "pipeline": "flux",
        "steps": 4,
        "guidance_scale": 0.0,
        "max_sequence_length": 256,
        "max_batch": 1,
    },
    "zimage-turbo": {
        "repo": "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
        "pipeline": "zimage",
        "steps": 9,
        "guidance_scale": 0.0,
        "max_batch": 4,
    },
}


def get_max_batch(model: str) -> int:
    """Return the max images-per-call this local model supports.

    Falls back to 1 if the model isn't recognised — generator code
    treats that as 'no batching'.
    """
    lower = model.lower().replace(".", "").replace(" ", "")
    for key, cfg in LOCAL_MODELS.items():
        if key in lower:
            return int(cfg.get("max_batch", 1))
    return 1


def parse_model_spec(spec: str) -> tuple[str, int | None]:
    """Parse a model spec like ``zimage-turbo:4`` into ``(name, 4)``.

    The trailing ``:N`` is an optional per-call batch override. If
    omitted (or N doesn't parse as a sane positive int) the spec is
    treated as a plain model name. We split on the *last* colon so
    legitimate names that happen to contain a colon (uncommon for
    image models but possible) won't be mangled.
    """
    name = spec.strip()
    head, sep, tail = name.rpartition(":")
    if sep and head:
        try:
            n = int(tail)
            if 1 <= n <= 100:
                return (head.strip(), n)
        except ValueError:
            pass
    return (name, None)

# Prompt suffix appended to all generation requests (matches openrouter behavior)
_PROMPT_SUFFIX = (
    "\nGenerate this as a single logo design on a clean solid white background. "
    "No text, no letters, no words anywhere in the image. "
    "The design should work as a standalone icon or app icon."
)


def _get_cache_dir() -> str:
    """Return the HuggingFace cache directory inside .models/."""
    cache = MODELS_DIR / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    return str(cache)


def _load_flux_pipeline():
    """Load FLUX.1 Schnell with BNB 4-bit quantization."""
    from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
    from diffusers import FluxPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBnBConfig

    quant_config = PipelineQuantizationConfig(
        quant_mapping={
            "transformer": DiffusersBnBConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            "text_encoder_2": TransformersBnBConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        }
    )

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        cache_dir=_get_cache_dir(),
        token=os.getenv("HF_TOKEN") or None,
        disable_mmap=True,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _load_zimage_pipeline():
    """Load Z-Image-Turbo with Unsloth BNB 4-bit quantization."""
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(
        "unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
        torch_dtype=torch.bfloat16,
        cache_dir=_get_cache_dir(),
        token=os.getenv("HF_TOKEN") or None,
        disable_mmap=True,
    )
    pipe.enable_model_cpu_offload()
    return pipe


def _get_pipeline(model_key: str):
    """Get or load a pipeline, unloading the previous one if different."""
    global _current_model

    if _current_model == model_key and model_key in _pipelines:
        return _pipelines[model_key]

    # Unload previous pipeline to free VRAM
    if _current_model and _current_model in _pipelines:
        del _pipelines[_current_model]
        gc.collect()
        torch.cuda.empty_cache()

    loaders = {
        "flux1-schnell": _load_flux_pipeline,
        "zimage-turbo": _load_zimage_pipeline,
    }

    if model_key not in loaders:
        raise ValueError(f"Unknown local model: {model_key}. Available: {list(loaders)}")

    pipe = loaders[model_key]()
    _pipelines[model_key] = pipe
    _current_model = model_key
    return pipe


def _generate_sync(
    prompt: str,
    model_key: str,
    seeds: list[int],
    save_dir: Path | None = None,
) -> list[Path]:
    """Synchronous image generation (runs on GPU).

    `seeds` is a list of one or more seeds. When the list has multiple
    entries we use diffusers' num_images_per_prompt + a parallel
    generator list to batch them in a single pipeline call — far faster
    than N sequential calls because the GPU schedules the work
    together. Each model declares its own max_batch in LOCAL_MODELS;
    the generator is responsible for splitting larger requests into
    chunks that respect that ceiling.
    """
    if not seeds:
        return []

    config = LOCAL_MODELS[model_key]
    pipe = _get_pipeline(model_key)

    full_prompt = prompt + _PROMPT_SUFFIX
    n = len(seeds)

    gen_kwargs = {
        "prompt": full_prompt,
        "num_inference_steps": config["steps"],
        "guidance_scale": config["guidance_scale"],
        "height": 1024,
        "width": 1024,
        "num_images_per_prompt": n,
    }

    if config["pipeline"] == "flux":
        gen_kwargs["max_sequence_length"] = config.get("max_sequence_length", 256)
        gen_kwargs["generator"] = [
            torch.Generator("cpu").manual_seed(s or 42) for s in seeds
        ]
    else:
        gen_kwargs["generator"] = [
            torch.Generator("cuda").manual_seed(s or 42) for s in seeds
        ]

    images = pipe(**gen_kwargs).images

    # Save each image with the seed it was generated from.
    save_dir = save_dir or Path(settings.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    slug = re.sub(r"[^a-z0-9]", "-", model_key.lower())[:30]
    paths: list[Path] = []
    for image, seed in zip(images, seeds):
        seed_str = seed or 0
        timestamp = int(time.time() * 1000) % 1_000_000
        filename = f"{slug}_s{seed_str}_{timestamp}.png"
        path = save_dir / filename
        image.save(str(path))
        paths.append(path)
        # Bump the timestamp slot so multi-image batches don't collide.
        time.sleep(0.001)

    return paths


def _resolve_model_key(model: str) -> str:
    """Map a possibly-decorated model name back to a LOCAL_MODELS key."""
    lower = model.lower().replace(".", "").replace(" ", "")
    for key in LOCAL_MODELS:
        if key in lower:
            return key
    return model


async def generate_image(
    prompt: str,
    model: str,
    seed: int | None = None,
    seeds: list[int] | None = None,
    save_dir: Path | None = None,
) -> list[Path]:
    """Async wrapper matching openrouter.generate_image()'s interface,
    extended with a `seeds` list for batched local generation.

    Pass either `seed` (single image, legacy callers) or `seeds`
    (batched, generator caller). When both are omitted we use a single
    None-seeded image so the call still works as a smoke test.
    """
    model_key = _resolve_model_key(model)

    if seeds is None:
        seeds = [seed if seed is not None else 0]

    return await asyncio.to_thread(
        _generate_sync, prompt, model_key, seeds, save_dir
    )


def is_local_model(model: str) -> bool:
    """Check if a model name refers to a local model."""
    lower = model.lower().replace(".", "").replace(" ", "")
    return any(key in lower for key in LOCAL_MODELS)


def unload():
    """Unload all pipelines and free VRAM."""
    global _current_model
    _pipelines.clear()
    _current_model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
