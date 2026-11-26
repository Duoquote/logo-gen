"""Run installed local models with diffusers. One pipeline is kept in memory."""

from __future__ import annotations

import gc
import threading
from typing import Callable

from PIL import Image

from logo_gen import system
from logo_gen.local_models.manager import is_installed, model_dir
from logo_gen.local_models.registry import MODELS, LocalModel

_lock = threading.Lock()
_pipe = None
_pipe_id: str | None = None

PROMPT_SUFFIX = " Centered on a plain white background, flat vector logo icon, no text."


def _torch():
    import torch

    return torch


def _nf4(torch, components: list[str]):
    from diffusers.quantizers import PipelineQuantizationConfig

    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
        components_to_quantize=components,
    )


def _load(spec: LocalModel):
    torch = _torch()
    path = str(model_dir(spec))
    device = system.device_kind()
    if spec.support(device) == "no":
        raise RuntimeError(f"{spec.name} needs an NVIDIA GPU with CUDA.")

    if spec.pipeline == "zimage":
        from diffusers import ZImagePipeline

        pipe = ZImagePipeline.from_pretrained(path, dtype=torch.bfloat16, disable_mmap=True)
        pipe.enable_model_cpu_offload()
    elif spec.pipeline == "flux":
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            path, dtype=torch.bfloat16, disable_mmap=True,
            quantization_config=_nf4(torch, ["transformer", "text_encoder_2"]),
        )
        pipe.enable_model_cpu_offload()
    elif spec.pipeline == "flux2":
        from diffusers import Flux2KleinPipeline

        if device == "cuda":
            pipe = Flux2KleinPipeline.from_pretrained(
                path, dtype=torch.bfloat16, disable_mmap=True,
                quantization_config=_nf4(torch, ["transformer", "text_encoder"]),
            )
            pipe.enable_model_cpu_offload()
        else:
            # bitsandbytes 4-bit is CUDA only; run in bf16 on Apple Silicon.
            pipe = Flux2KleinPipeline.from_pretrained(path, dtype=torch.bfloat16).to(device)
    elif spec.pipeline == "sd":
        from diffusers import AutoPipelineForText2Image

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            path, dtype=dtype, variant=spec.variant,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        )
        pipe = pipe.to(device)
    else:
        raise ValueError(f"unknown pipeline {spec.pipeline}")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def unload(model_id: str | None = None) -> None:
    global _pipe, _pipe_id
    with _lock:
        if model_id is None or _pipe_id == model_id:
            _pipe = None
            _pipe_id = None
            gc.collect()
            try:
                torch = _torch()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


def _get(spec: LocalModel, on_status: Callable[[str], None]):
    global _pipe, _pipe_id
    if _pipe_id == spec.id and _pipe is not None:
        return _pipe
    if _pipe is not None:
        on_status("Unloading previous model")
    _pipe = None
    _pipe_id = None
    gc.collect()
    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    on_status(f"Loading {spec.name} (first run takes a while)")
    _pipe = _load(spec)
    _pipe_id = spec.id
    return _pipe


def generate(
    model_id: str,
    prompt: str,
    seeds: list[int],
    on_step: Callable[[int, int], None],
    on_status: Callable[[str], None],
    should_stop: Callable[[], bool],
) -> list[Image.Image]:
    """Blocking. Generates len(seeds) images in one batched call."""
    spec = MODELS.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown local model {model_id}")
    if not is_installed(spec):
        raise RuntimeError(f"{spec.name} is not installed. Install it on the Models page.")
    torch = _torch()
    with _lock:
        pipe = _get(spec, on_status)
        on_status(f"{spec.name}: generating")
        # CPU generators give the same image for a seed on every device.
        generators = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]

        def callback(p, step: int, timestep, kwargs):
            on_step(step + 1, spec.steps)
            if should_stop():
                p._interrupt = True
            return kwargs

        kwargs = dict(
            prompt=prompt + PROMPT_SUFFIX,
            num_inference_steps=spec.steps,
            guidance_scale=spec.guidance,
            height=spec.resolution,
            width=spec.resolution,
            num_images_per_prompt=len(seeds),
            generator=generators,
            callback_on_step_end=callback,
        )
        if spec.pipeline == "flux":
            kwargs["max_sequence_length"] = 256
        result = pipe(**kwargs)
        if should_stop():
            return []
        return list(result.images)
