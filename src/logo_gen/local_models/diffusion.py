"""Run installed local models with diffusers. One pipeline is kept in memory."""

from __future__ import annotations

import gc
import threading
import time
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


def _is_oom(e: BaseException) -> bool:
    return "OutOfMemoryError" in type(e).__name__ or "out of memory" in str(e).lower()


def _free(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run(pipe, spec: LocalModel, prompts: list[str], seeds: list[int], on_step, should_stop) -> list[Image.Image]:
    """One pipeline call: one image per (prompt, seed) pair."""
    torch = _torch()
    # CPU generators give the same image for a seed on every device.
    generators = [torch.Generator("cpu").manual_seed(int(s)) for s in seeds]

    def callback(p, step: int, timestep, kwargs):
        on_step(step + 1, spec.steps)
        if should_stop():
            p._interrupt = True
        return kwargs

    kwargs = dict(
        prompt=[p + PROMPT_SUFFIX for p in prompts],
        num_inference_steps=spec.steps,
        guidance_scale=spec.guidance,
        height=spec.resolution,
        width=spec.resolution,
        num_images_per_prompt=1,
        generator=generators,
        callback_on_step_end=callback,
    )
    if spec.pipeline == "flux":
        kwargs["max_sequence_length"] = 256
    return list(pipe(**kwargs).images)


def _run_stepping_down(pipe, spec, prompts, seeds, on_step, on_status, should_stop) -> list[Image.Image]:
    """Run a batch; on out-of-memory, free VRAM and retry as two halves, down to one image."""
    torch = _torch()
    try:
        return _run(pipe, spec, prompts, seeds, on_step, should_stop)
    except Exception as e:
        if not _is_oom(e):
            raise
        _free(torch)
        if len(seeds) == 1:
            raise
        half = len(seeds) // 2
        on_status(f"Out of memory at a batch of {len(seeds)}; retrying in batches of {half}")
        first = _run_stepping_down(pipe, spec, prompts[:half], seeds[:half], on_step, on_status, should_stop)
        if should_stop():
            return first
        return first + _run_stepping_down(pipe, spec, prompts[half:], seeds[half:], on_step, on_status, should_stop)


def _check(model_id: str) -> LocalModel:
    spec = MODELS.get(model_id)
    if spec is None:
        raise ValueError(f"Unknown local model {model_id}")
    if not is_installed(spec):
        raise RuntimeError(f"{spec.name} is not installed. Install it on the Models page.")
    return spec


def generate(
    model_id: str,
    prompts: list[str],
    seeds: list[int],
    on_step: Callable[[int, int], None],
    on_status: Callable[[str], None],
    should_stop: Callable[[], bool],
) -> list[Image.Image]:
    """Blocking. One image per (prompt, seed) pair, in one batched call when memory allows."""
    spec = _check(model_id)
    with _lock:
        pipe = _get(spec, on_status)
        on_status(f"{spec.name}: generating a batch of {len(seeds)}")
        images = _run_stepping_down(pipe, spec, prompts, seeds, on_step, on_status, should_stop)
        return [] if should_stop() else images


CALIBRATION_PROMPT = "A flat geometric logo icon of a fox head built from triangles, two colors"


def calibrate(
    model_id: str,
    max_batch: int,
    on_status: Callable[[str], None],
    should_stop: Callable[[], bool],
) -> dict:
    """Time batch sizes 1, 2, 4, ... and record peak GPU memory for each.

    Stops at the first out-of-memory, when a batch spills past the card's
    memory (Windows then pages into shared system memory and gets very slow),
    or at max_batch.
    """
    spec = _check(model_id)
    torch = _torch()
    cuda = torch.cuda.is_available() and system.device_kind() == "cuda"
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if cuda else None
    results: list[dict] = []
    with _lock:
        t0 = time.perf_counter()
        pipe = _get(spec, on_status)
        on_status("Warming up")
        _run(pipe, spec, [CALIBRATION_PROMPT], [1], lambda *_: None, should_stop)
        load_s = time.perf_counter() - t0
        sizes = [b for b in (1, 2, 3, 4, 6, 8, 12, 16) if b <= max_batch]
        for b in sizes:
            if should_stop():
                break
            ok = [r for r in results if r["ok"] and r.get("peak_gb")]
            if total_gb and len(ok) >= 2:
                # Peak memory grows about linearly with batch size; skip a size that can't fit.
                a, z = ok[-2], ok[-1]
                per = (z["peak_gb"] - a["peak_gb"]) / (z["batch"] - a["batch"])
                predicted = z["peak_gb"] + per * (b - z["batch"])
                if predicted > total_gb:
                    results.append({"batch": b, "ok": False,
                                    "error": f"skipped, would need about {predicted:.1f} GB"})
                    break
            on_status(f"Testing a batch of {b}")
            _free(torch)
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            try:
                _run(pipe, spec, [CALIBRATION_PROMPT] * b, list(range(1, b + 1)), lambda *_: None, should_stop)
            except Exception as e:
                if not _is_oom(e):
                    raise
                _free(torch)
                results.append({"batch": b, "ok": False, "error": "out of memory"})
                break
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_reserved() / 1e9 if cuda else None
            row = {"batch": b, "ok": True, "seconds": round(elapsed, 2), "per_image": round(elapsed / b, 2),
                   "peak_gb": round(peak, 2) if peak is not None else None}
            results.append(row)
            if total_gb and peak and peak > total_gb:
                row["note"] = "spilled past the card's memory into system RAM, so it slowed down"
                break
            if total_gb and peak and peak > total_gb * 0.95:
                row["note"] = "at the card's memory limit"
                break
            ok = [r for r in results if r["ok"]]
            # Two sizes in a row slower per image than the best: bigger won't help.
            if len(ok) >= 3 and ok[-1]["per_image"] > min(r["per_image"] for r in ok[:-1]) and (
                ok[-2]["per_image"] > min(r["per_image"] for r in ok[:-2])
            ):
                break
    return {"model_id": model_id, "device": system.device_kind(), "vram_gb": round(total_gb, 1) if total_gb else None,
            "load_seconds": round(load_s, 1), "results": results, "options": calibration_options(results, total_gb)}


def calibration_options(results: list[dict], total_gb: float | None) -> list[dict]:
    ok = [r for r in results if r["ok"]]
    if not ok:
        return []
    fastest = min(ok, key=lambda r: r["per_image"])
    balanced = next(
        (r for r in ok if r["per_image"] <= fastest["per_image"] * 1.15
         and (not total_gb or not r["peak_gb"] or r["peak_gb"] <= total_gb * 0.8)),
        fastest,
    )
    one = ok[0]
    options, seen = [], set()
    for key, label, r in (
        ("fastest", "Fastest overall", fastest),
        ("single", "One image at a time (lowest VRAM)", one),
        ("balanced", "Fast, with VRAM to spare", balanced),
    ):
        if r["batch"] in seen:
            continue
        seen.add(r["batch"])
        options.append({"key": key, "label": label, **r})
    return options
