"""Local image models the app knows how to install and run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LocalModel:
    id: str
    name: str
    repo: str
    pipeline: str  # zimage | flux | flux2 | sd
    allow_patterns: tuple[str, ...]
    license: str
    license_url: str
    vram_gb: float  # what it needs on the GPU with CPU offload
    size_gb: float  # download size, refreshed from the Hub when online
    steps: int
    guidance: float
    resolution: int
    blurb: str
    # How it runs without an NVIDIA GPU: ok, slow or no
    mps: str = "no"
    cpu: str = "no"
    max_batch: int = 1
    variant: str | None = None
    ignore_patterns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_cuda(self) -> bool:
        return self.mps == "no" and self.cpu == "no"

    def support(self, device: str) -> str:
        """ok, slow or no for a device kind (cuda, mps, cpu)."""
        if device == "cuda":
            return "ok"
        return self.mps if device == "mps" else self.cpu

    @property
    def page_url(self) -> str:
        return f"https://huggingface.co/{self.repo}"


_DIFFUSERS_FILES = ("model_index.json", "scheduler/*")

MODELS: dict[str, LocalModel] = {}


def register(m: LocalModel) -> None:
    MODELS[m.id] = m


register(LocalModel(
    id="zimage-turbo",
    name="Z-Image Turbo (4-bit)",
    repo="unsloth/Z-Image-Turbo-unsloth-bnb-4bit",
    pipeline="zimage",
    allow_patterns=(*_DIFFUSERS_FILES, "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"),
    license="Apache-2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0",
    size_gb=6.8,
    vram_gb=8,
    steps=9,
    guidance=1.0,
    resolution=1024,
    max_batch=4,
    blurb="6B model, pre-quantized to 4-bit by Unsloth. Clean, minimal results in about 15 s.",
))

register(LocalModel(
    id="flux2-klein-4b",
    name="FLUX.2 klein 4B",
    repo="black-forest-labs/FLUX.2-klein-4B",
    pipeline="flux2",
    allow_patterns=(*_DIFFUSERS_FILES, "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*"),
    license="Apache-2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0",
    size_gb=16.0,
    vram_gb=8,
    steps=4,
    guidance=1.0,
    resolution=1024,
    blurb="Black Forest Labs' small FLUX.2 model, 4 steps. Quantized to 4-bit on NVIDIA GPUs.",
    mps="slow",
))

register(LocalModel(
    id="flux1-schnell",
    name="FLUX.1 schnell",
    repo="black-forest-labs/FLUX.1-schnell",
    pipeline="flux",
    allow_patterns=(
        *_DIFFUSERS_FILES, "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*",
        "transformer/*", "vae/*",
    ),
    license="Apache-2.0",
    license_url="https://www.apache.org/licenses/LICENSE-2.0",
    size_gb=34.0,
    vram_gb=10,
    steps=4,
    guidance=0.0,
    resolution=1024,
    blurb="12B model, 4 steps, most detail. Gated: accept the terms on Hugging Face first. Large download.",
))

register(LocalModel(
    id="sd-turbo",
    name="SD-Turbo",
    repo="stabilityai/sd-turbo",
    pipeline="sd",
    allow_patterns=(
        *_DIFFUSERS_FILES, "text_encoder/config.json", "text_encoder/model.fp16.safetensors", "tokenizer/*",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
    ),
    license="Stability AI Community",
    license_url="https://stability.ai/license",
    size_gb=2.6,
    vram_gb=3,
    steps=2,
    guidance=0.0,
    resolution=512,
    max_batch=4,
    variant="fp16",
    mps="ok",
    cpu="slow",
    blurb="Small and fast, 512 px. Runs on modest GPUs, Apple Silicon and (slowly) CPU. Rough results.",
))
