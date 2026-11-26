"""Image upscaling with multiple methods."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
from PIL import Image

from logo_gen import paths

logger = logging.getLogger(__name__)

UPSCALED_DIR = paths.UPSCALED_DIR

# --- Model URLs ---
_MODELS = {
    "realesrgan-anime": {
        "name": "Real-ESRGAN Anime 6B - Best for flat/illustrative logos, clean edges",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "native_scale": 4,
    },
    "realesrgan-general": {
        "name": "Real-ESRGAN x4plus - Best for complex/photorealistic logos",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "filename": "RealESRGAN_x4plus.pth",
        "native_scale": 4,
    },
    "lanczos": {
        "name": "Lanczos (Pillow) - Fast, no GPU needed, good for clean art",
    },
    "cubic": {
        "name": "Bicubic (OpenCV) - Fast, no GPU needed, smooth interpolation",
    },
}

METHODS = {k: v["name"] for k, v in _MODELS.items()}

# Scale factor options
SCALES = {
    "2x": 2,
    "4x": 4,
    "8x (2x4x)": 8,
}

_LEGACY_CACHE = Path.home() / ".cache" / "logo-gen" / "models"
_CACHE_DIR = paths.MODELS_DIR / "upscalers"

# Singleton model cache
_loaded_model = None
_loaded_model_name = None


def _download_weights(url: str, filename: str) -> Path:
    """Download model weights if not cached."""
    legacy = _LEGACY_CACHE / filename
    if legacy.exists():
        return legacy
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / filename
    if not path.exists():
        logger.info("Downloading %s -> %s", url, path)
        urlretrieve(url, path)
        logger.info("Download complete: %s", path)
    return path


def _get_device() -> str:
    from logo_gen import system

    kind = system.device_kind()
    try:
        import torch

        if kind == "cuda" and torch.cuda.is_available():
            return "cuda"
        if kind == "mps" and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _load_ai_model(method: str):
    """Load a spandrel-compatible model."""
    global _loaded_model, _loaded_model_name

    if _loaded_model_name == method and _loaded_model is not None:
        return _loaded_model

    # Unload previous
    if _loaded_model is not None:
        del _loaded_model
        _loaded_model = None
        _loaded_model_name = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    model_info = _MODELS[method]
    weights_path = _download_weights(model_info["url"], model_info["filename"])

    model = ModelLoader().load_from_file(weights_path)
    assert isinstance(model, ImageModelDescriptor)
    model.eval()

    device = _get_device()
    model = model.to(torch.device(device))

    _loaded_model = model
    _loaded_model_name = method
    return model


def _tiled_inference(model, tensor, tile_size: int = 512, scale: int = 4):
    """Run inference in tiles to avoid VRAM OOM."""
    import torch

    _, _, h, w = tensor.shape
    if h <= tile_size and w <= tile_size:
        return model(tensor)

    overlap = 32
    out_h, out_w = h * scale, w * scale
    output = torch.zeros(1, 3, out_h, out_w, device=tensor.device)
    count = torch.zeros(1, 1, out_h, out_w, device=tensor.device)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            tile = tensor[:, :, y_start:y_end, x_start:x_end]
            tile_out = model(tile)

            oy, ox = y_start * scale, x_start * scale
            oh, ow = tile_out.shape[2], tile_out.shape[3]
            output[:, :, oy:oy + oh, ox:ox + ow] += tile_out
            count[:, :, oy:oy + oh, ox:ox + ow] += 1

    return output / count.clamp(min=1)


def _upscale_ai(image: np.ndarray, method: str, scale: int) -> np.ndarray:
    """Upscale with Real-ESRGAN via spandrel."""
    import torch

    model = _load_ai_model(method)
    device = _get_device()
    native_scale = _MODELS[method]["native_scale"]

    # Handle RGBA
    has_alpha = image.shape[2] == 4 if len(image.shape) == 3 else False
    if has_alpha:
        rgb = image[..., :3]
        alpha = image[..., 3]
    else:
        rgb = image

    # To torch: HWC uint8 -> BCHW float32 [0,1]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        output = _tiled_inference(model, tensor, tile_size=512, scale=native_scale)

    # Back to numpy
    output_np = (output.squeeze(0).permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    h_out, w_out = output_np.shape[:2]

    # Upscale alpha with Lanczos
    if has_alpha:
        alpha_up = cv2.resize(alpha, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)

    # If requested scale != native scale, resize
    target_h = image.shape[0] * scale
    target_w = image.shape[1] * scale
    if (h_out, w_out) != (target_h, target_w):
        output_np = cv2.resize(output_np, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        if has_alpha:
            alpha_up = cv2.resize(alpha_up, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    if has_alpha:
        return np.dstack([output_np, alpha_up])
    return output_np


def _upscale_lanczos(image: np.ndarray, scale: int) -> np.ndarray:
    """Upscale with Pillow Lanczos."""
    h, w = image.shape[:2]
    pil_img = Image.fromarray(image)
    upscaled = pil_img.resize((w * scale, h * scale), Image.LANCZOS)
    return np.array(upscaled)


def _upscale_cubic(image: np.ndarray, scale: int) -> np.ndarray:
    """Upscale with OpenCV bicubic interpolation."""
    h, w = image.shape[:2]
    return cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def upscale_image(
    image_path: Path,
    method: str = "lanczos",
    scale: int = 4,
) -> Path:
    """Upscale an image and save to the upscaled directory.

    Returns path to the upscaled image.
    """
    UPSCALED_DIR.mkdir(parents=True, exist_ok=True)

    img = np.array(Image.open(image_path).convert("RGBA"))

    if method not in _MODELS:
        raise ValueError(f"Unknown upscale method {method}")
    if scale not in SCALES.values():
        raise ValueError(f"Scale must be one of {sorted(SCALES.values())}")
    if method in ("realesrgan-anime", "realesrgan-general"):
        from logo_gen import system

        if not system.local_stack_installed():
            raise ValueError("Real-ESRGAN needs local model support (PyTorch). Install it on the Models page, "
                             "or use Lanczos.")
        result = _upscale_ai(img, method, scale)
    elif method == "cubic":
        result = _upscale_cubic(img, scale)
    else:
        result = _upscale_lanczos(img, scale)

    out_path = UPSCALED_DIR / f"{image_path.stem}_{method}_{scale}x.png"
    i = 2
    while out_path.exists():
        out_path = UPSCALED_DIR / f"{image_path.stem}_{method}_{scale}x_{i}.png"
        i += 1
    Image.fromarray(result).save(out_path)

    return out_path
