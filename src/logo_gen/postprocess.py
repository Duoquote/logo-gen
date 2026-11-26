"""Background removal: rembg models or a flat color key."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image

from logo_gen import paths

MODELS = {
    "birefnet-general": "BiRefNet: best on gradients and shadows. Slow on CPU, first run takes a minute",
    "birefnet-general-lite": "BiRefNet Lite: faster, slightly lower quality",
    "isnet-general-use": "ISNet: precise edges, good for thin lines",
    "u2net": "U2-Net: fast, fine for simple logos on solid backgrounds",
    "u2netp": "U2-Net Portable: smallest and fastest, lowest quality",
}
DEFAULT_MODEL = "isnet-general-use"

_sessions: dict[str, object] = {}
_lock = threading.Lock()


def _session(model_name: str):
    from rembg import new_session

    with _lock:
        if model_name not in _sessions:
            _sessions[model_name] = new_session(model_name)
        return _sessions[model_name]


def _out_path(src: Path, tag: str) -> Path:
    paths.CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.CLEANED_DIR / f"{src.stem}_{tag}.png"
    i = 2
    while out.exists():
        out = paths.CLEANED_DIR / f"{src.stem}_{tag}_{i}.png"
        i += 1
    return out


def _erode_alpha(img: Image.Image, pixels: int) -> Image.Image:
    """Shrink the alpha mask by N pixels to remove a light fringe."""
    if pixels <= 0:
        return img
    import cv2

    arr = np.array(img)
    y, x = np.ogrid[-pixels : pixels + 1, -pixels : pixels + 1]
    kernel = ((x * x + y * y) <= pixels * pixels).astype(np.uint8)
    arr[:, :, 3] = cv2.erode(arr[:, :, 3], kernel, iterations=1)
    return Image.fromarray(arr)


def _corner_color(img: Image.Image) -> tuple[int, int, int]:
    im = img.convert("RGB")
    w, h = im.size
    corners = [im.getpixel((0, 0)), im.getpixel((w - 1, 0)), im.getpixel((0, h - 1)), im.getpixel((w - 1, h - 1))]
    return tuple(int(sum(c[i] for c in corners) / 4) for i in range(3))  # type: ignore[return-value]


def remove_background(src: Path, model_name: str = DEFAULT_MODEL, alpha_matting: bool = False, erode_pixels: int = 0) -> Path:
    from rembg import remove

    if model_name not in MODELS:
        raise ValueError(f"Unknown background model {model_name}")
    img = Image.open(src).convert("RGBA")
    kwargs = {}
    if alpha_matting:
        kwargs = {
            "alpha_matting": True,
            "alpha_matting_foreground_threshold": 240,
            "alpha_matting_background_threshold": 10,
            "alpha_matting_erode_size": 10,
        }
    result = remove(img, session=_session(model_name), **kwargs)
    result = _erode_alpha(result, erode_pixels)
    out = _out_path(src, "clean")
    result.save(out)
    return out


def color_distance(rgb: np.ndarray, target: tuple[int, int, int]) -> np.ndarray:
    """Euclidean RGB distance per pixel, 0 to about 441.

    Computed in int32: a squared difference of two 0-255 values reaches
    65025, which overflows int16 and turns the distance into NaN.
    """
    diff = rgb.astype(np.int32) - np.asarray(target, dtype=np.int32)
    return np.sqrt(np.sum(diff * diff, axis=-1, dtype=np.int64))


def color_key(img: Image.Image, target: tuple[int, int, int], tolerance: int) -> Image.Image:
    """Make pixels within tolerance (0-100) of the target color fully transparent."""
    arr = np.array(img.convert("RGBA"))
    dist = color_distance(arr[:, :, :3], target)
    arr[:, :, 3] = np.where(dist <= tolerance * 2.21, 0, arr[:, :, 3]).astype(np.uint8)
    return Image.fromarray(arr)


def remove_background_color(
    src: Path, tolerance: int = 30, color: tuple[int, int, int] | None = None, erode_pixels: int = 0
) -> Path:
    """Make pixels near one color transparent. Without a color, use the corners."""
    img = Image.open(src).convert("RGBA")
    result = _erode_alpha(color_key(img, color or _corner_color(img), tolerance), erode_pixels)
    out = _out_path(src, "key")
    result.save(out)
    return out


def parse_hex(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    s = value.lstrip("#")
    if len(s) != 6:
        return None
    try:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    except ValueError:
        return None
