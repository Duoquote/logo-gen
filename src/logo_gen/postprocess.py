"""Background removal for generated logos."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from rembg import remove, new_session

from logo_gen.config import settings

# Available rembg models with descriptions
MODELS = {
    "u2net": "U2-Net (default) - Fast, good for simple logos on solid backgrounds",
    "u2netp": "U2-Net Portable - Lightweight (~4MB), fastest, lower quality",
    "isnet-general-use": "ISNet/DIS - Best edge precision, good for thin lines & details",
    "birefnet-general": "BiRefNet - Best overall quality, handles gradients & shadows",
    "birefnet-general-lite": "BiRefNet Lite - Faster BiRefNet, slightly lower quality",
}

DEFAULT_MODEL = "birefnet-general"


def list_generated_images() -> list[Path]:
    """List all PNG images from both generated/ and upscaled/ directories."""
    images = []
    for dirname in (settings.output_dir, str(Path(settings.output_dir).parent / "upscaled")):
        d = Path(dirname)
        if d.exists():
            images.extend(d.glob("*.png"))
    return sorted(images, key=lambda p: p.stat().st_mtime, reverse=True)


def list_generated_images_labeled() -> list[tuple[str, str]]:
    """List images with labels showing source folder (generated vs upscaled)."""
    results = []
    gen_dir = Path(settings.output_dir)
    up_dir = Path(settings.output_dir).parent / "upscaled"

    if gen_dir.exists():
        for p in gen_dir.glob("*.png"):
            results.append((str(p), f"[generated] {p.name}"))
    if up_dir.exists():
        for p in up_dir.glob("*.png"):
            results.append((str(p), f"[upscaled] {p.name}"))

    results.sort(key=lambda x: Path(x[0]).stat().st_mtime, reverse=True)
    return results


def list_cleaned_images() -> list[Path]:
    """List all PNG images in the cleaned output directory."""
    clean_dir = Path(settings.cleaned_dir)
    if not clean_dir.exists():
        return []
    return sorted(clean_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)


def _erode_alpha(img: Image.Image, pixels: int) -> Image.Image:
    """Erode the alpha channel by N pixels to remove white fringe/halo."""
    if pixels <= 0:
        return img
    import numpy as np

    arr = np.array(img)
    alpha = arr[:, :, 3]

    # Use a circular kernel for natural-looking erosion
    kernel_size = pixels * 2 + 1
    y, x = np.ogrid[-pixels:pixels + 1, -pixels:pixels + 1]
    kernel = ((x * x + y * y) <= pixels * pixels).astype(np.uint8)

    import cv2
    eroded = cv2.erode(alpha, kernel, iterations=1)

    arr[:, :, 3] = eroded
    return Image.fromarray(arr)


def remove_background(
    image_path: Path,
    model_name: str = DEFAULT_MODEL,
    alpha_matting: bool = False,
    erode_pixels: int = 0,
) -> Path:
    """Remove background from an image and save to cleaned directory.

    Args:
        erode_pixels: Erode alpha mask by N pixels to remove white fringe.

    Returns the path to the cleaned image.
    """
    clean_dir = Path(settings.cleaned_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    img = Image.open(image_path).convert("RGBA")

    session = new_session(model_name)

    kwargs = {}
    if alpha_matting:
        kwargs["alpha_matting"] = True
        kwargs["alpha_matting_foreground_threshold"] = 240
        kwargs["alpha_matting_background_threshold"] = 10
        kwargs["alpha_matting_erode_size"] = 10

    result = remove(img, session=session, **kwargs)
    result = _erode_alpha(result, erode_pixels)

    out_name = f"{image_path.stem}_clean.png"
    out_path = clean_dir / out_name
    result.save(out_path)

    return out_path


def remove_background_batch(
    image_paths: list[Path],
    model_name: str = DEFAULT_MODEL,
    alpha_matting: bool = False,
    erode_pixels: int = 0,
    progress_callback=None,
) -> list[Path]:
    """Remove background from multiple images.

    Returns list of cleaned image paths.
    """
    session = new_session(model_name)
    clean_dir = Path(settings.cleaned_dir)
    clean_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(image_paths)

    kwargs = {}
    if alpha_matting:
        kwargs["alpha_matting"] = True
        kwargs["alpha_matting_foreground_threshold"] = 240
        kwargs["alpha_matting_background_threshold"] = 10
        kwargs["alpha_matting_erode_size"] = 10

    for i, path in enumerate(image_paths):
        if progress_callback:
            progress_callback(i, total, f"Processing {i+1}/{total}: {path.name}")

        try:
            img = Image.open(path).convert("RGBA")
            result = remove(img, session=session, **kwargs)
            result = _erode_alpha(result, erode_pixels)

            out_name = f"{path.stem}_clean.png"
            out_path = clean_dir / out_name
            result.save(out_path)
            results.append(out_path)
        except Exception as e:
            if progress_callback:
                progress_callback(i, total, f"Error on {path.name}: {e}")

    if progress_callback:
        progress_callback(total, total, f"Done! {len(results)}/{total} cleaned")

    return results
