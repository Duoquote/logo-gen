# 13. Background Removal for Logo Generation

## Overview

Background removal is a critical post-processing step in automated logo generation pipelines. Most image generation models output logos on colored or textured backgrounds, even when prompted for transparency. This document covers the tools, models, techniques, and code needed to produce clean transparent PNGs suitable for production use.

---

## 1. rembg: The Standard Python Library

### Installation

```bash
# Basic install (includes default U2-Net model)
pip install rembg

# With GPU support (CUDA)
pip install rembg[gpu]

# With CLI support
pip install rembg[cli]

# Full install (all optional dependencies)
pip install rembg[gpu,cli]
```

rembg depends on onnxruntime (CPU) or onnxruntime-gpu (GPU). Models are downloaded automatically on first use and cached in `~/.u2net/`.

### Basic Usage

```python
from rembg import remove
from PIL import Image

input_image = Image.open("logo_with_bg.png")
output_image = remove(input_image)
output_image.save("logo_transparent.png")
```

### CLI Usage

```bash
rembg i input.png output.png
rembg p input_folder/ output_folder/   # batch processing
```

---

## 2. rembg Models: Which to Use When

rembg supports multiple models, selectable via the `model_name` parameter.

### U2-Net (default: `u2net`)

- **Architecture**: U-squared Net, a two-level nested U-structure for salient object detection.
- **Size**: ~176 MB.
- **Strengths**: Good general-purpose segmentation; fast; reliable on high-contrast subjects.
- **Weaknesses**: Can struggle with fine edges (hair, thin lines), semi-transparent areas.
- **Best for**: Simple logos with clear subject/background separation. The default choice for most logo work.

```python
output = remove(input_image, model_name="u2net")
```

**Variants**:
- `u2netp` -- lightweight version (~4 MB), faster but less accurate. Good for previews or resource-constrained environments.
- `u2net_human_seg` -- trained specifically on human segmentation. Not useful for logos.
- `u2net_cloth_seg` -- trained on clothing segmentation. Not useful for logos.

### ISNet / DIS (model: `isnet-general-use`)

- **Architecture**: Intermediate Supervision Net, designed for dichotomous image segmentation.
- **Size**: ~170 MB.
- **Strengths**: Excellent edge precision; designed for highly accurate foreground/background separation.
- **Weaknesses**: Slightly slower than U2-Net.
- **Best for**: Logos with intricate details, thin strokes, or fine typography where edge accuracy matters.

```python
output = remove(input_image, model_name="isnet-general-use")
```

### BiRefNet (model: `birefnet-general`)

- **Architecture**: Bilateral Reference Network. Uses bilateral reference approach for high-resolution segmentation.
- **Size**: ~200-400 MB depending on variant.
- **Strengths**: State-of-the-art accuracy (as of 2025-2026); excellent with complex edges, semi-transparent regions, and fine details.
- **Weaknesses**: Heavier model, slower inference; requires more VRAM.
- **Best for**: High-quality production output where accuracy is paramount. Complex logos with gradients, shadows, or overlapping elements.

```python
output = remove(input_image, model_name="birefnet-general")
```

**BiRefNet variants available in rembg**:
- `birefnet-general` -- general purpose, best overall quality.
- `birefnet-general-lite` -- smaller, faster, slightly lower quality.
- `birefnet-portrait` -- optimized for portrait photos, less relevant for logos.
- `birefnet-massive` -- trained on larger dataset, can be better for unusual subjects.

### SAM (Segment Anything Model)

SAM is not directly included in rembg as a model option but can be used alongside it or independently. SAM excels at interactive segmentation where you provide point or box prompts.

- **Architecture**: Vision Transformer (ViT) based encoder with prompt-based mask decoder.
- **Size**: ViT-H ~2.4 GB, ViT-L ~1.2 GB, ViT-B ~375 MB.
- **Strengths**: Extremely flexible; can segment any object given a prompt; zero-shot generalization.
- **Weaknesses**: Requires prompts (points, boxes, or masks) -- not fully automatic without additional logic; large model size; slower.
- **Best for**: Cases where automatic models fail and you need precise control. Useful for logos embedded in complex scenes. SAM 2 (2024-2025) improves speed and adds video support.

```python
# SAM usage (standalone, not via rembg)
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image_array)

# Provide center point of the logo as prompt
masks, scores, logits = predictor.predict(
    point_coords=np.array([[256, 256]]),
    point_labels=np.array([1]),
    multimask_output=True
)
best_mask = masks[np.argmax(scores)]
```

### Model Selection Guide for Logos

| Scenario | Recommended Model | Reason |
|---|---|---|
| Simple logo on white/solid bg | `u2net` or threshold-based | Fast, reliable |
| Logo with fine text/thin lines | `isnet-general-use` | Best edge precision |
| Complex logo with gradients/shadows | `birefnet-general` | Best overall quality |
| Batch processing (speed matters) | `u2netp` | Lightweight, fast |
| Production-quality final output | `birefnet-general` | State of the art |
| Auto models fail, need manual control | SAM (standalone) | Prompt-based precision |

---

## 3. Alpha Matting Parameters

rembg supports alpha matting for improved edge handling. Alpha matting refines the binary mask produced by the segmentation model, creating smooth semi-transparent edges instead of hard cutoffs.

```python
output = remove(
    input_image,
    alpha_matting=True,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=10
)
```

### Parameters

- **`alpha_matting`** (bool, default `False`): Enable alpha matting post-processing.
- **`alpha_matting_foreground_threshold`** (int, default `240`): Pixels with alpha above this value are treated as definite foreground. Range 0-255. Higher = stricter foreground classification.
- **`alpha_matting_background_threshold`** (int, default `10`): Pixels with alpha below this value are treated as definite background. Range 0-255. Lower = stricter background classification.
- **`alpha_matting_erode_size`** (int, default `10`): Size of the erosion kernel applied to the trimap. Larger values create a wider "unknown" region at edges, giving alpha matting more room to work. Increase for logos with soft edges or anti-aliased borders.

### Tuning for Logos

- **Flat/geometric logos**: Use defaults or slightly tighten thresholds (fg=245, bg=5). Clean edges are usually sufficient.
- **Logos with glows/shadows**: Widen the unknown region (erode_size=15-20) and relax thresholds (fg=220, bg=20) to preserve soft effects.
- **Logos with thin strokes**: Be cautious with alpha matting -- it can erode thin features. Use a smaller erode_size (5-8) or skip alpha matting and rely on model quality.

---

## 4. Threshold-Based Removal for Simple Backgrounds

For logos generated on pure white or pure black backgrounds, threshold-based removal is often faster and more reliable than neural network approaches.

### White Background Removal

```python
from PIL import Image
import numpy as np

def remove_white_background(image_path, threshold=240, output_path=None):
    """Remove white or near-white background from an image."""
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)

    # Identify pixels where R, G, B are all above threshold
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)

    # Set alpha to 0 for white pixels
    data[white_mask, 3] = 0

    result = Image.fromarray(data)
    if output_path:
        result.save(output_path)
    return result
```

### Black Background Removal

```python
def remove_black_background(image_path, threshold=15, output_path=None):
    """Remove black or near-black background from an image."""
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]
    black_mask = (r < threshold) & (g < threshold) & (b < threshold)

    data[black_mask, 3] = 0

    result = Image.fromarray(data)
    if output_path:
        result.save(output_path)
    return result
```

### Gradient Threshold (Smooth Edges)

A hard threshold produces jagged edges. A gradient approach maps near-background colors to partial transparency:

```python
def remove_background_gradient(image_path, bg_color=(255, 255, 255),
                                threshold_low=230, threshold_high=250):
    """Remove background with smooth alpha transition at edges."""
    img = Image.open(image_path).convert("RGBA")
    data = np.array(img, dtype=np.float32)

    # Calculate distance from background color
    bg = np.array(bg_color, dtype=np.float32)
    distance = np.sqrt(np.sum((data[:, :, :3] - bg) ** 2, axis=2))

    # Map distance to alpha: close to bg = transparent, far = opaque
    # distance 0 -> alpha 0, distance at threshold -> alpha 255
    max_dist = np.sqrt(3 * (threshold_high - threshold_low) ** 2) + 1
    alpha = np.clip(distance / max_dist * 255, 0, 255).astype(np.uint8)

    # Only apply to pixels that are near the background color
    near_bg = distance < max_dist
    result = np.array(img)
    result[near_bg, 3] = alpha[near_bg]

    return Image.fromarray(result)
```

### When to Use Threshold vs. Neural Network

| Factor | Threshold-Based | Neural Network (rembg) |
|---|---|---|
| Pure white/black background | Preferred -- fast, exact | Works but overkill |
| Gradient or textured background | Fails | Required |
| Logo contains white/black elements | Risky (may remove logo parts) | Handles correctly |
| Speed requirement | ~1ms | ~100ms-2s |
| Batch of 10,000+ images | Excellent | Feasible with GPU |

---

## 5. Post-Removal Cleanup

### Edge Smoothing

After background removal, edges may appear jagged or have remnant fringe pixels. Common cleanup steps:

```python
from PIL import Image, ImageFilter
import numpy as np

def cleanup_edges(image, feather_radius=1, defringe=True):
    """Clean up edges after background removal."""
    img = image.copy()

    if defringe:
        img = remove_color_fringe(img)

    if feather_radius > 0:
        img = feather_edges(img, feather_radius)

    return img


def remove_color_fringe(image, fringe_width=2):
    """Remove color fringe (halo) at transparent edges.

    Common issue: white/colored fringe from the original background
    bleeding into edge pixels.
    """
    data = np.array(image)
    alpha = data[:, :, 3]

    # Find edge pixels (partially transparent)
    edge_mask = (alpha > 0) & (alpha < 255)

    if not np.any(edge_mask):
        return image

    # For edge pixels, blend RGB toward nearest fully opaque pixel color
    # Simple approach: erode alpha slightly and premultiply
    from scipy.ndimage import binary_erosion, binary_dilation

    opaque_mask = alpha == 255
    eroded = binary_erosion(opaque_mask, iterations=fringe_width)

    # Replace fringe pixel colors with average of nearby opaque pixels
    from scipy.ndimage import uniform_filter
    for c in range(3):
        channel = data[:, :, c].astype(np.float32)
        weight = opaque_mask.astype(np.float32)
        smoothed = uniform_filter(channel * weight, size=fringe_width * 2 + 1)
        weight_smoothed = uniform_filter(weight, size=fringe_width * 2 + 1)
        weight_smoothed = np.maximum(weight_smoothed, 1e-6)
        filled = smoothed / weight_smoothed

        # Apply only to fringe pixels
        fringe = edge_mask | (opaque_mask & ~eroded)
        data[fringe, c] = np.clip(filled[fringe], 0, 255).astype(np.uint8)

    return Image.fromarray(data)


def feather_edges(image, radius=1):
    """Apply slight feathering to alpha channel edges for anti-aliasing."""
    data = np.array(image)
    alpha = Image.fromarray(data[:, :, 3])

    # Slight Gaussian blur on the alpha channel
    alpha_blurred = alpha.filter(ImageFilter.GaussianBlur(radius=radius))

    # Only apply blur where alpha was already partially transparent or at edges
    alpha_arr = np.array(alpha)
    blurred_arr = np.array(alpha_blurred)

    # Preserve fully opaque interior, smooth only edges
    edge_region = (alpha_arr > 0) & (alpha_arr < 255)
    # Expand edge region slightly
    from scipy.ndimage import binary_dilation
    edge_region = binary_dilation(edge_region, iterations=radius)

    result_alpha = alpha_arr.copy()
    result_alpha[edge_region] = blurred_arr[edge_region]

    data[:, :, 3] = result_alpha
    return Image.fromarray(data)
```

### Morphological Cleanup

```python
from PIL import ImageFilter, ImageOps
import numpy as np

def morphological_cleanup(image, close_size=2, open_size=1):
    """Use morphological operations to clean up the alpha mask.

    - Closing: fills small holes in the foreground
    - Opening: removes small noise in the background
    """
    data = np.array(image)
    alpha = data[:, :, 3]

    from scipy.ndimage import binary_closing, binary_opening

    mask = alpha > 128

    # Close small gaps
    if close_size > 0:
        struct = np.ones((close_size * 2 + 1, close_size * 2 + 1))
        mask = binary_closing(mask, structure=struct)

    # Remove small specks
    if open_size > 0:
        struct = np.ones((open_size * 2 + 1, open_size * 2 + 1))
        mask = binary_opening(mask, structure=struct)

    # Reapply: keep original alpha where mask is True, 0 elsewhere
    data[:, :, 3] = np.where(mask, data[:, :, 3], 0)

    return Image.fromarray(data)
```

---

## 6. Transparent PNG Generation and Handling

### Saving Transparent PNGs

```python
from PIL import Image

# Ensure RGBA mode
img = img.convert("RGBA")

# Save with full quality (lossless)
img.save("logo.png", format="PNG")

# Save with compression optimization (still lossless, smaller file)
img.save("logo.png", format="PNG", optimize=True)

# Save with specific compression level (0=none, 9=max compression)
img.save("logo.png", format="PNG", compress_level=9)
```

### Handling Alpha Channel Correctly

```python
def verify_transparency(image_path):
    """Verify that an image has proper transparency."""
    img = Image.open(image_path)

    if img.mode != "RGBA":
        print(f"Warning: Image mode is {img.mode}, not RGBA")
        return False

    alpha = np.array(img)[:, :, 3]
    has_transparent = np.any(alpha == 0)
    has_opaque = np.any(alpha == 255)
    has_partial = np.any((alpha > 0) & (alpha < 255))

    print(f"Transparent pixels: {np.sum(alpha == 0)}")
    print(f"Opaque pixels: {np.sum(alpha == 255)}")
    print(f"Semi-transparent pixels: {np.sum((alpha > 0) & (alpha < 255))}")

    return has_transparent and has_opaque


def composite_on_background(image, bg_color=(255, 255, 255)):
    """Preview transparent image on a solid background."""
    bg = Image.new("RGBA", image.size, bg_color + (255,))
    return Image.alpha_composite(bg, image)
```

### WebP as an Alternative

For web delivery, WebP offers better compression with transparency:

```python
# Save as WebP with transparency (lossy, much smaller file size)
img.save("logo.webp", format="WEBP", quality=90, lossless=False)

# Lossless WebP (smaller than PNG in most cases)
img.save("logo.webp", format="WEBP", lossless=True)
```

---

## 7. Generating with Transparent Background Directly

Some models and services can produce transparent backgrounds without post-processing.

### Models with Native Transparency Support

**Recraft V3/V4**:
- Supports transparent background as a generation parameter.
- API parameter: `"background": "transparent"` in the style config.
- Produces actual alpha channel, not white background.

**Flux (via fal.ai)**:
- The `fal-ai/flux-pro` endpoint supports an `image_format` parameter.
- Some wrappers support transparent background generation.

**DALL-E 3 / GPT Image 1.5 (OpenAI)**:
- `gpt-image-1` supports `background="transparent"` parameter directly.
- Returns PNG with proper alpha channel.

```python
# OpenAI example
from openai import OpenAI
client = OpenAI()

result = client.images.generate(
    model="gpt-image-1",
    prompt="A minimalist geometric logo for a tech company called 'Nexus'",
    size="1024x1024",
    background="transparent",
    output_format="png"
)
```

**Recraft example**:
```python
import requests

response = requests.post(
    "https://external.api.recraft.ai/v1/images/generations",
    headers={"Authorization": f"Bearer {api_key}"},
    json={
        "prompt": "minimalist logo for tech startup",
        "style": "icon",
        "background": "transparent",
        "size": "1024x1024"
    }
)
```

### Limitations of Native Transparency

- Not all models support it; many still output RGB (no alpha).
- Quality varies -- some models produce rough edges at the transparency boundary.
- Prompting for "transparent background" or "no background" without API support often produces a checkerboard pattern or grey background rather than actual transparency.
- Even with native support, post-processing cleanup may still be beneficial.

### Recommended Strategy

1. **If the model supports native transparency**: Use it, then apply light cleanup (edge smoothing).
2. **If the model does not support transparency**: Generate on a solid white background (easiest to remove), then apply rembg or threshold-based removal.
3. **For maximum quality**: Generate at 2x resolution, remove background, then downscale. This produces cleaner edges.

---

## 8. Full Background Removal Pipeline

### Complete Pipeline Implementation

```python
"""
Full background removal pipeline for logo generation.

Dependencies:
    pip install rembg[gpu] Pillow numpy scipy
"""

from PIL import Image, ImageFilter
import numpy as np
from pathlib import Path
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RemovalMethod(Enum):
    THRESHOLD_WHITE = "threshold_white"
    THRESHOLD_BLACK = "threshold_black"
    REMBG_U2NET = "rembg_u2net"
    REMBG_ISNET = "rembg_isnet"
    REMBG_BIREFNET = "rembg_birefnet"
    NATIVE_TRANSPARENT = "native_transparent"


class BackgroundRemover:
    """Automated background removal pipeline for logos."""

    def __init__(self, default_method: RemovalMethod = RemovalMethod.REMBG_U2NET):
        self.default_method = default_method
        self._rembg_sessions = {}

    def _get_rembg_session(self, model_name: str):
        """Lazy-load and cache rembg sessions."""
        if model_name not in self._rembg_sessions:
            from rembg import new_session
            self._rembg_sessions[model_name] = new_session(model_name)
        return self._rembg_sessions[model_name]

    def remove_background(
        self,
        image: Image.Image,
        method: Optional[RemovalMethod] = None,
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        threshold: int = 240,
    ) -> Image.Image:
        """Remove background from a logo image.

        Args:
            image: Input PIL Image.
            method: Removal method. Uses default if None.
            alpha_matting: Enable alpha matting (rembg methods only).
            alpha_matting_foreground_threshold: Foreground threshold for matting.
            alpha_matting_background_threshold: Background threshold for matting.
            alpha_matting_erode_size: Erosion size for matting trimap.
            threshold: Color threshold for threshold-based methods.

        Returns:
            PIL Image with transparent background (RGBA).
        """
        method = method or self.default_method
        image = image.convert("RGBA")

        if method == RemovalMethod.THRESHOLD_WHITE:
            return self._threshold_remove(image, (255, 255, 255), threshold)
        elif method == RemovalMethod.THRESHOLD_BLACK:
            return self._threshold_remove(image, (0, 0, 0), threshold)
        elif method == RemovalMethod.NATIVE_TRANSPARENT:
            # Already transparent, just return
            return image
        else:
            return self._rembg_remove(
                image, method,
                alpha_matting=alpha_matting,
                fg_threshold=alpha_matting_foreground_threshold,
                bg_threshold=alpha_matting_background_threshold,
                erode_size=alpha_matting_erode_size,
            )

    def _threshold_remove(self, image: Image.Image, bg_color: tuple,
                          threshold: int) -> Image.Image:
        """Threshold-based background removal with gradient edges."""
        data = np.array(image, dtype=np.float32)
        bg = np.array(bg_color, dtype=np.float32)

        distance = np.sqrt(np.sum((data[:, :, :3] - bg) ** 2, axis=2))

        # Hard threshold with slight gradient at boundary
        hard_thresh = 255 - threshold if bg_color == (0, 0, 0) else threshold
        gradient_width = 15  # pixels of gradient transition

        if bg_color == (0, 0, 0):
            # For black: low distance = background
            alpha = np.clip(
                (distance - (255 - hard_thresh)) / gradient_width * 255,
                0, 255
            ).astype(np.uint8)
        else:
            # For white: low distance = background
            max_bg_dist = np.sqrt(3 * ((255 - threshold) ** 2))
            alpha = np.clip(
                distance / max(max_bg_dist, 1) * 255,
                0, 255
            ).astype(np.uint8)

        result = np.array(image)
        # Only modify alpha for near-background pixels
        near_bg = distance < (max_bg_dist * 1.5 if bg_color != (0, 0, 0)
                              else 255 - hard_thresh + gradient_width)
        result[near_bg, 3] = np.minimum(result[near_bg, 3], alpha[near_bg])

        return Image.fromarray(result)

    def _rembg_remove(self, image: Image.Image, method: RemovalMethod,
                      alpha_matting: bool, fg_threshold: int,
                      bg_threshold: int, erode_size: int) -> Image.Image:
        """Neural network background removal via rembg."""
        from rembg import remove

        model_map = {
            RemovalMethod.REMBG_U2NET: "u2net",
            RemovalMethod.REMBG_ISNET: "isnet-general-use",
            RemovalMethod.REMBG_BIREFNET: "birefnet-general",
        }
        model_name = model_map[method]
        session = self._get_rembg_session(model_name)

        result = remove(
            image,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=fg_threshold,
            alpha_matting_background_threshold=bg_threshold,
            alpha_matting_erode_size=erode_size,
        )
        return result

    def cleanup(
        self,
        image: Image.Image,
        defringe: bool = True,
        feather_radius: float = 0.5,
        remove_small_artifacts: bool = True,
        min_artifact_size: int = 50,
    ) -> Image.Image:
        """Post-removal cleanup: defringing, feathering, artifact removal.

        Args:
            image: RGBA image with transparent background.
            defringe: Remove color fringe at edges.
            feather_radius: Gaussian blur radius for edge feathering.
            remove_small_artifacts: Remove small disconnected opaque regions.
            min_artifact_size: Minimum pixel count to keep a connected component.
        """
        data = np.array(image)

        if defringe:
            data = self._defringe(data)

        if remove_small_artifacts:
            data = self._remove_artifacts(data, min_artifact_size)

        result = Image.fromarray(data)

        if feather_radius > 0:
            result = self._feather(result, feather_radius)

        return result

    def _defringe(self, data: np.ndarray) -> np.ndarray:
        """Remove color fringe from edges."""
        alpha = data[:, :, 3]
        edge_mask = (alpha > 0) & (alpha < 245)

        if not np.any(edge_mask):
            return data

        from scipy.ndimage import uniform_filter

        opaque_mask = alpha > 200
        for c in range(3):
            channel = data[:, :, c].astype(np.float32)
            weight = opaque_mask.astype(np.float32)
            smoothed = uniform_filter(channel * weight, size=5)
            weight_smoothed = uniform_filter(weight, size=5)
            weight_smoothed = np.maximum(weight_smoothed, 1e-6)
            filled = smoothed / weight_smoothed

            data[edge_mask, c] = np.clip(filled[edge_mask], 0, 255).astype(np.uint8)

        return data

    def _remove_artifacts(self, data: np.ndarray,
                          min_size: int) -> np.ndarray:
        """Remove small disconnected opaque regions."""
        from scipy.ndimage import label

        mask = data[:, :, 3] > 128
        labeled, num_features = label(mask)

        if num_features <= 1:
            return data

        # Find the largest component (the logo)
        component_sizes = np.bincount(labeled.ravel())
        component_sizes[0] = 0  # ignore background

        # Keep only components above min_size
        for i in range(1, num_features + 1):
            if component_sizes[i] < min_size:
                data[labeled == i, 3] = 0

        return data

    def _feather(self, image: Image.Image, radius: float) -> Image.Image:
        """Light feathering of alpha edges."""
        data = np.array(image)
        alpha = Image.fromarray(data[:, :, 3])
        alpha_blurred = alpha.filter(ImageFilter.GaussianBlur(radius=radius))

        orig = np.array(alpha)
        blurred = np.array(alpha_blurred)

        # Only feather edge pixels, preserve solid interior
        edge = (orig > 0) & (orig < 255)
        near_edge = np.zeros_like(edge)

        from scipy.ndimage import binary_dilation
        near_edge = binary_dilation(edge, iterations=max(1, int(radius)))

        result_alpha = orig.copy()
        result_alpha[near_edge] = np.minimum(orig[near_edge], blurred[near_edge])
        # Do not add transparency to fully opaque interior pixels
        result_alpha[orig == 255] = 255

        data[:, :, 3] = result_alpha
        return Image.fromarray(data)

    def process(
        self,
        image: Image.Image,
        method: Optional[RemovalMethod] = None,
        cleanup: bool = True,
        **kwargs,
    ) -> Image.Image:
        """Full pipeline: remove background + cleanup.

        Args:
            image: Input image.
            method: Background removal method.
            cleanup: Whether to run post-processing cleanup.
            **kwargs: Passed to remove_background() and cleanup().
        """
        # Separate kwargs for each step
        removal_keys = {
            "alpha_matting", "alpha_matting_foreground_threshold",
            "alpha_matting_background_threshold", "alpha_matting_erode_size",
            "threshold"
        }
        cleanup_keys = {
            "defringe", "feather_radius", "remove_small_artifacts",
            "min_artifact_size"
        }

        removal_kwargs = {k: v for k, v in kwargs.items() if k in removal_keys}
        cleanup_kwargs = {k: v for k, v in kwargs.items() if k in cleanup_keys}

        result = self.remove_background(image, method=method, **removal_kwargs)

        if cleanup:
            result = self.cleanup(result, **cleanup_kwargs)

        return result


def auto_detect_method(image: Image.Image) -> RemovalMethod:
    """Heuristic to pick the best removal method based on image properties."""
    data = np.array(image.convert("RGB"))
    h, w = data.shape[:2]

    # Sample border pixels (top/bottom 5 rows, left/right 5 cols)
    border_size = 5
    border = np.concatenate([
        data[:border_size, :].reshape(-1, 3),
        data[-border_size:, :].reshape(-1, 3),
        data[:, :border_size].reshape(-1, 3),
        data[:, -border_size:].reshape(-1, 3),
    ])

    mean_color = border.mean(axis=0)
    color_std = border.std(axis=0).mean()

    # If border is nearly uniform and white
    if color_std < 10 and mean_color.min() > 235:
        logger.info("Detected white background, using threshold method")
        return RemovalMethod.THRESHOLD_WHITE

    # If border is nearly uniform and black
    if color_std < 10 and mean_color.max() < 20:
        logger.info("Detected black background, using threshold method")
        return RemovalMethod.THRESHOLD_BLACK

    # Otherwise use neural network
    logger.info("Complex background detected, using BiRefNet")
    return RemovalMethod.REMBG_BIREFNET


# --- Convenience functions ---

def remove_logo_background(
    input_path: str,
    output_path: str,
    method: str = "auto",
    quality: str = "high",
) -> str:
    """One-call function for the full pipeline.

    Args:
        input_path: Path to input image.
        output_path: Path for output transparent PNG.
        method: "auto", "threshold_white", "threshold_black",
                "u2net", "isnet", "birefnet".
        quality: "draft" (fast), "standard", "high" (best quality).

    Returns:
        Output path.
    """
    image = Image.open(input_path)

    remover = BackgroundRemover()

    if method == "auto":
        removal_method = auto_detect_method(image)
    else:
        method_map = {
            "threshold_white": RemovalMethod.THRESHOLD_WHITE,
            "threshold_black": RemovalMethod.THRESHOLD_BLACK,
            "u2net": RemovalMethod.REMBG_U2NET,
            "isnet": RemovalMethod.REMBG_ISNET,
            "birefnet": RemovalMethod.REMBG_BIREFNET,
        }
        removal_method = method_map[method]

    quality_settings = {
        "draft": {
            "alpha_matting": False,
            "defringe": False,
            "feather_radius": 0,
            "remove_small_artifacts": True,
        },
        "standard": {
            "alpha_matting": False,
            "defringe": True,
            "feather_radius": 0.5,
            "remove_small_artifacts": True,
        },
        "high": {
            "alpha_matting": True,
            "defringe": True,
            "feather_radius": 0.5,
            "remove_small_artifacts": True,
            "min_artifact_size": 30,
        },
    }

    settings = quality_settings.get(quality, quality_settings["standard"])
    result = remover.process(image, method=removal_method, **settings)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, format="PNG", optimize=True)
    logger.info(f"Saved transparent logo to {output_path}")

    return output_path


# --- Batch processing ---

def batch_remove_backgrounds(
    input_dir: str,
    output_dir: str,
    method: str = "auto",
    quality: str = "standard",
    extensions: tuple = (".png", ".jpg", ".jpeg", ".webp"),
) -> list[str]:
    """Process all images in a directory.

    Returns list of output paths.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for file in sorted(input_path.iterdir()):
        if file.suffix.lower() in extensions:
            out_file = output_path / f"{file.stem}.png"
            try:
                remove_logo_background(
                    str(file), str(out_file),
                    method=method, quality=quality,
                )
                results.append(str(out_file))
                logger.info(f"Processed: {file.name}")
            except Exception as e:
                logger.error(f"Failed: {file.name}: {e}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python bg_remove.py <input> <output> [method] [quality]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    method = sys.argv[3] if len(sys.argv) > 3 else "auto"
    quality = sys.argv[4] if len(sys.argv) > 4 else "high"

    remove_logo_background(input_file, output_file, method, quality)
```

---

## 9. Integration into an Automated Pipeline

### Pipeline Architecture

```
[Generate Logo] --> [Background Removal] --> [Cleanup] --> [Format & Save]
       |                    |                     |              |
  Model API call     Auto-detect method    Defringe, feather   PNG/WebP/SVG
  or local model     Threshold or rembg    Remove artifacts    Multiple sizes
```

### Example Integration

```python
class LogoGenerationPipeline:
    def __init__(self, generation_api, bg_remover=None):
        self.api = generation_api
        self.bg_remover = bg_remover or BackgroundRemover()

    async def generate_logo(self, prompt, transparent=True, **gen_kwargs):
        """Generate a logo with optional background removal."""

        # Step 1: Generate
        raw_image = await self.api.generate(prompt, **gen_kwargs)

        if not transparent:
            return raw_image

        # Step 2: Check if already transparent
        if self._has_transparency(raw_image):
            # Light cleanup only
            return self.bg_remover.cleanup(raw_image, feather_radius=0.3)

        # Step 3: Remove background
        method = auto_detect_method(raw_image)
        result = self.bg_remover.process(raw_image, method=method)

        return result

    def _has_transparency(self, image):
        if image.mode != "RGBA":
            return False
        alpha = np.array(image)[:, :, 3]
        transparent_ratio = np.sum(alpha == 0) / alpha.size
        return transparent_ratio > 0.05  # >5% transparent pixels

    def save_variants(self, image, base_path, sizes=None):
        """Save multiple size variants of the transparent logo."""
        sizes = sizes or [
            ("original", None),
            ("512", (512, 512)),
            ("256", (256, 256)),
            ("128", (128, 128)),
            ("64", (64, 64)),
            ("favicon", (32, 32)),
        ]

        base = Path(base_path)
        base.mkdir(parents=True, exist_ok=True)
        paths = {}

        for name, size in sizes:
            if size is None:
                resized = image
            else:
                resized = image.resize(size, Image.LANCZOS)

            png_path = base / f"logo_{name}.png"
            resized.save(str(png_path), format="PNG", optimize=True)
            paths[name] = str(png_path)

            # Also save WebP for web use
            if name != "favicon":
                webp_path = base / f"logo_{name}.webp"
                resized.save(str(webp_path), format="WEBP",
                             quality=95, lossless=False)

        return paths
```

---

## 10. Performance Benchmarks and Practical Notes

### Approximate Inference Times (per image, 1024x1024)

| Method | CPU | GPU (RTX 3090) |
|---|---|---|
| Threshold-based | ~2 ms | N/A |
| u2netp (lightweight) | ~200 ms | ~30 ms |
| u2net | ~800 ms | ~80 ms |
| isnet-general-use | ~1.0 s | ~100 ms |
| birefnet-general | ~2.5 s | ~200 ms |
| SAM (ViT-H) | ~5 s | ~500 ms |

### Memory Requirements

| Model | RAM/VRAM |
|---|---|
| u2netp | ~200 MB |
| u2net | ~1 GB |
| isnet-general-use | ~1 GB |
| birefnet-general | ~2 GB |
| SAM ViT-H | ~8 GB |

### Key Practical Notes

1. **Always generate on white backgrounds** when the model does not support native transparency. White is easiest to remove cleanly and does not bleed color into edges.

2. **Generate at 2x resolution** and downscale after background removal. This produces much cleaner edges than removing the background at the target resolution.

3. **Cache rembg sessions**. Model loading takes several seconds; reuse sessions across multiple images in batch processing.

4. **Verify output quality** programmatically by checking that: (a) >10% of pixels are transparent, (b) the largest connected opaque component is a reasonable size, (c) no color fringe remains.

5. **Alpha matting is not always an improvement**. For logos with thin strokes or fine text, it can erode important details. Test with and without on representative samples.

6. **BiRefNet is the current best** for general-purpose quality but U2-Net is often sufficient for logos on simple backgrounds and runs 3-5x faster.

7. **For production pipelines**, implement fallback logic: try threshold first (if uniform background detected), fall back to U2-Net, escalate to BiRefNet if quality check fails.
