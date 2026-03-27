# Post-Processing and Refinement for AI-Generated Logos

## Overview

AI-generated logos (from Stable Diffusion, DALL-E, Midjourney, etc.) typically require
post-processing to reach production quality. Common issues include: soft/blurry edges,
color banding, noise artifacts, limited resolution, and raster-only output. This document
covers a complete post-processing pipeline from raw generation to export-ready assets.

---

## 1. Upscaling with Real-ESRGAN

### Why Real-ESRGAN

Most AI image generators output at 512x512 or 1024x1024. Logos need to be sharp at
large print sizes (3000px+). Real-ESRGAN uses Enhanced Super-Resolution GANs to upscale
images 2-4x while reconstructing detail, far exceeding bicubic interpolation.

### Installation

```bash
# Option A: pip package (simpler, Python >= 3.10)
pip install py-real-esrgan

# Option B: Full repo (more control, Python >= 3.7, PyTorch >= 1.7)
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install basicsr facexlib gfpgan
pip install -r requirements.txt
python setup.py develop
```

### Available Models

| Model | Best For | Notes |
|-------|----------|-------|
| `RealESRGAN_x4plus` | Photos, general images | Primary 4x model, best overall quality |
| `RealESRGAN_x2plus` | Moderate upscaling | 2x variant, faster |
| `RealESRGAN_x4plus_anime_6B` | Anime, illustrations, flat-color logos | Smaller model, sharper edges on flat art |
| `realesr-general-x4v3` | Compact general-purpose | Has denoising strength control |
| `realesr-animevideov3` | Animation/video frames | Temporal consistency |

**For logos**: Use `RealESRGAN_x4plus_anime_6B` for flat/illustrative logos (cleaner edges,
less hallucinated texture). Use `RealESRGAN_x4plus` for photorealistic or complex logos.

### Python API Usage

```python
import torch
from PIL import Image
from py_real_esrgan.model import RealESRGAN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# For illustrative/flat logos - anime model preserves clean edges
model = RealESRGAN(device, scale=4)
model.load_weights('weights/RealESRGAN_x4.pth', download=True)

image = Image.open('logo_raw.png').convert('RGB')
sr_image = model.predict(image)
sr_image.save('logo_upscaled.png')
```

### CLI Usage

```bash
# General upscale
python inference_realesrgan.py -n RealESRGAN_x4plus -i logo_raw.png -o output/ --outscale 4

# Anime/illustration model (better for flat logos)
python inference_realesrgan.py -n RealESRGAN_x4plus_anime_6B -i logo_raw.png -o output/

# Handle large images with tiling (saves VRAM)
python inference_realesrgan.py -n RealESRGAN_x4plus -i logo_raw.png --tile 256
```

### Tips

- Real-ESRGAN supports alpha channels natively -- important for logos with transparency.
- Use `--tile` option to process large images on GPUs with limited VRAM.
- For a 512x512 input with 4x scale, output is 2048x2048. Chain two 2x passes for
  higher quality than a single 4x pass.

---

## 2. Image Cleanup: Denoising and Artifact Removal

AI-generated images often contain subtle noise, color banding, and compression artifacts.

### OpenCV Non-Local Means Denoising

```python
import cv2
import numpy as np

def denoise_logo(image_path: str, output_path: str, strength: int = 10) -> None:
    """
    Denoise an image using Non-Local Means algorithm.

    Args:
        strength: Filter strength. Higher removes more noise but may blur.
                  For logos, use 5-15 (lower than photos to preserve edges).
    """
    img = cv2.imread(image_path)

    # For color images: fastNlMeansDenoisingColored
    # h=strength for luminance, hForColorComponents=strength for color
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=strength,              # luminance noise filter strength
        hForColorComponents=strength,  # color noise filter strength
        templateWindowSize=7,     # patch size (odd number)
        searchWindowSize=21       # search area (odd number)
    )

    cv2.imwrite(output_path, denoised)
```

### Bilateral Filter (Edge-Preserving)

Better for logos because it smooths flat areas while preserving sharp edges:

```python
def bilateral_denoise(image_path: str, output_path: str) -> None:
    """Edge-preserving denoising ideal for logos with flat color regions."""
    img = cv2.imread(image_path)

    # d=9: diameter of pixel neighborhood
    # sigmaColor=75: filter sigma in color space
    # sigmaSpace=75: filter sigma in coordinate space
    denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    cv2.imwrite(output_path, denoised)
```

### Median Filter for Speckle Removal

```python
def remove_speckle(image_path: str, output_path: str, ksize: int = 3) -> None:
    """Remove salt-and-pepper noise / speckles from logo."""
    img = cv2.imread(image_path)
    cleaned = cv2.medianBlur(img, ksize)  # ksize must be odd: 3, 5, 7
    cv2.imwrite(output_path, cleaned)
```

### Pillow-Based Simple Denoising

```python
from PIL import Image, ImageFilter

def pillow_smooth(image_path: str, output_path: str) -> None:
    """Light smoothing to reduce minor artifacts."""
    img = Image.open(image_path)
    # SMOOTH_MORE for stronger effect, SMOOTH for lighter
    smoothed = img.filter(ImageFilter.SMOOTH)
    smoothed.save(output_path)
```

---

## 3. Edge Refinement and Sharpening

### Unsharp Mask (Recommended for Logos)

The classic approach: subtract a blurred version from the original to enhance edges.

```python
import cv2
import numpy as np
from scipy.ndimage import median_filter

def unsharp_mask_opencv(image_path: str, output_path: str,
                         sigma: float = 1.0, strength: float = 1.5) -> None:
    """
    Apply unsharp mask sharpening.

    Formula: sharpened = original + strength * (original - blurred)

    Args:
        sigma: Gaussian blur radius. Larger = broader edge enhancement.
        strength: Sharpening intensity. 0.5-2.0 typical for logos.
    """
    img = cv2.imread(image_path).astype(np.float64)

    # Create blurred version
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)

    # Unsharp mask formula
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)

    # Clip to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, sharpened)


def unsharp_mask_pillow(image_path: str, output_path: str,
                         radius: int = 2, percent: int = 150,
                         threshold: int = 3) -> None:
    """
    Pillow's built-in unsharp mask.

    Args:
        radius: Size of blur kernel.
        percent: Strength of sharpening (100 = normal, 150 = strong).
        threshold: Minimum brightness change to sharpen (reduces noise sharpening).
    """
    img = Image.open(image_path)
    sharpened = img.filter(ImageFilter.UnsharpMask(
        radius=radius, percent=percent, threshold=threshold
    ))
    sharpened.save(output_path)
```

### Laplacian Edge Enhancement

```python
def laplacian_sharpen(image_path: str, output_path: str,
                       strength: float = 0.7) -> None:
    """
    Sharpen using Laplacian edge detection.
    Formula: sharp = image - strength * Laplacian(image)
    """
    img = cv2.imread(image_path)

    # Apply per-channel
    result = np.zeros_like(img, dtype=np.float64)
    for i in range(3):
        channel = img[:, :, i].astype(np.float64)
        # Median filter to reduce noise before edge detection
        channel_mf = median_filter(channel, size=1)
        lap = cv2.Laplacian(channel_mf, cv2.CV_64F)
        sharp = channel - strength * lap
        sharp[sharp > 255] = 255
        sharp[sharp < 0] = 0
        result[:, :, i] = sharp

    cv2.imwrite(output_path, result.astype(np.uint8))
```

### Custom Sharpening Kernel

```python
def kernel_sharpen(image_path: str, output_path: str,
                    mode: str = "moderate") -> None:
    """Apply convolution-based sharpening with preset kernels."""
    img = cv2.imread(image_path)

    kernels = {
        "light": np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ]),
        "moderate": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]),
        "strong": np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ]),
    }

    sharpened = cv2.filter2D(img, -1, kernels[mode])
    cv2.imwrite(output_path, sharpened)
```

---

## 4. Color Correction and Normalization

### Pillow Color Enhancement

```python
from PIL import Image, ImageEnhance, ImageOps

def enhance_colors(image_path: str, output_path: str,
                    color_factor: float = 1.2,
                    contrast_factor: float = 1.3,
                    brightness_factor: float = 1.0,
                    sharpness_factor: float = 1.5) -> None:
    """
    Multi-step Pillow enhancement.

    Factor guide (for all enhancers):
        0.0 = degenerate (gray/blurred/black)
        1.0 = original image (no change)
        >1.0 = enhance (more color/contrast/brightness/sharpness)
    """
    img = Image.open(image_path)

    # Step 1: Color saturation
    img = ImageEnhance.Color(img).enhance(color_factor)

    # Step 2: Contrast
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Step 3: Brightness
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Step 4: Sharpness
    img = ImageEnhance.Sharpness(img).enhance(sharpness_factor)

    img.save(output_path)
```

### Histogram Equalization (OpenCV)

```python
def equalize_histogram(image_path: str, output_path: str) -> None:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Better than global histogram equalization for logos.
    """
    img = cv2.imread(image_path)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel only (luminance)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge and convert back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    cv2.imwrite(output_path, result)
```

### White Balance / Gray World Correction

```python
def gray_world_correction(image_path: str, output_path: str) -> None:
    """
    Apply gray world assumption white balance correction.
    Useful when AI generation introduces color casts.
    """
    img = cv2.imread(image_path).astype(np.float64)

    # Calculate mean of each channel
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    # Scale each channel
    img[:, :, 0] *= avg_gray / avg_b
    img[:, :, 1] *= avg_gray / avg_g
    img[:, :, 2] *= avg_gray / avg_r

    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, img)
```

### Normalize to Target Color Palette

```python
from PIL import Image
import numpy as np
from typing import List, Tuple

def quantize_to_palette(image_path: str, output_path: str,
                         colors: int = 8) -> None:
    """
    Reduce logo to a limited color palette.
    Useful for cleaning up AI-generated color noise.
    """
    img = Image.open(image_path).convert('RGB')

    # Pillow's built-in quantization
    quantized = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

    # Convert back to RGB
    result = quantized.convert('RGB')
    result.save(output_path)


def snap_to_brand_colors(image_path: str, output_path: str,
                          brand_colors: List[Tuple[int, int, int]]) -> None:
    """
    Snap each pixel to the nearest brand color.
    Great for ensuring AI output uses exact brand colors.
    """
    img = Image.open(image_path).convert('RGB')
    pixels = np.array(img, dtype=np.float64)

    brand = np.array(brand_colors, dtype=np.float64)

    # For each pixel, find nearest brand color (Euclidean distance)
    h, w, _ = pixels.shape
    flat = pixels.reshape(-1, 3)

    # Vectorized distance calculation
    distances = np.linalg.norm(
        flat[:, np.newaxis, :] - brand[np.newaxis, :, :], axis=2
    )
    nearest_idx = np.argmin(distances, axis=1)
    result = brand[nearest_idx].reshape(h, w, 3).astype(np.uint8)

    Image.fromarray(result).save(output_path)
```

---

## 5. Multi-Stage Refinement Pipeline

### Complete Pipeline Design

```python
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the logo post-processing pipeline."""
    # Upscaling
    upscale_factor: int = 4
    upscale_model: str = "RealESRGAN_x4plus_anime_6B"

    # Denoising
    denoise_strength: int = 8
    use_bilateral: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75

    # Sharpening
    sharpen_method: str = "unsharp"  # "unsharp", "laplacian", "kernel"
    unsharp_radius: int = 2
    unsharp_percent: int = 150
    unsharp_threshold: int = 3

    # Color enhancement
    color_factor: float = 1.1
    contrast_factor: float = 1.2
    brightness_factor: float = 1.0
    sharpness_factor: float = 1.3

    # Color quantization
    quantize_colors: Optional[int] = None  # None = skip, 8-16 typical
    brand_colors: Optional[List[Tuple[int, int, int]]] = None

    # Background removal
    remove_background: bool = True

    # Output
    output_sizes: List[int] = field(default_factory=lambda: [
        4096, 2048, 1024, 512, 256, 128, 64, 32, 16
    ])
    output_formats: List[str] = field(default_factory=lambda: [
        "png", "ico", "svg"
    ])


class LogoPostProcessor:
    """Multi-stage pipeline for refining AI-generated logos."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._esrgan_model = None

    def _load_upscaler(self):
        """Lazy-load Real-ESRGAN model."""
        if self._esrgan_model is None:
            import torch
            from py_real_esrgan.model import RealESRGAN

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._esrgan_model = RealESRGAN(device, scale=self.config.upscale_factor)
            self._esrgan_model.load_weights(
                f'weights/{self.config.upscale_model}.pth',
                download=True
            )
        return self._esrgan_model

    def step_1_upscale(self, img: Image.Image) -> Image.Image:
        """Stage 1: AI upscaling for higher resolution."""
        model = self._load_upscaler()
        rgb = img.convert('RGB')
        upscaled = model.predict(rgb)

        # Restore alpha channel if present
        if img.mode == 'RGBA':
            alpha = img.getchannel('A')
            alpha_upscaled = alpha.resize(upscaled.size, Image.LANCZOS)
            upscaled = upscaled.convert('RGBA')
            upscaled.putalpha(alpha_upscaled)

        return upscaled

    def step_2_denoise(self, img: Image.Image) -> Image.Image:
        """Stage 2: Remove noise and artifacts."""
        arr = np.array(img.convert('RGB'))
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if self.config.use_bilateral:
            denoised = cv2.bilateralFilter(
                arr_bgr,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color,
                sigmaSpace=self.config.bilateral_sigma_space
            )
        else:
            denoised = cv2.fastNlMeansDenoisingColored(
                arr_bgr, None,
                h=self.config.denoise_strength,
                hForColorComponents=self.config.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )

        arr_rgb = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(arr_rgb)

        # Preserve alpha
        if img.mode == 'RGBA':
            result = result.convert('RGBA')
            result.putalpha(img.getchannel('A'))

        return result

    def step_3_sharpen(self, img: Image.Image) -> Image.Image:
        """Stage 3: Edge refinement and sharpening."""
        if self.config.sharpen_method == "unsharp":
            return img.filter(ImageFilter.UnsharpMask(
                radius=self.config.unsharp_radius,
                percent=self.config.unsharp_percent,
                threshold=self.config.unsharp_threshold
            ))
        elif self.config.sharpen_method == "kernel":
            kernel = np.array([
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ])
            arr = np.array(img.convert('RGB'))
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            sharpened = cv2.filter2D(arr_bgr, -1, kernel)
            arr_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(arr_rgb)
            if img.mode == 'RGBA':
                result = result.convert('RGBA')
                result.putalpha(img.getchannel('A'))
            return result

        return img

    def step_4_color_correct(self, img: Image.Image) -> Image.Image:
        """Stage 4: Color correction and enhancement."""
        # Apply Pillow enhancements (works on RGB, preserves alpha separately)
        alpha = None
        if img.mode == 'RGBA':
            alpha = img.getchannel('A')
            img = img.convert('RGB')

        img = ImageEnhance.Color(img).enhance(self.config.color_factor)
        img = ImageEnhance.Contrast(img).enhance(self.config.contrast_factor)
        img = ImageEnhance.Brightness(img).enhance(self.config.brightness_factor)
        img = ImageEnhance.Sharpness(img).enhance(self.config.sharpness_factor)

        if alpha is not None:
            img = img.convert('RGBA')
            img.putalpha(alpha)

        return img

    def step_5_quantize(self, img: Image.Image) -> Image.Image:
        """Stage 5: Optional color palette reduction."""
        if self.config.brand_colors:
            pixels = np.array(img.convert('RGB'), dtype=np.float64)
            brand = np.array(self.config.brand_colors, dtype=np.float64)
            h, w, _ = pixels.shape
            flat = pixels.reshape(-1, 3)
            distances = np.linalg.norm(
                flat[:, np.newaxis, :] - brand[np.newaxis, :, :], axis=2
            )
            nearest_idx = np.argmin(distances, axis=1)
            result_arr = brand[nearest_idx].reshape(h, w, 3).astype(np.uint8)
            result = Image.fromarray(result_arr)
            if img.mode == 'RGBA':
                result = result.convert('RGBA')
                result.putalpha(img.getchannel('A'))
            return result

        if self.config.quantize_colors:
            rgb = img.convert('RGB')
            quantized = rgb.quantize(
                colors=self.config.quantize_colors,
                method=Image.Quantize.MEDIANCUT
            )
            result = quantized.convert('RGB')
            if img.mode == 'RGBA':
                result = result.convert('RGBA')
                result.putalpha(img.getchannel('A'))
            return result

        return img

    def step_6_remove_background(self, img: Image.Image) -> Image.Image:
        """Stage 6: Optional background removal for transparent output."""
        if not self.config.remove_background:
            return img

        from rembg import remove
        return remove(img)

    def process(self, input_path: str, output_dir: str) -> dict:
        """
        Run the full pipeline and export all variants.

        Returns dict mapping format/size to output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        outputs = {}

        img = Image.open(input_path)

        # Run pipeline stages
        print("[1/6] Upscaling...")
        img = self.step_1_upscale(img)

        print("[2/6] Denoising...")
        img = self.step_2_denoise(img)

        print("[3/6] Sharpening...")
        img = self.step_3_sharpen(img)

        print("[4/6] Color correction...")
        img = self.step_4_color_correct(img)

        print("[5/6] Color quantization...")
        img = self.step_5_quantize(img)

        print("[6/6] Background removal...")
        img = self.step_6_remove_background(img)

        # Save master high-res version
        master_path = os.path.join(output_dir, "logo_master.png")
        img.save(master_path, "PNG", optimize=True)
        outputs["master"] = master_path

        # Generate sized variants
        for size in self.config.output_sizes:
            resized = img.copy()
            resized.thumbnail((size, size), Image.LANCZOS)
            path = os.path.join(output_dir, f"logo_{size}x{size}.png")
            resized.save(path, "PNG", optimize=True)
            outputs[f"png_{size}"] = path

        return outputs


# Usage
if __name__ == "__main__":
    config = PipelineConfig(
        upscale_factor=4,
        upscale_model="RealESRGAN_x4plus_anime_6B",
        denoise_strength=8,
        use_bilateral=True,
        sharpen_method="unsharp",
        unsharp_percent=150,
        color_factor=1.1,
        contrast_factor=1.2,
        remove_background=True,
    )

    processor = LogoPostProcessor(config)
    results = processor.process("raw_logo.png", "output/refined/")

    for key, path in results.items():
        print(f"  {key}: {path}")
```

### Pipeline Stage Order Rationale

1. **Upscale first** -- gives later stages more pixels to work with, sharpening
   and denoising are more effective at higher resolution.
2. **Denoise second** -- removes artifacts introduced by AI generation and any
   artifacts from the upscaling process.
3. **Sharpen third** -- must come after denoising (sharpening amplifies noise).
4. **Color correct fourth** -- adjust on the clean, sharp image.
5. **Quantize fifth** -- snap to brand colors after all other corrections.
6. **Background removal last** -- operates on the fully refined image for best mask quality.

---

## 6. Pillow-Based Image Enhancement

### Quick Reference: ImageEnhance Classes

```python
from PIL import Image, ImageEnhance

img = Image.open("logo.png")

# Color (saturation): 0.0=grayscale, 1.0=original, 2.0=double saturation
ImageEnhance.Color(img).enhance(1.3)

# Contrast: 0.0=solid gray, 1.0=original, 2.0=double contrast
ImageEnhance.Contrast(img).enhance(1.2)

# Brightness: 0.0=black, 1.0=original, 2.0=double brightness
ImageEnhance.Brightness(img).enhance(1.05)

# Sharpness: 0.0=blurred, 1.0=original, 2.0=double sharpness
ImageEnhance.Sharpness(img).enhance(1.5)
```

### ImageFilter Built-In Filters

```python
from PIL import ImageFilter

# Predefined filters
img.filter(ImageFilter.SHARPEN)           # Basic sharpening
img.filter(ImageFilter.DETAIL)            # Enhance detail
img.filter(ImageFilter.EDGE_ENHANCE)      # Light edge enhancement
img.filter(ImageFilter.EDGE_ENHANCE_MORE) # Strong edge enhancement
img.filter(ImageFilter.SMOOTH)            # Light smoothing
img.filter(ImageFilter.SMOOTH_MORE)       # Strong smoothing

# Configurable unsharp mask
img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

# Configurable Gaussian blur (for pipeline use)
img.filter(ImageFilter.GaussianBlur(radius=1))
```

### Recommended Enhancement Presets for Logos

```python
LOGO_PRESETS = {
    "flat_illustration": {
        "color": 1.1,
        "contrast": 1.3,
        "brightness": 1.0,
        "sharpness": 1.8,
    },
    "photorealistic": {
        "color": 1.2,
        "contrast": 1.15,
        "brightness": 1.05,
        "sharpness": 1.3,
    },
    "minimal_text": {
        "color": 1.0,
        "contrast": 1.4,
        "brightness": 1.0,
        "sharpness": 2.0,
    },
    "gradient_rich": {
        "color": 1.15,
        "contrast": 1.1,
        "brightness": 1.0,
        "sharpness": 1.2,
    },
}

def apply_preset(img: Image.Image, preset_name: str) -> Image.Image:
    """Apply a named enhancement preset to a logo."""
    p = LOGO_PRESETS[preset_name]
    img = ImageEnhance.Color(img).enhance(p["color"])
    img = ImageEnhance.Contrast(img).enhance(p["contrast"])
    img = ImageEnhance.Brightness(img).enhance(p["brightness"])
    img = ImageEnhance.Sharpness(img).enhance(p["sharpness"])
    return img
```

---

## 7. Format Conversion

### PNG Export (Optimized)

```python
from PIL import Image

def export_png(img: Image.Image, output_path: str,
               optimize: bool = True) -> None:
    """Export optimized PNG with transparency support."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')
    img.save(output_path, "PNG", optimize=optimize)
```

### ICO (Favicon, Multi-Size)

```python
def export_ico(img: Image.Image, output_path: str,
               sizes: list = None) -> None:
    """
    Export multi-size ICO file.

    Standard favicon sizes: 16, 24, 32, 48, 64, 128, 256
    """
    if sizes is None:
        sizes = [16, 24, 32, 48, 64, 128, 256]

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Pillow can save ICO with multiple sizes embedded
    icon_sizes = [(s, s) for s in sizes]
    img.save(output_path, format="ICO", sizes=icon_sizes)
```

### PDF Export

```python
def export_pdf(img: Image.Image, output_path: str) -> None:
    """Export logo as PDF (raster-based)."""
    rgb = img.convert('RGB')
    rgb.save(output_path, "PDF", resolution=300)
```

### SVG Conversion with VTracer (Raster to Vector)

```python
import vtracer

def export_svg_vtracer(input_png: str, output_svg: str,
                        colormode: str = "color",
                        filter_speckle: int = 4,
                        color_precision: int = 6,
                        corner_threshold: int = 60,
                        segment_length: float = 4.0,
                        splice_threshold: int = 45,
                        mode: str = "spline") -> None:
    """
    Convert raster logo to vector SVG using VTracer.

    Args:
        colormode: "color" for full color, "bw" for black & white.
        filter_speckle: Discard patches smaller than this (px).
                        Higher = cleaner but may lose small details.
        color_precision: Bits per RGB channel (1-8). Lower = fewer colors.
        corner_threshold: Min angle (degrees) to be a corner.
        segment_length: Max curve segment length. Lower = more accurate.
        mode: "pixel" (blocky), "polygon" (angular), "spline" (smooth curves).
    """
    vtracer.convert_image_to_svg_py(
        input_png,
        output_svg,
        colormode=colormode,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        corner_threshold=corner_threshold,
        segment_length=segment_length,
        splice_threshold=splice_threshold,
        mode=mode,
    )
```

### SVG Conversion with Potrace (Black & White)

```python
import subprocess
from PIL import Image

def export_svg_potrace(input_png: str, output_svg: str,
                        threshold: int = 128) -> None:
    """
    Convert logo to SVG via Potrace. Best for monochrome logos.
    Potrace only handles B&W, so the image is binarized first.

    Requires: potrace installed (apt install potrace / brew install potrace)
    """
    # Convert to BMP (Potrace's preferred input)
    img = Image.open(input_png).convert('L')  # Grayscale
    # Binarize
    img = img.point(lambda x: 0 if x < threshold else 255, '1')
    bmp_path = input_png.rsplit('.', 1)[0] + '.bmp'
    img.save(bmp_path)

    # Run potrace
    subprocess.run([
        'potrace', bmp_path,
        '-s',              # SVG output
        '-o', output_svg,
        '--turdsize', '2', # Remove small speckles
        '--alphamax', '1', # Corner smoothness (0=sharp, 1.34=smooth)
    ], check=True)
```

### SVG to PNG with CairoSVG

```python
import cairosvg

def svg_to_png(svg_path: str, png_path: str,
               width: int = None, height: int = None) -> None:
    """Convert SVG to PNG at specified dimensions."""
    cairosvg.svg2png(
        url=svg_path,
        write_to=png_path,
        output_width=width,
        output_height=height,
    )

def svg_to_pdf(svg_path: str, pdf_path: str) -> None:
    """Convert SVG to PDF."""
    cairosvg.svg2pdf(url=svg_path, write_to=pdf_path)
```

---

## 8. Generating Logo Variants

### Dark Mode Variant

```python
from PIL import Image, ImageOps, ImageChops
import numpy as np

def create_dark_mode(img: Image.Image,
                      method: str = "invert") -> Image.Image:
    """
    Create a dark-mode variant of the logo.

    Methods:
        "invert": Simple color inversion (good for monochrome logos).
        "light_on_dark": Make dark elements light, preserve alpha.
        "white_version": Convert all non-transparent pixels to white.
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    if method == "invert":
        r, g, b, a = img.split()
        rgb = Image.merge('RGB', (r, g, b))
        inverted = ImageOps.invert(rgb)
        inverted = inverted.convert('RGBA')
        inverted.putalpha(a)
        return inverted

    elif method == "white_version":
        r, g, b, a = img.split()
        # Create white image
        white = Image.new('RGB', img.size, (255, 255, 255))
        white = white.convert('RGBA')
        white.putalpha(a)
        return white

    elif method == "light_on_dark":
        arr = np.array(img, dtype=np.float64)
        # Invert RGB channels, keep alpha
        arr[:, :, :3] = 255.0 - arr[:, :, :3]
        # Boost brightness slightly for dark backgrounds
        arr[:, :, :3] = np.clip(arr[:, :, :3] * 1.1, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    return img


def create_dark_mode_svg(svg_content: str) -> str:
    """
    Create SVG with embedded dark mode support using CSS media query.
    The browser automatically switches based on user's system preference.
    """
    dark_mode_css = """
    <style>
      @media (prefers-color-scheme: dark) {
        .logo-dark { display: inline; }
        .logo-light { display: none; }
      }
      @media (prefers-color-scheme: light) {
        .logo-dark { display: none; }
        .logo-light { display: inline; }
      }
    </style>
    """
    # Insert CSS after opening <svg> tag
    return svg_content.replace('>', f'>{dark_mode_css}', 1)
```

### Monochrome Variant

```python
def create_monochrome(img: Image.Image,
                       color: Tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """
    Create a single-color monochrome version of the logo.
    Preserves alpha channel for transparency.
    """
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    r, g, b, a = img.split()

    # Create solid color image
    mono = Image.new('RGB', img.size, color)
    mono = mono.convert('RGBA')
    mono.putalpha(a)

    return mono


def create_grayscale(img: Image.Image) -> Image.Image:
    """Convert logo to grayscale while preserving alpha."""
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    r, g, b, a = img.split()
    gray = img.convert('L').convert('RGB').convert('RGBA')
    gray.putalpha(a)
    return gray
```

### Favicon Generation

```python
def generate_favicon_set(img: Image.Image, output_dir: str) -> dict:
    """
    Generate a complete favicon set for web deployment.

    Produces:
        - favicon.ico (16, 32, 48px multi-size)
        - favicon-16x16.png
        - favicon-32x32.png
        - apple-touch-icon.png (180x180)
        - android-chrome-192x192.png
        - android-chrome-512x512.png
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    outputs = {}

    # Multi-size ICO
    ico_path = os.path.join(output_dir, "favicon.ico")
    img.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    outputs["ico"] = ico_path

    # Individual PNGs
    favicon_sizes = {
        "favicon-16x16.png": 16,
        "favicon-32x32.png": 32,
        "apple-touch-icon.png": 180,
        "android-chrome-192x192.png": 192,
        "android-chrome-512x512.png": 512,
    }

    for filename, size in favicon_sizes.items():
        resized = img.copy()
        resized = resized.resize((size, size), Image.LANCZOS)
        path = os.path.join(output_dir, filename)
        resized.save(path, "PNG")
        outputs[filename] = path

    # Generate webmanifest
    manifest = {
        "icons": [
            {"src": "/android-chrome-192x192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "/android-chrome-512x512.png", "sizes": "512x512", "type": "image/png"},
        ]
    }
    import json
    manifest_path = os.path.join(output_dir, "site.webmanifest")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    outputs["webmanifest"] = manifest_path

    # Generate HTML snippet
    html_snippet = """
<!-- Favicon HTML -->
<link rel="icon" type="image/x-icon" href="/favicon.ico">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
<link rel="manifest" href="/site.webmanifest">
""".strip()

    snippet_path = os.path.join(output_dir, "favicon_html.txt")
    with open(snippet_path, 'w') as f:
        f.write(html_snippet)
    outputs["html_snippet"] = snippet_path

    return outputs
```

### Social Media Sized Variants

```python
def generate_social_media_set(img: Image.Image, output_dir: str,
                               bg_color: Tuple[int, int, int] = (255, 255, 255)
                               ) -> dict:
    """
    Generate social media profile and banner images.

    Common sizes:
        - Profile: square, centered logo
        - Open Graph / Link Preview: 1200x630
        - Twitter Card: 1200x628
        - LinkedIn Banner: 1584x396
        - Instagram Post: 1080x1080
        - Facebook Cover: 820x312
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    social_sizes = {
        "og-image.png": (1200, 630),
        "twitter-card.png": (1200, 628),
        "linkedin-banner.png": (1584, 396),
        "instagram-post.png": (1080, 1080),
        "facebook-cover.png": (820, 312),
        "profile-square-400.png": (400, 400),
        "profile-square-200.png": (200, 200),
    }

    outputs = {}

    for filename, (width, height) in social_sizes.items():
        # Create background canvas
        canvas = Image.new('RGB', (width, height), bg_color)

        # Scale logo to fit with padding (60% of smaller dimension)
        max_logo_size = int(min(width, height) * 0.6)
        logo_copy = img.copy()
        logo_copy.thumbnail((max_logo_size, max_logo_size), Image.LANCZOS)

        # Center the logo
        x = (width - logo_copy.width) // 2
        y = (height - logo_copy.height) // 2

        canvas.paste(logo_copy, (x, y), logo_copy)  # alpha as mask

        path = os.path.join(output_dir, filename)
        canvas.save(path, "PNG")
        outputs[filename] = path

    return outputs
```

### Complete Variant Generator

```python
def generate_all_variants(master_logo_path: str, output_dir: str,
                           brand_color: Tuple[int, int, int] = (0, 0, 0),
                           bg_color: Tuple[int, int, int] = (255, 255, 255)
                           ) -> dict:
    """
    Generate all logo variants from a single master logo.

    Output structure:
        output_dir/
            master/
                logo_master.png
            png/
                logo_4096.png ... logo_16.png
            dark/
                logo_dark_inverted.png
                logo_dark_white.png
            mono/
                logo_mono_black.png
                logo_mono_white.png
                logo_grayscale.png
            favicon/
                favicon.ico, apple-touch-icon.png, etc.
            social/
                og-image.png, twitter-card.png, etc.
            vector/
                logo.svg
                logo_mono.svg
            pdf/
                logo.pdf
    """
    import os

    all_outputs = {}
    img = Image.open(master_logo_path).convert('RGBA')

    # Master PNG
    master_dir = os.path.join(output_dir, "master")
    os.makedirs(master_dir, exist_ok=True)
    img.save(os.path.join(master_dir, "logo_master.png"), "PNG", optimize=True)

    # Sized PNGs
    png_dir = os.path.join(output_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
    for size in [4096, 2048, 1024, 512, 256, 128, 64, 32]:
        resized = img.copy()
        resized.thumbnail((size, size), Image.LANCZOS)
        resized.save(os.path.join(png_dir, f"logo_{size}.png"), "PNG", optimize=True)

    # Dark mode variants
    dark_dir = os.path.join(output_dir, "dark")
    os.makedirs(dark_dir, exist_ok=True)
    create_dark_mode(img, "invert").save(
        os.path.join(dark_dir, "logo_dark_inverted.png"))
    create_dark_mode(img, "white_version").save(
        os.path.join(dark_dir, "logo_dark_white.png"))

    # Monochrome
    mono_dir = os.path.join(output_dir, "mono")
    os.makedirs(mono_dir, exist_ok=True)
    create_monochrome(img, (0, 0, 0)).save(
        os.path.join(mono_dir, "logo_mono_black.png"))
    create_monochrome(img, (255, 255, 255)).save(
        os.path.join(mono_dir, "logo_mono_white.png"))
    create_grayscale(img).save(
        os.path.join(mono_dir, "logo_grayscale.png"))

    # Favicons
    favicon_dir = os.path.join(output_dir, "favicon")
    all_outputs["favicons"] = generate_favicon_set(img, favicon_dir)

    # Social media
    social_dir = os.path.join(output_dir, "social")
    all_outputs["social"] = generate_social_media_set(img, social_dir, bg_color)

    # Vector (SVG) -- requires vtracer
    vector_dir = os.path.join(output_dir, "vector")
    os.makedirs(vector_dir, exist_ok=True)
    try:
        temp_png = os.path.join(vector_dir, "_temp.png")
        img.save(temp_png, "PNG")
        export_svg_vtracer(temp_png, os.path.join(vector_dir, "logo.svg"))
        # Mono SVG
        mono_png = os.path.join(vector_dir, "_temp_mono.png")
        create_monochrome(img, (0, 0, 0)).save(mono_png, "PNG")
        export_svg_vtracer(mono_png, os.path.join(vector_dir, "logo_mono.svg"),
                           colormode="bw")
        os.remove(temp_png)
        os.remove(mono_png)
    except ImportError:
        print("Warning: vtracer not installed, skipping SVG export.")

    # PDF
    pdf_dir = os.path.join(output_dir, "pdf")
    os.makedirs(pdf_dir, exist_ok=True)
    export_pdf(img, os.path.join(pdf_dir, "logo.pdf"))

    return all_outputs
```

---

## 9. Dependencies and Installation

### Core Dependencies

```bash
# Image processing fundamentals
pip install Pillow opencv-python numpy scipy

# AI upscaling
pip install py-real-esrgan torch torchvision

# Background removal
pip install rembg

# Raster-to-vector conversion
pip install vtracer

# SVG rendering
pip install cairosvg
```

### Optional Dependencies

```bash
# Potrace for B&W vectorization (system package)
# Ubuntu/Debian: apt install potrace
# macOS: brew install potrace
# Windows: download from https://potrace.sourceforge.net/

# ImageMagick for advanced format conversions
# Ubuntu/Debian: apt install imagemagick
# macOS: brew install imagemagick
```

### Minimal Install (Pillow-Only Pipeline)

If GPU/heavy dependencies are not wanted, a Pillow-only pipeline still provides:
- Denoising (ImageFilter.SMOOTH)
- Sharpening (UnsharpMask, SHARPEN, EDGE_ENHANCE)
- Color correction (ImageEnhance)
- PNG/ICO/PDF export
- Resizing and variant generation

```bash
pip install Pillow numpy
```

---

## 10. Key Recommendations

### Model Selection for Upscaling

| Logo Type | Recommended Model | Reason |
|-----------|-------------------|--------|
| Flat/icon style | `RealESRGAN_x4plus_anime_6B` | Preserves hard edges, avoids texture hallucination |
| Photorealistic | `RealESRGAN_x4plus` | Better detail reconstruction |
| Text-heavy | `RealESRGAN_x4plus_anime_6B` | Cleaner text edge preservation |
| Gradient-rich | `RealESRGAN_x4plus` | Smoother gradient handling |

### Vectorization Strategy

- **Full-color logos**: Use VTracer with `mode="spline"`, `filter_speckle=4`,
  `color_precision=6` for smooth, clean SVG output.
- **Monochrome logos**: Use Potrace for the cleanest, most efficient SVG paths.
- **Complex logos**: Export a high-res PNG master; vectorize a simplified monochrome
  version for scalable use cases.
- Always keep the raster master as the source of truth -- re-vectorize as needed.

### Quality Checklist

1. Upscale to at least 4096x4096 master resolution.
2. Denoise with bilateral filter (preserves edges better than Gaussian).
3. Apply moderate unsharp mask (percent=150, threshold=3).
4. Enhance contrast slightly (1.1-1.3x) for visual punch.
5. Verify colors match brand palette; use snap_to_brand_colors if needed.
6. Remove background for transparent PNG.
7. Export all format variants from the single refined master.
8. Visually inspect at target sizes -- especially 16x16 favicon.

### Performance Notes

- Real-ESRGAN 4x on a 512x512 image: ~1-3 seconds on GPU, ~30-60 seconds on CPU.
- Bilateral filter on 2048x2048: ~0.5 seconds.
- VTracer on a 2048x2048 PNG: ~2-5 seconds.
- Full pipeline for one logo: ~10-30 seconds on GPU, ~2-5 minutes on CPU.
- Use `--tile 256` for Real-ESRGAN on GPUs with less than 4GB VRAM.

---

## Sources

- [Real-ESRGAN GitHub (xinntao)](https://github.com/xinntao/Real-ESRGAN)
- [py-real-esrgan on PyPI](https://pypi.org/project/py-real-esrgan/)
- [Real-ESRGAN Anime Model Documentation](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/anime_model.md)
- [VTracer - Raster to Vector Converter](https://github.com/visioncortex/vtracer)
- [Potrace - Bitmap Tracing](https://potrace.sourceforge.net/)
- [Pillow ImageEnhance Documentation](https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html)
- [OpenCV Image Denoising](https://docs.opencv.org/4.x/d5/d69/tutorial_py_non_local_means.html)
- [Unsharp Masking with Python and OpenCV](https://www.idtools.com.au/unsharp-masking-with-python-and-opencv/)
- [rembg - Background Removal](https://github.com/danielgatis/rembg)
- [CairoSVG](https://cairosvg.org/)
- [Pillow ICO Export](https://pillow.readthedocs.io/en/stable/_modules/PIL/IcoImagePlugin.html)
