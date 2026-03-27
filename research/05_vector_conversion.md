# Raster-to-Vector Conversion for Logos

Research on converting raster logo outputs (PNG/JPG) into clean, scalable SVG vector graphics.

---

## 1. VTracer

### Overview

VTracer (by VisionCortex) is a Rust-based raster-to-vector converter with official Python bindings. It uses a stacking strategy that avoids producing shapes with holes, making it well-suited for logo work. It handles both color and binary (monochrome) tracing.

- **Repository**: https://github.com/visioncortex/vtracer
- **PyPI**: https://pypi.org/project/vtracer/ (v0.6.15 as of March 2026)
- **License**: MIT

### Installation

```bash
pip install vtracer
```

Pre-built wheels are available for Windows, macOS, and Linux (compiled from Rust via PyO3).

### Python API

VTracer exposes three main functions:

#### `convert_image_to_svg_py(input_path, output_path, **kwargs)`

File-to-file conversion.

```python
import vtracer

vtracer.convert_image_to_svg_py(
    "logo_raster.png",
    "logo_vector.svg",
    colormode='color',        # 'color' | 'binary'
    hierarchical='stacked',   # 'stacked' | 'cutout'
    mode='spline',            # 'spline' | 'polygon' | 'none'
    filter_speckle=4,         # discard patches smaller than N px (default: 4)
    color_precision=6,        # color quantization precision 1-8 (default: 6)
    layer_difference=16,      # minimum color layer difference (default: 16)
    corner_threshold=60,      # angle (degrees) to detect corners (default: 60)
    length_threshold=4.0,     # path segment length threshold, range 3.5-10 (default: 4.0)
    max_iterations=10,        # max curve-fitting iterations (default: 10)
    splice_threshold=45,      # path splicing angle threshold (default: 45)
    path_precision=8,         # decimal places in SVG path output (default: 8)
)
```

#### `convert_raw_image_to_svg(image_bytes, img_format, **kwargs)`

Bytes-to-string conversion (useful for in-memory pipelines).

```python
with open("logo.png", "rb") as f:
    image_bytes = f.read()

svg_string = vtracer.convert_raw_image_to_svg(
    image_bytes,
    img_format='png',
    colormode='color',
    filter_speckle=4,
    # ... same kwargs as above
)
```

#### `convert_pixels_to_svg(pixels, size, **kwargs)`

Raw RGBA pixel data to SVG string.

```python
from PIL import Image

img = Image.open("logo.png").convert("RGBA")
pixels = list(img.getdata())
svg_string = vtracer.convert_pixels_to_svg(pixels, img.size, colormode='color')
```

### Parameter Tuning for Logos

| Scenario | colormode | filter_speckle | color_precision | corner_threshold | path_precision | mode |
|---|---|---|---|---|---|---|
| **Flat color logo** | `'color'` | 8-15 | 4-5 | 60 | 3 | `'spline'` |
| **Monochrome / line art** | `'binary'` | 4-8 | -- | 60 | 2-3 | `'spline'` |
| **Gradient logo** | `'color'` | 4 | 6-8 | 60 | 5-6 | `'spline'` |
| **Pixel art style** | `'color'` | 1 | 8 | 90 | 0 | `'polygon'` |
| **Noisy / scanned logo** | `'color'` | 10-20 | 4-5 | 60 | 3 | `'spline'` |

**Key tuning guidelines:**

- **`filter_speckle`**: The most important cleanup parameter. For logos, 8-15 removes noise without losing small features like dots in "i" letters. Start at 8 and increase if you see artifacts.
- **`color_precision`**: Lower values (4-5) merge similar colors, producing cleaner output with fewer shapes. Higher values preserve subtle gradients but produce larger files.
- **`layer_difference`**: Increase (e.g., 32-48) to merge similar color layers and reduce shape count.
- **`corner_threshold`**: 60 degrees is good for most logos. Lower values (30-45) detect more corners (sharper result). Higher values (90) smooth corners away.
- **`path_precision`**: For logos displayed at small-to-medium sizes, 2-3 decimal places are sufficient. Reduces file size significantly vs. the default of 8.
- **`mode='spline'`**: Almost always preferred for logos -- produces smooth Bezier curves. Use `'polygon'` only for pixel art.
- **`hierarchical='stacked'`**: Better for logos with overlapping colors. `'cutout'` produces shapes with holes (less predictable).

### Quality Comparison: VTracer vs. Alternatives

| Feature | VTracer | Potrace | Vector Magic | Vectorizer.AI |
|---|---|---|---|---|
| Color support | Full color | Monochrome only | Full color | Full color |
| Python API | Native bindings | Via pypotrace/CLI | Web API only | Web API only |
| Output quality (logos) | Very good | Excellent (B&W) | Excellent | Excellent |
| Speed | Fast (Rust) | Fast (C) | N/A (cloud) | N/A (cloud) |
| Cost | Free/open source | Free/open source | Paid | Paid |
| Spline fitting | Yes | Yes | Yes | Yes |
| Batch processing | Easy via Python | Easy via CLI | Limited | Limited |

---

## 2. Potrace

### When to Use Over VTracer

Potrace is the gold standard for **monochrome / two-tone** vectorization. Use Potrace when:

- The logo is black-and-white or can be thresholded to two tones
- You need the absolute smoothest Bezier curves for single-color artwork
- You want fine control over curve fitting (turnpolicy, alphamax, opttolerance)
- The input is already a clean bitmap (scanned at high resolution)

Use VTracer instead when:

- The logo has multiple colors
- You want a single-tool solution for both color and binary images
- You prefer a simpler Python API without C compilation requirements

### Setup

**Option A: pypotrace (Python bindings)**

```bash
# Linux: install system dependencies first
sudo apt-get install build-essential python-dev libagg-dev libpotrace-dev

pip install pypotrace
```

Note: pypotrace can be difficult to install on Windows. Consider using WSL or Option B.

**Option B: potrace CLI + subprocess**

```bash
# Linux
sudo apt-get install potrace

# macOS
brew install potrace

# Windows: download from https://potrace.sourceforge.net/#downloading
```

**Option C: potracer (pure Python wrapper)**

```bash
pip install potracer
```

### Python Usage

```python
# --- Using pypotrace ---
import numpy as np
from PIL import Image
import potrace

# Load and binarize
img = Image.open("logo.png").convert("L")
bitmap = np.array(img) > 128  # threshold

# Trace
bmp = potrace.Bitmap(bitmap)
path = bmp.trace(
    turdsize=2,        # suppress speckles up to this size
    turnpolicy=potrace.TURNPOLICY_MINORITY,  # resolve ambiguities
    alphamax=1.0,      # corner threshold (0=sharp, 1.334=smooth)
    opttolerance=0.2,  # curve optimization tolerance
)

# Convert to SVG
with open("logo.svg", "w") as f:
    f.write('<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{img.width}" height="{img.height}">')
    f.write('<path d="')
    for curve in path:
        f.write(f"M {curve.start_point[0]},{curve.start_point[1]} ")
        for segment in curve:
            if segment.is_corner:
                f.write(f"L {segment.c[0]},{segment.c[1]} "
                        f"L {segment.end_point[0]},{segment.end_point[1]} ")
            else:
                f.write(f"C {segment.c1[0]},{segment.c1[1]} "
                        f"{segment.c2[0]},{segment.c2[1]} "
                        f"{segment.end_point[0]},{segment.end_point[1]} ")
    f.write('z" fill="black"/></svg>')

# --- Using potrace CLI via subprocess ---
import subprocess
from PIL import Image

# Potrace requires PBM/PGM/PPM input
img = Image.open("logo.png").convert("1")  # 1-bit
img.save("logo.pbm")

subprocess.run([
    "potrace",
    "logo.pbm",
    "-s",               # SVG output
    "-o", "logo.svg",
    "-t", "5",          # turdsize: suppress speckles
    "-a", "1.0",        # alphamax: corner smoothing
    "-O", "0.2",        # opttolerance: curve optimization
])
```

---

## 3. Full Pipeline: Raster -> Cleanup -> Vectorize -> Optimize SVG

### Architecture Overview

```
Input Raster (PNG/JPG)
    |
    v
[1. Pre-processing / Cleanup]
    - Background removal
    - Denoising
    - Contrast enhancement
    - Color quantization
    - Upscaling (if low-res)
    |
    v
[2. Vectorization]
    - VTracer (color) or Potrace (B&W)
    - Parameter tuning per logo type
    |
    v
[3. SVG Optimization]
    - scour (Python) or SVGO (Node.js)
    - Path simplification
    - Precision reduction
    |
    v
[4. Quality Validation]
    - Render SVG back to raster
    - Compare with original (SSIM, visual diff)
    - File size check
    |
    v
Output SVG
```

### Complete Python Pipeline

```python
import vtracer
import subprocess
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np


def preprocess_logo(input_path: str, output_path: str, target_size: int = 1024) -> str:
    """Clean up raster logo before vectorization."""
    img = Image.open(input_path).convert("RGBA")

    # 1. Upscale if too small (vectorization works better on larger images)
    max_dim = max(img.size)
    if max_dim < target_size:
        scale = target_size / max_dim
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # 2. Remove near-white background (common in logo PNGs)
    data = np.array(img)
    # If pixel is close to white and mostly opaque, make transparent
    white_mask = (
        (data[:, :, 0] > 240) &
        (data[:, :, 1] > 240) &
        (data[:, :, 2] > 240)
    )
    data[white_mask, 3] = 0
    img = Image.fromarray(data)

    # 3. Slight denoise to remove JPEG artifacts
    rgb = img.convert("RGB")
    rgb = rgb.filter(ImageFilter.MedianFilter(size=3))

    # Recompose with original alpha
    result = Image.merge("RGBA", (*rgb.split(), img.split()[3]))

    # 4. Color quantization to reduce unique colors (optional for flat logos)
    # result = result.quantize(colors=16, method=Image.MEDIANCUT).convert("RGBA")

    result.save(output_path)
    return output_path


def vectorize_logo(
    input_path: str,
    output_path: str,
    logo_type: str = "flat_color"
) -> str:
    """Vectorize a cleaned-up raster logo using VTracer."""

    # Parameter presets by logo type
    presets = {
        "flat_color": {
            "colormode": "color",
            "hierarchical": "stacked",
            "mode": "spline",
            "filter_speckle": 10,
            "color_precision": 4,
            "layer_difference": 24,
            "corner_threshold": 60,
            "length_threshold": 4.0,
            "max_iterations": 10,
            "splice_threshold": 45,
            "path_precision": 3,
        },
        "monochrome": {
            "colormode": "binary",
            "hierarchical": "stacked",
            "mode": "spline",
            "filter_speckle": 6,
            "color_precision": 6,
            "layer_difference": 16,
            "corner_threshold": 60,
            "length_threshold": 4.0,
            "max_iterations": 10,
            "splice_threshold": 45,
            "path_precision": 2,
        },
        "detailed": {
            "colormode": "color",
            "hierarchical": "stacked",
            "mode": "spline",
            "filter_speckle": 4,
            "color_precision": 6,
            "layer_difference": 16,
            "corner_threshold": 45,
            "length_threshold": 4.0,
            "max_iterations": 15,
            "splice_threshold": 45,
            "path_precision": 4,
        },
    }

    params = presets.get(logo_type, presets["flat_color"])
    vtracer.convert_image_to_svg_py(input_path, output_path, **params)
    return output_path


def optimize_svg_scour(input_path: str, output_path: str) -> str:
    """Optimize SVG using scour (Python-native)."""
    subprocess.run([
        "scour",
        "-i", input_path,
        "-o", output_path,
        "--enable-viewboxing",
        "--enable-id-stripping",
        "--enable-comment-stripping",
        "--shorten-ids",
        "--indent=none",
        "--remove-metadata",
        "--strip-xml-prolog",
        "--no-line-breaks",
    ], check=True)
    return output_path


def optimize_svg_svgo(input_path: str, output_path: str) -> str:
    """Optimize SVG using SVGO (Node.js, must be installed globally)."""
    subprocess.run([
        "npx", "svgo",
        "-i", input_path,
        "-o", output_path,
        "--multipass",
        "--precision=2",
    ], check=True)
    return output_path


def validate_quality(original_path: str, svg_path: str) -> dict:
    """Render SVG and compare to original raster for quality check."""
    try:
        import cairosvg
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return {"error": "Install cairosvg and scikit-image for quality validation"}

    # Load original
    original = Image.open(original_path).convert("RGB")
    w, h = original.size

    # Render SVG to PNG at same size
    png_data = cairosvg.svg2png(url=svg_path, output_width=w, output_height=h)
    rendered = Image.open(io.BytesIO(png_data)).convert("RGB")

    # Compute SSIM
    orig_arr = np.array(original)
    rend_arr = np.array(rendered)
    score = ssim(orig_arr, rend_arr, channel_axis=2)

    # File size comparison
    original_size = os.path.getsize(original_path)
    svg_size = os.path.getsize(svg_path)

    return {
        "ssim": round(score, 4),
        "original_size_kb": round(original_size / 1024, 1),
        "svg_size_kb": round(svg_size / 1024, 1),
        "compression_ratio": round(original_size / max(svg_size, 1), 2),
    }


def full_pipeline(
    input_path: str,
    output_path: str,
    logo_type: str = "flat_color",
    optimizer: str = "scour",
) -> dict:
    """Run the complete raster-to-vector pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Preprocess
        cleaned = os.path.join(tmpdir, "cleaned.png")
        preprocess_logo(input_path, cleaned)

        # Step 2: Vectorize
        raw_svg = os.path.join(tmpdir, "raw.svg")
        vectorize_logo(cleaned, raw_svg, logo_type=logo_type)

        # Step 3: Optimize
        if optimizer == "scour":
            optimize_svg_scour(raw_svg, output_path)
        elif optimizer == "svgo":
            optimize_svg_svgo(raw_svg, output_path)
        else:
            # No optimization, just copy
            import shutil
            shutil.copy(raw_svg, output_path)

        # Step 4: Validate
        quality = validate_quality(input_path, output_path)

    return quality


# Usage
if __name__ == "__main__":
    result = full_pipeline(
        "my_logo.png",
        "my_logo.svg",
        logo_type="flat_color",
        optimizer="scour",
    )
    print(f"SSIM: {result.get('ssim', 'N/A')}")
    print(f"SVG size: {result.get('svg_size_kb', 'N/A')} KB")
```

---

## 4. SVG Optimization

### SVGO (Node.js)

The most widely used SVG optimizer. Provides plugin-based architecture for fine-grained control.

```bash
npm install -g svgo
```

**CLI usage:**

```bash
# Basic optimization
svgo input.svg -o output.svg

# Aggressive with multipass
svgo input.svg -o output.svg --multipass --precision=2

# Custom plugin configuration
svgo input.svg -o output.svg --config svgo.config.js
```

**Configuration for logos (`svgo.config.js`):**

```javascript
module.exports = {
  multipass: true,
  precision: 2,
  plugins: [
    'preset-default',
    'removeDimensions',        // use viewBox instead of width/height
    'removeOffCanvasPaths',
    'reusePaths',
    'sortAttrs',
    {
      name: 'removeAttrs',
      params: { attrs: ['data-*', 'class'] }
    },
    {
      name: 'preset-default',
      params: {
        overrides: {
          removeViewBox: false,  // keep viewBox for scalability
          cleanupNumericValues: { floatPrecision: 2 },
          convertPathData: {
            floatPrecision: 2,
            transformPrecision: 3,
          },
          mergePaths: true,
          convertShapeToPath: true,
        }
      }
    }
  ]
};
```

**Key plugins for logos:**

| Plugin | Effect | Size Impact |
|---|---|---|
| `convertPathData` | Optimizes path commands (relative coords, shorthand) | 15-30% |
| `mergePaths` | Combines paths with same attributes | 5-15% |
| `cleanupNumericValues` | Reduces decimal precision | 10-20% |
| `removeUselessStrokeAndFill` | Strips redundant style attrs | 2-5% |
| `convertShapeToPath` | Unifies shapes as paths | Varies |
| `collapseGroups` | Flattens unnecessary `<g>` wrappers | 2-5% |
| `multipass` | Runs all optimizations multiple times | 5-10% additional |

### Scour (Python)

Python-native alternative. Integrates directly into Python pipelines without shelling out to Node.js.

```bash
pip install scour
```

**Python API usage:**

```python
from scour.scour import scourString

with open("input.svg", "r") as f:
    svg_input = f.read()

options = {
    "enable_viewboxing": True,
    "strip_ids": True,
    "remove_metadata": True,
    "strip_comments": True,
    "shorten_ids": True,
    "indent_type": "none",
    "newlines": False,
    "strip_xml_prolog": True,
    "remove_descriptive_elements": True,
}

# scour expects an options object, not a dict
from scour.scour import parse_args
scour_options = parse_args([
    "--enable-viewboxing",
    "--enable-id-stripping",
    "--enable-comment-stripping",
    "--shorten-ids",
    "--indent=none",
    "--remove-metadata",
    "--strip-xml-prolog",
    "--no-line-breaks",
])

svg_output = scourString(svg_input, options=scour_options)

with open("output.svg", "w") as f:
    f.write(svg_output)
```

### Comparison: SVGO vs. Scour

| Feature | SVGO | Scour |
|---|---|---|
| Language | Node.js | Python |
| Plugin system | Extensive | Limited |
| Compression | Typically 30-60% | Typically 20-40% |
| Speed | Fast | Moderate |
| Python integration | Via subprocess | Native |
| Multipass | Built-in | Manual |
| Precision control | Per-plugin | Global |
| Best for | Production builds | Python-native pipelines |

**Recommendation**: Use scour for a pure-Python pipeline. For maximum compression, run SVGO as a final pass. They can be chained (scour first, then SVGO) for best results.

---

## 5. SVG Path Simplification and Cleanup Techniques

### Path Command Optimization

**Relative vs. Absolute coordinates**: Relative coordinates (`m`, `l`, `c`) are often shorter than absolute (`M`, `L`, `C`) because the numbers are smaller.

```xml
<!-- Absolute (longer) -->
<path d="M 100,200 L 150,200 L 150,250 L 100,250 Z"/>

<!-- Relative (shorter) -->
<path d="M100,200l50,0l0,50l-50,0z"/>
```

**Implicit line commands**: After `M`/`m`, subsequent coordinate pairs are implicitly `L`/`l` commands.

```xml
<!-- Verbose -->
<path d="M 10,10 L 20,20 L 30,10"/>

<!-- Compact -->
<path d="M10,10 20,20 30,10"/>
```

**H/V shorthand**: Replace `L` with `H` (horizontal) or `V` (vertical) when only one axis changes.

```xml
<!-- Before -->
<path d="M 0,0 L 100,0 L 100,100"/>

<!-- After -->
<path d="M0,0H100V100"/>
```

### Python Libraries for Path Manipulation

#### svgpathtools

```bash
pip install svgpathtools
```

```python
from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier

# Load SVG paths
paths, attributes = svg2paths("logo.svg")

# Analyze path complexity
for i, path in enumerate(paths):
    print(f"Path {i}: {len(path)} segments, length={path.length():.1f}")

# Simplify: remove very short segments
def simplify_path(path, min_length=1.0):
    """Remove segments shorter than min_length."""
    return Path(*[seg for seg in path if seg.length() > min_length])

simplified = [simplify_path(p) for p in paths]
wsvg(simplified, attributes=attributes, filename="simplified.svg")
```

#### picosvg (Google)

Normalizes SVGs to a simplified subset -- all shapes become cubic Bezier paths. Useful for consistent downstream processing.

```bash
pip install picosvg
```

```bash
# CLI usage
picosvg input.svg > normalized.svg
```

#### svgpathtools_simplify

Companion to svgpathtools that breaks apart discontinuous paths and matches endpoints.

```bash
pip install svgpathtools_simplify
```

### Manual Cleanup Patterns

1. **Remove hidden elements**: Paths with `display:none`, zero opacity, or off-canvas coordinates.
2. **Merge adjacent paths**: Paths with identical fill/stroke can be combined into a single `<path>`.
3. **Remove empty groups**: `<g>` elements with no children or only whitespace.
4. **Flatten transforms**: Bake `transform` attributes directly into path coordinates.
5. **Round coordinates**: Reduce decimal places (2-3 is sufficient for logos displayed up to ~1000px).
6. **Remove editor metadata**: Inkscape, Illustrator, and Sketch embed custom namespaces and metadata.

### Douglas-Peucker Path Simplification

For reducing point count in polygon paths:

```python
import numpy as np

def douglas_peucker(points, epsilon):
    """Simplify a polyline using the Douglas-Peucker algorithm."""
    if len(points) <= 2:
        return points

    # Find point with maximum distance from line between first and last
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)

    if line_len == 0:
        return [points[0], points[-1]]

    line_unit = line_vec / line_len
    distances = np.abs(np.cross(line_unit, start - points))
    max_idx = np.argmax(distances)
    max_dist = distances[max_idx]

    if max_dist > epsilon:
        left = douglas_peucker(points[:max_idx + 1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]
```

---

## 6. Direct SVG Generation Approaches

Instead of raster-then-vectorize, these methods generate SVG directly.

### 6.1 PyTorch-SVGRender / DiffVG

A framework for differentiable SVG rendering, enabling gradient-based optimization of SVG parameters.

- **Repository**: https://github.com/ximinng/PyTorch-SVGRender
- **Paper**: Multiple (CVPR 2025, T-PAMI)

**Supported methods include:**

| Method | Task | Approach |
|---|---|---|
| DiffVG | Image-to-SVG | Differentiable rasterizer |
| LIVE | Image-to-SVG | Layer-wise vectorization |
| CLIPDraw | Text-to-SVG | CLIP-guided drawing |
| SVGDreamer | Text-to-SVG | Diffusion + vectorization |
| SVGDreamer++ | Text-to-SVG | Hierarchical vectorization |

**Example (image vectorization with LIVE):**

```bash
# Install
pip install torch torchvision
git clone https://github.com/ximinng/PyTorch-SVGRender.git
cd PyTorch-SVGRender
pip install -r requirements.txt

# Run LIVE vectorization
python svg_render.py x=live target="logo.png" \
    --num_paths 128 \
    --num_iter 500
```

**Pros**: Can produce semantically meaningful SVGs with clean, editable paths.
**Cons**: Slow (minutes per image on GPU), requires PyTorch + CUDA, complex setup.

### 6.2 SVGDreamer / SVGDreamer++

Text-to-SVG synthesis using diffusion models.

- Decomposes scene into foreground objects + background (SIVE)
- Supports multiple styles: iconography, sketch, pixel art, low-poly, painting
- SVGDreamer++ (T-PAMI 2025) adds hierarchical vectorization and adaptive primitive control

```bash
# Using PyTorch-SVGRender framework
python svg_render.py x=svgdreamer prompt="minimalist tech company logo, geometric" \
    --style iconography \
    --num_paths 64
```

**Relevance to logos**: The iconography style mode is directly applicable to logo generation, producing clean geometric SVGs.

### 6.3 StarVector (CVPR 2025)

A multimodal LLM that directly generates SVG code from images or text.

- **Repository**: https://github.com/joanrod/star-vector
- **Models**: StarVector-1B (lightweight) and StarVector-8B (highest quality)
- **Training**: Pre-trained on SVG-Stack dataset (2.1M samples)
- **Benchmark**: Introduces SVG-Bench across 10 datasets

```python
# Using StarVector for image-to-SVG
from starvector import StarVector

model = StarVector.from_pretrained("starvector/starvector-8b-im2svg")
svg_code = model.generate(image_path="logo.png")

with open("logo.svg", "w") as f:
    f.write(svg_code)
```

**Pros**: Produces semantically structured SVG (named layers, logical grouping). Understands image content.
**Cons**: Large model, requires GPU, output may need cleanup.

### 6.4 Chat2SVG (CVPR 2025)

Hybrid LLM + diffusion framework for text-to-SVG.

- **Architecture**: Three-stage pipeline:
  1. LLM generates SVG template from basic primitives (rect, ellipse, path)
  2. SDEdit + ControlNet enhances visual details
  3. Dual-stage optimization refines paths in latent space
- **Repository**: https://github.com/kingnobro/Chat2SVG

**Relevance**: The LLM template stage produces clean, editable SVG structures ideal for logo frameworks.

### 6.5 LLM4SVG (CVPR 2025)

Empowers standard LLMs to understand and generate complex vector graphics.

- **Repository**: https://github.com/ximinng/LLM4SVG
- Trains LLMs to work in SVG code space with specialized tokenization

### 6.6 LLM Direct Code Generation (Claude, GPT-4)

Modern LLMs can generate SVG code directly from text prompts.

```python
# Example: Using Claude API for SVG generation
import anthropic

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": (
            "Generate a clean, minimal SVG logo for a tech company called 'Nexus'. "
            "Use geometric shapes, blue color palette (#1a73e8, #4285f4, #8ab4f8). "
            "Output only the SVG code, no explanation. "
            "Keep paths simple with max 2 decimal places. "
            "Use viewBox='0 0 100 100'."
        )
    }]
)
svg_code = message.content[0].text
# Extract SVG from markdown code block if present
if "```" in svg_code:
    svg_code = svg_code.split("```")[1].strip()
    if svg_code.startswith("svg") or svg_code.startswith("xml"):
        svg_code = svg_code.split("\n", 1)[1]
```

**Strengths**: Fast iteration, natural language control, produces clean/editable SVG.
**Weaknesses**: Struggles with complex curved shapes, inconsistent quality, hard to match a specific raster reference.

### Comparison of Direct SVG Approaches

| Method | Input | Quality | Speed | GPU Required | Logo Suitability |
|---|---|---|---|---|---|
| VTracer (trace) | Raster | Good | Fast | No | High |
| DiffVG/LIVE | Raster | Very good | Slow | Yes | Medium |
| SVGDreamer | Text | Good | Very slow | Yes | Medium (iconography) |
| StarVector | Raster/Text | Very good | Moderate | Yes | High |
| Chat2SVG | Text | Good | Moderate | Yes | High |
| LLM codegen | Text | Variable | Fast | No | Medium-High (simple) |

---

## 7. Quality Metrics for Vector Output

### Visual Fidelity Metrics

#### SSIM (Structural Similarity Index)

The primary metric for comparing rasterized SVG against the original. Measures luminance, contrast, and structural similarity.

```python
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

def compute_ssim(original_path: str, svg_path: str, render_size: int = 512) -> float:
    """Compute SSIM between original raster and rendered SVG."""
    import cairosvg
    import io

    # Load original and resize
    orig = Image.open(original_path).convert("RGB").resize((render_size, render_size))

    # Render SVG
    png_data = cairosvg.svg2png(
        url=svg_path,
        output_width=render_size,
        output_height=render_size
    )
    rendered = Image.open(io.BytesIO(png_data)).convert("RGB")

    return ssim(
        np.array(orig),
        np.array(rendered),
        channel_axis=2,        # color images
        data_range=255,
    )

# Interpretation:
# > 0.95: Excellent (visually identical)
# 0.90-0.95: Very good (minor differences)
# 0.80-0.90: Good (noticeable but acceptable)
# < 0.80: Poor (visible artifacts or missing details)
```

#### Multi-Scale SSIM (MS-SSIM)

Better correlates with human perception by evaluating at multiple resolutions.

```python
# pip install piq
import piq
import torch

def compute_ms_ssim(orig_tensor, rendered_tensor):
    """Compute MS-SSIM (requires tensors in [B, C, H, W] format, range [0, 1])."""
    return piq.multi_scale_ssim(orig_tensor, rendered_tensor, data_range=1.0)
```

#### LPIPS (Learned Perceptual Image Patch Similarity)

Neural-network-based perceptual distance. Lower is better.

```python
# pip install lpips
import lpips

loss_fn = lpips.LPIPS(net='alex')  # or 'vgg'
distance = loss_fn(orig_tensor, rendered_tensor)
# < 0.05: Excellent
# 0.05-0.15: Good
# > 0.15: Noticeable differences
```

### SVG-Specific Metrics

#### Path Complexity

```python
import re

def svg_complexity(svg_path: str) -> dict:
    """Measure SVG complexity metrics."""
    with open(svg_path, "r") as f:
        svg_content = f.read()

    # Count elements
    paths = len(re.findall(r'<path', svg_content))
    groups = len(re.findall(r'<g[ >]', svg_content))

    # Count path commands
    d_attrs = re.findall(r'd="([^"]*)"', svg_content)
    total_commands = 0
    total_points = 0
    for d in d_attrs:
        commands = re.findall(r'[MmLlHhVvCcSsQqTtAaZz]', d)
        total_commands += len(commands)
        # Rough point count from coordinate pairs
        coords = re.findall(r'-?\d+\.?\d*', d)
        total_points += len(coords) // 2

    file_size = os.path.getsize(svg_path)

    return {
        "file_size_bytes": file_size,
        "num_paths": paths,
        "num_groups": groups,
        "total_commands": total_commands,
        "total_points": total_points,
        "avg_commands_per_path": total_commands / max(paths, 1),
    }
```

#### Color Count

```python
def count_svg_colors(svg_path: str) -> int:
    """Count unique fill/stroke colors in SVG."""
    with open(svg_path, "r") as f:
        svg_content = f.read()
    colors = set(re.findall(r'(?:fill|stroke)="(#[0-9a-fA-F]{3,8})"', svg_content))
    colors.update(re.findall(r'(?:fill|stroke):\s*(#[0-9a-fA-F]{3,8})', svg_content))
    return len(colors)
```

### Composite Quality Score

```python
def logo_quality_score(
    original_path: str,
    svg_path: str,
    target_max_size_kb: float = 50.0,
) -> dict:
    """Compute a composite quality score for a vectorized logo."""
    # Visual fidelity (weight: 0.5)
    ssim_score = compute_ssim(original_path, svg_path)

    # Complexity (weight: 0.3) - penalize overly complex SVGs
    complexity = svg_complexity(svg_path)
    size_kb = complexity["file_size_bytes"] / 1024
    size_score = max(0, 1.0 - (size_kb / target_max_size_kb)) if size_kb > target_max_size_kb else 1.0
    cmd_score = max(0, 1.0 - (complexity["total_commands"] / 5000))

    complexity_score = 0.6 * size_score + 0.4 * cmd_score

    # Composite
    composite = 0.5 * ssim_score + 0.3 * complexity_score + 0.2 * 1.0  # 0.2 reserved for editability

    return {
        "ssim": round(ssim_score, 4),
        "file_size_kb": round(size_kb, 1),
        "path_count": complexity["num_paths"],
        "total_commands": complexity["total_commands"],
        "complexity_score": round(complexity_score, 4),
        "composite_score": round(composite, 4),
    }
```

---

## 8. Recommendations for Logo-Gen Project

### Recommended Default Pipeline

1. **Primary vectorizer**: VTracer (via Python bindings) -- handles both color and monochrome, fast, no GPU needed.
2. **SVG optimizer**: scour (Python-native) for integration, optionally SVGO as final pass.
3. **Quality validation**: SSIM at 512x512 render, with threshold of 0.90 minimum.

### When to Use Each Approach

| Scenario | Recommended Approach |
|---|---|
| Raster logo exists, need SVG | VTracer pipeline (Section 3) |
| Need black-and-white logo SVG | Potrace (cleaner curves for monochrome) |
| Generating logo from text prompt | LLM codegen (Claude) for simple, or Chat2SVG for complex |
| Need editable/semantic SVG | StarVector or LLM codegen |
| Maximum quality, time not critical | DiffVG/LIVE optimization |
| Batch processing many logos | VTracer + scour pipeline |

### Dependencies Summary

```
# Core pipeline
pip install vtracer Pillow numpy scour

# Quality metrics
pip install scikit-image cairosvg

# Path manipulation
pip install svgpathtools picosvg

# Advanced metrics (optional)
pip install piq lpips

# Potrace (optional, for monochrome)
pip install pypotrace  # or use CLI: apt install potrace

# SVGO (optional, for max compression)
npm install -g svgo
```

---

## Sources

- [VTracer GitHub](https://github.com/visioncortex/vtracer)
- [VTracer PyPI](https://pypi.org/project/vtracer/)
- [VTracer Documentation](https://www.visioncortex.org/vtracer-docs/)
- [Potrace](https://potrace.sourceforge.net/)
- [pypotrace](https://pypi.org/project/pypotrace/)
- [SVGO GitHub](https://github.com/svg/svgo)
- [Scour GitHub](https://github.com/scour-project/scour)
- [svgpathtools GitHub](https://github.com/mathandy/svgpathtools)
- [picosvg GitHub](https://github.com/googlefonts/picosvg)
- [PyTorch-SVGRender](https://github.com/ximinng/PyTorch-SVGRender)
- [SVGDreamer](https://ximinng.github.io/SVGDreamer-project/)
- [StarVector (CVPR 2025)](https://arxiv.org/abs/2312.11556)
- [Chat2SVG (CVPR 2025)](https://github.com/kingnobro/Chat2SVG)
- [LLM4SVG (CVPR 2025)](https://github.com/ximinng/LLM4SVG)
- [SVG Code Optimization Guide](https://www.svgai.org/blog/svg-code-optimization-guide)
- [SVG Path Optimization Techniques](https://vectosolve.com/blog/svg-path-optimization-techniques)
- [AI-Powered Vectorization Workflows 2026](https://www.svggenie.com/blog/raster-to-vector-ai-vectorization-2026)
- [LLM SVG Benchmark](https://www.communeify.com/en/blog/ai-image-generation-showdown-9-llms-svg-benchmark/)
