# Color Palette Control for Logo Generation

## 1. Prompt-Based Color Control Across Different Models

### Stable Diffusion / SDXL

Stable Diffusion responds well to explicit color instructions in prompts, but control is approximate rather than exact.

**Effective prompt patterns:**
- Direct color naming: `"a logo in navy blue and gold"` -- works for common color names
- Material/finish references: `"metallic gold logo"`, `"matte black emblem"` -- leverages training associations
- Negative prompts: `"--no red, no warm colors"` -- removes unwanted hues
- Style anchors: `"monochrome logo"`, `"duotone in teal and coral"` -- constrains the palette

**Limitations:**
- Cannot specify exact hex values through prompts alone
- Color bleeding between elements is common
- Background color often leaks into foreground objects
- Model interprets "blue" broadly (could produce any shade)

**ControlNet color grids:** A ControlNet preprocessor can accept a color-blocked reference image (simple rectangles of desired colors) to bias generation toward a specific palette. This is the most reliable prompt-adjacent method for SD.

### DALL-E 3

- Responds to natural language color descriptions: `"using only #2563EB blue and white"`
- Hex codes in prompts are sometimes respected but not guaranteed
- Best results come from descriptive color language: `"deep ocean blue"` rather than `"#003366"`
- Color palette images as references are not supported (no image-to-image)

### Midjourney

- Supports `--style` and color-related suffixes
- Hex codes can be included in prompts and are sometimes loosely respected
- Image references (via URL) can transfer color palettes from reference images
- `--no red` style negation helps exclude colors

### Flux (Black Forest Labs)

- Strong prompt adherence for color descriptions
- Similar to SDXL in that explicit color names work well
- ControlNet-style conditioning is emerging for Flux

### Ideogram

- Particularly strong at text rendering and color adherence
- Color descriptions in prompts are well-respected
- Handles multi-color specifications more reliably than most models

### General Best Practices for Prompt-Based Color

1. Use specific, well-known color names (e.g., "cerulean" not "light blue")
2. Reference materials that imply color (e.g., "copper", "ivory", "obsidian")
3. Specify background color explicitly
4. Use negative prompts to exclude unwanted colors
5. Generate multiple candidates and select the closest match
6. Combine with post-generation recoloring for precision

---

## 2. Post-Generation Recoloring Techniques

Since prompt-based color control is imprecise, post-generation recoloring is essential for brand-accurate output.

### Raster Image Recoloring

**HSL/HSV channel manipulation:**
```python
import cv2
import numpy as np
from PIL import Image

def recolor_hue_shift(image_path: str, target_hue: int, output_path: str):
    """
    Shift all hues in an image toward a target hue.
    target_hue: 0-179 in OpenCV's HSV space (0=red, 60=green, 120=blue).
    """
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate shift needed
    current_mean_hue = np.mean(hsv[:, :, 0][hsv[:, :, 1] > 30])  # ignore low-saturation pixels
    shift = target_hue - current_mean_hue

    hsv[:, :, 0] = np.clip(hsv[:, :, 0].astype(np.float32) + shift, 0, 179).astype(np.uint8)

    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, result)
```

**Selective color replacement (replace specific color ranges):**
```python
def replace_color_range(
    image_path: str,
    lower_hsv: tuple,
    upper_hsv: tuple,
    replacement_bgr: tuple,
    output_path: str,
):
    """Replace pixels within an HSV range with a specific BGR color."""
    img = cv2.imread(image_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

    # Optional: smooth the mask to avoid hard edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask_float = mask.astype(np.float32) / 255.0

    result = img.copy().astype(np.float32)
    for c in range(3):
        result[:, :, c] = (
            result[:, :, c] * (1 - mask_float) + replacement_bgr[c] * mask_float
        )

    cv2.imwrite(output_path, result.astype(np.uint8))
```

**Color transfer (transfer palette from a reference image):**
```python
def color_transfer_lab(source_path: str, target_path: str, output_path: str):
    """
    Transfer color statistics from target to source using LAB color space.
    Based on Reinhard et al. (2001) color transfer algorithm.
    """
    source = cv2.imread(source_path).astype(np.float32)
    target = cv2.imread(target_path).astype(np.float32)

    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # Compute mean and std for each channel
    for ch in range(3):
        src_mean, src_std = source_lab[:, :, ch].mean(), source_lab[:, :, ch].std()
        tgt_mean, tgt_std = target_lab[:, :, ch].mean(), target_lab[:, :, ch].std()

        # Normalize source, then scale to target statistics
        source_lab[:, :, ch] = (
            (source_lab[:, :, ch] - src_mean) * (tgt_std / (src_std + 1e-6)) + tgt_mean
        )

    source_lab = np.clip(source_lab, 0, 255)
    result = cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, result)
```

### Palette-Constrained Recoloring

```python
from scipy.spatial import KDTree

def recolor_to_palette(image_path: str, palette_rgb: list[tuple], output_path: str):
    """
    Snap every pixel to the nearest color in a fixed palette.
    palette_rgb: list of (R, G, B) tuples.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w, _ = img.shape
    pixels = img.reshape(-1, 3).astype(np.float32)

    tree = KDTree(np.array(palette_rgb, dtype=np.float32))
    _, indices = tree.query(pixels)

    palette_arr = np.array(palette_rgb, dtype=np.uint8)
    result = palette_arr[indices].reshape(h, w, 3)
    Image.fromarray(result).save(output_path)
```

---

## 3. Color Palette Extraction from Images

### colorthief (Python)

The most popular library for extracting dominant colors from images.

```python
from colorthief import ColorThief

def extract_palette(image_path: str, color_count: int = 6, quality: int = 10):
    """
    Extract a palette from an image using colorthief.
    quality: 1 = highest quality (slowest), 10 = default.
    Returns list of (R, G, B) tuples.
    """
    ct = ColorThief(image_path)

    # Get the single dominant color
    dominant = ct.get_color(quality=quality)

    # Get a full palette
    palette = ct.get_palette(color_count=color_count, quality=quality)

    return {"dominant": dominant, "palette": palette}

# Example usage:
# result = extract_palette("logo.png", color_count=5)
# result = {'dominant': (41, 98, 255), 'palette': [(41,98,255), (255,255,255), ...]}
```

**How it works internally:** colorthief uses the Median Cut algorithm -- it recursively splits the color space along the axis of greatest range, producing clusters that represent dominant colors. It is fast but not always perceptually optimal.

### K-Means Clustering (scikit-learn)

More configurable and often more accurate for logo work:

```python
from sklearn.cluster import KMeans
from collections import Counter

def extract_palette_kmeans(
    image_path: str,
    n_colors: int = 5,
    ignore_white: bool = True,
    ignore_black: bool = True,
    sample_size: int = 10000,
):
    """
    Extract palette using K-Means clustering in LAB color space.
    LAB clustering produces more perceptually uniform palette groupings.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    pixels = img.reshape(-1, 3).astype(np.float32)

    # Filter out near-white and near-black if requested
    if ignore_white:
        mask = np.all(pixels < 240, axis=1)
        pixels = pixels[mask]
    if ignore_black:
        mask = np.all(pixels > 15, axis=1)
        pixels = pixels[mask]

    # Subsample for speed
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]

    # Convert to LAB for perceptually uniform clustering
    pixels_bgr = pixels[:, ::-1].reshape(1, -1, 3).astype(np.uint8)
    pixels_lab = cv2.cvtColor(pixels_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_colors, n_init=10, random_state=42)
    kmeans.fit(pixels_lab)

    # Convert cluster centers back to RGB
    centers_lab = kmeans.cluster_centers_.reshape(1, -1, 3).astype(np.uint8)
    centers_bgr = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2BGR).reshape(-1, 3)
    centers_rgb = centers_bgr[:, ::-1]

    # Sort by cluster size (most dominant first)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_indices = np.argsort(-counts)

    palette = [tuple(centers_rgb[i]) for i in sorted_indices]
    proportions = [counts[i] / counts.sum() for i in sorted_indices]

    return list(zip(palette, proportions))
```

### Other Extraction Libraries

| Library | Method | Notes |
|---------|--------|-------|
| `colorthief` | Median Cut | Fast, simple API, good enough for most cases |
| `sklearn.cluster.KMeans` | K-Means | More configurable, works in LAB space |
| `sklearn.cluster.MiniBatchKMeans` | Mini-Batch K-Means | Faster for large images |
| `extcolors` | Pixel counting with tolerance | Groups similar colors, returns counts |
| `haishoku` | Custom algorithm | Designed for Japanese aesthetic palettes |
| `Pillow` (manual) | `getcolors()` | Built-in, limited but no extra deps |
| `fast-colorthief` | Optimized Median Cut | Rust-backed, faster than colorthief |

### extcolors Example

```python
import extcolors

def extract_with_extcolors(image_path: str, tolerance: int = 32, limit: int = 8):
    """
    Extract colors with tolerance-based grouping.
    tolerance: how different two colors must be to count as separate (0-255).
    """
    colors, pixel_count = extcolors.extract_from_path(image_path, tolerance=tolerance, limit=limit)
    # colors is a list of ((R, G, B), count) tuples
    return [(rgb, count / pixel_count) for rgb, count in colors]
```

---

## 4. AI-Based Palette Generation from Brand Descriptions

### LLM-Driven Palette Generation

Use an LLM to translate brand attributes into color palettes:

```python
import json
from anthropic import Anthropic

def generate_brand_palette(brand_description: str, num_colors: int = 5) -> dict:
    """
    Use Claude to generate a brand-appropriate color palette
    from a natural language description.
    """
    client = Anthropic()

    prompt = f"""Generate a color palette for the following brand:

{brand_description}

Return exactly {num_colors} colors as a JSON object with this structure:
{{
    "palette": [
        {{
            "hex": "#RRGGBB",
            "name": "descriptive color name",
            "role": "primary|secondary|accent|neutral|background",
            "rationale": "why this color fits the brand"
        }}
    ],
    "color_relationships": "description of how colors work together",
    "mood": "overall mood the palette conveys"
}}

Guidelines:
- Include at least one neutral/dark and one light color
- Ensure sufficient contrast between text colors and backgrounds
- Consider cultural color associations relevant to the brand's market
- Prefer colors that reproduce well in both digital and print"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse JSON from response
    response_text = message.content[0].text
    # Find JSON block in response
    start = response_text.index("{")
    end = response_text.rindex("}") + 1
    return json.loads(response_text[start:end])

# Usage:
# palette = generate_brand_palette(
#     "A sustainable outdoor gear company targeting millennials. "
#     "Values: adventure, environmental responsibility, premium quality. "
#     "Competitors use greens and browns; we want to stand out."
# )
```

### Generating Palettes from Color Psychology Rules

```python
# Color psychology associations for brand contexts
COLOR_PSYCHOLOGY = {
    "trust":       ["#1E40AF", "#1D4ED8", "#2563EB"],  # blues
    "energy":      ["#DC2626", "#EF4444", "#F97316"],  # reds, oranges
    "growth":      ["#059669", "#10B981", "#34D399"],  # greens
    "luxury":      ["#1F2937", "#7C3AED", "#D4AF37"],  # dark, purple, gold
    "innovation":  ["#6366F1", "#8B5CF6", "#06B6D4"],  # purples, cyan
    "health":      ["#059669", "#10B981", "#60A5FA"],  # greens, light blue
    "warmth":      ["#F59E0B", "#F97316", "#EF4444"],  # ambers, oranges
    "calm":        ["#6366F1", "#818CF8", "#A5B4FC"],  # soft purples, lavenders
    "professional":["#1F2937", "#374151", "#6B7280"],  # grays, dark neutrals
    "playful":     ["#EC4899", "#F472B6", "#FBBF24"],  # pinks, yellows
}

def palette_from_attributes(attributes: list[str], accent_count: int = 2) -> list[str]:
    """
    Generate a palette from brand attribute keywords.
    Returns list of hex codes.
    """
    candidates = []
    for attr in attributes:
        attr_lower = attr.lower()
        if attr_lower in COLOR_PSYCHOLOGY:
            candidates.extend(COLOR_PSYCHOLOGY[attr_lower])

    # Deduplicate and limit
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique[:accent_count + 3]  # primary + secondary + accents + neutral space
```

---

## 5. Color Harmony Algorithms

### Core Harmony Types

```python
import colorsys

def hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    """Convert hex to HSL (H: 0-360, S: 0-1, L: 0-1)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s, l

def hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL (H: 0-360, S: 0-1, L: 0-1) to hex."""
    r, g, b = colorsys.hls_to_rgb(h / 360.0, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))

def complementary(hex_color: str) -> list[str]:
    """Return the base color and its complement (180 degrees opposite)."""
    h, s, l = hex_to_hsl(hex_color)
    return [hex_color, hsl_to_hex((h + 180) % 360, s, l)]

def analogous(hex_color: str, angle: float = 30) -> list[str]:
    """Return base color and two neighbors on the color wheel."""
    h, s, l = hex_to_hsl(hex_color)
    return [
        hsl_to_hex((h - angle) % 360, s, l),
        hex_color,
        hsl_to_hex((h + angle) % 360, s, l),
    ]

def triadic(hex_color: str) -> list[str]:
    """Return three evenly spaced colors (120 degrees apart)."""
    h, s, l = hex_to_hsl(hex_color)
    return [
        hex_color,
        hsl_to_hex((h + 120) % 360, s, l),
        hsl_to_hex((h + 240) % 360, s, l),
    ]

def split_complementary(hex_color: str, angle: float = 30) -> list[str]:
    """Base color plus two colors adjacent to its complement."""
    h, s, l = hex_to_hsl(hex_color)
    comp = (h + 180) % 360
    return [
        hex_color,
        hsl_to_hex((comp - angle) % 360, s, l),
        hsl_to_hex((comp + angle) % 360, s, l),
    ]

def tetradic(hex_color: str) -> list[str]:
    """Four colors forming a rectangle on the color wheel."""
    h, s, l = hex_to_hsl(hex_color)
    return [
        hex_color,
        hsl_to_hex((h + 90) % 360, s, l),
        hsl_to_hex((h + 180) % 360, s, l),
        hsl_to_hex((h + 270) % 360, s, l),
    ]

def monochromatic(hex_color: str, variations: int = 5) -> list[str]:
    """Generate lightness variations of a single hue."""
    h, s, l = hex_to_hsl(hex_color)
    step = 0.7 / (variations - 1) if variations > 1 else 0
    return [
        hsl_to_hex(h, s, 0.15 + step * i)
        for i in range(variations)
    ]
```

### Full Palette Builder with Harmony + Roles

```python
def build_logo_palette(
    primary_hex: str,
    harmony: str = "complementary",
    include_neutrals: bool = True,
) -> dict:
    """
    Build a complete logo palette from a primary color and a harmony rule.
    Returns colors with assigned roles.
    """
    harmony_funcs = {
        "complementary": complementary,
        "analogous": analogous,
        "triadic": triadic,
        "split_complementary": split_complementary,
        "tetradic": tetradic,
        "monochromatic": monochromatic,
    }

    func = harmony_funcs.get(harmony, complementary)
    colors = func(primary_hex)

    palette = {
        "primary": colors[0],
        "secondary": colors[1] if len(colors) > 1 else colors[0],
        "accent": colors[2] if len(colors) > 2 else colors[-1],
    }

    if include_neutrals:
        h, s, l = hex_to_hsl(primary_hex)
        palette["dark"] = hsl_to_hex(h, s * 0.15, 0.12)
        palette["light"] = hsl_to_hex(h, s * 0.08, 0.96)

    return palette
```

---

## 6. Brand Color Consistency Enforcement

### Delta-E Color Distance

The standard method for measuring perceptual color difference. Delta-E < 2.0 is considered imperceptible to most observers.

```python
from skimage.color import deltaE_ciede2000, rgb2lab

def color_distance(hex1: str, hex2: str) -> float:
    """
    Compute CIEDE2000 color distance between two hex colors.
    < 1.0: imperceptible
    1-2: perceptible on close inspection
    2-10: noticeable at a glance
    > 10: colors are clearly different
    """
    def hex_to_lab(h):
        h = h.lstrip("#")
        rgb = np.array([[[int(h[i:i+2], 16) for i in (0, 2, 4)]]], dtype=np.uint8)
        return rgb2lab(rgb)[0, 0]

    lab1 = hex_to_lab(hex1)
    lab2 = hex_to_lab(hex2)
    return float(deltaE_ciede2000(lab1.reshape(1, 1, 3), lab2.reshape(1, 1, 3))[0, 0])
```

### Brand Color Validator

```python
class BrandColorValidator:
    """Validate that generated images conform to brand color guidelines."""

    def __init__(self, brand_colors: dict[str, str], tolerance: float = 5.0):
        """
        brand_colors: {"primary": "#2563EB", "secondary": "#F59E0B", ...}
        tolerance: maximum Delta-E for a color to be considered "on-brand"
        """
        self.brand_colors = brand_colors
        self.tolerance = tolerance

    def validate_image(self, image_path: str, min_brand_coverage: float = 0.6) -> dict:
        """
        Check if an image's colors match the brand palette.
        min_brand_coverage: minimum fraction of non-neutral pixels that must
            be within tolerance of a brand color.
        """
        # Extract image palette
        palette_with_proportions = extract_palette_kmeans(image_path, n_colors=8)

        brand_hex_list = list(self.brand_colors.values())
        on_brand_coverage = 0.0
        off_brand_colors = []

        for (r, g, b), proportion in palette_with_proportions:
            pixel_hex = "#{:02x}{:02x}{:02x}".format(r, g, b)

            # Check if this color is close to any brand color
            min_dist = min(color_distance(pixel_hex, bh) for bh in brand_hex_list)

            if min_dist <= self.tolerance:
                on_brand_coverage += proportion
            else:
                # Check if it is a neutral (low saturation)
                _, s, l = hex_to_hsl(pixel_hex)
                if s < 0.1 or l < 0.1 or l > 0.95:
                    continue  # neutrals are acceptable
                off_brand_colors.append((pixel_hex, proportion, min_dist))

        passed = on_brand_coverage >= min_brand_coverage

        return {
            "passed": passed,
            "on_brand_coverage": round(on_brand_coverage, 3),
            "off_brand_colors": off_brand_colors,
            "message": (
                "Brand color check passed"
                if passed
                else f"Only {on_brand_coverage:.0%} of colors are on-brand "
                     f"(need {min_brand_coverage:.0%})"
            ),
        }
```

### Automated Recoloring to Enforce Brand

```python
def enforce_brand_colors(
    image_path: str,
    brand_colors: list[tuple[int, int, int]],
    output_path: str,
    strength: float = 0.7,
):
    """
    Soft-snap image colors toward the nearest brand color.
    strength: 0.0 = no change, 1.0 = full snap to nearest brand color.
    """
    img = np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
    h, w, _ = img.shape
    pixels = img.reshape(-1, 3)

    tree = KDTree(np.array(brand_colors, dtype=np.float32))
    distances, indices = tree.query(pixels)

    brand_arr = np.array(brand_colors, dtype=np.float32)
    nearest = brand_arr[indices]

    # Blend original with nearest brand color
    result = pixels * (1 - strength) + nearest * strength
    result = np.clip(result, 0, 255).astype(np.uint8).reshape(h, w, 3)
    Image.fromarray(result).save(output_path)
```

---

## 7. SVG Color Replacement

SVG is the ideal format for logo color manipulation because colors are stored as editable text attributes.

### Basic SVG Color Replacement

```python
import re
from lxml import etree

def replace_svg_colors(
    svg_path: str,
    color_map: dict[str, str],
    output_path: str,
):
    """
    Replace colors in an SVG file based on a mapping.
    color_map: {"#ff0000": "#2563EB", "#00ff00": "#F59E0B"}
    Handles fill, stroke, style attributes, and inline CSS.
    """
    tree = etree.parse(svg_path)
    root = tree.getroot()

    # Normalize color map keys to lowercase
    color_map = {k.lower(): v for k, v in color_map.items()}

    # Namespaces
    nsmap = {"svg": "http://www.w3.org/2000/svg"}

    def normalize_color(color: str) -> str:
        """Normalize color representations for matching."""
        color = color.strip().lower()
        # Expand 3-digit hex: #abc -> #aabbcc
        if re.match(r"^#[0-9a-f]{3}$", color):
            color = "#" + "".join(c * 2 for c in color[1:])
        # Convert rgb(r,g,b) to hex
        rgb_match = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", color)
        if rgb_match:
            r, g, b = (int(x) for x in rgb_match.groups())
            color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        return color

    def replace_in_attr(element, attr_name):
        """Replace a color in a direct attribute (fill, stroke)."""
        val = element.get(attr_name)
        if val:
            normalized = normalize_color(val)
            if normalized in color_map:
                element.set(attr_name, color_map[normalized])

    def replace_in_style(element):
        """Replace colors within inline style attributes."""
        style = element.get("style")
        if not style:
            return
        for old_color, new_color in color_map.items():
            # Match color values in style properties
            style = re.sub(
                re.escape(old_color),
                new_color,
                style,
                flags=re.IGNORECASE,
            )
        element.set("style", style)

    # Walk all elements
    for elem in root.iter():
        replace_in_attr(elem, "fill")
        replace_in_attr(elem, "stroke")
        replace_in_attr(elem, "stop-color")  # for gradients
        replace_in_attr(elem, "flood-color")  # for filters
        replace_in_style(elem)

    # Also handle <style> blocks (embedded CSS)
    for style_elem in root.iter("{http://www.w3.org/2000/svg}style"):
        if style_elem.text:
            text = style_elem.text
            for old_color, new_color in color_map.items():
                text = re.sub(re.escape(old_color), new_color, text, flags=re.IGNORECASE)
            style_elem.text = text

    tree.write(output_path, xml_declaration=True, encoding="utf-8", pretty_print=True)
```

### Extracting Colors from SVG

```python
def extract_svg_colors(svg_path: str) -> dict[str, int]:
    """
    Extract all unique colors from an SVG and their usage count.
    Returns {"#aabbcc": 5, "#112233": 2, ...}
    """
    tree = etree.parse(svg_path)
    root = tree.getroot()

    color_pattern = re.compile(
        r"(#[0-9a-fA-F]{3,6}|rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\))"
    )
    color_counts: dict[str, int] = {}

    def count_color(raw: str):
        normalized = raw.strip().lower()
        if re.match(r"^#[0-9a-f]{3}$", normalized):
            normalized = "#" + "".join(c * 2 for c in normalized[1:])
        rgb_match = re.match(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", normalized)
        if rgb_match:
            r, g, b = (int(x) for x in rgb_match.groups())
            normalized = "#{:02x}{:02x}{:02x}".format(r, g, b)
        color_counts[normalized] = color_counts.get(normalized, 0) + 1

    for elem in root.iter():
        for attr in ("fill", "stroke", "stop-color", "flood-color"):
            val = elem.get(attr)
            if val and val.lower() not in ("none", "transparent", "inherit", "currentcolor"):
                count_color(val)

        style = elem.get("style", "")
        for match in color_pattern.finditer(style):
            count_color(match.group(1))

    # Check <style> blocks
    for style_elem in root.iter("{http://www.w3.org/2000/svg}style"):
        if style_elem.text:
            for match in color_pattern.finditer(style_elem.text):
                count_color(match.group(1))

    return dict(sorted(color_counts.items(), key=lambda x: -x[1]))
```

### Automated SVG Rebranding

```python
def rebrand_svg(
    svg_path: str,
    brand_colors: dict[str, str],
    output_path: str,
):
    """
    Automatically map existing SVG colors to brand colors.
    Assigns brand roles (primary, secondary, accent) to SVG colors
    based on usage frequency and lightness.
    """
    existing_colors = extract_svg_colors(svg_path)
    if not existing_colors:
        return

    sorted_colors = sorted(existing_colors.items(), key=lambda x: -x[1])

    # Build mapping: most-used color -> primary, next -> secondary, etc.
    role_order = ["primary", "secondary", "accent", "dark", "light"]
    color_map = {}

    for i, (svg_color, _count) in enumerate(sorted_colors):
        if i < len(role_order) and role_order[i] in brand_colors:
            color_map[svg_color] = brand_colors[role_order[i]]

    replace_svg_colors(svg_path, color_map, output_path)
```

---

## 8. Color Accessibility Checking

### WCAG Contrast Ratio

```python
def relative_luminance(hex_color: str) -> float:
    """
    Calculate relative luminance per WCAG 2.1.
    https://www.w3.org/TR/WCAG21/#dfn-relative-luminance
    """
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = linearize(r), linearize(g), linearize(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def contrast_ratio(hex1: str, hex2: str) -> float:
    """
    WCAG contrast ratio between two colors.
    Ranges from 1:1 (identical) to 21:1 (black vs white).

    WCAG requirements:
      - AA normal text:  >= 4.5:1
      - AA large text:   >= 3.0:1
      - AAA normal text: >= 7.0:1
      - AAA large text:  >= 4.5:1
      - Non-text (UI):   >= 3.0:1
    """
    l1 = relative_luminance(hex1)
    l2 = relative_luminance(hex2)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

def check_wcag_compliance(
    foreground: str,
    background: str,
) -> dict:
    """Check WCAG compliance for a color pair."""
    ratio = contrast_ratio(foreground, background)
    return {
        "contrast_ratio": round(ratio, 2),
        "aa_normal_text": ratio >= 4.5,
        "aa_large_text": ratio >= 3.0,
        "aaa_normal_text": ratio >= 7.0,
        "aaa_large_text": ratio >= 4.5,
        "non_text_ui": ratio >= 3.0,
    }
```

### Colorblind Simulation

```python
# Simulation matrices for common color vision deficiencies
# Based on Machado et al. (2009) model at full severity
COLORBLIND_MATRICES = {
    "protanopia": np.array([     # No red cones (~1% of males)
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281,  0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    "deuteranopia": np.array([   # No green cones (~1% of males)
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501,  0.047413],
        [-0.011820, 0.042940, 0.968881],
    ]),
    "tritanopia": np.array([     # No blue cones (~0.003% of population)
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367,  0.303900],
    ]),
}

def simulate_colorblind(hex_color: str, deficiency: str) -> str:
    """Simulate how a color appears to someone with a color vision deficiency."""
    hex_color = hex_color.lstrip("#")
    rgb = np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float64) / 255.0

    matrix = COLORBLIND_MATRICES[deficiency]
    result = np.clip(matrix @ rgb, 0, 1)

    r, g, b = (int(x * 255) for x in result)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def is_palette_colorblind_safe(
    palette: list[str],
    min_distance: float = 10.0,
) -> dict:
    """
    Check if all colors in a palette are distinguishable
    under all common color vision deficiencies.
    """
    issues = []

    for deficiency in COLORBLIND_MATRICES:
        simulated = [simulate_colorblind(c, deficiency) for c in palette]

        for i in range(len(simulated)):
            for j in range(i + 1, len(simulated)):
                dist = color_distance(simulated[i], simulated[j])
                if dist < min_distance:
                    issues.append({
                        "deficiency": deficiency,
                        "color_a": palette[i],
                        "color_b": palette[j],
                        "simulated_a": simulated[i],
                        "simulated_b": simulated[j],
                        "delta_e": round(dist, 2),
                    })

    return {
        "is_safe": len(issues) == 0,
        "issues": issues,
    }
```

### Full Accessibility Report for a Palette

```python
def palette_accessibility_report(
    palette: dict[str, str],
    background: str = "#ffffff",
) -> dict:
    """
    Generate a full accessibility report for a brand palette.
    palette: {"primary": "#2563EB", "secondary": "#F59E0B", ...}
    """
    hex_list = list(palette.values())

    # Contrast checks against background
    contrast_results = {}
    for name, color in palette.items():
        contrast_results[name] = check_wcag_compliance(color, background)

    # Colorblind safety
    cb_safety = is_palette_colorblind_safe(hex_list, min_distance=8.0)

    # Find best text colors for each palette color as background
    text_recommendations = {}
    for name, bg_color in palette.items():
        white_ratio = contrast_ratio("#ffffff", bg_color)
        black_ratio = contrast_ratio("#000000", bg_color)
        text_recommendations[name] = {
            "recommended_text": "#ffffff" if white_ratio > black_ratio else "#000000",
            "contrast_ratio": round(max(white_ratio, black_ratio), 2),
        }

    return {
        "contrast_vs_background": contrast_results,
        "colorblind_safety": cb_safety,
        "text_on_color": text_recommendations,
    }
```

---

## 9. Python Code Examples for Color Management

### Unified Color Utility Class

```python
"""
color_manager.py -- Unified color management for logo generation pipeline.
"""
import colorsys
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial import KDTree


@dataclass
class Color:
    """Represents a single color with conversion utilities."""
    r: int
    g: int
    b: int

    @classmethod
    def from_hex(cls, hex_str: str) -> "Color":
        hex_str = hex_str.lstrip("#")
        if len(hex_str) == 3:
            hex_str = "".join(c * 2 for c in hex_str)
        return cls(
            r=int(hex_str[0:2], 16),
            g=int(hex_str[2:4], 16),
            b=int(hex_str[4:6], 16),
        )

    @property
    def hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(self.r, self.g, self.b)

    @property
    def rgb(self) -> tuple[int, int, int]:
        return (self.r, self.g, self.b)

    @property
    def hsl(self) -> tuple[float, float, float]:
        r, g, b = self.r / 255.0, self.g / 255.0, self.b / 255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return (h * 360, s, l)

    @property
    def luminance(self) -> float:
        def lin(c):
            c = c / 255.0
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return 0.2126 * lin(self.r) + 0.7152 * lin(self.g) + 0.0722 * lin(self.b)

    def contrast_with(self, other: "Color") -> float:
        l1, l2 = self.luminance, other.luminance
        lighter, darker = max(l1, l2), min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)

    def distance_to(self, other: "Color") -> float:
        """Euclidean distance in RGB space (quick approximation)."""
        return float(np.sqrt(
            (self.r - other.r) ** 2 +
            (self.g - other.g) ** 2 +
            (self.b - other.b) ** 2
        ))


@dataclass
class BrandPalette:
    """A complete brand color palette with validation and manipulation."""
    primary: Color
    secondary: Color
    accent: Color
    dark: Color = field(default_factory=lambda: Color(31, 41, 55))
    light: Color = field(default_factory=lambda: Color(249, 250, 251))

    @classmethod
    def from_hex_dict(cls, colors: dict[str, str]) -> "BrandPalette":
        return cls(**{k: Color.from_hex(v) for k, v in colors.items()})

    def to_hex_dict(self) -> dict[str, str]:
        return {
            "primary": self.primary.hex,
            "secondary": self.secondary.hex,
            "accent": self.accent.hex,
            "dark": self.dark.hex,
            "light": self.light.hex,
        }

    def all_colors(self) -> list[Color]:
        return [self.primary, self.secondary, self.accent, self.dark, self.light]

    def check_accessibility(self) -> dict:
        """Run accessibility checks on the palette."""
        results = {}
        bg = self.light

        for name in ("primary", "secondary", "accent", "dark"):
            color = getattr(self, name)
            ratio = color.contrast_with(bg)
            results[name] = {
                "hex": color.hex,
                "contrast_vs_light_bg": round(ratio, 2),
                "aa_text": ratio >= 4.5,
                "aa_large": ratio >= 3.0,
            }

        return results

    def apply_to_svg(self, svg_path: str, output_path: str):
        """Apply this palette to an SVG, mapping by usage frequency."""
        rebrand_svg(svg_path, self.to_hex_dict(), output_path)

    def validate_image(self, image_path: str, tolerance: float = 5.0) -> dict:
        """Check if an image conforms to this palette."""
        validator = BrandColorValidator(self.to_hex_dict(), tolerance=tolerance)
        return validator.validate_image(image_path)


class PaletteGenerator:
    """Generate palettes using various strategies."""

    @staticmethod
    def from_harmony(base_hex: str, harmony: str = "triadic") -> BrandPalette:
        """Generate a palette from a base color and harmony rule."""
        palette = build_logo_palette(base_hex, harmony=harmony, include_neutrals=True)
        return BrandPalette.from_hex_dict({
            "primary": palette["primary"],
            "secondary": palette["secondary"],
            "accent": palette["accent"],
            "dark": palette.get("dark", "#1f2937"),
            "light": palette.get("light", "#f9fafb"),
        })

    @staticmethod
    def from_image(image_path: str, n_colors: int = 3) -> BrandPalette:
        """Extract a palette from a reference image."""
        colors_with_props = extract_palette_kmeans(image_path, n_colors=n_colors)
        hexes = [
            "#{:02x}{:02x}{:02x}".format(*rgb)
            for rgb, _ in colors_with_props
        ]
        while len(hexes) < 3:
            hexes.append(hexes[-1])

        return BrandPalette.from_hex_dict({
            "primary": hexes[0],
            "secondary": hexes[1],
            "accent": hexes[2],
        })

    @staticmethod
    def from_keywords(keywords: list[str]) -> BrandPalette:
        """Generate palette from brand attribute keywords."""
        hex_list = palette_from_attributes(keywords)
        while len(hex_list) < 3:
            hex_list.append("#6B7280")

        return BrandPalette.from_hex_dict({
            "primary": hex_list[0],
            "secondary": hex_list[1],
            "accent": hex_list[2],
        })
```

### Integration Example: Full Pipeline

```python
def logo_color_pipeline(
    generated_image_path: str,
    brand_description: str,
    brand_colors: dict[str, str] | None = None,
    output_dir: str = "output",
):
    """
    Full pipeline: validate generated logo colors, recolor if needed,
    produce accessibility report.
    """
    output = Path(output_dir)
    output.mkdir(exist_ok=True)

    # Step 1: Establish palette
    if brand_colors:
        palette = BrandPalette.from_hex_dict(brand_colors)
    else:
        # Generate palette from description via LLM
        palette_data = generate_brand_palette(brand_description)
        hex_dict = {}
        for entry in palette_data["palette"]:
            role = entry["role"]
            if role in ("primary", "secondary", "accent", "neutral", "background"):
                key = {"neutral": "dark", "background": "light"}.get(role, role)
                hex_dict[key] = entry["hex"]
        palette = BrandPalette.from_hex_dict(hex_dict)

    # Step 2: Validate generated image against palette
    validation = palette.validate_image(generated_image_path, tolerance=8.0)

    # Step 3: If off-brand, apply recoloring
    if not validation["passed"]:
        enforce_brand_colors(
            generated_image_path,
            [c.rgb for c in palette.all_colors()],
            str(output / "recolored.png"),
            strength=0.5,
        )

    # Step 4: Accessibility report
    accessibility = palette.check_accessibility()
    colorblind = is_palette_colorblind_safe([c.hex for c in palette.all_colors()])

    report = {
        "palette": palette.to_hex_dict(),
        "validation": validation,
        "accessibility": accessibility,
        "colorblind_safety": colorblind,
    }

    with open(output / "color_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
```

---

## Key Libraries Summary

| Library | Install | Purpose |
|---------|---------|---------|
| `colorthief` | `pip install colorthief` | Quick palette extraction |
| `scikit-learn` | `pip install scikit-learn` | K-Means clustering |
| `scikit-image` | `pip install scikit-image` | Delta-E color distance (CIEDE2000) |
| `opencv-python` | `pip install opencv-python` | Color space conversions, image manipulation |
| `Pillow` | `pip install Pillow` | Image I/O, basic processing |
| `lxml` | `pip install lxml` | SVG XML parsing and modification |
| `extcolors` | `pip install extcolors` | Tolerance-based color extraction |
| `scipy` | `pip install scipy` | KDTree for nearest-color queries |
| `colour-science` | `pip install colour-science` | Advanced colorimetry (optional) |
| `anthropic` | `pip install anthropic` | LLM-based palette generation |

## Recommendations for Logo Generation Pipeline

1. **Generate with loose color guidance** via prompts, then **recolor precisely** in post-processing
2. **Prefer SVG output** when possible -- color replacement is exact and lossless
3. **Always validate accessibility** before finalizing: check contrast ratios and colorblind safety
4. **Use LAB/CIEDE2000** for perceptual color comparisons, not RGB Euclidean distance
5. **Store brand palettes as structured data** (JSON with roles) not just hex lists
6. **Build a feedback loop**: extract colors from output, compare to brand spec, auto-correct
7. **Provide multiple harmony options** to users starting from their primary brand color
8. **Test with all three CVD types** (protanopia, deuteranopia, tritanopia) before deployment
