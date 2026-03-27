# Image-to-Logo and Reference-Based Generation

## March 2026 - Research Notes

---

## 1. IP-Adapter for Flux: Loading, Usage, and Weight Tuning

### What is IP-Adapter?

IP-Adapter (Image Prompt Adapter) is a lightweight adapter (~100MB) that enables text-to-image diffusion models to accept image prompts alongside text. It works by adding new cross-attention layers that process image features extracted by an image encoder (CLIP or SigLIP), while keeping the original model's text cross-attention layers frozen. This decoupled design allows fine-grained control over how much the image reference influences generation.

### FLUX.1-dev IP-Adapter (InstantX)

The InstantX team released the first IP-Adapter for FLUX.1-dev. Architecture details:

- **Image encoder**: `google/siglip-so400m-patch14-384` (SigLIP, not CLIP)
- **Projection**: MLPProjModel with 2 linear layers
- **Image token count**: 128 tokens
- **Integration points**: New layers added to all 38 single blocks and 19 double blocks
- **Training**: 10M open-source images, batch size 128, 80K training steps

### Loading and Usage (InstantX Native Code)

The InstantX IP-Adapter is not yet fully integrated into the HuggingFace diffusers library. It uses custom pipeline files from the model repository:

```python
import torch
from PIL import Image

# Custom imports from InstantX repository
from pipeline_flux_ipa import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from attention_processor import IPAFluxAttnProcessor2_0
from transformers import AutoProcessor, SiglipVisionModel
from infer_flux_ipa_siglip import resize_img, MLPProjModel, IPAdapter

# Configuration
image_encoder_path = "google/siglip-so400m-patch14-384"
ipadapter_path = "./ip-adapter.bin"

# Load transformer
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16
)

# Initialize pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16
)

# Initialize IP-Adapter
ip_model = IPAdapter(
    pipe,
    image_encoder_path,
    ipadapter_path,
    device="cuda",
    num_tokens=128
)

# Load reference image
ref_image = Image.open("reference_logo.png").convert("RGB")
ref_image = resize_img(ref_image)

# Generate with image reference
images = ip_model.generate(
    pil_image=ref_image,
    prompt="a minimalist tech company logo, clean vector style",
    scale=0.7,          # IP-Adapter influence strength
    width=1024,
    height=1024,
    seed=42
)
images[0].save("generated_logo.png")
```

### IP-Adapter with Diffusers (SDXL - Fully Integrated)

For production use today, the SDXL IP-Adapter has full diffusers integration:

```python
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)

# Set influence scale (0.0 = text only, 1.0 = image only)
pipeline.set_ip_adapter_scale(0.6)

# Load reference logo
reference = load_image("existing_brand_logo.png")

image = pipeline(
    prompt="modern technology company logo, geometric shapes, blue gradient",
    ip_adapter_image=reference,
    negative_prompt="blurry, low quality, complex, photorealistic",
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]
```

### Weight Tuning Strategy for Logos

| Scale Value | Effect | Logo Use Case |
|-------------|--------|---------------|
| 0.2-0.3 | Subtle style influence | Borrow color palette only |
| 0.4-0.5 | Balanced text/image | Inspired-by generation |
| 0.6-0.7 | Strong image influence | Style transfer from reference |
| 0.8-1.0 | Image-dominant | Near-reproduction with tweaks |

**Logo-specific recommendations:**
- Start at `scale=0.5` and adjust up/down
- For style transfer (keep colors/feel, change shape): use 0.3-0.5
- For variations (similar logo, minor changes): use 0.7-0.9
- Flux IP-Adapter needs **lower strength** than SDXL to avoid over-adherence

### InstantStyle: Separating Style from Content

InstantStyle is a technique that applies IP-Adapter influence only to specific model layers, separating style (color, texture) from layout/content:

```python
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)

# Style-only injection (colors, textures, artistic feel)
scale = {
    "up": {"block_0": [0.0, 1.0, 0.0]},  # style layer only
}
pipeline.set_ip_adapter_scale(scale)

# Layout + style injection
scale = {
    "down": {"block_2": [0.0, 1.0]},   # layout information
    "up": {"block_0": [0.0, 1.0, 0.0]}, # style information
}
pipeline.set_ip_adapter_scale(scale)
```

This is particularly useful for logos: extract the **style** (brand colors, visual language) from existing brand assets while generating entirely new logo layouts.

### Pre-computing Image Embeddings

For batch generation with the same reference image, pre-compute embeddings:

```python
image_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=reference_image,
    ip_adapter_image_embeds=None,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)

# Save for later reuse
torch.save(image_embeds, "brand_reference_embeds.ipadpt")

# Load and use without image encoder
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    image_encoder_folder=None,  # skip loading encoder
    weight_name="ip-adapter_sdxl.bin"
)
image_embeds = torch.load("brand_reference_embeds.ipadpt")
pipeline(
    prompt="logo design",
    ip_adapter_image_embeds=image_embeds,
).images[0]
```

---

## 2. Img2Img with Controlled Denoising Strength

### How Denoising Strength Works

The `strength` parameter controls how much the input image is preserved vs. regenerated:

- **strength=0.0**: Output is identical to input (no denoising)
- **strength=0.5**: 50% of noise steps applied; significant transformation while retaining structure
- **strength=1.0**: Maximum noise; effectively ignores the input image

Mathematically: if `num_inference_steps=50` and `strength=0.6`, the pipeline adds 30 steps of noise to the input and then denoises for 30 steps.

### FluxImg2ImgPipeline

```python
import torch
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image

pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Load existing logo or sketch
init_image = load_image("rough_logo_sketch.png").resize((1024, 1024))

prompt = "professional minimalist logo, clean vector art, single color, centered composition"

# Moderate transformation - refine the sketch
refined = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.6,           # preserve general structure
    num_inference_steps=28,
    guidance_scale=7.0,
).images[0]
refined.save("refined_logo.png")
```

### Strength Tuning Guide for Logos

```python
# Minimal touch-up (clean edges, smooth colors)
pipe(prompt=prompt, image=image, strength=0.2, num_inference_steps=28)

# Moderate refinement (improve quality, add detail)
pipe(prompt=prompt, image=image, strength=0.4, num_inference_steps=28)

# Significant rework (new details, style change)
pipe(prompt=prompt, image=image, strength=0.6, num_inference_steps=28)

# Major transformation (sketch to polished logo)
pipe(prompt=prompt, image=image, strength=0.8, num_inference_steps=28)

# Near-complete regeneration (use structure as loose guide)
pipe(prompt=prompt, image=image, strength=0.95, num_inference_steps=28)
```

**Logo-specific strength recommendations:**

| Input Type | Recommended Strength | Rationale |
|-----------|---------------------|-----------|
| Polished logo needing color change | 0.2-0.3 | Preserve shape, change palette |
| Rough vector needing refinement | 0.4-0.5 | Clean up while keeping layout |
| Hand sketch to digital logo | 0.6-0.8 | Major transformation needed |
| Photo/icon to logo style | 0.7-0.9 | Style conversion required |

### FLUX.1 Schnell for Fast Iteration

For rapid prototyping with img2img:

```python
pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
).to("cuda")

# Fast 4-step generation for quick iterations
image = pipe(
    prompt="clean vector logo",
    image=init_image,
    strength=0.95,
    num_inference_steps=4,
    guidance_scale=0.0,  # schnell works best with guidance_scale=0
).images[0]
```

---

## 3. ControlNet + Reference Image Workflows

### Flux ControlNet for Logo Structure Control

ControlNet enables generating images that follow the structural guide of an input image (edges, depth, lines) while allowing creative freedom in style and detail.

#### Canny Edge Control (Preserve Logo Outlines)

```python
import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.utils import load_image

# Load ControlNet for canny edges
controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny",
    torch_dtype=torch.bfloat16
)
pipeline = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
).to("cuda")

# Generate canny edge map from existing logo
original = load_image("existing_logo.png")
image_array = np.array(original)
canny = cv2.Canny(image_array, 100, 200)
canny = np.stack([canny] * 3, axis=-1)
canny_image = Image.fromarray(canny)

# Generate new logo following the same structure
result = pipeline(
    prompt="modern gradient logo, vibrant colors, professional design",
    control_image=canny_image,
    controlnet_conditioning_scale=0.5,  # 0.0-1.0, how strictly to follow edges
    control_guidance_start=0.0,         # when to start applying control
    control_guidance_end=0.8,           # when to stop (allows creative freedom at end)
    num_inference_steps=50,
    guidance_scale=3.5,
).images[0]
```

#### ControlNet Conditioning Scale for Logos

| Scale | Effect | Use Case |
|-------|--------|----------|
| 0.3 | Loose structural guidance | Inspired-by redesign |
| 0.5 | Balanced | Style transfer with structure |
| 0.7 | Strong structural adherence | Restyle same layout |
| 1.0 | Strict following | Colorize/texture existing outline |

#### Lineart Control (Best for Sketches)

```python
# Using lineart ControlNet for sketch-to-logo
controlnet = FluxControlNetModel.from_pretrained(
    "promeai/FLUX.1-controlnet-lineart-promeai",
    torch_dtype=torch.bfloat16
)
pipeline = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
).to("cuda")

sketch = load_image("hand_drawn_sketch.png")
result = pipeline(
    prompt="professional logo design, clean vector, minimal",
    control_image=sketch,
    controlnet_conditioning_scale=0.7,
    num_inference_steps=50,
).images[0]
```

#### ControlNet Union (Multiple Controls)

The ControlNet Union model supports multiple control modes in a single model:

```python
# Using FLUX.1-dev-ControlNet-Union for multi-modal control
from diffusers import FluxControlNetModel

controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Union",
    torch_dtype=torch.bfloat16
)

# Supports: canny, depth, pose, tile, blur, lineart, etc.
# Mode is selected by the type of control_image provided
```

#### Combining ControlNet + IP-Adapter

The most powerful approach for reference-based logo generation combines structural control (ControlNet) with style control (IP-Adapter):

```python
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

# Load ControlNet for structure
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)

pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load IP-Adapter for style
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin"
)
pipeline.set_ip_adapter_scale(0.5)

# Structure from sketch, style from reference
canny_image = load_image("logo_sketch_edges.png")       # structural guide
style_reference = load_image("brand_style_reference.png") # style guide

result = pipeline(
    prompt="professional logo design",
    image=canny_image,                     # ControlNet input
    ip_adapter_image=style_reference,      # IP-Adapter input
    controlnet_conditioning_scale=0.6,
    num_inference_steps=50,
).images[0]
```

---

## 4. Sketch-to-Logo Pipeline Design

### Architecture Overview

A robust sketch-to-logo pipeline combines multiple stages:

```
[User Sketch] --> [Preprocessing] --> [Edge Extraction] --> [ControlNet Generation]
                                                                    |
[Brand Kit / Style Ref] --> [IP-Adapter / Style Encoding] ----------+
                                                                    |
[Text Prompt] --> [Prompt Engineering] ----> [Diffusion Model] --> [Refinement]
                                                                    |
                                                              [SVG Conversion]
                                                                    |
                                                              [Final Output]
```

### Complete Sketch-to-Logo Pipeline

```python
import cv2
import torch
import numpy as np
from PIL import Image, ImageFilter
from diffusers import FluxControlNetPipeline, FluxControlNetModel, FluxImg2ImgPipeline
from diffusers.utils import load_image


class SketchToLogoPipeline:
    """Multi-stage pipeline: sketch --> clean edges --> controlled generation --> refinement."""

    def __init__(self, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self._load_models()

    def _load_models(self):
        """Load ControlNet + base pipeline."""
        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny",
            torch_dtype=self.dtype
        )
        self.controlnet_pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            controlnet=controlnet,
            torch_dtype=self.dtype
        ).to(self.device)

        self.img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=self.dtype
        ).to(self.device)

    def preprocess_sketch(
        self,
        sketch_path: str,
        target_size: int = 1024,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> Image.Image:
        """Clean up a hand-drawn sketch and extract edges."""
        img = Image.open(sketch_path).convert("L")

        # Resize to square, maintaining aspect ratio with padding
        img = img.resize((target_size, target_size), Image.LANCZOS)

        # Convert to numpy for OpenCV processing
        arr = np.array(img)

        # Denoise the sketch
        arr = cv2.GaussianBlur(arr, (3, 3), 0)

        # Threshold to clean up pencil marks
        _, arr = cv2.threshold(arr, 200, 255, cv2.THRESH_BINARY)

        # Extract canny edges
        edges = cv2.Canny(arr, canny_low, canny_high)
        edges = np.stack([edges] * 3, axis=-1)

        return Image.fromarray(edges)

    def generate_from_sketch(
        self,
        sketch_path: str,
        prompt: str,
        style: str = "minimalist",
        controlnet_scale: float = 0.6,
        num_steps: int = 50,
        seed: int = 42,
    ) -> list[Image.Image]:
        """Generate logo variations from a sketch."""
        # Stage 1: Preprocess sketch
        edges = self.preprocess_sketch(sketch_path)

        # Build prompt with style modifiers
        style_prompts = {
            "minimalist": "minimalist flat logo, clean vector art, simple shapes, single color",
            "gradient": "modern gradient logo, smooth color transitions, professional",
            "vintage": "vintage retro logo, textured, classic typography, badge style",
            "geometric": "geometric logo, precise shapes, mathematical, abstract",
            "hand_drawn": "hand-lettered logo, organic shapes, artisanal, craft style",
        }
        style_suffix = style_prompts.get(style, style_prompts["minimalist"])
        full_prompt = f"{prompt}, {style_suffix}, white background, centered"

        # Stage 2: ControlNet generation (multiple seeds for variations)
        results = []
        for i in range(4):
            generator = torch.Generator(self.device).manual_seed(seed + i)
            image = self.controlnet_pipe(
                prompt=full_prompt,
                control_image=edges,
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=num_steps,
                guidance_scale=3.5,
                generator=generator,
            ).images[0]
            results.append(image)

        return results

    def refine_logo(
        self,
        logo_image: Image.Image,
        prompt: str,
        strength: float = 0.3,
    ) -> Image.Image:
        """Refine a generated logo with img2img for cleanup."""
        refined = self.img2img_pipe(
            prompt=f"{prompt}, clean vector logo, sharp edges, professional",
            image=logo_image,
            strength=strength,
            num_inference_steps=28,
            guidance_scale=7.0,
        ).images[0]
        return refined


# Usage
pipeline = SketchToLogoPipeline()
logos = pipeline.generate_from_sketch(
    sketch_path="my_sketch.png",
    prompt="tech startup logo for 'NovaByte'",
    style="geometric",
    controlnet_scale=0.6,
)
# Refine the best candidate
final = pipeline.refine_logo(logos[0], "NovaByte tech startup logo")
final.save("final_logo.png")
```

### Sketch Input Requirements

For best results with sketch-to-logo:
- **Format**: PNG or JPG
- **Resolution**: At least 1000x1000 pixels (will be resized to model input size)
- **Background**: Clean white background preferred
- **Lines**: Dark, clear lines (pencil sketches may need thresholding)
- **Content**: Keep sketch simple; avoid shading or hatching

---

## 5. Brand Kit Reference Approach

### Extracting Brand Colors from Existing Assets

```python
from colorthief import ColorThief
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


def extract_dominant_colors_colorthief(image_path: str, n_colors: int = 5) -> list[tuple]:
    """Extract dominant colors using ColorThief (median cut algorithm)."""
    ct = ColorThief(image_path)
    palette = ct.get_palette(color_count=n_colors, quality=1)
    return palette  # list of (R, G, B) tuples


def extract_dominant_colors_kmeans(image_path: str, n_colors: int = 5) -> list[dict]:
    """Extract dominant colors using K-Means clustering with proportions."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((200, 200))  # downsample for speed
    pixels = np.array(img).reshape(-1, 3)

    # Remove near-white and near-black pixels (background)
    mask = (pixels.sum(axis=1) > 30) & (pixels.sum(axis=1) < 720)
    pixels = pixels[mask]

    if len(pixels) == 0:
        return []

    kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)

    # Calculate proportions
    counts = Counter(labels)
    total = sum(counts.values())

    colors = []
    for i, center in enumerate(kmeans.cluster_centers_):
        r, g, b = int(center[0]), int(center[1]), int(center[2])
        proportion = counts[i] / total
        colors.append({
            "rgb": (r, g, b),
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "proportion": round(proportion, 3),
        })

    # Sort by proportion (most dominant first)
    colors.sort(key=lambda x: x["proportion"], reverse=True)
    return colors


def rgb_to_hex(rgb: tuple) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


# Usage
colors = extract_dominant_colors_kmeans("existing_logo.png", n_colors=5)
for c in colors:
    print(f"  {c['hex']} - {c['proportion']*100:.1f}%")

# Convert to prompt-friendly format
color_str = ", ".join([c["hex"] for c in colors[:3]])
prompt = f"logo design using colors {color_str}"
```

### Full Brand Kit Extraction

```python
import json
from PIL import Image
from colorthief import ColorThief


class BrandKitExtractor:
    """Extract brand identity elements from existing logo assets."""

    def __init__(self, logo_path: str):
        self.logo_path = logo_path
        self.image = Image.open(logo_path).convert("RGBA")

    def extract_colors(self, n_colors: int = 5) -> list[dict]:
        """Extract dominant colors excluding transparency."""
        # Save as RGB temp for ColorThief
        rgb = self.image.convert("RGB")
        temp_path = "/tmp/brand_temp.png"
        rgb.save(temp_path)

        ct = ColorThief(temp_path)
        dominant = ct.get_color(quality=1)
        palette = ct.get_palette(color_count=n_colors, quality=1)

        return {
            "primary": rgb_to_hex(dominant),
            "palette": [rgb_to_hex(c) for c in palette],
        }

    def extract_dimensions(self) -> dict:
        """Analyze logo proportions and aspect ratio."""
        w, h = self.image.size
        return {
            "width": w,
            "height": h,
            "aspect_ratio": round(w / h, 2),
            "is_square": abs(w - h) < max(w, h) * 0.1,
        }

    def extract_complexity(self) -> dict:
        """Estimate visual complexity via edge density."""
        import cv2
        import numpy as np

        gray = np.array(self.image.convert("L"))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size

        return {
            "edge_density": round(edge_density, 4),
            "complexity": (
                "simple" if edge_density < 0.02
                else "moderate" if edge_density < 0.05
                else "complex"
            ),
        }

    def extract_transparency(self) -> dict:
        """Check if logo uses transparency."""
        if self.image.mode != "RGBA":
            return {"has_transparency": False, "transparent_ratio": 0.0}

        alpha = np.array(self.image.split()[-1])
        transparent_pixels = np.count_nonzero(alpha < 128)
        total = alpha.size

        return {
            "has_transparency": transparent_pixels > 0,
            "transparent_ratio": round(transparent_pixels / total, 3),
        }

    def to_brand_kit(self) -> dict:
        """Generate complete brand kit analysis."""
        return {
            "source": self.logo_path,
            "colors": self.extract_colors(),
            "dimensions": self.extract_dimensions(),
            "complexity": self.extract_complexity(),
            "transparency": self.extract_transparency(),
        }

    def to_prompt_context(self) -> str:
        """Convert brand kit to prompt context string."""
        kit = self.to_brand_kit()
        colors = kit["colors"]
        complexity = kit["complexity"]["complexity"]

        prompt_parts = [
            f"brand colors: {', '.join(colors['palette'][:3])}",
            f"primary color: {colors['primary']}",
            f"style: {complexity} design",
        ]
        if kit["dimensions"]["is_square"]:
            prompt_parts.append("square format")

        return ", ".join(prompt_parts)


# Usage
extractor = BrandKitExtractor("company_logo.png")
kit = extractor.to_brand_kit()
print(json.dumps(kit, indent=2))

# Use in generation prompt
context = extractor.to_prompt_context()
prompt = f"new logo variation, {context}, modern refresh"
```

---

## 6. Vision Model Analysis of Reference Images

### Using Claude Vision to Analyze and Re-describe Logos

```python
import anthropic
import base64


def analyze_logo_with_claude(image_path: str) -> dict:
    """Use Claude vision to analyze a logo and generate structured description."""
    client = anthropic.Anthropic()

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type
    ext = image_path.lower().rsplit(".", 1)[-1]
    media_types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
    media_type = media_types.get(ext, "image/png")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Analyze this logo image and provide a structured description. Return a JSON object with:

{
  "description": "One-sentence description of the logo",
  "style": "e.g., minimalist, vintage, geometric, hand-drawn, gradient, 3D",
  "shapes": ["list of geometric shapes used"],
  "colors": {
    "primary": "#hex",
    "secondary": "#hex",
    "accent": "#hex or null",
    "background": "#hex"
  },
  "typography": {
    "has_text": true/false,
    "text_content": "visible text or null",
    "font_style": "e.g., sans-serif, serif, script, display"
  },
  "composition": "e.g., centered, asymmetric, badge/emblem, wordmark, icon+text",
  "mood": "e.g., professional, playful, elegant, bold, tech-forward",
  "similar_to": "brief comparison to well-known logo styles"
}

Return ONLY the JSON, no other text.""",
                    },
                ],
            }
        ],
    )

    import json
    return json.loads(message.content[0].text)


def generate_reproduction_prompt(analysis: dict) -> str:
    """Convert logo analysis into an image generation prompt."""
    parts = [
        f"A {analysis['style']} logo design",
        f"using {', '.join(analysis['shapes'])} shapes" if analysis['shapes'] else "",
        f"in {analysis['colors']['primary']} and {analysis['colors']['secondary']} colors",
        f"with a {analysis['mood']} feel",
        f"{analysis['composition']} composition",
    ]

    if analysis['typography']['has_text']:
        parts.append(
            f"featuring the text '{analysis['typography']['text_content']}' "
            f"in {analysis['typography']['font_style']} font"
        )

    parts.extend([
        "clean vector art",
        "white background",
        "professional quality",
    ])

    return ", ".join([p for p in parts if p])


# Usage
analysis = analyze_logo_with_claude("reference_logo.png")
prompt = generate_reproduction_prompt(analysis)
print(f"Generated prompt: {prompt}")

# Feed prompt into image generation
# pipe(prompt=prompt, ...)
```

### Using Claude Vision via URL

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/logo.png",
                    },
                },
                {
                    "type": "text",
                    "text": "Describe this logo in detail for recreation by an AI image generator."
                },
            ],
        }
    ],
)
print(message.content[0].text)
```

---

## 7. Using LLM Vision to Analyze Logos and Generate Prompts

### Claude Vision: Logo Analysis Pipeline

```python
import anthropic
import base64
import json


class LogoAnalyzer:
    """Use LLM vision models to analyze logos and generate image-gen prompts."""

    def __init__(self, provider: str = "claude"):
        self.provider = provider
        if provider == "claude":
            self.client = anthropic.Anthropic()
        elif provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """Read and base64-encode an image file."""
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        ext = image_path.lower().rsplit(".", 1)[-1]
        media_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}
        return data, media_map.get(ext, "image/png")

    def analyze_for_regeneration(self, image_path: str) -> str:
        """Analyze a logo and produce an optimized prompt for image generation."""
        image_data, media_type = self._encode_image(image_path)

        system_prompt = """You are an expert logo designer and AI image generation prompt engineer.
When shown a logo, you produce detailed, optimized prompts that would recreate
a similar logo using text-to-image AI models like Flux, DALL-E, or Stable Diffusion.

Focus on:
- Exact geometric shapes and their arrangement
- Color values (use hex codes when possible)
- Typography style (if text is present)
- Overall composition and layout
- Visual style (flat, gradient, 3D, etc.)
- Negative space usage
- Level of detail and complexity

Output a single optimized prompt string. Do NOT include the brand name in the prompt
(the user will add that). Focus on visual description only."""

        if self.provider == "claude":
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Analyze this logo and generate an optimized image generation prompt.",
                        },
                    ],
                }],
            )
            return message.content[0].text

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this logo and generate an optimized image generation prompt."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=1024,
            )
            return response.choices[0].message.content

    def compare_logos(self, image_paths: list[str]) -> str:
        """Compare multiple logos and describe their common brand language."""
        if self.provider == "claude":
            content = []
            for i, path in enumerate(image_paths):
                image_data, media_type = self._encode_image(path)
                content.append({"type": "text", "text": f"Logo {i+1}:"})
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": image_data},
                })

            content.append({
                "type": "text",
                "text": """Analyze these logos as a brand family. Describe:
1. Common visual elements (shapes, motifs)
2. Shared color palette (hex codes)
3. Typography consistency
4. Design language / style system
5. An image generation prompt that captures the shared brand identity""",
            })

            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": content}],
            )
            return message.content[0].text

        elif self.provider == "openai":
            content = [
                {"type": "text", "text": "Analyze these logos as a brand family. Describe common elements and generate a unified style prompt."}
            ]
            for path in image_paths:
                image_data, media_type = self._encode_image(path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{image_data}", "detail": "high"},
                })

            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
            )
            return response.choices[0].message.content

    def extract_style_keywords(self, image_path: str) -> list[str]:
        """Extract a list of style keywords from a logo for use as prompt tags."""
        image_data, media_type = self._encode_image(image_path)

        if self.provider == "claude":
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": image_data},
                        },
                        {
                            "type": "text",
                            "text": "List 10-15 style keywords that describe this logo's visual style. Return ONLY a JSON array of strings. Example: [\"minimalist\", \"geometric\", \"flat\"]",
                        },
                    ],
                }],
            )
            return json.loads(message.content[0].text)

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List 10-15 style keywords for this logo as a JSON array of strings."},
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_data}"}},
                    ],
                }],
                max_tokens=256,
            )
            return json.loads(response.choices[0].message.content)


# Usage
analyzer = LogoAnalyzer(provider="claude")

# Single logo analysis
prompt = analyzer.analyze_for_regeneration("company_logo.png")
print(f"Regeneration prompt: {prompt}")

# Style keywords
keywords = analyzer.extract_style_keywords("company_logo.png")
print(f"Style keywords: {keywords}")

# Brand family comparison
brand_prompt = analyzer.compare_logos([
    "logo_main.png",
    "logo_icon.png",
    "logo_wordmark.png",
])
print(f"Brand family analysis:\n{brand_prompt}")
```

### Best Practices for Vision-Based Logo Analysis

1. **Image before text**: Place the image content block before the text prompt in the API call. Claude performs better with this ordering.

2. **Image sizing**: Claude internally resizes images to max 1568px on the longest edge. Send images at that size or smaller to avoid wasted bandwidth. For logos, 1024x1024 is typically sufficient.

3. **Structured output**: Request JSON output for programmatic parsing. Use a clear schema in the prompt.

4. **Cost considerations**:
   - Claude: ~1,334 tokens for a 1000x1000 image (~$0.004 per image at Sonnet pricing)
   - GPT-4.1: ~765 tokens in high detail for 1024x1024 (~$0.008 per image)
   - For batch analysis, pre-compute once and cache results

5. **Multi-image comparison**: Claude supports up to 20 images per request on claude.ai, up to 600 via API. Use this for analyzing brand families.

---

## 8. FLUX.1 Redux: Native Image Variation

### What is FLUX.1 Redux?

FLUX.1 Redux is an official adapter from Black Forest Labs for reference-based image generation. Unlike img2img (which adds noise to the input image and denoises), Redux **encodes** the reference image with SigLIP and injects the embeddings alongside (or instead of) text embeddings. This produces variations that capture the semantic essence of the reference without directly copying pixels.

### Usage with Diffusers

```python
import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from diffusers.utils import load_image

device = "cuda"
dtype = torch.bfloat16

# Load Redux adapter (encodes reference images)
pipe_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev",
    torch_dtype=dtype
).to(device)

# Load base Flux pipeline (skip text encoders to save VRAM)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    text_encoder_2=None,
    torch_dtype=dtype
).to(device)

# Encode reference image
reference = load_image("existing_logo.png")
redux_output = pipe_redux(reference)

# Generate variation
images = pipe(
    guidance_scale=2.5,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(42),
    **redux_output,  # passes prompt_embeds and pooled_prompt_embeds
).images

images[0].save("logo_variation.png")
```

### Redux vs. IP-Adapter vs. Img2Img

| Approach | How It Works | Best For |
|----------|-------------|----------|
| **Img2Img** | Adds noise to input, denoises with prompt | Refining/restyling existing images |
| **IP-Adapter** | Encodes reference with CLIP/SigLIP, injects via cross-attention | Style transfer, combining text+image prompts |
| **Redux** | Encodes reference, replaces text embeddings entirely | Pure image variations, no text needed |
| **ControlNet** | Extracts structural guide (edges/depth), conditions generation | Preserving layout/structure exactly |

### Combining Redux with Text Prompts

While Redux can work without text, combining it with text guidance gives more control:

```python
# First get Redux embeddings
redux_output = pipe_redux(reference_image)

# Then add text guidance back by loading full pipeline
pipe_with_text = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Use both text prompt and Redux embeddings
# (requires custom embedding merging - not natively supported in all versions)
```

---

## 9. End-to-End Reference-Based Logo Generation Pipeline

### Complete Pipeline Combining All Techniques

```python
import torch
import json
import base64
from pathlib import Path
from PIL import Image
import anthropic
from colorthief import ColorThief
from diffusers import (
    FluxImg2ImgPipeline,
    FluxControlNetPipeline,
    FluxControlNetModel,
    FluxPriorReduxPipeline,
    FluxPipeline,
)
from diffusers.utils import load_image
import cv2
import numpy as np


class ReferenceBasedLogoGenerator:
    """
    Complete pipeline for generating logos based on reference images.

    Workflow:
    1. Analyze reference with Claude vision
    2. Extract brand colors
    3. Generate with ControlNet (structure) or Redux (variation) or img2img (refinement)
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.dtype = torch.bfloat16
        self.claude = anthropic.Anthropic()
        self._models_loaded = False

    def _ensure_models(self):
        if self._models_loaded:
            return
        # Load only what's needed lazily
        self._models_loaded = True

    def analyze_reference(self, image_path: str) -> dict:
        """Step 1: Use Claude to analyze the reference logo."""
        with open(image_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")

        ext = Path(image_path).suffix.lstrip(".")
        media_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}

        message = self.claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_map.get(ext, "image/png"),
                            "data": data,
                        },
                    },
                    {
                        "type": "text",
                        "text": """Analyze this logo and return JSON:
{
  "description": "one-line description",
  "generation_prompt": "optimized prompt for AI image generation to recreate this style",
  "style_tags": ["list", "of", "style", "keywords"],
  "colors": {"primary": "#hex", "secondary": "#hex", "accent": "#hex"},
  "complexity": "simple|moderate|complex",
  "has_text": true/false,
  "text_content": "visible text or null",
  "composition_type": "icon|wordmark|icon+text|emblem|abstract"
}
Return ONLY valid JSON.""",
                    },
                ],
            }],
        )
        return json.loads(message.content[0].text)

    def extract_colors(self, image_path: str) -> list[str]:
        """Step 2: Extract brand colors programmatically."""
        ct = ColorThief(image_path)
        palette = ct.get_palette(color_count=5, quality=1)
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette]

    def generate_variation(
        self,
        reference_path: str,
        prompt_override: str = None,
        method: str = "img2img",  # "img2img", "controlnet", "redux"
        strength: float = 0.5,
        n_variations: int = 4,
        seed: int = 42,
    ) -> list[Image.Image]:
        """Step 3: Generate logo variations using chosen method."""

        # Analyze reference first
        analysis = self.analyze_reference(reference_path)
        colors = self.extract_colors(reference_path)

        prompt = prompt_override or analysis["generation_prompt"]
        prompt += f", brand colors {' '.join(colors[:3])}"

        ref_image = Image.open(reference_path).convert("RGB").resize((1024, 1024))
        results = []

        if method == "img2img":
            pipe = FluxImg2ImgPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=self.dtype,
            ).to(self.device)

            for i in range(n_variations):
                gen = torch.Generator(self.device).manual_seed(seed + i)
                img = pipe(
                    prompt=prompt,
                    image=ref_image,
                    strength=strength,
                    num_inference_steps=28,
                    guidance_scale=7.0,
                    generator=gen,
                ).images[0]
                results.append(img)

        elif method == "controlnet":
            # Extract edges from reference
            arr = np.array(ref_image)
            edges = cv2.Canny(arr, 100, 200)
            edges = np.stack([edges] * 3, axis=-1)
            edge_image = Image.fromarray(edges)

            controlnet = FluxControlNetModel.from_pretrained(
                "InstantX/FLUX.1-dev-Controlnet-Canny",
                torch_dtype=self.dtype,
            )
            pipe = FluxControlNetPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                controlnet=controlnet,
                torch_dtype=self.dtype,
            ).to(self.device)

            for i in range(n_variations):
                gen = torch.Generator(self.device).manual_seed(seed + i)
                img = pipe(
                    prompt=prompt,
                    control_image=edge_image,
                    controlnet_conditioning_scale=strength,
                    num_inference_steps=50,
                    guidance_scale=3.5,
                    generator=gen,
                ).images[0]
                results.append(img)

        elif method == "redux":
            pipe_redux = FluxPriorReduxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Redux-dev",
                torch_dtype=self.dtype,
            ).to(self.device)

            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                text_encoder=None,
                text_encoder_2=None,
                torch_dtype=self.dtype,
            ).to(self.device)

            redux_output = pipe_redux(ref_image)

            for i in range(n_variations):
                gen = torch.Generator("cpu").manual_seed(seed + i)
                img = pipe(
                    guidance_scale=2.5,
                    num_inference_steps=50,
                    generator=gen,
                    **redux_output,
                ).images[0]
                results.append(img)

        return results


# Usage
generator = ReferenceBasedLogoGenerator()

# Analyze a reference logo
analysis = generator.analyze_reference("client_logo.png")
print(json.dumps(analysis, indent=2))

# Generate variations using different methods
img2img_results = generator.generate_variation(
    "client_logo.png",
    method="img2img",
    strength=0.5,
)

controlnet_results = generator.generate_variation(
    "client_logo.png",
    method="controlnet",
    strength=0.6,
)

redux_results = generator.generate_variation(
    "client_logo.png",
    method="redux",
)

# Save results
for i, img in enumerate(img2img_results):
    img.save(f"variation_img2img_{i}.png")
```

---

## 10. Summary: Method Selection Guide

| Scenario | Best Method | Key Parameters |
|----------|------------|----------------|
| Refine existing logo quality | Img2Img | strength=0.2-0.4 |
| Restyle logo (new colors, same shape) | ControlNet (canny) | conditioning_scale=0.7 |
| Generate from hand sketch | ControlNet (lineart) | conditioning_scale=0.5-0.7 |
| Create style variations | IP-Adapter | scale=0.5-0.7 |
| Pure image variations | FLUX Redux | guidance_scale=2.5 |
| Extract style, apply to new design | IP-Adapter + InstantStyle | style layers only |
| Complete brand-aware generation | Vision analysis + any method | Claude/GPT-4o pre-analysis |
| Sketch + brand style combined | ControlNet + IP-Adapter | both conditioning params |

### Required Python Packages

```
pip install diffusers transformers accelerate torch
pip install anthropic openai
pip install opencv-python pillow colorthief scikit-learn
pip install safetensors sentencepiece
```

### VRAM Requirements

| Setup | Minimum VRAM |
|-------|-------------|
| FLUX.1-dev (bf16) | ~24GB |
| FLUX.1-dev + ControlNet | ~32GB |
| FLUX.1-dev + IP-Adapter | ~32GB |
| FLUX.1-schnell (bf16) | ~20GB |
| SDXL + IP-Adapter | ~12GB |
| SDXL + ControlNet | ~12GB |
| Any + CPU offloading | ~8GB (slower) |

Enable memory optimization with:
```python
pipe.enable_model_cpu_offload()   # offload to CPU when not in use
pipe.enable_vae_slicing()          # process VAE in slices
pipe.enable_vae_tiling()           # process VAE in tiles (for large images)
```
