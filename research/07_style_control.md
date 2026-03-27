# Style Control for Logo Generation

Research into LoRAs, ControlNet, IP-Adapter, InstantStyle, and combined techniques for
controlling style in AI-generated logos using the Hugging Face `diffusers` library.

---

## Table of Contents

1. [Logo-Specific LoRAs](#1-logo-specific-loras)
2. [ControlNet for Logos](#2-controlnet-for-logos)
3. [IP-Adapter for Style Transfer](#3-ip-adapter-for-style-transfer)
4. [InstantStyle for Style-Content Separation](#4-instantstyle-for-style-content-separation)
5. [Combining ControlNet + IP-Adapter + LoRA](#5-combining-controlnet--ip-adapter--lora)
6. [Style Consistency Across Multiple Logo Variations](#6-style-consistency-across-multiple-logo-variations)
7. [Summary and Recommendations](#7-summary-and-recommendations)

---

## 1. Logo-Specific LoRAs

LoRA (Low-Rank Adaptation) models are small adapter weights (~50-500 MB) that fine-tune a base
model toward a specific style or concept without retraining the full model. Several LoRAs target
logo generation specifically.

### 1.1 Shakker-Labs FLUX Logo LoRA

**Model**: `Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design`
**Base model**: FLUX.1-dev (black-forest-labs/FLUX.1-dev)
**License**: flux-1-dev-non-commercial-license

**Trigger words**: `wablogo`, `logo`, `Minimalist`

**Recommended settings**:
- LoRA scale: 0.8
- Inference steps: 24
- Guidance scale: 3.5

**Prompt structure tips** (from the model card):
- Dual combination: "something and something" (e.g. "cat and coffee")
- Font combination: "a shape plus a letter" (e.g. "a book with the word 'M'")
- Text below graphic: "Below the graphic is the word 'XX'"

**Example prompts**:
- `logo,Minimalist,A bunch of grapes and a wine glass`
- `logo,Minimalist,A man stands in front of a door,his shadow forming the word "A"`
- `logo,Minimalist,A pair of chopsticks and a bowl of rice with the word "Lee"`
- `wablogo,Minimalist,Leaf and cat,logo`

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights(
    "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design",
    weight_name="FLUX-dev-lora-Logo-Design.safetensors"
)
pipe.fuse_lora(lora_scale=0.8)
pipe.to("cuda")

prompt = "logo,Minimalist,A bunch of grapes and a wine glass"
image = pipe(
    prompt,
    num_inference_steps=24,
    guidance_scale=3.5,
).images[0]
image.save("flux_logo.png")
```

### 1.2 Shakker-Labs FLUX Vector Journey LoRA

**Model**: `Shakker-Labs/FLUX.1-dev-LoRA-Vector-Journey`

Produces vector-style illustrations useful for logo work. Same loading pattern as above
but with different weight file and trigger words.

```python
pipe.load_lora_weights(
    "Shakker-Labs/FLUX.1-dev-LoRA-Vector-Journey",
    weight_name="FLUX-dev-lora-Vector-Journey.safetensors"
)
pipe.fuse_lora(lora_scale=0.8)
```

### 1.3 Harrlogos XL

**Model**: `HarroweD/HarrlogosXL` (available on HuggingFace and CivitAI)
**Base model**: Stable Diffusion XL
**Purpose**: Custom text generation in logos -- renders readable text within graphics.

**Prompt format** (comma-separated terms in order):
1. Text content: `"YOUR TEXT" text logo`
2. Text color: blue, teal, gold, rainbow, red, orange, white, cyan, purple, green, yellow,
   grey, silver, black
3. Style modifiers: dripping, colorful, graffiti, tattoo, anime, pixel art, 8-bit, 16-bit,
   32-bit, metal, metallic, spikey, stone, splattered, comic book, 80s, neon, 3D
4. Accent modifiers: smoke, fire, flames, tentacles, glow, horns, wings, halo

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipe.load_lora_weights(
    "HarroweD/HarrlogosXL",
    weight_name="HarrlogosXL.safetensors"
)

prompt = '"ACME" text logo, gold, metallic, 3D, glow'
negative_prompt = "lowres, bad anatomy, worst quality, low quality"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save("harrlogos_output.png")
```

### 1.4 Other Notable Logo LoRAs

| Model | Base | Notes |
|-------|------|-------|
| `dber123/lora-logo-simple` | SDXL | Simple/clean logo style |
| Logo Maker 9000 SDXL (CivitAI #436281) | SDXL | Concept-driven logo generation |
| Shakker-Labs/FLUX.1-dev-LoRA-add-details | FLUX | Adds fine detail; useful for logo refinement |

### 1.5 General LoRA Loading Patterns

**Loading a LoRA**:
```python
pipe.load_lora_weights("org/model-name", weight_name="file.safetensors")
```

**Fusing for performance** (merges weights into the base model permanently for the session):
```python
pipe.fuse_lora(lora_scale=0.8)  # scale 0.0-1.0
```

**Unfusing** (to swap LoRAs):
```python
pipe.unfuse_lora()
pipe.unload_lora_weights()
```

**Stacking multiple LoRAs** (diffusers supports loading multiple):
```python
pipe.load_lora_weights("org/lora-a", weight_name="a.safetensors", adapter_name="style_a")
pipe.load_lora_weights("org/lora-b", weight_name="b.safetensors", adapter_name="style_b")
pipe.set_adapters(["style_a", "style_b"], adapter_weights=[0.7, 0.5])
```

---

## 2. ControlNet for Logos

ControlNet adds structural control to diffusion models by conditioning on edge maps, depth maps,
sketches, and other spatial information. For logos, the most relevant conditioning types are
**Canny** (edge detection), **Scribble** (rough sketches), and **Lineart** (clean line drawings).

### 2.1 Architecture Overview

ControlNet creates a trainable copy of the encoder blocks of the diffusion model, connected
via "zero convolution" layers. The original model weights remain frozen. A conditioning image
(canny edges, scribble, etc.) is fed through the copy, and its outputs are added to the
main model's intermediate features.

Key parameter: `controlnet_conditioning_scale` (0.0-1.0+) controls how much influence the
conditioning has over the output.

### 2.2 FLUX ControlNet (Canny)

**Model**: `InstantX/FLUX.1-dev-Controlnet-Canny`

```python
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.utils import load_image

# ----- Prepare canny edge image from a logo sketch -----
original_image = load_image("path/to/logo_sketch.png")
image_np = np.array(original_image)
edges = cv2.Canny(image_np, 100, 200)
edges = np.stack([edges] * 3, axis=2)
canny_image = Image.fromarray(edges)

# ----- Load pipeline -----
controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny",
    torch_dtype=torch.bfloat16,
)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
).to("cuda")

prompt = "a clean minimalist logo design, vector art, professional branding"
image = pipe(
    prompt,
    control_image=canny_image,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=50,
    guidance_scale=3.5,
).images[0]
image.save("flux_controlnet_logo.png")
```

### 2.3 FLUX ControlNet Union (Multiple Control Types)

**Model**: `InstantX/FLUX.1-dev-Controlnet-Union`

Supports multiple control types in a single model (canny, depth, lineart, etc.)
via control mode selection.

### 2.4 FLUX Lineart ControlNet

**Model**: `promeai/FLUX.1-controlnet-lineart-promeai`

Trained specifically for line art conditioning with FLUX. Artists often prefer lineart
over canny because it more closely matches hand-drawn sketches. The MistoLine variant
(`TheMistoAI/MistoControlNet-Flux-dev`) handles any type of line input.

### 2.5 SDXL ControlNet -- Canny

**Model**: `diffusers/controlnet-canny-sdxl-1.0`

```python
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image

# Prepare canny edge map
original = load_image("path/to/logo_draft.png")
image_np = np.array(original)
edges = cv2.Canny(image_np, 100, 200)
canny_image = Image.fromarray(np.stack([edges] * 3, axis=2))

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    prompt="minimalist professional logo, vector style, clean lines",
    negative_prompt="lowres, blurry, bad quality",
    image=canny_image,
    controlnet_conditioning_scale=0.5,
    num_inference_steps=30,
).images[0]
image.save("sdxl_canny_logo.png")
```

### 2.6 SDXL ControlNet -- Scribble

**Model**: `xinsir/controlnet-scribble-sdxl-1.0`

Supports any type and width of lines. Thicker lines = coarser control (follows text prompt
more); thinner lines = stronger structural control (adheres to conditioning image more).

```python
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler,
)
from controlnet_aux import HEDdetector

def nms(x, t, s):
    """Non-maximum suppression for edge thinning."""
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)
    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    y = np.zeros_like(x)
    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)
    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z

# Load models
controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-scribble-sdxl-1.0", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler"
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=torch.float16,
).to("cuda")

# --- Option A: Extract scribble from existing image ---
processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
scribble = processor("path/to/reference_logo.png", scribble=False)
scribble_np = np.array(scribble)
scribble_np = nms(scribble_np, 127, 3)
scribble_np = cv2.GaussianBlur(scribble_np, (0, 0), 3)
scribble_np[scribble_np > 20] = 255
scribble_np[scribble_np < 255] = 0
controlnet_img = Image.fromarray(scribble_np)

# --- Option B: Load a hand-drawn sketch directly ---
# controlnet_img = Image.open("path/to/sketch.png")  # must be black/white

# Resize to ~1024x1024 total pixels
w, h = controlnet_img.size
ratio = np.sqrt(1024.0 * 1024.0 / (w * h))
new_w, new_h = int(w * ratio), int(h * ratio)
controlnet_img = controlnet_img.resize((new_w, new_h))

image = pipe(
    prompt="professional logo design, clean vector art, modern branding",
    negative_prompt="lowres, bad anatomy, worst quality, low quality",
    image=controlnet_img,
    controlnet_conditioning_scale=1.0,
    width=new_w,
    height=new_h,
    num_inference_steps=30,
).images[0]
image.save("scribble_logo.png")
```

### 2.7 SDXL ControlNet -- Lineart

**Model**: `ShermanG/ControlNet-Standard-Lineart-for-SDXL` or `TheMistoAI/MistoLine`

MistoLine is particularly versatile -- handles any type of line input (canny, HED, scribble,
hand-drawn) with a single model, making it ideal for logo sketch-to-design workflows.

```python
controlnet = ControlNetModel.from_pretrained(
    "ShermanG/ControlNet-Standard-Lineart-for-SDXL",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

line_art = Image.open("path/to/logo_lineart.png")
image = pipe(
    prompt="premium brand logo, elegant design",
    negative_prompt="lowres, blurry",
    image=line_art,
    controlnet_conditioning_scale=0.9,
    num_inference_steps=30,
).images[0]
```

### 2.8 Multi-ControlNet

Combine multiple ControlNets (e.g., canny + depth) by passing lists to the pipeline:

```python
controlnets = [
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0-small", torch_dtype=torch.float16
    ),
    ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
    ),
]

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnets,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

images = [canny_image.resize((1024, 1024)), depth_image.resize((1024, 1024))]

result = pipe(
    prompt="modern logo design",
    negative_prompt="lowres, bad quality",
    image=images,
    controlnet_conditioning_scale=[0.5, 0.5],
    num_inference_steps=50,
).images[0]
```

### 2.9 Logo-Specific ControlNet Tips

- **Canny**: Best for preserving sharp edges and geometric shapes in logos. Use low_threshold=100,
  high_threshold=200 for clean results. Lower `controlnet_conditioning_scale` (0.3-0.5) to allow
  creative interpretation; higher (0.7-1.0) for strict adherence.
- **Scribble**: Best for rough concept sketches. Allows the most creative freedom. Thicker lines
  give more freedom to the text prompt.
- **Lineart**: Best for final logo refinement where precise line work matters. Closest to
  production-quality output.

---

## 3. IP-Adapter for Style Transfer

IP-Adapter (Image Prompt Adapter) is a ~100 MB lightweight adapter that enables conditioning
image generation on a reference image. It injects image features through decoupled cross-attention
layers alongside the existing text cross-attention, preserving both image and text guidance.

### 3.1 How It Works

1. A reference image is encoded by a CLIP image encoder into image embeddings.
2. These embeddings are passed to newly added cross-attention layers in the UNet/transformer.
3. The original text cross-attention layers are kept frozen and separate.
4. The `ip_adapter_scale` parameter (0.0-1.0) controls how much the reference image influences
   generation vs. the text prompt.

### 3.2 IP-Adapter Variants

| Variant | Encoder | Best For |
|---------|---------|----------|
| `ip-adapter_sdxl.bin` | CLIP ViT-G | General style transfer |
| `ip-adapter-plus_sdxl_vit-h.safetensors` | CLIP ViT-H (patch) | Higher fidelity style matching |
| `ip-adapter-plus-face_sdxl_vit-h.safetensors` | CLIP ViT-H (patch) | Face/character preservation |
| `ip-adapter-faceid_sdxl.bin` | InsightFace | Strongest face identity |

### 3.3 Basic Usage (SDXL)

```python
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin",
)
pipeline.set_ip_adapter_scale(0.6)

# Load a reference logo whose style you want to transfer
reference_logo = load_image("path/to/reference_logo.png")

image = pipeline(
    prompt="minimalist tech company logo, clean vector design",
    ip_adapter_image=reference_logo,
    negative_prompt="deformed, ugly, low quality, blurry",
    num_inference_steps=30,
).images[0]
image.save("style_transferred_logo.png")
```

### 3.4 IP-Adapter Plus (Higher Fidelity)

Uses patch embeddings and a ViT-H image encoder for more accurate style matching:

```python
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
).to("cuda")

pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
)
pipeline.set_ip_adapter_scale(0.7)
```

### 3.5 IP-Adapter for FLUX

FLUX IP-Adapter support is available from InstantX (`InstantX/FLUX.1-dev-IP-Adapter`) and
XLabs-AI (`XLabs-AI/flux-ip-adapter`). The InstantX version uses a SigLIP vision encoder
rather than CLIP.

**InstantX FLUX IP-Adapter** (uses custom pipeline files, not yet fully integrated into
stock diffusers):

```python
import torch
from PIL import Image

# These imports come from the InstantX repo files (download from HuggingFace)
from pipeline_flux_ipa import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from infer_flux_ipa_siglip import resize_img, IPAdapter

image_encoder_path = "google/siglip-so400m-patch14-384"
ipadapter_path = "./ip-adapter.bin"  # downloaded from InstantX/FLUX.1-dev-IP-Adapter

transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

ip_model = IPAdapter(
    pipe, image_encoder_path, ipadapter_path,
    device="cuda", num_tokens=128,
)

reference = Image.open("reference_logo.jpg").convert("RGB")
reference = resize_img(reference)

images = ip_model.generate(
    pil_image=reference,
    prompt="a modern tech startup logo",
    scale=0.7,        # IP-Adapter influence strength
    width=1024,
    height=1024,
    seed=42,
)
images[0].save("flux_ipadapter_logo.png")
```

### 3.6 Pre-computing Image Embeddings (Efficiency)

When generating multiple variations from the same reference, pre-compute and reuse embeddings:

```python
# Compute once
image_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=reference_logo,
    ip_adapter_image_embeds=None,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)
torch.save(image_embeds, "logo_style_embeds.ipadpt")

# Reuse for each variation
image_embeds = torch.load("logo_style_embeds.ipadpt")
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    image_encoder_folder=None,  # skip loading encoder
    weight_name="ip-adapter_sdxl.bin",
)
pipeline.set_ip_adapter_scale(0.6)

for i, prompt in enumerate(["tech logo", "nature logo", "food logo"]):
    img = pipeline(
        prompt=prompt,
        ip_adapter_image_embeds=image_embeds,
        negative_prompt="lowres, blurry",
    ).images[0]
    img.save(f"variation_{i}.png")
```

### 3.7 Regional Masking with IP-Adapter

Assign different reference images to different regions of the output using binary masks:

```python
from diffusers.image_processor import IPAdapterMaskProcessor

processor = IPAdapterMaskProcessor()
masks = processor.preprocess([mask1, mask2], height=1024, width=1024)

pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"],
)
pipeline.set_ip_adapter_scale([[0.7, 0.7]])

result = pipeline(
    prompt="company logo with icon and text",
    ip_adapter_image=[[icon_ref, text_style_ref]],
    cross_attention_kwargs={"ip_adapter_masks": [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]},
).images[0]
```

---

## 4. InstantStyle for Style-Content Separation

InstantStyle is a technique (not a separate model) that disentangles **style** (color, texture,
atmosphere) from **content** (subject, layout) in reference images. It works by activating
IP-Adapter in specific attention blocks of the UNet only.

### 4.1 Key Insight: Block-Level Control

Different attention layers in the UNet capture different semantic information:
- **`up_blocks.0.attentions.1`** -- captures **style** (color, material, atmosphere)
- **`down_blocks.2.attentions.1`** -- captures **layout** (structure, composition)

By activating IP-Adapter only in the style blocks, you transfer the visual style of a
reference image without copying its content/layout.

### 4.2 Content Subtraction

InstantStyle also subtracts content text features from image features at the CLIP level.
After subtracting the content text embedding from the image embedding, what remains is
predominantly style information. This reduces content leakage even further.

### 4.3 Style-Only Transfer (Diffusers Native)

Natively supported in diffusers (v0.28.0+) via per-block scale dictionaries:

```python
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin",
)

# ---- Style ONLY (no layout leakage) ----
scale_style_only = {
    "up": {"block_0": [0.0, 1.0, 0.0]},
}
pipeline.set_ip_adapter_scale(scale_style_only)

style_reference = load_image("path/to/reference_with_desired_style.png")

image = pipeline(
    prompt="a lion logo, professional brand identity",
    ip_adapter_image=style_reference,
    negative_prompt="text, watermark, lowres, worst quality, deformed, blurry",
    guidance_scale=5,
    num_inference_steps=30,
).images[0]
image.save("instantstyle_logo.png")
```

### 4.4 Style + Layout Transfer

To transfer both style and spatial layout from a reference:

```python
# Style + Layout
scale_style_and_layout = {
    "down": {"block_2": [0.0, 1.0]},   # layout information
    "up":   {"block_0": [0.0, 1.0, 0.0]},  # style information
}
pipeline.set_ip_adapter_scale(scale_style_and_layout)

image = pipeline(
    prompt="a cat logo, professional brand identity",
    ip_adapter_image=style_reference,
    negative_prompt="text, watermark, lowres, worst quality, deformed, blurry",
    guidance_scale=5,
    num_inference_steps=30,
).images[0]
```

### 4.5 InstantStyle with the Original Library

The original InstantStyle library (`instantX-research/InstantStyle`) allows explicit targeting
of attention blocks:

```python
from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterXL
from PIL import Image
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_tiling()

ip_model = IPAdapterXL(
    pipe,
    image_encoder_path="sdxl_models/image_encoder",
    ip_ckpt="sdxl_models/ip-adapter_sdxl.bin",
    device="cuda",
    # Style blocks only:
    target_blocks=["up_blocks.0.attentions.1"],
    # Style + Layout:
    # target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"],
)

ref = Image.open("./assets/reference_style.jpg").resize((512, 512))

images = ip_model.generate(
    pil_image=ref,
    prompt="a modern tech company logo, clean vector art",
    negative_prompt="text, watermark, lowres, worst quality",
    scale=1.0,
    guidance_scale=5,
    num_samples=1,
    num_inference_steps=30,
    seed=42,
)
images[0].save("instantstyle_result.png")
```

### 4.6 Practical Tips for Logo Style Transfer

- **Scale 0.6-0.8**: Good balance between style fidelity and prompt adherence.
- **Style-only blocks**: Strongly recommended for logos to avoid copying the reference
  logo's shape/layout -- you want the style (color palette, texture, treatment) without
  the specific design.
- **Multiple references**: Feed several examples of the same style for more robust
  style extraction.
- Layers **not** included in the scale dict default to 0 (IP-Adapter disabled there).

---

## 5. Combining ControlNet + IP-Adapter + LoRA

The most powerful approach for logo generation combines all three:
- **LoRA** -- steers the base model toward logo aesthetics
- **ControlNet** -- provides structural/spatial guidance (from a sketch or edge map)
- **IP-Adapter** -- transfers a reference style (color palette, texture, mood)

### 5.1 ControlNet + IP-Adapter (SDXL, Diffusers Native)

```python
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

# 1. Load ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth",
    torch_dtype=torch.float16,
)

# 2. Create pipeline with ControlNet
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

# 3. Load IP-Adapter
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter_sd15.bin",
)
pipeline.set_ip_adapter_scale(0.6)

# 4. Generate with both controls
depth_map = load_image("path/to/logo_depth_map.png")
style_ref = load_image("path/to/style_reference.png")

result = pipeline(
    prompt="professional company logo, clean design",
    image=depth_map,                       # ControlNet conditioning
    ip_adapter_image=style_ref,            # IP-Adapter style reference
    negative_prompt="lowres, bad quality",
    num_inference_steps=50,
).images[0]
result.save("controlnet_ipadapter_logo.png")
```

### 5.2 Full Combo: ControlNet + IP-Adapter + LoRA (SDXL)

```python
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from diffusers.utils import load_image

# --- 1. Load ControlNet (canny) ---
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
)

# --- 2. Load VAE fix ---
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

# --- 3. Create pipeline ---
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

# --- 4. Load LoRA (logo style) ---
pipe.load_lora_weights(
    "HarroweD/HarrlogosXL",
    weight_name="HarrlogosXL.safetensors",
)

# --- 5. Load IP-Adapter ---
pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin",
)
pipe.set_ip_adapter_scale(0.5)

# --- 6. Prepare canny edge conditioning ---
sketch = np.array(Image.open("path/to/logo_sketch.png"))
edges = cv2.Canny(sketch, 100, 200)
canny_img = Image.fromarray(np.stack([edges] * 3, axis=2)).resize((1024, 1024))

# --- 7. Load style reference ---
style_ref = load_image("path/to/brand_style_reference.png")

# --- 8. Generate ---
image = pipe(
    prompt='"BRAND" text logo, metallic, 3D, professional',
    negative_prompt="lowres, bad anatomy, worst quality, low quality",
    image=canny_img,
    ip_adapter_image=style_ref,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save("full_combo_logo.png")
```

### 5.3 Full Combo with InstantStyle Block Targeting

Add InstantStyle's per-block scale to the combo for cleaner style separation:

```python
# After loading IP-Adapter, set block-level scales
pipe.set_ip_adapter_scale({
    "up": {"block_0": [0.0, 1.0, 0.0]},  # style only from IP-Adapter
})

# The LoRA handles logo aesthetics globally
# ControlNet handles structure from the sketch
# IP-Adapter (style blocks only) handles color/texture/mood
image = pipe(
    prompt='"BRAND" text logo, professional branding',
    negative_prompt="lowres, bad quality",
    image=canny_img,
    ip_adapter_image=style_ref,
    controlnet_conditioning_scale=0.6,
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
```

### 5.4 IP-Adapter + LoRA with LCM (Fast Generation)

For rapid iteration during logo exploration (4 steps):

```python
import torch
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers.utils import load_image

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)

# Load IP-Adapter
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin",
)

# Load LCM LoRA for speed
pipeline.load_lora_weights("latent-consistency/lcm-lora-sdxl")
pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_model_cpu_offload()

pipeline.set_ip_adapter_scale(0.4)

style_ref = load_image("path/to/reference.png")
image = pipeline(
    prompt="minimalist logo design, vector art",
    ip_adapter_image=style_ref,
    num_inference_steps=4,       # only 4 steps!
    guidance_scale=1.0,
).images[0]
```

### 5.5 Scale Tuning Guide

| Component | Parameter | Logo Range | Notes |
|-----------|-----------|------------|-------|
| LoRA | `lora_scale` / `fuse_lora(lora_scale=)` | 0.6-1.0 | Higher = stronger logo aesthetic |
| ControlNet | `controlnet_conditioning_scale` | 0.3-0.8 | Lower = creative freedom; higher = strict structure |
| IP-Adapter | `set_ip_adapter_scale()` | 0.4-0.8 | Lower = more text prompt influence |
| InstantStyle | Block-level dict | 0.0 or 1.0 per block | Binary on/off per layer works best |

---

## 6. Style Consistency Across Multiple Logo Variations

Generating a family of logos (different layouts, sizes, color variants) that maintain a
consistent style requires several strategies.

### 6.1 Fixed Seed + Shared Embeddings

Use the same random seed and pre-computed IP-Adapter embeddings across variations:

```python
import torch
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter_sdxl.bin",
)
pipeline.set_ip_adapter_scale(0.7)

# Pre-compute style embeddings from your brand reference
brand_ref = load_image("path/to/brand_reference.png")
style_embeds = pipeline.prepare_ip_adapter_image_embeds(
    ip_adapter_image=brand_ref,
    ip_adapter_image_embeds=None,
    device="cuda",
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
)
torch.save(style_embeds, "brand_style.ipadpt")

# Generate consistent variations
base_seed = 12345
prompts = [
    "logo icon, minimalist mountain symbol",
    "logo icon, minimalist mountain with text 'ALPINE'",
    "logo icon, minimalist mountain, horizontal layout",
    "logo icon, minimalist mountain, monochrome version",
]

for i, prompt in enumerate(prompts):
    generator = torch.Generator(device="cuda").manual_seed(base_seed + i)
    img = pipeline(
        prompt=prompt,
        ip_adapter_image_embeds=style_embeds,
        negative_prompt="lowres, blurry, bad quality",
        generator=generator,
        num_inference_steps=30,
    ).images[0]
    img.save(f"brand_variation_{i}.png")
```

### 6.2 Shared LoRA + Consistent Prompting

Use the same LoRA across all variations with a structured prompt template:

```python
base_style = "wablogo, logo, Minimalist, clean vector, "
color_palette = "blue and white color scheme, "

variations = [
    base_style + color_palette + "mountain icon",
    base_style + color_palette + "mountain icon with text 'ALPINE'",
    base_style + color_palette + "horizontal mountain landscape icon",
]
```

### 6.3 ControlNet for Layout Variants

Use different ControlNet conditioning images to create layout variants while keeping
the same style:

```python
# Same style (LoRA + IP-Adapter) but different structural conditioning
layouts = {
    "square": "path/to/square_layout_sketch.png",
    "horizontal": "path/to/horizontal_layout_sketch.png",
    "icon_only": "path/to/icon_only_sketch.png",
    "with_text": "path/to/icon_and_text_sketch.png",
}

for name, sketch_path in layouts.items():
    sketch = Image.open(sketch_path)
    edges = cv2.Canny(np.array(sketch), 100, 200)
    canny = Image.fromarray(np.stack([edges] * 3, axis=2)).resize((1024, 1024))

    img = pipe(
        prompt="professional brand logo, clean vector",
        image=canny,
        ip_adapter_image_embeds=style_embeds,
        controlnet_conditioning_scale=0.5,
        num_inference_steps=30,
    ).images[0]
    img.save(f"layout_{name}.png")
```

### 6.4 Multiple IP-Adapter References for Robust Style

Feed multiple examples of your desired style to make the style extraction more robust
and less dependent on any single reference image:

```python
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    torch_dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    image_encoder=image_encoder,
).to("cuda")

pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"],
)
pipeline.set_ip_adapter_scale(0.6)

# Load multiple style reference images
style_refs = [
    load_image("style_example_1.png"),
    load_image("style_example_2.png"),
    load_image("style_example_3.png"),
]

# Pass as a list -- the model averages the style signals
image = pipeline(
    prompt="company logo, professional design",
    ip_adapter_image=[style_refs],
    negative_prompt="lowres, bad quality",
).images[0]
```

### 6.5 Consistency Checklist

1. **Same LoRA** across all variants (same adapter weights and scale)
2. **Same IP-Adapter style embeddings** (pre-compute once, reuse everywhere)
3. **InstantStyle style-only blocks** to prevent content leakage between variants
4. **Consistent negative prompts** to maintain quality baseline
5. **Structured prompt templates** with shared prefix/suffix
6. **Fixed or sequential seeds** for reproducible exploration
7. **Same guidance scale and step count** across the batch
8. **Post-processing**: apply consistent color correction / SVG tracing to all outputs

---

## 7. Summary and Recommendations

### For Logo Generation Pipeline

| Stage | Tool | Purpose |
|-------|------|---------|
| Base aesthetic | FLUX Logo LoRA or Harrlogos XL | Push the model toward logo-quality output |
| Structural control | ControlNet (Canny/Lineart/Scribble) | Guide layout from sketches |
| Style transfer | IP-Adapter + InstantStyle | Apply brand colors/texture/mood from references |
| Fast iteration | LCM LoRA | 4-step generation for rapid exploration |
| Consistency | Pre-computed embeddings + fixed seeds | Coherent brand family |

### Recommended Stack by Base Model

**FLUX.1-dev** (highest quality, non-commercial):
- LoRA: `Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design`
- ControlNet: `InstantX/FLUX.1-dev-Controlnet-Canny` or Union variant
- IP-Adapter: `InstantX/FLUX.1-dev-IP-Adapter` (requires custom pipeline files)

**SDXL** (mature ecosystem, full combo support):
- LoRA: `HarroweD/HarrlogosXL` (for text in logos) or custom-trained
- ControlNet: `diffusers/controlnet-canny-sdxl-1.0`, `xinsir/controlnet-scribble-sdxl-1.0`
- IP-Adapter: `h94/IP-Adapter` with `ip-adapter_sdxl.bin` or Plus variant
- InstantStyle: Native in diffusers via per-block scale dicts

### Installation

```bash
pip install diffusers transformers accelerate safetensors
pip install opencv-python controlnet-aux
# For InsightFace (FaceID variant):
# pip install insightface onnxruntime-gpu
```

### Memory Optimization

```python
# CPU offloading (load IP-Adapter BEFORE enabling offload)
pipeline.load_ip_adapter(...)
pipeline.enable_model_cpu_offload()

# VAE tiling for high resolution
pipeline.enable_vae_tiling()

# Attention slicing
pipeline.enable_attention_slicing()

# For FLUX with bfloat16 -- requires ~24GB VRAM without offloading
# With enable_model_cpu_offload() -- ~12GB VRAM
```

---

## Sources

- [Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design)
- [HarroweD/HarrlogosXL](https://huggingface.co/HarroweD/HarrlogosXL)
- [Harrlogos XL on CivitAI](https://civitai.com/models/176555/harrlogos-xl-finally-custom-text-generation-in-sd)
- [Diffusers ControlNet Guide](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)
- [Diffusers IP-Adapter Guide](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)
- [InstantX/FLUX.1-dev-Controlnet-Canny](https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny)
- [InstantX/FLUX.1-dev-IP-Adapter](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter)
- [XLabs-AI/flux-ip-adapter](https://huggingface.co/XLabs-AI/flux-ip-adapter)
- [xinsir/controlnet-scribble-sdxl-1.0](https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0)
- [InstantStyle GitHub](https://github.com/instantX-research/InstantStyle)
- [promeai/FLUX.1-controlnet-lineart-promeai](https://huggingface.co/promeai/FLUX.1-controlnet-lineart-promeai)
- [ShermanG/ControlNet-Standard-Lineart-for-SDXL](https://huggingface.co/ShermanG/ControlNet-Standard-Lineart-for-SDXL)
- [TheMistoAI/MistoLine](https://github.com/TheMistoAI/MistoControlNet-Flux-dev)
