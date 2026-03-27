# Flux Deep Dive: Models, Setup, and Logo Generation

> Research date: 2026-03-27

## Table of Contents

1. [Model Variants](#1-flux-model-variants)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Diffusers Setup](#3-diffusers-setup-for-flux)
4. [Logo-Specific LoRAs](#4-flux-loras-for-logo-generation)
5. [ControlNet Models](#5-flux-controlnet-models)
6. [IP-Adapter Setup](#6-flux-ip-adapter-setup)
7. [Optimal Inference Parameters for Logos](#7-optimal-inference-parameters-for-logos)
8. [API Providers: Pricing and Endpoints](#8-flux-via-api)
9. [Memory Optimization](#9-memory-optimization)

---

## 1. Flux Model Variants

Flux is developed by **Black Forest Labs** (BFL), founded by former Stability AI researchers. The model family spans two generations: FLUX.1 (12B parameters) and FLUX.2 (32B parameters, released November 2025).

### FLUX.1 Family

| Variant | Parameters | License | Steps | Guidance | Use Case |
|---------|-----------|---------|-------|----------|----------|
| **FLUX.1 [schnell]** | 12B | Apache 2.0 | 1-4 | None (distilled) | Fast drafts, prototyping |
| **FLUX.1 [dev]** | 12B | Non-commercial | 20-50 | Guidance-distilled (3.5) | Local experimentation, LoRA training |
| **FLUX.1 [pro]** | 12B | API-only | ~28 | Internal | Production quality |
| **FLUX.1.1 [pro]** | 12B | API-only | ~28 | Internal | Faster/cheaper than 1.0 Pro |
| **FLUX.1.1 [pro] Ultra** | 12B | API-only | ~28 | Internal | Up to 4MP (2K+) output |
| **FLUX.1 Kontext [pro]** | 12B | API-only | ~28 | Internal | In-context editing, character consistency |
| **FLUX.1 Kontext [dev]** | 12B | Non-commercial | ~28 | 2.5 | Local in-context editing |

### FLUX.2 Family (November 2025+)

FLUX.2 is a **complete architectural overhaul**: ~32B parameters, single Mistral Small 3.1 text encoder (replacing dual CLIP+T5), 8 double-stream + 48 single-stream transformer blocks, native multi-reference image composition (up to 10 images), and a new `AutoencoderKLFlux2`.

| Variant | Parameters | License | Speed | Use Case |
|---------|-----------|---------|-------|----------|
| **FLUX.2 [klein] 4B** | 4B | API/weights | <0.5s | Real-time, edge devices, ~8GB VRAM |
| **FLUX.2 [klein] 9B** | 9B | API/weights | <0.5s | Fast local, ~13GB VRAM |
| **FLUX.2 [dev]** | 32B | Non-commercial | ~3-5s | Local experimentation |
| **FLUX.2 [flex]** | 32B | API | ~2-4s | Adjustable quality/speed tradeoff |
| **FLUX.2 [pro]** | 32B | API | ~3-5s | Production, API workflows |
| **FLUX.2 [max]** | 32B | API | ~3-5s | Highest quality, final renders |

### Key Architecture Differences (FLUX.1 vs FLUX.2)

| Feature | FLUX.1 | FLUX.2 |
|---------|--------|--------|
| Parameters | ~12B | ~32B (3x larger) |
| Text encoders | CLIP + T5-XXL (dual) | Mistral Small 3.1 (single) |
| Double-stream blocks | 19 | 8 |
| Single-stream blocks | 38 | 48 |
| Multi-image input | No (Redux adapter needed) | Native (up to 10 images) |
| MLP activation | GELU | SwiGLU |
| Bias parameters | Yes | No |
| Pipeline class | `FluxPipeline` | `Flux2Pipeline` |

### Relevance for Logo Generation

- **FLUX.1 [dev]** is the best choice for local logo generation with LoRAs -- most logo LoRAs target this model.
- **FLUX.1 [schnell]** is useful for rapid iteration on prompts (4 steps).
- **FLUX.2 [klein] 4B/9B** are interesting for fast local logo drafts if LoRA ecosystem catches up.
- **API models** (Pro, Max) provide the best raw quality when LoRAs are not needed.

---

## 2. Hardware Requirements

### FLUX.1 VRAM Requirements

| Configuration | VRAM | RAM | Speed | Notes |
|--------------|------|-----|-------|-------|
| BF16 full precision | ~33GB | 64GB+ | Fast | Requires A100/H100 or dual GPUs |
| FP16 + CPU offload | ~24GB | 64GB+ | Medium | RTX 3090/4090 |
| FP8 quantized (quanto) | ~16-18GB | 32GB+ | Medium | RTX 4080/4090 |
| NF4 quantized (bitsandbytes) | ~12-14GB | 32GB+ | Slower | RTX 4070 Ti+ |
| SVDQuant INT4 (Nunchaku) | ~10GB | 32GB+ | Fast (3x NF4) | RTX 4060 Ti+ |
| Sequential CPU offload + VAE tiling | ~6GB | 64GB+ | Very slow | Any 6GB GPU (slow!) |

### FLUX.2 VRAM Requirements

| Configuration | VRAM | Notes |
|--------------|------|-------|
| BF16 unoptimized | 80GB+ | Requires H100/A100 |
| NF4 quantized + CPU offload | ~18GB | RTX 4090 |
| NF4 + group offload + remote text enc | ~8GB | RTX 3060+ (slow) |
| FLUX.2 Klein 4B | ~8GB | Consumer GPUs |
| FLUX.2 Klein 9B | ~13GB | RTX 4070+ |

### System RAM Note

Quantizing Flux at startup requires **~50GB of system RAM** temporarily (loading BF16 weights before quantizing). Use pre-quantized checkpoints to avoid this.

### Training Requirements

- **LoRA training (FLUX.1)**: 24GB VRAM minimum (RTX 3090/4090)
- **LoRA training (FLUX.2)**: 24GB+ with FP8/NF4 + gradient checkpointing + remote text encoder
- **Full fine-tuning**: Multiple A100s / H100s

---

## 3. Diffusers Setup for Flux

### Installation

```bash
pip install diffusers transformers accelerate torch torchvision
pip install sentencepiece protobuf  # for T5 tokenizer (FLUX.1)
pip install bitsandbytes             # for quantization
pip install controlnet-aux           # for ControlNet preprocessors
pip install image-gen-aux            # for depth preprocessing
```

### FLUX.1 Dev -- Basic Text-to-Image

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()  # saves ~9GB VRAM

prompt = "A minimalist logo of a mountain and sun, clean vector style"
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_inference_steps=28,
    height=1024,
    width=1024,
    max_sequence_length=512,
).images[0]
image.save("logo.png")
```

### FLUX.1 Schnell -- Fast Generation (4 steps)

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="logo, minimalist coffee cup icon, flat design",
    guidance_scale=0.0,        # schnell doesn't use guidance
    num_inference_steps=4,
    height=1024,
    width=1024,
    max_sequence_length=256,
).images[0]
image.save("logo_fast.png")
```

### FLUX.2 Dev -- Next-Gen Pipeline

```python
import torch
from diffusers import Flux2Pipeline

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="A professional logo for a tech startup called 'Nexus'",
    num_inference_steps=28,
    guidance_scale=4.0,
    height=1024,
    width=1024,
).images[0]
image.save("logo_flux2.png")
```

### FLUX.2 Dev -- Multi-Reference Composition

```python
import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image

pipe = Flux2Pipeline.from_pretrained(
    "black-forest-labs/FLUX.2-dev",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

ref_icon = load_image("my_icon_reference.png")
ref_style = load_image("my_style_reference.png")

image = pipe(
    prompt="Combine the icon from image 1 with the color palette and style from image 2, minimalist logo design",
    image=[ref_icon, ref_style],
    num_inference_steps=50,
    guidance_scale=2.5,
    width=1024,
    height=1024,
).images[0]
image.save("logo_multiref.png")
```

---

## 4. Flux LoRAs for Logo Generation

### Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design (Recommended)

The most established logo LoRA for Flux. Trained on minimalist logo designs.

- **Trigger words**: `wablogo, logo, Minimalist`
- **Recommended LoRA scale**: `0.8`
- **Recommended steps**: 24
- **Recommended guidance**: 3.5
- **Model**: `Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design`

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.load_lora_weights(
    "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design",
    weight_name="FLUX-dev-lora-Logo-Design.safetensors"
)
pipe.fuse_lora(lora_scale=0.8)
pipe.to("cuda")

# Dual combination (two objects)
prompt = "wablogo, logo, Minimalist, A mountain and a compass"
image = pipe(prompt, num_inference_steps=24, guidance_scale=3.5).images[0]
image.save("logo_mountain.png")

# Font combination (shape + letter)
prompt = "wablogo, logo, Minimalist, A tree with the letter N"
image = pipe(prompt, num_inference_steps=24, guidance_scale=3.5).images[0]
image.save("logo_tree_n.png")

# Text below graphic
prompt = "wablogo, logo, Minimalist, A coffee cup, below the graphic is the word 'BREW'"
image = pipe(prompt, num_inference_steps=24, guidance_scale=3.5).images[0]
image.save("logo_brew.png")
```

**Design pattern tips** (from Shakker-Labs documentation):
- **Dual combination**: "A [thing] and a [thing]" -- merges two concepts
- **Font combination**: "A [shape] with the letter [X]" or "fingerprint pattern with letters hp"
- **Text below graphic**: "Below the graphic is the word '[BRAND]'"

### prithivMLmods/Logo-Design-Flux-LoRA

- **Trigger word**: `Logo Design`
- **Base model**: FLUX.1-dev

```python
pipe.load_lora_weights("prithivMLmods/Logo-Design-Flux-LoRA")
pipe.fuse_lora(lora_scale=0.8)

prompt = "Logo Design, modern tech company logo, abstract geometric shapes"
image = pipe(prompt, num_inference_steps=24, guidance_scale=3.5).images[0]
```

### CivitAI Logo LoRAs

| Model | Trigger Word | Style | Weight |
|-------|-------------|-------|--------|
| **Icon & Logos (Flux)** (CivitAI #936661) | varies | Simple icons/logos | 0.7-1.0 |
| **Logo Flux Simple Color Style** (CivitAI #716810) | varies | Few-color, single subject | 0.7-0.9 |
| **gmic Flat Icon** (CivitAI #1149808) | `gmic_(heibaitubiao)` | Flat icons | 0.4-1.0 |
| **Vintage Logo Design** (CivitAI #817337) | varies | Vintage/retro logos | 0.7-1.0 |
| **LogoDesign V2** (CivitAI #1850266) | varies | General logo design | 0.7-1.0 |

### Shakker AI Platform Additional Logo LoRAs

- **Flux.1 Standard Logo Production Draft** -- production-ready logo drafts
- **Flux.1 Logo Icon Design** -- icon-focused generation

### Loading CivitAI LoRAs

```python
# Download .safetensors file from CivitAI, then:
pipe.load_lora_weights("/path/to/logo_lora.safetensors")
pipe.fuse_lora(lora_scale=0.8)
```

---

## 5. Flux ControlNet Models

ControlNet enables structural control over generation -- essential for logo refinement where you want to maintain a specific layout or shape.

### Available ControlNet Models

| Model | Type | Source |
|-------|------|--------|
| `black-forest-labs/FLUX.1-Canny-dev` | Full model (12B) | BFL official |
| `black-forest-labs/FLUX.1-Depth-dev` | Full model (12B) | BFL official |
| `black-forest-labs/FLUX.1-Canny-dev-lora` | LoRA adapter | BFL official |
| `black-forest-labs/FLUX.1-Depth-dev-lora` | LoRA adapter | BFL official |
| `InstantX/FLUX.1-dev-Controlnet-Canny` | Community ControlNet | InstantX |
| `InstantX/FLUX.1-dev-Controlnet-Canny-alpha` | Community ControlNet | InstantX |

### Canny ControlNet -- Full Model (FluxControlPipeline)

Best for logos: use an existing logo sketch/outline as a canny edge map to guide generation.

```python
import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Canny-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Load your logo sketch
control_image = load_image("logo_sketch.png")

# Extract canny edges
processor = CannyDetector()
control_image = processor(
    control_image,
    low_threshold=50,
    high_threshold=200,
    detect_resolution=1024,
    image_resolution=1024
)

image = pipe(
    prompt="A clean minimalist logo, professional design, vector style",
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=50,
    guidance_scale=30.0,
).images[0]
image.save("logo_controlled.png")
```

### Canny ControlNet -- LoRA Variant (Lighter)

```python
import torch
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
from diffusers.utils import load_image

pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("black-forest-labs/FLUX.1-Canny-dev-lora")

# Same control image processing and generation as above
```

### Depth ControlNet

```python
import torch
from diffusers import FluxControlPipeline
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Depth-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

control_image = load_image("reference_logo.png")
processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = pipe(
    prompt="Professional logo, clean design",
    control_image=control_image,
    height=1024,
    width=1024,
    num_inference_steps=30,
    guidance_scale=10.0,
).images[0]
image.save("logo_depth_controlled.png")
```

### Community ControlNet (FluxControlNetPipeline)

```python
import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.utils import load_image

controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny",
    torch_dtype=torch.bfloat16
)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    prompt="Minimalist logo design",
    control_image=canny_image,
    controlnet_conditioning_scale=0.7,
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
```

### Logo Workflow with ControlNet

1. **Sketch** a rough logo layout (even in MS Paint)
2. **Extract canny edges** from the sketch
3. **Generate** with ControlNet + logo LoRA + descriptive prompt
4. **Iterate** by adjusting `controlnet_conditioning_scale` (0.5-1.0)

---

## 6. Flux IP-Adapter Setup

IP-Adapter enables image-based style guidance -- pass a reference logo/style image to influence generation.

### XLabs-AI IP-Adapter (FLUX.1)

```python
import torch
from diffusers import FluxPipeline
from diffusers.utils import load_image

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Load reference style image
style_ref = load_image("existing_brand_logo.png").resize((1024, 1024))

pipe.load_ip_adapter(
    "XLabs-AI/flux-ip-adapter",
    weight_name="ip_adapter.safetensors",
    image_encoder_pretrained_model_name_or_path="openai/clip-vit-large-patch14"
)
pipe.set_ip_adapter_scale(0.7)  # lower for more prompt influence, higher for more image influence

image = pipe(
    prompt="minimalist logo, clean vector design, professional",
    ip_adapter_image=style_ref,
    width=1024,
    height=1024,
    true_cfg_scale=4.0,
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("logo_ip_adapted.png")
```

### InstantX IP-Adapter (Alternative)

```python
pipe.load_ip_adapter(
    "InstantX/FLUX.1-dev-IP-Adapter",
    weight_name="ip-adapter.safetensors",
    image_encoder_pretrained_model_name_or_path="google/siglip-so400m-patch14-384"
)
pipe.set_ip_adapter_scale(0.6)
```

### FLUX.1 Redux (BFL Official Image Adapter)

Redux is BFL's official image-to-image adapter -- useful for style transfer from an existing logo.

```python
import torch
from diffusers import FluxPriorReduxPipeline, FluxPipeline
from diffusers.utils import load_image

# Stage 1: Encode reference image
pipe_redux = FluxPriorReduxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Redux-dev",
    torch_dtype=torch.bfloat16
).to("cuda")

# Stage 2: Generate with encoded reference
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder=None,
    text_encoder_2=None,
    torch_dtype=torch.bfloat16
).to("cuda")

ref_image = load_image("brand_style_reference.png")
redux_output = pipe_redux(ref_image)

image = pipe(
    guidance_scale=2.5,
    num_inference_steps=50,
    **redux_output,
).images[0]
image.save("logo_redux.png")
```

### IP-Adapter Tips for Logos

- **Scale 0.3-0.5**: Subtle style influence (color palette, general feel)
- **Scale 0.6-0.8**: Strong style transfer (layout + style)
- **Scale 0.9-1.0**: Near-reproduction (useful for variations)
- Combine with LoRA for best results: IP-Adapter for style, LoRA for "logo-ness"
- Flux IP-Adapter is less refined than SDXL's; expect to experiment more

---

## 7. Optimal Inference Parameters for Logos

### Understanding Flux Guidance

Flux uses a **fundamentally different guidance system** than Stable Diffusion:

- **`guidance_scale`** in Flux Dev is a "guidance-distilled" value. A value of 3.5 approximates what CFG 7 does in SD 1.5. Do NOT use values like 7-12 as you would with SD.
- **`true_cfg_scale`** is actual classifier-free guidance (requires negative prompt). Usually set to 1.0 (disabled) unless you need negative prompting.
- **Flux Schnell** ignores guidance entirely -- it is fully distilled.

### Recommended Logo Parameters

| Parameter | FLUX.1 Dev | FLUX.1 Schnell | FLUX.2 Dev |
|-----------|-----------|----------------|------------|
| `guidance_scale` | **3.5** | **0.0** | **4.0** |
| `num_inference_steps` | **24-28** | **4** | **28-50** |
| `height` | **1024** | **1024** | **1024** |
| `width` | **1024** | **1024** | **1024** |
| `max_sequence_length` | **512** | **256** | **512** |
| `true_cfg_scale` | 1.0 (off) | N/A | 1.0 (off) |

### Resolution Recommendations for Logos

- **1024x1024**: Default, best quality. Use this for most logo generation.
- **768x768**: Acceptable for drafts, saves ~40% compute.
- **Square aspect ratio**: Strongly recommended for logos.
- Avoid non-standard resolutions that deviate far from training distribution.

### Logo Prompting Strategy

Flux has excellent text understanding. Use detailed, structured prompts:

```
# Good logo prompts for Flux
"wablogo, logo, Minimalist, A stylized fox head in orange and black, geometric, flat design, white background"

"logo, clean vector illustration of a lighthouse, navy blue and gold, professional corporate branding, simple shapes"

"wablogo, Minimalist logo, abstract interlocking letters AB, modern sans-serif, gradient from blue to purple, white background"
```

**Key prompt elements for logos:**
1. Trigger words (if using LoRA): `wablogo, logo, Minimalist`
2. Subject description: what the logo depicts
3. Style keywords: `flat design`, `vector`, `geometric`, `minimalist`, `clean`
4. Color specification: name specific colors
5. Background: almost always `white background` or `transparent background`
6. Anti-realism: avoid photographic terms; use `illustration`, `icon`, `symbol`

### Negative Prompting (with true_cfg_scale > 1)

```python
image = pipe(
    prompt="logo, minimalist mountain icon, flat design, white background",
    negative_prompt="photorealistic, 3d render, shadow, gradient, complex, detailed texture",
    true_cfg_scale=4.0,
    guidance_scale=3.5,
    num_inference_steps=28,
).images[0]
```

---

## 8. Flux via API

### Black Forest Labs Direct API

The official BFL API at `api.bfl.ml`.

| Model | Pricing |
|-------|---------|
| FLUX.2 [pro] | Calculator-based (per megapixel) |
| FLUX.2 [max] | Calculator-based (per megapixel) |
| FLUX.2 [flex] | Calculator-based (per megapixel) |
| FLUX.2 [klein] 4B | Calculator-based |
| FLUX.2 [klein] 9B | Calculator-based |

### Together AI

| Model | Endpoint ID | Price/Image | Default Steps |
|-------|------------|-------------|---------------|
| FLUX.1 [schnell] | `black-forest-labs/FLUX.1-schnell` | **$0.003** | 4 |
| FLUX.1 Krea [dev] | `black-forest-labs/FLUX.1-krea-dev` | **$0.025** | 28 |
| FLUX.1 Kontext [pro] | `black-forest-labs/FLUX.1-kontext-pro` | **$0.04** | 28 |
| FLUX.1 Kontext [max] | `black-forest-labs/FLUX.1-kontext-max` | **$0.08** | 28 |
| FLUX1.1 [pro] | `black-forest-labs/FLUX1.1-pro` | **$0.04** | - |
| FLUX.2 [dev] | `black-forest-labs/FLUX.2-dev` | **$0.015** | - |
| FLUX.2 [flex] | `black-forest-labs/FLUX.2-flex` | **$0.03** | - |
| FLUX.2 [pro] | `black-forest-labs/FLUX.2-pro` | **$0.03** | - |
| FLUX.2 [max] | `black-forest-labs/FLUX.2-max` | **$0.07** | - |

```python
# Together AI example
import together

client = together.Together(api_key="YOUR_KEY")

response = client.images.generate(
    model="black-forest-labs/FLUX.1-schnell",
    prompt="wablogo, logo, Minimalist, A mountain and compass",
    width=1024,
    height=1024,
    steps=4,
    n=1,
)
# response.data[0].url contains the image URL
```

### fal.ai

| Model | Endpoint ID | Price/Image |
|-------|------------|-------------|
| FLUX Dev | `fal-ai/flux/dev` | **$0.025** |
| FLUX Schnell | `fal-ai/flux/schnell` | ~$0.003 |
| FLUX Pro | `fal-ai/flux/pro` | **$0.05** |
| FLUX1.1 [pro] Ultra | `fal-ai/flux-pro/v1.1-ultra` | **$0.06** |
| FLUX.2 [pro] | `fal-ai/flux-2/pro` | **$0.03/MP** |
| FLUX Dev + LoRA | `fal-ai/flux-lora` | ~$0.025 |

```python
# fal.ai example
import fal_client

result = fal_client.subscribe(
    "fal-ai/flux/dev",
    arguments={
        "prompt": "logo, Minimalist, a stylized fox head",
        "image_size": "square_hd",   # 1024x1024
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
    },
)
# result["images"][0]["url"]
```

**fal.ai with LoRA:**
```python
result = fal_client.subscribe(
    "fal-ai/flux-lora",
    arguments={
        "prompt": "wablogo, logo, Minimalist, coffee cup and leaf",
        "loras": [
            {
                "path": "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design",
                "scale": 0.8
            }
        ],
        "image_size": "square_hd",
        "num_inference_steps": 24,
        "guidance_scale": 3.5,
    },
)
```

### Replicate

| Model | Endpoint ID | Price |
|-------|------------|-------|
| FLUX.1 Schnell | `black-forest-labs/flux-schnell` | ~$0.003/image |
| FLUX.1 Dev | `black-forest-labs/flux-dev` | ~$0.03/image |
| FLUX1.1 [pro] | `black-forest-labs/flux-1.1-pro` | ~$0.04/image |
| FLUX1.1 [pro] Ultra | `black-forest-labs/flux-1.1-pro-ultra` | ~$0.06/image |
| FLUX.2 [dev] | `black-forest-labs/flux-2-dev` | $0.012/MP |
| FLUX.2 [pro] | `black-forest-labs/flux-2-pro` | $0.015 + $0.015/MP |
| FLUX.2 [flex] | `black-forest-labs/flux-2-flex` | $0.06/MP |
| FLUX Kontext [pro] | `black-forest-labs/flux-kontext-pro` | ~$0.04/image |
| FLUX Kontext [max] | `black-forest-labs/flux-kontext-max` | ~$0.08/image |

```python
# Replicate example
import replicate

output = replicate.run(
    "black-forest-labs/flux-dev",
    input={
        "prompt": "logo, minimalist mountain icon, flat vector, white background",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 28,
        "guidance": 3.5,
    }
)
# output is a list of URLs
```

### API Cost Comparison (1024x1024, single image)

| Provider | Schnell | Dev | Pro | Best For |
|----------|---------|-----|-----|----------|
| **Together AI** | $0.003 | $0.015 | $0.03 | Cheapest Dev/Pro |
| **fal.ai** | ~$0.003 | $0.025 | $0.05 | LoRA support, fast |
| **Replicate** | ~$0.003 | ~$0.03 | ~$0.04 | Ecosystem, easy deploy |
| **BFL Direct** | N/A | N/A | varies | Official, latest models |

**Recommendation for logo generation**: Use **fal.ai** for LoRA-based generation (supports custom LoRAs natively) or **Together AI** for cheapest raw Flux access.

---

## 9. Memory Optimization

### Technique 1: Model CPU Offloading (Easiest)

Moves entire model components to CPU when not in use. ~24GB VRAM for FLUX.1.

```python
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()  # automatic offloading
```

### Technique 2: Sequential CPU Offloading (More Aggressive)

Moves individual layers to CPU. Slower but uses less VRAM (~12-16GB).

```python
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
```

### Technique 3: 8-bit Quantization (bitsandbytes)

Quantize transformer and text encoder to 8-bit. ~18-20GB VRAM.

```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
from transformers import T5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBnBConfig

# Quantize T5 text encoder
text_encoder_config = TransformersBnBConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    quantization_config=text_encoder_config,
    torch_dtype=torch.float16,
)

# Quantize transformer
transformer_config = DiffusersBnBConfig(load_in_8bit=True)
transformer_8bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=transformer_config,
    torch_dtype=torch.float16,
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

image = pipe(
    "logo, minimalist design",
    guidance_scale=3.5,
    num_inference_steps=28,
).images[0]
```

### Technique 4: NF4 Quantization (4-bit, bitsandbytes)

~12-14GB VRAM. Some quality loss but usually acceptable for logos.

```python
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
from transformers import T5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBnBConfig

text_encoder_config = TransformersBnBConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
text_encoder_4bit = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    quantization_config=text_encoder_config,
)

transformer_config = DiffusersBnBConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
transformer_4bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=transformer_config,
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder_4bit,
    transformer=transformer_4bit,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)
```

### Technique 5: FP8 Quantization (optimum-quanto)

Good quality/memory tradeoff. ~16-18GB VRAM.

```python
import torch
from diffusers import FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
from optimum.quanto import freeze, qfloat8, quantize

bfl_repo = "black-forest-labs/FLUX.1-dev"

transformer = FluxTransformer2DModel.from_single_file(
    "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8.safetensors",
    torch_dtype=torch.bfloat16
)
quantize(transformer, weights=qfloat8)
freeze(transformer)

text_encoder_2 = T5EncoderModel.from_pretrained(
    bfl_repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
)
quantize(text_encoder_2, weights=qfloat8)
freeze(text_encoder_2)

pipe = FluxPipeline.from_pretrained(
    bfl_repo, transformer=None, text_encoder_2=None, torch_dtype=torch.bfloat16
)
pipe.transformer = transformer
pipe.text_encoder_2 = text_encoder_2
pipe.enable_model_cpu_offload()
```

### Technique 6: SVDQuant / Nunchaku (Best Speed + Memory)

4-bit quantization with 3x speedup over NF4. **~10GB VRAM** on RTX 4090 laptop. ICLR 2025 spotlight paper.

```bash
pip install torch==2.6 torchvision==0.21
# Install nunchaku wheel from https://github.com/nunchaku-ai/nunchaku/releases
```

```python
import torch
from diffusers import FluxPipeline
from nunchaku.models.transformer_flux import NunchakuFluxTransformer2dModel

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    "mit-han-lab/svdq-int4-flux.1-dev"
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16
).to("cuda")

image = pipe(
    "wablogo, logo, Minimalist, a geometric wolf head",
    num_inference_steps=28,
    guidance_scale=3.5,
).images[0]
image.save("logo_svdquant.png")
```

**SVDQuant also supports LoRAs** -- all diffusers features (schedulers, callbacks, multi-GPU) work seamlessly since only the transformer component is swapped.

### Technique 7: Group Offloading (Diffusers Native)

Fine-grained offloading at the layer level. Works with quantization.

```python
import torch
from diffusers import FluxPipeline
from diffusers.hooks import apply_group_offloading

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

for component in [pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.vae]:
    apply_group_offloading(
        component,
        offload_type="leaf_level",
        offload_device=torch.device("cpu"),
        onload_device=torch.device("cuda"),
        use_stream=True,
    )
```

### Technique 8: FLUX.2 NF4 + Remote Text Encoder (8GB GPU)

For FLUX.2 on very constrained hardware:

```python
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel

transformer = Flux2Transformer2DModel.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)

pipe = Flux2Pipeline.from_pretrained(
    "diffusers/FLUX.2-dev-bnb-4bit",
    text_encoder=None,      # use remote text encoder
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
# Use HuggingFace remote text encoder endpoint for prompt encoding
```

### Technique 9: Pre-Quantized Checkpoints (Skip Startup RAM)

Avoid the ~50GB RAM spike during quantization by using pre-quantized models:

- `diffusers/FLUX.2-dev-bnb-4bit` -- pre-quantized FLUX.2
- `Kijai/flux-fp8/flux1-dev-fp8.safetensors` -- pre-quantized FLUX.1 FP8
- `mit-han-lab/svdq-int4-flux.1-dev` -- pre-quantized SVDQuant INT4

### Memory Optimization Summary

| Technique | VRAM | Speed | Quality | Complexity |
|-----------|------|-------|---------|------------|
| BF16 (baseline) | 33GB | Fast | Best | Low |
| CPU offload | 24GB | Medium | Best | Low |
| Sequential offload + tiling | 6-8GB | Very slow | Best | Low |
| 8-bit (bnb) | 18-20GB | Medium | Very good | Medium |
| FP8 (quanto) | 16-18GB | Medium | Very good | Medium |
| NF4 (bnb) | 12-14GB | Slower | Good | Medium |
| SVDQuant INT4 (Nunchaku) | ~10GB | **Fast** | Good | Medium |
| Group offloading | Variable | Variable | Best | Medium |
| NF4 + group offload | ~8GB | Slow | Good | High |

**Recommendation for logo generation on consumer hardware:**
- **RTX 4090 (24GB)**: FP8 quantization or CPU offload -- best quality
- **RTX 4070 Ti (16GB)**: SVDQuant/Nunchaku -- best speed/quality/memory balance
- **RTX 4060 (8GB)**: NF4 + sequential offload + VAE tiling (slow but works)
- **No GPU / low VRAM**: Use API (fal.ai with LoRA support, ~$0.025/image)

---

## Sources

- [FLUX Models Comparison Guide](https://www.techlifeadventures.com/post/flux-1-comparison-pro-dev-and-schnell-models)
- [FLUX Schnell vs Dev vs Pro vs Max](https://melies.co/compare/flux-models)
- [Diffusers Flux API Documentation](https://huggingface.co/docs/diffusers/api/pipelines/flux)
- [Diffusers Welcomes FLUX-2](https://huggingface.co/blog/flux-2)
- [Shakker-Labs FLUX.1-dev-LoRA-Logo-Design](https://huggingface.co/Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design)
- [ControlNet with Flux.1 (Diffusers)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_flux)
- [IP-Adapter Documentation](https://huggingface.co/docs/diffusers/en/using-diffusers/ip_adapter)
- [XLabs-AI Flux IP-Adapter](https://huggingface.co/XLabs-AI/flux-ip-adapter)
- [InstantX FLUX.1-dev-IP-Adapter](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter)
- [GPUStack Recommended Parameters](https://docs.gpustack.ai/0.4/tutorials/recommended-parameters-for-image-generation-models/)
- [Together AI Pricing](https://www.together.ai/pricing)
- [Together AI Flux Models](https://www.together.ai/models)
- [fal.ai Flux API](https://fal.ai/flux)
- [fal.ai Pricing](https://fal.ai/docs/platform-apis/v1/models/pricing)
- [Replicate Flux Collection](https://replicate.com/collections/flux)
- [Replicate FLUX.2 Pro](https://replicate.com/black-forest-labs/flux-2-pro)
- [BFL Pricing](https://bfl.ai/pricing)
- [AI Image Model Pricing Comparison](https://pricepertoken.com/image)
- [SVDQuant Blog Post](https://hanlab.mit.edu/blog/svdquant)
- [Nunchaku GitHub](https://github.com/nunchaku-ai/nunchaku)
- [Diffusers Quantization Backends](https://huggingface.co/blog/diffusers-quantization)
- [Diffusers Speed/Memory Optimization](https://huggingface.co/docs/diffusers/en/optimization/speed-memory-optims)
- [Flux Limited Resources Gist](https://gist.github.com/sayakpaul/b664605caf0aa3bf8585ab109dd5ac9c)
- [How to Run FLUX Locally (2026)](https://localaimaster.com/blog/flux-local-image-generation)
- [CivitAI Icon & Logos Flux LoRA](https://civitai.com/models/936661/icon-and-logos-flux)
- [CivitAI Flat Icon LoRA](https://civitai.com/models/1149808/gmic-iconf1-flat-icon)
- [Wikipedia: Flux (text-to-image model)](https://en.wikipedia.org/wiki/Flux_(text-to-image_model))
