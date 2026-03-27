# Multi-Generation and Batch Processing for Logo Creation

## Research Date: March 2026

---

## 1. Seed Variation Strategies for Controlled Diversity

### How Seeds Work

The seed parameter initializes the random noise tensor that the diffusion model denoises. Using the same seed with identical parameters produces the same image every time, enabling reproducibility while allowing controlled exploration of the latent space.

### Core Strategies

**Sequential Seed Sweep**
Generate images with seeds `N, N+1, N+2, ...` to explore nearby latent space regions. In batch generation, each image is assigned a sequential seed starting from the specified value. This produces diverse but not radically different outputs.

**Variation Seed Blending (AUTOMATIC1111 / ComfyUI)**
Two parameters control this:
- **Variation Seed**: A second seed value that blends with the primary seed
- **Variation Strength**: Float from 0.0 to 1.0 controlling blend intensity
  - 0.0 = pure original seed image
  - 1.0 = pure variation seed image
  - 0.05-0.3 = subtle variations preserving overall composition

This allows fine-grained exploration around a "good" seed without losing the core composition.

**Random Seed Sampling**
Use `random.sample(range(0, 2**32), count)` to pick seeds spread across the full latent space. Best for initial exploration when you have no baseline.

**Golden Seed Libraries**
Maintain a curated dictionary of seeds known to produce good results for specific styles:

```python
LOGO_SEEDS = {
    "clean_minimal": [42, 1337, 8675309, 2847561],
    "bold_geometric": [314159, 271828, 161803],
    "organic_flowing": [999999, 5555555, 7777777],
}
```

**Deterministic Grid Search**
For production pipelines, generate a grid crossing seeds with other parameters:

```python
seeds = [42, 123, 456, 789, 1024]
cfg_values = [5.0, 7.5, 10.0]
combinations = [(s, c) for s in seeds for c in cfg_values]  # 15 variants
```

### Sampler Considerations

Ancestral samplers (Euler A, DPM++ 2M SDE) introduce additional randomness per step, so seed reproducibility is less deterministic. For strict seed control, use non-ancestral samplers (Euler, DPM++ 2M, DDIM).

---

## 2. Batch Generation with Different Parameters

### Key Parameters for Logo Generation

| Parameter | Logo-Optimal Range | Effect |
|-----------|-------------------|--------|
| CFG Scale | 5-9 (diffusion), 1-2 (LCM/Turbo) | Prompt adherence vs. creativity |
| Steps | 20-30 (standard), 4-8 (LCM/Turbo/Schnell) | Detail level and coherence |
| LoRA Weight | 0.5-1.0 | Style influence strength |
| Resolution | 1024x1024 (SDXL/Flux) | Base canvas size |
| Sampler | DPM++ 2M Karras | Balance of speed and quality |

### CFG Scale Deep Dive

- **2-5**: Abstract, loose interpretation. Good for creative exploration.
- **5-9**: Sweet spot for logos. Strong prompt adherence with natural-looking results.
- **10-15**: Very literal prompt following. Can over-saturate colors and produce artifacts.
- **15+**: Typically produces artifacts, burned colors, distorted shapes. Avoid for logos.

For fast-inference models (LCM-LoRA, SDXL Turbo, Flux Schnell), CFG must be set to 1.0-2.0 or artifacts will appear.

### LoRA Weight Variation

LoRA weights control how strongly a fine-tuned style is applied:
- **0.3-0.5**: Subtle influence, blended with base model style
- **0.6-0.8**: Clear style application while maintaining prompt coherence
- **0.9-1.0**: Strong style override, may reduce prompt flexibility
- **>1.0**: Over-application, typically causes artifacts (some LoRAs tolerate up to 1.5)

Multiple LoRAs can be combined with independent weights:
```
<lora:logo_style:0.7> <lora:flat_design:0.4> <lora:sharp_text:0.5>
```

### Parameter Sweep Pipeline

```python
from dataclasses import dataclass
from itertools import product

@dataclass
class GenerationConfig:
    prompt: str
    seed: int
    cfg_scale: float
    steps: int
    lora_weight: float
    width: int = 1024
    height: int = 1024

def build_sweep_configs(
    prompt: str,
    seeds: list[int],
    cfg_values: list[float],
    step_values: list[int],
    lora_weights: list[float],
) -> list[GenerationConfig]:
    """Generate all combinations for a parameter sweep."""
    configs = []
    for seed, cfg, steps, lora_w in product(seeds, cfg_values, step_values, lora_weights):
        configs.append(GenerationConfig(
            prompt=prompt,
            seed=seed,
            cfg_scale=cfg,
            steps=steps,
            lora_weight=lora_w,
        ))
    return configs

# Example: 3 seeds x 3 CFG x 2 steps x 2 LoRA = 36 variants
configs = build_sweep_configs(
    prompt="minimalist geometric logo for tech startup, clean lines, flat design",
    seeds=[42, 123, 789],
    cfg_values=[5.0, 7.0, 9.0],
    step_values=[20, 30],
    lora_weights=[0.6, 0.9],
)
print(f"Total variants to generate: {len(configs)}")
```

---

## 3. Multi-Model Comparison Pipeline

### Model Landscape (March 2026)

| Model | Elo Score | Price/Image | Best For |
|-------|-----------|-------------|----------|
| Flux 2 Pro v1.1 | 1,265 | $0.055 | Premium creative work |
| GPT Image 1.5 | 1,264 | $0.04 | Quality leader, multimodal |
| Gemini 3 Pro Image | 1,252 | $0.035 | Editing, batch discount |
| Flux 2 Dev | 1,245 | $0.025 | Open-weight, self-hostable |
| Hunyuan Image 3.0 | 1,238 | $0.030 | Cost-effective quality |
| Flux 2 Schnell | 1,232 | $0.015 | Speed-optimized |
| Seedream 4.5 | 1,225 | $0.035 | Text rendering |
| Imagen 4 Fast | ~1,220 | $0.02 | Best price-to-quality |
| Ideogram 2.0 | 1,218 | $0.04 | Text in logos |
| Recraft V4 | N/A | $0.04-0.08 | Native SVG, vector logos |

### Pipeline Architecture

```python
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GenerationResult:
    model_name: str
    prompt: str
    image_bytes: bytes
    seed: int | None
    generation_time_s: float
    cost_usd: float
    metadata: dict

class ImageGenerator(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        ...

class OpenAIGenerator(ImageGenerator):
    def __init__(self, client):
        self.client = client

    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        import time, base64
        start = time.monotonic()
        result = self.client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            quality=kwargs.get("quality", "medium"),
            size="1024x1024",
        )
        elapsed = time.monotonic() - start
        img_b64 = result.data[0].b64_json
        return GenerationResult(
            model_name="gpt-image-1",
            prompt=prompt,
            image_bytes=base64.b64decode(img_b64),
            seed=None,
            generation_time_s=elapsed,
            cost_usd=0.042,
            metadata={"quality": kwargs.get("quality", "medium")},
        )

class FalAIGenerator(ImageGenerator):
    """Supports Flux, Recraft, and other fal.ai-hosted models."""

    def __init__(self, model_id: str, model_name: str, cost: float):
        self.model_id = model_id
        self.model_name = model_name
        self.cost = cost

    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        import fal_client, time
        start = time.monotonic()
        result = await fal_client.subscribe_async(
            self.model_id,
            arguments={
                "prompt": prompt,
                "seed": kwargs.get("seed"),
                "guidance_scale": kwargs.get("cfg_scale", 7.0),
                "num_inference_steps": kwargs.get("steps", 28),
                "image_size": "square_hd",
            },
        )
        elapsed = time.monotonic() - start
        # Download the image from the returned URL
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(result["images"][0]["url"]) as resp:
                image_bytes = await resp.read()
        return GenerationResult(
            model_name=self.model_name,
            prompt=prompt,
            image_bytes=image_bytes,
            seed=kwargs.get("seed"),
            generation_time_s=elapsed,
            cost_usd=self.cost,
            metadata=result,
        )

class GoogleImaGenGenerator(ImageGenerator):
    def __init__(self, client):
        self.client = client

    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        import time
        start = time.monotonic()
        result = self.client.models.generate_images(
            model="imagen-4.0-generate-001",
            prompt=prompt,
            config={"number_of_images": 1},
        )
        elapsed = time.monotonic() - start
        return GenerationResult(
            model_name="imagen-4",
            prompt=prompt,
            image_bytes=result.generated_images[0].image.image_bytes,
            seed=None,
            generation_time_s=elapsed,
            cost_usd=0.04,
            metadata={},
        )

class MultiModelPipeline:
    def __init__(self, generators: list[ImageGenerator]):
        self.generators = generators

    async def run_comparison(
        self, prompt: str, **kwargs
    ) -> list[GenerationResult]:
        """Run same prompt through all models concurrently."""
        tasks = [gen.generate(prompt, **kwargs) for gen in self.generators]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out failures
        successful = []
        for r in results:
            if isinstance(r, Exception):
                print(f"Generation failed: {r}")
            else:
                successful.append(r)
        return successful
```

### Usage

```python
import asyncio
from openai import OpenAI
from google import genai

openai_client = OpenAI()
google_client = genai.Client(api_key="...")

pipeline = MultiModelPipeline([
    OpenAIGenerator(openai_client),
    FalAIGenerator("fal-ai/flux-pro/v1.1", "flux-2-pro", 0.055),
    FalAIGenerator("fal-ai/flux/schnell", "flux-2-schnell", 0.015),
    FalAIGenerator("fal-ai/recraft-v4", "recraft-v4", 0.04),
    GoogleImaGenGenerator(google_client),
])

results = asyncio.run(pipeline.run_comparison(
    "minimalist geometric logo for AI startup called 'Nexus', clean vector style"
))

for r in results:
    print(f"{r.model_name}: {r.generation_time_s:.1f}s, ${r.cost_usd}")
    with open(f"output/{r.model_name}.png", "wb") as f:
        f.write(r.image_bytes)
```

---

## 4. Automated Quality Scoring

### Tier 1: CLIP Score (Prompt Alignment)

CLIP score measures how well an image matches its text prompt. Higher score = better alignment.

```python
import torch
from torchmetrics.multimodal.clip_score import CLIPScore

def compute_clip_score(image_tensor: torch.Tensor, prompt: str) -> float:
    """
    Compute CLIP score between an image and prompt.

    Args:
        image_tensor: uint8 tensor [C, H, W] in range [0, 255]
        prompt: The text prompt used to generate the image
    Returns:
        CLIP score (higher = better alignment, typically 20-35 for good images)
    """
    metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
    score = metric(image_tensor.unsqueeze(0), [prompt])
    return score.item()

# Usage with PIL Image
from PIL import Image
from torchvision import transforms

def score_image_file(image_path: str, prompt: str) -> float:
    img = Image.open(image_path).convert("RGB")
    tensor = transforms.ToTensor()(img) * 255
    tensor = tensor.to(torch.uint8)
    return compute_clip_score(tensor, prompt)
```

**Interpreting CLIP Scores:**
- 15-20: Weak alignment, image may not match prompt
- 20-25: Moderate alignment
- 25-30: Good alignment
- 30+: Strong alignment

### Tier 2: Aesthetic Score (Visual Quality)

Uses LAION's CLIP+MLP aesthetic predictor trained on human aesthetic ratings.

```python
import torch
import torch.nn as nn
import clip
from PIL import Image

class AestheticScorer:
    """LAION aesthetic score predictor (CLIP + MLP)."""

    def __init__(self, model_path: str = "sac+logos+ava1-l14-linearMSE.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device
        )
        # Simple linear model on top of CLIP embeddings
        self.mlp = nn.Linear(768, 1)
        state_dict = torch.load(model_path, map_location=self.device)
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device).eval()

    @torch.no_grad()
    def score(self, image: Image.Image) -> float:
        """Return aesthetic score (1-10 scale, higher = more aesthetic)."""
        img_input = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.clip_model.encode_image(img_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        score = self.mlp(embedding.float())
        return score.item()

# Usage
scorer = AestheticScorer("sac+logos+ava1-l14-linearMSE.pth")
score = scorer.score(Image.open("logo.png"))
print(f"Aesthetic score: {score:.2f}/10")
```

**Interpreting Aesthetic Scores:**
- 1-3: Low quality, unappealing
- 4-5: Average quality
- 5-6: Above average, decent
- 6-7: Good quality, visually appealing
- 7+: Excellent aesthetic quality

The `sac+logos+ava1-l14-linearMSE.pth` model is particularly relevant for logo work as it was trained on a combination of SAC (Simulacra Aesthetic Captions), logo images, and AVA (Aesthetic Visual Analysis) datasets.

### Tier 3: LLM Vision Ranking

Use vision-capable LLMs to evaluate logos against specific design criteria.

```python
import base64
from openai import OpenAI

LOGO_EVAL_SYSTEM_PROMPT = """You are an expert logo designer and brand strategist.
Evaluate the provided logo image. Always respond in JSON format only.

Score each criterion from 1-10:
- simplicity: Clean lines, minimal elements, not cluttered
- memorability: Distinctive, easy to recall
- scalability: Would look good at 16px favicon and 1000px billboard
- text_legibility: Any text is clear and readable (or N/A if no text)
- color_harmony: Colors work well together, appropriate contrast
- professionalism: Looks polished, production-ready
- versatility: Works on light/dark backgrounds, monochrome-friendly
- overall: Holistic quality assessment

Also provide:
- strengths: List of 2-3 specific strengths
- weaknesses: List of 2-3 specific weaknesses
- recommendation: "accept", "revise", or "reject"
"""

def evaluate_logo_with_vlm(
    image_path: str,
    brand_context: str = "",
    model: str = "gpt-4o",
) -> dict:
    """Evaluate a logo using a vision LLM."""
    client = OpenAI()

    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode()

    user_prompt = "Evaluate this logo."
    if brand_context:
        user_prompt += f"\n\nBrand context: {brand_context}"

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": LOGO_EVAL_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64_image}"
                }},
            ]},
        ],
    )
    import json
    return json.loads(response.choices[0].message.content)
```

### Multi-Model VLM Consensus

For higher confidence, run the same evaluation through multiple vision models and average the scores:

```python
import asyncio
import anthropic

async def evaluate_with_claude(image_path: str, brand_context: str = "") -> dict:
    """Evaluate a logo using Claude's vision."""
    client = anthropic.AsyncAnthropic()
    import base64, json

    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode()

    message = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=LOGO_EVAL_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64_image,
                }},
                {"type": "text", "text": f"Evaluate this logo.\nBrand context: {brand_context}"},
            ],
        }],
    )
    return json.loads(message.content[0].text)

async def consensus_evaluation(image_path: str, brand_context: str = "") -> dict:
    """Run evaluation across multiple VLMs and average scores."""
    import numpy as np

    results = await asyncio.gather(
        # GPT-4o evaluation (wrapped in async)
        asyncio.to_thread(evaluate_logo_with_vlm, image_path, brand_context, "gpt-4o"),
        evaluate_with_claude(image_path, brand_context),
    )

    score_keys = [
        "simplicity", "memorability", "scalability", "text_legibility",
        "color_harmony", "professionalism", "versatility", "overall",
    ]

    averaged = {}
    for key in score_keys:
        values = [r[key] for r in results if key in r and r[key] != "N/A"]
        averaged[key] = round(np.mean(values), 1) if values else "N/A"

    averaged["individual_results"] = results
    averaged["model_count"] = len(results)
    return averaged
```

### Combined Scoring Pipeline

```python
@dataclass
class LogoQualityReport:
    file_path: str
    clip_score: float
    aesthetic_score: float
    vlm_scores: dict
    composite_score: float

def compute_composite_score(
    clip_score: float,
    aesthetic_score: float,
    vlm_overall: float,
    weights: tuple[float, float, float] = (0.2, 0.3, 0.5),
) -> float:
    """
    Weighted composite score normalized to 0-100.

    Weights default: 20% CLIP, 30% aesthetic, 50% VLM judgment.
    VLM judgment is weighted highest because it evaluates
    logo-specific criteria (scalability, memorability, etc.).
    """
    # Normalize each to 0-100 scale
    clip_norm = min(clip_score / 35 * 100, 100)      # 35 is ~max CLIP score
    aesthetic_norm = aesthetic_score / 10 * 100         # Already 1-10
    vlm_norm = vlm_overall / 10 * 100                  # Already 1-10

    w_clip, w_aes, w_vlm = weights
    return w_clip * clip_norm + w_aes * aesthetic_norm + w_vlm * vlm_norm
```

---

## 5. Grid/Gallery Generation for Comparison

### Contact Sheet Generator

```python
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import math

def create_comparison_grid(
    images: list[tuple[Image.Image, str]],  # (image, label) pairs
    columns: int = 4,
    thumb_size: int = 512,
    padding: int = 20,
    label_height: int = 40,
    bg_color: str = "#f5f5f5",
    font_size: int = 16,
) -> Image.Image:
    """
    Create a labeled comparison grid from a list of images.

    Args:
        images: List of (PIL.Image, label_string) tuples
        columns: Number of columns in the grid
        thumb_size: Size of each thumbnail (square)
        padding: Padding between images
        label_height: Height reserved for text label below each image
        bg_color: Background color
        font_size: Label font size
    Returns:
        PIL.Image of the assembled grid
    """
    n = len(images)
    rows = math.ceil(n / columns)

    cell_w = thumb_size + padding
    cell_h = thumb_size + label_height + padding
    grid_w = columns * cell_w + padding
    grid_h = rows * cell_h + padding

    grid = Image.new("RGB", (grid_w, grid_h), bg_color)
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for idx, (img, label) in enumerate(images):
        row = idx // columns
        col = idx % columns

        # Resize maintaining aspect ratio
        img_copy = img.copy()
        img_copy.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

        # Center the image in the cell
        x = padding + col * cell_w + (thumb_size - img_copy.width) // 2
        y = padding + row * cell_h + (thumb_size - img_copy.height) // 2

        grid.paste(img_copy, (x, y))

        # Draw label centered below
        label_x = padding + col * cell_w + thumb_size // 2
        label_y = padding + row * cell_h + thumb_size + 5
        draw.text(
            (label_x, label_y), label,
            fill="black", font=font, anchor="mt",
        )

    return grid

# Usage: compare across models
def grid_from_generation_results(results: list, output_path: str):
    """Build comparison grid from GenerationResult objects."""
    from io import BytesIO

    image_label_pairs = []
    for r in results:
        img = Image.open(BytesIO(r.image_bytes)).convert("RGB")
        label = f"{r.model_name}\n{r.generation_time_s:.1f}s | ${r.cost_usd:.3f}"
        image_label_pairs.append((img, label))

    grid = create_comparison_grid(image_label_pairs, columns=min(len(results), 4))
    grid.save(output_path, quality=95)
    print(f"Grid saved to {output_path}")
```

### Parameter Sweep Grid

```python
def create_parameter_sweep_grid(
    images: list[tuple[Image.Image, dict]],
    row_param: str,   # e.g., "cfg_scale"
    col_param: str,   # e.g., "seed"
    thumb_size: int = 384,
) -> Image.Image:
    """
    Create a grid where rows and columns represent different parameter values.
    Each image's dict must contain the row_param and col_param keys.
    """
    # Extract unique values for row and column parameters
    row_values = sorted(set(d[row_param] for _, d in images))
    col_values = sorted(set(d[col_param] for _, d in images))

    # Build lookup
    lookup = {}
    for img, params in images:
        key = (params[row_param], params[col_param])
        lookup[key] = img

    padding = 15
    header_size = 50
    cell = thumb_size + padding
    grid_w = header_size + len(col_values) * cell + padding
    grid_h = header_size + len(row_values) * cell + padding

    grid = Image.new("RGB", (grid_w, grid_h), "#ffffff")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    # Column headers
    for ci, cv in enumerate(col_values):
        x = header_size + ci * cell + thumb_size // 2
        draw.text((x, 10), f"{col_param}={cv}", fill="black", font=font, anchor="mt")

    # Row headers and images
    for ri, rv in enumerate(row_values):
        y_label = header_size + ri * cell + thumb_size // 2
        draw.text((5, y_label), f"{row_param}={rv}", fill="black", font=font, anchor="lm")

        for ci, cv in enumerate(col_values):
            key = (rv, cv)
            if key in lookup:
                img = lookup[key].copy()
                img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                x = header_size + ci * cell
                y = header_size + ri * cell
                grid.paste(img, (x, y))

    return grid
```

### HTML Gallery (Interactive)

For richer comparison, generate an HTML gallery with filtering and scoring:

```python
def generate_html_gallery(
    results: list[dict],  # Each has: image_path, model, scores, params
    output_path: str = "gallery.html",
):
    """Generate an interactive HTML comparison gallery."""
    cards_html = ""
    for r in results:
        scores = r.get("scores", {})
        cards_html += f"""
        <div class="card" data-model="{r['model']}"
             data-composite="{scores.get('composite', 0)}">
          <img src="{r['image_path']}" loading="lazy">
          <div class="info">
            <strong>{r['model']}</strong>
            <span>CLIP: {scores.get('clip', 'N/A')}</span>
            <span>Aesthetic: {scores.get('aesthetic', 'N/A')}</span>
            <span>Composite: {scores.get('composite', 'N/A')}</span>
          </div>
          <div class="params">
            seed={r.get('seed', '?')} cfg={r.get('cfg', '?')}
            steps={r.get('steps', '?')}
          </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html><head><style>
  body {{ font-family: system-ui; background: #1a1a1a; color: #eee; padding: 20px; }}
  .controls {{ margin-bottom: 20px; }}
  select, button {{ padding: 8px 16px; margin-right: 10px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
  .card {{ background: #2a2a2a; border-radius: 8px; overflow: hidden; }}
  .card img {{ width: 100%; aspect-ratio: 1; object-fit: cover; }}
  .info, .params {{ padding: 8px 12px; font-size: 13px; }}
  .info span {{ display: block; color: #aaa; }}
  .params {{ color: #666; font-size: 11px; border-top: 1px solid #333; }}
</style></head><body>
  <h1>Logo Generation Comparison</h1>
  <div class="controls">
    <button onclick="sortBy('composite')">Sort by Score</button>
    <button onclick="sortBy('model')">Sort by Model</button>
  </div>
  <div class="grid" id="grid">{cards_html}</div>
  <script>
    function sortBy(attr) {{
      const grid = document.getElementById('grid');
      const cards = [...grid.children];
      cards.sort((a, b) => {{
        if (attr === 'composite')
          return parseFloat(b.dataset.composite) - parseFloat(a.dataset.composite);
        return a.dataset[attr].localeCompare(b.dataset[attr]);
      }});
      cards.forEach(c => grid.appendChild(c));
    }}
  </script>
</body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"Gallery saved to {output_path}")
```

---

## 6. A/B Testing Approaches for Logo Selection

### Automated A/B Testing Framework

**Method 1: Pairwise VLM Tournament**

Run every pair of logo candidates through a vision LLM and ask it to choose the better one. This produces an Elo-like ranking.

```python
import itertools
import random
from collections import defaultdict

async def pairwise_vlm_compare(
    image_a_path: str,
    image_b_path: str,
    brand_context: str,
    model: str = "gpt-4o",
) -> str:  # Returns "A" or "B"
    """Ask a VLM to choose the better logo from a pair."""
    client = OpenAI()
    import base64

    def load_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""You are a professional logo designer.
Compare Logo A and Logo B for the brand: {brand_context}

Consider: simplicity, memorability, scalability, professionalism, and brand fit.

Respond with ONLY "A" or "B" (the better logo), followed by a one-sentence reason."""},
                {"type": "text", "text": "Logo A:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{load_b64(image_a_path)}"}},
                {"type": "text", "text": "Logo B:"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{load_b64(image_b_path)}"}},
            ],
        }],
    )
    answer = response.choices[0].message.content.strip()
    return "A" if answer.startswith("A") else "B"


async def run_tournament(
    image_paths: list[str],
    brand_context: str,
) -> list[tuple[str, int]]:
    """
    Run a round-robin tournament comparing all pairs.
    Returns sorted list of (image_path, win_count).
    """
    wins = defaultdict(int)
    pairs = list(itertools.combinations(range(len(image_paths)), 2))

    # Randomize presentation order to reduce position bias
    for i, j in pairs:
        if random.random() > 0.5:
            a_idx, b_idx = i, j
        else:
            a_idx, b_idx = j, i

        winner = await pairwise_vlm_compare(
            image_paths[a_idx], image_paths[b_idx], brand_context
        )
        winning_idx = a_idx if winner == "A" else b_idx
        wins[image_paths[winning_idx]] += 1

    ranked = sorted(wins.items(), key=lambda x: x[1], reverse=True)
    return ranked
```

**Method 2: Multi-Criteria Weighted Score**

Rather than pairwise comparison, score each logo independently and rank by composite:

| Criterion | Weight | Measurement |
|-----------|--------|-------------|
| Prompt alignment | 15% | CLIP score |
| Visual quality | 20% | LAION aesthetic score |
| Simplicity | 15% | VLM assessment |
| Memorability | 15% | VLM assessment |
| Scalability | 15% | VLM + automated resize test |
| Text legibility | 10% | VLM + OCR confidence |
| Brand fit | 10% | VLM with brand context |

**Method 3: Human-in-the-Loop A/B Testing**

For final selection, present the top 3-5 AI-ranked logos to human evaluators:
- Present pairs randomly to a panel (50-100 respondents for statistical confidence)
- Measure: preference rate, recall after delay, emotional association
- Tools: Poll the People, UsabilityHub, SurveyMonkey, or a simple custom web form
- Combine human preference data with automated scores for final decision

**Method 4: Automated Scalability Test**

```python
def test_logo_scalability(image_path: str) -> dict:
    """Test a logo at multiple sizes to check if it remains recognizable."""
    from PIL import Image, ImageFilter
    import imagehash

    img = Image.open(image_path).convert("RGB")
    original_hash = imagehash.phash(img)

    sizes = [512, 256, 128, 64, 32, 16]
    results = {}

    for size in sizes:
        resized = img.resize((size, size), Image.Resampling.LANCZOS)
        # Re-enlarge to compare
        enlarged = resized.resize((512, 512), Image.Resampling.LANCZOS)
        test_hash = imagehash.phash(enlarged)
        similarity = 1 - (original_hash - test_hash) / 64  # Normalize to 0-1
        results[f"{size}px"] = round(similarity, 3)

    results["scalability_score"] = round(
        sum(results.values()) / len(results), 3
    )
    return results
```

---

## 7. Parallelizing Generation Across Multiple APIs

### Architecture Overview

```
                    +------------------+
                    |  Orchestrator    |
                    | (asyncio event   |
                    |  loop)           |
                    +--------+---------+
                             |
            +----------------+----------------+
            |                |                |
    +-------v------+  +-----v--------+  +----v---------+
    | OpenAI API   |  | fal.ai API   |  | Google API   |
    | (GPT Image)  |  | (Flux,Recraft)|  | (Imagen)     |
    +-------+------+  +-----+--------+  +----+---------+
            |                |                |
            +----------------+----------------+
                             |
                    +--------v---------+
                    | Result Collector  |
                    | & Quality Scorer  |
                    +------------------+
```

### Concurrency Control with Semaphores

```python
import asyncio
import aiohttp
from dataclasses import dataclass

@dataclass
class RateLimitConfig:
    requests_per_minute: int
    max_concurrent: int

API_LIMITS = {
    "openai": RateLimitConfig(requests_per_minute=50, max_concurrent=5),
    "fal_ai": RateLimitConfig(requests_per_minute=100, max_concurrent=10),
    "google": RateLimitConfig(requests_per_minute=60, max_concurrent=5),
    "replicate": RateLimitConfig(requests_per_minute=60, max_concurrent=10),
    "together": RateLimitConfig(requests_per_minute=60, max_concurrent=10),
}

class RateLimitedClient:
    """API client with semaphore-based concurrency and rate limiting."""

    def __init__(self, provider: str):
        config = API_LIMITS[provider]
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.min_interval = 60.0 / config.requests_per_minute
        self._last_request = 0.0

    async def throttled_request(self, coro):
        """Execute a coroutine with rate limiting and concurrency control."""
        import time
        async with self.semaphore:
            # Enforce minimum interval between requests
            now = time.monotonic()
            wait = self._last_request + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request = time.monotonic()

            # Retry with exponential backoff
            for attempt in range(3):
                try:
                    return await coro
                except Exception as e:
                    if attempt == 2:
                        raise
                    wait_time = 2 ** attempt
                    print(f"Retry {attempt+1}/3 after {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
```

### Full Parallel Pipeline

```python
import asyncio
import json
import time
from pathlib import Path

class ParallelLogoPipeline:
    """
    End-to-end pipeline: generate across APIs, score, rank, and produce gallery.
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_batch(
        self,
        prompt: str,
        seeds: list[int],
        generators: dict[str, ImageGenerator],
    ) -> list[GenerationResult]:
        """Generate images across all models and seeds concurrently."""
        tasks = []
        for model_name, gen in generators.items():
            for seed in seeds:
                tasks.append(gen.generate(prompt, seed=seed))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = []
        for r in results:
            if isinstance(r, Exception):
                print(f"Failed: {r}")
            else:
                successful.append(r)
                # Save image
                fname = f"{r.model_name}_seed{r.seed}.png"
                path = self.output_dir / fname
                with open(path, "wb") as f:
                    f.write(r.image_bytes)
                r.metadata["saved_path"] = str(path)

        return successful

    async def score_all(
        self, results: list[GenerationResult], prompt: str
    ) -> list[dict]:
        """Score all generated images with CLIP + aesthetic + VLM."""
        scored = []
        for r in results:
            path = r.metadata.get("saved_path")
            if not path:
                continue

            # CLIP and aesthetic scores (CPU-bound, run in thread pool)
            clip_s = await asyncio.to_thread(score_image_file, path, prompt)
            aesthetic_s = await asyncio.to_thread(
                lambda p: AestheticScorer().score(Image.open(p)), path
            )

            # VLM score (async API call)
            vlm_s = await asyncio.to_thread(
                evaluate_logo_with_vlm, path, prompt
            )

            composite = compute_composite_score(
                clip_s, aesthetic_s, vlm_s.get("overall", 5)
            )

            scored.append({
                "model": r.model_name,
                "seed": r.seed,
                "image_path": path,
                "clip_score": round(clip_s, 2),
                "aesthetic_score": round(aesthetic_s, 2),
                "vlm_scores": vlm_s,
                "composite_score": round(composite, 2),
                "generation_time": r.generation_time_s,
                "cost": r.cost_usd,
            })

        # Sort by composite score
        scored.sort(key=lambda x: x["composite_score"], reverse=True)
        return scored

    async def run(
        self,
        prompt: str,
        seeds: list[int],
        generators: dict[str, ImageGenerator],
    ) -> list[dict]:
        """Full pipeline: generate, score, rank, create gallery."""
        print(f"Generating {len(seeds)} seeds x {len(generators)} models "
              f"= {len(seeds) * len(generators)} images...")

        start = time.monotonic()
        results = await self.generate_batch(prompt, seeds, generators)
        gen_time = time.monotonic() - start
        print(f"Generation complete in {gen_time:.1f}s")

        print("Scoring all images...")
        scored = await self.score_all(results, prompt)

        # Save results JSON
        report_path = self.output_dir / "results.json"
        with open(report_path, "w") as f:
            json.dump(scored, f, indent=2, default=str)

        # Create comparison grid
        images_for_grid = []
        for s in scored[:16]:  # Top 16 for grid
            img = Image.open(s["image_path"]).convert("RGB")
            label = f"{s['model']} (s={s['seed']})\nScore: {s['composite_score']}"
            images_for_grid.append((img, label))

        grid = create_comparison_grid(images_for_grid, columns=4)
        grid_path = self.output_dir / "comparison_grid.png"
        grid.save(str(grid_path), quality=95)

        # Generate HTML gallery
        generate_html_gallery(scored, str(self.output_dir / "gallery.html"))

        print(f"\nTop 5 results:")
        for i, s in enumerate(scored[:5]):
            print(f"  {i+1}. {s['model']} seed={s['seed']} "
                  f"composite={s['composite_score']:.1f}")

        return scored
```

### Running the Pipeline

```python
async def main():
    from openai import OpenAI

    pipeline = ParallelLogoPipeline(output_dir="logo_output")

    generators = {
        "gpt-image-1": OpenAIGenerator(OpenAI()),
        "flux-2-pro": FalAIGenerator("fal-ai/flux-pro/v1.1", "flux-2-pro", 0.055),
        "flux-2-schnell": FalAIGenerator("fal-ai/flux/schnell", "flux-2-schnell", 0.015),
        "recraft-v4": FalAIGenerator("fal-ai/recraft-v4", "recraft-v4", 0.04),
    }

    scored = await pipeline.run(
        prompt=(
            "minimalist geometric logo for tech company 'Nexus', "
            "clean vector style, flat design, white background, "
            "professional, simple, memorable"
        ),
        seeds=[42, 123, 456, 789, 1024],
        generators=generators,
    )

    # Cost summary
    total_cost = sum(s["cost"] for s in scored)
    print(f"\nTotal generation cost: ${total_cost:.2f}")
    print(f"Total images: {len(scored)}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 8. Batch Pipeline: Complete Working Example

### Dependencies

```
pip install openai anthropic fal-client aiohttp pillow torch torchvision
pip install torchmetrics transformers clip imagehash
```

### Minimal Self-Contained Batch Script

```python
#!/usr/bin/env python3
"""
logo_batch.py - Batch logo generation, scoring, and comparison pipeline.

Usage:
    python logo_batch.py --prompt "logo for ..." --seeds 5 --output ./output

Requires API keys in environment:
    OPENAI_API_KEY, FAL_KEY
"""

import argparse
import asyncio
import base64
import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenRequest:
    prompt: str
    seed: int
    model: str
    provider: str
    cfg_scale: float = 7.0
    steps: int = 28

@dataclass
class GenResult:
    request: GenRequest
    image_bytes: bytes = field(repr=False)
    elapsed_s: float = 0.0
    cost_usd: float = 0.0
    file_path: str = ""

@dataclass
class ScoredResult:
    file_path: str
    model: str
    seed: int
    clip_score: float = 0.0
    aesthetic_score: float = 0.0
    vlm_overall: float = 0.0
    composite: float = 0.0
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

async def generate_openai(req: GenRequest) -> GenResult:
    from openai import OpenAI
    client = OpenAI()
    start = time.monotonic()
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=req.prompt,
        quality="medium",
        size="1024x1024",
    )
    elapsed = time.monotonic() - start
    img_bytes = base64.b64decode(resp.data[0].b64_json)
    return GenResult(req, img_bytes, elapsed, cost_usd=0.042)


async def generate_fal(req: GenRequest) -> GenResult:
    import fal_client
    model_map = {
        "flux-2-pro": ("fal-ai/flux-pro/v1.1", 0.055),
        "flux-2-schnell": ("fal-ai/flux/schnell", 0.015),
        "recraft-v4": ("fal-ai/recraft-v4", 0.04),
    }
    model_id, cost = model_map[req.model]
    start = time.monotonic()
    result = await fal_client.subscribe_async(
        model_id,
        arguments={
            "prompt": req.prompt,
            "seed": req.seed,
            "guidance_scale": req.cfg_scale,
            "num_inference_steps": req.steps,
            "image_size": "square_hd",
        },
    )
    elapsed = time.monotonic() - start
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.get(result["images"][0]["url"]) as r:
            img_bytes = await r.read()
    return GenResult(req, img_bytes, elapsed, cost_usd=cost)


GENERATORS = {
    "openai": generate_openai,
    "fal": generate_fal,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def quick_clip_score(image_path: str, prompt: str) -> float:
    """CLIP score via torchmetrics (requires GPU for speed)."""
    try:
        import torch
        from torchmetrics.multimodal.clip_score import CLIPScore
        from torchvision import transforms

        img = Image.open(image_path).convert("RGB")
        tensor = (transforms.ToTensor()(img) * 255).to(torch.uint8)
        metric = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14")
        return metric(tensor.unsqueeze(0), [prompt]).item()
    except Exception as e:
        print(f"CLIP score failed: {e}")
        return 0.0


def quick_vlm_score(image_path: str, prompt: str) -> float:
    """Fast VLM evaluation via GPT-4o-mini."""
    try:
        from openai import OpenAI
        client = OpenAI()
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Rate this logo 1-10 for overall quality "
                        "(simplicity, memorability, professionalism). "
                        "Respond with ONLY a number."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        )
        return float(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"VLM score failed: {e}")
        return 5.0


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(
    scored: list[ScoredResult],
    output_path: str,
    cols: int = 4,
    thumb: int = 400,
):
    n = len(scored)
    rows = math.ceil(n / cols)
    pad, label_h = 15, 50
    w = cols * (thumb + pad) + pad
    h = rows * (thumb + label_h + pad) + pad

    canvas = Image.new("RGB", (w, h), "#f0f0f0")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    for idx, sr in enumerate(scored):
        r, c = divmod(idx, cols)
        img = Image.open(sr.file_path).convert("RGB")
        img.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        x = pad + c * (thumb + pad)
        y = pad + r * (thumb + label_h + pad)
        canvas.paste(img, (x, y))
        label = (
            f"{sr.model} s={sr.seed}\n"
            f"CLIP={sr.clip_score:.1f} VLM={sr.vlm_overall:.0f} "
            f"C={sr.composite:.1f}"
        )
        draw.text((x, y + thumb + 2), label, fill="black", font=font)

    canvas.save(output_path, quality=95)
    print(f"Grid saved: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    prompt: str,
    num_seeds: int,
    output_dir: str,
    models: list[str] | None = None,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = ["flux-2-schnell"]

    model_provider = {
        "gpt-image-1": "openai",
        "flux-2-pro": "fal",
        "flux-2-schnell": "fal",
        "recraft-v4": "fal",
    }

    seeds = [random.randint(0, 2**32 - 1) for _ in range(num_seeds)]
    requests = [
        GenRequest(prompt=prompt, seed=s, model=m, provider=model_provider[m])
        for m in models
        for s in seeds
    ]

    print(f"Generating {len(requests)} images "
          f"({len(models)} models x {num_seeds} seeds)...")

    # --- Generate concurrently ---
    sem = asyncio.Semaphore(8)

    async def guarded_generate(req: GenRequest) -> GenResult | None:
        async with sem:
            try:
                gen_fn = GENERATORS[req.provider]
                result = await gen_fn(req)
                fname = f"{req.model}_s{req.seed}.png"
                path = out / fname
                with open(path, "wb") as f:
                    f.write(result.image_bytes)
                result.file_path = str(path)
                return result
            except Exception as e:
                print(f"FAILED {req.model} seed={req.seed}: {e}")
                return None

    gen_results = await asyncio.gather(
        *[guarded_generate(r) for r in requests]
    )
    gen_results = [r for r in gen_results if r is not None]
    print(f"Successfully generated {len(gen_results)} images")

    # --- Score ---
    print("Scoring images...")
    scored: list[ScoredResult] = []
    for gr in gen_results:
        clip_s = quick_clip_score(gr.file_path, prompt)
        vlm_s = quick_vlm_score(gr.file_path, prompt)
        composite = 0.3 * (clip_s / 35 * 10) + 0.7 * vlm_s
        scored.append(ScoredResult(
            file_path=gr.file_path,
            model=gr.request.model,
            seed=gr.request.seed,
            clip_score=clip_s,
            vlm_overall=vlm_s,
            composite=composite,
            cost_usd=gr.cost_usd,
        ))

    scored.sort(key=lambda x: x.composite, reverse=True)

    # --- Output ---
    build_grid(scored, str(out / "grid.png"))

    report = {
        "prompt": prompt,
        "total_images": len(scored),
        "total_cost": round(sum(s.cost_usd for s in scored), 3),
        "results": [asdict(s) for s in scored],
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nTop 3:")
    for i, s in enumerate(scored[:3]):
        print(f"  {i+1}. {s.model} seed={s.seed} "
              f"composite={s.composite:.2f}")

    print(f"\nTotal cost: ${report['total_cost']:.3f}")
    print(f"Output: {out.resolve()}")
    return scored


def main():
    parser = argparse.ArgumentParser(description="Batch logo generation pipeline")
    parser.add_argument("--prompt", required=True, help="Logo prompt")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--output", default="./logo_output", help="Output dir")
    parser.add_argument(
        "--models", nargs="+",
        default=["flux-2-schnell"],
        choices=["gpt-image-1", "flux-2-pro", "flux-2-schnell", "recraft-v4"],
        help="Models to use",
    )
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.prompt, args.seeds, args.output, args.models))


if __name__ == "__main__":
    main()
```

---

## Cost Estimation Guide

| Scenario | Models | Seeds | Total Images | Est. Cost |
|----------|--------|-------|--------------|-----------|
| Quick test | 1 (Schnell) | 5 | 5 | $0.08 |
| Standard sweep | 3 models | 5 | 15 | $0.55 |
| Full comparison | 5 models | 10 | 50 | $2.00 |
| Production run | 5 models | 20 | 100 | $4.00 |
| + VLM scoring (GPT-4o-mini) | - | - | per image | ~$0.002/img |
| + VLM scoring (GPT-4o) | - | - | per image | ~$0.01/img |

Add approximately 10-20% overhead for retries and failed generations.

### Batch API Discounts

- **Google Gemini 3 Pro Image Batch API**: 50% discount for async processing (results within 24 hours instead of seconds). Reduces per-image cost from $0.035 to $0.0175.
- **xAI Grok Batch API**: Supports batch image generation via JSONL file upload with similar async discount model.

---

## Recommended Workflow

1. **Explore** (5 min, ~$0.50): Run prompt through 3 fast models (Flux Schnell, Imagen Fast, GPT Image Mini) with 5 random seeds each. Create comparison grid.

2. **Refine** (10 min, ~$1.50): Take the top 2-3 seeds/styles. Run through premium models (Flux Pro, GPT Image 1.5, Recraft V4) with parameter variations (3 CFG values).

3. **Score** (2 min, ~$0.30): Run CLIP + aesthetic + VLM scoring on all results. Auto-rank by composite score.

4. **Select** (5 min, ~$0.10): Run pairwise VLM tournament on top 5 candidates. Present top 3 to human decision-maker with score annotations.

5. **Finalize**: Send winner to Recraft V4 for SVG vector output, or to a vectorization pipeline.

---

## Sources

- [Seed Management - AUTOMATIC1111 DeepWiki](https://deepwiki.com/AUTOMATIC1111/stable-diffusion-webui-feature-showcase/2.4-seed-management-and-variations)
- [Guide to Seed in Stable Diffusion - getimg.ai](https://getimg.ai/guides/guide-to-seed-parameter-in-stable-diffusion)
- [CLIP Score - PyTorch Metrics](https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html)
- [Improved Aesthetic Predictor - GitHub](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [LAION Aesthetic Predictor - GitHub](https://github.com/LAION-AI/aesthetic-predictor)
- [AI Image Generation API Comparison 2026](https://blog.laozhang.ai/en/posts/ai-image-generation-api-comparison-2026)
- [Vision LLM Image Quality Evaluation (9 models) - Medium](https://garystafford.medium.com/evaluating-image-quality-using-nine-different-multimodal-generative-ai-vision-models-a1044de936e3)
- [A/B Testing for Logo Design - LogoDiffusion](https://logodiffusion.com/blog/a-b-testing-for-logo-design-prompts)
- [Logo Rank - AI Logo Testing](https://brandmark.io/logo-rank/)
- [CFG Scale Explained - ArtSmart](https://artsmart.ai/blog/what-is-cfg-scale/)
- [Guide to CFG Scale - getimg.ai](https://getimg.ai/guides/interactive-guide-to-stable-diffusion-guidance-scale-parameter)
- [fal.ai API Guide 2026](https://www.glmimages.com/blog/fal-ai-api-guide-2026)
- [Asyncio Parallel API Requests - Medium](https://medium.com/@ghaelen.m/how-to-run-multiple-parallel-api-requests-to-llm-apis-without-freezing-your-cpu-in-python-asyncio-af0da7e240e3)
- [Best Vision Models January 2026 - WhatLLM](https://whatllm.org/blog/best-vision-models-january-2026)
- [Batch AI Image Generation - MindStudio](https://www.mindstudio.ai/blog/batch-ai-image-generation-hundreds-visuals-minutes)
