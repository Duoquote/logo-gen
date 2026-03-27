# Image Generation Models for Logo Creation
## Research Notes - March 2026

---

## 1. Models Available via OpenRouter API

OpenRouter provides image generation through the `/api/v1/chat/completions` endpoint using the `modalities` parameter (set to `["image"]` or `["image", "text"]`). Images are returned as base64-encoded PNG data URLs.

### Confirmed OpenRouter Image Models

| Model | Model ID | Pricing | Notes |
|-------|----------|---------|-------|
| FLUX.2 Pro | `black-forest-labs/flux.2-pro` | $0.03/MP first, $0.015/MP after | Image-only output, up to 4MP |
| FLUX.2 Klein 4B | `black-forest-labs/flux.2-klein` | ~$0.014/MP first | Fastest/cheapest in FLUX.2 family |
| GPT-5 Image | `openai/gpt-5-image` | $10/M input tokens, $40/M image output tokens | 400K context, text+image output |
| GPT-5 Image Mini | `openai/gpt-5-image-mini` | Lower than GPT-5 Image | Cost-efficient variant |
| Gemini 3.1 Flash Image (Nano Banana 2) | `google/gemini-3.1-flash-image` | ~$0.0000003/token in, $0.00003/token out | Pro-level quality at Flash speed |
| Gemini 2.5 Flash Image (Nano Banana) | `google/gemini-2.5-flash-image` | ~$0.0000003/token in, $0.00003/token out | Multi-turn image conversations |
| Gemini 3 Pro Image (Nano Banana Pro) | `google/nano-banana-pro` | Higher than Flash | 2K/4K output, best text rendering |
| Seedream 4.5 | (ByteDance) | TBD | Small-text rendering, portrait refinement |

### NOT on OpenRouter (as of March 2026)

- **Recraft V4** - Not available on OpenRouter. Use Recraft API directly, fal.ai, or WaveSpeedAI.
- **Ideogram 3.0** - Not available on OpenRouter. Use Ideogram API directly, Together AI, or Segmind.
- **Stable Diffusion 3.5** - Not available on OpenRouter. Use Stability AI API directly, fal.ai, or self-host.

### OpenRouter Configuration Options

- **Aspect ratios**: 1:1, 3:2, 2:3, 4:3, 3:4, 16:9, 9:16, plus extended (1:4, 4:1, 1:8, 8:1) for select models
- **Image size**: 0.5K, 1K (default), 2K, 4K
- **Font inputs** (Sourceful/Riverflow only): Custom text with specific fonts, +$0.03/font input
- **Super Resolution references** (Sourceful only): Up to 4 reference URLs, $0.20/reference

---

## 2. Model-by-Model Analysis for Logo Generation

### Recraft V4 -- BEST FOR LOGOS

**Released**: February 2026

**API Endpoint**:
```
POST https://external.api.recraft.ai/v1/images/generations
Authorization: Bearer RECRAFT_API_TOKEN
```

**Pricing**:
| Variant | Price | Resolution |
|---------|-------|-----------|
| Recraft V4 (raster) | $0.04/image | 1MP |
| Recraft V4 Vector (SVG) | $0.04/image | 1MP |
| Recraft V4 Pro (raster) | $0.08/image | 4MP |
| Recraft V4 Pro Vector (SVG) | $0.25/image | 4MP |

**Logo Quality**: Best in class. #1 on HuggingFace Text-to-Image Arena. Design-aware composition with strong brand aesthetics. Treats typography as a structural element of the composition, not an overlay.

**Text Rendering**: Excellent. Accurate letterforms, proper kerning, appropriate weight. Text interacts with the scene -- bridging visual elements and fitting naturally into the design.

**Key Differentiator**: Only model producing **native SVG output** -- actual editable vector paths, structured layers, clean geometry. Opens directly in Illustrator, Figma, or Sketch. This is critical for logo work where vector output is a production requirement.

**Available via**: Recraft API (direct), fal.ai, WaveSpeedAI, Replicate, Vercel AI Gateway

---

### Ideogram 3.0 -- BEST TEXT RENDERING

**Released**: March 2025

**API Endpoint (direct)**:
```
POST https://api.ideogram.ai/generate
Header: Api-Key: YOUR_API_KEY
```

**API Endpoint (Together AI)**:
```
POST https://api.together.xyz/v1/images/generations
Model: ideogram-3-0
```

**Pricing**: ~$0.06/image (varies by quality tier; $0.03-$0.09 on fal.ai depending on tier)

**Logo Quality**: Excellent for branding, logos, and structured professional visuals. Clean compositions with high customization. Supports color palette input with hex codes.

**Text Rendering**: Industry-leading accuracy at 95%+ correct spelling (vs. 30-50% for most competitors). Handles complex multi-element typography, multiple fonts, and sizes reliably.

**Key Differentiator**: Near-perfect text rendering accuracy. If your logo heavily features text/wordmarks, Ideogram is the safest bet for getting the typography right on the first try.

**Available via**: Ideogram API (direct), Together AI, Segmind, fal.ai, Leonardo.AI

---

### FLUX.2 (Black Forest Labs) -- BEST VALUE / OPEN WEIGHT

**Released**: November 2025

**API Endpoint (OpenRouter)**:
```
POST https://openrouter.ai/api/v1/chat/completions
Model: black-forest-labs/flux.2-pro
Modalities: ["image"]
```

**API Endpoint (direct)**:
```
POST https://api.bfl.ai/v1/flux-2-pro
```

**Pricing**:
| Variant | Price | Elo Score | Notes |
|---------|-------|-----------|-------|
| FLUX.2 Schnell | $0.015/image | 1,232 | Fast, 1-4 steps |
| FLUX.2 Dev | $0.025/image | 1,245 | Open weight, self-hostable |
| FLUX.2 Pro v1.1 | $0.055/image | 1,265 | Highest quality |
| FLUX.2 Klein 4B | ~$0.014/MP | -- | Ultra-fast, cost-optimized |

**Logo Quality**: Strong. Sharp textures, consistent style reproduction, strong prompt adherence. Ties with GPT Image 1.5 for quality crown at the Pro tier (Elo 1,265). 32B parameter model.

**Text Rendering**: Good. Excellent for signs, logos, and infographics. Not quite at Ideogram's level for complex multi-text compositions, but reliable for single text elements.

**Key Differentiator**: Best price-to-quality ratio. FLUX.2 Schnell at $0.015/image is the cheapest option that still delivers strong results. Open-weight Dev variant allows self-hosting and fine-tuning.

**Available via**: OpenRouter, BFL API (direct), fal.ai, Together AI, Replicate, DeepInfra, self-hosted

---

### GPT Image 1.5 / GPT-5 Image (OpenAI) -- BEST MULTIMODAL

**Released**: GPT Image 1.5 (Dec 2025), GPT-5 Image (Oct 2025)

**API Endpoint (direct)**:
```
POST https://api.openai.com/v1/images/generations
Model: gpt-image-1.5
```

**API Endpoint (OpenRouter)** (GPT-5 Image):
```
POST https://openrouter.ai/api/v1/chat/completions
Model: openai/gpt-5-image
Modalities: ["image", "text"]
```

**Pricing**:
| Variant | Quality | Price | Elo Score |
|---------|---------|-------|-----------|
| GPT Image 1 Mini | Low | $0.005/image | ~1,200 |
| GPT Image 1 | Low | $0.011/image | ~1,250 |
| GPT Image 1.5 | Standard | $0.04/image | 1,264 |
| GPT Image 1 | High | $0.167/image | ~1,260 |
| GPT-5 Image (OpenRouter) | -- | $10/M in, $40/M img out | -- |

**Logo Quality**: High. Industry-leading Elo score (1,264). Strong instruction following, detailed editing capabilities. Mandatory reasoning capabilities improve prompt interpretation.

**Text Rendering**: Very good. Significantly improved over DALL-E series. Clear text on signage, UI elements, product labels. Can still struggle with complex multi-text placement vs. Ideogram.

**Key Differentiator**: Multimodal reasoning -- can accept text + image inputs together, enabling iterative editing workflows. GPT Image 1 Mini at $0.005 is the absolute cheapest option for draft/exploration.

**Available via**: OpenAI API (direct), OpenRouter (GPT-5 Image variant)

---

### Stable Diffusion 3.5 Large -- BEST FOR CUSTOMIZATION

**Released**: Late 2024 (auto-upgraded from SD 3.0 in April 2025)

**API Endpoint (Stability AI)**:
```
POST https://api.stability.ai/v2beta/stable-image/generate/sd3
```

**Pricing**:
| Variant | Price | Notes |
|---------|-------|-------|
| SD 3.5 Large (via Stability API) | $0.065/image (6.5 credits) | 8B params, MMDiT |
| SD 3.5 Medium | $0.035/image (3.5 credits) | 2.5B params |
| SD 3.5 Large Turbo | ~$0.04/image | 4-step generation |
| Stable Image Ultra (SD3.5) | $0.08/image | Highest quality tier |
| Stable Image Core | $0.03/image | Budget option |

**Logo Quality**: Good. Strong prompt adherence and aesthetic quality (Elo >1,020). Best resolution at 1MP (1024x1024). Not in the top tier against Recraft/Flux/GPT for logos specifically.

**Text Rendering**: Improved over SD 2.x but still the weakest of the five models reviewed. Acceptable for simple text, unreliable for complex typography.

**Key Differentiator**: Open weights allow full self-hosting. Largest ecosystem of LoRAs, ControlNets, and community fine-tunes. Best choice if you need to train custom logo styles or need complete control over the pipeline. Inference at 2.8-3.5s on RTX 4090.

**Available via**: Stability AI API (direct), fal.ai, Replicate, self-hosted (ComfyUI, A1111)

---

## 3. Comparative Summary

### Quality Ranking for Logos (March 2026)

| Rank | Model | Logo Score | Text Score | SVG? | Price Range |
|------|-------|-----------|-----------|------|-------------|
| 1 | Recraft V4 | 10/10 | 9/10 | YES | $0.04-$0.25 |
| 2 | Ideogram 3.0 | 8/10 | 10/10 | No | $0.03-$0.09 |
| 3 | FLUX.2 Pro v1.1 | 9/10 | 8/10 | No | $0.015-$0.055 |
| 4 | GPT Image 1.5 | 8/10 | 8/10 | No | $0.005-$0.167 |
| 5 | SD 3.5 Large | 6/10 | 5/10 | No | $0.03-$0.08 |

### API Aggregator Options

Rather than integrating each API separately, consider using an aggregator:

| Aggregator | Models Available | Advantage |
|-----------|-----------------|-----------|
| **OpenRouter** | FLUX.2, GPT-5 Image, Gemini Image | Unified chat completions API, easy model switching |
| **fal.ai** | 600+ models including Flux, Recraft, Ideogram | 30-50% cheaper, fastest inference, broadest coverage |
| **Together AI** | Flux, Ideogram 3.0 | Simple REST API, competitive pricing |
| **Replicate** | Recraft V4, Flux, SD 3.5 | Pay-per-second compute, good for variable loads |

---

## 4. OpenRouter-Specific Availability

Models accessible through OpenRouter as of March 2026:
- FLUX.2 Pro, FLUX.2 Klein -- full image generation
- GPT-5 Image, GPT-5 Image Mini -- text + image multimodal
- Gemini 3.1 Flash Image, Gemini 2.5 Flash Image, Gemini 3 Pro Image -- Google's image models
- Seedream 4.5 -- ByteDance
- Sourceful Riverflow v2 -- with font input support

**NOT on OpenRouter**: Recraft V4, Ideogram 3.0, Stable Diffusion 3.5

**Implication for logo pipeline**: OpenRouter alone cannot cover the best logo models (Recraft V4, Ideogram 3.0). A multi-provider strategy is required.

---

## 5. Recommended Multi-Model Logo Pipeline Strategy

### Architecture: Three-Stage Pipeline

```
Stage 1: EXPLORATION (cheap, fast, many variations)
  |-- Model: FLUX.2 Schnell ($0.015/image) or GPT Image 1 Mini ($0.005/image)
  |-- Generate 8-16 concept variations per prompt
  |-- Goal: Identify promising aesthetic directions
  |
Stage 2: REFINEMENT (quality text, tight composition)
  |-- Model: Ideogram 3.0 ($0.06/image) for text-heavy logos
  |-- Model: FLUX.2 Pro ($0.055/image) for icon/symbol logos
  |-- Generate 4-8 refined candidates per direction
  |-- Goal: Nail typography and compositional details
  |
Stage 3: PRODUCTION OUTPUT (final vector-ready assets)
  |-- Model: Recraft V4 Vector ($0.04/image) or Recraft V4 Pro Vector ($0.25/image)
  |-- Generate 2-4 final candidates with SVG output
  |-- Goal: Production-ready vector files for Figma/Illustrator
```

### Cost Estimate Per Logo Project

| Stage | Images | Model | Cost |
|-------|--------|-------|------|
| Exploration | 16 | FLUX.2 Schnell | $0.24 |
| Refinement | 8 | Ideogram 3.0 | $0.48 |
| Production | 4 | Recraft V4 Vector | $0.16 |
| **Total** | **28** | | **$0.88** |

For Pro-tier production output ($0.25/vector), total rises to ~$1.48 per logo project.

### Model Selection Logic

```
IF logo_type == "wordmark" or "text-heavy":
    primary = Ideogram 3.0     # best text rendering
    production = Recraft V4     # SVG output

ELIF logo_type == "icon" or "symbol" or "abstract":
    primary = FLUX.2 Pro        # best visual quality for money
    production = Recraft V4     # SVG output

ELIF logo_type == "combination_mark":
    primary = Ideogram 3.0     # text accuracy critical
    secondary = FLUX.2 Pro     # icon exploration
    production = Recraft V4    # SVG output

ELIF budget == "minimal":
    all_stages = GPT Image 1 Mini ($0.005/image)
    # Acceptable quality, no SVG output
```

### Required API Integrations

For a complete logo pipeline, you need at minimum:

1. **FLUX.2** via OpenRouter or BFL direct -- exploration & general generation
2. **Ideogram 3.0** via Ideogram API or Together AI -- text-heavy logos
3. **Recraft V4** via Recraft API or fal.ai -- production SVG output

**Alternatively**: Use **fal.ai** as a single aggregator for all three (Flux, Ideogram, Recraft) under one API key and billing account, at 30-50% lower cost than direct APIs.

### Key Technical Considerations

- **SVG output is only available from Recraft V4**. All other models output raster (PNG/JPEG). For professional logo delivery, you either use Recraft for final output or need a raster-to-vector conversion step (potrace, vectorizer.ai, etc.).
- **Prompt engineering matters more than model choice** for logo quality. Specify: logo style (wordmark, icon, combination), color palette, typography style, geometric constraints, and negative prompts.
- **Aspect ratio**: Use 1:1 for most logos. OpenRouter and direct APIs all support aspect ratio control.
- **Batch generation**: Generate many variations and select, rather than trying to perfect a single prompt.

---

## Sources

- [OpenRouter Image Generation Docs](https://openrouter.ai/docs/guides/overview/multimodal/image-generation)
- [OpenRouter Image Models Collection](https://openrouter.ai/collections/image-models)
- [OpenRouter FLUX.2 Pro](https://openrouter.ai/black-forest-labs/flux.2-pro)
- [OpenRouter GPT-5 Image](https://openrouter.ai/openai/gpt-5-image)
- [Recraft V4 API Getting Started](https://www.recraft.ai/docs/api-reference/getting-started)
- [Recraft API Pricing](https://www.recraft.ai/docs/api-reference/pricing)
- [Recraft V4 on Replicate](https://replicate.com/blog/recraft-v4)
- [Ideogram API Documentation](https://developer.ideogram.ai/ideogram-api/api-overview)
- [Ideogram API Pricing](https://ideogram.ai/features/api-pricing)
- [Black Forest Labs FLUX Pricing](https://bfl.ai/pricing)
- [Stability AI API Pricing](https://platform.stability.ai/pricing)
- [OpenAI Image Generation Docs](https://developers.openai.com/api/docs/guides/image-generation)
- [AI Image Generation API Comparison 2026](https://blog.laozhang.ai/en/posts/ai-image-generation-api-comparison-2026)
- [Best AI Image Models 2026 - TeamDay](https://www.teamday.ai/blog/best-ai-image-models-2026)
- [fal.ai Best AI Image Generators 2026](https://fal.ai/learn/tools/ai-image-generators)
- [Recraft V4 - WaveSpeedAI](https://wavespeed.ai/blog/posts/recraft-v4-small-company-tops-ai-image-generation-2026/)
- [Complete Guide to AI Image Generation APIs 2026 - WaveSpeedAI](https://wavespeed.ai/blog/posts/complete-guide-ai-image-apis-2026/)
