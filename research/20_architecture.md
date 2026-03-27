# Logo Generation Pipeline: Architecture Document

## March 2026

---

## Table of Contents

1. [End-to-End Flow](#1-end-to-end-flow)
2. [Module Breakdown](#2-module-breakdown)
3. [Data Flow Between Components](#3-data-flow-between-components)
4. [Configuration and Settings Management](#4-configuration-and-settings-management)
5. [Storage Strategy](#5-storage-strategy)
6. [API Design](#6-api-design)
7. [CLI Design](#7-cli-design)
8. [Web UI Design](#8-web-ui-design)
9. [Pipeline Orchestration](#9-pipeline-orchestration)
10. [Error Handling and Fallback Strategies](#10-error-handling-and-fallback-strategies)
11. [Tech Stack](#11-tech-stack)
12. [MVP vs Full Feature Set](#12-mvp-vs-full-feature-set)

---

## 1. End-to-End Flow

```
User Input -> Prompt Enhancement -> Generation -> Post-Processing -> Delivery
```

### Detailed Pipeline Stages

```
[1] INPUT
    User provides:
      - Brand name
      - Brand description / industry / personality
      - Style preferences (minimalist, modern, vintage, playful, etc.)
      - Color preferences (optional hex codes or palette name)
      - Logo type (wordmark, symbol, combination mark, emblem, abstract mark, mascot)
      - Reference image (optional, for style transfer)
      - Text to include (optional)
         |
         v
[2] PROMPT ENHANCEMENT (LLM)
    Stage 2a: Brand Analysis
      - LLM extracts brand attributes, audience, personality
    Stage 2b: Design Direction
      - LLM proposes 3-5 design directions with rationale
    Stage 2c: Prompt Generation
      - LLM produces model-specific prompts (Flux, SDXL, Recraft, Ideogram, etc.)
      - Includes positive prompt, negative prompt, and suggested parameters
    Stage 2d: Negative Prompt Generation
      - LLM generates exclusion terms based on the selected direction
         |
         v
[3] GENERATION (Image Model)
    - Dispatch to selected backend(s):
        a. API backends: Recraft V4, Ideogram 3.0, Flux 2 Pro, GPT Image 1.5
        b. Local backends: Flux 2 Dev (diffusers), SDXL + LoRA (diffusers), ComfyUI
    - Batch generation: N variations per prompt (seed variation)
    - Optional multi-model: same prompt across 2-3 models for comparison
    - ControlNet / IP-Adapter conditioning if reference image provided
         |
         v
[4] POST-PROCESSING
    Stage 4a: Background Removal (rembg)
    Stage 4b: Upscaling (Real-ESRGAN, optional)
    Stage 4c: Color Palette Extraction (colorthief)
    Stage 4d: Vector Conversion (VTracer for color, Potrace for B&W)
    Stage 4e: SVG Optimization (path cleanup, size reduction)
    Stage 4f: Typography Overlay (Pillow/Cairo, if text was requested and model output is imperfect)
         |
         v
[5] QUALITY EVALUATION (Optional)
    - CLIP score for prompt adherence
    - Aesthetic scoring model
    - LLM Vision evaluation (Claude/GPT-4V ranks candidates)
    - Duplicates / near-duplicates removed
         |
         v
[6] DELIVERY
    - Output bundle per logo: PNG (multiple sizes), SVG, metadata JSON
    - Gallery view for selection (UI) or filesystem output (CLI)
    - Selected logo packaged as brand asset kit
```

---

## 2. Module Breakdown

### Core Package Structure

```
logo_gen/
|-- __init__.py
|-- __main__.py              # CLI entry point
|-- config.py                # Configuration loading and validation
|-- pipeline.py              # Pipeline orchestrator
|
|-- input/
|   |-- __init__.py
|   |-- schema.py            # Pydantic models for user input
|   |-- validators.py        # Input validation and normalization
|
|-- prompt/
|   |-- __init__.py
|   |-- enhancer.py          # LLM-based prompt enhancement
|   |-- templates.py         # System prompts and prompt templates
|   |-- brand_analyzer.py    # Brand attribute extraction
|
|-- generation/
|   |-- __init__.py
|   |-- base.py              # Abstract generator interface
|   |-- recraft.py           # Recraft V4 API backend
|   |-- ideogram.py          # Ideogram 3.0 API backend
|   |-- flux_api.py          # Flux via fal.ai / Together AI
|   |-- openai_gen.py        # GPT Image 1.5 backend
|   |-- flux_local.py        # Local Flux 2 Dev via diffusers
|   |-- sdxl_local.py        # Local SDXL + LoRA via diffusers
|   |-- comfyui.py           # ComfyUI workflow dispatch
|   |-- registry.py          # Generator registry and factory
|
|-- postprocess/
|   |-- __init__.py
|   |-- background.py        # Background removal (rembg)
|   |-- upscale.py           # Real-ESRGAN upscaling
|   |-- vectorize.py         # Raster-to-SVG (VTracer, Potrace)
|   |-- svg_optimize.py      # SVG path cleanup and optimization
|   |-- typography.py        # Text overlay with real fonts
|   |-- color.py             # Color palette extraction and analysis
|
|-- evaluate/
|   |-- __init__.py
|   |-- clip_score.py        # CLIP-based prompt adherence scoring
|   |-- aesthetic.py          # Aesthetic quality scoring
|   |-- llm_judge.py         # LLM vision-based ranking
|
|-- storage/
|   |-- __init__.py
|   |-- local.py             # Local filesystem storage
|   |-- models.py            # Metadata Pydantic models
|   |-- packaging.py         # Brand asset kit packaging
|
|-- api/
|   |-- __init__.py
|   |-- app.py               # FastAPI application
|   |-- routes.py            # API route definitions
|   |-- deps.py              # Dependency injection
|
|-- ui/
|   |-- __init__.py
|   |-- gradio_app.py        # Gradio web interface
|
|-- utils/
|   |-- __init__.py
|   |-- image.py             # Image format conversion helpers
|   |-- async_helpers.py     # Async utilities
|   |-- logging.py           # Structured logging setup
```

### Component Responsibilities

| Component | Responsibility | Key Dependencies |
|-----------|---------------|------------------|
| `input/` | Parse, validate, and normalize user input into a structured `LogoRequest` | pydantic |
| `prompt/` | Transform user intent into optimized image-model prompts via LLM | anthropic, openai, httpx |
| `generation/` | Dispatch to image models, manage batch runs, collect raw outputs | httpx, diffusers, torch |
| `postprocess/` | Refine raw outputs: bg removal, upscale, vectorize, typography | rembg, vtracer, Pillow, realesrgan |
| `evaluate/` | Score and rank generated candidates | transformers (CLIP), anthropic |
| `storage/` | Persist images, SVGs, metadata; package deliverables | pathlib, json |
| `pipeline.py` | Orchestrate stages, manage concurrency, handle errors | asyncio |
| `config.py` | Load and validate settings from YAML/env/CLI flags | pydantic-settings, pyyaml |
| `api/` | HTTP interface for the pipeline | fastapi, uvicorn |
| `ui/` | Interactive browser-based interface | gradio |

---

## 3. Data Flow Between Components

### Core Data Models

```python
# input/schema.py
class LogoRequest(BaseModel):
    brand_name: str
    description: str
    industry: str | None = None
    style: LogoStyle = LogoStyle.MINIMALIST  # enum
    logo_type: LogoType = LogoType.COMBINATION  # enum
    colors: list[str] | None = None  # hex codes
    text_to_include: str | None = None
    reference_image: bytes | None = None
    num_variations: int = 4
    models: list[str] = ["recraft_v4"]
    output_formats: list[str] = ["png", "svg"]

# prompt/enhancer.py
class EnhancedPrompt(BaseModel):
    positive_prompt: str
    negative_prompt: str
    design_rationale: str
    suggested_params: dict  # model-specific params (steps, cfg, etc.)
    target_model: str

class PromptSet(BaseModel):
    request: LogoRequest
    prompts: list[EnhancedPrompt]  # one per design direction

# generation/base.py
class GeneratedImage(BaseModel):
    image_data: bytes
    format: str  # "png" or "webp"
    width: int
    height: int
    model: str
    prompt_used: str
    seed: int | None = None
    generation_params: dict
    generation_time_ms: int

class GenerationBatch(BaseModel):
    request: LogoRequest
    prompt: EnhancedPrompt
    images: list[GeneratedImage]

# postprocess/
class ProcessedLogo(BaseModel):
    original: GeneratedImage
    png_transparent: bytes       # background removed
    png_sizes: dict[str, bytes]  # {"1024": bytes, "512": bytes, "256": bytes, "128": bytes}
    svg: str | None              # vectorized SVG content
    color_palette: list[str]     # extracted hex codes
    metadata: LogoMetadata

# evaluate/
class ScoredLogo(BaseModel):
    logo: ProcessedLogo
    clip_score: float | None = None
    aesthetic_score: float | None = None
    llm_rank: int | None = None
    llm_feedback: str | None = None

# storage/models.py
class LogoMetadata(BaseModel):
    id: str                      # uuid
    created_at: datetime
    request: LogoRequest
    prompt: EnhancedPrompt
    generation: GeneratedImage   # stripped of image_data
    scores: dict[str, float]
    color_palette: list[str]
    file_paths: dict[str, str]   # {"png_1024": "path", "svg": "path", ...}
```

### Data Flow Diagram

```
LogoRequest
    |
    v
[PromptEnhancer] --LLM call--> PromptSet (N EnhancedPrompts)
    |
    v
[Generator.generate()] --API/local call per prompt--> list[GenerationBatch]
    |                                                   (each batch has M images)
    v
[PostProcessor.process()] --per image--> list[ProcessedLogo]
    |
    v
[Evaluator.score()] --per logo--> list[ScoredLogo]
    |
    v
[Storage.save()] --filesystem write--> LogoMetadata + files on disk
    |
    v
[Delivery] --sorted by score--> final output (CLI files / API JSON / UI gallery)
```

---

## 4. Configuration and Settings Management

### Configuration Hierarchy (lowest to highest priority)

1. Built-in defaults (hardcoded)
2. Config file: `config.yaml` (project root or `~/.logo-gen/config.yaml`)
3. Environment variables (prefixed `LOGOGEN_`)
4. CLI flags / API request parameters

### Config File Structure

```yaml
# config.yaml

# --- LLM Settings (for prompt enhancement) ---
llm:
  provider: "anthropic"          # anthropic | openai | ollama
  model: "claude-sonnet-4-20250514"
  api_key_env: "ANTHROPIC_API_KEY"  # env var name holding the key
  temperature: 0.7
  max_tokens: 2048

# --- Generation Backends ---
generation:
  default_backend: "recraft_v4"
  backends:
    recraft_v4:
      api_key_env: "RECRAFT_API_KEY"
      base_url: "https://external.api.recraft.ai"
      default_style: "digital_illustration"
      default_size: "1024x1024"

    ideogram:
      api_key_env: "IDEOGRAM_API_KEY"
      base_url: "https://api.ideogram.ai"
      default_model: "V_3"

    flux_api:
      provider: "fal"             # fal | together | replicate
      api_key_env: "FAL_KEY"
      model: "fal-ai/flux-pro/v1.1"

    openai:
      api_key_env: "OPENAI_API_KEY"
      model: "gpt-image-1.5"
      quality: "high"

    flux_local:
      model_id: "black-forest-labs/FLUX.1-dev"
      device: "cuda"
      dtype: "bfloat16"
      offload: true               # CPU offload for low VRAM

    sdxl_local:
      model_id: "stabilityai/stable-diffusion-xl-base-1.0"
      lora: "Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design"
      lora_scale: 0.8
      device: "cuda"

  batch:
    default_variations: 4
    max_variations: 16
    seeds: [42, 123, 456, 789, 1024, 2048, 4096, 8192]

# --- Post-Processing ---
postprocess:
  background_removal: true
  upscale: false
  upscale_factor: 2
  vectorize: true
  vectorize_mode: "spline"        # pixel | polygon | spline
  vectorize_colormode: "color"    # color | bw
  svg_optimize: true
  output_sizes: [1024, 512, 256, 128]

# --- Evaluation ---
evaluate:
  enabled: false                  # disabled for MVP
  clip_score: false
  aesthetic_score: false
  llm_judge: false

# --- Storage ---
storage:
  base_dir: "./output"
  structure: "by_project"         # by_project | by_date | flat
  metadata_format: "json"         # json | sqlite

# --- API Server ---
api:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  rate_limit: 10                  # requests per minute
  max_upload_size_mb: 10

# --- UI ---
ui:
  provider: "gradio"              # gradio | streamlit
  share: false                    # gradio public link
  theme: "soft"
```

### Settings Implementation

```python
# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import yaml

class LLMConfig(BaseSettings):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key_env: str = "ANTHROPIC_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 2048

class GenerationConfig(BaseSettings):
    default_backend: str = "recraft_v4"
    # ... backend sub-configs

class PostProcessConfig(BaseSettings):
    background_removal: bool = True
    upscale: bool = False
    vectorize: bool = True
    # ...

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOGOGEN_",
        env_nested_delimiter="__",
    )
    llm: LLMConfig = LLMConfig()
    generation: GenerationConfig = GenerationConfig()
    postprocess: PostProcessConfig = PostProcessConfig()
    # ...

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

### API Key Management

API keys are never stored in config files. They are resolved at runtime:

1. Config specifies the env var name (e.g., `api_key_env: "RECRAFT_API_KEY"`)
2. At runtime, `os.environ[config.api_key_env]` is read
3. For local dev, use a `.env` file loaded by `python-dotenv`
4. `.env` is in `.gitignore`

---

## 5. Storage Strategy

### Directory Structure

```
output/
|-- projects/
|   |-- {project_id}/                    # UUID or slugified brand name
|   |   |-- metadata.json               # Project-level metadata
|   |   |-- prompts/
|   |   |   |-- enhanced_prompts.json   # All generated prompts
|   |   |-- raw/
|   |   |   |-- {gen_id}_seed42.png     # Raw generation outputs
|   |   |   |-- {gen_id}_seed123.png
|   |   |-- processed/
|   |   |   |-- {logo_id}/
|   |   |   |   |-- logo_1024.png       # Transparent, full size
|   |   |   |   |-- logo_512.png
|   |   |   |   |-- logo_256.png
|   |   |   |   |-- logo_128.png
|   |   |   |   |-- logo.svg            # Vectorized
|   |   |   |   |-- metadata.json       # Per-logo metadata
|   |   |-- selected/                    # User's final picks
|   |   |   |-- brand_kit/
|   |   |   |   |-- {brand}_logo.svg
|   |   |   |   |-- {brand}_logo_1024.png
|   |   |   |   |-- {brand}_logo_512.png
|   |   |   |   |-- {brand}_logo_256.png
|   |   |   |   |-- {brand}_logo_128.png
|   |   |   |   |-- palette.json
|   |   |   |   |-- brand_kit.zip
```

### Metadata Schema (per-logo metadata.json)

```json
{
  "id": "a1b2c3d4-...",
  "project_id": "nova-tech-2026",
  "created_at": "2026-03-27T14:30:00Z",
  "request": {
    "brand_name": "Nova",
    "description": "Cloud computing startup",
    "style": "minimalist",
    "logo_type": "combination",
    "colors": ["#2563EB", "#1E40AF"]
  },
  "prompt": {
    "positive": "Minimalist vector logo, abstract rising star ...",
    "negative": "no shadows, no gradients, no realistic ...",
    "target_model": "recraft_v4"
  },
  "generation": {
    "model": "recraft_v4",
    "seed": 42,
    "params": {"style": "digital_illustration", "size": "1024x1024"},
    "time_ms": 3200
  },
  "postprocess": {
    "bg_removed": true,
    "vectorized": true,
    "upscaled": false
  },
  "scores": {
    "clip_score": 0.312,
    "aesthetic_score": 6.8
  },
  "color_palette": ["#2563EB", "#1E40AF", "#FFFFFF", "#0F172A"],
  "files": {
    "raw": "raw/gen_a1b2_seed42.png",
    "png_1024": "processed/a1b2c3d4/logo_1024.png",
    "png_512": "processed/a1b2c3d4/logo_512.png",
    "svg": "processed/a1b2c3d4/logo.svg"
  }
}
```

### Storage Backend Considerations

For the MVP, local filesystem with JSON metadata is sufficient. If the project scales:

- **SQLite**: single-file database for metadata queries (filter by style, date, model)
- **S3-compatible storage**: for cloud deployment (MinIO for self-hosted, AWS S3 for production)
- **PostgreSQL + pgvector**: if CLIP embeddings are stored for similarity search

---

## 6. API Design

### REST API (FastAPI)

#### Endpoints

```
POST   /api/v1/generate           # Full pipeline: input -> logos
POST   /api/v1/enhance-prompt     # Prompt enhancement only
POST   /api/v1/generate-images    # Image generation only (from pre-built prompt)
POST   /api/v1/postprocess        # Post-process an uploaded image
GET    /api/v1/projects           # List all projects
GET    /api/v1/projects/{id}      # Get project details and logos
GET    /api/v1/logos/{id}         # Get a specific logo and its metadata
GET    /api/v1/logos/{id}/file/{variant}  # Download a specific file (png_1024, svg, etc.)
DELETE /api/v1/projects/{id}      # Delete a project and all its outputs
GET    /api/v1/health             # Health check
GET    /api/v1/models             # List available generation backends
```

#### Request / Response Examples

**POST /api/v1/generate**

Request:
```json
{
  "brand_name": "Nova",
  "description": "Cloud computing startup focused on developer tools",
  "industry": "technology",
  "style": "minimalist",
  "logo_type": "combination",
  "colors": ["#2563EB", "#1E40AF"],
  "text_to_include": "NOVA",
  "num_variations": 4,
  "models": ["recraft_v4"],
  "output_formats": ["png", "svg"]
}
```

Response (async job):
```json
{
  "job_id": "job_abc123",
  "status": "processing",
  "project_id": "nova-tech-20260327-143000",
  "estimated_time_seconds": 45,
  "poll_url": "/api/v1/jobs/job_abc123"
}
```

**GET /api/v1/jobs/{job_id}** (poll for completion)

Response (completed):
```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "project_id": "nova-tech-20260327-143000",
  "logos": [
    {
      "id": "logo_001",
      "scores": {"clip_score": 0.32, "aesthetic_score": 7.1},
      "color_palette": ["#2563EB", "#1E40AF", "#FFFFFF"],
      "files": {
        "png_1024": "/api/v1/logos/logo_001/file/png_1024",
        "svg": "/api/v1/logos/logo_001/file/svg"
      },
      "prompt_used": "Minimalist vector logo, abstract rising star ..."
    }
  ]
}
```

#### Async Pattern

Generation takes 5-60 seconds depending on the backend and number of variations. The API uses an async job pattern:

1. Client POSTs to `/generate`, receives a `job_id` immediately
2. Client polls `/jobs/{job_id}` or connects via WebSocket for real-time updates
3. Optional: Server-Sent Events (SSE) for streaming progress

```python
# api/routes.py
@router.post("/generate")
async def generate(request: LogoRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    background_tasks.add_task(run_pipeline, job_id, request)
    return {"job_id": job_id, "status": "processing"}

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    return job_store.get(job_id)

@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str):
    async def event_generator():
        async for update in job_store.subscribe(job_id):
            yield f"data: {update.model_dump_json()}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

---

## 7. CLI Design

### Command Structure

```
logo-gen [COMMAND] [OPTIONS]

Commands:
  generate     Run the full logo generation pipeline
  enhance      Generate enhanced prompts without image generation
  postprocess  Post-process existing images (bg removal, vectorize, etc.)
  models       List available generation backends
  config       Show or edit configuration
  serve        Start the API server
  ui           Launch the web UI
```

### Primary Command: `generate`

```
logo-gen generate [OPTIONS]

Required:
  --name, -n TEXT           Brand name

Options:
  --description, -d TEXT    Brand description
  --industry TEXT           Industry / sector
  --style TEXT              Logo style: minimalist, modern, vintage, playful, etc.
                            [default: minimalist]
  --type TEXT               Logo type: wordmark, symbol, combination, emblem,
                            abstract, mascot [default: combination]
  --colors TEXT             Comma-separated hex colors (e.g., "#2563EB,#1E40AF")
  --text TEXT               Text to include in the logo
  --reference PATH          Reference image for style transfer
  --model, -m TEXT          Generation backend(s), comma-separated
                            [default: recraft_v4]
  --variations, -v INT      Number of variations [default: 4]
  --output, -o PATH         Output directory [default: ./output]
  --format TEXT             Output formats: png,svg [default: png,svg]
  --no-bg-remove            Skip background removal
  --no-vectorize            Skip SVG vectorization
  --upscale                 Enable upscaling
  --config PATH             Path to config.yaml
  --verbose                 Enable verbose logging
  --json                    Output results as JSON to stdout
```

### Usage Examples

```bash
# Minimal usage
logo-gen generate --name "Nova" --description "Cloud computing startup"

# Full control
logo-gen generate \
  --name "Nova" \
  --description "Cloud computing startup for developers" \
  --style minimalist \
  --type combination \
  --colors "#2563EB,#1E40AF" \
  --text "NOVA" \
  --model recraft_v4,ideogram \
  --variations 8 \
  --output ./nova-logos \
  --upscale

# Prompt enhancement only
logo-gen enhance \
  --name "Nova" \
  --description "Cloud computing startup" \
  --model recraft_v4 \
  --directions 5

# Post-process existing images
logo-gen postprocess \
  --input ./raw-logos/ \
  --vectorize \
  --bg-remove \
  --output ./processed/

# Launch services
logo-gen serve --port 8000
logo-gen ui --share
```

### CLI Implementation

Use `click` for argument parsing (or `typer` for Pydantic integration):

```python
# __main__.py
import click

@click.group()
@click.option("--config", type=click.Path(), default=None)
@click.option("--verbose", is_flag=True)
@click.pass_context
def cli(ctx, config, verbose):
    ctx.ensure_object(dict)
    ctx.obj["settings"] = Settings.from_yaml(config) if config else Settings()
    ctx.obj["verbose"] = verbose

@cli.command()
@click.option("--name", "-n", required=True)
@click.option("--description", "-d", default="")
# ... all options ...
@click.pass_context
def generate(ctx, name, description, ...):
    request = LogoRequest(brand_name=name, description=description, ...)
    pipeline = LogoPipeline(ctx.obj["settings"])
    results = asyncio.run(pipeline.run(request))
    # display results

if __name__ == "__main__":
    cli()
```

### CLI Output

```
$ logo-gen generate --name "Nova" --description "Cloud computing startup"

[1/4] Analyzing brand and generating prompts...
  - Generated 3 design directions
  - Direction 1: Abstract rising star, minimalist geometric
  - Direction 2: Cloud + compass combination mark
  - Direction 3: Wordmark with stylized 'N'

[2/4] Generating logo variations...
  - Backend: recraft_v4
  - Generating 4 variations per direction (12 total)...
  [################] 12/12 complete (34.2s)

[3/4] Post-processing...
  - Background removal: 12 images
  - Vectorization: 12 SVGs generated
  - Resizing: 4 sizes per image
  [################] 12/12 complete (8.1s)

[4/4] Saving results...
  - Output directory: ./output/nova-20260327-143000/
  - 12 logos generated (PNG + SVG)
  - Metadata saved to metadata.json

Done! Open ./output/nova-20260327-143000/ to view results.
```

---

## 8. Web UI Design

### Gradio Interface (Recommended for MVP)

Gradio is chosen over Streamlit for the MVP because:
- Native image gallery components
- Built-in API generation (every Gradio app is automatically an API)
- Simpler state management for a single-pipeline flow
- Easy public sharing via `share=True`

### Layout

```
+----------------------------------------------------------------------+
|  LOGO GENERATOR                                          [Settings]  |
+----------------------------------------------------------------------+
|                                                                      |
|  LEFT PANEL (Inputs)              |  RIGHT PANEL (Results)           |
|  +--------------------------+     |  +--------------------------+    |
|  | Brand Name:  [________]  |     |  |                          |    |
|  | Description: [________]  |     |  |  [Gallery Grid]          |    |
|  |              [________]  |     |  |                          |    |
|  | Industry:    [________]  |     |  |  [img1] [img2] [img3]   |    |
|  |                          |     |  |  [img4] [img5] [img6]   |    |
|  | Style:  [v Minimalist ]  |     |  |                          |    |
|  | Type:   [v Combination]  |     |  +--------------------------+    |
|  |                          |     |                                  |
|  | Colors: [#hex] [+Add]    |     |  Selected Logo Detail:           |
|  |  [swatch] [swatch]       |     |  +--------------------------+    |
|  |                          |     |  | [Large Preview]          |    |
|  | Text:    [________]      |     |  |                          |    |
|  |                          |     |  | Prompt: "..."            |    |
|  | Reference: [Upload]      |     |  | Colors: [#] [#] [#]     |    |
|  |                          |     |  | Score: 7.2/10            |    |
|  | Model:  [v recraft_v4 ]  |     |  |                          |    |
|  | Variations: [--4--]      |     |  | [Download PNG]           |    |
|  |                          |     |  | [Download SVG]           |    |
|  | [   Generate Logos   ]   |     |  | [Download Brand Kit]     |    |
|  +--------------------------+     |  +--------------------------+    |
|                                                                      |
|  Progress: [##########---------] 50% - Generating variations...      |
+----------------------------------------------------------------------+
```

### Gradio Implementation Sketch

```python
# ui/gradio_app.py
import gradio as gr

def create_app(settings: Settings) -> gr.Blocks:
    pipeline = LogoPipeline(settings)

    with gr.Blocks(title="Logo Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Logo Generator")

        with gr.Row():
            # Left panel - inputs
            with gr.Column(scale=1):
                brand_name = gr.Textbox(label="Brand Name", placeholder="e.g., Nova")
                description = gr.Textbox(label="Description", lines=3)
                industry = gr.Textbox(label="Industry")
                style = gr.Dropdown(
                    choices=["minimalist", "modern", "vintage", "playful", "geometric"],
                    value="minimalist", label="Style"
                )
                logo_type = gr.Dropdown(
                    choices=["combination", "wordmark", "symbol", "emblem", "abstract", "mascot"],
                    value="combination", label="Logo Type"
                )
                colors = gr.Textbox(label="Colors (hex, comma-separated)", placeholder="#2563EB, #1E40AF")
                text_input = gr.Textbox(label="Text to Include")
                reference = gr.Image(label="Reference Image (optional)", type="filepath")
                model = gr.Dropdown(
                    choices=list(settings.generation.backends.keys()),
                    value=settings.generation.default_backend,
                    label="Model"
                )
                variations = gr.Slider(1, 16, value=4, step=1, label="Variations")
                generate_btn = gr.Button("Generate Logos", variant="primary")

            # Right panel - results
            with gr.Column(scale=2):
                progress = gr.Markdown("Ready")
                gallery = gr.Gallery(label="Generated Logos", columns=3, height=400)
                with gr.Row():
                    selected_preview = gr.Image(label="Selected Logo")
                    with gr.Column():
                        prompt_display = gr.Textbox(label="Prompt Used", interactive=False)
                        palette_display = gr.JSON(label="Color Palette")
                        download_png = gr.File(label="Download PNG")
                        download_svg = gr.File(label="Download SVG")
                        download_kit = gr.File(label="Download Brand Kit (.zip)")

        # Event handlers
        generate_btn.click(
            fn=run_pipeline_for_ui,
            inputs=[brand_name, description, industry, style, logo_type,
                    colors, text_input, reference, model, variations],
            outputs=[gallery, progress]
        )
        gallery.select(
            fn=on_logo_selected,
            outputs=[selected_preview, prompt_display, palette_display,
                     download_png, download_svg, download_kit]
        )

    return app
```

### Advanced UI Features (Post-MVP)

- **Live prompt preview**: show the enhanced prompt before generation starts
- **Side-by-side model comparison**: run same prompt on 2-3 models
- **Refinement mode**: select a logo and run img2img variations
- **Color picker**: interactive palette builder with harmony suggestions
- **Typography editor**: choose font, size, position for text overlay
- **History tab**: browse previous generation sessions
- **Favorites**: star logos across sessions

---

## 9. Pipeline Orchestration

### Sequential vs Parallel Steps

```
SEQUENTIAL (each stage depends on the previous):
  Input Validation -> Prompt Enhancement -> [parallel zone] -> Delivery

PARALLEL within Generation stage:
  - If multiple models selected: run all models concurrently
  - If batch variations: run seed variations concurrently (up to concurrency limit)

PARALLEL within Post-Processing:
  - Each image is independent: process all images concurrently
  - Within a single image: bg_remove -> (upscale, vectorize) can run in parallel

PARALLEL within Evaluation (if enabled):
  - CLIP scoring and aesthetic scoring can run concurrently per image
  - LLM judge runs once over the full batch (sequential)
```

### Pipeline Orchestrator

```python
# pipeline.py
import asyncio
from contextlib import asynccontextmanager

class LogoPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enhancer = PromptEnhancer(settings.llm)
        self.generators = GeneratorRegistry(settings.generation)
        self.postprocessor = PostProcessor(settings.postprocess)
        self.evaluator = Evaluator(settings.evaluate) if settings.evaluate.enabled else None
        self.storage = LocalStorage(settings.storage)

    async def run(
        self,
        request: LogoRequest,
        progress_callback: Callable | None = None,
    ) -> list[ScoredLogo]:

        # Stage 1: Prompt Enhancement (sequential, single LLM call)
        self._report(progress_callback, "Enhancing prompts...")
        prompt_set = await self.enhancer.enhance(request)

        # Stage 2: Generation (parallel across models and seeds)
        self._report(progress_callback, "Generating images...")
        generation_tasks = []
        for prompt in prompt_set.prompts:
            for model_name in request.models:
                generator = self.generators.get(model_name)
                generation_tasks.append(
                    generator.generate_batch(prompt, request.num_variations)
                )

        batches: list[GenerationBatch] = await asyncio.gather(
            *generation_tasks, return_exceptions=True
        )
        # Filter out failed batches, log errors
        batches = [b for b in batches if not isinstance(b, Exception)]

        all_images = [img for batch in batches for img in batch.images]

        # Stage 3: Post-Processing (parallel across images)
        self._report(progress_callback, "Post-processing...")
        semaphore = asyncio.Semaphore(4)  # limit concurrent heavy ops

        async def process_one(img):
            async with semaphore:
                return await self.postprocessor.process(img)

        processed = await asyncio.gather(
            *[process_one(img) for img in all_images],
            return_exceptions=True,
        )
        processed = [p for p in processed if not isinstance(p, Exception)]

        # Stage 4: Evaluation (optional)
        if self.evaluator:
            self._report(progress_callback, "Evaluating quality...")
            scored = await self.evaluator.score_batch(processed)
        else:
            scored = [ScoredLogo(logo=p) for p in processed]

        # Stage 5: Storage
        self._report(progress_callback, "Saving results...")
        for logo in scored:
            await self.storage.save(request, logo)

        # Sort by score (best first)
        scored.sort(key=lambda s: s.aesthetic_score or 0, reverse=True)

        self._report(progress_callback, "Done!")
        return scored
```

### Concurrency Limits

| Operation | Default Concurrency | Rationale |
|-----------|-------------------|-----------|
| API generation calls | 3 | Avoid rate limits |
| Local GPU generation | 1 | GPU is a single resource |
| Background removal | 4 | CPU-bound, moderate memory |
| Vectorization | 4 | CPU-bound |
| Upscaling | 1 | GPU-heavy |
| LLM calls | 2 | API rate limits |

### Progress Reporting

The pipeline accepts a `progress_callback` function that receives stage name, percentage, and message. This powers:
- CLI progress bars (via `rich` or `click`)
- Gradio progress updates
- API SSE/WebSocket events

---

## 10. Error Handling and Fallback Strategies

### Error Categories and Responses

| Error Category | Example | Strategy |
|----------------|---------|----------|
| **API rate limit** | 429 from Recraft | Exponential backoff (3 retries, 2s/4s/8s) |
| **API auth failure** | 401/403 | Fail fast with clear message about missing/invalid key |
| **API timeout** | Generation takes >120s | Retry once; if still fails, skip model and log |
| **Model unavailable** | API 503 or local model not found | Fall back to next model in priority list |
| **Invalid input** | Missing brand name | Reject at validation layer with specific error |
| **LLM failure** | Prompt enhancement fails | Use template-based fallback prompts (no LLM) |
| **Post-process failure** | VTracer crashes on an image | Skip vectorization for that image, deliver PNG only |
| **Out of memory** | GPU OOM during local generation | Reduce batch size to 1, enable CPU offload, retry |
| **Partial batch failure** | 2 of 8 seeds fail | Deliver the 6 that succeeded, log failures |
| **Storage failure** | Disk full | Fail with clear message, do not lose in-memory results |

### Fallback Chain for Generation

```
Primary:   User-selected model (e.g., recraft_v4)
    |-- fails -->
Fallback 1: flux_api (fal.ai)
    |-- fails -->
Fallback 2: openai (GPT Image 1.5)
    |-- fails -->
Fallback 3: flux_local (if GPU available)
    |-- fails -->
Error: "All generation backends failed. Check API keys and network."
```

### Fallback for Prompt Enhancement

```
Primary:   LLM-based enhancement (Claude/GPT)
    |-- fails -->
Fallback:  Template-based prompt construction

# Template fallback example:
def template_fallback(request: LogoRequest) -> EnhancedPrompt:
    return EnhancedPrompt(
        positive_prompt=(
            f"{request.style} vector logo, {request.logo_type}, "
            f"for {request.brand_name}, {request.description}, "
            f"clean lines, professional, scalable, white background"
        ),
        negative_prompt="blurry, low quality, realistic photo, complex background",
        design_rationale="Template-based fallback (LLM unavailable)",
        suggested_params={},
        target_model=request.models[0],
    )
```

### Retry Implementation

```python
# utils/retry.py
import asyncio
from functools import wraps

def with_retry(max_retries=3, backoff_base=2.0, retriable_exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retriable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait = backoff_base ** attempt
                        logger.warning(f"{func.__name__} attempt {attempt+1} failed: {e}. "
                                       f"Retrying in {wait}s...")
                        await asyncio.sleep(wait)
            raise last_exception
        return wrapper
    return decorator
```

### Graceful Degradation Principle

The pipeline always attempts to deliver something rather than failing entirely:
- If vectorization fails, deliver PNGs without SVG
- If background removal fails, deliver original images
- If 3 of 4 models fail, deliver results from the 1 that worked
- If evaluation fails, deliver unsorted results
- If upscaling fails, deliver at original resolution

---

## 11. Tech Stack

### Core Dependencies

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Runtime** | Python | 3.11+ | Language runtime |
| **Data Validation** | pydantic | 2.x | Request/response models, settings |
| **Config** | pydantic-settings | 2.x | Settings from env/yaml |
| **Config** | pyyaml | 6.x | YAML config file parsing |
| **Config** | python-dotenv | 1.x | `.env` file loading |
| **HTTP Client** | httpx | 0.27+ | Async API calls to generation backends |
| **CLI** | click | 8.x | Command-line interface |
| **CLI Output** | rich | 13.x | Progress bars, tables, colored output |
| **Logging** | structlog | 24.x | Structured logging |

### Image Generation (API)

| Package | Purpose |
|---------|---------|
| httpx | Direct API calls to Recraft, Ideogram |
| fal-client | fal.ai API (Flux, SDXL, Recraft) |
| together | Together AI API (Flux Pro, Ideogram) |
| openai | OpenAI API (GPT Image 1.5) |
| anthropic | Claude API (prompt enhancement, LLM judge) |

### Image Generation (Local, optional)

| Package | Purpose |
|---------|---------|
| torch | PyTorch backend |
| diffusers | HuggingFace diffusion pipelines |
| transformers | CLIP, model loading |
| accelerate | Multi-GPU, memory optimization |
| safetensors | Fast model weight loading |

### Post-Processing

| Package | Purpose |
|---------|---------|
| Pillow | Image manipulation, resizing, compositing |
| rembg | Background removal |
| vtracer | Raster to SVG vectorization |
| realesrgan | Image upscaling (optional) |
| colorthief | Color palette extraction |

### API Server

| Package | Purpose |
|---------|---------|
| fastapi | REST API framework |
| uvicorn | ASGI server |

### Web UI

| Package | Purpose |
|---------|---------|
| gradio | Interactive web interface |

### Development / Testing

| Package | Purpose |
|---------|---------|
| pytest | Testing framework |
| pytest-asyncio | Async test support |
| ruff | Linting and formatting |
| mypy | Static type checking |
| pre-commit | Git hooks |

### Dependency Groups (pyproject.toml)

```toml
[project]
name = "logo-gen"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "httpx>=0.27",
    "click>=8.0",
    "rich>=13.0",
    "structlog>=24.0",
    "Pillow>=10.0",
    "rembg>=2.0",
    "vtracer>=0.6",
    "colorthief>=0.2",
    "anthropic>=0.40",
]

[project.optional-dependencies]
api = [
    "fastapi>=0.115",
    "uvicorn>=0.32",
]
ui = [
    "gradio>=5.0",
]
local = [
    "torch>=2.4",
    "diffusers>=0.31",
    "transformers>=4.46",
    "accelerate>=1.0",
    "safetensors>=0.4",
]
eval = [
    "realesrgan>=0.3",
]
all-backends = [
    "openai>=1.50",
    "together>=1.3",
    "fal-client>=0.5",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "mypy>=1.13",
    "pre-commit>=4.0",
]

[project.scripts]
logo-gen = "logo_gen.__main__:cli"
```

---

## 12. MVP vs Full Feature Set

### MVP (v0.1) - Target: 1-2 weeks

**Goal**: End-to-end pipeline that works for the most common case.

| Feature | Scope |
|---------|-------|
| **Input** | Brand name + description via CLI |
| **Prompt Enhancement** | Single LLM call (Claude) with hardcoded system prompt |
| **Generation** | One API backend: Recraft V4 |
| **Variations** | 4 variations via seed control |
| **Post-Processing** | Background removal + VTracer vectorization |
| **Output** | PNG (1024px) + SVG per variation, saved to local filesystem |
| **CLI** | `logo-gen generate --name "X" --description "Y"` |
| **Config** | `.env` for API keys, minimal YAML config |
| **Error Handling** | Basic retry (3x) on API failure, template fallback for LLM |

**Not in MVP**: Web UI, API server, evaluation/scoring, upscaling, multiple models, reference images, brand kit packaging, typography overlay.

### v0.2 - Multi-Model + UI (2-3 weeks after MVP)

| Feature | Addition |
|---------|----------|
| **Generation** | Add Ideogram 3.0 and Flux 2 Pro API backends |
| **Multi-Model** | Run same prompt across 2-3 models, compare results |
| **Web UI** | Gradio interface with gallery view |
| **Output Sizes** | PNG at 1024, 512, 256, 128 |
| **Color Extraction** | Extract and display palette per logo |
| **Style Control** | Full style/type dropdowns in CLI and UI |
| **Config** | Full `config.yaml` support |

### v0.3 - Quality + Polish (2-3 weeks)

| Feature | Addition |
|---------|----------|
| **Evaluation** | CLIP score ranking, optional LLM judge |
| **API Server** | FastAPI with async job pattern |
| **Brand Kit** | Zip packaging with all assets + metadata |
| **Typography** | Post-generation text overlay with real fonts |
| **Reference Image** | IP-Adapter style transfer via API backends |
| **History** | Browse previous generations, metadata search |
| **Upscaling** | Optional Real-ESRGAN upscaling |

### v0.4 - Local + Advanced (ongoing)

| Feature | Addition |
|---------|----------|
| **Local Generation** | Flux 2 Dev and SDXL via diffusers (GPU required) |
| **LoRA Support** | Logo-specific LoRAs for local generation |
| **ControlNet** | Sketch-to-logo via Canny/Scribble ControlNet |
| **ComfyUI** | Workflow dispatch for advanced pipelines |
| **Fine-Tuning** | Custom model fine-tuning on brand assets |
| **Batch Mode** | CSV input for bulk logo generation |
| **Plugin System** | Custom post-processors and generators |

### MVP File Checklist

Files to create for v0.1:

```
logo_gen/
|-- __init__.py
|-- __main__.py          # CLI with click
|-- config.py            # Settings with pydantic-settings
|-- pipeline.py          # Orchestrator
|-- input/
|   |-- __init__.py
|   |-- schema.py        # LogoRequest model
|-- prompt/
|   |-- __init__.py
|   |-- enhancer.py      # Claude-based enhancement
|   |-- templates.py     # System prompt + fallback templates
|-- generation/
|   |-- __init__.py
|   |-- base.py          # Abstract Generator
|   |-- recraft.py       # Recraft V4 implementation
|-- postprocess/
|   |-- __init__.py
|   |-- background.py    # rembg wrapper
|   |-- vectorize.py     # VTracer wrapper
|-- storage/
|   |-- __init__.py
|   |-- local.py         # Filesystem save + metadata JSON
|   |-- models.py        # Metadata models

config.yaml              # Default config
.env.example             # Template for API keys
pyproject.toml           # Project metadata + dependencies
```

---

## Appendix: Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Config format | YAML + env vars | YAML is human-readable for complex config; env vars for secrets |
| CLI framework | click | Mature, well-documented, good for nested commands |
| HTTP client | httpx | Async-native, modern API, used widely in Python ecosystem |
| Validation | pydantic v2 | Industry standard, integrates with FastAPI and settings |
| API framework | FastAPI | Async, auto-docs, pydantic integration, widely adopted |
| UI framework | Gradio | Image gallery built-in, auto-API, easy sharing, fast prototyping |
| Vector conversion | VTracer | Color support, O(n) performance, smaller SVGs than alternatives |
| Background removal | rembg | Best open-source option, works well on logos |
| Default gen backend | Recraft V4 | Native SVG, #1 on leaderboard, good pricing |
| LLM for prompts | Claude (Anthropic) | Strong instruction following, good at creative prompt writing |
| Logging | structlog | Structured JSON logs, good for both dev and production |
| Project tooling | ruff + mypy | Fast linting/formatting, strong type checking |
