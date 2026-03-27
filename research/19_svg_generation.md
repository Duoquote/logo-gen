# Research: Direct SVG Generation for Logos

## 1. PyTorch-SVGRender Framework

PyTorch-SVGRender is the primary unified library for differentiable SVG rendering via neural networks. It consolidates state-of-the-art methods for text-to-SVG, image-to-SVG, and SVG editing.

### Supported Methods

**Image-to-SVG (Vectorization):**
| Method | Venue | Notes |
|--------|-------|-------|
| DiffVG | SIGGRAPH 2020 | Core differentiable rasterizer; foundation for most other methods |
| LIVE | CVPR 2022 | Layer-wise image vectorization preserving topology |
| CLIPasso | SIGGRAPH 2022 | Object-to-sketch with varying abstraction levels |
| CLIPascene | ICCV 2023 | Scene-level sketching with multiple abstraction types |

**Text-to-SVG Synthesis:**
| Method | Venue | Notes |
|--------|-------|-------|
| CLIPDraw | NeurIPS 2022 | Text-to-drawing via CLIP language-image encoders |
| StyleCLIPDraw | - | Couples content and style in text-to-drawing |
| CLIPFont | BMVC 2022 | Texture-guided vector WordArt generation |
| VectorFusion | CVPR 2023 | Text-to-SVG via pixel-based diffusion model abstraction; 3 primitive styles |
| DiffSketcher | NeurIPS 2023 | Text-guided sketch synthesis through latent diffusion |
| Word-As-Image | SIGGRAPH 2023 | Semantic typography |
| SVGDreamer | CVPR 2024 | Best quality; 6 primitive types; SIVE + VPSD pipeline |

### Setup Requirements
- Python 3.10+
- PyTorch with CUDA
- DiffVG (compiled differentiable rasterizer)
- HuggingFace diffusers library
- CLIP vision-language models
- xFormers for optimized transformer computation
- U2Net for object segmentation (some methods)

### Installation
```bash
git clone https://github.com/ximinng/PyTorch-SVGRender.git
cd PyTorch-SVGRender
# Run provided setup scripts from the repo's top-level directory
# Docker deployment also available
```

### Technical Approach
Most methods use **Score Distillation Sampling (SDS)**: a pretrained text-to-image diffusion model provides gradients that flow through the differentiable rasterizer (DiffVG) back to SVG path parameters. This optimizes Bezier control points, colors, and stroke widths without requiring paired training data.

**Key limitation for logos:** These methods are slow (minutes to hours per generation) and produce artistic/illustrative results. They are not optimized for clean, geometric logo output.

Source: https://github.com/ximinng/PyTorch-SVGRender

---

## 2. SVGDreamer: Vector Primitives and Logo Quality

SVGDreamer (CVPR 2024) is the most capable method in the PyTorch-SVGRender ecosystem. It introduces Semantic-driven Image Vectorization (SIVE) and Vectorized Particle-based Score Distillation (VPSD).

### Six Vector Primitive Types

| Primitive | Representation | Best For |
|-----------|---------------|----------|
| **Iconography** | Closed Bezier curves with trainable control points + fill colors | Logos, icons, flat design |
| **Sketch** | Open Bezier curves with control points + opacity | Line art, sketches |
| **Pixel Art** | Square SVG polygons with fill colors | Retro/pixel style |
| **Low-Poly** | Square polygons with trainable control points + fill colors | Geometric/faceted style |
| **Painting** | Open Bezier curves with stroke colors + widths | Artistic illustrations |
| **Ink and Wash** | Open curves with opacity + stroke widths | Traditional East Asian style |

### SIVE (Semantic-driven Image Vectorization)
1. **Primitive Initialization**: Uses cross-attention maps from the diffusion model to allocate control points to different semantic regions (foreground vs background)
2. **Semantic-aware Optimization**: Attention-based mask loss hierarchically optimizes vector objects, keeping elements within designated regions for editability

### VPSD (Vectorized Particle-based Score Distillation)
- Models SVGs as distributions of control points and colors
- Maintains k particle groups of vector parameters
- Uses LoRA network for distribution estimation
- Reward Feedback Learning accelerates convergence by ~50%

### Quality Metrics (vs VectorFusion)
| Metric | SVGDreamer | VectorFusion |
|--------|-----------|--------------|
| FID | **59.13** | 100.68 |
| PSNR | **14.54** | 8.01 |
| Aesthetic Score | **5.54** | 4.89 |

### SVGDreamer++ (Late 2024)
- Adaptive vector primitives control: dynamically adjusts primitive count
- Improved editability and visual quality
- Better diversity through enhanced VPSD

### Logo Suitability
The **Iconography** primitive is most relevant for logos -- it produces clean filled shapes with discrete color regions. However, generation is slow (10-30 minutes on GPU) and results still tend toward illustration rather than the clean geometric precision expected in professional logos.

Sources:
- https://arxiv.org/html/2312.16476v3
- https://ximinng.github.io/SVGDreamer-project/

---

## 3. LLM-Based SVG Code Generation

### Direct SVG Markup from LLMs

LLMs can generate SVG code directly as text output. This is fundamentally different from diffusion-based methods -- the model writes `<svg>`, `<path>`, `<circle>`, etc. tags as code.

### Benchmark Results (2025)

A benchmark by Tom Gally tested 9 frontier models on 30 creative SVG prompts:

**Top performers for SVG generation:**
1. **Gemini 3.0 Pro Preview** -- Best at detailed compositions with proper gradients and spatial layout
2. **Claude Sonnet 4.5** -- Strong character design and recognizable objects
3. **Gemini 2.5 Pro** -- Good execution on complex subjects

**Weaker performers:**
- DeepSeek V3.2-Exp produced abstract rather than coherent imagery
- Several models struggled with complex spatial relationships

### Strengths for Logo Generation
- **Instant output** (seconds, not minutes)
- **Fully editable** -- output is clean SVG code with named elements
- **Deterministic structure** -- circles, rects, paths are explicit; no optimization artifacts
- **Iterative refinement** -- can ask the LLM to modify specific elements
- **Small file sizes** -- LLMs tend to use simple primitives
- **No GPU required** -- just an API call

### Limitations
- Limited visual complexity compared to diffusion methods
- Models struggle with complex organic shapes
- No photorealistic capability
- Quality varies significantly between models
- 16K-100K token context limits SVG complexity

### StarVector (CVPR 2025)

A specialized multimodal LLM for SVG generation:
- **Architecture**: Image encoder + StarCoder LLM adapter
- **Variants**: 1B and 8B parameter models
- **Training**: SVG-Stack dataset (2.1M samples)
- **Capabilities**: Image-to-SVG and Text-to-SVG; handles circles, polygons, text elements, complex paths
- **Strengths**: Icons, logotypes, technical diagrams
- **Limitation**: 16K token context insufficient for highly complex SVGs

### OmniSVG (NeurIPS 2025)

The first end-to-end multimodal SVG generator using Vision-Language Models:
- Built on Qwen2.5-VL with custom SVG tokenizer
- Parameterizes SVG commands/coordinates as discrete tokens
- Generates from simple icons to complex anime characters
- Also supports Lottie animation generation (OmniLottie, March 2026)
- Trained on MMSVG-2M dataset (2M annotated SVG assets)
- Model weights: OmniSVG1.1_4B and OmniSVG1.1_8B

### Claude Code Skill Approach

A practical workflow for logo generation:
- Use Claude as a conversational SVG designer
- Describe the logo concept in natural language
- Claude generates SVG code directly
- Iterate through conversation ("make the circle larger", "change color to blue")
- Export final SVG, optionally render to PNG

Source: https://simonwillison.net/2025/Nov/25/llm-svg-generation-benchmark/

---

## 4. Recraft V4 Native SVG API

Recraft V4 (February 2026) is currently the only production image generation model that produces native vector output -- true SVG geometry, not raster-to-trace conversion.

### Model Variants

| Model | Output | Time | Price |
|-------|--------|------|-------|
| Recraft V4 | Raster 1024x1024 | ~10s | $0.04 |
| Recraft V4 Vector | SVG | ~15s | $0.08 |
| Recraft V4 Pro | Raster 2048x2048 | ~30s | $0.25 |
| Recraft V4 Pro Vector | SVG (higher detail) | ~45s | $0.30 |

### API Usage (via fal.ai)

```python
import fal_client

result = fal_client.submit(
    "fal-ai/recraft/v4/text-to-vector",
    arguments={
        "prompt": "minimalist fox logo, flat design, geometric",
        "image_size": "square_hd",
        "colors": [
            {"r": 255, "g": 107, "b": 53},
            {"r": 33, "g": 33, "b": 33}
        ],
        "background_color": {"r": 255, "g": 255, "b": 255}
    }
)
# Returns: {"images": [{"url": "...", "content_type": "image/svg+xml"}]}
```

### API Parameters
- `prompt` (required): Text description, 1-10000 chars
- `image_size`: square_hd, square, portrait_4_3, portrait_16_9, landscape_4_3, landscape_16_9
- `colors`: Array of RGB objects for color palette control
- `background_color`: RGB object
- `enable_safety_checker`: Boolean (default true)

### Output Formats
SVG, PNG, JPG, PDF, TIFF, and Lottie

### Key Characteristics
- Produces editable SVG with clean paths and structured layers
- Discrete color regions suitable for professional design workflows
- 100+ available styles for consistent visual systems
- Available via Recraft API, Replicate, fal.ai, WaveSpeedAI
- Commercial use licensed

### Logo Suitability
**Strong choice for logos.** Native SVG output means clean vector paths without tracing artifacts. Color palette control is particularly useful for brand guidelines. The $0.08 per generation makes iterative exploration cheap. Main limitation: less control over exact geometric structure compared to hand-crafted or LLM-generated SVG.

Sources:
- https://www.recraft.ai/docs/recraft-models/recraft-V4
- https://fal.ai/models/fal-ai/recraft/v4/text-to-vector

---

## 5. Hybrid Approach: AI Raster -> Vectorize vs Direct SVG

### Approach A: Generate Raster, Then Vectorize

**Pipeline:** Text prompt -> Raster image (FLUX, DALL-E, Midjourney, etc.) -> Vectorization (vtracer, potrace, Vectorizer.AI)

**Vectorization Tools:**

| Tool | Algorithm | Color | Speed | Best For |
|------|-----------|-------|-------|----------|
| **Potrace** | O(n^2) fitting | Binary (B&W only) | Moderate | Simple logos, text; fewest paths, cleanest output |
| **vtracer** | O(n) clustering | Full color | Fast (linear scaling) | Colored logos, complex images; Python: `pip install vtracer` |
| **Vectorizer.AI** | AI-enhanced | Full color | Cloud API | High-fidelity conversion; subscription service |
| **Adobe Image Trace** | Proprietary | Full color | Desktop | Professional workflow integration |

**vtracer Python usage:**
```python
import vtracer

vtracer.convert_image_to_svg_py(
    input_path="logo.png",
    output_path="logo.svg",
    colormode="color",        # or "binary"
    hierarchical="stacked",   # or "cutout"
    mode="spline",            # or "polygon", "none"
    filter_speckle=4,         # remove noise
    color_precision=6,        # fewer = simpler SVG
    layer_difference=16,      # color clustering threshold
    corner_threshold=60,
    length_threshold=4.0,
    max_iterations=10,
    splice_threshold=45,
    path_precision=3
)
```

**Pros of raster-then-vectorize:**
- Access to the full power of raster image generation models
- More visual complexity and photorealism possible
- Well-understood pipeline with mature tools

**Cons:**
- Lossy conversion: vector tracing is interpretation, not reconstruction
- More anchor points = larger files; fewer = lost detail
- Requires clean input (flat colors, crisp edges) for good results
- Two-step process with potential quality degradation
- Bloated SVG output compared to hand-crafted vectors

### Approach B: Direct SVG Generation

**Methods:** Recraft V4 Vector, LLM code generation, SVGDreamer, StarVector, OmniSVG

**Pros:**
- Clean vector paths from the start
- Smaller file sizes with proper structure
- No tracing artifacts
- Editable, semantic structure

**Cons:**
- Limited model choices
- Less visual complexity achievable
- Newer technology, less proven at scale

### Recommendation for Logos

**Direct SVG is strongly preferred for logos** because:
1. Logos need clean geometry, not traced approximations
2. File size matters for web deployment
3. Editability is essential for brand systems
4. Logos use simple shapes that direct methods handle well

Source: https://www.svggenie.com/blog/raster-to-vector-ai-vectorization-2026

---

## 6. Quality Comparison of SVG Generation Methods

### Method Ranking for Logo Use Cases

| Rank | Method | Quality | Speed | Editability | Cost |
|------|--------|---------|-------|-------------|------|
| 1 | **Recraft V4 Vector** | High (native SVG, clean paths) | 15-45s | High | $0.08-0.30/image |
| 2 | **LLM (Claude/Gemini)** | Medium-High (geometric, clean code) | 2-10s | Very High | API token cost |
| 3 | **OmniSVG** | High (icons to complex) | Varies | High | Self-hosted (GPU) |
| 4 | **StarVector** | Medium-High (icons, logotypes) | Varies | High | Self-hosted (GPU) |
| 5 | **SVGDreamer (Iconography)** | Medium (artistic, less precise) | 10-30min | Medium | Self-hosted (GPU) |
| 6 | **Raster + vtracer** | Medium (tracing artifacts) | 10-30s total | Low-Medium | Raster model cost |
| 7 | **Raster + potrace** | Medium for B&W logos | 5-15s total | Low-Medium | Raster model cost |

### Key Quality Observations

1. **Most "AI SVG generators" are fakes**: They generate raster images then auto-trace, producing bloated, unusable vector code. Only Recraft, direct LLM generation, and research models produce true native SVG.

2. **LLM-generated SVGs are surprisingly good for logos**: Simple geometric shapes, text, and icons are well within LLM capabilities. The code is clean and immediately editable.

3. **Recraft V4 is the production standard**: Only commercially available model producing native SVG at scale with consistent quality.

4. **Research models (OmniSVG, StarVector) are promising but require GPU hosting**: Not yet available as simple API calls for production use.

5. **For complex illustrations**: Raster-then-vectorize still wins on visual richness, though at the cost of editability and file bloat.

---

## 7. SVG Manipulation in Python

### Library Comparison

#### svgpathtools
```bash
pip install svgpathtools
```
- **Focus**: Path analysis and geometric operations
- **Core classes**: `Line`, `QuadraticBezier`, `CubicBezier`, `Arc`
- **Capabilities**: Read/write SVG files, path intersections, bounding boxes, transforms, path splitting/joining, arc length computation
- **Latest**: v1.7.2 (November 2025)

```python
from svgpathtools import svg2paths, wsvg, Path, Line, CubicBezier

# Read SVG
paths, attributes = svg2paths('logo.svg')

# Analyze paths
for path in paths:
    print(f"Length: {path.length()}")
    print(f"Bbox: {path.bbox()}")

# Create new paths
new_path = Path(
    Line(complex(0, 0), complex(100, 0)),
    Line(complex(100, 0), complex(100, 100)),
    Line(complex(100, 100), complex(0, 0))
)

# Write SVG
wsvg(paths + [new_path], filename='modified.svg')
```

#### drawsvg
```bash
pip install drawsvg
```
- **Focus**: Programmatic SVG creation and visualization
- **Capabilities**: Most common SVG tags, animations, Jupyter widget, PNG/MP4 rendering
- **Best for**: Creating SVGs from scratch, interactive notebooks

```python
import drawsvg as draw

d = draw.Drawing(200, 200, origin='center')
d.append(draw.Circle(0, 0, 60, fill='#E94560', stroke='white', stroke_width=3))
d.append(draw.Rectangle(-30, -10, 60, 20, fill='white', rx=5))
d.append(draw.Text('LOGO', 16, 0, 40, center=True, fill='#333'))
d.save_svg('logo.svg')
d.save_png('logo.png')  # requires cairo
```

#### xml.etree.ElementTree (stdlib)
```python
import xml.etree.ElementTree as ET

# Parse existing SVG
tree = ET.parse('logo.svg')
root = tree.getroot()

# SVG namespace
ns = {'svg': 'http://www.w3.org/2000/svg'}

# Find and modify elements
for path in root.findall('.//svg:path', ns):
    path.set('fill', '#FF5733')

# Add new element
circle = ET.SubElement(root, 'circle')
circle.set('cx', '50')
circle.set('cy', '50')
circle.set('r', '25')
circle.set('fill', '#333')

# Write back
tree.write('modified.svg', xml_declaration=True)
```

#### Other Notable Libraries
- **svgelements**: SVG parsing with element-level access; `pip install svgelements`
- **svg_ultralight**: Lightweight SVG creation; emphasis on simplicity
- **cairosvg**: SVG to PNG/PDF rendering; `pip install cairosvg`
- **lxml**: More powerful XML parsing than ElementTree; XPath support

### Recommended Stack for Logo Pipeline

| Task | Library |
|------|---------|
| Generate SVG from scratch | `drawsvg` |
| Parse and analyze paths | `svgpathtools` |
| Simple XML modifications | `xml.etree.ElementTree` |
| Complex XML queries | `lxml` |
| SVG to PNG rendering | `cairosvg` |
| Path optimization | `svgpathtools` + custom code |
| Vectorize raster images | `vtracer` (Python bindings) |

---

## 8. When to Use Which Approach

### Decision Matrix

```
START
  |
  v
Is the logo simple/geometric (text, shapes, icons)?
  |
  YES --> Use LLM (Claude/Gemini) for direct SVG code generation
  |         - Fastest iteration cycle
  |         - Most editable output
  |         - Best for: tech logos, wordmarks, abstract geometric marks
  |
  NO --> Does the logo need complex illustration/organic shapes?
          |
          YES --> Use Recraft V4 Vector API
          |         - Native SVG with visual complexity
          |         - Best for: mascot logos, illustrated marks, detailed icons
          |         - Fallback: Raster generation + vtracer if Recraft quality insufficient
          |
          NO --> Do you need full artistic control over vector primitives?
                  |
                  YES --> Use SVGDreamer (Iconography mode)
                  |         - Research-grade quality
                  |         - Slow but controllable
                  |         - Best for: artistic/stylized logos
                  |
                  NO --> Use hybrid pipeline
                          - Generate with best raster model
                          - Vectorize with vtracer (color) or potrace (B&W)
                          - Post-process with svgpathtools
```

### Practical Recommendations by Scenario

**Startup MVP / Quick Prototyping:**
- Primary: LLM direct SVG generation (Claude)
- Cost: ~$0.01-0.05 per iteration
- Iterate in conversation, export when satisfied

**Production Logo Pipeline:**
- Primary: Recraft V4 Vector API ($0.08/generation)
- Generate multiple variants, select best
- Post-process with svgpathtools for optimization
- Final touches in Figma/Illustrator

**Brand System with Multiple Variants:**
- Generate base logo with Recraft V4 or LLM
- Use drawsvg/svgpathtools to programmatically create color variants
- Use xml.etree to batch-modify attributes across variants

**High-Volume Icon/Logo Generation:**
- Self-host OmniSVG or StarVector if GPU available
- Otherwise Recraft V4 API at $0.08/image
- Post-process with vtracer for any raster-generated assets

**Research/Experimentation:**
- PyTorch-SVGRender with SVGDreamer for exploring artistic styles
- OmniSVG for cutting-edge quality
- Requires GPU infrastructure

### Cost-Quality Tradeoff Summary

| Approach | Cost/Logo | Quality | Speed | GPU Required |
|----------|-----------|---------|-------|-------------|
| Claude/GPT SVG code | $0.01-0.05 | 7/10 for simple logos | Seconds | No |
| Recraft V4 Vector | $0.08-0.30 | 8/10 | 15-45s | No |
| Raster + vtracer | $0.04 + free | 6/10 | 15-30s | No |
| OmniSVG (self-hosted) | GPU cost | 8/10 | Minutes | Yes |
| SVGDreamer (self-hosted) | GPU cost | 7/10 artistic | 10-30min | Yes |
