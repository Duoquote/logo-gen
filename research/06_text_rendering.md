# Text Rendering in AI-Generated Logos
## Research Report - March 2026

---

## 1. Model Comparison: Text Rendering Accuracy

### Tier 1: Near-Perfect Text Rendering

**Ideogram 3.0** - Best overall text accuracy (~90%)
- Built by former Google Brain researchers who specifically targeted text rendering
- Handles multi-word text, mixed case, and special characters
- Supports text placement control via "text in the center", "text at the top"
- Color palette feature accepts hex codes for brand consistency
- Weakness: occasional letter duplication on words >10 characters
- API: ~$0.06/image, Together AI integration available

**Reve (by Reve AI)** - Trained on 50M+ font samples
- Purpose-built typography model, excels at serif and script fonts
- Handles ligatures, kerning, and baseline alignment natively
- Best for wordmark-style logos where text IS the logo
- Renders up to ~15 characters reliably
- Weakness: less control over non-text visual elements

**Recraft V4** - Best for vector text output
- Native SVG output means text paths are mathematically precise
- Text rendered as vector paths, not rasterized glyphs
- Excellent for logos that need to scale from favicon to billboard
- API: $0.08/image for vector output
- Weakness: limited font style variety compared to Ideogram

### Tier 2: Good Text Rendering (70-85% accuracy)

**Flux 2 Pro (Black Forest Labs)** - Best open-weight text rendering
- 32B parameter model with dedicated text encoding pathways
- Reliable up to ~8 characters, degrades on longer strings
- FLUX.2 Dev (open-weight) nearly matches Pro for text quality
- Can be enhanced with text-specific LoRAs (see Section 3)
- Community LoRAs like Harrlogos push accuracy to ~85%

**GPT Image 1.5 (OpenAI)** - Strong multimodal text
- Leverages language model understanding for text coherence
- Good at rendering text in context (on signs, labels, banners)
- Handles longer phrases better than most diffusion models
- Weakness: sometimes adds unwanted decorative elements to text

**Seedream 4.5 (ByteDance)** - Underrated text capability
- Strong CJK character rendering (Chinese, Japanese, Korean)
- Good Latin text up to ~10 characters
- Available via API, less community tooling

### Tier 3: Acceptable with Workarounds (50-70%)

**Stable Diffusion 3.5 Large** - Improved but inconsistent
- MMDiT architecture improved text vs. SD 1.5/SDXL
- Requires LoRAs or ControlNet guidance for reliable text
- Best with short text (3-5 characters), single words
- Community LoRAs essential for production use

**Midjourney V8** - Aesthetic but unreliable text
- Beautiful typography styling but frequent misspellings
- Best for single-word logos or monograms (1-3 letters)
- No API access limits automation potential

**Imagen 4 (Google)** - Improving rapidly
- Better than Imagen 3 for text but behind Ideogram
- Limited API access through Vertex AI

### Model Selection for Text-in-Logo Use Cases

| Scenario | Recommended Model | Fallback Strategy |
|----------|------------------|-------------------|
| Wordmark (text-only logo) | Ideogram 3.0 or Reve | Hybrid: generate font style, composite with Pillow |
| Monogram (1-3 letters) | Recraft V4 (SVG) | Any Tier 1-2 model works |
| Symbol + company name | Ideogram 3.0 | Hybrid: AI symbol + Pillow text |
| Tagline under logo | Hybrid approach | Text too long for reliable AI rendering |
| Multi-language text | Seedream 4.5 (CJK) | Always use hybrid for non-Latin scripts |
| Vector/scalable output | Recraft V4 | Generate raster, trace with potrace/vtracer |

---

## 2. Prompt Techniques for Accurate Text Rendering

### Core Rules for Text Prompts

**Rule 1: Quote the exact text**
```
a logo with the text "AURORA" in bold sans-serif letters
```
Not: `a logo that says Aurora` (case and format ambiguity).

**Rule 2: Spell out letter by letter for difficult words**
```
the word "SYNAPSE" spelled S-Y-N-A-P-S-E in geometric uppercase letters
```
This technique reduces character swapping/duplication by 20-30%.

**Rule 3: Specify typography attributes explicitly**
```
the text "NEXUS" in heavyweight sans-serif, tracked-out uppercase,
modern geometric letterforms, consistent stroke width
```
Vague descriptions like "nice font" produce inconsistent results.

**Rule 4: Keep text SHORT**
- 1-5 characters: high reliability across all models
- 6-10 characters: reliable on Tier 1 models
- 11-15 characters: only Ideogram 3.0 / Reve with careful prompting
- 16+ characters: use hybrid approach (Section 4)

**Rule 5: Separate text from symbol in the prompt**
```
A minimalist mountain logo mark above the text "ALPINE",
the word "ALPINE" in clean sans-serif capitals below the mountain symbol,
white background, flat vector style
```
Repeating the text and its placement improves accuracy.

### Model-Specific Prompt Strategies

**Ideogram 3.0:**
```
Logo design: a geometric lion head icon above the text "BRAVERA".
Typography: "BRAVERA" in bold modern sans-serif, tracked uppercase.
Style: flat vector, navy blue and gold, white background.
Aspect ratio: 1:1.
```
- Use structured sections (Logo design / Typography / Style)
- Repeat text string 2x in prompt
- Specify aspect ratio explicitly

**Flux 2 (with or without LoRA):**
```
professional logo design, the word "NOVA" in thick geometric sans-serif
letters, minimalist style, solid white background, clean vector illustration,
high contrast, centered composition
```
- Front-load the text element in the prompt
- Include "professional" and "clean" to reduce artifacts
- Use with Harrlogos LoRA for best results (see Section 3)

**Recraft V4:**
```
Modern tech startup logo. The letters "QR" interlinked as a monogram.
Geometric, minimal, single color navy blue. Vector style.
```
- Recraft responds well to design brief-style prompts
- Specify "vector style" to trigger SVG output mode
- Works best for monograms and short wordmarks

**Stable Diffusion 3.5 / SDXL (with LoRA):**
```
wablogo, logo design, the text "DRIFT" in bold modern font,
minimalist flat vector, solid white background, professional branding
```
- Include LoRA trigger words (e.g., "wablogo" for Flux Logo LoRA)
- Keep text to single word
- Add "solid white background" to prevent text blending into complex backgrounds

### Negative Prompt Patterns for Text
```
blurry text, misspelled, extra letters, missing letters, distorted text,
overlapping letters, illegible, low quality, watermark, multiple logos,
decorative borders, gradient background, 3D text, drop shadow on text
```

### Advanced Techniques

**Two-pass generation (Ideogram/GPT Image):**
1. First pass: generate the symbol/icon only with "no text" in the prompt
2. Second pass: use image-to-image with the symbol as input, add text prompt
3. This separates the model's attention between symbol and text tasks

**Seed locking for text iteration:**
- Fix the seed, change only the text portion of the prompt
- Maintains consistent symbol while iterating on text placement
- Works well in ComfyUI with KSampler seed control

**ControlNet text guidance (SD/Flux):**
1. Create a text layout image in Pillow/Figma with exact positioning
2. Feed as Canny ControlNet input with strength 0.4-0.6
3. Model follows text placement without distorting the artistic style

---

## 3. LoRA-Based Text Generation

### Harrlogos XL v2.0

The most widely used LoRA for text rendering in logos on SDXL.

**Model details:**
- Architecture: SDXL LoRA
- Training: ~5,000 logo images with accurate text
- Download: CivitAI (search "Harrlogos XL"), HuggingFace
- File size: ~150MB
- Trigger word: `text logo` or `harrlogos`

**Usage in ComfyUI:**
```
Nodes: CheckpointLoader (SDXL base) -> LoraLoader (Harrlogos) -> CLIPTextEncode -> KSampler

Prompt: "text logo, the word 'SPARK' in bold uppercase letters,
minimalist design, white background, professional"

Settings:
- LoRA strength: 0.7-0.9 (higher = stronger text adherence)
- Steps: 30-40
- CFG: 7-8
- Sampler: DPM++ 2M Karras
- Resolution: 1024x1024
```

**Prompt formula for Harrlogos:**
```
text logo, the word "[YOUR TEXT]", [font style], [color scheme],
[background], [additional style keywords]
```

**Limitations:**
- Best with 3-8 character words
- Single word only; multi-word text unreliable
- English/Latin alphabet only
- Can conflict with other LoRAs if stacked

### Shakker-Labs FLUX.1-dev-LoRA-Logo-Design

**Model details:**
- Architecture: Flux.1-dev LoRA
- Trigger words: "wablogo, logo, Minimalist"
- Download: HuggingFace (Shakker-Labs)
- Recommended settings: 24 steps, guidance_scale=3.5, lora_scale=0.8

**Usage:**
```python
# With diffusers
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.load_lora_weights("Shakker-Labs/FLUX.1-dev-LoRA-Logo-Design")

image = pipe(
    prompt='wablogo, logo, Minimalist, the text "APEX" in bold geometric font, clean lines',
    num_inference_steps=24,
    guidance_scale=3.5,
    width=1024,
    height=1024,
).images[0]
```

### Other Notable Text/Logo LoRAs

| LoRA | Base Model | Strength | Best For |
|------|-----------|----------|----------|
| Harrlogos XL v2.0 | SDXL | 0.7-0.9 | Text logos, wordmarks |
| FLUX Logo Design | Flux.1-dev | 0.8 | Minimalist logo icons + text |
| Logo.Redmond | SDXL | 0.6-0.8 | Corporate/brand logos |
| LogoRedmond V2 | SDXL | 0.7 | Improved text, icon marks |
| Neon Logo LoRA | SDXL | 0.7 | Neon/glowing text effects |
| Vintage Logo LoRA | SD 1.5 | 0.8 | Retro badge/emblem style text |

### Training a Custom Text LoRA

For project-specific needs, training a custom LoRA is feasible:

```
Dataset requirements:
- 50-200 logo images with accurate text
- Consistent style within the dataset
- Text captions describing each logo accurately
- Resolution: 1024x1024 for SDXL, 512x512 for SD 1.5

Training parameters (SDXL LoRA):
- Learning rate: 1e-4
- Rank: 32-64
- Steps: 2000-5000
- Optimizer: AdamW8bit
- Tools: kohya_ss, SimpleTuner, ai-toolkit
```

---

## 4. Hybrid Approach: AI Symbol + Programmatic Text

### Why Hybrid?

AI models excel at generating unique symbols, icons, and abstract marks but still struggle with precise typography. The hybrid approach plays to each tool's strengths:

- **AI model** generates the symbol/icon/mark (what it does best)
- **Programmatic text** (Pillow, Cairo, SVG) handles typography (deterministic, pixel-perfect)
- Result: professional logos with accurate text every time

### Architecture Overview

```
[User Input: brand name, style, colors]
        |
        v
[Prompt Generator] -- builds symbol-only prompt
        |
        v
[AI Model API] -- generates symbol/icon (no text)
        |
        v
[Background Removal] -- rembg / transparent background
        |
        v
[Text Renderer] -- Pillow/Cairo adds brand name + tagline
        |
        v
[Compositor] -- positions symbol + text in final layout
        |
        v
[Output: PNG/SVG logo]
```

### Implementation Pipeline

```python
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from rembg import remove

class HybridLogoGenerator:
    """Generate logo symbol with AI, add text programmatically."""

    def __init__(self, api_client, font_dir="./fonts"):
        self.api = api_client
        self.font_dir = font_dir

    def generate_symbol(self, concept, style="minimalist", colors=None):
        """Generate symbol-only logo via AI model."""
        prompt = (
            f"Minimalist {style} logo icon, {concept}, "
            f"simple geometric shapes, clean lines, flat vector style, "
            f"centered on solid white background, no text, no letters, "
            f"no words, professional brand mark"
        )
        if colors:
            prompt += f", color palette: {', '.join(colors)}"

        # Call your preferred API (Ideogram, Flux, Recraft, etc.)
        image = self.api.generate(prompt=prompt, size=(1024, 1024))
        return image

    def remove_background(self, image):
        """Remove white/solid background from symbol."""
        return remove(image)  # rembg library

    def render_text(self, text, font_path, font_size, color="#000000",
                    tracking=0, uppercase=True):
        """Render brand text with precise typography control."""
        if uppercase:
            text = text.upper()

        font = ImageFont.truetype(font_path, font_size)

        # Calculate text dimensions with tracking (letter-spacing)
        if tracking > 0:
            total_width = sum(
                font.getbbox(char)[2] + tracking for char in text
            ) - tracking
        else:
            bbox = font.getbbox(text)
            total_width = bbox[2] - bbox[0]

        bbox = font.getbbox(text)
        text_height = bbox[3] - bbox[1]

        # Create transparent text image
        text_img = Image.new("RGBA", (total_width + 20, text_height + 20), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)

        if tracking > 0:
            x = 10
            for char in text:
                draw.text((x, 10), char, fill=color, font=font)
                x += font.getbbox(char)[2] + tracking
        else:
            draw.text((10, 10), text, fill=color, font=font)

        return text_img

    def composite_logo(self, symbol, text_img, tagline_img=None,
                       layout="vertical", padding=40, canvas_size=(1200, 1200)):
        """Compose symbol and text into final logo."""
        canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 0))

        if layout == "vertical":
            # Symbol on top, text below
            sym_scale = min(
                (canvas_size[0] - 2 * padding) / symbol.width,
                (canvas_size[1] * 0.55) / symbol.height
            )
            sym_resized = symbol.resize(
                (int(symbol.width * sym_scale), int(symbol.height * sym_scale)),
                Image.LANCZOS
            )

            # Center symbol horizontally, position in upper portion
            sym_x = (canvas_size[0] - sym_resized.width) // 2
            sym_y = padding
            canvas.paste(sym_resized, (sym_x, sym_y), sym_resized)

            # Scale and center text below symbol
            text_scale = min(1.0, (canvas_size[0] - 2 * padding) / text_img.width)
            text_resized = text_img.resize(
                (int(text_img.width * text_scale), int(text_img.height * text_scale)),
                Image.LANCZOS
            )
            text_x = (canvas_size[0] - text_resized.width) // 2
            text_y = sym_y + sym_resized.height + padding
            canvas.paste(text_resized, (text_x, text_y), text_resized)

            # Optional tagline
            if tagline_img:
                tag_scale = min(1.0, (canvas_size[0] - 2 * padding) / tagline_img.width)
                tag_resized = tagline_img.resize(
                    (int(tagline_img.width * tag_scale), int(tagline_img.height * tag_scale)),
                    Image.LANCZOS
                )
                tag_x = (canvas_size[0] - tag_resized.width) // 2
                tag_y = text_y + text_resized.height + padding // 2
                canvas.paste(tag_resized, (tag_x, tag_y), tag_resized)

        elif layout == "horizontal":
            # Symbol on left, text on right
            sym_scale = min(
                (canvas_size[0] * 0.35) / symbol.width,
                (canvas_size[1] - 2 * padding) / symbol.height
            )
            sym_resized = symbol.resize(
                (int(symbol.width * sym_scale), int(symbol.height * sym_scale)),
                Image.LANCZOS
            )
            sym_x = padding
            sym_y = (canvas_size[1] - sym_resized.height) // 2
            canvas.paste(sym_resized, (sym_x, sym_y), sym_resized)

            # Text to the right of symbol, vertically centered
            text_scale = min(
                1.0,
                (canvas_size[0] - sym_resized.width - 3 * padding) / text_img.width
            )
            text_resized = text_img.resize(
                (int(text_img.width * text_scale), int(text_img.height * text_scale)),
                Image.LANCZOS
            )
            text_x = sym_x + sym_resized.width + padding
            text_y = (canvas_size[1] - text_resized.height) // 2
            canvas.paste(text_resized, (text_x, text_y), text_resized)

        return canvas

    def generate(self, brand_name, concept, tagline=None,
                 font_path=None, style="minimalist", colors=None,
                 layout="vertical"):
        """Full pipeline: generate symbol + add text."""
        # 1. Generate AI symbol
        symbol = self.generate_symbol(concept, style, colors)

        # 2. Remove background
        symbol = self.remove_background(symbol)

        # 3. Render brand name
        if font_path is None:
            font_path = f"{self.font_dir}/Montserrat-Bold.ttf"
        text_img = self.render_text(
            brand_name, font_path, font_size=72,
            color=colors[0] if colors else "#1a1a1a",
            tracking=8
        )

        # 4. Optional tagline
        tagline_img = None
        if tagline:
            tagline_font = font_path.replace("Bold", "Regular")
            tagline_img = self.render_text(
                tagline, tagline_font, font_size=28,
                color="#666666", tracking=4
            )

        # 5. Composite
        logo = self.composite_logo(symbol, text_img, tagline_img, layout)
        return logo
```

### Symbol-Only Prompt Patterns

Critical: always explicitly exclude text from symbol generation.

```
# Good - explicit text exclusion
"abstract geometric flame icon, no text, no letters, no words,
 no typography, simple flat design, centered, white background"

# Good - describe as icon/mark only
"minimalist bird logo mark, symbol only, no text whatsoever,
 clean vector, single color navy blue, white background"

# Bad - ambiguous, model may add text
"modern tech company logo, sleek design"
```

---

## 5. Font Management in Python

### Pillow ImageFont

```python
from PIL import ImageFont

# Load TrueType/OpenType font
font = ImageFont.truetype("fonts/Montserrat-Bold.ttf", size=72)

# Load with specific encoding
font = ImageFont.truetype("fonts/NotoSans-Regular.ttf", size=48, encoding="unic")

# System font fallback (not recommended for production)
font = ImageFont.load_default()

# Get text bounding box (Pillow 10+)
bbox = font.getbbox("HELLO")  # (left, top, right, bottom)
width = bbox[2] - bbox[0]
height = bbox[3] - bbox[1]

# Get text size for multi-line (Pillow 10+)
left, top, right, bottom = ImageDraw.Draw(Image.new("RGB", (1,1))).multiline_textbbox(
    (0, 0), "Line 1\nLine 2", font=font
)

# Font metrics
ascent, descent = font.getmetrics()
```

### fonttools Library

`fonttools` is the standard Python library for reading, inspecting, and manipulating font files (TTF, OTF, WOFF, WOFF2).

```python
from fontTools.ttLib import TTFont

# Load and inspect font
font = TTFont("fonts/Montserrat-Bold.ttf")

# Get font family name
name_table = font['name']
for record in name_table.names:
    if record.nameID == 1:  # Font Family
        print(f"Family: {record.toUnicode()}")
    if record.nameID == 2:  # Font Subfamily (style)
        print(f"Style: {record.toUnicode()}")

# List available glyphs
glyph_order = font.getGlyphOrder()
print(f"Total glyphs: {len(glyph_order)}")

# Check if font supports specific characters
cmap = font.getBestCmap()
has_ampersand = ord('&') in cmap  # True/False

# Get font metrics
os2 = font['OS/2']
print(f"Weight class: {os2.usWeightClass}")  # 400=Regular, 700=Bold
print(f"Width class: {os2.usWidthClass}")

# Subsetting (reduce font size for embedding)
from fontTools.subset import Subsetter, Options

options = Options()
subsetter = Subsetter(options=options)
subsetter.populate(text="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
subsetter.subset(font)
font.save("fonts/Montserrat-Bold-subset.ttf")
```

### Font Discovery and Management

```python
import os
import glob
from pathlib import Path
from fontTools.ttLib import TTFont

class FontManager:
    """Manage and discover fonts for logo text rendering."""

    # Common system font directories
    SYSTEM_FONT_DIRS = {
        "win32": [
            os.path.expandvars(r"%WINDIR%\Fonts"),
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts"),
        ],
        "darwin": [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.expanduser("~/Library/Fonts"),
        ],
        "linux": [
            "/usr/share/fonts",
            "/usr/local/share/fonts",
            os.path.expanduser("~/.local/share/fonts"),
            os.path.expanduser("~/.fonts"),
        ],
    }

    def __init__(self, custom_dirs=None, platform="win32"):
        self.fonts = {}  # {family_name: {style: path}}
        self.font_dirs = self.SYSTEM_FONT_DIRS.get(platform, [])
        if custom_dirs:
            self.font_dirs = custom_dirs + self.font_dirs
        self._scan_fonts()

    def _scan_fonts(self):
        """Scan directories and index all available fonts."""
        for font_dir in self.font_dirs:
            for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
                for path in glob.glob(os.path.join(font_dir, "**", ext), recursive=True):
                    try:
                        tt = TTFont(path, fontNumber=0)
                        names = tt['name']
                        family = None
                        style = "Regular"
                        for record in names.names:
                            if record.nameID == 1:
                                family = record.toUnicode()
                            if record.nameID == 2:
                                style = record.toUnicode()
                        tt.close()
                        if family:
                            if family not in self.fonts:
                                self.fonts[family] = {}
                            self.fonts[family][style] = path
                    except Exception:
                        continue

    def get_font(self, family, style="Regular", size=72):
        """Get a Pillow ImageFont by family name and style."""
        if family in self.fonts and style in self.fonts[family]:
            return ImageFont.truetype(self.fonts[family][style], size)
        # Fuzzy match
        for fam in self.fonts:
            if family.lower() in fam.lower():
                styles = self.fonts[fam]
                if style in styles:
                    return ImageFont.truetype(styles[style], size)
                # Return any available style
                path = next(iter(styles.values()))
                return ImageFont.truetype(path, size)
        raise FileNotFoundError(f"Font '{family}' ({style}) not found")

    def list_families(self):
        """List all available font families."""
        return sorted(self.fonts.keys())

    def get_logo_fonts(self):
        """Return fonts commonly used in logos."""
        logo_families = [
            "Montserrat", "Poppins", "Inter", "Raleway", "Oswald",
            "Bebas Neue", "Playfair Display", "Roboto", "Futura",
            "Lato", "Nunito", "Work Sans", "Source Sans 3", "DM Sans",
            "Space Grotesk", "Outfit", "Clash Display", "Satoshi",
            "General Sans", "Cabinet Grotesk"
        ]
        available = {}
        for family in logo_families:
            if family in self.fonts:
                available[family] = self.fonts[family]
        return available
```

### Downloading Google Fonts Programmatically

```python
import requests
import zipfile
from io import BytesIO

def download_google_font(family_name, output_dir="./fonts"):
    """Download a font family from Google Fonts."""
    os.makedirs(output_dir, exist_ok=True)

    # Google Fonts API (no key required for download)
    url = f"https://fonts.google.com/download?family={family_name.replace(' ', '+')}"
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        for name in zf.namelist():
            if name.endswith(('.ttf', '.otf')):
                zf.extract(name, output_dir)
                print(f"Extracted: {name}")

# Download popular logo fonts
for font in ["Montserrat", "Poppins", "Inter", "Bebas Neue", "Raleway"]:
    download_google_font(font)
```

---

## 6. Text Compositing onto AI-Generated Symbols

### Basic Compositing with Pillow

```python
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def composite_text_on_symbol(
    symbol_path,
    brand_name,
    font_path,
    font_size=72,
    text_color="#1a1a1a",
    layout="below",          # "below", "above", "right", "overlay"
    padding=30,
    canvas_color=(255, 255, 255, 0),  # transparent
    output_size=(1200, 1200),
):
    """Composite text onto/around an AI-generated symbol."""
    symbol = Image.open(symbol_path).convert("RGBA")
    font = ImageFont.truetype(font_path, font_size)

    # Measure text
    temp_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    text_bbox = temp_draw.textbbox((0, 0), brand_name, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    canvas = Image.new("RGBA", output_size, canvas_color)

    if layout == "below":
        # Scale symbol to fit upper portion
        available_h = output_size[1] - text_h - 3 * padding
        scale = min(
            (output_size[0] - 2 * padding) / symbol.width,
            available_h / symbol.height
        )
        sym_w = int(symbol.width * scale)
        sym_h = int(symbol.height * scale)
        symbol_resized = symbol.resize((sym_w, sym_h), Image.LANCZOS)

        # Place symbol centered in upper area
        sym_x = (output_size[0] - sym_w) // 2
        sym_y = padding
        canvas.paste(symbol_resized, (sym_x, sym_y), symbol_resized)

        # Place text centered below
        text_x = (output_size[0] - text_w) // 2
        text_y = sym_y + sym_h + padding
        draw = ImageDraw.Draw(canvas)
        draw.text((text_x, text_y), brand_name, fill=text_color, font=font)

    elif layout == "right":
        # Symbol left, text right, vertically centered
        available_w = output_size[0] - 3 * padding
        sym_max = available_w * 0.4
        scale = min(sym_max / symbol.width, (output_size[1] - 2 * padding) / symbol.height)
        sym_w = int(symbol.width * scale)
        sym_h = int(symbol.height * scale)
        symbol_resized = symbol.resize((sym_w, sym_h), Image.LANCZOS)

        sym_x = padding
        sym_y = (output_size[1] - sym_h) // 2
        canvas.paste(symbol_resized, (sym_x, sym_y), symbol_resized)

        draw = ImageDraw.Draw(canvas)
        text_x = sym_x + sym_w + padding
        text_y = (output_size[1] - text_h) // 2
        draw.text((text_x, text_y), brand_name, fill=text_color, font=font)

    elif layout == "overlay":
        # Text centered over symbol
        scale = min(
            (output_size[0] - 2 * padding) / symbol.width,
            (output_size[1] - 2 * padding) / symbol.height
        )
        sym_w = int(symbol.width * scale)
        sym_h = int(symbol.height * scale)
        symbol_resized = symbol.resize((sym_w, sym_h), Image.LANCZOS)

        sym_x = (output_size[0] - sym_w) // 2
        sym_y = (output_size[1] - sym_h) // 2
        canvas.paste(symbol_resized, (sym_x, sym_y), symbol_resized)

        draw = ImageDraw.Draw(canvas)
        text_x = (output_size[0] - text_w) // 2
        text_y = (output_size[1] - text_h) // 2
        draw.text((text_x, text_y), brand_name, fill=text_color, font=font)

    return canvas
```

### Advanced Compositing with Cairo (pycairo)

Cairo provides superior text rendering with anti-aliasing, subpixel positioning, and path-based operations.

```python
import cairo
from PIL import Image
import numpy as np

def composite_with_cairo(
    symbol_path,
    brand_name,
    font_family="Montserrat",
    font_size=60,
    font_weight=cairo.FONT_WEIGHT_BOLD,
    text_color=(0.1, 0.1, 0.1),
    letter_spacing=4,
    output_size=(1200, 1200),
):
    """High-quality text compositing using Cairo."""
    width, height = output_size

    # Create Cairo surface
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)

    # Transparent background
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()

    # Load and draw symbol
    symbol = Image.open(symbol_path).convert("RGBA")
    sym_data = np.array(symbol)
    # Convert RGBA to Cairo's BGRA format
    sym_data[:, :, [0, 2]] = sym_data[:, :, [2, 0]]
    sym_surface = cairo.ImageSurface.create_for_data(
        sym_data, cairo.FORMAT_ARGB32, symbol.width, symbol.height
    )

    # Scale and center symbol in upper 60%
    sym_scale = min((width - 80) / symbol.width, (height * 0.6) / symbol.height)
    sym_w = symbol.width * sym_scale
    sym_h = symbol.height * sym_scale
    sym_x = (width - sym_w) / 2
    sym_y = 40

    ctx.save()
    ctx.translate(sym_x, sym_y)
    ctx.scale(sym_scale, sym_scale)
    ctx.set_source_surface(sym_surface, 0, 0)
    ctx.paint()
    ctx.restore()

    # Render text with Cairo
    ctx.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, font_weight)
    ctx.set_font_size(font_size)

    # Measure text
    text_extents = ctx.text_extents(brand_name)

    # Account for letter spacing
    total_spacing = letter_spacing * (len(brand_name) - 1)
    total_width = text_extents.width + total_spacing

    # Position text centered below symbol
    text_x = (width - total_width) / 2
    text_y = sym_y + sym_h + 60 + text_extents.height

    ctx.set_source_rgb(*text_color)

    if letter_spacing > 0:
        # Render character by character with spacing
        x = text_x
        for char in brand_name:
            ctx.move_to(x, text_y)
            ctx.show_text(char)
            char_extents = ctx.text_extents(char)
            x += char_extents.x_advance + letter_spacing
    else:
        ctx.move_to(text_x, text_y)
        ctx.show_text(brand_name)

    # Convert Cairo surface to PIL Image
    buf = surface.get_data()
    arr = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=buf)
    # Convert BGRA back to RGBA
    arr[:, :, [0, 2]] = arr[:, :, [2, 0]]
    result = Image.fromarray(arr, "RGBA")
    return result
```

### SVG-Based Compositing

For truly scalable output, compose in SVG:

```python
def create_svg_logo(symbol_svg_path, brand_name, font_family="Montserrat",
                    font_size=48, text_color="#1a1a1a", width=400, height=400):
    """Create an SVG logo combining symbol and text."""
    # Read symbol SVG content
    with open(symbol_svg_path, 'r') as f:
        symbol_svg = f.read()

    # Extract viewBox from symbol or default
    # (simplified - production code should parse the SVG properly)
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"
     width="{width}" height="{height}">

  <!-- Symbol -->
  <g transform="translate({width*0.1}, 20) scale(0.8)">
    {symbol_svg}
  </g>

  <!-- Brand Name -->
  <text x="{width/2}" y="{height - 40}"
        font-family="{font_family}, sans-serif"
        font-size="{font_size}"
        font-weight="700"
        fill="{text_color}"
        text-anchor="middle"
        letter-spacing="3">
    {brand_name.upper()}
  </text>
</svg>'''
    return svg
```

---

## 7. Typography Best Practices for Logos

### Font Selection Rules

**Sans-Serif (most common for modern logos):**
- Geometric sans: Montserrat, Poppins, Futura, Outfit, DM Sans
- Grotesque sans: Inter, Roboto, Helvetica Neue, General Sans
- Humanist sans: Lato, Nunito, Source Sans
- Best for: tech, SaaS, startups, modern brands

**Serif (authority, tradition, luxury):**
- Transitional: Playfair Display, Libre Baskerville
- Modern: Bodoni, Didot
- Slab: Roboto Slab, Rockwell
- Best for: law firms, finance, luxury, editorial

**Display/Statement fonts:**
- Bebas Neue, Clash Display, Cabinet Grotesk
- Use sparingly, only for brand name (not taglines)

### Typography Rules for Logo Design

1. **Maximum two typefaces** - one for the brand name, optionally a second for the tagline. Never more than two.

2. **Weight hierarchy** - brand name in Bold/Black (700-900), tagline in Regular/Light (300-400). The contrast creates visual hierarchy.

3. **Letter-spacing (tracking)** - uppercase text needs increased tracking (+2-8% of font size). Lowercase typically needs less or none. Wider tracking conveys elegance and openness.

4. **Optical sizing** - at small sizes, increase weight and spacing slightly. At large sizes, tighter tracking is acceptable.

5. **Alignment with symbol:**
   - Vertical layout: text width should be 60-100% of symbol width
   - Horizontal layout: text baseline should align with symbol's visual center
   - Text should never overpower the symbol or vice versa

6. **Color contrast** - text must have sufficient contrast against background. Minimum 4.5:1 ratio for accessibility (WCAG AA). For logos, aim for 7:1+ (AAA).

7. **Scalability test** - the logo text must be legible at:
   - Favicon (16x16px) - often just the symbol
   - Mobile header (120px wide)
   - Business card (50mm wide)
   - Billboard (3m wide)

8. **Avoid these common mistakes:**
   - Stretching or distorting font shapes
   - Using more than two font weights
   - Tight tracking on lightweight fonts
   - Script/handwriting fonts at small sizes
   - Trendy fonts that will look dated in 2 years

### Recommended Font Pairings for Logos

| Brand Name Font | Tagline Font | Vibe |
|----------------|-------------|------|
| Montserrat Bold | Montserrat Light | Clean modern |
| Bebas Neue | Lato Regular | Bold statement |
| Playfair Display Bold | Source Sans Regular | Elegant classic |
| Space Grotesk Bold | Inter Regular | Tech forward |
| Outfit Bold | Outfit Light | Contemporary |
| DM Sans Bold | DM Sans Regular | Friendly modern |
| Clash Display Semibold | General Sans Regular | Creative agency |

---

## 8. Complete Text Compositing Pipeline

### Full Working Example

```python
"""
Complete logo generation pipeline:
1. Generate symbol with AI (Flux/Ideogram/Recraft)
2. Remove background
3. Render text with precise typography
4. Composite into final logo
5. Export in multiple formats
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor
from rembg import remove
from io import BytesIO
import json


class LogoTextPipeline:
    """Production text compositing pipeline for AI-generated logos."""

    def __init__(self, font_dir="./fonts", output_dir="./output"):
        self.font_dir = Path(font_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Text Rendering -----

    def render_text_line(
        self,
        text: str,
        font_path: str,
        font_size: int,
        color: str = "#1a1a1a",
        tracking: int = 0,
        uppercase: bool = True,
    ) -> Image.Image:
        """Render a single line of text as a transparent RGBA image."""
        if uppercase:
            text = text.upper()

        font = ImageFont.truetype(str(font_path), font_size)
        color_rgb = ImageColor.getrgb(color)

        # Measure with tracking
        chars = list(text)
        char_widths = []
        max_h = 0
        for ch in chars:
            bbox = font.getbbox(ch)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            char_widths.append((w, bbox))
            max_h = max(max_h, h)

        total_w = sum(w for w, _ in char_widths) + tracking * (len(chars) - 1)

        # Add padding for descenders and anti-aliasing
        pad = font_size // 4
        img = Image.new("RGBA", (total_w + 2 * pad, max_h + 2 * pad), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        x = pad
        ascent, descent = font.getmetrics()
        y_baseline = pad

        if tracking > 0:
            for i, ch in enumerate(chars):
                draw.text((x, y_baseline), ch, fill=color_rgb, font=font)
                w, bbox = char_widths[i]
                x += font.getbbox(ch)[2] + tracking
        else:
            draw.text((pad, y_baseline), text, fill=color_rgb, font=font)

        # Trim transparent edges
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        return img

    # ----- Symbol Processing -----

    def process_symbol(self, symbol_image: Image.Image, remove_bg: bool = True) -> Image.Image:
        """Process AI-generated symbol: remove background, clean edges."""
        if remove_bg:
            symbol_image = remove(symbol_image)

        # Convert to RGBA if needed
        if symbol_image.mode != "RGBA":
            symbol_image = symbol_image.convert("RGBA")

        # Trim transparent borders
        bbox = symbol_image.getbbox()
        if bbox:
            symbol_image = symbol_image.crop(bbox)

        return symbol_image

    # ----- Layout Engine -----

    def compute_layout(
        self,
        symbol_size: tuple,
        text_size: tuple,
        tagline_size: tuple = None,
        canvas_size: tuple = (1200, 1200),
        layout: str = "vertical",
        padding: int = 40,
        text_symbol_ratio: float = 0.7,
    ) -> dict:
        """Compute positions for all elements. Returns coordinate dict."""
        cw, ch = canvas_size

        if layout == "vertical":
            # Allocate vertical space: symbol gets 55-65%, text gets rest
            tag_h = tagline_size[1] + padding if tagline_size else 0
            total_content_h = ch - 3 * padding - tag_h

            # Symbol takes upper portion
            sym_alloc_h = total_content_h * 0.65
            sym_alloc_w = cw - 2 * padding

            sym_scale = min(sym_alloc_w / symbol_size[0], sym_alloc_h / symbol_size[1])
            sym_w = int(symbol_size[0] * sym_scale)
            sym_h = int(symbol_size[1] * sym_scale)

            # Target text width relative to symbol
            target_text_w = int(sym_w * text_symbol_ratio)
            text_scale = min(1.0, target_text_w / text_size[0])
            # Don't upscale text; only downscale
            text_w = int(text_size[0] * text_scale)
            text_h = int(text_size[1] * text_scale)

            # Center everything horizontally
            sym_x = (cw - sym_w) // 2
            sym_y = padding
            text_x = (cw - text_w) // 2
            text_y = sym_y + sym_h + padding

            result = {
                "symbol": (sym_x, sym_y, sym_w, sym_h),
                "text": (text_x, text_y, text_w, text_h),
            }

            if tagline_size:
                tag_scale = text_scale
                tag_w = int(tagline_size[0] * tag_scale)
                tag_h = int(tagline_size[1] * tag_scale)
                tag_x = (cw - tag_w) // 2
                tag_y = text_y + text_h + padding // 2
                result["tagline"] = (tag_x, tag_y, tag_w, tag_h)

            return result

        elif layout == "horizontal":
            sym_alloc = cw * 0.35
            sym_scale = min(
                sym_alloc / symbol_size[0],
                (ch - 2 * padding) / symbol_size[1],
            )
            sym_w = int(symbol_size[0] * sym_scale)
            sym_h = int(symbol_size[1] * sym_scale)
            sym_x = padding
            sym_y = (ch - sym_h) // 2

            text_max_w = cw - sym_w - 3 * padding
            text_scale = min(1.0, text_max_w / text_size[0])
            text_w = int(text_size[0] * text_scale)
            text_h = int(text_size[1] * text_scale)
            text_x = sym_x + sym_w + padding
            text_y = (ch - text_h) // 2

            result = {
                "symbol": (sym_x, sym_y, sym_w, sym_h),
                "text": (text_x, text_y, text_w, text_h),
            }

            if tagline_size:
                tag_scale = text_scale * 0.6
                tag_w = int(tagline_size[0] * tag_scale)
                tag_h_val = int(tagline_size[1] * tag_scale)
                tag_x = text_x
                tag_y = text_y + text_h + padding // 3
                result["tagline"] = (tag_x, tag_y, tag_w, tag_h_val)

            return result

    def composite(
        self,
        symbol: Image.Image,
        text_img: Image.Image,
        tagline_img: Image.Image = None,
        canvas_size: tuple = (1200, 1200),
        canvas_color=(255, 255, 255, 0),
        layout: str = "vertical",
        padding: int = 40,
    ) -> Image.Image:
        """Composite symbol, text, and optional tagline onto canvas."""
        canvas = Image.new("RGBA", canvas_size, canvas_color)

        positions = self.compute_layout(
            symbol_size=symbol.size,
            text_size=text_img.size,
            tagline_size=tagline_img.size if tagline_img else None,
            canvas_size=canvas_size,
            layout=layout,
            padding=padding,
        )

        # Place symbol
        sx, sy, sw, sh = positions["symbol"]
        sym_resized = symbol.resize((sw, sh), Image.LANCZOS)
        canvas.paste(sym_resized, (sx, sy), sym_resized)

        # Place text
        tx, ty, tw, th = positions["text"]
        text_resized = text_img.resize((tw, th), Image.LANCZOS)
        canvas.paste(text_resized, (tx, ty), text_resized)

        # Place tagline
        if tagline_img and "tagline" in positions:
            tlx, tly, tlw, tlh = positions["tagline"]
            tag_resized = tagline_img.resize((tlw, tlh), Image.LANCZOS)
            canvas.paste(tag_resized, (tlx, tly), tag_resized)

        return canvas

    # ----- Export -----

    def export_multi_format(self, logo: Image.Image, name: str):
        """Export logo in multiple sizes and formats."""
        base = self.output_dir / name

        # Full size PNG (transparent)
        logo.save(base.with_suffix(".png"), "PNG")

        # Full size on white background
        white_bg = Image.new("RGB", logo.size, (255, 255, 255))
        white_bg.paste(logo, mask=logo.split()[3])
        white_bg.save(base.with_name(f"{name}_white_bg.png"), "PNG")

        # Common export sizes
        sizes = {
            "favicon": (64, 64),
            "small": (256, 256),
            "medium": (512, 512),
            "large": (1024, 1024),
        }
        for label, size in sizes.items():
            resized = logo.resize(size, Image.LANCZOS)
            resized.save(base.with_name(f"{name}_{label}.png"), "PNG")

        # ICO for favicon
        ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64)]
        ico_images = [logo.resize(s, Image.LANCZOS) for s in ico_sizes]
        ico_images[0].save(
            base.with_suffix(".ico"), format="ICO",
            sizes=ico_sizes, append_images=ico_images[1:]
        )

        print(f"Exported {name} in {len(sizes) + 3} formats to {self.output_dir}")

    # ----- Full Pipeline -----

    def run(
        self,
        symbol_image: Image.Image,
        brand_name: str,
        tagline: str = None,
        font_name: str = "Montserrat-Bold.ttf",
        tagline_font_name: str = "Montserrat-Regular.ttf",
        font_size: int = 72,
        tagline_font_size: int = 28,
        text_color: str = "#1a1a1a",
        tagline_color: str = "#666666",
        tracking: int = 6,
        layout: str = "vertical",
        canvas_size: tuple = (1200, 1200),
        remove_bg: bool = True,
        export_name: str = "logo",
    ) -> Image.Image:
        """Run the full compositing pipeline."""

        # 1. Process symbol
        symbol = self.process_symbol(symbol_image, remove_bg=remove_bg)

        # 2. Render brand name
        font_path = self.font_dir / font_name
        text_img = self.render_text_line(
            brand_name, font_path, font_size,
            color=text_color, tracking=tracking
        )

        # 3. Render tagline (optional)
        tagline_img = None
        if tagline:
            tagline_font_path = self.font_dir / tagline_font_name
            tagline_img = self.render_text_line(
                tagline, tagline_font_path, tagline_font_size,
                color=tagline_color, tracking=tracking // 2,
                uppercase=False
            )

        # 4. Composite
        logo = self.composite(
            symbol, text_img, tagline_img,
            canvas_size=canvas_size, layout=layout
        )

        # 5. Export
        self.export_multi_format(logo, export_name)

        return logo


# ----- Usage Example -----

if __name__ == "__main__":
    pipeline = LogoTextPipeline(font_dir="./fonts", output_dir="./output")

    # Assuming you already generated a symbol image via AI API
    # symbol = api_client.generate("minimalist mountain icon, no text, white bg")
    symbol = Image.open("generated_symbol.png")

    logo = pipeline.run(
        symbol_image=symbol,
        brand_name="ALPINE",
        tagline="Adventure Awaits",
        font_name="Montserrat-Bold.ttf",
        font_size=80,
        text_color="#1B3A4B",
        tracking=8,
        layout="vertical",
        canvas_size=(1200, 1200),
        export_name="alpine_logo",
    )

    # Also generate horizontal variant
    logo_h = pipeline.run(
        symbol_image=symbol,
        brand_name="ALPINE",
        tagline="Adventure Awaits",
        font_name="Montserrat-Bold.ttf",
        font_size=60,
        text_color="#1B3A4B",
        tracking=6,
        layout="horizontal",
        canvas_size=(1600, 800),
        export_name="alpine_logo_horizontal",
    )
```

### Integration with AI API (Flux via fal.ai Example)

```python
import fal_client
from PIL import Image
from io import BytesIO
import requests

def generate_symbol_flux(concept, style="minimalist"):
    """Generate a text-free symbol using Flux 2 via fal.ai."""
    prompt = (
        f"minimalist {style} logo icon of {concept}, "
        f"simple geometric shapes, clean vector illustration, "
        f"centered on pure white background, "
        f"no text, no letters, no words, no typography, "
        f"professional brand mark, flat design"
    )

    result = fal_client.subscribe(
        "fal-ai/flux-pro/v1.1",
        arguments={
            "prompt": prompt,
            "image_size": {"width": 1024, "height": 1024},
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
            "num_images": 1,
            "safety_tolerance": "5",
        },
    )

    image_url = result["images"][0]["url"]
    response = requests.get(image_url)
    return Image.open(BytesIO(response.content))


# Full workflow
symbol = generate_symbol_flux("a phoenix rising", style="geometric")
pipeline = LogoTextPipeline(font_dir="./fonts")
logo = pipeline.run(
    symbol_image=symbol,
    brand_name="PHOENIX",
    tagline="Rise Above",
    font_size=80,
    text_color="#C0392B",
    export_name="phoenix_logo",
)
```

---

## Summary: Decision Matrix

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| Pure AI (Ideogram 3.0) | Wordmarks, monograms, quick prototypes | Fast, single-step | ~10% error rate, limited font control |
| Pure AI (Recraft V4) | SVG output needed | True vector, scalable | Limited font variety |
| LoRA-enhanced (Harrlogos) | Self-hosted, single words | Free, customizable | SDXL only, short text |
| Hybrid (AI + Pillow) | Production logos, any text length | 100% text accuracy, full font control | Two-step process, needs compositing code |
| Hybrid (AI + Cairo) | High-quality anti-aliased text | Superior rendering | More complex setup |
| Hybrid (AI + SVG) | Scalable vector output | Infinite scalability | Symbol must be SVG (Recraft) |

**Recommended default approach:** Hybrid with Pillow for raster output, hybrid with SVG for vector output. Use pure AI (Ideogram 3.0) only for rapid prototyping or when the text is 1-5 characters.

---

## Dependencies

```
pip install Pillow>=10.0
pip install fonttools>=4.40
pip install rembg>=2.0       # background removal
pip install pycairo>=1.24    # optional, for Cairo rendering
pip install fal-client       # optional, for Flux API
pip install requests
```
