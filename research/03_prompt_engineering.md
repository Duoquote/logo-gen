# Prompt Engineering for AI Logo Generation

## Deep-Dive Research - March 2026

---

## Table of Contents

1. [Prompt Structure Fundamentals](#1-prompt-structure-fundamentals)
2. [Logo-Type-Specific Templates](#2-logo-type-specific-templates)
3. [Model-Specific Prompt Optimization](#3-model-specific-prompt-optimization)
4. [Negative Prompt Strategies](#4-negative-prompt-strategies)
5. [Keyword Database and Style Modifiers](#5-keyword-database-and-style-modifiers)
6. [Describing Colors, Typography, and Composition](#6-describing-colors-typography-and-composition)
7. [Common Mistakes and How to Avoid Them](#7-common-mistakes-and-how-to-avoid-them)
8. [Excellent Prompt Examples](#8-excellent-prompt-examples)

---

## 1. Prompt Structure Fundamentals

### The Universal Logo Prompt Formula

Every effective logo prompt follows a layered structure. The order of elements matters -- models give more weight to tokens that appear earlier in the prompt.

```
[Logo Type] + [Subject/Symbol] + [Style Descriptors] + [Color Specification] + [Typography Notes] + [Composition/Layout] + [Technical Modifiers] + [Background]
```

### Core Principle: Specificity Over Abstraction

The single most important rule is: **be specific, not vague**. Generic prompts produce generic results. Do not say "modern" -- say "minimalist geometric lines with generous whitespace." Do not say "professional" -- say "clean sans-serif typography, muted navy and grey palette, balanced symmetrical composition."

### Prompt Length Guidelines

- **Flux models**: Prefer natural language, 1-3 concise sentences. Avoid keyword soup.
- **SDXL**: Tag-based syntax works well. Comma-separated descriptors with optional weighting.
- **Ideogram**: Full descriptive sentences. Place quoted text early in the prompt.
- **Recraft V4**: Accepts up to 10,000 characters. Follows both short and detailed prompts closely.
- **Midjourney**: Shorter prompts (under 60 words) tend to perform better. Use parameters for control.

### The Three Pillars of a Logo Prompt

1. **What** -- The subject, symbol, or icon (e.g., "a stylized fox head," "interlocking geometric hexagons")
2. **How** -- The style, technique, and aesthetic (e.g., "flat vector illustration, minimalist, clean edges")
3. **Context** -- The intended use and brand feeling (e.g., "for a premium fintech startup, conveying trust and innovation")

---

## 2. Logo-Type-Specific Templates

### 2.1 Minimalist Logo

Minimalist logos depend on restraint. The prompt must explicitly request simplicity, or models will add unwanted detail.

**Template:**
```
Minimalist [icon/symbol] logo, [subject described in geometric terms],
flat vector design, [1-2 colors only], clean precise edges,
simple geometric shapes, ample negative space,
on [white/solid color] background
```

**Example:**
```
Minimalist logo, a single continuous line forming a mountain peak and sun,
flat vector design, deep navy blue on white background, clean precise edges,
simple geometric shapes, ample negative space, no gradients, no shadows
```

**Key modifiers:** `minimalist`, `simple`, `clean lines`, `negative space`, `flat design`, `single line`, `geometric`, `reductive`, `essential forms only`

### 2.2 Mascot Logo

Mascot logos require character description, expression, and stylistic treatment. They are more complex and benefit from specifying the illustration style explicitly.

**Template:**
```
Mascot logo, [character description with expression and pose],
[illustration style] style, [color palette],
bold outlines, [mood/energy level],
suitable for [use case], on [background]
```

**Example:**
```
Mascot logo, a fierce owl with spread wings and determined eyes,
bold comic book vector illustration style, purple and gold color scheme,
thick clean outlines, dynamic and powerful energy,
suitable for an esports team, on white background
```

**Key modifiers:** `mascot`, `character`, `bold outlines`, `expressive`, `cartoon style`, `comic book style`, `thick lines`, `dynamic pose`, `friendly` or `fierce`

### 2.3 Wordmark Logo

Wordmarks live and die by typography. The challenge is that most models still struggle with text rendering, so the prompt must be extremely explicit about the letterforms.

**Template:**
```
Wordmark logo, the text "[EXACT TEXT]" in [font style description],
[letter spacing description], [weight and case],
[color] on [background], clean typography,
[additional style notes]
```

**Example:**
```
Wordmark logo, the text "AURORA" in modern geometric sans-serif typeface,
wide letter spacing, bold uppercase, pure black on white background,
clean crisp typography, minimal and elegant, editorial style
```

**Key modifiers:** `wordmark`, `typographic logo`, `letterforms`, `kerning`, `tracking`, `serif`, `sans-serif`, `script`, `monospace`, `bold`, `light weight`, `uppercase`, `lowercase`

**Important:** Use Ideogram 3.0 or Recraft V4 for wordmarks. Enclose exact text in quotation marks. Spell out individual letters if needed (e.g., "the letters A-U-R-O-R-A").

### 2.4 Emblem Logo

Emblems combine iconography with a containing shape (shield, circle, badge). They require describing the container, the central element, and any surrounding text or decorative elements.

**Template:**
```
Emblem logo, [container shape] containing [central icon/symbol],
[decorative border or frame details], [text placement if any],
[style era -- vintage/modern/classical], [color palette],
detailed but clean, on [background]
```

**Example:**
```
Emblem logo, circular badge with ornate thin border containing
a stylized anchor at center, the text "HARBOR BREWING CO." arched
along the top and "EST. 2024" along the bottom,
vintage craft brewery style, navy blue and cream,
detailed but clean line work, on white background
```

**Key modifiers:** `emblem`, `badge`, `crest`, `shield`, `seal`, `circular frame`, `banner`, `ribbon`, `ornate border`, `contained`, `enclosed`

### 2.5 Abstract Logo

Abstract logos rely on shape, color, and composition to convey meaning without literal representation. Prompts must describe the shapes and their relationships precisely.

**Template:**
```
Abstract logo, [shape description and spatial relationship],
[what the shapes suggest/symbolize], [color treatment],
[style -- geometric/organic/fluid], clean vector design,
balanced composition, on [background]
```

**Example:**
```
Abstract logo, three overlapping translucent circles forming
a triangular intersection at center, suggesting connectivity and unity,
gradient from teal to deep blue to purple, modern geometric style,
clean vector design, balanced symmetrical composition, on white background
```

**Key modifiers:** `abstract`, `non-representational`, `geometric forms`, `organic shapes`, `fluid`, `overlapping`, `interlocking`, `gradient`, `translucent`, `dynamic`, `flowing`

### 2.6 Lettermark / Monogram Logo

**Template:**
```
Lettermark logo, the letter(s) "[LETTERS]" designed as [style description],
[how letters interact -- intertwined/stacked/overlapping],
[font style], [color], modern and distinctive,
on [background]
```

**Example:**
```
Lettermark logo, the letters "JK" intertwined in a single elegant monogram,
modern serif typeface with thin and thick stroke contrast,
gold metallic on dark charcoal background, luxury and sophisticated,
clean vector design
```

### 2.7 Combination Mark

**Template:**
```
Combination mark logo, [icon/symbol description] positioned [left of / above / integrated with]
the text "[BRAND NAME]" in [typography description],
[overall style], [color palette], balanced layout,
on [background]
```

---

## 3. Model-Specific Prompt Optimization

### 3.1 Flux (Black Forest Labs) -- Flux 2 Pro / Dev / Schnell

**Prompt syntax:** Natural language sentences. Avoid tag-based keyword soup.

**Architecture:** Uses dual text encoders -- CLIP (clip_l) and T5-XXL. In practice, the same prompt text goes to both, but the T5 encoder handles natural language understanding while CLIP handles visual-semantic alignment.

**Best practices:**
- Write in descriptive sentences, not comma-separated tags
- Follow the frame: Subject -> Action/State -> Environment -> Style/Modifiers
- Keep prompts concise (1-3 sentences). Overly long prompts degrade quality
- No native negative prompt support in base Flux; use guidance scale to control adherence
- Guidance scale of 3.0-4.0 works well for logos (lower = more creative, higher = more literal)
- For logo LoRAs (e.g., Shakker-Labs FLUX.1-dev-LoRA-Logo-Design), use trigger words: `wablogo, logo, Minimalist`
- LoRA scale of 0.7-0.8 prevents overpowering the base model
- 20-28 inference steps for Dev; 1-4 steps for Schnell

**Example Flux prompt:**
```
A minimalist vector logo for a technology company called "Nexus". The design features
an abstract geometric knot formed from three interlocking triangles in deep blue.
Clean flat design on a pure white background with no shadows or gradients.
```

**ComfyUI nodes:** `CLIPTextEncode` for prompt encoding, `KSampler` with 24 steps, `LoraLoader` for logo LoRAs.

### 3.2 SDXL (Stable Diffusion XL)

**Prompt syntax:** Tag-based, comma-separated keywords. Supports prompt weighting with `(token:weight)` syntax.

**Best practices:**
- Comma-separated descriptors work well: `minimalist logo, fox head, geometric, flat vector, blue and white`
- Use prompt weighting to emphasize key elements: `(minimalist logo:1.3), fox head, (flat vector:1.2), blue and white`
- Keep weights between 0.5 and 1.6 -- going higher produces artifacts
- SDXL has a second text encoder (refiner prompt) -- use it for style refinement
- **Requires negative prompts** (see Section 4) to avoid photorealistic artifacts
- Best resolution: 1024x1024 for square logos
- Use logo-specific LoRAs like Harrlogos XL v2.0 (trigger word: `text logo`)
- Harrlogos LoRA strength: 0.6-0.8 to prevent overpowering
- For text in logos with Harrlogos, weight the text: `("BRAND":1.5) text logo, red, black`
- Harrlogos works best with single words; multi-word text is possible but less reliable

**Example SDXL prompt:**
```
(minimalist logo:1.3), geometric owl icon, (flat vector style:1.2),
clean edges, navy blue and gold, professional, modern,
white background, no texture, centered composition
```

**Example SDXL negative prompt:**
```
photorealistic, 3d render, photograph, shadow, gradient, texture,
blurry, noisy, complex, busy, cluttered, watermark, ugly, messy
```

### 3.3 Ideogram 3.0

**Prompt syntax:** Full descriptive sentences. Ideogram excels at understanding natural language with placement instructions.

**Key advantage:** Near-perfect text rendering (90%+ accuracy). The best choice for logos that contain words.

**Best practices:**
- Enclose exact text in quotation marks: `a sign that says "HELLO"`
- Place quoted text early in the prompt for highest accuracy
- Describe text placement explicitly: "headline at the top," "brand name centered below the icon"
- Pair text with font style descriptions: "bold grotesk typeface, tight tracking"
- Use **Design mode** for graphic output (logos, posters, cards) -- it handles text placement more deliberately
- Use **Style References** (up to 3 reference images) to control aesthetic consistency
- Supports color palette input with hex codes
- Supports negative prompts (see Section 4)
- Specify "logo design" or "brand mark" to activate logo-appropriate aesthetics

**Example Ideogram prompt:**
```
A professional logo design for a coffee brand. A minimalist line drawing of a
steaming coffee cup integrated into the letter "C". The text "CATALYST COFFEE"
appears below in a clean modern sans-serif font, bold weight. Color palette:
warm brown (#6F4E37) and cream (#FFFDD0). White background, flat vector style.
```

**Ideogram Design mode prompt:**
```
Logo design, centered composition. A geometric mountain peak icon above the
wordmark "SUMMIT" in wide-tracked uppercase geometric sans-serif.
Two colors only: deep forest green and white. Minimalist, clean, professional.
```

### 3.4 Recraft V4

**Prompt syntax:** Natural language, flexible length (up to 10,000 characters).

**Key advantage:** Native SVG vector output. The only model producing real editable vector paths, not rasterized images.

**Best practices:**
- Describe brand personality, preferred shapes, and color palette
- Specify RGB/hex values for exact brand colors: "use #1A365D for the primary blue"
- Output opens directly in Illustrator, Figma, or Sketch with editable paths
- Standard tier ($0.08/vector) vs Pro tier ($0.30/vector) for higher quality
- Works on fal.ai, WaveSpeedAI, and native Recraft API
- Style keywords: "flat color," "minimalist," "geometric," "line art"
- Available in ComfyUI via Recraft partner nodes

**Example Recraft prompt:**
```
A logo for a sustainable energy startup. A stylized leaf seamlessly merging
with a lightning bolt, creating a unified symbol. Flat vector design using
only two colors: #2D8B4E (forest green) and #F4A623 (amber). Clean geometric
lines, no gradients, no shadows. White background.
```

### 3.5 Midjourney V8

**Prompt syntax:** Short descriptive phrases + parameters appended at the end.

**Key parameters for logos:**
- `--style raw` -- Disables Midjourney's aesthetic auto-pilot. Produces cleaner, more literal results. Essential for logos.
- `--stylize [0-1000]` (or `--s`) -- Controls artistic interpretation. Default 100. For logos, use 50-250 (lower = more faithful to prompt).
- `--ar 1:1` -- Square aspect ratio, standard for logo marks
- `--no [element]` -- Exclude unwanted elements (functions as negative prompt)
- `--chaos [0-100]` -- Variation between generated images. Use 0-20 for consistent logo batches.
- `--q 2` -- Higher quality rendering

**Best practices:**
- Keep prompts under 60 words
- Always use `--style raw` for logo work
- Use `--no` for exclusions: `--no text, photograph, realistic, gradient`
- Midjourney has no direct API -- Discord-based only, limiting automation
- Best for initial creative exploration and concept ideation, not production pipelines

**Example Midjourney prompt:**
```
flat vector logo, geometric fox head, minimalist, clean lines,
orange and dark grey, white background, professional brand mark
--style raw --s 150 --ar 1:1 --no text photograph gradient shadow
```

### 3.6 GPT Image 1.5 (OpenAI / ChatGPT)

**Prompt syntax:** Conversational natural language. Supports multi-turn refinement.

**Best practices:**
- Conversational iteration is the key strength: generate, then refine through dialogue
- Supports image input for reference-based generation
- Specify "vector style" or "flat illustration" to avoid photorealistic defaults
- Explicitly state "no background" or "white background" -- the model defaults to scenes
- Useful for rapid ideation before moving to specialized models

**Example ChatGPT prompt:**
```
Create a logo for a fitness app called "PulseTrack." The logo should be a
minimalist heart shape formed by two curved lines that suggest a pulse/ECG wave.
Use a single color: vibrant coral (#FF6B6B). Flat vector style, clean lines,
no shadows, no gradients, white background.
```

### Model Selection Quick Reference

| Scenario | Best Model | Reason |
|----------|-----------|--------|
| Logo with text/wordmark | Ideogram 3.0 | 90%+ text accuracy |
| Production-ready SVG | Recraft V4 | Native vector output |
| Creative exploration | Midjourney V8 | Best aesthetics |
| Automated pipeline | Flux 2 Pro/Dev | API + open weights |
| Maximum LoRA control | SDXL + LoRAs | Largest ecosystem |
| Conversational iteration | GPT Image 1.5 | Multi-turn dialogue |
| Budget-conscious | Flux 2 Schnell | Free, fast, decent |

---

## 4. Negative Prompt Strategies

### Why Negative Prompts Matter for Logos

Logos require clean, simple, scalable output. Without negative prompts, AI models default toward photorealism, complex textures, and excessive detail -- the three failures that make an AI image look impressive but function terribly as a brand mark.

### Universal Negative Prompt for Logos (SDXL / Stable Diffusion)

```
photorealistic, photograph, 3d render, 3d, CGI, realistic,
shadow, drop shadow, gradient, texture, noise, grain, film grain,
blurry, bokeh, depth of field, out of focus,
complex, busy, cluttered, detailed background,
watermark, signature, text (if unwanted), letters (if unwanted),
ugly, messy, dirty, distorted, deformed,
multiple logos, duplicated elements, border, frame (if unwanted)
```

### Category-Specific Negative Prompts

**For Minimalist Logos:**
```
ornate, decorative, complex, detailed, intricate, busy, cluttered,
3d, shadow, gradient, texture, pattern, realistic, photograph,
multiple colors, rainbow, neon, glossy, shiny, metallic
```

**For Mascot Logos:**
```
photorealistic, uncanny valley, blurry, low quality, deformed face,
extra limbs, mutated, grotesque, scary (unless intended),
no outline, thin lines, watercolor, sketch, rough
```

**For Wordmark/Typography Logos:**
```
blurry text, distorted letters, misspelled, illegible, overlapping letters,
decorative illustration, icon, mascot, complex background,
handwriting (unless intended), serif (if you want sans-serif)
```

**For Emblem Logos:**
```
asymmetrical (if unwanted), broken border, incomplete frame,
photorealistic, blurry, low resolution, noisy, grainy,
modern minimalist (if you want vintage), flat (if you want dimensional)
```

### Model-Specific Negative Prompt Support

| Model | Native Negative Prompt | Workaround |
|-------|----------------------|------------|
| SDXL | Yes -- dedicated negative prompt field | N/A |
| Flux | No native support | Use CFG guidance scale; use LoRA conditioning |
| Ideogram 3.0 | Yes -- "Negative prompt" field in settings | N/A |
| Midjourney | Partial -- `--no` parameter | `--no text, gradient, shadow` |
| Recraft V4 | Limited | Rely on positive prompt specificity |
| GPT Image 1.5 | No | Use "do not include" in prompt |

### Negative Prompt Best Practices

1. **Mirror your positive prompt**: If you ask for "flat vector," negate "3d, realistic, textured"
2. **Start broad, then refine**: Begin with the universal negative list, then add specific exclusions based on output
3. **Do not contradict yourself**: Avoid negating something your positive prompt implies
4. **Weight critical negatives** (SDXL): `(photorealistic:1.5), (blurry:1.3), shadow, gradient`
5. **Less is more**: An overly long negative prompt can confuse the model. Focus on the top 10-15 most impactful terms
6. **For models without negative prompts**: Embed exclusions in the positive prompt -- "no shadows, no gradients, no text, flat design only"

---

## 5. Keyword Database and Style Modifiers

### Logo Type Keywords

| Category | Keywords |
|----------|----------|
| **Minimalist** | minimalist, simple, clean, reductive, essential, stripped-down, uncluttered, sparse, refined, pared-back |
| **Geometric** | geometric, angular, polygonal, hexagonal, triangular, circular, symmetrical, mathematical, structured, grid-based |
| **Organic** | organic, natural, flowing, fluid, curved, biomorphic, hand-drawn, freeform, asymmetric |
| **Vintage/Retro** | vintage, retro, classic, heritage, aged, distressed, nostalgic, 1950s, art deco, letterpress |
| **Futuristic** | futuristic, sci-fi, cyber, neon, holographic, tech, digital, high-tech, sleek, sharp |
| **Luxury** | luxury, premium, elegant, sophisticated, refined, exclusive, high-end, opulent, gold foil |
| **Playful** | playful, fun, whimsical, cheerful, friendly, bouncy, rounded, bubbly, cartoon |
| **Corporate** | corporate, professional, trustworthy, authoritative, stable, balanced, conservative, institutional |

### Style Modifier Keywords

| Category | Keywords |
|----------|----------|
| **Rendering** | flat vector, line art, solid fill, gradient, duotone, monochrome, outlined, silhouette |
| **Technique** | vector illustration, digital art, hand-lettered, engraved, stencil, stamp, woodcut, linocut |
| **Edges** | clean edges, crisp lines, sharp, smooth, soft, rounded corners, hard angles |
| **Dimension** | flat, 2D, isometric, 3D, embossed, debossed, layered, stacked |
| **Texture** | smooth, textured, grungy, distressed, paper texture, metallic, glossy, matte |
| **Weight** | bold, thick lines, thin lines, light, heavy, delicate, chunky, ultra-thin |

### Quality and Output Modifiers

These terms push the model toward higher-quality, more usable output:

```
high resolution, 4K, crisp, sharp, professional quality,
centered composition, isolated on white background,
scalable, vector art, clean design, brand mark,
award-winning logo design, behance, dribbble
```

**Caution:** "behance" and "dribbble" can bias output toward trendy aesthetics. Use sparingly and only when that style is desired.

### Industry-Specific Modifier Sets

| Industry | Suggested Modifiers |
|----------|-------------------|
| **Technology** | circuit, digital, pixel, node, network, binary, gradient blue, geometric |
| **Healthcare** | cross, heart, pulse, leaf, shield, clean, blue and green, trustworthy |
| **Food & Beverage** | organic, fresh, handcrafted, warm colors, rustic, artisanal, natural |
| **Finance** | shield, column, graph, ascending, stable, navy, gold, serif typography |
| **Fashion** | elegant, haute, monogram, editorial, high contrast, black and white |
| **Sports/Gaming** | dynamic, fierce, bold, shield, flame, claw, aggressive angles, neon |
| **Education** | book, owl, lamp, academic, classical, balanced, blue, green |
| **Environment** | leaf, tree, water, earth tones, green, organic shapes, circular |

### Emotional/Mood Modifiers

| Mood | Keywords |
|------|----------|
| **Trustworthy** | stable, balanced, grounded, symmetrical, blue tones, shield |
| **Innovative** | dynamic, forward-leaning, angular, gradient, purple and blue |
| **Friendly** | rounded, warm colors, smile, open, approachable, soft edges |
| **Powerful** | bold, angular, dark palette, high contrast, sharp, strong |
| **Calm** | soft, muted, pastel, flowing, circular, natural tones |
| **Energetic** | bright, vibrant, diagonal, dynamic, warm colors, flame |

---

## 6. Describing Colors, Typography, and Composition

### 6.1 Color Specification

#### Methods (Most to Least Precise)

1. **Hex codes** (where supported): `#1A365D`, `#F4A623` -- Recraft V4 and Ideogram support this
2. **Named color + qualifier**: `deep navy blue`, `warm amber orange`, `muted sage green`
3. **Color system reference**: `Pantone 2768 C blue` (models approximate this)
4. **Relative description**: `the blue of a clear winter sky`, `burgundy like aged wine`

#### Color Palette Patterns for Logos

| Pattern | Description | Example |
|---------|-------------|---------|
| **Monochrome** | Single hue with variations | `monochrome navy, ranging from light to dark` |
| **Duotone** | Two contrasting colors | `deep blue and bright orange duotone` |
| **Triadic** | Three evenly spaced colors | `red, yellow, blue triadic palette` |
| **Analogous** | Adjacent colors | `teal, blue, and purple analogous palette` |
| **Neutral + Accent** | Grayscale with one pop color | `charcoal grey with a single coral accent` |

#### Color Do's and Don'ts

- **Do:** Specify exact number of colors: "two-color design using only black and teal"
- **Do:** Name the dominant color: "primarily deep blue with gold accents"
- **Do:** Specify color relationships: "dark icon on light background"
- **Don't:** Say "colorful" without specifics -- you will get rainbow chaos
- **Don't:** Assume "blue" is enough -- there are hundreds of blues. Say "deep navy" or "electric cyan"
- **Don't:** Forget background color -- always specify it explicitly

### 6.2 Typography Description

AI models cannot use specific font names (they do not have font libraries). Instead, describe the characteristics of the typography you want.

#### Typography Descriptor Framework

```
[Weight] + [Style] + [Classification] + [Spacing] + [Case] + [Personality]
```

| Attribute | Options |
|-----------|---------|
| **Weight** | ultra-light, thin, light, regular, medium, semi-bold, bold, extra-bold, black |
| **Style** | upright, italic, oblique, condensed, extended, rounded |
| **Classification** | serif, sans-serif, slab serif, geometric sans, humanist sans, script, display, monospace |
| **Spacing** | tight tracking, normal tracking, wide tracking/letter-spacing, compressed, expanded |
| **Case** | uppercase, lowercase, title case, small caps, mixed case |
| **Personality** | elegant, sturdy, playful, technical, classical, modern, editorial |

#### Typography Examples in Prompts

- Luxury brand: `"elegant thin serif typeface, wide letter spacing, uppercase, refined and exclusive"`
- Tech startup: `"bold geometric sans-serif, tight tracking, lowercase, modern and clean"`
- Children's brand: `"rounded bubbly sans-serif, medium weight, playful and friendly"`
- Law firm: `"classical serif, medium weight, small caps, authoritative and traditional"`
- Creative agency: `"hand-lettered script, flowing and organic, irregular baseline, artistic"`

### 6.3 Composition and Layout

#### Spatial Arrangement Keywords

| Layout | Keywords |
|--------|----------|
| **Centered** | centered, symmetrical, balanced, middle-aligned |
| **Stacked** | vertically stacked, icon above text, top-to-bottom |
| **Horizontal** | side by side, icon left of text, horizontal layout |
| **Contained** | enclosed in circle, within a shield, inside a badge |
| **Integrated** | icon merged with text, letter incorporated into symbol |
| **Isolated** | standalone icon, no text, mark only |

#### Aspect Ratio Guidance

| Use Case | Recommended | Notes |
|----------|------------|-------|
| App icon | 1:1 square | `centered, square composition` |
| Social media avatar | 1:1 square | `fits within circle crop` |
| Website header | 3:1 or 4:1 horizontal | `wide horizontal layout` |
| Favicon | 1:1 square, simple | `extremely simplified, recognizable at 16x16` |
| Business card | Flexible | `balanced, works at small scale` |
| Billboard | Flexible | `bold, readable at distance` |

#### Composition Modifiers

```
centered composition, balanced layout, rule of thirds,
golden ratio proportions, symmetrical, asymmetrical but balanced,
generous whitespace, tight crop, isolated subject,
breathing room around icon, compact arrangement
```

---

## 7. Common Mistakes and How to Avoid Them

### Mistake 1: Vague, Underspecified Prompts

**Bad:** `A cool logo for my tech company`

**Good:** `Minimalist geometric logo for a cloud computing company. An abstract icon of overlapping hexagons suggesting network nodes. Flat vector, two colors only: deep blue (#0A1628) and electric cyan (#00D4FF). Clean edges, no gradients, on white background.`

**Why:** "Cool" and "tech" mean nothing to a model. Every detail you omit is a random roll.

### Mistake 2: Overloading with Conflicting Instructions

**Bad:** `A vintage yet futuristic minimalist but detailed ornate simple geometric organic flowing angular logo with gold silver bronze copper metallic matte finish`

**Good:** Pick ONE primary style and stay consistent. Contradictions confuse the model and produce incoherent output.

**Rule of thumb:** Focus on 2-3 key descriptors. If two descriptors conflict, choose one.

### Mistake 3: Expecting Perfect Text Rendering

Most models struggle with text in images. Common failures:
- Misspelled words
- Extra or missing letters
- Garbled or illegible text
- Wrong font style despite description

**Solutions:**
- Use Ideogram 3.0 for text-heavy logos (90%+ accuracy)
- Enclose exact text in quotation marks
- Spell out letters individually for critical text: `"the letters N-E-X-U-S"`
- Generate the icon/symbol with AI, then add text manually in Figma/Illustrator
- Use Harrlogos LoRA with SDXL for text (single words work best)

### Mistake 4: Ignoring the Background

**Bad:** (No background specification)

The model will generate a random scene, gradient, or textured background.

**Good:** Always explicitly specify: `on white background`, `on solid black background`, `on transparent background`, `isolated, no background elements`

### Mistake 5: Not Using Negative Prompts (When Available)

Without negatives, SDXL and Ideogram will gravitate toward photorealistic, textured, shadowed output. Always pair your positive prompt with relevant negatives (see Section 4).

### Mistake 6: Treating AI Generation as One-Shot

**Wrong approach:** Write one prompt, accept the first result.

**Right approach:** Treat it as a dialogue:
1. Start with a clear base prompt
2. Generate 4-8 variations
3. Identify what works and what does not
4. Refine the prompt based on observations
5. Iterate 3-5 rounds minimum
6. Use seed locking to preserve good compositions while adjusting details

### Mistake 7: Forgetting Scalability Keywords

A logo must work at 16px (favicon) and 16 feet (billboard). If you don't specify scalability, the model may produce detail that disappears at small sizes.

**Add:** `scalable, works at small sizes, simple enough for favicon, clean at any resolution, vector art`

### Mistake 8: Using Model-Wrong Prompt Syntax

- Writing SDXL-style tags for Flux (natural language is better)
- Writing long paragraphs for Midjourney (short phrases + parameters work better)
- Not using quotation marks for text in Ideogram
- Not using `--style raw` in Midjourney for logos

### Mistake 9: Ignoring Aspect Ratio

Generating a square logo in 16:9 wastes context. Generating a horizontal wordmark in 1:1 wastes space.

Match aspect ratio to logo type:
- Icon/mark: 1:1
- Wordmark: 3:1 or wider
- Combination mark: 2:1 or 3:2
- Emblem: 1:1

### Mistake 10: Not Specifying the Number of Colors

Without a color count, models use as many as they want. Professionally, most logos use 1-3 colors.

**Always state:** `two colors only`, `monochrome`, `three-color palette limited to navy, gold, and white`

---

## 8. Excellent Prompt Examples

### Example 1: Minimalist Tech Startup (Flux 2 Dev)

```
A minimalist vector logo for a cloud computing startup called "Nimbus."
The icon is an abstract, geometric cloud shape composed of three overlapping
rounded rectangles, suggesting layers of infrastructure. Single color: deep
indigo (#312E81). No text, icon only. Flat design, no shadows, no gradients,
clean precise edges on a pure white background.
```

**Why it works:** Clear subject (geometric cloud), specific color (hex code), explicit style constraints (flat, no shadows/gradients), defined scope (icon only), and background specification.

### Example 2: Luxury Fashion Wordmark (Ideogram 3.0)

```
A luxury fashion brand wordmark logo. The text "MAISON NOIR" in an elegant
thin serif typeface with extremely wide letter spacing. All uppercase, light
font weight. Pure black (#000000) on white background. Minimalist, editorial
style with no icon or decoration. Clean, sophisticated, high-end fashion
aesthetic. Centered composition.
```

**Why it works:** Leverages Ideogram's text rendering. Quoted exact text. Detailed typography description (thin, serif, wide spacing, uppercase, light weight). Strong style anchoring (editorial, luxury, high-end).

### Example 3: Esports Mascot (Midjourney V8)

```
fierce dragon mascot logo for esports team, stylized dragon head with
sharp angular features, glowing red eyes, bold thick outlines,
flat vector illustration, red and black color scheme, dynamic and aggressive,
white background, professional esports branding
--style raw --s 100 --ar 1:1 --no photograph realistic gradient text 3d
```

**Why it works:** Strong character description, explicit style (flat vector, bold outlines), emotional modifiers (fierce, aggressive, dynamic), Midjourney-specific parameters, and comprehensive `--no` exclusions.

### Example 4: Craft Brewery Emblem (SDXL + Negative Prompt)

**Positive prompt:**
```
(emblem logo:1.3), circular badge design, craft brewery, stylized hop flower
at center, (ornate vintage border:1.1), the text "IRONCLAD BREWING" arched
along top, "EST. 2023" along bottom, (vintage woodcut style:1.2),
cream and dark brown color scheme, detailed line work, white background
```

**Negative prompt:**
```
photorealistic, photograph, 3d, modern, minimalist, gradient, blurry,
noisy, low quality, deformed text, misspelled, neon, bright colors,
complex background, watermark
```

**Why it works:** Weighted key terms appropriately, detailed compositional structure (arched text, center icon, border), strong style commitment (vintage woodcut), and comprehensive negative prompt that steers away from defaults.

### Example 5: Abstract SaaS Logo (Recraft V4 Vector)

```
An abstract logo for a data analytics SaaS platform. Three ascending bars
of increasing height, integrated with a subtle upward arrow in the negative
space between them. Colors: #6366F1 (indigo) and #A5B4FC (light periwinkle).
Flat vector design, geometric precision, no rounded corners, clean sharp
edges. Minimalist, modern, professional. White background.
```

**Why it works:** Describes the exact geometric construction. Uses hex colors for Recraft's precise color matching. Specifies edge treatment (sharp, no rounding). Output will be native SVG with editable paths.

### Example 6: Children's Education App (GPT Image 1.5)

```
Create a friendly, playful logo for a children's reading app called "BookBuddy."
The icon is a smiling cartoon book character with small arms and legs, as if the
book is waving. Rounded friendly shapes, thick outlines. Color palette: bright
primary blue, sunshine yellow, and soft white. The text "BookBuddy" appears below
in a rounded, bubbly sans-serif font. Flat vector illustration style, clean lines,
white background.
```

**Why it works:** Conversational tone suits GPT. Detailed character description with personality (smiling, waving). Named color descriptions match the playful mood. Explicit typography description.

### Example 7: Environmental Nonprofit (Ideogram 3.0 Design Mode)

```
Logo design for an ocean conservation nonprofit. A circular mark containing
a minimalist whale tail rising from stylized waves. The text "DEEP BLUE
FOUNDATION" is set below in a clean sans-serif font, medium weight, wide
tracking, uppercase. Two colors only: ocean blue (#1E40AF) and white.
Flat vector style, clean and professional, centered composition, white background.
```

**Why it works:** Uses Ideogram Design mode for deliberate graphic layout. Clear visual hierarchy (icon in circle, text below). Limited color palette. Composition explicitly described.

### Example 8: Iterative Refinement Demonstration

**Round 1 (too vague):**
```
A logo for a coffee shop
```
*Result: Cluttered, photorealistic coffee cup with random text, complex background*

**Round 2 (better structure):**
```
Minimalist logo for a specialty coffee roaster, a stylized coffee bean icon,
flat vector, brown and cream, white background
```
*Result: Cleaner, but still has unwanted shadows and too many details*

**Round 3 (precise and constrained):**
```
Minimalist logo, single stylized coffee bean formed by two curved leaf-like
shapes meeting at center, creating an "S" curve in the negative space.
Flat vector design, two colors only: dark roast brown (#3E2723) and cream
(#FFF8E1). No shadows, no gradients, no texture, clean precise edges.
Isolated on pure white background. Simple enough to work as a favicon.
```
*Result: Clean, professional, usable logo mark*

**Lesson:** Each iteration adds specificity. The final version describes the exact geometric construction, specifies colors by hex, explicitly excludes unwanted elements, and considers practical usage (favicon).

---

## Appendix A: Quick-Reference Prompt Templates

### Universal Starter Template
```
[Logo type] logo for [industry/brand context].
[Icon/symbol description with geometric specifics].
[Style: flat vector / line art / vintage / etc.].
[Color palette: specific colors, limited count].
[Typography description if text is included].
[Composition: centered / stacked / horizontal].
No [list unwanted elements]. On [background color] background.
```

### Speed Template (Rapid Ideation)
```
[style] logo, [subject], [2-3 key modifiers], [colors], white background
```

### Production Template (Final Output)
```
Professional [logo type] logo design for [brand name], a [industry] company.
The mark features [detailed icon description with shapes, spatial relationships,
and what they symbolize]. Style: [rendering technique], [aesthetic modifiers].
Color palette limited to [N] colors: [color 1 with hex], [color 2 with hex].
Typography: [weight] [classification] [spacing] [case] for the brand name.
Composition: [layout description]. Technical: clean edges, scalable,
works at all sizes from favicon to billboard. On pure white background.
No shadows, no gradients, no photorealistic elements, no decorative borders.
```

---

## Appendix B: Prompt Engineering Checklist

Before submitting any logo generation prompt, verify:

- [ ] Logo type is explicitly stated (minimalist, mascot, wordmark, emblem, abstract, lettermark, combination)
- [ ] Subject/symbol is described in concrete geometric terms, not abstract concepts
- [ ] Style is specified with 2-3 consistent, non-contradictory modifiers
- [ ] Color palette is named with specific shades or hex codes, with a stated maximum count
- [ ] Typography is described by characteristics (weight, style, classification, spacing, case), not font names
- [ ] Composition and layout are specified (centered, stacked, horizontal, contained)
- [ ] Background is explicitly stated (white, black, transparent, solid color)
- [ ] Unwanted elements are excluded (via negative prompt or explicit "no X" statements)
- [ ] Prompt syntax matches the target model (natural language for Flux/Ideogram, tags for SDXL, short + params for Midjourney)
- [ ] Scalability is considered ("works at small sizes," "simple enough for favicon")
- [ ] Aspect ratio matches the logo format
- [ ] Text (if any) is enclosed in quotation marks and placed early in the prompt

---

## Appendix C: Sources

- [Superside - 20 Best AI Prompts for Logo Design in 2026](https://www.superside.com/blog/ai-prompts-logo-design)
- [DesignRush - AI Logos: What You Need to Get Right in 2026](https://www.designrush.com/best-designs/logo/trends/logo-design-prompts)
- [SocialSight - Ultimate Guide to AI Logo Generation Prompts](https://socialsight.ai/prompt-guides/ultimate-guide-ai-logo-generation)
- [Nebius - Creating Images with Flux: Prompt Guide](https://nebius.com/blog/posts/creating-images-with-flux-prompt-guide)
- [Skywork - Flux Prompting Ultimate Guide](https://skywork.ai/blog/flux-prompting-ultimate-guide-flux1-dev-schnell/)
- [Civitai - Harrlogos XL v2.0 LoRA](https://civitai.com/models/176555/harrlogos-xl-finally-custom-text-generation-in-sd)
- [Ideogram Docs - Text and Typography](https://docs.ideogram.ai/using-ideogram/prompting-guide/2-prompting-fundamentals/text-and-typography)
- [Ideogram Docs - Prompt Structure](https://docs.ideogram.ai/using-ideogram/prompting-guide/3-prompt-structure)
- [Midjourney Docs - Stylize](https://docs.midjourney.com/hc/en-us/articles/32196176868109-Stylize)
- [Midjourney Docs - Raw Mode](https://docs.midjourney.com/docs/style)
- [fal.ai - Recraft V4 Text-to-Vector](https://fal.ai/models/fal-ai/recraft/v4/text-to-vector)
- [Replicate - Recraft V4: Image Generation with Design Taste](https://replicate.com/blog/recraft-v4)
- [LogoCrafter - 43 AI Logo Examples with Prompts](https://www.logocrafter.app/blog/ai-logos-with-prompts)
- [LogoDiffusion - Common Prompt Mistakes and How Feedback Fixes Them](https://logodiffusion.com/blog/common-prompt-mistakes-and-how-feedback-fixes-them)
- [GodOfPrompt - 10 AI Image Generation Mistakes](https://www.godofprompt.ai/blog/10-ai-image-generation-mistakes-99percent-of-people-make-and-how-to-fix-them)
- [Promptomania - Best Logo Design Prompts](https://promptomania.com/prompts/logo-prompts)
- [Kalon - AI Logo Prompts: Brand Marks & Icon Design](https://www.kalon.ai/templates/ai-logo-prompts)
- [Free AI Prompt Maker - Stable Diffusion Negative Prompts Guide](https://freeaipromptmaker.com/blog/2025-11-29-stable-diffusion-negative-prompts-guide)
- [Aituts - 50+ Midjourney Logo Design Prompts](https://aituts.com/how-to-create-actual-ai-generated-logos/)
- [LogoAI - How to Use Ideogram AI for Logos](https://www.logoai.com/design/blog/ideogram-ai-generate-logos-posters-typography)
- [eBaq Design - How to Use Ideogram for Logo Design in 2026](https://www.ebaqdesign.com/blog/ideogram-logo-design)
