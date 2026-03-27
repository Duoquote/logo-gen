# LLM-Based Prompt Enhancement System for Logo Generation

## Research Notes - March 2026

---

## 1. System Prompt Design for Logo Prompt Enhancement

### Core Concept

An LLM acts as an intermediary between a user's simple brand description (e.g., "a modern tech startup called Nexus") and the detailed, structured prompt that image generation models need to produce high-quality logos. The LLM's system prompt must encode deep knowledge of logo design principles, image generation model syntax, and brand strategy.

### System Prompt Architecture

The system prompt should encode three domains of knowledge:

1. **Brand/Design Knowledge** - Logo styles, typography categories, color theory, composition rules, industry conventions
2. **Image Generation Syntax** - What phrasing produces the best results for each target model (Flux, Recraft, Ideogram, SDXL, etc.)
3. **Output Formatting** - Structured output so the application can parse and use the enhanced prompt programmatically

### Reference System Prompt

```text
You are an expert logo design prompt engineer. Your role is to transform simple
brand descriptions into detailed, effective prompts for AI image generation models.

You have deep expertise in:
- Logo design principles: simplicity, memorability, scalability, versatility
- Typography: serif, sans-serif, slab, display, handwritten, monospace categories
- Color theory: complementary, analogous, triadic schemes; color psychology by industry
- Composition: symmetry, golden ratio, negative space, geometric construction
- Industry conventions: tech=geometric/minimal, food=warm/organic, luxury=thin serifs/gold

When generating prompts, always include:
1. Subject: The core visual element (icon, lettermark, emblem, mascot, abstract mark)
2. Style: The aesthetic approach (minimalist, vintage, geometric, hand-drawn, 3D, flat)
3. Typography: Font style and treatment if text is included
4. Color: Specific palette or scheme with rationale
5. Composition: Layout, spacing, and arrangement details
6. Technical: Background, resolution, format specifications
7. Quality modifiers: "professional", "clean", "vector-style", "high contrast"

Output format: Return a JSON object with fields:
- "prompt": The full positive prompt string
- "negative_prompt": Things to exclude
- "style_tags": Array of style keywords
- "rationale": Brief explanation of design choices
```

### Prompt Formula (Three-Layer Approach)

Structure prompts using three layers as recommended by branding prompt engineering guides:

1. **Foundation Layer** - Brand identity: industry, values, target audience, personality
2. **Design Layer** - Style, composition, color, typography, iconography
3. **Technical Layer** - Model-specific syntax, quality boosters, format specs

---

## 2. Multi-Stage Enhancement Pipeline

### Architecture: Brand Analysis -> Design Direction -> Prompt Generation

A single LLM call often produces generic results. A multi-stage pipeline where each stage has a focused role produces significantly better prompts.

#### Stage 1: Brand Analysis

Input: Raw user description
Output: Structured brand profile

```text
System: You are a brand strategist. Analyze the following brand description and
extract a structured profile.

Output JSON with:
- brand_name: string
- industry: string
- target_audience: string
- brand_personality: array of 3-5 adjectives
- competitors: likely competitor brands
- key_differentiators: what makes this brand unique
- emotional_tone: the feeling the brand should evoke
- color_associations: colors commonly associated with this industry/personality
```

#### Stage 2: Design Direction

Input: Brand profile from Stage 1
Output: Concrete design specifications

```text
System: You are a senior logo designer. Given a brand profile, propose concrete
design directions.

Output JSON with:
- logo_type: "wordmark" | "lettermark" | "icon" | "combination" | "emblem" | "mascot" | "abstract"
- style: "minimalist" | "geometric" | "vintage" | "hand-drawn" | "3d" | "gradient" | "flat"
- primary_colors: array of hex codes with rationale
- secondary_colors: array of hex codes
- typography_style: description of font characteristics
- icon_concept: description of the visual symbol if applicable
- composition_notes: layout and spatial arrangement
- references: similar real-world logos for inspiration context
```

#### Stage 3: Prompt Generation

Input: Design direction from Stage 2 + target model info
Output: Model-optimized image generation prompt

```text
System: You are a prompt engineer specializing in AI image generation for logos.
Convert design specifications into optimized prompts for the target model.

Model-specific guidance:
- Flux: Prefers natural language descriptions, responds well to "vector logo",
  "flat design", "on white background", "simple and clean"
- SDXL: Benefits from comma-separated tags, weighted tokens with (parentheses:1.2),
  explicit negative prompts
- Recraft V4: Supports style presets, color palette hex codes, vector output mode
- Ideogram: Wrap desired text in quotation marks, use --style descriptors

Output JSON with:
- prompt: string (the positive prompt)
- negative_prompt: string
- model_params: dict of recommended model parameters (cfg_scale, steps, etc.)
- variations: array of 3-5 prompt variations exploring different angles
```

### Pipeline Benefits

- Each stage can use a different model (e.g., Claude for analysis, GPT-4o for design direction)
- Intermediate results are inspectable and editable by the user
- Stages can be cached and reused (same brand analysis, different design directions)
- Error isolation: if design direction is wrong, re-run Stage 2 without re-analyzing the brand

---

## 3. Using Claude/GPT via OpenRouter API

### Overview

OpenRouter (https://openrouter.ai/api/v1) provides a unified API compatible with the OpenAI SDK that routes to 200+ models from Anthropic, OpenAI, Google, Meta, Mistral, and others. This is ideal for a prompt enhancement system because:

- Switch between models without code changes
- Automatic fallback if a model is unavailable
- Single API key, unified billing
- Consistent request/response format across all providers

### Setup with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="sk-or-v1-xxxxxxxxxxxx",  # OpenRouter API key
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",  # or "openai/gpt-4o", etc.
    extra_headers={
        "HTTP-Referer": "https://your-app.com",  # Optional, for leaderboard
        "X-Title": "LogoGen",                      # Optional, app name
    },
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "A modern fintech startup called PayFlow"},
    ],
    temperature=0.7,
    max_tokens=2000,
)

enhanced_prompt = response.choices[0].message.content
```

### Setup with httpx (Direct HTTP)

```python
import httpx
import json

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
API_KEY = "sk-or-v1-xxxxxxxxxxxx"

async def enhance_prompt(description: str, model: str = "anthropic/claude-sonnet-4") -> dict:
    """Enhance a brand description into a logo generation prompt."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-app.com",
                "X-Title": "LogoGen",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": description},
                ],
                "temperature": 0.7,
                "max_tokens": 2000,
                "response_format": {"type": "json_object"},
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
```

### Recommended Models for Prompt Enhancement

| Model | OpenRouter ID | Strengths | Cost (per 1M tokens) |
|-------|--------------|-----------|---------------------|
| Claude Sonnet 4 | `anthropic/claude-sonnet-4` | Structured output, design reasoning | ~$3 in / $15 out |
| GPT-4o | `openai/gpt-4o` | Fast, good at creative writing | ~$2.50 in / $10 out |
| Claude Haiku 3.5 | `anthropic/claude-3.5-haiku` | Cheapest for simple enhancement | ~$0.80 in / $4 out |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct` | Free/cheap tier, good quality | ~$0.10-0.50 |
| DeepSeek V3 | `deepseek/deepseek-chat` | Very cheap, strong reasoning | ~$0.14 in / $0.28 out |

### Structured Output (JSON Mode)

For reliable parsing, use `response_format: {"type": "json_object"}` or use the `instructor` library:

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI

class LogoPrompt(BaseModel):
    prompt: str
    negative_prompt: str
    style_tags: list[str]
    color_palette: list[str]
    rationale: str

client = instructor.from_openai(
    OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-v1-xxxxxxxxxxxx",
    )
)

result = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    response_model=LogoPrompt,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "A craft brewery called Ironwood Ales"},
    ],
)
# result is a validated LogoPrompt instance
print(result.prompt)
```

### Model Fallback Pattern

```python
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4",
    extra_body={
        "models": [
            "anthropic/claude-sonnet-4",
            "openai/gpt-4o",
            "meta-llama/llama-3.3-70b-instruct",
        ],
        "route": "fallback",
    },
    messages=messages,
)
```

---

## 4. Chain-of-Thought Prompting for Logo Prompts

### Why CoT Matters for Logo Generation

Chain-of-thought prompting forces the LLM to reason through design decisions step by step rather than jumping to a generic prompt. This produces prompts grounded in actual design logic rather than pattern-matched cliches.

### Implementation

Embed reasoning steps directly in the system prompt:

```text
System: You are a logo design prompt engineer. When given a brand description,
think through the following steps before generating the final prompt:

Step 1 - BRAND IDENTITY: What industry is this? Who is the target customer?
What emotions should the logo evoke? What are the brand's core values?

Step 2 - COMPETITIVE LANDSCAPE: What do logos in this industry typically look
like? How can this logo differentiate while remaining appropriate?

Step 3 - DESIGN PRINCIPLES: Based on the brand identity, select:
  - Logo type (wordmark, icon, combination, emblem, abstract)
  - Visual style (minimalist, geometric, vintage, playful, corporate)
  - Color psychology (what colors align with the brand emotions?)
  - Typography mood (strong, elegant, friendly, technical)

Step 4 - VISUAL CONCEPT: Describe the specific visual elements. What shapes,
symbols, or metaphors communicate the brand message?

Step 5 - PROMPT SYNTHESIS: Combine all decisions into an optimized prompt for
the target image generation model.

Show your reasoning for each step, then provide the final prompt.
```

### Zero-Shot CoT (Simple Version)

For lighter-weight enhancement, append a reasoning trigger:

```text
User: Create a logo prompt for "a sustainable fashion brand called Verdant"

Think step by step about brand positioning, design conventions in sustainable
fashion, color psychology, and typography before writing the prompt.
```

### Few-Shot CoT with Examples

Provide worked examples to guide the reasoning pattern:

```text
Example:
Input: "A pet grooming service called Paws & Claws"
Reasoning:
- Industry: Pet services. Audience: Pet owners (25-55, middle-upper income)
- Emotion: Trust, warmth, playfulness, cleanliness
- Competitors use: Paw prints, cartoon animals, soft colors (blues, greens, oranges)
- Differentiation: Use a more sophisticated/modern take to stand out
- Logo type: Combination mark (icon + wordmark)
- Style: Clean modern with a playful touch
- Colors: Teal (#2EC4B6) for trust/cleanliness + warm coral (#FF6B6B) for warmth
- Typography: Rounded sans-serif (friendly but professional)
- Icon: Minimalist cat and dog silhouette formed by negative space
Output prompt: "Professional logo design, minimalist combination mark, stylized
cat and dog silhouette using negative space, clean geometric shapes, rounded
sans-serif typography reading 'Paws & Claws', teal and coral color scheme,
white background, vector style, flat design, modern and friendly, high quality"
```

---

## 5. Generating Multiple Diverse Prompt Variations

### Strategy: Controlled Diversity

Generating multiple variations from one description requires intentional diversity across specific design axes. Without explicit guidance, the LLM tends to produce near-identical prompts with minor wording changes.

### Method 1: Axis-Based Variation

Define explicit axes of variation in the system prompt:

```text
Generate 5 diverse logo prompt variations. Each variation MUST differ on at
least 2 of the following axes:

1. Logo Type: wordmark, lettermark, icon-only, combination, emblem, abstract mark
2. Style: minimalist, geometric, vintage/retro, hand-drawn/organic, 3D/gradient, corporate
3. Color Approach: monochrome, complementary pair, analogous triad, bold/saturated, muted/pastel
4. Mood: serious/corporate, playful/friendly, elegant/luxury, bold/energetic, organic/natural
5. Composition: centered/symmetric, asymmetric, circular/badge, horizontal lockup, stacked

Label each variation with its axis choices.
```

### Method 2: Temperature + Sampling

Make multiple calls with increasing temperature:

```python
async def generate_variations(
    description: str,
    n_variations: int = 5,
    base_temp: float = 0.5,
    temp_step: float = 0.15,
) -> list[dict]:
    """Generate diverse prompt variations using temperature scaling."""
    variations = []
    for i in range(n_variations):
        temp = min(base_temp + (i * temp_step), 1.5)
        response = await client.chat.completions.create(
            model="anthropic/claude-sonnet-4",
            messages=[
                {"role": "system", "content": VARIATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Variation {i+1}/{n_variations} "
                    f"(explore {'conservative' if i == 0 else 'creative'} direction): "
                    f"{description}"},
            ],
            temperature=temp,
        )
        variations.append(parse_response(response))
    return variations
```

### Method 3: Persona-Based Variation

Use different designer personas to get diverse perspectives:

```python
PERSONAS = [
    "You are a Swiss/International style designer who values grids, clean geometry, and Helvetica.",
    "You are a boutique branding specialist who creates warm, hand-crafted, artisanal identities.",
    "You are a tech startup designer who favors bold gradients, geometric marks, and modern sans-serifs.",
    "You are a luxury brand designer who uses thin serifs, generous whitespace, and monochrome palettes.",
    "You are an illustrator-designer who creates character-driven, playful, and colorful brand marks.",
]

async def generate_persona_variations(description: str) -> list[dict]:
    tasks = [
        enhance_with_persona(description, persona)
        for persona in PERSONAS
    ]
    return await asyncio.gather(*tasks)
```

### Method 4: Single-Call Batch with Constraints

```text
Generate exactly 5 logo prompt variations for the brand below. Requirements:
- Variation 1: Minimalist wordmark only (no icon)
- Variation 2: Abstract geometric icon with brand initial
- Variation 3: Vintage/retro emblem or badge style
- Variation 4: Modern combination mark (icon + text)
- Variation 5: The most creative/unexpected interpretation you can imagine

Each must be a complete, standalone prompt ready for image generation.
```

---

## 6. Negative Prompt Generation

### Purpose

Negative prompts tell the image generation model what to exclude. For logo generation, they are critical for avoiding common failure modes: photorealism, excessive detail, unwanted text artifacts, and non-logo visual elements.

### Universal Logo Negative Prompt Template

```text
photorealistic, photograph, 3d render, realistic lighting, shadows, gradient
background, busy background, complex scene, multiple objects, person, human,
face, hands, body parts, landscape, nature scene, watermark, signature,
copyright text, url, website, low quality, blurry, pixelated, noisy, grainy,
jpeg artifacts, cropped, cut off, out of frame, duplicate, double image,
distorted, deformed, ugly, amateur, clip art, stock photo, frame, border,
mockup, perspective, angled view
```

### Style-Specific Negative Prompts

```python
NEGATIVE_PROMPTS = {
    "minimalist": (
        "complex, detailed, ornate, decorative, busy, cluttered, gradient, "
        "3d, shadow, texture, pattern, photorealistic, many colors, "
        "illustration, hand-drawn, sketchy, rough"
    ),
    "vintage": (
        "modern, futuristic, neon, gradient, flat design, minimalist, "
        "digital, tech, clean, sterile, sans-serif, geometric, abstract, "
        "photorealistic, 3d render"
    ),
    "geometric": (
        "organic, hand-drawn, sketchy, rough, natural, flowing, curved, "
        "irregular, asymmetric, photorealistic, detailed, ornate, vintage, "
        "retro, grunge, textured"
    ),
    "corporate": (
        "playful, childish, cartoon, hand-drawn, grunge, distressed, "
        "vintage, retro, neon, psychedelic, complex, busy, photorealistic, "
        "3d render, illustration"
    ),
    "hand_drawn": (
        "perfect, geometric, symmetric, digital, clean, vector, minimal, "
        "corporate, sterile, mechanical, 3d render, photorealistic, "
        "gradient, neon, futuristic"
    ),
}
```

### LLM-Generated Negative Prompts

Let the LLM generate contextual negative prompts as part of the enhancement:

```text
System: After generating the positive prompt, also generate a negative prompt
that specifically addresses:
1. The opposite of the chosen style (if minimalist, exclude ornate/complex)
2. Common failure modes for this type of logo
3. Elements that would be inappropriate for this brand's industry
4. Technical quality issues (blur, artifacts, low resolution)

The negative prompt should be specific to this particular logo, not generic.
```

### Model-Specific Considerations

- **Flux models**: Support negative prompts via `negative_prompt` parameter but respond less strongly to them than SDXL; keep them focused and short
- **SDXL/SD 3.5**: Strong negative prompt support; can use weighted tokens like `(photorealistic:1.5)`
- **Recraft V4**: Uses style presets instead of negative prompts; set `style: "vector_illustration"` or `style: "icon"`
- **Ideogram**: Limited negative prompt support; rely on positive prompt specificity instead

---

## 7. Interactive Chat-Based Refinement

### Architecture

A conversational refinement loop where the user can iteratively improve the generated logo prompt before (or after) image generation.

### Conversation Flow

```
User: "I need a logo for my coffee shop called Bean & Brew"
  |
  v
[Stage 1: Initial Enhancement]
Assistant: Here's what I understand about your brand:
  - Industry: Food & Beverage / Coffee
  - Personality: Warm, artisanal, inviting
  - Suggested direction: Hand-lettered wordmark with a coffee bean icon

  Generated prompt: "Hand-lettered logo design for 'Bean & Brew'..."

  Want to adjust anything? I can modify:
  - Style (more modern? more vintage?)
  - Colors (warm browns? or something unexpected?)
  - Icon (coffee bean? cup? steam?)
  - Typography (script? sans-serif? serif?)
  |
  v
User: "Make it more modern and use a teal color scheme"
  |
  v
[Stage 2: Refinement]
Assistant: Updated prompt with modern geometric style and teal palette...

  Changes made:
  - Switched from hand-lettered to clean geometric sans-serif
  - Color: Teal (#0D9488) primary, dark charcoal (#1F2937) secondary
  - Icon: Simplified, abstract coffee bean using geometric shapes
  |
  v
User: "Perfect, generate it"
  |
  v
[Stage 3: Image Generation with final prompt]
```

### Implementation with Conversation History

```python
class LogoPromptChat:
    """Interactive chat-based logo prompt refinement."""

    def __init__(self, client: OpenAI, model: str = "anthropic/claude-sonnet-4"):
        self.client = client
        self.model = model
        self.messages = [
            {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        ]
        self.current_prompt = None
        self.current_negative = None

    async def send_message(self, user_input: str) -> dict:
        """Send a message and get refined prompt back."""
        self.messages.append({"role": "user", "content": user_input})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )

        assistant_msg = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": assistant_msg})

        result = json.loads(assistant_msg)
        if "prompt" in result:
            self.current_prompt = result["prompt"]
            self.current_negative = result.get("negative_prompt", "")

        return result

    def get_current_prompt(self) -> tuple[str, str]:
        """Return current (prompt, negative_prompt) pair."""
        return self.current_prompt, self.current_negative
```

### Refinement System Prompt

```text
You are an interactive logo design assistant. You help users iteratively refine
their logo generation prompts through conversation.

On each turn, you MUST return a JSON object with:
- "message": Your conversational response explaining what you did/suggesting options
- "prompt": The current full positive prompt (updated with any changes)
- "negative_prompt": The current negative prompt
- "changes_made": Array of strings describing what changed from the previous version
- "suggestions": Array of 2-3 specific things the user could adjust next

Guidelines for refinement:
- When the user gives vague feedback ("make it better"), ask a specific clarifying question
- When the user gives specific feedback ("use blue"), apply it and explain the impact
- Preserve previous decisions unless the user explicitly overrides them
- After each change, briefly explain WHY this change improves the logo
- Track the conversation state so you can undo changes if asked
```

### Feedback-After-Generation Loop

For post-generation refinement (after the user has seen the generated image):

```text
The user has seen the generated logo and wants changes. Common refinement patterns:

- "Too complex" -> Simplify the prompt, add "minimalist" and "simple" modifiers
- "Text is wrong" -> Emphasize text in quotes, add "clear readable text" to prompt
- "Wrong colors" -> Replace color descriptors, be more specific with hex codes
- "Too generic" -> Add more specific visual metaphors and unique design elements
- "Not professional" -> Add "professional", "corporate", "clean" modifiers
- "Too corporate" -> Add "friendly", "approachable", "warm" modifiers

When refining after generation, make targeted changes (1-2 things at a time)
rather than rewriting the entire prompt.
```

---

## 8. Complete Code Example

### Full Pipeline Implementation

```python
"""
LLM-based prompt enhancement system for logo generation.
Uses OpenRouter API with OpenAI-compatible client.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from openai import AsyncOpenAI

# --- Configuration ---

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4"


# --- Data Models ---

@dataclass
class BrandProfile:
    brand_name: str
    industry: str
    target_audience: str
    personality: list[str]
    emotional_tone: str
    color_associations: list[str]
    key_differentiators: str


@dataclass
class DesignDirection:
    logo_type: str
    style: str
    primary_colors: list[str]
    secondary_colors: list[str]
    typography_style: str
    icon_concept: str
    composition_notes: str


@dataclass
class EnhancedPrompt:
    prompt: str
    negative_prompt: str
    style_tags: list[str]
    rationale: str
    model_params: dict = field(default_factory=dict)


# --- Client Setup ---

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
    )


async def llm_call(
    client: AsyncOpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> dict:
    """Make a single LLM call and parse JSON response."""
    response = await client.chat.completions.create(
        model=model,
        extra_headers={
            "HTTP-Referer": "https://logo-gen.app",
            "X-Title": "LogoGen Prompt Enhancer",
        },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return json.loads(content)


# --- Stage 1: Brand Analysis ---

BRAND_ANALYSIS_PROMPT = """You are a brand strategist. Analyze the brand description and extract a structured profile.

Think step by step:
1. What industry is this brand in?
2. Who is the target audience?
3. What personality traits define this brand?
4. What emotions should the brand evoke?
5. What colors are associated with this industry and personality?

Return JSON with fields:
- brand_name: string
- industry: string
- target_audience: string (demographic description)
- personality: array of 3-5 adjectives
- emotional_tone: string (the primary feeling to evoke)
- color_associations: array of color names with reasoning
- key_differentiators: string"""


async def analyze_brand(client: AsyncOpenAI, description: str) -> BrandProfile:
    data = await llm_call(client, BRAND_ANALYSIS_PROMPT, description)
    return BrandProfile(**{k: data[k] for k in BrandProfile.__dataclass_fields__})


# --- Stage 2: Design Direction ---

DESIGN_DIRECTION_PROMPT = """You are a senior logo designer. Given a brand profile, propose a concrete design direction.

Think step by step:
1. What logo type best suits this brand? (wordmark, lettermark, icon, combination, emblem, abstract)
2. What visual style matches the brand personality?
3. Select specific colors (hex codes) based on the brand's color associations and industry
4. What typography style communicates the right personality?
5. What icon or visual concept would be memorable and relevant?
6. How should the elements be composed?

Return JSON with fields:
- logo_type: string
- style: string
- primary_colors: array of hex code strings
- secondary_colors: array of hex code strings
- typography_style: string (detailed description)
- icon_concept: string (description of the visual element)
- composition_notes: string"""


async def determine_design_direction(
    client: AsyncOpenAI, brand: BrandProfile
) -> DesignDirection:
    user_prompt = json.dumps(brand.__dict__, indent=2)
    data = await llm_call(client, DESIGN_DIRECTION_PROMPT, user_prompt)
    return DesignDirection(**{k: data[k] for k in DesignDirection.__dataclass_fields__})


# --- Stage 3: Prompt Generation ---

PROMPT_GENERATION_SYSTEM = """You are a prompt engineer specializing in AI image generation for logos.
Convert design specifications into optimized prompts.

Rules for prompt writing:
- Start with the logo type and style
- Include specific color hex codes
- Describe typography in detail
- Describe the icon/visual concept clearly
- Add quality modifiers: "professional logo design", "vector style", "clean lines"
- End with technical specs: "white background", "centered composition", "high resolution"
- Keep the prompt under 200 words for best results
- Generate a targeted negative prompt that excludes the opposite of the chosen style

Return JSON with fields:
- prompt: string (the complete positive prompt)
- negative_prompt: string (things to avoid)
- style_tags: array of keyword strings
- rationale: string (brief explanation of key design choices)
- model_params: object with recommended cfg_scale (float), steps (int)"""


async def generate_prompt(
    client: AsyncOpenAI, brand: BrandProfile, design: DesignDirection
) -> EnhancedPrompt:
    user_prompt = json.dumps({
        "brand": brand.__dict__,
        "design": design.__dict__,
    }, indent=2)
    data = await llm_call(client, PROMPT_GENERATION_SYSTEM, user_prompt)
    return EnhancedPrompt(**{k: data[k] for k in EnhancedPrompt.__dataclass_fields__})


# --- Variation Generation ---

VARIATION_PROMPT = """You are a logo prompt engineer. Generate {n} diverse variations of a logo prompt.

Each variation must differ significantly in at least 2 of these axes:
- Logo type (wordmark, icon, combination, emblem, abstract)
- Visual style (minimalist, vintage, geometric, hand-drawn, gradient, corporate)
- Color approach (monochrome, complementary, analogous, bold, muted)
- Mood (serious, playful, elegant, energetic, organic)

Return JSON with field "variations": array of objects, each with:
- prompt: string
- negative_prompt: string
- style_description: string (brief label like "Minimalist Geometric")
- axes: object describing the chosen axes"""


async def generate_variations(
    client: AsyncOpenAI,
    brand: BrandProfile,
    n: int = 5,
) -> list[dict]:
    system = VARIATION_PROMPT.format(n=n)
    user_prompt = json.dumps(brand.__dict__, indent=2)
    data = await llm_call(client, system, user_prompt, temperature=0.9)
    return data["variations"]


# --- Full Pipeline ---

async def enhance_prompt(description: str) -> dict:
    """
    Full multi-stage prompt enhancement pipeline.

    Args:
        description: Simple brand description from the user.

    Returns:
        Dictionary with brand analysis, design direction,
        primary prompt, and variations.
    """
    client = get_client()

    # Stage 1: Analyze the brand
    brand = await analyze_brand(client, description)
    print(f"[Stage 1] Brand: {brand.brand_name} ({brand.industry})")

    # Stage 2: Determine design direction
    design = await determine_design_direction(client, brand)
    print(f"[Stage 2] Direction: {design.logo_type} / {design.style}")

    # Stage 3: Generate the optimized prompt
    enhanced = await generate_prompt(client, brand, design)
    print(f"[Stage 3] Prompt generated ({len(enhanced.prompt)} chars)")

    # Stage 4: Generate variations (parallel with Stage 3 in production)
    variations = await generate_variations(client, brand, n=4)
    print(f"[Stage 4] {len(variations)} variations generated")

    return {
        "brand_profile": brand.__dict__,
        "design_direction": design.__dict__,
        "primary_prompt": {
            "prompt": enhanced.prompt,
            "negative_prompt": enhanced.negative_prompt,
            "style_tags": enhanced.style_tags,
            "rationale": enhanced.rationale,
            "model_params": enhanced.model_params,
        },
        "variations": variations,
    }


# --- Interactive Chat Refinement ---

class LogoPromptChat:
    """Conversational prompt refinement session."""

    SYSTEM = """You are an interactive logo design assistant helping refine logo
generation prompts through conversation.

Return JSON on every turn:
- "message": Your conversational response
- "prompt": Current full positive prompt (always include, even if unchanged)
- "negative_prompt": Current negative prompt
- "changes_made": Array of change descriptions (empty if first turn)
- "suggestions": Array of 2-3 things the user could adjust next
- "ready_to_generate": boolean (true when the user confirms they're satisfied)

Rules:
- Apply specific feedback immediately, ask clarifying questions for vague feedback
- Preserve previous decisions unless explicitly overridden
- Explain WHY each change improves the logo
- Adjust one or two things at a time for predictable refinement"""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = get_client()
        self.model = model
        self.messages: list[dict] = [
            {"role": "system", "content": self.SYSTEM},
        ]
        self.current_prompt: str | None = None
        self.current_negative: str | None = None

    async def send(self, user_input: str) -> dict:
        self.messages.append({"role": "user", "content": user_input})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0.7,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": content})

        result = json.loads(content)
        self.current_prompt = result.get("prompt", self.current_prompt)
        self.current_negative = result.get("negative_prompt", self.current_negative)
        return result


# --- Entry Point ---

if __name__ == "__main__":
    result = asyncio.run(enhance_prompt(
        "A sustainable outdoor gear company called Trailmark that sells to "
        "environmentally-conscious millennials who love hiking and camping"
    ))
    print("\n=== RESULT ===")
    print(f"Prompt: {result['primary_prompt']['prompt']}")
    print(f"Negative: {result['primary_prompt']['negative_prompt']}")
    print(f"Variations: {len(result['variations'])}")
```

---

## Key Takeaways

1. **Multi-stage beats single-shot**: Breaking enhancement into brand analysis, design direction, and prompt generation produces more thoughtful, grounded prompts than asking an LLM to do everything at once.

2. **OpenRouter simplifies model access**: A single API key and the OpenAI SDK are sufficient to access Claude, GPT-4o, Llama, DeepSeek, and others. Model switching is a one-line change.

3. **Chain-of-thought is essential**: Forcing step-by-step design reasoning (brand analysis before color choice before prompt writing) prevents the LLM from defaulting to generic "professional modern clean" prompts.

4. **Structured output enables automation**: Using JSON response format and libraries like `instructor` with Pydantic models makes LLM outputs reliably parseable by downstream code.

5. **Diversity requires explicit constraints**: Without axis-based variation, persona-based generation, or temperature scaling, multiple LLM calls produce near-identical prompts.

6. **Negative prompts are style-dependent**: A minimalist logo's negative prompt should exclude ornate/complex elements, while a vintage logo's should exclude modern/digital elements. Generic negative prompts are less effective.

7. **Chat refinement closes the loop**: An interactive refinement session where the LLM tracks conversation state and applies incremental changes gives users fine-grained control without requiring prompt engineering knowledge.

8. **Cost is minimal**: At ~$3-15 per million tokens, even a 5-stage pipeline with 5 variations costs well under $0.01 per logo enhancement session.
