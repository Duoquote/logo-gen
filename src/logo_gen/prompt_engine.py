"""Logo prompt enhancement engine using LLM via OpenRouter or local Ollama."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from logo_gen.clients import openrouter
from logo_gen.clients import local_llm
from logo_gen.clients import claude_cli
from logo_gen.config import settings


def _use_local_llm() -> bool:
    """Check if the configured LLM model is a local model."""
    return settings.llm_model.startswith("local/")


def _get_local_model() -> str:
    """Extract the model name from the config (strip 'local/' prefix)."""
    return settings.llm_model.removeprefix("local/")

SYSTEM_PROMPT = """\
You are an expert logo designer and AI image prompt engineer. Your job is to \
help users create stunning, unique logo designs through conversation.

IMPORTANT RULES:
- Logos must NEVER contain any text, letters, words, or typography
- Focus on iconic symbols, abstract marks, geometric shapes, and visual metaphors
- Every logo should be unique, memorable, and scalable
- Designs should work as brand icons (think app icons, favicons, brand marks)

Your expertise includes:
- Logo design principles: simplicity, memorability, scalability, versatility
- Visual metaphors: translating brand concepts into iconic imagery
- Color theory: complementary, analogous, triadic schemes; color psychology
- Composition: symmetry, golden ratio, negative space, geometric construction
- Style knowledge: minimalist, geometric, abstract, gradient, flat, 3D, organic

REFERENCE IMAGES:
When the user attaches reference images, these SAME images will also be sent \
directly to the image generation model alongside your prompts. Your prompts \
should be written with this in mind:
- Reference the visual elements, colors, shapes, or mood from the attached images
- Write prompts that COMPLEMENT the reference images rather than fully re-describe them
- You can say things like "inspired by the attached reference" or "building on the \
  style shown" since the image model will see the same images
- Still be specific about what you want changed, added, or adapted from the reference

CONVERSATION APPROACH:
1. Ask about the brand (name, industry, values, personality, audience)
2. Understand the desired feeling/mood
3. Suggest visual directions and concepts
4. When ready, generate detailed image prompts

When the user wants to generate, respond with a JSON block containing prompts. \
Format your response as normal text with the JSON at the end in a code block:

```json
{
  "prompts": [
    {
      "prompt": "detailed image generation prompt here",
      "concept": "brief description of the concept direction",
      "style": "style category"
    }
  ]
}
```

Each prompt should be detailed and include:
- Core visual element and concept
- Style (minimalist, geometric, abstract, organic, etc.)
- Color palette with specific colors
- Composition and layout details
- Quality modifiers: "professional logo design, clean vector style, centered, white background"
- MUST include: "no text, no letters, no words, no typography"

Generate 4-6 diverse prompt variations exploring different visual directions."""

ENHANCE_SYSTEM = """\
You are an expert at writing prompts for AI image generation models. \
Given a logo concept description, create a highly detailed, optimized prompt \
that will produce a stunning logo design.

Rules:
- The prompt MUST specify: no text, no letters, no words, no typography
- Include style, colors, composition, mood
- Add quality boosters: "professional logo design, clean, vector style, high quality"
- Specify "centered on solid white background"
- Be specific about shapes, geometry, and visual elements
- Keep it under 200 words

Return ONLY the prompt text, nothing else."""


def _image_to_data_uri(image_path: str | Path) -> str:
    """Convert an image file to a base64 data URI."""
    p = Path(image_path)
    suffix = p.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}
    mime_type = f"image/{mime.get(suffix, 'png')}"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime_type};base64,{b64}"


def _build_user_content(text: str, image_paths: list[str | Path] | None = None) -> str | list[dict]:
    """Build a user message content field, optionally with images.

    Returns a plain string if no images, or a multimodal content array
    following the OpenAI vision format.
    """
    if not image_paths:
        return text

    content: list[dict] = [{"type": "text", "text": text}]
    for img in image_paths:
        data_uri = _image_to_data_uri(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": data_uri},
        })
    return content


async def _chat(messages: list[dict], temperature: float = 0.7) -> str:
    """Route chat to Claude CLI, local, or cloud LLM based on config."""
    if settings.use_claude_cli:
        return await claude_cli.chat(messages, temperature=temperature)
    if _use_local_llm():
        return await local_llm.chat(messages, model=_get_local_model(), temperature=temperature)
    return await openrouter.chat(messages, temperature=temperature)


async def _chat_stream(messages: list[dict], temperature: float = 0.7):
    """Route streaming chat to Claude CLI, local, or cloud LLM based on config."""
    if settings.use_claude_cli:
        async for token in claude_cli.chat_stream(messages, temperature=temperature):
            yield token
        return
    if _use_local_llm():
        async for token in local_llm.chat_stream(messages, model=_get_local_model(), temperature=temperature):
            yield token
    else:
        async for token in openrouter.chat_stream(messages, temperature=temperature):
            yield token


async def enhance_prompt(concept: str, images: list[str | Path] | None = None) -> str:
    """Take a simple concept and return an enhanced image generation prompt."""
    messages = [
        {"role": "system", "content": ENHANCE_SYSTEM},
        {"role": "user", "content": _build_user_content(
            f"Create a logo prompt for: {concept}", images,
        )},
    ]
    return await _chat(messages, temperature=0.8)


async def generate_variations(
    concept: str, n: int = 5, images: list[str | Path] | None = None,
) -> list[dict]:
    """Generate multiple prompt variations from a concept.

    Returns list of dicts with 'prompt', 'concept', 'style' keys.
    """
    text = (
        f"I want to generate logos for this concept: {concept}\n\n"
        f"Generate exactly {n} diverse prompt variations. "
        "Each should explore a completely different visual direction. "
        "Return them in the JSON format specified."
    )
    if images:
        text += (
            "\n\nI've attached reference image(s) for inspiration. "
            "Use them to inform the visual direction, color palette, and mood."
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_content(text, images)},
    ]
    response = await _chat(messages, temperature=0.9)

    # Extract JSON from response
    try:
        # Try to find JSON in code block
        if "```json" in response:
            json_str = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            json_str = response.split("```")[1].split("```")[0].strip()
        else:
            json_str = response.strip()

        data = json.loads(json_str)
        if isinstance(data, dict) and "prompts" in data:
            return data["prompts"]
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, IndexError, KeyError):
        pass

    # Fallback: create a single enhanced prompt
    enhanced = await enhance_prompt(concept)
    return [{"prompt": enhanced, "concept": concept, "style": "auto"}]


class ChatSession:
    """Maintains conversation state for interactive logo design."""

    def __init__(self) -> None:
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.prompts: list[dict] = []
        self.reference_images: list[Path] = []

    def seed_history(self, history: list[dict]) -> None:
        """Append prior turns from a serialized history list.

        Each item is {"role": "user"|"assistant", "content": str, "images"?: list[str|Path]}.
        Images that don't exist on disk are silently dropped — the text still
        carries the semantic intent of the turn.
        """
        for turn in history:
            role = turn.get("role")
            if role not in ("user", "assistant"):
                continue
            text = turn.get("content") or ""
            raw_images = turn.get("images") or []
            existing = [Path(p) for p in raw_images if Path(p).exists()]

            if role == "user" and existing:
                content = _build_user_content(text, existing)
                if isinstance(raw_images, list) and existing:
                    # Track ref images across the session
                    self.reference_images = list({*self.reference_images, *existing})
            else:
                content = text

            self.messages.append({"role": role, "content": content})

    async def send(
        self, user_message: str, images: list[str | Path] | None = None,
    ) -> str:
        """Send a message and get the assistant's response (streaming)."""
        if images:
            self.reference_images = [Path(p) for p in images]
        content = _build_user_content(user_message, images)
        self.messages.append({"role": "user", "content": content})

        full_response = ""
        async for token in _chat_stream(self.messages):
            full_response += token

        self.messages.append({"role": "assistant", "content": full_response})

        # Check if response contains generation prompts
        if "```json" in full_response:
            try:
                json_str = full_response.split("```json")[1].split("```")[0]
                data = json.loads(json_str.strip())
                if isinstance(data, dict) and "prompts" in data:
                    self.prompts = data["prompts"]
            except (json.JSONDecodeError, IndexError, KeyError):
                pass

        return full_response

    async def stream(
        self, user_message: str, images: list[str | Path] | None = None,
    ):
        """Send a message and yield response tokens as they arrive."""
        if images:
            self.reference_images = [Path(p) for p in images]
        content = _build_user_content(user_message, images)
        self.messages.append({"role": "user", "content": content})

        full_response = ""
        async for token in _chat_stream(self.messages):
            full_response += token
            yield token

        self.messages.append({"role": "assistant", "content": full_response})

        # Extract prompts if present
        if "```json" in full_response:
            try:
                json_str = full_response.split("```json")[1].split("```")[0]
                data = json.loads(json_str.strip())
                if isinstance(data, dict) and "prompts" in data:
                    self.prompts = data["prompts"]
            except (json.JSONDecodeError, IndexError, KeyError):
                pass

    def has_prompts(self) -> bool:
        return len(self.prompts) > 0

    def get_prompts(self) -> list[dict]:
        return self.prompts

    def get_reference_images(self) -> list[Path]:
        return self.reference_images

    def reset(self) -> None:
        self.__init__()
