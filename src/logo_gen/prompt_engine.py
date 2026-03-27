"""Logo prompt enhancement engine using LLM via OpenRouter."""

from __future__ import annotations

import json

from logo_gen.clients import openrouter

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


async def enhance_prompt(concept: str) -> str:
    """Take a simple concept and return an enhanced image generation prompt."""
    messages = [
        {"role": "system", "content": ENHANCE_SYSTEM},
        {"role": "user", "content": f"Create a logo prompt for: {concept}"},
    ]
    return await openrouter.chat(messages, temperature=0.8)


async def generate_variations(concept: str, n: int = 5) -> list[dict]:
    """Generate multiple prompt variations from a concept.

    Returns list of dicts with 'prompt', 'concept', 'style' keys.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"I want to generate logos for this concept: {concept}\n\n"
                f"Generate exactly {n} diverse prompt variations. "
                "Each should explore a completely different visual direction. "
                "Return them in the JSON format specified."
            ),
        },
    ]
    response = await openrouter.chat(messages, temperature=0.9)

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

    async def send(self, user_message: str) -> str:
        """Send a message and get the assistant's response (streaming)."""
        self.messages.append({"role": "user", "content": user_message})

        full_response = ""
        async for token in openrouter.chat_stream(self.messages):
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

    async def stream(self, user_message: str):
        """Send a message and yield response tokens as they arrive."""
        self.messages.append({"role": "user", "content": user_message})

        full_response = ""
        async for token in openrouter.chat_stream(self.messages):
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

    def reset(self) -> None:
        self.__init__()
