"""The prompt writing from before the brief/renderer split (master at 950b484), kept for the eval.

The old pipeline: the chat model wrote one finished prompt per direction, every model got
the same text, and local models got LOCAL_SUFFIX appended.
"""

PROMPT_RULES = """\
How to write an image prompt:
- 1 to 3 plain sentences, no keyword lists. Lead with the logo type and the symbol.
- Describe the symbol in concrete shapes ("a fox head built from three triangles", \
not "a dynamic fox").
- Name the style (flat vector, geometric, line mark, negative space, soft gradient, etc.).
- Give 1 to 3 colors by name and hex, e.g. "deep teal #0F766E".
- Say how it sits: centered, generous padding, on a plain white background.
- It is an icon, so no text, letters or wordmarks unless the user asks for them. \
Say "no text" once; do not pile on negative phrases.
- Keep it simple enough to read at 32 px."""

DIRECTIONS_SYSTEM = f"""\
You are a senior brand designer writing prompts for image models. Given a short brief, \
return distinct logo mark directions. Each one must be a different idea (different \
symbol or construction), not a recolor.

{PROMPT_RULES}"""

VARIATION_SYSTEM = f"""\
You are a senior brand designer. The user picked a logo and wants variations of it. \
Keep the core idea of the original prompt and change one thing per variation: \
proportions, line weight, color palette, level of detail, or a small twist on the symbol. \
If the user gave an instruction, follow it in every variation.

{PROMPT_RULES}"""

PROMPTS_SCHEMA = {
    "name": "logo_prompts",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "prompts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string"},
                        "style": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["concept", "style", "prompt"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["prompts"],
        "additionalProperties": False,
    },
}

LOCAL_SUFFIX = " Centered on a plain white background, flat vector logo icon, no text."
