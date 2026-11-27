"""How to prompt each image model, as data.

Each profile says how a model (or a family of models) reads prompts: the
style it was tuned on, how long a prompt it actually uses, whether it takes a
negative prompt, how well it draws text, and which request parameters help.
The renderers in render.py read these; nothing model-specific lives there.

Lookup for a model id (see ``profile_for``):

1. a profile that lists the id exactly,
2. the first profile with a glob pattern that matches the id,
3. the first profile for the id's provider (the part before the slash),
4. ``DEFAULT``, a cautious natural-language profile.

To add a model, add or extend a profile below. research/prompting-2026.md
has the reasoning and sources behind each entry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from typing import Literal

Style = Literal["natural", "structured", "recraft", "keywords", "tags"]
TextSkill = Literal["strong", "fair", "weak", "none"]
Negation = Literal["clause", "positive", "never"]


@dataclass(frozen=True)
class ModelProfile:
    id: str
    name: str
    style: Style
    encoder: str
    max_words: int  # past this, the model ignores or blurs what comes later
    text: TextSkill  # how reliably it renders the exact words you ask for
    match: tuple[str, ...] = ()  # glob patterns on the model id
    providers: tuple[str, ...] = ()  # fallback by provider prefix
    token_limit: int | None = None  # hard text-encoder limit, if there is one
    negative: bool = False  # the request can carry a separate negative prompt
    # How to say what to leave out, when there's no negative prompt:
    # clause: one short "No x, no y." at the end (LLM encoders, which mostly cope),
    # positive: only positive rephrasing (providers that ask for it, T5),
    # never: CLIP-only models, where any mention pulls the concept in.
    negation: Negation = "positive"
    max_avoid: int = 3  # most exclusions worth stating
    max_text_words: int = 3  # longest wording it renders reliably
    lead: str = "flat vector"  # the medium phrase that opens the prompt
    no_text: str = "Symbol only."  # what to say when the brief has no text
    text_tail: str = ""  # appended after the text clause ("No other text.")
    # Request parameters. Cloud: OpenRouter Images API fields, sent only when
    # the model lists them ("lowest" picks the smallest listed value), plus
    # `provider` passthrough options. Local: steps, guidance, resolution.
    params: dict = field(default_factory=dict)
    # Recraft: send the palette and background as RGB `controls`.
    color_controls: bool = False
    negative_default: str = ""  # used only when negative is True
    quality_tags: str = ""  # leading tags for tag-trained models
    do: tuple[str, ...] = ()
    dont: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES: list[ModelProfile] = [
    # ── OpenAI ───────────────────────────────────────────────────────
    ModelProfile(
        id="gpt-image",
        name="OpenAI GPT Image",
        style="structured",
        encoder="GPT (multimodal LLM)",
        max_words=170,
        text="strong",
        max_text_words=8,
        match=("openai/gpt-image*", "openai/gpt-*-image*", "openai/dall-e*"),
        providers=("openai",),
        negation="clause",
        max_avoid=4,
        lead="flat vector",
        no_text="No text, letters or numbers anywhere.",
        text_tail="No extra text.",
        params={"quality": "medium", "output_format": "png"},
        do=(
            "Labeled sections: subject, style, colors, composition, background, text, constraints.",
            "Quote exact wording; spell unusual names letter by letter.",
            "Use the background=transparent parameter and also say \"transparent background\".",
            "\"Original, vector-like shapes, strong silhouette, balanced negative space, reads at small sizes.\"",
        ),
        dont=(
            "quality=low smudges letters; medium is the cost and legibility middle ground.",
            "Don't describe a backdrop when you want transparency.",
        ),
        sources=("https://developers.openai.com/api/docs/guides/image-prompting.md",),
    ),
    # ── Google ───────────────────────────────────────────────────────
    ModelProfile(
        id="gemini-image",
        name="Google Gemini image (Nano Banana)",
        style="natural",
        encoder="Gemini (multimodal LLM)",
        max_words=110,
        text="strong",
        max_text_words=5,
        match=("google/gemini-*image*", "google/*nano-banana*", "google/imagen*"),
        providers=("google",),
        negation="positive",
        lead="modern, flat vector",
        no_text="The mark is a pure pictorial symbol.",
        params={"resolution": "1K"},
        do=(
            "A narrative paragraph of a few sentences: purpose first (\"a logo for ...\"), then shapes, colors, background.",
            "Describe what should be there instead of what shouldn't (Google's \"semantic negative prompts\").",
            "Quote exact text and describe the font in words.",
        ),
        dont=(
            "No keyword lists, no \"no X\" lists.",
            "No transparency parameter: ask for plain white and remove it afterwards.",
        ),
        sources=(
            "https://aistudio.google.com/docs/image-generation",
            "https://developers.googleblog.com/en/how-to-prompt-gemini-2-5-flash-image-generation-for-the-best-results/",
        ),
    ),
    # ── Recraft ──────────────────────────────────────────────────────
    ModelProfile(
        id="recraft",
        name="Recraft",
        style="recraft",
        encoder="Recraft (not documented)",
        max_words=80,
        text="strong",
        max_text_words=6,
        match=("recraft/*",),
        providers=("recraft",),
        negation="clause",
        lead="flat vector",
        no_text="No text.",
        color_controls=True,
        do=(
            "General to specific: graphic type, subject, shape logic, color system, line rules.",
            "\"Flat colors only, no gradients, shadows, or texture.\" Ask for one unified silhouette.",
            "The palette goes in as RGB controls.colors (OpenRouter passes provider.options.recraft.controls).",
            "Vector variants return SVG; utility variants suit icons.",
        ),
        dont=(
            "V4 and V4.1 ignore curated style/substyle, so the prompt carries the style.",
            "No texture or material words, no stacked adjectives.",
        ),
        sources=(
            "https://www.recraft.ai/docs/prompt-engineering-guide/prompting-with-recraft-v4.md",
            "https://www.recraft.ai/docs/api-reference/models/recraft-v4-1.md",
        ),
    ),
    # ── Black Forest Labs ────────────────────────────────────────────
    ModelProfile(
        id="flux-3",
        name="FLUX 3 (cloud)",
        style="natural",
        encoder="not documented",
        max_words=90,
        text="strong",
        max_text_words=5,
        match=("black-forest-labs/flux-3*",),
        negation="positive",
        lead="flat-color graphic",
        params={"output_format": "png", "resolution": "1K", "provider": {"options": {"black-forest-labs": {"grounding": False}}}},
        do=(
            "Open with one sentence that says what it is; add detail after.",
            "Tie each hex code to one element (\"the mark in #0F766E\").",
            "grounding=false: an invented brand name gains nothing from a web search.",
        ),
        dont=("BFL's docs don't use \"vector\"; say \"flat-color graphic\".", "No quality tags, no \"no X\"."),
        sources=("https://docs.bfl.ml/api-reference/utility/generate-an-image-with-flux-3.md",),
    ),
    ModelProfile(
        id="flux-2-klein",
        name="FLUX.2 klein (cloud)",
        style="natural",
        encoder="Qwen3 (512 tokens)",
        max_words=85,
        token_limit=512,
        text="fair",
        max_text_words=2,
        match=("black-forest-labs/flux.2-klein*",),
        negation="clause",
        max_avoid=2,
        params={"output_format": "png"},
        do=("Prose under about 100 words; what you write is exactly what it gets (no upsampling).",),
        dont=("Text \"may be inaccurate\" per the model card; keep wording to a word or two.",),
        sources=("https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",),
    ),
    ModelProfile(
        id="flux-2",
        name="FLUX.2 pro, flex, max (cloud)",
        style="natural",
        encoder="Mistral 3 (FLUX.2 dev; API encoder not documented)",
        max_words=85,
        text="strong",
        max_text_words=5,
        match=("black-forest-labs/flux.2-*", "black-forest-labs/flux-2*"),
        providers=("black-forest-labs",),
        negation="positive",
        params={"output_format": "png"},
        do=(
            "Subject, then style, then context. Word order matters; front-load the mark.",
            "Hex codes tied to elements; quoted text with font and placement.",
        ),
        dont=("No negative prompt and no \"no X\" phrasing.", "No quality tags."),
        sources=("https://docs.bfl.ai/guides/prompting_guide_flux2.md",),
    ),
    ModelProfile(
        id="flux-2-klein-local",
        name="FLUX.2 klein 4B (local)",
        style="natural",
        encoder="Qwen3 4B (512 tokens)",
        max_words=85,
        token_limit=512,
        text="fair",
        max_text_words=2,
        match=("local/flux2-klein*",),
        negation="clause",
        max_avoid=2,
        params={"steps": 4, "guidance": 1.0},
        dont=("Distilled: guidance is ignored, so negative prompts do nothing.",),
        sources=("https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",),
    ),
    ModelProfile(
        id="flux-1-schnell-local",
        name="FLUX.1 schnell (local)",
        style="natural",
        encoder="T5-XXL (256 tokens here) plus CLIP-L (77 tokens)",
        max_words=55,
        token_limit=256,
        text="fair",
        max_text_words=2,
        match=("local/flux1-schnell*",),
        negation="positive",
        params={"steps": 4, "guidance": 0.0},
        do=("Key nouns first: the CLIP half only sees the first 77 tokens.",),
        dont=("Guidance 0: no negative prompt. Negated words are unreliable with T5 and CLIP.",),
        sources=("https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux",),
    ),
    # ── Local Z-Image ────────────────────────────────────────────────
    ModelProfile(
        id="z-image-turbo",
        name="Z-Image Turbo (local)",
        style="natural",
        encoder="Qwen3 4B (512 tokens)",
        max_words=120,
        token_limit=512,
        text="strong",
        max_text_words=4,
        match=("local/zimage*", "*/z-image*"),
        negation="clause",
        max_avoid=2,
        params={"steps": 9, "guidance": 0.0},
        do=(
            "Plain, objective description, 40 to 150 words; it was trained on that caption style.",
            "Strong English and Chinese text: quote it.",
        ),
        dont=("Turbo has no CFG, so negative prompts are ignored.",),
        sources=("https://huggingface.co/Tongyi-MAI/Z-Image-Turbo", "https://arxiv.org/html/2511.22699v2"),
    ),
    # ── Local SD-Turbo ───────────────────────────────────────────────
    ModelProfile(
        id="sd-turbo",
        name="SD-Turbo (local)",
        style="keywords",
        encoder="OpenCLIP ViT-H (77 tokens)",
        max_words=25,
        token_limit=77,
        text="none",
        max_text_words=0,
        match=("local/sd-turbo*", "local/sdxl-turbo*"),
        negation="never",
        no_text="",
        params={"steps": 2, "guidance": 0.0},
        do=("10 to 25 comma-separated keywords, most important first.",),
        dont=("It can't render legible text: never ask for words, and don't write \"no text\" either.", "Guidance 0: no negatives."),
        sources=("https://huggingface.co/stabilityai/sd-turbo",),
    ),
    # ── ByteDance Seedream ───────────────────────────────────────────
    ModelProfile(
        id="seedream",
        name="ByteDance Seedream",
        style="natural",
        encoder="not documented",
        max_words=110,
        text="strong",
        max_text_words=6,
        match=("bytedance-seed/seedream*", "bytedance*/seedream*"),
        providers=("bytedance-seed", "bytedance"),
        negation="clause",
        no_text="No text.",
        text_tail="No other text.",
        params={"resolution": "lowest", "output_format": "png"},
        do=("A brief in clauses: deliverable, subject, style, exclusions.", "Quote the copy and end with \"no other text\"."),
        dont=("No negative prompt field on OpenRouter.",),
        sources=("https://runware.ai/docs/models/bytedance-seedream-5-5/guides/prompting",),
    ),
    # ── Qwen ─────────────────────────────────────────────────────────
    ModelProfile(
        id="qwen-image",
        name="Qwen Image",
        style="natural",
        encoder="Qwen (LLM)",
        max_words=100,
        text="strong",
        max_text_words=8,
        match=("qwen/qwen-image*",),
        providers=("qwen",),
        negation="clause",
        no_text="No readable text.",
        params={"resolution": "lowest"},
        do=("Layered prose: subject, composition, style.", "It renders lettering readily; say \"no readable text\" for symbols."),
        dont=("Prompt rewriting may be on upstream, so seeds may not reproduce.",),
        sources=("https://www.alibabacloud.com/help/en/model-studio/qwen-image-api",),
    ),
    # ── Other cloud models ───────────────────────────────────────────
    ModelProfile(
        id="hunyuan-image",
        name="Tencent HunyuanImage",
        style="natural",
        encoder="Hunyuan multimodal LLM",
        max_words=120,
        text="fair",
        max_text_words=4,
        match=("tencent/hy-image*", "tencent/hunyuan*"),
        providers=("tencent",),
        negation="clause",
        params={"resolution": "lowest"},
        sources=("https://huggingface.co/tencent/HunyuanImage-3.0",),
    ),
    ModelProfile(
        id="krea",
        name="Krea 2",
        style="natural",
        encoder="not documented",
        max_words=60,
        text="fair",
        match=("krea/*",),
        providers=("krea",),
        negation="positive",
        params={"resolution": "lowest"},
        do=("Short prose; it's tuned to explore, so expect looser results.",),
        sources=("https://www.krea.ai/docs/user-guide/features/krea-2",),
    ),
    ModelProfile(
        id="mai-image",
        name="Microsoft MAI Image",
        style="natural",
        encoder="not documented (32k-token input)",
        max_words=100,
        text="strong",
        max_text_words=5,
        match=("microsoft/mai-image*",),
        providers=("microsoft",),
        negation="clause",
        params={"output_format": "png"},
        sources=("https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-mai",),
    ),
    ModelProfile(
        id="muse-image",
        name="Meta Muse Image",
        style="structured",
        encoder="Meta multimodal LLM (plans before rendering)",
        max_words=140,
        text="fair",
        max_text_words=3,
        match=("meta/muse*",),
        providers=("meta",),
        negation="clause",
        no_text="No text, letters or numbers anywhere.",
        params={"output_format": "png"},
        dont=("Text varies run to run (Meta); keep it short.",),
        sources=("https://developer.meta.com/ai/models/muse-image/",),
    ),
    ModelProfile(
        id="riverflow",
        name="Sourceful Riverflow",
        style="structured",
        encoder="not documented",
        max_words=130,
        text="strong",
        max_text_words=6,
        match=("sourceful/riverflow*",),
        providers=("sourceful",),
        negation="clause",
        no_text="No text, letters or numbers anywhere.",
        params={"resolution": "lowest", "output_format": "png"},
        sources=("https://replicate.com/sourceful/riverflow-2.0-pro/readme",),
    ),
    ModelProfile(
        id="grok-imagine",
        name="xAI Grok Imagine",
        style="natural",
        encoder="Grok (multimodal LLM)",
        max_words=100,
        text="fair",
        max_text_words=4,
        match=("x-ai/grok-imagine*", "x-ai/grok-*image*"),
        providers=("x-ai",),
        negation="clause",
        params={"resolution": "lowest", "quality": "medium"},
        dont=("quality=auto means low for generation; pin medium.",),
        sources=("https://openrouter.ai/x-ai/grok-imagine-image-2.0",),
    ),
    # ── Not offered on OpenRouter right now, kept for when they are ──
    ModelProfile(
        id="ideogram",
        name="Ideogram",
        style="structured",
        encoder="not documented",
        max_words=110,
        text="strong",
        max_text_words=10,
        match=("ideogram*/*", "*/ideogram*"),
        providers=("ideogram", "ideogram-ai"),
        negative=True,
        negation="positive",
        params={"magic_prompt": "OFF", "style_type": "DESIGN"},
        negative_default="photo, 3d render, gradient, extra text, watermark",
        do=("Sentences, with quoted text near the start.", "Magic prompt off so it doesn't rewrite the brief."),
        dont=("`::1` weights and `--ar` flags are read as plain words.",),
        sources=("https://docs.ideogram.ai/using-ideogram/getting-started/prompting-guide/in-a-nutshell",),
    ),
    ModelProfile(
        id="booru-tags",
        name="Booru-tag SDXL models (Illustrious, NoobAI, Pony)",
        style="tags",
        encoder="CLIP-L + CLIP-G (77-token chunks)",
        max_words=40,
        token_limit=77,
        text="none",
        max_text_words=0,
        match=("*illustrious*", "*noobai*", "*pony*", "*animagine*"),
        negative=True,
        negation="never",
        no_text="",
        params={"steps": 25, "guidance": 5.5},
        quality_tags="masterpiece, best quality",
        negative_default="worst quality, low quality, lowres, text, signature, watermark, jpeg artifacts, "
        "photorealistic, 3d, gradient background",
        do=("Danbooru tags, model-specific quality tags first.", "\"simple background, white background, no humans, symbol\"."),
        dont=("Pony score_* tags don't apply to Illustrious.", "A poor fit for logos: the training data is anime illustration."),
        sources=("https://huggingface.co/Laxhar/noobai-XL-1.0",),
    ),
    ModelProfile(
        id="sdxl",
        name="SDXL and SD 1.5 style models",
        style="keywords",
        encoder="CLIP-L + OpenCLIP bigG (77 tokens each)",
        max_words=45,
        token_limit=77,
        text="weak",
        max_text_words=1,
        match=("*sdxl*", "*stable-diffusion*", "stabilityai/*", "local/sd-*"),
        providers=("stabilityai",),
        negative=True,
        negation="never",
        no_text="",
        params={"steps": 30, "guidance": 6.0},
        negative_default="photo, realistic, 3d render, shading, gradient, text, letters, watermark, blurry, "
        "busy background, clutter",
        do=("Comma-separated keywords, subject first; the negative field works at CFG 5 to 8.",),
        dont=("Anything past 77 tokens is truncated.",),
        sources=("https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/60",),
    ),
]

DEFAULT = ModelProfile(
    id="default",
    name="Unknown model (natural language)",
    style="natural",
    encoder="unknown",
    max_words=80,
    text="fair",
    max_text_words=3,
    negation="positive",
    do=("A compact descriptive paragraph is the safest bet for current models.",),
    dont=("Used when no profile matches. Add a profile to profiles.py to tune a new model.",),
)

_BY_ID = {p.id: p for p in PROFILES} | {DEFAULT.id: DEFAULT}


def get(profile_id: str) -> ModelProfile:
    return _BY_ID.get(profile_id, DEFAULT)


def profile_for(model: str) -> ModelProfile:
    """The profile for an OpenRouter id ("recraft/recraft-v4.1") or a local ref ("local/zimage-turbo")."""
    mid = (model or "").lower().split(":", 1)[0]
    for p in PROFILES:
        if mid in p.match:
            return p
    for p in PROFILES:
        if any(fnmatchcase(mid, pat) for pat in p.match):
            return p
    provider = mid.split("/", 1)[0] if "/" in mid else ""
    if provider and provider != "local":
        for p in PROFILES:
            if provider in p.providers:
                return p
    return DEFAULT
