"""Turn a design brief into the prompt and parameters one image model reads best.

There is one renderer per prompt style (see profiles.Style). They read only
the brief and the profile, so a new model usually needs a profile entry and no
code. Every renderer:

- keeps within the profile's useful length, dropping the least important
  parts first,
- asks for text only when the model can draw it, and says so in `notes`
  when it had to leave text out,
- states what to leave out the way the model copes with: a negative prompt
  where one exists, one short "No x, no y." clause for LLM-encoder models,
  positive rephrasing for providers that ask for it, and nothing at all for
  CLIP-only models, where naming a thing tends to put it in the picture.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace

from logo_gen.prompting import brief as briefs
from logo_gen.prompting.brief import Brief
from logo_gen.prompting.profiles import ModelProfile, profile_for


@dataclass
class Rendered:
    model: str
    profile: str
    profile_name: str
    style: str
    prompt: str
    negative_prompt: str = ""
    params: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Shared phrasing ──────────────────────────────────────────────────

MARK_PHRASE = {
    "pictorial": "logo symbol",
    "abstract": "abstract logo mark",
    "lettermark": "lettermark logo",
    "wordmark": "wordmark logo",
    "combination": "logo with a symbol and a wordmark",
    "emblem": "emblem logo",
    "mascot": "mascot logo",
}

# Avoids with a clean positive form, for models that should only hear what
# to draw. Anything not in this table is left out for them.
POSITIVE = {
    "gradient": "flat solid colors",
    "gradients": "flat solid colors",
    "shading": "flat fills without shading",
    "shadow": "flat fills",
    "shadows": "flat fills",
    "drop shadow": "flat fills",
    "drop shadows": "flat fills",
    "soft shadows": "flat fills",
    "3d": "flat 2D",
    "3d render": "flat 2D",
    "3d lighting": "flat 2D",
    "bevel": "flat 2D",
    "bevels": "flat 2D",
    "3d bevels": "flat 2D",
    "gloss": "matte flat colors",
    "photo": "flat graphic",
    "photorealistic": "flat graphic",
    "realism": "flat graphic",
    "detail": "few, bold shapes",
    "fine detail": "few, bold shapes",
    "too much detail": "few, bold shapes",
    "clutter": "few, bold shapes",
    "busy": "few, bold shapes",
    "thin lines": "sturdy line weight",
    "thin hairlines": "sturdy line weight",
    "hairlines": "sturdy line weight",
    "hairline strokes": "sturdy line weight",
    "texture": "clean solid fills",
    "textures": "clean solid fills",
    "glow": "crisp edges",
    "background clutter": "a plain empty background",
    "busy background": "a plain empty background",
    "many colors": "a limited palette",
    "rainbow": "a limited palette",
}

ROLES = ("Main shapes in", "accents in", "details in")

# Rough names for hex colors, for CLIP models that can't read hex codes.
_NAMED = {
    "black": (20, 20, 20), "white": (245, 245, 245), "gray": (128, 128, 128), "red": (210, 40, 40),
    "orange": (240, 130, 30), "yellow": (240, 210, 40), "gold": (200, 160, 50), "green": (50, 160, 70),
    "teal": (20, 130, 120), "cyan": (40, 190, 210), "blue": (40, 90, 210), "navy": (20, 35, 90),
    "purple": (120, 60, 170), "pink": (230, 110, 170), "brown": (120, 75, 40), "beige": (225, 205, 170),
    "cream": (245, 235, 210), "maroon": (110, 25, 35), "olive": (110, 120, 40), "mint": (150, 220, 190),
}


# "..., reading as a drone capture footprint", "..., the small wedge standing for taxes".
# Image models draw every noun they get, so the meaning has to stay out of the prompt.
_MEANING = re.compile(
    r"[,;]\s*(?:[\w\s-]{0,30}?\s)?(?:reading|standing|stands|representing|represents|symbolizing|symbolising|"
    r"suggesting|suggests|evoking|evokes|signifying|hinting|alluding|referencing|a nod|a metaphor|as if)\b[^,;]*",
    re.I,
)


def drawable(subject: str) -> str:
    """The subject without clauses that explain what it means."""
    out = _MEANING.sub("", subject or "").strip(" ,;.")
    return out or (subject or "").strip()


def _rgb(hex_value: str) -> tuple[int, int, int] | None:
    m = re.search(r"#?([0-9a-fA-F]{6})\b", hex_value or "")
    if not m:
        return None
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def color_name(hex_value: str) -> str:
    rgb = _rgb(hex_value) if re.fullmatch(r"#?[0-9a-fA-F]{6}", hex_value or "") else None
    if not rgb:
        return ""
    return min(_NAMED, key=lambda n: sum((a - c) ** 2 for a, c in zip(_NAMED[n], rgb)))


def _words(s: str) -> int:
    return len(s.split())


def _sentence(s: str) -> str:
    s = " ".join(s.split()).strip().rstrip(".,;")
    if not s:
        return ""
    return s[0].upper() + s[1:] + "."


def _article(phrase: str) -> str:
    return "An" if phrase[:1].lower() in "aeiou" else "A"


def _join(*parts: str, sep: str = ", ") -> str:
    return sep.join(p.strip() for p in parts if p and p.strip())


def _fit(parts: list[tuple[int, str]], max_words: int, sep: str = " ") -> str:
    """Join (priority, text) parts in order, dropping the lowest priority ones until it fits.

    Priority 9 and up is never dropped; if those alone are too long, the end is cut.
    """
    keep = [p for p in parts if p[1]]
    while sum(_words(t) for _, t in keep) > max_words:
        droppable = [i for i, (pr, _) in enumerate(keep) if pr < 9]
        if not droppable:
            break
        keep.pop(min(droppable, key=lambda i: (keep[i][0], -i)))
    text = sep.join(t for _, t in keep)
    words = text.split(" ")
    if sum(_words(t) for _, t in keep) > max_words:
        text = " ".join(" ".join(words).split()[:max_words]).rstrip(",;") + "."
    return text


def _palette_list(b: Brief, hex_ok: bool = True) -> list[str]:
    if hex_ok:
        return [c.label() for c in b.palette]
    names = [c.name.split()[-1].lower() if c.name else color_name(c.hex) for c in b.palette]
    return [x for x in dict.fromkeys(names) if x]


def _and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _colors_by_role(b: Brief) -> str:
    """'Main shapes in deep teal #0F766E, accents in warm amber #F59E0B.' BFL: tie each hex to an element."""
    labels = _palette_list(b)
    if not labels:
        return ""
    if len(labels) == 1:
        return f"All shapes in {labels[0]}"
    return ", ".join(f"{role} {c}" for role, c in zip(ROLES, labels))


def _exclusions(b: Brief, p: ModelProfile, extra: list[str]) -> tuple[list[str], list[str]]:
    """(things to exclude in words, positive rephrasings) for this profile."""
    wanted = " ".join((b.style, b.shape_language, b.subject)).lower()
    extra = [a for a in extra if a.strip() and a.strip().lower() not in wanted]  # never fight the brief
    items = list(dict.fromkeys(a.strip() for a in [*b.avoid, *extra] if a.strip()))
    if b.format == "app icon":
        items.append("gloss")  # "app icon" drifts toward glossy tiles
    items = [i.lower().removeprefix("no ").removeprefix("any ") for i in items]
    if b.has_text:
        items = [i for i in items if i not in ("text", "letters", "letterforms")]
    positive: list[str] = []
    for i in items:  # at most two, and not two ways of saying "flat"
        pos = POSITIVE.get(i)
        if pos and pos not in positive and not (pos.startswith(("flat", "matte")) and any(x.startswith("flat") for x in positive)):
            positive.append(pos)
    positive = positive[:2]
    direct = items[: p.max_avoid] if (p.negative or p.negation == "clause") else []
    return direct, positive


def _text_plan(b: Brief, p: ModelProfile) -> tuple[str, list[str]]:
    """The text to ask for ('' for none) and notes about it."""
    if not b.has_text:
        return "", []
    if p.text == "none":
        return "", [f"{p.name} can't draw lettering, so this prompt asks for the symbol only. "
                    f"Add \"{b.text}\" in an editor, or use a model that renders text."]
    reliable = _words(b.text) <= p.max_text_words and p.text != "weak"
    if b.mark_type in ("wordmark", "lettermark", "combination", "emblem"):
        # The text is the logo here, so ask for it anyway and say how likely it is to come out right.
        return b.text, ([] if reliable else [f"{p.name} often misspells lettering; check \"{b.text}\" closely."])
    if not reliable:
        return "", [f"{p.name} is unreliable with text, so this prompt leaves out \"{b.text}\"."]
    return b.text, []


def _mark(b: Brief, text: str) -> str:
    mark = b.mark_type
    if not text and mark in ("wordmark", "combination"):
        mark = "pictorial"
    if b.format == "app icon":
        return "app icon"
    if b.format == "favicon":
        return "logo symbol, bold enough to read at 16 px"
    return MARK_PHRASE.get(mark, "logo mark")


def _style_head(b: Brief, lead: str) -> tuple[str, str]:
    """Lead phrase and the rest of the style: "flat vector, two colors" -> ("flat vector", "two colors")."""
    parts = [x.strip() for x in re.split(r"[,;]", b.style or "") if x.strip()]
    if not parts:
        return lead, ""
    head = parts[0]
    if lead == "flat-color graphic":  # FLUX 3: BFL's vocabulary has no "vector"
        head = re.sub(r"\bflat vector\b|\bvector\b", "flat-color graphic", head, flags=re.I)
    return head, ", ".join(parts[1:])


def _near_white(bg: str) -> bool:
    """White, off-white and pale tints. They read as a colored canvas to most models and make removal harder."""
    low = bg.lower()
    if any(w in low for w in ("white", "transparent", "off-white", "ivory")) or not low.strip():
        return True
    rgb = _rgb(bg)
    if rgb is None:
        return False
    r, g, b = (c / 255 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.9 and max(rgb) - min(rgb) < 40


def _background(b: Brief, transparent: bool) -> str:
    if transparent:
        return "transparent background"
    bg = b.background.strip() or "white"
    if _near_white(bg):
        return "plain white background"
    return bg if "background" in bg.lower() else f"solid {bg} background"


def _placement(b: Brief, transparent: bool) -> str:
    centered = any(w in b.composition.lower() for w in ("center", "centre"))
    head = "A single mark with generous even padding" if centered else "A single mark, centered with generous even padding"
    return f"{head}, on a {_background(b, transparent)}, nothing else in frame"


def _text_clause(b: Brief, text: str) -> str:
    where = {
        "wordmark": "as the whole logo",
        "lettermark": "as a single bold monogram",
        "combination": "set beside the symbol",
        "emblem": "inside the emblem",
    }.get(b.mark_type, "below the symbol")
    return f"the text \"{text}\" {where}, spelled exactly, in clean lettering"


def _no_clause(direct: list[str], with_text: bool) -> str:
    items = (["text"] if with_text else []) + direct
    return _sentence("No " + ", no ".join(items)) if items else ""


# ── Renderers ────────────────────────────────────────────────────────


def render_natural(b: Brief, p: ModelProfile, transparent: bool, extra_avoid: list[str]) -> tuple[str, str, list[str]]:
    text, notes = _text_plan(b, p)
    direct, positive = _exclusions(b, p, extra_avoid)
    mark = _mark(b, text)
    head, rest = _style_head(b, p.lead)
    lead = f"{_article(head)} {head} {mark}" + (f" of {b.subject.rstrip('.')}" if b.subject else "")
    clause = p.negation == "clause" and not p.negative
    no_text_word = clause and not text and p.text != "none"
    parts: list[tuple[int, str]] = [
        (10, _sentence(lead)),
        (6, _sentence(_join(rest, b.shape_language))),
        (8, _sentence(_colors_by_role(b))),
        (9, _sentence(_text_clause(b, text)) + (f" {p.text_tail}" if p.text_tail else "") if text else ""),
        (5, _sentence(_join(*positive))),
        (4, _sentence(b.composition)),
        (9, _sentence(_placement(b, transparent))),
        (3, _sentence("Bold, simple shapes that read clearly at small sizes")),
    ]
    if clause:
        parts.append((7 if no_text_word else 2, _no_clause(direct, no_text_word)))
    if not text and p.no_text and not no_text_word:
        parts.append((1, p.no_text))
    prompt = _fit(parts, p.max_words)
    negative = _join(p.negative_default, *direct) if p.negative else ""
    return prompt, negative, notes


def render_structured(b: Brief, p: ModelProfile, transparent: bool, extra_avoid: list[str]) -> tuple[str, str, list[str]]:
    text, notes = _text_plan(b, p)
    direct, positive = _exclusions(b, p, extra_avoid)
    mark = _mark(b, text)
    what = "an app icon" if b.format == "app icon" else f"{_article(mark).lower()} {mark}"
    head, rest = _style_head(b, p.lead)
    look = _join(head, rest, b.shape_language, *positive, sep="; ")
    bg = "transparent" if transparent else _background(b, False).removesuffix(" background")
    lines: list[tuple[int, str]] = [
        (10, f"Design {what}, an original mark."),
        (10, f"Subject: {b.subject.rstrip('.')}." if b.subject else ""),
        (8, f"Style: {look}. Clean vector-like shapes, a strong silhouette, balanced negative space."),
        (8, f"Colors: {_and(_palette_list(b))} only." if b.palette else ""),
        (6, f"Composition: {_join(b.composition, 'a single mark, centered, generous even padding, nothing else in frame', sep='; ')}."),
        (9, f"Background: {bg}." if transparent else f"Background: {bg}, filling the whole canvas."),
    ]
    if text:
        lines.append((9, f"Text: {_text_clause(b, text)}. {p.text_tail}".strip()))
    elif p.no_text:
        lines.append((7, f"Text: none. {p.no_text}"))
    if direct and not p.negative and p.negation == "clause":
        lines.append((5, f"Constraints: {_no_clause(direct, False)}"))
    lines.append((4, "It must read clearly at 32 px, so keep the shapes bold and the details few."))
    prompt = _fit(lines, p.max_words, sep="\n")
    negative = _join(p.negative_default, *direct) if p.negative else ""
    return prompt, negative, notes


def render_recraft(b: Brief, p: ModelProfile, transparent: bool, extra_avoid: list[str]) -> tuple[str, str, list[str]]:
    """Recraft V4: general to specific, then "flat colors only, no ..." (its own guide's wording)."""
    text, notes = _text_plan(b, p)
    direct, _ = _exclusions(b, p, extra_avoid)
    mark = _mark(b, text)
    head, rest = _style_head(b, p.lead)
    lead = f"{head} {mark}" + (f" of {b.subject.rstrip('.')}" if b.subject else "")
    flat = ["gradients", "shadows", "texture"] + [d for d in direct if d not in ("gradients", "shadows", "texture")]
    parts: list[tuple[int, str]] = [
        (10, _sentence(lead)),
        (7, _sentence(_join(rest, b.shape_language))),
        (8, _sentence(_colors_by_role(b))),
        (9, _sentence(_text_clause(b, text)) if text else ""),
        (8, _sentence("Flat colors only, one unified silhouette, no " + ", no ".join(flat[:5]))),
        (4, _sentence(b.composition)),
        (9, _sentence(_placement(b, transparent))),
        (5, "" if text else p.no_text),
    ]
    return _fit(parts, p.max_words), "", notes


def _short_subject(s: str, n: int = 12) -> str:
    s = re.sub(r"\b(a|an|the|that|which|whose|is|are)\b", " ", s, flags=re.I)
    return " ".join(s.replace(".", ",").split()[:n]).strip(", ")


def render_keywords(b: Brief, p: ModelProfile, transparent: bool, extra_avoid: list[str]) -> tuple[str, str, list[str]]:
    text, notes = _text_plan(b, p)
    direct, positive = _exclusions(b, p, extra_avoid)
    mark = "app icon" if b.format == "app icon" else "logo icon"
    style = [x.strip() for x in re.split(r"[,;]", b.style) if x.strip()]
    shape = [x.strip() for x in re.split(r"[,;]", b.shape_language) if x.strip()]
    # (priority, phrase), in prompt order. CLIP reads the front hardest, and
    # the budget keeps the highest priorities.
    bits: list[tuple[int, str]] = [
        (10, f"flat vector {mark}"),
        (10, _short_subject(b.subject, 9)),
        (8, _and(_palette_list(b, hex_ok=False))),
        (6, style[0] if style else ""),
        (5, shape[0] if shape else ""),
        (7, f'text "{text}"' if text else ""),
        *[(4, x) for x in positive[:2]],
        (9, "white background"),
        (7, "centered"),
        (6, "simple"),
        *[(3, x) for x in style[1:2]],
        *[(2, x) for x in shape[1:2]],
    ]
    seen: list[tuple[int, str]] = []
    for pr, x in bits:
        x = x.strip(" ,.")
        if x and x.lower() not in (t.lower() for _, t in seen):
            seen.append((pr, x))
    budget, chosen = p.max_words, set()
    for i in sorted(range(len(seen)), key=lambda i: -seen[i][0]):
        n = _words(seen[i][1])
        if n <= budget:
            chosen.add(i)
            budget -= n
    out = [t for i, (_, t) in enumerate(seen) if i in chosen]
    negative = _join(p.negative_default, *direct) if p.negative else ""
    return ", ".join(out), negative, notes


def _tagify(s: str) -> list[str]:
    s = _short_subject(s, 10).lower()
    return [t.strip() for t in re.split(r",| and | with ", s) if t.strip()]


def render_tags(b: Brief, p: ModelProfile, transparent: bool, extra_avoid: list[str]) -> tuple[str, str, list[str]]:
    text, notes = _text_plan(b, p)
    direct, _ = _exclusions(b, p, extra_avoid)
    tags = [
        *[t for t in p.quality_tags.split(", ") if t],
        "logo", "symbol", "flat color", "vector", "simple background", "white background",
        "no humans" if b.mark_type != "mascot" else "solo",
        *_tagify(b.subject),
        *[f"{c} theme" for c in _palette_list(b, hex_ok=False)[:2]],
        "limited palette",
        "centered",
    ]
    tags = list(dict.fromkeys(t for t in tags if t))
    out, count = [], 0
    for t in tags:
        if count + _words(t) > p.max_words:
            break
        out.append(t)
        count += _words(t)
    negative = _join(p.negative_default, *direct) if p.negative else ""
    return ", ".join(out), negative, notes


RENDERERS = {
    "natural": render_natural,
    "structured": render_structured,
    "recraft": render_recraft,
    "keywords": render_keywords,
    "tags": render_tags,
}


# ── Parameters ───────────────────────────────────────────────────────


def _size_rank(v: str) -> float:
    s = str(v).strip().upper()
    scale = 1000.0 if s.endswith("K") else 1.0
    try:
        return float(s.removesuffix("K")) * scale
    except ValueError:
        return 1e9


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def recraft_controls(b: Brief, transparent: bool) -> dict:
    """Recraft's `controls`: the palette and the background as RGB."""
    colors = [{"rgb": list(rgb)} for c in b.palette if (rgb := _rgb(c.hex))]
    controls: dict = {"colors": colors} if colors else {}
    if not transparent:
        bg = b.background.lower()
        rgb = (255, 255, 255) if _near_white(bg) else _rgb(bg)
        if rgb:
            controls["background_color"] = {"rgb": list(rgb)}
    return {"provider": {"options": {"recraft": {"controls": controls}}}} if controls else {}


def cloud_params(p: ModelProfile, info: dict | None, transparent: bool, b: Brief | None = None) -> dict:
    """The profile's params that this OpenRouter model actually accepts, plus provider passthrough."""
    info = info or {}
    supported = info.get("params") or {}
    out: dict = {}
    for k, v in p.params.items():
        if k == "provider":
            out = _merge(out, {"provider": v})
            continue
        spec = supported.get(k)
        if spec is None:
            continue
        values = spec.get("values") or []
        if v == "lowest":
            if values:
                out[k] = min(values, key=_size_rank)
            continue
        if values and v not in values:
            continue
        out[k] = v
    if transparent and info.get("supports_transparent"):
        out["background"] = "transparent"
    if p.color_controls and b is not None:
        out = _merge(out, recraft_controls(b, transparent))
    return out


def local_params(p: ModelProfile) -> dict:
    return {k: v for k, v in p.params.items() if k in ("steps", "guidance", "resolution")}


# ── Entry points ─────────────────────────────────────────────────────


def _is_local(model: str) -> bool:
    return model.startswith("local/")


def _can_clear(model: str, info: dict | None) -> bool:
    return not _is_local(model) and bool((info or {}).get("supports_transparent"))


def render(
    b: Brief | dict,
    model: str,
    info: dict | None = None,
    transparent: bool = False,
    extra_avoid: list[str] | None = None,
) -> Rendered:
    """Render one brief for one model. `info` is the OpenRouter catalog entry, if any."""
    if isinstance(b, dict):
        b = briefs.from_dict(b)
    p = profile_for(model)
    b = replace(b, subject=drawable(b.subject))
    clear = (transparent or b.wants_transparent) and _can_clear(model, info)
    prompt, negative, notes = RENDERERS[p.style](b, p, clear, list(extra_avoid or []))
    params = local_params(p) if _is_local(model) else cloud_params(p, info, clear, b)
    if b.wants_transparent and not clear:
        notes.append("This model has no transparent output, so it gets a plain background. Remove it afterwards.")
    return Rendered(model, p.id, p.name, p.style, prompt, negative if p.negative else "", params, notes)


FRAME = {
    "natural": "Flat vector logo mark, a single mark centered with generous padding on a plain white background.",
    "structured": "Make it a flat vector logo mark: a single mark, centered with generous padding on a plain white background.",
    "recraft": "Flat vector logo, flat colors only, centered on a plain white background.",
    "keywords": "flat vector logo icon, white background, centered, simple",
    "tags": "logo, symbol, flat color, simple background, white background",
}


def render_raw(text: str, model: str, info: dict | None = None, transparent: bool = False) -> Rendered:
    """A prompt the user typed without AI help: add the profile's logo framing unless it already has some."""
    p = profile_for(model)
    text = " ".join(text.split())
    framed = any(w in text.lower() for w in ("logo", "icon", "background", "emblem", "wordmark"))
    if p.style in ("keywords", "tags"):
        prompt = text if framed else f"{text}, {FRAME[p.style]}"
        if p.style == "tags" and p.quality_tags:
            prompt = f"{p.quality_tags}, {prompt}"
    else:
        prompt = text if framed else f"{_sentence(text)} {FRAME[p.style]}"
    clear = transparent and _can_clear(model, info)
    params = local_params(p) if _is_local(model) else cloud_params(p, info, clear)
    return Rendered(model, p.id, p.name, p.style, prompt, p.negative_default if p.negative else "", params, [])


def render_direction(
    direction: dict,
    model: str,
    info: dict | None = None,
    transparent: bool = False,
    extra_avoid: list[str] | None = None,
) -> Rendered:
    """A direction from the designer, the Generate page or an edit.

    - an override for this model (the user edited the rendered prompt) is sent as written,
    - `verbatim` sends `prompt` as written to every model,
    - a direction with a brief is rendered for the model,
    - anything else is a plain prompt and gets light framing.
    """
    p = profile_for(model)
    override = (direction.get("overrides") or {}).get(model)
    hand = override if override and str(override.get("prompt") or "").strip() else (direction if direction.get("verbatim") else None)
    if hand is not None:
        b = briefs.from_dict(direction["brief"]) if briefs.is_brief(direction.get("brief")) else None
        clear = transparent and _can_clear(model, info)
        params = local_params(p) if _is_local(model) else cloud_params(p, info, clear, b)
        if isinstance(hand.get("params"), dict):  # explicit params replace the profile's
            params = dict(hand["params"])
        neg = str(hand.get("negative_prompt") or "") if p.negative else ""
        return Rendered(model, p.id, p.name, p.style, str(hand.get("prompt") or "").strip(), neg, params, ["Edited by hand"])
    if briefs.is_brief(direction.get("brief")):
        return render(direction["brief"], model, info, transparent, extra_avoid)
    return render_raw(str(direction.get("prompt") or ""), model, info, transparent)
