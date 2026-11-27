# Prompting image models for logos and icons (state as of 2026-11-27)

Scope: the model families logo-gen reaches through the OpenRouter Images API (71 image model ids in the live catalog on 2026-11-27, including `:free` and `-preview` aliases), plus the local models (FLUX.2 klein 4B, FLUX.1 schnell, Z-Image Turbo, SD-Turbo) and a few families not offered today (Ideogram, SDXL, booru-tag SDXL fine-tunes).

Rule used throughout: if a fact was not found in an official doc, model card, provider guide or the live OpenRouter catalog, it is marked "not documented" or attributed to the third-party source it came from. Parameter facts for OpenRouter come from `GET https://openrouter.ai/api/v1/images/models` and the per-model `/endpoints` records, fetched on 2026-11-27.

---

## 1. Summary

- Almost every model offered in 2026 uses an LLM or VLM text encoder (Qwen3, Mistral 3, Gemini, GPT) or is itself a multimodal LLM. For these, write a short brief in plain sentences: what the mark is, its shape language, palette, background, and what it is for. Keyword soup and quality tags ("masterpiece, 8k, trending") do not help; BFL says to leave quality tags out.
- The main idea goes first. BFL, Recraft, Ideogram and Seedream (Runware guide) all say early words carry more weight.
- Exact text goes in double quotes, with font traits (weight, case, serif or sans) and placement. All the strong text renderers document this (OpenAI, Google, BFL, Recraft, Ideogram, Qwen, Seedream, MAI).
- Hex colors work on FLUX.2 / FLUX 3 when each code is tied to a named element ("the mark in color #1F6FEB"). Recraft takes colors as RGB through `controls.colors`, which OpenRouter passes through.
- Negative prompts: no OpenRouter model exposes a `negative_prompt` parameter. Distilled local models (FLUX.2 klein distilled, FLUX.1 schnell, Z-Image Turbo, SD-Turbo) run at CFG 0 or 1, where negatives do nothing. So exclusions have to go into the positive prompt. With LLM encoders a short exclusion clause ("no gradients, no text") works reasonably, and OpenAI, Recraft and Seedream examples use them. With CLIP-only models, negated words tend to pull the concept in.
- Transparency is a parameter, not a prompt word, on models that support it (OpenAI gpt-image-1 / 1-mini / 2.5 / 2.6 / gpt-5-image, Seedream 5.0 flash / pro / 5.5, Riverflow 2.5+, MAI 2.7). Elsewhere, ask for a flat solid background (usually white) and remove it in post. Ideogram's docs say writing "transparent PNG" in the prompt does not replace the setting.
- For SVG, use Recraft `*-vector` models (the only OpenRouter models whose `output_format` includes `svg`). OpenAI documents SVG output for gpt-image-2.5 / 2.6, but OpenRouter's catalog does not list `svg` for them.

---

## 2. Per-family table

| Family | Prompt style | Text encoder / useful limit | Ideal length | Negative prompt | Guidance / steps | Text in image | Background / transparency | Logo do / don't |
|---|---|---|---|---|---|---|---|---|
| Recraft V4 / V4.1 (raster, vector, pro, utility, flash) | Structured brief, global to local | Not documented; prompt cap 10,000 chars | Short = interpretive; 30-80 words for control (Recraft guide, no number given) | No (V2/V3 only) | None exposed | Good; quote the text. V4.1 Flash V2 "better with text" | `controls.background_color` (RGB) via passthrough; SVG from `*-vector` | Do: graphic type, shape logic, color system, line discipline, "flat colors only". Don't: texture/material words, stacked adjectives |
| Recraft V3 | Short descriptive + curated `style` | Not documented; 1,000 chars | Under 1,000 chars | Yes on Recraft API, but not passable via OpenRouter | `controls.artistic_level` 0-5 | Good; `text_layout` word boxes | `controls.background_color`; `controls.no_text` | Do: pass `style` (e.g. "Vintage Emblem") via passthrough. Don't: expect `logo_raster` to work (reported 400) |
| FLUX.2 pro / max / flex | Prose or JSON; Subject + Action + Style + Context | FLUX.2 [dev] uses Mistral 3, `max_sequence_length` 512 (diffusers); API encoder not documented | 30-80 words (BFL) | No | flex: guidance 1.5-10 (4.5 default), steps up to 50; passthrough on pro/max/flex | Strong; quotes, font, placement, hex | No transparency param; ask for solid background | Do: hex tied to element, front-load. Don't: "no X" phrasing, quality tags |
| FLUX.2 klein 4B (API and local) | Prose, "like a novelist" | Qwen3 (diffusers), 512 tokens | Under ~100 words (fal) | No (distilled, guidance ignored) | Local: 4 steps, guidance 1.0 | Usable for short words; model card says it "may be inaccurate" | Solid background, post-remove | Do: one-sentence mark description + style line. Don't: long lists of exclusions |
| FLUX 3 Image / FLUX 3 klein | Prose; optional layout boxes | Not documented | Start with one sentence, add as needed | No field; unknown fields 422 | No guidance field; klein: `steps` passthrough; `grounding` (web search) default true | Strong; quote exact case and punctuation | Not documented | Do: set `grounding: false` for invented brands. Don't: rely on "vector" (not in BFL vocabulary) |
| FLUX.1 schnell (local) | Short prose or keywords | CLIP-L (77 tokens, pooled) + T5-XXL, max 256 | 20-60 words | No (guidance 0) | 4 steps, guidance 0, `max_sequence_length` 256 | Fair for 1-2 short words | Solid background | Do: put key nouns early. Don't: negation words |
| Seedream 4.5 / 5.0 / 5.5 | Brief in clauses: deliverable, subject, setting, style, exclusions | Not documented; Runware gives a 2,000-char budget for 4.5 | Up to ~600 words claimed (attributed to ByteDance by Runware/APIMart); shorter is better | Not via OpenRouter (no passthrough) | None exposed | Strong (5.0 Pro: dense text, many scripts) | `background` transparent on 5.0 flash, 5.0 pro, 5.5 (not 4.5, not 5.0 lite) | Do: quote copy, end with "no other text". Don't: adjective lists |
| Gemini image (Nano Banana family) | Narrative paragraph | Gemini LLM; limit not documented | 1-4 sentences for a logo | No; "semantic negative prompts" (describe what you want) | `resolution` 1K/2K/4K (512 on 3.1 flash); thinking level not exposed via OpenRouter | Strong on 3 Pro / 3.1 / 2.1; small text and spelling "may not work perfectly" | No `background` param on OpenRouter; ask for plain white | Do: state purpose, font style, colors, shape. Don't: keyword lists, "no cars" style negation |
| OpenAI GPT Image (1, 1-mini, 2, 2.5/2.6 flare/sunburst, gpt-5-image) | Labeled sections: scene, subject, details, constraints | GPT; prompt cap 32,000 chars | Logo example is ~100 words | No param; plain-language exclusions work | `quality` low/medium/high (+ xhigh/max on 2.5/2.6); 2.6 `complexity` | Strong; quotes, spell unusual names letter by letter | `background: transparent` + png/webp on 1, 1-mini, 2.5, 2.6, gpt-5-image; not on gpt-image-2 or gpt-5.4-image-2 via OpenRouter | Do: say transparent in prompt and param, "generous padding", "reads at small sizes". Don't: describe a backdrop when you want alpha |
| Qwen Image 2.1 / 3 | Layered prose: subject, composition, style | Qwen LLM; 2.0 series 1,300 tokens (Alibaba); 3.0 not documented | 30-100 words | Alibaba API yes (500 chars); not passable via OpenRouter | None exposed; prompt rewriting default on in Alibaba API, not controllable via OpenRouter | Very strong (EN/ZH, small text) | Solid background | Do: quote text; add "no text" when you want none. Don't: assume seed reproducibility (rewriter) |
| Z-Image Turbo (local) | Prose, objective description | Qwen3-4B, `max_sequence_length` 512 | 40-150 words (community guides disagree) | No at CFG 0/1 | 8-9 steps, guidance 0 | Strong EN/ZH bilingual | Solid background | Do: describe plainly, quote text. Don't: rely on negative prompt |
| SD-Turbo (local) | Short keywords | CLIP (OpenCLIP ViT-H), 77 tokens | 10-25 words | No (guidance 0) | 1-4 steps, guidance 0, 512 px | Cannot render legible text | White background, post-remove | Do: "flat vector logo, simple geometric fox, white background". Don't: ask for words; don't write "no text" |
| Ideogram 3.0 / 4.x (not offered) | Sentences; 4.x accepts JSON | Not documented | 1-3 sentences, add detail where it misses | Yes (`negative_prompt`; prompt wins on conflict) | `magic_prompt` on/off/auto; `style_type` incl. DESIGN | Best-in-class claim; quote text near the start | Separate setting; prompt words do not make it transparent | Do: magic prompt off for exact layouts. Don't: `::1` weights or `--ar` flags (read as words) |
| SDXL base (not offered) | Comma keywords + short phrases | CLIP-L + OpenCLIP bigG, 77 tokens each (truncated) | Under ~60 words | Yes, CFG 5-8 | 25-40 steps | Poor | White background | Do: front-load; use negative field. Don't: write past 77 tokens |
| Illustrious / NoobAI / Pony (not offered) | Danbooru tags | CLIP (SDXL), 77-token chunks | 15-40 tags | Yes, important for Illustrious | CFG ~4-7 | Poor | "simple background, white background" tags | Do: model-specific quality tags. Don't: Pony `score_*` on Illustrious |
| Krea 2 | Short prose + style refs/moodboards | Not documented | Short | No | `creativity`, `styles`, `moodboards` passthrough | Not documented | Not documented | Do: use style references for brand consistency |
| MAI Image 2.5-2.7 | Prose | Foundry lists 32,000-token text input (MAI-Image-2) | Not documented | No | None | Strong claim; quote exact text | `background` transparent on 2.7 / 2.7 flash only | Do: quote text, specify font and background |
| Meta Muse Image | Explicit instructions; agentic | Not documented | Not documented | No | None | Varies run to run (Meta) | Not exposed | Do: keep baked-in text short; re-run |
| Sourceful Riverflow 2 / 2.5 / 2.6 | Prose instruction | Not documented | Not documented | No | `font_inputs` passthrough (custom font + exact text) | Strong, especially with font_inputs | `background` transparent on 2.5 / 2.6 | Do: pass brand font via `font_inputs` |
| Tencent Hy Image 3.5 | Long prose OK | Not documented for 3.5 (3.0 is an LLM-based model) | Long OK | No | None | Good; quote each string | Not exposed | Do: break text into separate quoted strings |
| xAI Grok Imagine 2.0 | Prose | Not documented | Not documented | No | `quality` low/medium/auto (auto = low for generation) | Not documented | Not exposed | Do: pin `quality: medium` for finals |

---

## 3. Families in detail

### 3.1 Recraft

**Models and API surface.** OpenRouter offers recraft-v3, v4, v4-pro, v4-vector, v4-pro-vector, v4.1, v4.1-pro, v4.1-vector, v4.1-pro-vector, v4.1-utility, v4.1-utility-pro, v4.1-flash, v4.1-flash-v2 (and `:free`), and v4-styles / styles-pro / styles-vector / styles-pro-vector. The `*-vector` endpoints list `output_format: ["svg"]`. None of the Recraft endpoints list `seed`, `background`, `quality` or `resolution`. `n` is 1-6.

**Passthrough (live endpoint records, 2026-11-27):**
- v3, v4*, v4.1 (base, pro, vector, pro-vector, utility, utility-pro): `style`, `controls`, `text_layout`.
- v4.1-flash, v4.1-flash-v2: `controls` only.
- v4-styles family: `style_id`, `style_match`, `controls`, `random_seed`, and `input_references` 1-10 are required (style reference images).

**What Recraft itself supports per model** (Recraft API reference):
- Curated `style`: V2 and V3 only. The V4.1 page says: "Not supported by V4.1: `negative_prompt`, `style` (curated styles), `text_layout`". The styles page says that for V4, V4.1 and Utility models, curated styles are not supported. So OpenRouter listing `style` and `text_layout` as allowed passthrough keys for V4 / V4.1 does not mean Recraft will honor them. Expect them to be ignored or rejected (untested).
- `controls.colors` (up to 10 colors, each `{"rgb":[r,g,b]}`, optional `weight`) and `controls.background_color`: all models. These are the main thing worth passing to V4 / V4.1.
- `controls.artistic_level` (0 "static, clean" to 5 "dynamic, eccentric") and `controls.no_text`: V3 only.
- `negative_prompt`: V2 / V3 only, and not in OpenRouter's passthrough list, so not reachable.
- `text_layout` (single words with quadrilateral `bbox` in 0-1 coordinates): V3 only.
- Prompt length: 10,000 chars for V4 / V4.1 / Utility / Flash V2 / Styles; 1,000 for V2 / V3.
- Custom styles (`style_id`, style references) work on V2, V3, V4, V4.1, not V4.1 Flash V2. `style_match` is `flexible` (default) or `precise` on V4 / V4.1.

**V3 curated styles relevant to logos** (current names on the styles page): raster V3 has Emblem styles `Prestige Emblem`, `Pop Graphic`, `Stamp`, `Punk Graphic`, `Vintage Emblem`, plus illustration styles. V3 Vector has `Vector art`, `Line art`, `Linocut`, `Bold stroke`, `Emotional flat`, `Roundish flat`, `Marker outline`, `Thin`, `Vivid shapes`, and others. V2 Vector has the `Icon` group (`Icon`, `Outline`, `Pictogram`, `Colored outline`, `Colored shape`, `Gradient outline`, `Gradient shape`, `Broken line`, `Offset fill`, `Doodle`, `Offset doodle`). No current style name contains "logo". The older enum form (`style: vector_illustration` / `icon` / `logo_raster` plus `substyle`) still shows up in wrappers such as ComfyUI. A Recraft bug report (Nov 2025) says `recraftv3` + `logo_raster` returned HTTP 400 even though the docs listed it. The current docs do not document `substyle` at all. OpenRouter's `recraft/recraft-v3` is raster (no `output_format` listed), so vector-only style names like `Vector art` probably do not apply there. That is an inference, not documented.

**Prompting guidance (Recraft "Prompting with Recraft V4").** Short prompts "activate interpretive behavior", structured prompts give control, and "Elements placed earlier in the prompt receive higher priority". For vector work, define "graphic type, shape logic, color system, line discipline, layout structure, and constraints", and "Avoid texture or material-focused language". The guide's logo examples use explicit exclusions written as instructions: "Flat colors only - no gradients, shadows, or texture"; "Centered, compact composition forming one unified silhouette with strong legibility". Icon set example: "Ultra-minimal bold silhouette style, pure black solid shapes on a white background", "No gradients, no textures, no outlines - only simple geometric filled forms." Text: put all required text in quotes and "Define the typographic system - not just the phrase." The V4.1 page's vector example is "minimal fox logo, flat, two colors". `_utility` variants give "clean, predictable images with flat lighting and simple composition", which suits icons.

Sources: https://www.recraft.ai/docs/api-reference/endpoints.md, https://www.recraft.ai/docs/api-reference/models/recraft-v4-1.md, https://www.recraft.ai/docs/api-reference/models/recraft-v3.md, https://www.recraft.ai/docs/api-reference/styles.md, https://www.recraft.ai/docs/api-reference/appendix.md, https://www.recraft.ai/docs/api-reference/models/overview.md, https://www.recraft.ai/docs/prompt-engineering-guide/prompting-with-recraft-v4.md, https://feedback.recraft.ai/bug-reports/p/bug-report-invalid-model-style-combination-error

### 3.2 Black Forest Labs (FLUX.2, FLUX 3, FLUX.1 schnell)

**FLUX.2 [pro] / [max] / [flex] (BFL FLUX.2 prompting guide).**
- Framework "Subject + Action + Style + Context". FLUX.2 "pays more attention to what comes first".
- Length: 10-30 words for exploration, 30-80 "usually ideal", 80+ for complex scenes.
- "FLUX.2 does not support negative prompts." Describe the wanted state instead ("sharp focus throughout" rather than "no blur").
- JSON prompts are optional. Fields shown: `scene`, `subjects[]` (`description`, `position`, `action`), `style`, `color_palette`, `lighting`, `mood`, `background`, `composition`, `camera`. JSON can be flattened to prose.
- Hex: put "color" or "hex" before the code and tie it to one object ("The logo text 'ACME' in color #FF5733"). Vague "use #FF0000 somewhere" gives inconsistent results.
- Text: quotes plus placement, font style, size and color. The unified reference says pro / max have "Excellent typography" and advises keeping strings short.
- Guidance / steps are documented for [flex] only (guidance 4.5, range 1.5-10; steps 50, max 50). OpenRouter passthrough: pro and max accept `steps`, `guidance`, `safety_tolerance`, `disable_pup`; flex accepts `steps`, `guidance`, `safety_tolerance`, `prompt_upsampling`. Shape: `{"provider":{"options":{"black-forest-labs":{"steps":40,"guidance":3}}}}`. Flex and max endpoints report `provider_slug` `black-forest-labs/us-3`; whether the key must be that exact slug is not documented.
- Text encoder: FLUX.2 [dev] uses `Mistral3ForConditionalGeneration` with `max_sequence_length` 512 in diffusers. The encoder behind the pro / max / flex API is not documented.

**FLUX.2 [klein] 4B.** Diffusers `Flux2KleinPipeline` uses `Qwen3ForCausalLM`, `max_sequence_length` 512. For step-distilled checkpoints, "`guidance_scale` is ignored". The model card example uses 4 steps, guidance 1.0. The card says text "may be inaccurate or subject to distortion" and "Prompt following is heavily influenced by the prompting style." The BFL unified reference says for klein: "What you write is exactly what the model receives" (no upsampling), write prose, and end with `Style: [style]. Mood: [mood].` A fal guide says prompts over 100 words "create confusion" and recommends targeted negatives. That negatives advice does not apply to the distilled model at guidance 1.0, which has no CFG pass. OpenRouter passthrough: `steps`, `guidance`, `safety_tolerance`.

**FLUX 3 Image / FLUX 3 Image klein.** The API schema has `prompt`, `images` (up to 10), `width` / `height` or `resolution` (`768sq`, `1k`, `1.5k`, `2k`, `4k`), `aspect_ratio`, `grounding` (default true: "FLUX 3 searches the web to get real things right"), `version` (`latest` only), `safety_tolerance` 0-7. `additionalProperties: false`, so `negative_prompt` or `guidance` give a 422. No prompt length limit is documented. Text rules: quote each string "with the capitalization and punctuation you want", state where it sits, describe weight, case, color; one layout box per text block for posters. Hex: "Tie each hex code to one named object." Style: "Name the style specifically and describe its visual characteristics." The doc's style examples include "Flat-color digital illustration of a minimalist graphic design presented against a solid black background." The BFL docs do not use the word "vector". BFL also says to omit quality tags such as "masterpiece". OpenRouter passthrough: flux-3-image `safety_tolerance`, `grounding`, `version`; flux-3-image-klein `safety_tolerance`, `steps`. For invented brand names, `grounding: false` avoids web lookups. That is a reasonable inference; BFL describes the flag as faster and prompt-only, and says nothing logo-specific. Transparency: not documented.

**FLUX.1 schnell (local).** Diffusers: "`guidance_scale` needs to be 0", "`max_sequence_length` cannot be more than 256", 4 steps in examples. Encoders: CLIP ViT-L/14 (77 tokens; Flux uses its pooled embedding) and T5-XXL (the token sequence). `negative_prompt` only works with `true_cfg_scale > 1`, which is untested for schnell. Practical rule: keep the important nouns in the first ~60 words so the CLIP pooled vector and the start of the T5 sequence both carry them.

Sources: https://docs.bfl.ai/guides/prompting_guide_flux2.md, https://docs.bfl.ai/guides/prompting_unified_reference.md, https://docs.bfl.ai/guides/prompting_unified_style, https://docs.bfl.ai/guides/prompting_unified_building, https://docs.bfl.ml/api-reference/utility/generate-an-image-with-flux-3.md, https://docs.bfl.ml/flux_3/flux3_image_overview.md, https://huggingface.co/black-forest-labs/FLUX.2-klein-4B, https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2, https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux, https://fal.ai/learn/devs/flux-2-klein-prompt-guide

### 3.3 ByteDance Seedream (4.5, 5.0 flash / lite / pro, 5.5)

- The official BytePlus ModelArk "Seedream 4.0-4.5 prompt guide" exists, but its body did not load through fetch (JS-rendered). Its contents are not verified here.
- Runware's Seedream 5.5 guide: write the prompt "as a brief", one decision per clause, in the order deliverable, subject, setting, lighting, camera, style, exclusions. It attributes "keeping it under about 600 English words" to ByteDance. Quote each line of copy, give it position, type style and color, and end with "no other text". Exclusions are written as clauses ("no text, no logos").
- Runware's Seedream 4.5 guide: "Seedream weights the start of the prompt most heavily"; 2,000-character prompt budget; quote literal text and keep strings short.
- ByteDance's Seedream 5.0 Pro post claims dense text rendering, multi-script typography, hex color input, and layer separation with transparency. It admits "room to improve in finer-grained text rendering".
- Seedream 5.5 launched 2026-11-09 (press coverage). Coverage says it produces native RGBA transparent output.
- OpenRouter: `background` (auto / transparent / opaque) on seedream-5-0-flash, 5-0-pro, 5-5; not on 4.5 or 5-0-lite. `output_format` png / jpeg. `resolution` 1K / 1.5K / 2K (5.0 flash, pro, 5.5), 2K / 4K (5.0 lite), 1K / 2K / 4K (4.5). No negative prompt. Passthrough is empty except 5.5 (`layer_decomposition`, `layer_decomposition_options`).

Sources: https://runware.ai/docs/models/bytedance-seedream-5-5/guides/prompting, https://runware.ai/docs/models/bytedance-seedream-4-5/guides/prompting, https://seed.bytedance.com/en/blog/beyond-generation-it-understands-design-introducing-seedream-5-0-pro, https://docs.byteplus.com/en/docs/ModelArk/seedream-4-0-5-0-prompt-guide

### 3.4 Google Gemini image (Nano Banana)

- Ids per Google: `gemini-nano-banana-2.1` (Nano Banana 2.1), `gemini-3.1-flash-lite-image` (Nano Banana 2 Lite), `gemini-3.1-flash-image` (Nano Banana 2), `gemini-3-pro-image` (Nano Banana Pro), `gemini-2.5-flash-image` (legacy).
- Core advice from Google's Gemini 2.5 Flash Image post: "Describe the scene, don't just list keywords." Google's best-practices list: be hyper-specific, give context and intent, iterate, use step-by-step instructions, and use "Semantic Negative Prompts": "Instead of saying "no cars," describe the intended scene positively."
- Logo example in Google's docs starts: "Create a modern, minimalist logo for a coffee shop called 'The Daily Grind'". For text: "Be clear about the text, the font style (descriptively), and the overall design." Google recommends Gemini 3 Pro Image for professional asset production, and notes: "Gemini works best if you first generate the text and then ask for an image with the text."
- Limitations (Nano Banana Pro tips): "Rendering small text, fine details, and producing accurate spellings may not work perfectly."
- Resolution: 1K default on Gemini 3 models; 2K and 4K available; 512 only on 3.1 Flash Image; values need an uppercase K. Thinking cannot be disabled on Gemini 3 image models; levels are not exposed through OpenRouter.
- Transparency: not mentioned in any Google doc fetched; OpenRouter lists no `background` param for any Gemini image model. Ask for a plain white or flat single-color background and remove it in post.
- Every output carries a SynthID watermark (invisible).
- OpenRouter passthrough: `cachedContent` only.

Sources: https://aistudio.google.com/docs/image-generation (formerly ai.google.dev/gemini-api/docs/image-generation), https://developers.googleblog.com/en/how-to-prompt-gemini-2-5-flash-image-generation-for-the-best-results/, https://blog.google/products/gemini/prompting-tips-nano-banana-pro/, https://deepmind.google/models/gemini-image/prompt-guide/

### 3.5 OpenAI GPT Image

- Models on OpenRouter: gpt-image-1, gpt-image-1-mini, gpt-image-2, gpt-image-2.5-flare / -sunburst, gpt-image-2.6-flare / -sunburst, gpt-5-image, gpt-5-image-mini, gpt-5.4-image-2. OpenAI positions Flare as the fast tier and Sunburst as the detail tier. 2.6 adds `complexity` (low / medium / high, default high; planning effort, no separate fee).
- Prompt structure (OpenAI image prompting guide): define subject, intended use, composition and constraints; for complex requests "organize the prompt as scene, subject, details, and constraints, using labeled sections". No special syntax required.
- Logo example (verbatim fragments): "Create an original, non-infringing logo for a company called Field & Flour, a local bakery." "Use clean, vector-like shapes, a strong silhouette, and balanced negative space." "Favor simplicity over detail so it reads clearly at small and large sizes." "Flat design, minimal strokes, no gradients unless essential." "Fully transparent background. Deliver a single centered logo with generous padding, clean alpha edges," "and no solid backdrop, scenery, checkerboard, or watermark." Used with `quality="medium"`, `background="transparent"`, `output_format="png"`.
- Text: "Put required wording in quotes and describe its position and typography." Spell unusual names letter by letter, state how many times the text should appear, ask for no extra text, and use medium or high quality for small text.
- Transparency: request it in the prompt and the parameter; PNG or WebP; a drawn checkerboard is not transparency; if the prompt describes a backdrop, the model may draw it. On OpenRouter, `background: transparent` is listed for gpt-image-1, 1-mini, 2.5-*, 2.6-*, gpt-5-image, gpt-5-image-mini. It is not listed for gpt-image-2 or gpt-5.4-image-2 (auto / opaque only), even though OpenAI announced a transparent preview for gpt-image-2.
- Quality: low / medium / high / auto on older models; 2.5 and 2.6 add xhigh and max. Cost scales with output tokens. OpenAI per-image list prices for 1024x1024 low / medium / high: gpt-image-2 $0.006 / $0.053 / $0.211; gpt-image-1 $0.011 / $0.042 / $0.167; gpt-image-1-mini $0.005 / $0.011 / $0.036. Use xhigh / max "only when they fix an unmet requirement".
- Prompt cap: 32,000 characters (API reference).
- SVG and GIF output are documented by OpenAI for 2.5 / 2.6. OpenRouter's catalog lists png / jpeg / webp (plus gif on 2.6), not svg.
- OpenRouter passthrough: `moderation`; 2.6 also `complexity`.

Sources: https://developers.openai.com/api/docs/guides/image-prompting.md, https://developers.openai.com/api/docs/guides/image-generation, https://developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide.md, https://developers.openai.com/api/reference/resources/images/methods/generate.md, https://community.openai.com/t/transparent-backgrounds-are-now-available-in-preview-for-gpt-image-2-in-the-api/1391541

### 3.6 Other OpenRouter families (brief)

- **Qwen Image 2.1 (turbo, pro) and 3 (base, pro).** OpenRouter descriptions claim precise text "as small as 10px". Alibaba's API reference (2.0 series): prompt max 1,300 tokens, `negative_prompt` up to 500 chars, `prompt_extend` (LLM rewrite) on by default. The 3.0 limits are not in Alibaba's English reference. Runware's 3.0 guide sets `promptExtend: false` for reproducibility (the rewrite is not seeded), and recommends "no text" or "no readable text" when lettering is unwanted because the model "renders lettering readily". OpenRouter passthrough is empty, so neither negatives nor `prompt_extend` can be set, and whether OpenRouter leaves the rewrite on is not documented. Sources: https://www.alibabacloud.com/help/en/model-studio/qwen-image-api, https://runware.ai/docs/models/alibaba-qwen-image-3-0/guides/prompting.md
- **Krea 2 (large, medium, medium-turbo).** Krea says the model is meant to be "more explorative than deterministic". Steering goes through style references and moodboards. OpenRouter passthrough: `image_style_references`, `moodboards`, `styles`, `creativity`, `intensity`, `complexity`, `movement`, `strength`. Resolution 1K only. OpenRouter describes Medium Turbo as aimed at "graphic design exploration". Text ability: not documented. Source: https://www.krea.ai/docs/user-guide/features/krea-2, https://fal.ai/learn/tools/krea-2-prompting-guide
- **Microsoft MAI Image 2.5 / 2.6 / 2.7 (+ flash).** Foundry lists 32,000-token text input for MAI-Image-2. Microsoft's sample notebook says to quote the exact text. OpenRouter lists `background` transparent for 2.7 and 2.7-flash only; passthrough `web_grounding` on 2.6 / 2.7. Sources: https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-mai, https://microsoft-foundry.github.io/forgebook/notebook/mai-image-2
- **Meta Muse Image.** Agentic: plans and can call tools before rendering. Meta says exact text rendering varies run to run, so keep baked-in text short and re-run. No passthrough on OpenRouter. Source: https://developer.meta.com/ai/models/muse-image/, https://runware.ai/docs/models/meta-muse-image/guides/text-in-images
- **Sourceful Riverflow 2 / 2.5 / 2.6.** `font_inputs` passthrough (custom font file plus exact text, max 2 fonts, about $0.03 per font on OpenRouter's V2 page). Replicate's readme: state the exact text, casing, placement and a legibility requirement, and repeat the text in the font input. `background` transparent on 2.5 / 2.6. Sources: https://openrouter.ai/sourceful/riverflow-v2-pro, https://replicate.com/sourceful/riverflow-2.0-pro/readme
- **Tencent Hy Image 3.5 preview.** Docs for 3.5 not found. For HunyuanImage 3.0, Tencent's card says it handles "very long text inputs" and points to a Prompt Handbook; the handbook advises splitting long text into several quoted strings. No passthrough on OpenRouter. Source: https://huggingface.co/tencent/HunyuanImage-3.0
- **xAI Grok Imagine Image 2.0.** `quality` low / medium / auto ("Auto currently uses `low` for image generation"), `resolution` 1k / 2k. Text rendering, negatives and transparency are not documented. Source: https://docs.x.ai/developers/model-capabilities/images/generation.md
- **MiniMax H3 Image** (in the catalog, not in the brief): passthrough `disable_prompt_enhanced`. No prompting docs checked.

### 3.7 Local: Z-Image Turbo (Tongyi-MAI, 6B)

- Text encoder: Qwen3-4B (paper: "we employ the lightweight Qwen3-4B ... as the text encoder" for "bilingual proficiency"). Diffusers `ZImagePipeline` default `max_sequence_length` 512.
- Model card: 9 steps ("This actually results in 8 DiT forwards"), `guidance_scale=0.0` ("Guidance should be 0 for the Turbo models"); CFG is marked unsupported for Turbo. Diffusers: `negative_prompt` is "Ignored when not using guidance". So no negatives at 0 or 1.
- Text: "excels at accurately rendering complex Chinese and English text"; the paper reports 0.935 / 0.936 on LongText-Bench EN / ZH.
- Prompt style: training captions used "a plain and objective linguistic style". An official Prompt Enhancer exists (a VLM with a system prompt). Community guides disagree on length (one says hundreds of words are fine, another says long prompts add noise). Write 40-150 words of plain visual description.
- Sources: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo, https://huggingface.co/docs/diffusers/main/en/api/pipelines/z_image, https://arxiv.org/html/2511.22699v2

### 3.8 Local: SD-Turbo

- Model card: 512x512 preferred; "A single step is enough"; "SD-Turbo does not make use of `guidance_scale` or `negative_prompt`, we disable it with `guidance_scale=0.0`"; "The model cannot render legible text."
- Encoder: SD 2.1 base uses OpenCLIP ViT-H, 77 tokens. The SD-Turbo card does not state this; it is inferred from SD-Turbo being a distillation of SD 2.1.
- Use 10-25 words of comma-separated visual keywords. Never ask for text. Don't write "no text" either: with no CFG and a CLIP encoder, the word "text" is just another concept to attend to.
- Source: https://huggingface.co/stabilityai/sd-turbo

### 3.9 Not offered now: Ideogram, SDXL, booru-tag models

**Ideogram (3.0, 4.0, 4.5, 4.6).** The prompting guide says: write in sentences; "Words near the start of the prompt tend to carry more weight"; "Say what you want, not what you don't"; "Put any text you want in the image in quotation marks, near the start." Weights like `::1` or `(important)` and flags like `--ar` "are read as ordinary words". 4.x accepts plain language or JSON. Aspect ratio, seed and transparency belong in settings; writing "transparent PNG" in the prompt does not replace them. Magic Prompt rewrites short prompts, and it reruns each call when on or auto, so a fixed seed does not reproduce. Turn it off for layouts. API: `style_type` (AUTO, GENERAL, REALISTIC, DESIGN, FICTION), `negative_prompt` ("Descriptions in the prompt take precedence"), `color_palette`, `style_codes`, `rendering_speed`. The v1 / v3 generate endpoint is marked legacy in favor of `POST /v2/image/generate/ideogram-3`. Sources: https://docs.ideogram.ai/using-ideogram/getting-started/prompting-guide/in-a-nutshell, https://developer.ideogram.ai/api-reference/api-reference/generate-v3, https://runware.ai/docs/models/ideogram-4-6/guides/text-and-design.md

**SDXL.** Two CLIP encoders (CLIP-L and OpenCLIP bigG), 77 tokens each. Diffusers truncates past 77 unless you use Compel, prompt embeddings, or a long-prompt community pipeline. AUTOMATIC1111 and ComfyUI split the prompt into 75-token chunks and concatenate them, which is not the same as the model reading a longer sentence. Negative prompts work because SDXL runs real CFG (5-8 typical). Keyword prompts and `(word:1.2)` weighting are front-end conventions (A1111 / Comfy / Compel), not model features. Sources: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/60, https://arxiv.org/pdf/2403.15378

**Illustrious, NoobAI, Pony (SDXL fine-tunes).** Prompted with Danbooru tags (NoobAI also e621). NoobAI's card maps popularity percentiles to quality tags (masterpiece above the 95th, best quality 85-95th, good quality 60-85th) and lists negatives such as "worst quality, old, early, low quality". Pony V6 uses `score_9, score_8_up, score_7_up` (the `_up` form came from a training mistake, according to community guides) plus `source_*` and `rating_*` tags. Community guides say Illustrious does not understand Pony score tags and is "very sensitive to negative prompts". For logos these models are a poor fit. Tags such as `logo, simple background, white background, flat color, no humans, symbol` exist on Danbooru, but the training data is anime illustration. Sources: https://huggingface.co/Laxhar/noobai-XL-1.0, https://civitai.com/models/257749/pony-diffusion-v6-xl, https://stable-diffusion-art.com/pony-diffusion-prompt-tags/, https://civitai.com/articles/23210

---

## 4. Words that help or hurt, and how negation behaves

### 4.1 Common logo words

| Word / phrase | Effect (evidence level) |
|---|---|
| "logo" | Sets the genre on every model. On LLM-encoder models it also pulls in "brand presentation" framing, which can produce mockups (business cards, signage). Add "a single mark, centered, nothing else in frame". OpenAI and Google examples both lead with "logo". (Documented examples; mockup drift is a community observation.) |
| "vector", "vector-like" | Recraft and OpenAI use it ("clean, vector-like shapes"). BFL's docs never use it; they say "flat-color digital illustration". On raster models it signals clean edges and flat fills, not real vectors. (Documented usage; effect strength not measured.) |
| "flat", "flat colors only" | Helps on all families; Recraft, OpenAI and BFL examples use it. Best paired with "no gradients, no shadows" on LLM encoders, or with a positive restatement ("solid fills") on CLIP models. |
| "minimalist", "minimal" | Widely used (Google, Recraft, OpenAI). It is evaluative, and Recraft warns against stacked evaluative adjectives. Better: give the count ("two shapes, one color"). |
| "white background" / "plain white background" / "solid #FFFFFF background" | The most reliable way to get a removable background on models without a `background` param. Recraft and BFL examples use solid backgrounds. Do not combine with `background: transparent` on OpenAI (the prompt backdrop can override alpha). |
| "transparent background" | Only meaningful on models with the parameter (OpenAI says say it in both places). On others it can produce a drawn checkerboard (OpenAI warns about this; Ideogram says the prompt word does not replace the setting). |
| "no text" | Works acceptably on LLM-encoder models (Seedream "no other text", Qwen "no readable text", OpenAI "no extra text"). On CLIP-only models (SD-Turbo, SDXL) the token "text" is attended to and can add lettering. For pictorial or abstract marks on those models, leave text out entirely and describe the mark as "symbol only" or "pictogram". |
| "icon", "app icon" | "icon" pushes toward pictogram-style single objects. "app icon" tends to produce a rounded-square tile with a gradient and gloss, which is fine for app icons and wrong for a bare mark. Apple's HIG tells designers not to add effects or masks themselves. (Community observation; Apple guidance is for designers, not models.) |
| "favicon" | Weak signal; few training captions use it. Describe the property instead: "reads clearly at 16 px, one bold shape, no thin lines". |
| "monoline" | Known design term for uniform stroke width. Recognized by LLM-encoder models; helps line icons. Add the stroke ("uniform 2 px-equivalent stroke, rounded caps"). Not documented by any provider. |
| "negative space" | OpenAI's logo example uses "balanced negative space". Asking for a specific negative-space trick (an arrow hidden between letters) works sometimes on strong reasoning models (GPT Image, Gemini Pro) and rarely on small distilled models. (Community observation.) |
| "geometric" | Helps shape language on all families. Recraft's logo example lists "sharp geometry" as something to avoid for a playful brand, so pick it deliberately. |
| quality tags ("masterpiece", "best quality", "8k", "trending on artstation") | No benefit on LLM-encoder models; BFL says to omit them. Only the booru-tag fine-tunes are trained on them. |
| photographic words (lens, lighting, "photorealistic", "3D render", "texture") | Hurt flat logos. Recraft: "Avoid texture or material-focused language". |

### 4.2 Negation by encoder type

- **CLIP (SD-Turbo, SDXL, the CLIP half of FLUX.1).** CLIP-style models largely fail to tell affirmative captions from negated ones (Alhamoud et al., "Vision-Language Models Do Not Understand Negation", CVPR 2025). In a prompt, "no text" contributes the concept "text". Use the separate negative-prompt field when CFG is on (SDXL). When CFG is off (SD-Turbo), use only positive descriptions.
- **T5 (FLUX.1, SD3).** T5 can parse "no" linguistically, but the VSF paper notes that T5 lacks contrastive pretraining, and that "even if the text encoder understands the negation, the diffusion model might still fail to avoid certain items because of their strong association." Treat negation as unreliable.
- **LLM / VLM encoders and native multimodal models (Qwen3, Mistral 3, Gemini, GPT).** Better, but still imperfect. The T2I-NSI benchmark (Gautam et al., arXiv 2611.09304, WACV 2027) found that "no X" almost never inverts (0.00% inverse interpretation) but often fails to suppress X. Open-weights models scored NSI at or below 0.038; GPT-Image-1.5 and Gemini-2.5-Flash-Image scored about 0.09, roughly 3.5x better but still low. The paper does not break results down by encoder type. Practical rule: state the positive version first ("solid fills, flat color") and add one short exclusion clause at the end ("no gradients, no text"). Do not list ten exclusions.
- **Provider stance.** Google ("semantic negative prompts") and BFL ("describe what you want") say to phrase positively. OpenAI, Recraft and Seedream (Runware) examples use short "no X" clauses. Ideogram: "Say what you want, not what you don't."

---

## 5. What professional logo designers specify in a brief

Briefs from design practice usually cover these, and each maps to a prompt slot.

1. **Mark type.** Industry guides list the same core set: wordmark (name only, typographic), lettermark (initials), pictorial mark (recognizable object), abstract mark (non-literal geometric form), emblem (text inside a badge or crest), combination mark (symbol plus wordmark, side by side, stacked or nested), and mascot. Many guides add negative-space or dynamic marks. Prompt implication: name the type explicitly, since each one changes whether text should render at all. Sources: https://mailchimp.com/resources/types-of-logos/, https://www.fiverr.com/resources/guides/graphic-design/types-of-logos, https://www.vistaprint.com/hub/types-of-logos
2. **Brand attributes and audience.** Two to four adjectives and who it is for. Haviv (Chermayeff & Geismar & Haviv) says a logo should be appropriate, distinctive and memorable, and simple, and that "appropriate" does not mean expressive: a logo "can't say very much". Prompt implication: one mood clause, not a paragraph of feelings. Sources: https://medium.com/designorex/the-3-rules-of-good-logo-design-9ccc05567310, https://logogeek.uk/en/sagi
3. **Shape language.** Rounded vs angular, organic vs geometric, symmetric vs dynamic, filled vs outlined. Recraft's guide calls this "shape logic" and "line discipline".
4. **Palette with hex.** Usually 1-3 colors plus neutrals, given as hex, with a one-color (black or reversed) version required. Prompt implication: hex tied to elements for FLUX; RGB via `controls.colors` for Recraft; plain color names plus hex for others. Adobe recommends checking a logo small, large, and in black and white (https://www.adobe.com/uk/express/discover/tips/logo/simple).
5. **Stroke weight and grid.** Icon systems fix these numbers. Material system icons: 24 dp grid, 20 dp live area, 2 dp padding, 2 dp stroke, 2 dp corner radius, keyline shapes for consistent proportions (https://material.io/design/iconography/system-icons.html). Apple's HIG: "avoid extremely thin line weights and sharp corners, because they tend to lose detail and crispness in smaller icon sizes" (https://developer.apple.com/design/human-interface-guidelines/app-icons). Prompt implication: "uniform medium-weight stroke, rounded corners, built on a square grid with even padding".
6. **Scale behavior at 16-32 px.** Favicon guidance from practitioners: one or two solid colors and one or two bold shapes; gradients and thin lines collapse; keep counters open; no shadows, glows or bevels; design a simplified small-size variant rather than shrinking the main logo; test the actual 16 px render on light and dark tab backgrounds (https://dev.to/alphashark/what-makes-a-logo-work-at-16x16-pixels-c4i, https://webflow.com/blog/favicon-guide, https://creativebloq.com/illustrator/create-perfect-favicon-12112760). These are practitioner blogs, not standards. Apple's HIG adds: "Embrace simplicity", "Include text only when it's essential", keep primary content centered, prefer illustrations to photos, and "there's no need to include specular highlights, drop shadows ... bevels, blurs, glows".
7. **What to avoid.** Competitor marks and clichés, trademarked shapes, effects, photos, more detail than the smallest size can hold. OpenAI's example asks for an "original, non-infringing logo".
8. **Deliverables.** Formats (SVG, PNG with alpha), lockups (horizontal, stacked, icon-only), color variants, minimum sizes. Brief templates also list references and revision counts (https://designshack.net/articles/graphics/logo-design-brief, https://www.tailorbrands.com/blog/logo-design-brief).

A prompt template that follows the brief (LLM-encoder models):

> A [mark type] logo for "[Name]", a [what the business does] for [audience]. The mark is [one-sentence concept: object or abstract form], built from [shape language], [stroke rule]. Palette: [color A #hex] for [element], [color B #hex] for [element]. Flat solid fills, centered with generous padding on a plain [white] background. It must read clearly at 16 px. [Text rule: either 'The wordmark "Name" in a [weight] [sans/serif], [case]' or 'Symbol only, no lettering'.]

For CLIP / SD-Turbo, compress to: `flat vector logo, [object] symbol, [shape words], [color words], centered, plain white background, simple, bold shapes`.

---

## 6. OpenRouter Images API: what can be passed

From the OpenAPI spec, the image-generation guide, and live endpoint records (2026-11-27):

- **Top-level fields:** `model`, `prompt`, `aspect_ratio`, `background` (auto / transparent / opaque; transparent needs png or webp), `output_format` (png / jpeg / webp / svg / gif), `output_compression`, `quality` (auto / low / medium / high / xhigh / max), `resolution` (512 / 768 / 1K / 1.5K / 2K / 4K / 8K tiers, "Concrete pixel dimensions are derived per-provider"), `size`, `seed`, `n` (1-10), `input_references` (up to 16), `markup`, `frame_count`, `frame_duration_ms`, `response_format`, `stream`, `provider`, `trace`, `user`, `session_id`.
- **No `negative_prompt`, `style`, `guidance` or `steps` at top level.** None of the 71 endpoints lists a negative-prompt passthrough either.
- **Passthrough:** `provider.options.<provider_slug>.<key>`. Each endpoint's `allowed_passthrough_parameters` lists the accepted keys, and "Unrecognized keys are silently dropped." Notable lists: Recraft v3 / v4 / v4.1 `style`, `controls`, `text_layout`; Recraft flash `controls`; Recraft styles `style_id`, `style_match`, `controls`, `random_seed`; BFL flux.2 pro / max `steps`, `guidance`, `safety_tolerance`, `disable_pup`; flex adds `prompt_upsampling`; klein-4b `steps`, `guidance`, `safety_tolerance`; flux-3-image `safety_tolerance`, `grounding`, `version`; flux-3-image-klein `safety_tolerance`, `steps`; OpenAI `moderation` (+ `complexity` on 2.6); Krea 2 style controls; Riverflow `font_inputs`; Seedream 5.5 `layer_decomposition`; MAI 2.6 / 2.7 `web_grounding`; MiniMax `disable_prompt_enhanced`; Gemini `cachedContent`; Qwen, Seedream 4.5 / 5.0, Grok, Muse, Hy: none.
- **`quality`:** "Providers without a quality knob ignore this." Only OpenAI models and Grok list it (Grok: low / medium / auto).
- **`resolution`:** listed for Seedream, Gemini 3+, Qwen, FLUX 3, Krea, Riverflow, Hy, Grok, MiniMax. Not listed for Recraft, OpenAI, FLUX.2, MAI, Muse.
- **Unsupported top-level params:** the docs say an absent key means unsupported, but do not say whether sending one errors or is ignored (except `quality` and `frame_count`, which are ignored). logo-gen should filter request fields against `supported_parameters` per model.
- **Discovery:** `GET /api/v1/images/models` and `GET /api/v1/images/models/{author}/{slug}/endpoints` are described as "the definitive set of parameters".

Sources: https://openrouter.ai/docs/api/api-reference/images/generate-an-image.md, https://openrouter.ai/docs/guides/overview/multimodal/image-generation.md, https://openrouter.ai/api/v1/images/models

---

## 7. Sources

Provider docs and model cards
- Recraft API endpoints: https://www.recraft.ai/docs/api-reference/endpoints.md
- Recraft V4.1 API: https://www.recraft.ai/docs/api-reference/models/recraft-v4-1.md
- Recraft V3 API: https://www.recraft.ai/docs/api-reference/models/recraft-v3.md
- Recraft styles: https://www.recraft.ai/docs/api-reference/styles.md
- Recraft appendix (limits, sizes): https://www.recraft.ai/docs/api-reference/appendix.md
- Recraft models overview: https://www.recraft.ai/docs/api-reference/models/overview.md
- Recraft prompting with V4: https://www.recraft.ai/docs/prompt-engineering-guide/prompting-with-recraft-v4.md
- Recraft bug report on logo_raster: https://feedback.recraft.ai/bug-reports/p/bug-report-invalid-model-style-combination-error
- BFL FLUX.2 prompting guide: https://docs.bfl.ai/guides/prompting_guide_flux2.md
- BFL unified reference: https://docs.bfl.ai/guides/prompting_unified_reference.md
- BFL style, aesthetics and text (FLUX 3): https://docs.bfl.ai/guides/prompting_unified_style
- BFL building a prompt: https://docs.bfl.ai/guides/prompting_unified_building
- BFL typography use case: https://docs.bfl.ai/guides/usecases_t2i_typography_design.md
- BFL FLUX 3 API: https://docs.bfl.ml/api-reference/utility/generate-an-image-with-flux-3.md
- BFL FLUX 3 overview: https://docs.bfl.ml/flux_3/flux3_image_overview.md
- FLUX.2 klein 4B card: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Diffusers Flux2 / Flux2Klein: https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux2
- Diffusers Flux (schnell): https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux
- Google Gemini image generation: https://aistudio.google.com/docs/image-generation
- Google dev blog, Gemini 2.5 Flash Image prompting: https://developers.googleblog.com/en/how-to-prompt-gemini-2-5-flash-image-generation-for-the-best-results/
- Google, Nano Banana Pro tips: https://blog.google/products/gemini/prompting-tips-nano-banana-pro/
- DeepMind Gemini image prompt guide: https://deepmind.google/models/gemini-image/prompt-guide/
- OpenAI image prompting guide: https://developers.openai.com/api/docs/guides/image-prompting.md
- OpenAI image generation guide: https://developers.openai.com/api/docs/guides/image-generation
- OpenAI cookbook prompting guide: https://developers.openai.com/cookbook/examples/multimodal/image-gen-models-prompting-guide.md
- OpenAI images API reference: https://developers.openai.com/api/reference/resources/images/methods/generate.md
- OpenAI forum, gpt-image-2 transparency preview: https://community.openai.com/t/transparent-backgrounds-are-now-available-in-preview-for-gpt-image-2-in-the-api/1391541
- ByteDance Seedream 5.0 Pro blog: https://seed.bytedance.com/en/blog/beyond-generation-it-understands-design-introducing-seedream-5-0-pro
- BytePlus Seedream prompt guide (body not retrievable): https://docs.byteplus.com/en/docs/ModelArk/seedream-4-0-5-0-prompt-guide
- Alibaba Qwen-Image API: https://www.alibabacloud.com/help/en/model-studio/qwen-image-api
- Z-Image Turbo card: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- Diffusers Z-Image: https://huggingface.co/docs/diffusers/main/en/api/pipelines/z_image
- Z-Image paper: https://arxiv.org/html/2511.22699v2
- SD-Turbo card: https://huggingface.co/stabilityai/sd-turbo
- Ideogram prompting guide: https://docs.ideogram.ai/using-ideogram/getting-started/prompting-guide/in-a-nutshell
- Ideogram API (generate v3, legacy): https://developer.ideogram.ai/api-reference/api-reference/generate-v3
- xAI image generation: https://docs.x.ai/developers/model-capabilities/images/generation.md
- Microsoft MAI on Foundry: https://learn.microsoft.com/azure/foundry/foundry-models/how-to/use-foundry-models-mai
- Microsoft MAI-Image-2 notebook: https://microsoft-foundry.github.io/forgebook/notebook/mai-image-2
- Meta Muse Image: https://developer.meta.com/ai/models/muse-image/
- Krea 2 docs: https://www.krea.ai/docs/user-guide/features/krea-2
- HunyuanImage 3.0 card: https://huggingface.co/tencent/HunyuanImage-3.0
- NoobAI-XL card: https://huggingface.co/Laxhar/noobai-XL-1.0
- Pony Diffusion V6 XL: https://civitai.com/models/257749/pony-diffusion-v6-xl

OpenRouter
- Images API reference: https://openrouter.ai/docs/api/api-reference/images/generate-an-image.md
- Image generation guide: https://openrouter.ai/docs/guides/overview/multimodal/image-generation.md
- Live model list: https://openrouter.ai/api/v1/images/models
- Riverflow V2 Pro page: https://openrouter.ai/sourceful/riverflow-v2-pro

Third-party guides (used where official docs were missing; treat as secondary)
- Runware Seedream 5.5: https://runware.ai/docs/models/bytedance-seedream-5-5/guides/prompting
- Runware Seedream 4.5: https://runware.ai/docs/models/bytedance-seedream-4-5/guides/prompting
- Runware Qwen Image 3.0: https://runware.ai/docs/models/alibaba-qwen-image-3-0/guides/prompting.md
- Runware Ideogram 4.6: https://runware.ai/docs/models/ideogram-4-6/guides/text-and-design.md
- Runware Muse Image text: https://runware.ai/docs/models/meta-muse-image/guides/text-in-images
- fal FLUX.2 klein guide: https://fal.ai/learn/devs/flux-2-klein-prompt-guide
- fal Krea 2 guide: https://fal.ai/learn/tools/krea-2-prompting-guide
- Replicate Riverflow 2.0 Pro readme: https://replicate.com/sourceful/riverflow-2.0-pro/readme
- Stable Diffusion Art, Pony tags: https://stable-diffusion-art.com/pony-diffusion-prompt-tags/
- Civitai Illustrious guide: https://civitai.com/articles/23210
- HF SDXL 77-token discussion: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/discussions/60
- Long-CLIP paper: https://arxiv.org/pdf/2403.15378

Research on negation
- Alhamoud et al., Vision-Language Models Do Not Understand Negation (CVPR 2025): https://arxiv.org/abs/2501.09425
- Gautam et al., T2I Models Rarely Invert Negation but Often Fail to Suppress Negated Concepts (arXiv 2611.09304): https://arxiv.org/abs/2611.09304
- VSF, negative guidance in few-step models (T5 negation note): https://arxiv.org/html/2508.10931v5

Design sources
- Apple HIG, app icons: https://developer.apple.com/design/human-interface-guidelines/app-icons
- Material system icons: https://material.io/design/iconography/system-icons.html
- Material Symbols guide: https://developers.google.com/fonts/docs/material_symbols
- Mailchimp, types of logos: https://mailchimp.com/resources/types-of-logos/
- Fiverr, types of logos: https://www.fiverr.com/resources/guides/graphic-design/types-of-logos
- Vistaprint, types of logos: https://www.vistaprint.com/hub/types-of-logos
- Sagi Haviv, three rules: https://medium.com/designorex/the-3-rules-of-good-logo-design-9ccc05567310 and https://logogeek.uk/en/sagi
- Adobe, simple logos: https://www.adobe.com/uk/express/discover/tips/logo/simple
- Favicons at 16 px: https://dev.to/alphashark/what-makes-a-logo-work-at-16x16-pixels-c4i, https://webflow.com/blog/favicon-guide, https://creativebloq.com/illustrator/create-perfect-favicon-12112760
- Logo brief guides: https://designshack.net/articles/graphics/logo-design-brief, https://www.tailorbrands.com/blog/logo-design-brief
