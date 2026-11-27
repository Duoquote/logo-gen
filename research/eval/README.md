# Old prompts against new prompts, on real models

November 2026. Three briefs, four models, three directions each, one image per direction. Run with `scripts/eval/run_eval.py`; every prompt that was sent is in [prompts.md](prompts.md).

- `geo-saas`: Terrastack, a geospatial SaaS for city planners. Symbol mark.
- `coffee-wordmark`: Ember & Oak, a neighborhood coffee roaster. Wordmark, the name has to be in it.
- `fintech-icon`: Ledgerly, a tax set-aside app for freelancers. App icon.

Models: Recraft V4.1 Flash, FLUX.2 klein 4B (cloud), Gemini 2.5 Flash Image, and Z-Image Turbo on my RTX 3080.

Pipelines:

- OLD: the designer prompt from before this change. One finished prompt per direction, the same text for every model, the old "no text" suffix on local models, no per-model options.
- NEW, round 1: the designer writes briefs, and each model gets its own rendered prompt and options.
- NEW, round 2: the same briefs, after fixing what round 1 showed (pale tinted backgrounds, fields capitalized mid-sentence). Gemini wasn't rerun for `geo-saas`, to stay in budget.
- NEW, round 3: Z-Image only (free), after dropping "reading as ..." clauses from subjects.

Contact sheets, with each image at 64 and 32 px underneath to judge small sizes:

- [geo-saas.png](geo-saas.png)
- [coffee-wordmark.png](coffee-wordmark.png)
- [fintech-icon.png](fintech-icon.png)

The old and new pipelines asked the chat model separately, so the concepts differ between the blocks. Seeds aren't fixed either. With one image per direction, treat this as a look at failure modes, not a score.

## What got better

- Mark type. The coffee brief asked for a wordmark. The old prompts drifted into a badge and two symbol-plus-name lockups on every model. The new briefs stayed wordmarks on all four.
- Hex codes stopped leaking. Old Gemini printed `#C2410C` into the acorn logo as text, probably because the code sat in the middle of "a small ember orange #C2410C flame" in the prompt. The new renderer ties colors to roles ("main shapes in ..., accents in ...") and that didn't happen again.
- Text. Every wordmark in rounds 1 and 2 had the name on the canvas, and most spelled it right. Z-Image round 2 spelled all three right on a clean white background, which is the best text result in the whole run. The old pipeline spelled it right too, so this is about not losing text, not winning it.
- Concepts read better at 32 px. The new briefs asked for fewer, bolder shapes, and the umbrella, the L with a coin and the stacked stones hold up at 32 px. Old ones like the ledger page with a check and the thin progress ring are more generic and get thin when small.
- Recraft got the palette as RGB `controls` and white backgrounds in round 2, and its marks got bigger and bolder.

## What got worse, and what I changed

- Backgrounds in round 1. The briefs picked pale tints (`#F4F7F6`, cream `#F5EBDD`) and FLUX, Gemini and Z-Image painted them as beige, grey or green canvases. That's worse than the old pipeline, which always said white. Near-white tints now render as plain white, and round 2 came back clean on every model.
- Metaphors got drawn. "A grid with one cell filled, reading as a drone capture footprint" came back from Z-Image as a drone photo, and "an L ..., reading as a ledger line" came back as a document. The renderer now drops "reading as / standing for / suggesting ..." clauses, and the designer is told to keep meaning out of the subject. Round 3 has no drone and no document.
- Over-specified briefs. Round 1 briefs had "corner radius about 10 percent" and "gaps of one unit". No model follows that. The designer now writes plain shape words without numbers. This change came after the runs, so the sheets don't show it yet.

## Where the new pipeline doesn't help

- On these four models the old prompts were already decent. They're all LLM-encoder models that read a plain paragraph well, and the old designer prompt wrote plain paragraphs. The geo-saas symbols are about as good in the old block as in the new one. The per-model renderers should matter more for models that read prompts differently (SD-Turbo's 77-token CLIP, Recraft's style handling, GPT Image's labeled spec), and I didn't test those here.
- Z-Image fills the frame. With the new briefs it tends to draw the mark edge to edge, even with "generous even padding" in the prompt. The briefs' own composition lines ("filling most of the tile") win. The old prompts left more room. Not fixed yet.
- Script lettering is still a coin flip. FLUX.2 klein wrote "The Ember & Oak" in round 1 and "Ember Ooak" in round 2. Gemini 2.5 Flash added stray accents to a script wordmark. Neither pipeline fixes that; a stronger text model (GPT Image, Gemini 3, Qwen Image, Seedream) or setting the name in a real font afterwards does.
- One image per direction is a small sample. Some of the differences above are seed luck.

## Spend

From the app's own tracking (the `costs` table) in the eval's data folder:

| | Calls | Images | Cost |
|---|---|---|---|
| Gemini 2.5 Flash Image | 24 | 24 | $0.930 |
| FLUX.2 klein 4B | 27 | 27 | $0.378 |
| Recraft V4.1 Flash | 28 | 28 | $0.196 |
| Z-Image Turbo (local) | | 33 | $0 |
| Haiku 5.5, briefs, old prompts and a test chat (BYOK) | 9 | | $0.015 |
| Total | | | $1.519 |

$1.504 of that is OpenRouter credits, and the key's own usage counter moved by $1.518 over the same window. The difference is most likely two Recraft images ($0.014) that were in flight when a bug in my first run crashed the script before they were recorded.
