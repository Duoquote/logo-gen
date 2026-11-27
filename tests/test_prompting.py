"""Profile lookup, renderers (with snapshots), brief parsing and learned preferences.

Snapshots live in tests/snapshots/. After an intended change to a profile or
renderer, refresh them with:

    UPDATE_SNAPSHOTS=1 uv run pytest tests/test_prompting.py
"""

import json
import os
from pathlib import Path

import pytest

from logo_gen import config, library, paths, prompt_engine
from logo_gen.prompting import brief as briefs
from logo_gen.prompting import preferences, profiles, render

SNAPSHOTS = Path(__file__).parent / "snapshots"

BRIEFS = {
    "geo-saas": {
        "concept": "Contour pin",
        "mark_type": "pictorial",
        "format": "logo",
        "subject": "a map pin whose inner space is filled with three stacked contour lines",
        "composition": "single symbol, vertically symmetric",
        "shape_language": "rounded geometric, even 3 px monoline strokes",
        "style": "flat vector, two colors",
        "palette": [{"name": "deep teal", "hex": "#0F766E"}, {"name": "warm amber", "hex": "#F59E0B"}],
        "background": "white",
        "text": "",
        "avoid": ["gradients", "thin hairlines"],
    },
    "coffee-wordmark": {
        "concept": "Warm serif wordmark",
        "mark_type": "wordmark",
        "format": "logo",
        "subject": "the name in a warm, slightly rounded serif; the counter of the o is a coffee bean",
        "composition": "one line of text, horizontally centered",
        "shape_language": "soft serifs, medium weight",
        "style": "flat vector, one color",
        "palette": [{"name": "espresso brown", "hex": "#4B2E1F"}],
        "background": "cream #F6EFE4",
        "text": "Ember & Oak",
        "avoid": ["3d", "steam swirls"],
    },
    "fintech-icon": {
        "concept": "Stacked coin arrow",
        "mark_type": "abstract",
        "format": "app icon",
        "subject": "three stacked rounded bars forming an upward arrow",
        "composition": "centered inside a rounded square tile",
        "shape_language": "bold geometric, rounded corners",
        "style": "flat vector",
        "palette": [{"name": "electric indigo", "hex": "#4F46E5"}, {"name": "white", "hex": "#FFFFFF"}],
        "background": "transparent",
        "text": "",
        "avoid": ["fine detail"],
    },
}

MODELS = [
    "openai/gpt-image-1-mini",
    "google/gemini-2.5-flash-image",
    "recraft/recraft-v4.1-flash",
    "black-forest-labs/flux.2-klein-4b",
    "black-forest-labs/flux-3-image-klein",
    "bytedance-seed/seedream-5-5",
    "local/zimage-turbo",
    "local/sd-turbo",
    "someone/illustrious-xl-v2",
]

INFO = {
    "openai/gpt-image-1-mini": {
        "params": {"quality": {"values": ["auto", "low", "medium", "high"]}, "output_format": {"values": ["png", "jpeg"]},
                   "background": {"values": ["auto", "transparent", "opaque"]}},
        "supports_transparent": True,
    },
    "bytedance-seed/seedream-5-5": {
        "params": {"resolution": {"values": ["1K", "1.5K", "2K"]}, "output_format": {"values": ["png", "jpeg"]},
                   "background": {"values": ["auto", "transparent", "opaque"]}},
        "supports_transparent": True,
    },
    "black-forest-labs/flux.2-klein-4b": {"params": {"output_format": {"values": ["png", "jpeg"]}}},
}


@pytest.fixture
def settings(monkeypatch, tmp_path):
    s = config.Settings()
    monkeypatch.setattr(config, "_settings", s)
    monkeypatch.setattr(preferences, "get_settings", lambda: s)
    monkeypatch.setattr(paths, "DB_FILE", tmp_path / "test.sqlite3")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(library, "_conn", None)
    monkeypatch.setattr(preferences, "_ready", None)
    yield s
    if library._conn is not None:
        library._conn.close()
    library._conn = None


# ── Registry ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/gpt-image-1-mini", "gpt-image"),
        ("openai/gpt-5-image-mini", "gpt-image"),
        ("openai/some-future-image-model", "gpt-image"),  # provider fallback
        ("google/gemini-3.1-flash-image", "gemini-image"),
        ("google/gemini-nano-banana-2.1", "gemini-image"),
        ("recraft/recraft-v4.1-pro-vector", "recraft"),
        ("recraft/recraft-v4.1-flash-v2:free", "recraft"),
        ("black-forest-labs/flux.2-pro", "flux-2"),
        ("black-forest-labs/flux.2-klein-4b", "flux-2-klein"),
        ("black-forest-labs/flux-3-image-klein", "flux-3"),
        ("black-forest-labs/flux-9", "flux-2"),  # provider fallback
        ("bytedance-seed/seedream-5-0-lite", "seedream"),
        ("qwen/qwen-image-3", "qwen-image"),
        ("local/zimage-turbo", "z-image-turbo"),
        ("local/flux2-klein-4b", "flux-2-klein-local"),
        ("local/flux1-schnell", "flux-1-schnell-local"),
        ("local/sd-turbo", "sd-turbo"),
        ("civitai/noobai-xl-epsilon", "booru-tags"),
        ("stabilityai/sdxl-base", "sdxl"),
        ("ideogram-ai/ideogram-v3", "ideogram"),
        ("brand-new/mystery-model", "default"),
        ("", "default"),
        ("local/unknown-model", "default"),
    ],
)
def test_profile_lookup(model, expected):
    assert profiles.profile_for(model).id == expected


def test_profile_ids_are_unique_and_styles_have_renderers():
    ids = [p.id for p in profiles.PROFILES]
    assert len(ids) == len(set(ids))
    for p in [*profiles.PROFILES, profiles.DEFAULT]:
        assert p.style in render.RENDERERS
        assert p.max_words > 0


# ── Renderers ────────────────────────────────────────────────────────


def _snapshot_text(brief_name: str) -> str:
    out = []
    for m in MODELS:
        r = render.render(BRIEFS[brief_name], m, INFO.get(m))
        out.append(f"## {m} [{r.profile}, {r.style}]")
        out.append(r.prompt)
        if r.negative_prompt:
            out.append(f"NEGATIVE: {r.negative_prompt}")
        if r.params:
            out.append(f"PARAMS: {json.dumps(r.params, sort_keys=True)}")
        for n in r.notes:
            out.append(f"NOTE: {n}")
        out.append("")
    return "\n".join(out)


@pytest.mark.parametrize("name", list(BRIEFS))
def test_render_snapshots(name):
    text = _snapshot_text(name)
    path = SNAPSHOTS / f"{name}.txt"
    if os.environ.get("UPDATE_SNAPSHOTS") or not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    assert text == path.read_text(encoding="utf-8")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("name", list(BRIEFS))
def test_prompts_stay_within_the_useful_length(model, name):
    r = render.render(BRIEFS[name], model, INFO.get(model))
    assert r.prompt
    assert len(r.prompt.split()) <= profiles.profile_for(model).max_words


def test_no_negative_prompt_where_unsupported():
    for m in MODELS:
        r = render.render(BRIEFS["geo-saas"], m, INFO.get(m))
        assert bool(r.negative_prompt) == profiles.profile_for(m).negative


def test_clip_and_t5_models_never_get_negated_words():
    """'no text' or 'no gradients' in a CLIP/T5 prompt tends to draw the thing."""
    for m in ("local/sd-turbo", "local/flux1-schnell", "google/gemini-2.5-flash-image", "black-forest-labs/flux.2-pro"):
        prompt = render.render(BRIEFS["geo-saas"], m, INFO.get(m)).prompt.lower()
        assert "no text" not in prompt
        assert "gradients" not in prompt
        assert " no " not in f" {prompt} "


def test_llm_encoder_models_get_one_short_exclusion_clause():
    prompt = render.render(BRIEFS["geo-saas"], "local/zimage-turbo").prompt
    assert prompt.count("No ") == 1
    assert "No text, no gradients, no thin hairlines." in prompt


def test_text_is_quoted_for_strong_models_and_dropped_for_weak_ones():
    gpt = render.render(BRIEFS["coffee-wordmark"], "openai/gpt-image-1-mini", INFO["openai/gpt-image-1-mini"])
    assert '"Ember & Oak"' in gpt.prompt and not gpt.notes
    sd = render.render(BRIEFS["coffee-wordmark"], "local/sd-turbo")
    assert "Ember" not in sd.prompt and sd.notes


def test_transparent_background_uses_the_parameter_when_supported():
    r = render.render(BRIEFS["fintech-icon"], "openai/gpt-image-1-mini", INFO["openai/gpt-image-1-mini"])
    assert r.params.get("background") == "transparent"
    assert "transparent" in r.prompt.lower()
    g = render.render(BRIEFS["fintech-icon"], "google/gemini-2.5-flash-image")
    assert "background" not in g.params
    assert any("transparent" in n for n in g.notes)


def test_params_only_include_what_the_model_lists():
    r = render.render(BRIEFS["geo-saas"], "bytedance-seed/seedream-5-5", INFO["bytedance-seed/seedream-5-5"])
    assert r.params["resolution"] == "1K"  # "lowest"
    bare = render.render(BRIEFS["geo-saas"], "bytedance-seed/seedream-5-5", {"params": {}})
    assert bare.params == {}


def test_learned_avoids_reach_the_prompt_but_not_against_the_brief():
    r = render.render(BRIEFS["geo-saas"], "openai/gpt-image-1-mini", INFO["openai/gpt-image-1-mini"],
                      extra_avoid=["mascots", "monoline"])
    assert "Constraints: No gradients, no thin hairlines, no mascots." in r.prompt  # monoline is what the brief asks for


def test_direction_routing():
    edited = {"prompt": "x", "overrides": {"local/sd-turbo": {"prompt": "my own words"}}, "brief": BRIEFS["geo-saas"]}
    assert render.render_direction(edited, "local/sd-turbo").prompt == "my own words"
    assert render.render_direction(edited, "local/zimage-turbo").prompt.startswith("A flat vector")
    assert render.render_direction({"prompt": "x", "verbatim": True, "params": {}}, "openai/gpt-image-1-mini",
                                   INFO["openai/gpt-image-1-mini"]).params == {}
    assert render.render_direction({"prompt": "exact", "verbatim": True}, "local/zimage-turbo").prompt == "exact"
    raw = render.render_direction({"prompt": "a fox for a VPN app"}, "local/zimage-turbo").prompt
    assert raw.startswith("A fox for a VPN app.") and "logo" in raw
    assert render.render_direction({"prompt": "fox logo"}, "local/zimage-turbo").prompt == "fox logo"


def test_recraft_gets_palette_and_background_as_controls():
    r = render.render(BRIEFS["coffee-wordmark"], "recraft/recraft-v4.1-flash", {"params": {}})
    controls = r.params["provider"]["options"]["recraft"]["controls"]
    assert controls["colors"] == [{"rgb": [75, 46, 31]}]
    assert controls["background_color"] == {"rgb": [255, 255, 255]}  # pale cream counts as white
    assert "Flat colors only" in r.prompt


def test_flux3_turns_web_grounding_off_and_avoids_the_word_vector():
    r = render.render(BRIEFS["geo-saas"], "black-forest-labs/flux-3-image-klein", {"params": {}})
    assert r.params["provider"]["options"]["black-forest-labs"]["grounding"] is False
    assert "vector" not in r.prompt.lower()


def test_color_names_for_clip():
    assert render.color_name("#0F766E") == "teal"
    assert render.color_name("#F59E0B") == "orange"
    assert render.color_name("nope") == ""


# ── Briefs and the designer's reply ──────────────────────────────────


def test_brief_parsing_is_forgiving():
    b = briefs.from_dict({"subject": "a fox", "mark_type": "Monogram", "palette": "teal #0f766e, amber F59E0B",
                          "text": "none", "avoid": "gradients; 3d"})
    assert b.mark_type == "lettermark"
    assert [c.hex for c in b.palette] == ["#0F766E", "#F59E0B"]
    assert b.text == ""
    assert b.avoid == ["gradients", "3d"]


def test_schema_and_prompt_name_the_same_fields():
    schema_fields = briefs.json_schema()["schema"]["properties"]["directions"]["items"]["required"]
    assert schema_fields == list(briefs.FIELDS)
    for f in briefs.FIELDS:
        assert f"- {f}:" in prompt_engine.DESIGNER_SYSTEM


def test_parse_reply_reads_briefs_feedback_and_old_prompts():
    reply = (
        "Two ideas.\n```json\n"
        + json.dumps({"directions": [BRIEFS["geo-saas"], BRIEFS["fintech-icon"]], "feedback": {"more": ["bolder"], "less": []}})
        + "\n```"
    )
    directions, fb = prompt_engine.parse_reply(reply)
    assert [d["concept"] for d in directions] == ["Contour pin", "Stacked coin arrow"]
    assert directions[0]["brief"]["palette"][0]["hex"] == "#0F766E"
    assert directions[0]["prompt"].startswith("Pictorial logo: a map pin")
    assert fb == {"more": ["bolder"], "less": []}
    assert prompt_engine.strip_prompt_block(reply) == "Two ideas."

    old = '```json\n{"prompts": [{"concept": "A", "style": "flat", "prompt": "a fox"}]}\n```'
    assert prompt_engine.parse_prompts(old) == [{"concept": "A", "style": "flat", "prompt": "a fox"}]

    only_feedback = 'Noted.\n```json\n{"feedback": {"more": [], "less": ["gradients"]}}\n```'
    assert prompt_engine.parse_reply(only_feedback) == ([], {"more": [], "less": ["gradients"]})


def test_history_reattaches_briefs():
    d = prompt_engine.direction_from_brief(BRIEFS["geo-saas"])
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Here.", "prompts": [d]}]
    out = prompt_engine.build_history(msgs, lambda p: p, vision=False)
    assert '"directions"' in out[-1]["content"] and "contour" in out[-1]["content"]


# ── Preferences ──────────────────────────────────────────────────────


def _image(tmp_path, name, brief, model="recraft/recraft-v4.1-flash", chat_id="c1"):
    from PIL import Image

    paths.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p = paths.OUTPUT_DIR / f"{name}.png"
    Image.new("RGB", (8, 8), "white").save(p)
    return library.add_image(p, "generated", model=model, chat_id=chat_id, brief=brief, concept=brief["concept"],
                             style=brief["style"])


def _react(kind, image_id):
    """A reaction the way the chat UI records it: a feedback_events row, then the sync."""
    from logo_gen import chats

    chats.add_feedback(kind, image_id=image_id)
    preferences.sync_feedback()


def test_feedback_builds_a_profile_and_feeds_the_designer(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    liked = _image(tmp_path, "a", BRIEFS["geo-saas"])
    disliked = _image(tmp_path, "b", BRIEFS["coffee-wordmark"] | {"style": "3d render, glossy"}, model="openai/gpt-image-1-mini")

    _react("pick", liked["id"])
    _react("dislike", disliked["id"])
    p = preferences.profile("chat:c1")
    assert p["liked"]["mark_type"] == ["pictorial"]
    assert "flat vector" in p["liked"]["style"]
    assert "3d render" in p["disliked"]["style"]
    assert p["models"][0]["model"] == "recraft/recraft-v4.1-flash"
    assert p["counts"]["pick"] == 1 and p["counts"]["dislike"] == 1

    preferences.add_notes("c1", more=["thicker strokes"], less=["gradients", "the P lettermark"])
    ctx = preferences.designer_context("c1")
    assert "thicker strokes" in ctx and "gradients" in ctx and "3d render" in ctx and "P lettermark" in ctx
    assert "this user" in prompt_engine.designer_system("c1")
    # Only visual attributes reach image prompts; naming "the P lettermark" there would draw it.
    assert preferences.render_hints("c1") == ["gradients", "3d render"]

    # unrate clears a thumbs rating.
    _react("unrate", disliked["id"])
    assert "3d render" not in preferences.profile("chat:c1")["disliked"].get("style", [])
    # A later "more of" cancels an older "less of".
    preferences.add_notes("c1", more=["gradients"])
    assert "gradients" not in preferences.profile("chat:c1")["less"]

    # The global profile got the same events; clearing the chat leaves it.
    preferences.clear("chat:c1")
    assert preferences.profile("chat:c1")["empty"]
    assert not preferences.profile("global")["empty"]


def test_a_pick_outlasts_thumbs_but_not_a_dislike(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    img = _image(tmp_path, "a", BRIEFS["geo-saas"])
    _react("pick", img["id"])
    _react("like", img["id"])
    _react("unrate", img["id"])
    assert preferences.profile("chat:c1")["counts"]["pick"] == 1
    _react("dislike", img["id"])
    counts = preferences.profile("chat:c1")["counts"]
    assert counts["pick"] == 0 and counts["dislike"] == 1


def test_sync_reads_each_event_once_and_skips_deleted_images(settings, tmp_path, monkeypatch):
    from logo_gen import chats

    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    img = _image(tmp_path, "a", BRIEFS["geo-saas"])
    gone = _image(tmp_path, "b", BRIEFS["fintech-icon"])
    chats.add_feedback("like", image_id=img["id"])
    chats.add_feedback("like", image_id=gone["id"])
    library.delete_image(gone["id"])
    chats.add_feedback("pick", chat_id="c1")  # no image: nothing to learn
    assert preferences.sync_feedback() == 1
    assert preferences.sync_feedback() == 0


def test_api_feedback_updates_what_the_designer_learned(home):
    """POST /api/feedback (the chat UI's thumbs) shows up in GET /api/preferences."""
    import asyncio

    import httpx

    from logo_gen import server

    paths.GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    img = _image(home, "a", BRIEFS["geo-saas"])

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://t") as c:
            r = await c.post("/api/feedback", json={"kind": "like", "image_id": img["id"]})
            assert r.status_code == 200 and r.json()["kind"] == "like"
            prefs = (await c.get("/api/preferences", params={"chat_id": "c1"})).json()
            assert prefs["chat"]["liked"]["mark_type"] == ["pictorial"]
            assert "pictorial" in prefs["designer_context"]
            await c.post("/api/feedback", json={"kind": "unrate", "image_id": img["id"]})
            assert (await c.get("/api/preferences", params={"chat_id": "c1"})).json()["chat"]["empty"]

    asyncio.run(go())


def test_global_taste_can_be_turned_off(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    settings.use_global_taste = False
    img = _image(tmp_path, "a", BRIEFS["geo-saas"])
    _react("like", img["id"])
    assert preferences.profile("global")["empty"]
    assert not preferences.profile("chat:c1")["empty"]


def test_meaning_clauses_stay_out_of_image_prompts():
    """The eval drew a drone photo for 'a grid ..., reading as a drone capture footprint'."""
    b = BRIEFS["geo-saas"] | {"subject": "a square split into four cells, one filled, reading as a drone capture footprint"}
    for m in MODELS:
        assert "drone" not in render.render(b, m, INFO.get(m)).prompt
    assert render.drawable("a leaf shaped like a heart") == "a leaf shaped like a heart"
    assert render.drawable("a wedge pulled out, the small wedge standing for taxes") == "a wedge pulled out"


def test_other_chats_lend_only_general_taste(settings, tmp_path, monkeypatch):
    """Found in the browser test: notes from one project ("stacked stones") showed up in another chat's designer."""
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    for i, name in enumerate(("a", "b", "c")):
        _react("like", _image(tmp_path, name, BRIEFS["geo-saas"], chat_id="old")["id"])
    preferences.add_notes("old", more=["stacked stones"], less=["the P lettermark"])

    ctx = preferences.designer_context("new")
    assert "earlier chats" in ctx and "flat vector" in ctx and "pictorial" in ctx
    assert "stacked stones" not in ctx and "P lettermark" not in ctx and "#0F766E" not in ctx

    # In the chat that made them, the same votes count once, as this chat's own.
    ctx_old = preferences.designer_context("old")
    assert "stacked stones" in ctx_old and "earlier chats" not in ctx_old
