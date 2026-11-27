"""The designer proposes and renders right away, within caps and spend limits, unless asked to ask first."""

import asyncio
import json

import httpx
import pytest

from logo_gen import config, generator, prompt_engine, server, turns
from logo_gen.clients import openrouter

BRIEF = {
    "concept": "Folded hexagon", "mark_type": "abstract", "format": "logo", "subject": "a hexagon folded along one diagonal",
    "composition": "centered", "shape_language": "sharp geometric", "style": "flat vector",
    "palette": [{"name": "python blue", "hex": "#3776AB"}], "background": "white", "text": "", "avoid": [],
}
PROPOSAL = (
    "Defaults: a symbol in Python blue and yellow, for docs and the repo.\n\n1. **Folded hexagon**\n2. **Vertex dots**\n\n"
    "Want it to nod to Python?\n\n```json\n"
    + json.dumps({"directions": [BRIEF, BRIEF | {"concept": "Vertex dots"}], "feedback": {"more": [], "less": []}})
    + "\n```"
)
QUESTIONS = "What's the logo for? A name and one line about it is enough."


@pytest.fixture
def designer(home, monkeypatch):
    """A fake designer that proposes for a real brief and asks for an empty one, plus a fake image model."""
    seen_systems: list[str] = []

    async def stream_reply(messages):
        seen_systems.append(messages[0]["content"])
        text = QUESTIONS if messages[-1]["content"].strip() == "make me a logo" else PROPOSAL
        yield {"type": "token", "content": text}

    async def no_vision():
        return False

    rendered: list[tuple] = []

    async def fake_run(job, prompts, models, per, refs=None, chat_id=None, parent_id=None):
        work, notes = generator.plan(models, prompts, per or 1)
        rendered.append((job.id, chat_id, {m: len(ps) * n for m, (ps, n) in work.items()}))

    async def model_info(model):
        return {"price_per_image": 0.007, "params": {}}

    key = {"limit_remaining": None}

    async def key_status(k=None):
        return {"valid": True, **key}

    monkeypatch.setattr(prompt_engine, "stream_reply", stream_reply)
    monkeypatch.setattr(prompt_engine, "chat_supports_images", no_vision)
    monkeypatch.setattr(turns, "_turns", {})
    monkeypatch.setattr(generator, "run", fake_run)
    monkeypatch.setattr(openrouter, "model_info", model_info)
    monkeypatch.setattr(openrouter, "key_status", key_status)
    monkeypatch.setattr(server, "_usable", lambda m: not m.startswith("local/"))
    config.update_settings({"image_models": ["recraft/recraft-v4.1-flash"], "images_per_prompt": 1})
    return {"systems": seen_systems, "rendered": rendered, "key": key}


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")


async def say(c, text: str, chat_id: str | None = None) -> tuple[str, dict]:
    if chat_id is None:
        chat_id = (await c.post("/api/chats", json={})).json()["id"]
    r = (await c.post(f"/api/chats/{chat_id}/messages", json={"content": text})).json()
    for _ in range(200):
        t = turns.get(r["reply_id"])
        if t and t.finished:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)  # let the generate job start
    return chat_id, (await c.get(f"/api/chats/{chat_id}")).json()


def test_a_brief_is_proposed_and_rendered_in_the_same_turn(designer):
    async def go():
        async with client() as c:
            chat_id, view = await say(c, "An icon for polytini, a small Python library for polygon geometry.")
            reply = view["messages"][-1]
            assert len(reply["prompts"]) == 2
            assert "Rendering 2 images now: 2 on recraft-v4.1-flash (about $0.014)" in reply["content"]
            assert "Want it to nod to Python?" in reply["content"] and "```json" not in reply["content"]
            # A real server job, linked to the reply, so the images land under it.
            assert [j["message_id"] for j in view["jobs"]] == [reply["id"]]
            assert designer["rendered"][0][1:] == (chat_id, {"recraft/recraft-v4.1-flash": 2})
            assert "Act first" in designer["systems"][0]

    asyncio.run(go())


def test_an_empty_brief_only_gets_questions(designer):
    async def go():
        async with client() as c:
            _, view = await say(c, "make me a logo")
            assert view["messages"][-1]["content"] == QUESTIONS
            assert view["jobs"] == [] and designer["rendered"] == []

    asyncio.run(go())


def test_ask_first_setting_and_per_chat_override(designer):
    async def go():
        async with client() as c:
            config.update_settings({"ask_before_generating": True})
            chat_id, view = await say(c, "An icon for polytini")
            assert view["jobs"] == [] and "Rendering" not in view["messages"][-1]["content"]
            assert "Before proposing directions you need" in designer["systems"][-1]

            # This chat says act first, whatever the setting says.
            patched = (await c.patch(f"/api/chats/{chat_id}", json={"ask_first": False})).json()
            assert patched["ask_first"] is False
            _, view = await say(c, "go", chat_id)
            assert len(view["jobs"]) == 1 and "Act first" in designer["systems"][-1]

    asyncio.run(go())


def test_the_round_respects_caps_and_counts(designer):
    async def go():
        async with client() as c:
            config.update_settings({"images_per_prompt": 3, "model_caps": {"recraft/recraft-v4.1-flash": 4}})
            _, view = await say(c, "An icon for polytini")
            # 2 directions x 3 would be 6; the cap of 4 leaves 2 per direction.
            assert designer["rendered"][0][2] == {"recraft/recraft-v4.1-flash": 4}
            assert "Rendering 4 images" in view["messages"][-1]["content"]
            assert "trimmed to the per-model caps" in view["messages"][-1]["content"]

    asyncio.run(go())


def test_no_cloud_round_past_the_keys_spending_limit(designer):
    async def go():
        async with client() as c:
            designer["key"]["limit_remaining"] = 0.005
            config.update_settings({"image_models": ["recraft/recraft-v4.1-flash", "local/zimage-turbo"]})
            _, view = await say(c, "An icon for polytini")
            text = view["messages"][-1]["content"]
            assert view["jobs"] == [] and designer["rendered"] == []
            assert "the key has $0.005 left" in text and "zimage-turbo (not installed)" in text
            assert "Press Generate" in text

    asyncio.run(go())
