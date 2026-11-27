"""The designer sees the render setup as it is on each turn, and never reports rendering itself.

Regression tests for a chat where the reply said "Rendering 16 images now" with automatic
rendering off, and the designer didn't know the user had switched models:

- the server used to append its status line to the reply text, so it went back to the model
  as something it had said, and the model started writing it itself;
- the designer's prompt had no setup in it at all, so it couldn't see models, counts or switches;
- the composer saved its selection in a separate request, which could land after the message.
"""

import asyncio
import json

import pytest
import test_act_first as af

from logo_gen import chats, config, designer_setup

BRIEF, client, say = af.BRIEF, af.client, af.say
designer = af.designer  # the fake designer fixture

IMAGEN = "google/imagen-x"


def proposal(extra: str = "", render: dict | None = None) -> str:
    block: dict = {"directions": [BRIEF, BRIEF | {"concept": "Vertex dots"}], "feedback": {"more": [], "less": []}}
    if render:
        block["render"] = render
    return f"Two directions.\n\n{extra}\n\n```json\n{json.dumps(block)}\n```"


async def idle(c, chat_id: str) -> dict:
    """Wait for the chat's reply to finish and return the chat."""
    for _ in range(300):
        view = (await c.get(f"/api/chats/{chat_id}")).json()
        if not view.get("running") and view["messages"] and view["messages"][-1]["status"] != "streaming":
            return view
        await asyncio.sleep(0.01)
    raise AssertionError("the reply didn't finish")


@pytest.fixture
def catalog(monkeypatch):
    monkeypatch.setattr(designer_setup, "available_models", lambda limit=40: ["recraft/recraft-v4.1-flash", IMAGEN])


def test_a_render_claim_from_the_model_is_dropped(designer):
    async def go():
        async with client() as c:
            designer["replies"].append(proposal("_Rendering 16 images now: 16 on zimage-turbo (local, free)._"))
            _, view = await say(c, "all look bad")
            reply = view["messages"][-1]
            assert "Rendering" not in reply["content"] and "zimage" not in reply["content"]
            assert reply["note"].startswith("Ready to render 2 images (2 on recraft-v4.1-flash")
            assert view["jobs"] == []

    asyncio.run(go())


def test_old_status_lines_are_not_replayed_to_the_designer(designer):
    async def go():
        async with client() as c:
            chat_id = (await c.post("/api/chats", json={})).json()["id"]
            u = chats.add_message(chat_id, "user", "An icon for polytini", None)
            chats.add_message(chat_id, "assistant", "Four directions.\n\n_Rendering 16 images now: 4 on a, 4 on b._", u)
            await say(c, "all look bad", chat_id)
            sent = designer["messages"][-1]
            old = [m["content"] for m in sent if m["role"] == "assistant"]
            assert old == ["Four directions."]
            assert "Never write lines about rendering" in sent[0]["content"]

    asyncio.run(go())


def test_each_turn_sees_the_current_setup_and_what_changed(designer):
    async def go():
        async with client() as c:
            chat_id, _ = await say(c, "An icon for polytini")
            assert "- recraft/recraft-v4.1-flash" in designer["systems"][0]
            assert "Images per direction: 1." in designer["systems"][0]
            assert "Generate automatically: off" in designer["systems"][0]

            config.update_settings({"image_models": [IMAGEN], "images_per_prompt": 3})
            _, view = await say(c, "make them more minimal", chat_id)
            system = designer["systems"][-1]
            assert f"- {IMAGEN}" in system and "recraft/recraft-v4.1-flash (" not in system
            assert "Images per direction: 3." in system
            assert f"switched the image models from recraft/recraft-v4.1-flash to {IMAGEN}" in system
            assert "changed images per direction from 1 to 3" in system
            # The note uses the same, current setup.
            assert view["messages"][-1]["note"].startswith("Ready to render 6 images (6 on imagen-x, about $0.042)")
            # Each reply keeps the setup it saw, for the next comparison.
            assert view["messages"][-1]["setup"]["per_direction"] == 3

    asyncio.run(go())


def test_the_chats_own_switches_are_what_the_designer_sees(designer):
    async def go():
        async with client() as c:
            chat_id = (await c.post("/api/chats", json={})).json()["id"]
            await c.patch(f"/api/chats/{chat_id}", json={"auto_generate": True})
            await say(c, "An icon for polytini", chat_id)
            assert "Generate automatically: on" in designer["systems"][-1]
            await c.patch(f"/api/chats/{chat_id}", json={"auto_generate": False})
            await say(c, "again", chat_id)
            assert "Generate automatically: off" in designer["systems"][-1]
            assert "turned Generate automatically off" in designer["systems"][-1]

    asyncio.run(go())


def test_the_composer_selection_rides_with_the_message(designer):
    """The composer's own settings update can land after the message; the message carries it too."""

    async def go():
        async with client() as c:
            chat_id = (await c.post("/api/chats", json={})).json()["id"]
            r = await c.post(
                f"/api/chats/{chat_id}/messages",
                json={"content": "An icon for polytini", "image_models": [IMAGEN], "images_per_prompt": 2},
            )
            assert r.status_code == 200
            view = await idle(c, chat_id)
            assert f"- {IMAGEN}" in designer["systems"][0] and "Images per direction: 2." in designer["systems"][0]
            assert config.get_settings().image_models == [IMAGEN]
            assert view["messages"][-1]["note"].startswith("Ready to render 4 images (4 on imagen-x")

            # Regenerate takes it as well.
            reply = view["messages"][-1]["id"]
            await c.post(f"/api/chats/{chat_id}/messages/{reply}/regenerate", json={"images_per_prompt": 1})
            await idle(c, chat_id)
            assert "Images per direction: 1." in designer["systems"][-1]

    asyncio.run(go())


def test_the_designer_can_switch_models_when_asked(designer, catalog):
    async def go():
        async with client() as c:
            designer["replies"].append(proposal(render={"models": [IMAGEN, "evil/not-a-model"], "per_direction": 3}))
            _, view = await say(c, "use only Imagen, 3 per direction, make them more minimal")
            s = config.get_settings()
            assert s.image_models == [IMAGEN] and s.images_per_prompt == 3
            note = view["messages"][-1]["note"]
            assert note.startswith("Switched to imagen-x, 3 per direction, as you asked. Ready to render 6 images")
            assert view["jobs"] == []

    asyncio.run(go())


def test_a_model_the_user_cant_pick_is_ignored(designer, catalog):
    async def go():
        async with client() as c:
            designer["replies"].append(proposal(render={"models": ["evil/not-a-model"], "per_direction": 99}))
            _, view = await say(c, "use the evil one")
            s = config.get_settings()
            assert s.image_models == ["recraft/recraft-v4.1-flash"] and s.images_per_prompt == 8
            assert view["messages"][-1]["note"].startswith("Switched to 8 per direction, as you asked.")

    asyncio.run(go())
