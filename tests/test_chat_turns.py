"""Designer replies keep running and saving when nobody is watching, and branches never lose history."""

import asyncio

import httpx
import pytest

from logo_gen import chats, prompt_engine, server, turns

WORDS = [f"word{i} " for i in range(12)]
FULL = "".join(WORDS)


@pytest.fixture
def fake_model(home, monkeypatch):
    """A designer that thinks briefly, then answers one word every 30 ms."""

    async def stream_reply(messages):
        yield {"type": "reasoning", "content": "thinking about it. "}
        for w in WORDS:
            await asyncio.sleep(0.03)
            yield {"type": "token", "content": w}
        yield {"type": "usage", "cost": 0.0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.0012}}

    async def no_vision():
        return False

    monkeypatch.setattr(prompt_engine, "stream_reply", stream_reply)
    monkeypatch.setattr(prompt_engine, "chat_supports_images", no_vision)
    monkeypatch.setattr(turns, "SAVE_EVERY", 0)
    monkeypatch.setattr(turns, "_turns", {})


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://test")


async def wait_done(message_id: int) -> None:
    for _ in range(200):
        t = turns.get(message_id)
        if t and t.finished:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("turn did not finish")


def test_leaving_mid_stream_and_coming_back(fake_model):
    async def go():
        async with client() as c:
            chat = (await c.post("/api/chats", json={})).json()
            r = (await c.post(f"/api/chats/{chat['id']}/messages", json={"content": "A logo for a bakery"})).json()
            reply = r["reply_id"]

            # Watch the first few events, then leave (the browser closes the stream).
            seen, last_id = "", 0
            follow = turns.follow(reply, None)
            async for ev in follow:
                last_id = ev.get("id", last_id)
                if ev["type"] == "snapshot":
                    seen = ev["content"]
                elif ev["type"] == "token":
                    seen += ev["content"]
                if seen.count("word") >= 3:
                    break
            await follow.aclose()

            # Nobody is watching, but the reply keeps going and keeps being saved.
            await asyncio.sleep(0.15)
            saved = chats.get_message(reply)
            assert saved["status"] == "streaming"
            assert len(saved["content"]) > len(seen)
            assert saved["reasoning"] == "thinking about it. "

            # The chat view says a reply is running, so the UI knows to reattach.
            view = (await c.get(f"/api/chats/{chat['id']}")).json()
            assert view["running"] == reply
            assert view["messages"][-1]["status"] == "streaming"

            # Come back with the cursor: get exactly the rest, nothing twice.
            rest, end = "", None
            async for ev in turns.follow(reply, last_id):
                if ev["type"] == "token":
                    rest += ev["content"]
                if ev["type"] == "end":
                    end = ev["status"]
            assert seen + rest == FULL
            assert end == "done"

            # Come back without a cursor: one snapshot with everything.
            events = [ev async for ev in turns.follow(reply, None)]
            assert events[0]["type"] == "snapshot" and events[0]["content"] == FULL

            view = (await c.get(f"/api/chats/{chat['id']}")).json()
            msg = view["messages"][-1]
            assert (msg["status"], msg["content"], view["running"]) == ("done", FULL, None)
            assert msg["cost"] == {"credits": 0.0, "byok": 0.0012}
            assert view["spend"]["byok"] == 0.0012

            # The SSE endpoint for a finished reply replays it from the database and ends.
            sse = (await c.get(f"/api/chats/{chat['id']}/messages/{reply}/stream")).text
            assert '"type": "snapshot"' in sse and '"type": "end"' in sse

    asyncio.run(go())


def test_reply_survives_without_any_watcher_and_server_restart_marks_it(fake_model):
    async def go():
        async with client() as c:
            chat = (await c.post("/api/chats", json={})).json()
            r = (await c.post(f"/api/chats/{chat['id']}/messages", json={"content": "hi"})).json()
            # A second message while the first reply runs is refused, not queued silently.
            busy = await c.post(f"/api/chats/{chat['id']}/messages", json={"content": "again"})
            assert busy.status_code == 409
            await wait_done(r["reply_id"])
            assert chats.get_message(r["reply_id"])["content"] == FULL

            # A reply that was streaming when the server died is marked, not left spinning.
            stuck = chats.add_message(chat["id"], "assistant", "half", r["user_id"], status="streaming")
            turns.recover()
            m = chats.get_message(stuck)
            assert m["status"] == "error" and "Regenerate" in m["error"]

    asyncio.run(go())


def test_stop_keeps_the_partial_reply(fake_model):
    async def go():
        async with client() as c:
            chat = (await c.post("/api/chats", json={})).json()
            r = (await c.post(f"/api/chats/{chat['id']}/messages", json={"content": "hi"})).json()
            await asyncio.sleep(0.1)
            assert (await c.post(f"/api/chats/{chat['id']}/messages/{r['reply_id']}/stop")).json()["stopped"]
            await wait_done(r["reply_id"])
            m = chats.get_message(r["reply_id"])
            assert m["status"] == "stopped" and 0 < len(m["content"]) < len(FULL)

    asyncio.run(go())


def test_edit_regenerate_and_delete_keep_history(fake_model):
    async def go():
        async with client() as c:
            cid = (await c.post("/api/chats", json={})).json()["id"]
            first = (await c.post(f"/api/chats/{cid}/messages", json={"content": "a fox"})).json()
            await wait_done(first["reply_id"])

            # Edit the first message: a sibling branch, and the old one stays.
            edit = (await c.post(f"/api/chats/{cid}/messages", json={"content": "a wolf", "edit_of": first["user_id"]})).json()
            await wait_done(edit["reply_id"])
            view = (await c.get(f"/api/chats/{cid}")).json()
            assert [m["content"] for m in view["messages"]][0] == "a wolf"
            assert view["messages"][0]["siblings"] == [first["user_id"], edit["user_id"]]
            assert view["title"] == "a fox"  # auto title from the first message; edits don't retitle

            # Switch back to the old branch: its reply comes with it.
            view = (await c.put(f"/api/chats/{cid}/active", json={"message_id": first["user_id"]})).json()
            assert [m["id"] for m in view["messages"]] == [first["user_id"], first["reply_id"]]

            # Regenerate the reply: another sibling.
            regen = (await c.post(f"/api/chats/{cid}/messages/{first['reply_id']}/regenerate")).json()
            await wait_done(regen["reply_id"])
            view = (await c.get(f"/api/chats/{cid}")).json()
            assert view["messages"][-1]["siblings"] == [first["reply_id"], regen["reply_id"]]

            # Delete from here, then undo.
            stamp = (await c.delete(f"/api/chats/{cid}/messages/{regen['reply_id']}")).json()["stamp"]
            view = (await c.get(f"/api/chats/{cid}")).json()
            assert view["messages"][-1]["id"] == first["reply_id"]
            view = (await c.post(f"/api/chats/{cid}/messages/{regen['reply_id']}/restore", json={"stamp": stamp})).json()
            assert view["messages"][-1]["id"] == regen["reply_id"]

            # Delete the chat, then undo; the designer remembers it as last opened until then.
            await c.post(f"/api/chats/{cid}/open")
            assert (await c.get("/api/designer/last")).json()["chat_id"] == cid
            await c.delete(f"/api/chats/{cid}")
            assert (await c.get(f"/api/chats/{cid}")).status_code == 404
            assert (await c.get("/api/chats")).json() == []
            await c.post(f"/api/chats/{cid}/restore")
            assert [x["id"] for x in (await c.get("/api/chats")).json()] == [cid]

    asyncio.run(go())


def test_feedback_events(home):
    from logo_gen import library

    chat = chats.create()
    img_path = home / "output" / "generated" / "x.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (4, 4)).save(img_path)
    img = library.add_image(img_path, "generated", chat_id=chat["id"], prompt="p", concept="c", model="m/x")
    chats.add_feedback("like", image_id=img["id"])
    assert chats.current_ratings([img["id"]]) == {img["id"]: "like"}
    chats.add_feedback("unrate", image_id=img["id"])
    assert chats.current_ratings([img["id"]]) == {}
    chats.add_feedback("pick", image_id=img["id"])
    events = chats.list_feedback()
    assert [e["kind"] for e in events] == ["like", "unrate", "pick"]
    assert events[0]["chat_id"] == chat["id"] and events[0]["prompt"] == "p" and events[0]["model"] == "m/x"
    with pytest.raises(ValueError):
        chats.add_feedback("love", image_id=img["id"])
