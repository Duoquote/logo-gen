import asyncio

import pytest

from logo_gen import config, generator


@pytest.fixture
def settings(monkeypatch):
    s = config.Settings(max_images_cloud=3, max_images_local=8, model_caps={"a/special": 1})
    monkeypatch.setattr(generator, "get_settings", lambda: s)
    return s


def test_cap_for(settings):
    assert settings.cap_for("a/b") == 3
    assert settings.cap_for("local/zimage-turbo") == 8
    assert settings.cap_for("a/special") == 1


def test_plan_trims_to_the_cap(settings):
    prompts = [{"prompt": str(i)} for i in range(4)]
    work, notes = generator.plan(["a/b", "local/x", "a/special"], prompts, 2)
    # 4 directions x 2 = 8 > 3: one image per direction, and only 3 directions fit
    assert len(work["a/b"][0]) == 3 and work["a/b"][1] == 1
    # local cap 8 fits exactly
    assert len(work["local/x"][0]) == 4 and work["local/x"][1] == 2
    assert len(work["a/special"][0]) == 1
    assert len(notes) == 2


def test_limiter_backs_off_and_recovers():
    async def go():
        lim = generator.AdaptiveLimiter(4)
        lim.backoff()
        assert lim.limit == 2
        lim.backoff()
        lim.backoff()
        assert lim.limit == 1
        for _ in range(3):
            await lim.success()
        assert lim.limit == 3
        # never more than `limit` inside at once
        peak = 0

        async def task():
            nonlocal peak
            async with lim:
                peak = max(peak, lim.active)
                await asyncio.sleep(0.01)

        await asyncio.gather(*(task() for _ in range(10)))
        assert peak <= 3

    asyncio.run(go())


def test_run_renders_per_model_and_records_what_was_sent(settings, monkeypatch, tmp_path):
    """A cloud run with a brief: the model gets its own prompt and params, and the image row keeps them."""
    import io

    from PIL import Image

    from logo_gen import library, paths
    from logo_gen.clients import openrouter
    from logo_gen.jobs import Job

    monkeypatch.setattr(paths, "DB_FILE", tmp_path / "t.sqlite3")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    monkeypatch.setattr(paths, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(paths, "GENERATED_DIR", tmp_path / "out" / "generated")
    monkeypatch.setattr(library, "_conn", None)

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format="PNG")
    sent = []

    async def info(model):
        return {"params": {"quality": {"values": ["low", "medium"]}}, "max_n": 1}

    async def fake_generate(prompt, model, **kw):
        sent.append((prompt, kw.get("extra")))
        return [openrouter.ImageResult(buf.getvalue(), "image/png")], {"cost": 0.01}

    monkeypatch.setattr(openrouter, "model_info", info)
    monkeypatch.setattr(openrouter, "generate_images", fake_generate)
    brief = {"concept": "Fox", "mark_type": "pictorial", "subject": "a fox head from three triangles", "style": "flat vector",
             "palette": [{"name": "orange", "hex": "#F97316"}], "background": "white", "text": "", "avoid": []}
    job = Job(id="j1", kind="generate", title="t")
    try:
        asyncio.run(generator.run(job, [{"concept": "Fox", "style": "flat", "prompt": "summary", "brief": brief}],
                                  ["openai/gpt-image-1-mini"], 1))
        assert len(job.image_ids) == 1 and not job.errors
        prompt, extra = sent[0]
        assert prompt.startswith("Design a logo symbol") and extra == {"quality": "medium"}
        img = library.get_image(job.image_ids[0])
        assert img["prompt"] == prompt and img["profile"] == "gpt-image"
        assert img["brief"]["subject"] == brief["subject"] and img["params"] == {"quality": "medium"}
        assert job.result["rendered"][0]["model"] == "openai/gpt-image-1-mini"
    finally:
        if library._conn is not None:
            library._conn.close()
        library._conn = None
