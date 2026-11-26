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
