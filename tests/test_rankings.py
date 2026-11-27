import math

from logo_gen import rankings
from logo_gen.rankings import ALIASES, pareto_front, popularity_id, rank_models, value_score


def test_value_formula():
    # $0.01 is the reference: value equals the rating.
    assert value_score(1200, 0.01) == 1200
    # Twice the price costs 50 Elo, half the price earns 50.
    assert math.isclose(value_score(1200, 0.02), 1150)
    assert math.isclose(value_score(1200, 0.005), 1250)
    # Free and zero prices don't blow up: they count as $0.001.
    assert value_score(1200, 0) == value_score(1200, 0.001) == value_score(1200, -1)
    assert math.isfinite(value_score(1200, 0))


def test_pareto_front():
    items = {
        "cheap-ok": (1100, 0.007),
        "mid-good": (1200, 0.04),
        "pricey-best": (1300, 0.2),
        "dominated": (1150, 0.05),  # worse and pricier than mid-good
        "tie-pricier": (1200, 0.05),  # same quality as mid-good, costs more
    }
    assert pareto_front(items) == {"cheap-ok", "mid-good", "pricey-best"}


def test_every_alias_is_on_the_bundled_leaderboard():
    names = set(rankings.snapshot()["models"])
    assert {n for n in ALIASES.values()} <= names


def test_popularity_slugs_map_to_catalog_ids():
    ids = {"google/gemini-3.1-flash-image", "meta/muse-image", "openai/gpt-image-2", "openai/gpt-image-2.5-sunburst",
           "recraft/recraft-v4.1-flash-v2", "recraft/recraft-v4.1-flash-v2:free"}
    assert popularity_id("google/gemini-3.1-flash-image-20260528", ids) == "google/gemini-3.1-flash-image"
    assert popularity_id("meta/muse-image-1.0-eval-20260824", ids) == "meta/muse-image"
    assert popularity_id("openai/gpt-image-2.5-sunburst-20260908", ids) == "openai/gpt-image-2.5-sunburst"
    assert popularity_id("openai/gpt-image-2", ids) == "openai/gpt-image-2"
    assert popularity_id("recraft/recraft-v4.1-flash-v2-20261113:free", ids) == "recraft/recraft-v4.1-flash-v2:free"
    assert popularity_id("someone/unknown-model-20260101", ids) is None


def test_rank_models_handles_unranked_missing_and_free_prices():
    board = {"models": {
        "A": {"overall": {"rating": 1300, "ci": [1290, 1310], "votes": 10, "rank": 1}},
        "B": {"commercial_design": {"rating": 1250, "ci": [1240, 1260], "votes": 5, "rank": 2},
              "overall": {"rating": 1100, "ci": [1, 2], "votes": 5, "rank": 9}},
    }}
    old = dict(ALIASES)
    ALIASES.update({"x/a": "A", "x/b": "B", "x/b:free": "B"})
    try:
        out = rank_models(
            [
                {"id": "x/a", "price": 0.2},
                {"id": "x/b", "price": None},  # price unknown: ranked, but no value or Pareto
                {"id": "x/b:free", "price": 0.0},
                {"id": "x/unranked", "price": 0.01},
            ],
            board,
            {"tokens": {"x/b-20260101": 5.0, "x/a-20260101": 9.0}},
        )
    finally:
        ALIASES.clear()
        ALIASES.update(old)
    assert out["x/b"]["category"] == "commercial_design" and out["x/b"]["rating"] == 1250  # logo-relevant category first
    assert out["x/a"]["rank"] == 1 and {out["x/b"]["rank"], out["x/b:free"]["rank"]} == {2, 3}
    assert out["x/b"]["value"] is None and not out["x/b"]["pareto"]
    assert out["x/b:free"]["pareto"] and out["x/a"]["pareto"]  # free and good, or best: both on the frontier
    assert out["x/unranked"] == out["x/unranked"] | {"ranked": False, "rank": None, "value": None, "pareto": False}
    assert out["x/a"]["popularity_rank"] == 1 and out["x/b"]["popularity_rank"] == 2
