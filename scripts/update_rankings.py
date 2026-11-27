"""Refresh the bundled leaderboard snapshot (src/logo_gen/rankings_snapshot.json).

Run: uv run python scripts/update_rankings.py
The app fetches fresh data itself and caches it for a day. This snapshot is
only the offline fallback, so refresh it now and then and commit it.
"""

import asyncio
import json
import sys

from logo_gen import rankings


async def main() -> None:
    data = await rankings.fetch_lmarena()
    data.pop("fetched_at", None)
    rankings.SNAPSHOT_FILE.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    missing = sorted({n for n in rankings.ALIASES.values() if n not in data["models"]})
    print(f"{len(data['models'])} models, published {data['source']['published']}")
    if missing:
        print("Aliases pointing at names no longer on the leaderboard:", ", ".join(missing))
        sys.exit(1)


asyncio.run(main())
