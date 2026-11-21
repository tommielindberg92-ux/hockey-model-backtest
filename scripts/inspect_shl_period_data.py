
"""Test SHL game schedule with confirmed UUIDs."""

import json
import requests

URL = "https://www.shl.se/api/sports-v2/game-schedule"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.shl.se/",
}

SEASONS = {
    "2024/2025": "qeb-73bZkIm9A",
    "2025/2026": "xs4m9qupsi",
    "2026/2027": "ndcf81nlb3",
}

for season_name, season_uuid in SEASONS.items():
    params = {
        "seasonUuid": season_uuid,
        "seriesUuid": "qQ9-bb0bzEWUk",
        "gameTypeUuid": "qQ9-af37Ti40B",
    }

    print("\n" + "=" * 70)
    print("SEASON:", season_name)

    try:
        response = requests.get(
            URL,
            params=params,
            headers=HEADERS,
            timeout=30,
        )

        print("HTTP:", response.status_code)
        response.raise_for_status()

        data = response.json()
        print("Response type:", type(data).__name__)

        if isinstance(data, dict):
            print("Top-level keys:", list(data.keys()))

        if isinstance(data, list):
            print("Number of entries:", len(data))

        print(json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        )[:12000])

    except Exception as error:
        print("ERROR:", repr(error))

print("\nSchedule tests finished.")
