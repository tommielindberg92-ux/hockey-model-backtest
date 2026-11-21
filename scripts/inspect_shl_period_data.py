
"""Discover SHL game schedule and period-data API endpoints."""

import json
import requests

BASE = "https://www.shl.se/api"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.shl.se/",
}

session = requests.Session()
session.headers.update(HEADERS)

tests = [
    (
        "sports-v2 schedule, no parameters",
        f"{BASE}/sports-v2/game-schedule",
        {},
    ),
    (
        "sports-v2 schedule, season 2025",
        f"{BASE}/sports-v2/game-schedule",
        {"season": "2025"},
    ),
    (
        "sports-v2 schedule, seasonUuid",
        f"{BASE}/sports-v2/game-schedule",
        {"seasonUuid": "qeb-73qtP8d19"},
    ),
    (
        "gameday schedule",
        f"{BASE}/gameday/game-schedule",
        {},
    ),
    (
        "gameday overview using game ID",
        f"{BASE}/gameday/game-overview/20460",
        {},
    ),
    (
        "gameday play-by-play using game ID",
        f"{BASE}/gameday/play-by-play/20460",
        {},
    ),
    (
        "statistics game details",
        f"https://www.shl.se/api/statistics-v2/stats-info/games_games",
        {"count": "3", "ssgtUuid": "qeb-73qtP8d19"},
    ),
]

for label, url, params in tests:
    print("\n" + "=" * 75)
    print(label)
    print("URL:", url)
    print("Parameters:", params)

    try:
        response = session.get(
            url,
            params=params,
            timeout=30,
        )

        print("HTTP status:", response.status_code)
        print("Final URL:", response.url)
        print("Content type:", response.headers.get("Content-Type"))

        try:
            payload = response.json()
            print(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                )[:7000]
            )
        except ValueError:
            print(response.text[:3000])

    except requests.RequestException as error:
        print("REQUEST ERROR:", repr(error))

print("\nAPI discovery finished.")
