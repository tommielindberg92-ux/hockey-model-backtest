
"""Inspect SHL game overview and play-by-play endpoints."""

import json
import requests

BASE = "https://www.shl.se/api/gameday"

# Verified game UUID from the SHL schedule response.
GAME_UUID = "p2qoh7wot5"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.shl.se/",
}

ENDPOINTS = [
    "game-overview",
    "play-by-play",
]

for endpoint in ENDPOINTS:
    url = f"{BASE}/{endpoint}/{GAME_UUID}"

    print("\n" + "=" * 70)
    print("ENDPOINT:", endpoint)
    print("URL:", url)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
        )

        print("HTTP:", response.status_code)
        response.raise_for_status()

        try:
            data = response.json()
            print("Response type:", type(data).__name__)
            if isinstance(data, dict):
                print("Top-level keys:", list(data.keys()))

            print(json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            )[:15000])

        except ValueError:
            print(response.text[:15000])

    except Exception as error:
        print("ERROR:", repr(error))

print("\nEndpoint inspection finished.")
