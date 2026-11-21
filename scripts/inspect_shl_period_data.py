"""Inspect the actual structure of the SHL schedule response."""

import json
import requests

SCHEDULE_URL = "https://www.shl.se/api/sports-v2/game-schedule"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.shl.se/",
}

PARAMS = {
    "seasonUuid": "qeb-73bZkIm9A",
    "seriesUuid": "qQ9-bb0bzEWUk",
    "gameTypeUuid": "qQ9-af37Ti40B",
}

response = requests.get(
    SCHEDULE_URL,
    params=PARAMS,
    headers=HEADERS,
    timeout=30,
)

print("HTTP status:", response.status_code)
response.raise_for_status()

data = response.json()

print("\nTop-level type:", type(data).__name__)

if isinstance(data, dict):
    print("Top-level keys:", list(data.keys()))

    for key, value in data.items():
        print(f"\nKEY: {key}")
        print("TYPE:", type(value).__name__)

        if isinstance(value, list):
            print("NUMBER OF ITEMS:", len(value))

            if value:
                print("FIRST ITEM TYPE:", type(value[0]).__name__)
                print(
                    json.dumps(
                        value[0],
                        ensure_ascii=False,
                        indent=2,
                    )[:12000]
                )

        elif isinstance(value, dict):
            print(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                )[:6000]
            )

        else:
            print(str(value)[:2000])

elif isinstance(data, list):
    print("Number of items:", len(data))

    if data:
        print(
            json.dumps(
                data[0],
                ensure_ascii=False,
                indent=2,
            )[:12000]
        )

else:
    print("Response:")
    print(str(data)[:12000])

print("\nSchedule structure inspection finished.")