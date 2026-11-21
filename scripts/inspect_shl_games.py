"""Inspect sample SHL game records from the public statistics API."""

import json
import requests

BASE_URL = "https://www.shl.se/api/statistics-v2"
SEASON_UUID = "qeb-73qtP8d19"  # SHL 2024/25

URL = f"{BASE_URL}/stats-info/games_games"

params = {
    "count": 3,
    "ssgtUuid": SEASON_UUID,
}

headers = {
    "User-Agent": "Mozilla/5.0 (compatible; SHLDataInspector/1.0)",
    "Accept": "application/json, text/plain, */*",
}

response = requests.get(
    URL,
    params=params,
    headers=headers,
    timeout=45,
)

response.raise_for_status()
payload = response.json()

if not isinstance(payload, list):
    raise ValueError(
        f"Unexpected top-level type: {type(payload).__name__}"
    )

rows = []

for block in payload:
    if isinstance(block, dict) and isinstance(block.get("stats"), list):
        rows.extend(block["stats"])

if not rows:
    raise ValueError("No game rows found under payload[*].stats")

print(f"Request URL: {response.url}")
print(f"Game rows returned: {len(rows)}")

for index, row in enumerate(rows[:3], start=1):
    print(f"\n--- GAME {index} ---")
    print(json.dumps(row, ensure_ascii=False, indent=2))