
"""Find SHL season, series and game-type UUIDs."""

import json
import requests

URL = "https://www.shl.se/api/statistics-v2/layout-info"

response = requests.get(
    URL,
    headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    timeout=30,
)
response.raise_for_status()
data = response.json()

print("Layout downloaded successfully.")
print("Top-level type:", type(data).__name__)


def inspect(value, path="root"):
    if isinstance(value, dict):
        keys = {str(key).lower(): key for key in value}

        if "uuid" in keys:
            selected = {}
            for key, item in value.items():
                name = str(key).lower()
                if (
                    name in {"uuid", "code", "name", "displayname",
                             "translation", "type", "season", "series"}
                    or "name" in name
                ):
                    if isinstance(item, (str, int, float, bool)):
                        selected[key] = item

            text = json.dumps(selected, ensure_ascii=False).lower()
            if any(word in text for word in (
                "shl", "regular", "grund", "season", "game",
                "match", "serie", "2025", "2026"
            )):
                print("\nPATH:", path)
                print(json.dumps(selected, ensure_ascii=False))

        for key, item in value.items():
            inspect(item, f"{path}.{key}")

    elif isinstance(value, list):
        for index, item in enumerate(value):
            inspect(item, f"{path}[{index}]")


inspect(data)

print("\nUUID discovery finished.")
