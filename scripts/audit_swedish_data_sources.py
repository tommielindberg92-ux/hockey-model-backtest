
"""Audit av tillgängliga historiska HockeyAllsvenskan-data."""

import json
from collections import Counter
from datetime import datetime
from urllib.request import Request, urlopen

URL = "https://hockeyallsvenskan.se/api/all-time-games-stats"
SEASONS = [2010, 2016, 2020, 2024, 2026]


def fetch_season(season):
    body = {
        "league": "HA",
        "phase": "HA",
        "team_side": "all",
        "team": "all",
        "against_team": "all",
        "all_results": "all",
        "game_status": "played",
        "decided_in": "all",
        "categories": "all",
        "page": 1,
        "page_size": 1000,
        "season": str(season - 1),
    }

    request = Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "User-Agent": "hockey-model-backtest-data-audit/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_rows(payload):
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "games", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

    return []


def main():
    report = {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source": URL,
        "seasons": {},
    }

    for season in SEASONS:
        print(f"Kontrollerar säsong {season - 1}/{season}...")
        try:
            payload = fetch_season(season)
            rows = extract_rows(payload)

            all_keys = Counter()
            for row in rows:
                if isinstance(row, dict):
                    all_keys.update(row.keys())

            report["seasons"][str(season)] = {
                "row_count": len(rows),
                "field_names": sorted(all_keys.keys()),
                "field_frequency": dict(all_keys),
                "sample_rows": rows[:2],
                "top_level_type": type(payload).__name__,
                "top_level_keys": (
                    sorted(payload.keys())
                    if isinstance(payload, dict)
                    else []
                ),
            }

            print(f"  Matcher/rader: {len(rows)}")
            print(f"  Fält: {', '.join(sorted(all_keys.keys()))}")

        except Exception as exc:
            report["seasons"][str(season)] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            print(f"  FEL: {exc}")

    with open("ha_data_audit.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nAudit klar. Rapport sparad i ha_data_audit.json")


if __name__ == "__main__":
    main()
