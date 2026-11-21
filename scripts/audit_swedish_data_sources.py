
"""Detaljerad täckningsanalys av HockeyAllsvenskans historiska matchdata."""

import json
from collections import Counter
from datetime import datetime, timezone
from urllib.request import Request, urlopen

URL = "https://hockeyallsvenskan.se/api/all-time-games-stats"
SEASONS = [2010, 2016, 2020, 2024, 2026]

CATEGORIES = {
    "skott": ["shot", "shots", "skott"],
    "powerplay": ["power", "powerplay", "pp_"],
    "utvisningar": ["penalt", "penalty", "utvis"],
    "periodresultat": ["p1", "p2", "p3", "period"],
    "faceoffs": ["faceoff", "tekning"],
    "corsi_fenwick": ["corsi", "fenwick", "pdo"],
    "malvaktsdata": ["goalie", "save", "saves", "svs"],
    "publik": ["attendance", "spectator", "publik"],
}


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
            "User-Agent": "hockey-model-backtest-data-audit/2.0",
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


def flatten_fields(value, prefix=""):
    """Returnerar alla bladfält i en eventuell nästlad matchpost."""
    fields = {}

    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            fields.update(flatten_fields(child, path))

    elif isinstance(value, list):
        if value:
            for index, child in enumerate(value):
                path = f"{prefix}[{index}]"
                fields.update(flatten_fields(child, path))
        elif prefix:
            fields[prefix] = None

    elif prefix:
        fields[prefix] = value

    return fields


def has_value(value):
    return value is not None and value != ""


def category_matches(field_name, keywords):
    name = field_name.lower()
    return any(word in name for word in keywords)


def main():
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": URL,
        "method": "Non-empty field coverage per returned match row",
        "seasons": {},
    }

    for season in SEASONS:
        print(f"\nKontrollerar säsong {season - 1}/{season}...")

        try:
            payload = fetch_season(season)
            rows = extract_rows(payload)

            field_counts = Counter()
            category_counts = Counter()

            for row in rows:
                if not isinstance(row, dict):
                    continue

                fields = flatten_fields(row)

                for field, value in fields.items():
                    if has_value(value):
                        field_counts[field] += 1

                for category, keywords in CATEGORIES.items():
                    if any(
                        category_matches(field, keywords)
                        and has_value(value)
                        for field, value in fields.items()
                    ):
                        category_counts[category] += 1

            total = len(rows)

            coverage = {}
            for field, count in sorted(field_counts.items()):
                coverage[field] = {
                    "matches_with_value": count,
                    "coverage_percent": round(
                        100 * count / total, 2
                    ) if total else 0,
                }

            category_report = {}
            for category in CATEGORIES:
                count = category_counts[category]
                category_report[category] = {
                    "matches_with_data": count,
                    "coverage_percent": round(
                        100 * count / total, 2
                    ) if total else 0,
                }

            report["seasons"][str(season)] = {
                "season_label": f"{season - 1}/{season}",
                "row_count": total,
                "category_coverage": category_report,
                "field_coverage": coverage,
                "top_level_keys": (
                    sorted(payload.keys())
                    if isinstance(payload, dict)
                    else []
                ),
                "sample_row": rows[0] if rows else None,
            }

            print(f"Matcher/rader: {total}")
            for category, values in category_report.items():
                print(
                    f"  {category}: "
                    f"{values['matches_with_data']}/{total} "
                    f"({values['coverage_percent']} %)"
                )

        except Exception as exc:
            report["seasons"][str(season)] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
            print(f"FEL: {exc}")

    with open("ha_data_audit.json", "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\nDetaljerad audit klar.")
    print("Rapport sparad i ha_data_audit.json")


if __name__ == "__main__":
    main()
