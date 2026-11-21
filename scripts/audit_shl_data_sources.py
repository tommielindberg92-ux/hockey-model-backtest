
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE_URL = "https://www.shl.se/api/statistics-v2"
SEASONS = [2000, 2010, 2016, 2020, 2024, 2025, 2026]

MODULES = [
    "games_games",
    "teams_result",
    "teams_powerplay",
    "teams_penaltyKilling",
    "teams_shotsForward",
    "teams_shotsAgainst",
    "teams_faceoff",
    "goalkeepers_summary",
    "goalkeepers_shotAgainst",
    "players_extendedMetrics",
]

def fetch_json(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; HockeyDataAudit/1.0)",
            "Accept": "application/json",
        },
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read().decode("utf-8")
                return json.loads(body), None
        except Exception as error:
            if attempt == 2:
                return None, str(error)
            time.sleep(2)

def find_lists(obj, path="$"):
    found = []

    if isinstance(obj, list):
        dictionaries = [item for item in obj if isinstance(item, dict)]
        if dictionaries:
            found.append({
                "path": path,
                "count": len(obj),
                "rows": dictionaries,
            })
        for index, item in enumerate(obj[:3]):
            found.extend(find_lists(item, f"{path}[{index}]"))

    elif isinstance(obj, dict):
        for key, value in obj.items():
            found.extend(find_lists(value, f"{path}.{key}"))

    return found

def find_metadata(obj):
    metadata = {}

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = key.lower().replace("_", "")
                if normalized in (
                    "totalcount",
                    "count",
                    "total",
                    "page",
                    "pagesize",
                    "offset",
                ):
                    if isinstance(item, (int, float, str)):
                        metadata[key] = item
                walk(item)
        elif isinstance(value, list):
            for item in value[:3]:
                walk(item)

    walk(obj)
    return metadata

def flatten_row(row, prefix=""):
    result = {}

    if isinstance(row, dict):
        for key, value in row.items():
            name = f"{prefix}.{key}" if prefix else str(key)

            if isinstance(value, dict):
                result.update(flatten_row(value, name))
            elif isinstance(value, list):
                result[name] = value
            else:
                result[name] = value

    return result

def is_populated(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, dict)):
        return len(value) > 0
    return True

def audit_module(season, module):
    params = urllib.parse.urlencode({
        "count": 1000,
        "season": season,
        "gameType": "regular",
    })

    url = f"{BASE_URL}/stats-info/{module}?{params}"
    payload, error = fetch_json(url)

    result = {
        "season": season,
        "module": module,
        "url": url,
        "error": error,
    }

    if error or payload is None:
        return result

    result["response_type"] = type(payload).__name__

    if isinstance(payload, dict):
        result["top_level_keys"] = list(payload.keys())
    else:
        result["top_level_keys"] = []

    result["metadata"] = find_metadata(payload)

    candidates = find_lists(payload)
    candidates.sort(key=lambda item: len(item["rows"]), reverse=True)

    if not candidates:
        result["returned_rows"] = 0
        result["data_found"] = False
        result["sample"] = str(payload)[:1500]
        return result

    best = candidates[0]
    rows = best["rows"]

    result["data_path"] = best["path"]
    result["returned_rows"] = len(rows)
    result["data_found"] = True
    result["sample_row"] = rows[0]

    flattened = [flatten_row(row) for row in rows]
    all_fields = sorted({
        field
        for row in flattened
        for field in row
    })

    coverage = {}

    for field in all_fields:
        populated = sum(
            1 for row in flattened
            if is_populated(row.get(field))
        )
        coverage[field] = {
            "populated": populated,
            "total": len(flattened),
            "percentage": round(
                100 * populated / len(flattened), 1
            ) if flattened else 0,
        }

    result["field_coverage"] = coverage

    metadata = result["metadata"]
    total_count = None

    for key, value in metadata.items():
        if key.lower().replace("_", "") == "totalcount":
            try:
                total_count = int(value)
            except (TypeError, ValueError):
                pass

    result["total_count_reported"] = total_count

    if total_count is not None:
        result["possible_truncation"] = (
            total_count > len(rows) or len(rows) >= 1000
        )
    else:
        result["possible_truncation"] = len(rows) >= 1000

    return result

def main():
    started = datetime.now(timezone.utc).isoformat()

    print("Hämtar SHL:s säsongs- och layoutinformation...")
    layout_url = f"{BASE_URL}/layout-info"
    layout, layout_error = fetch_json(layout_url)

    report = {
        "created_utc": started,
        "layout_url": layout_url,
        "layout_error": layout_error,
        "layout": layout,
        "requested_seasons": SEASONS,
        "modules": MODULES,
        "results": [],
    }

    for season in SEASONS:
        for module in MODULES:
            print(f"Kontrollerar säsong {season}: {module}")

            result = audit_module(season, module)
            report["results"].append(result)

            print(
                "  Rader:",
                result.get("returned_rows", 0),
                "| Fel:",
                result.get("error"),
            )

            time.sleep(1)

    with open("shl_data_audit.json", "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    successful = [
        item for item in report["results"]
        if not item.get("error")
    ]

    populated = [
        item for item in successful
        if item.get("returned_rows", 0) > 0
    ]

    print("\n========== SHL AUDIT KLAR ==========")
    print("Kontrollerade kombinationer:", len(report["results"]))
    print("Lyckade API-anrop:", len(successful))
    print("Anrop med data:", len(populated))
    print("Rapport: shl_data_audit.json")

if __name__ == "__main__":
    main()
