
#!/usr/bin/env python3
"""Audit public SHL statistics API across representative seasons."""

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://www.shl.se/api/statistics-v2"
OUTPUT_FILE = Path("shl_data_audit.json")
COUNT = 1000

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

TARGET_YEARS = [2010, 2016, 2020, 2024, 2025, 2026]

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; SHLDataAudit/1.0)",
    "Accept": "application/json, text/plain, */*",
})


def request_json(url, params=None, retries=3):
    """Request JSON and retain useful diagnostics on failures."""
    last_error = None

    for attempt in range(retries):
        try:
            response = session.get(
                url,
                params=params,
                timeout=45,
            )

            body = response.text or ""

            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}; "
                    f"body={body[:500]!r}"
                )

            if not body.strip():
                raise RuntimeError("HTTP 200 but response body is empty")

            try:
                return response.json(), {
                    "http_status": response.status_code,
                    "response_bytes": len(response.content),
                    "request_url": response.url,
                }
            except ValueError as exc:
                raise RuntimeError(
                    f"Invalid JSON: {exc}; body={body[:500]!r}"
                ) from exc

        except Exception as exc:
            last_error = str(exc)
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return None, {
        "error": last_error or "Unknown request error",
        "request_url": url,
    }


def walk_dicts(value):
    """Yield every dictionary in a nested JSON structure."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def get_season_catalog(layout):
    """Extract SHL regular-season UUIDs from layout-info."""
    catalog = {}

    for item in walk_dicts(layout):
        season = item.get("season")
        game_type = item.get("gameType")
        series = item.get("series")
        uuid = item.get("ssgtUuid")

        if not all(isinstance(x, dict) for x in (
            season, game_type, series
        )):
            continue

        season_code = season.get("code")
        game_type_code = game_type.get("code")
        series_code = series.get("code")

        if not uuid or not season_code:
            continue

        try:
            year = int(season_code)
        except (TypeError, ValueError):
            continue

        if series_code == "SHL" and game_type_code == "regular":
            catalog[year] = {
                "ssgt_uuid": uuid,
                "season_name": season.get("name"),
                "state": item.get("state"),
            }

    return catalog


def choose_sample_years(available_years):
    """Choose representative years that actually exist in the catalog."""
    if not available_years:
        return []

    chosen = set()

    # Include the oldest available season.
    chosen.add(min(available_years))

    # Select the closest available year to each target.
    for target in TARGET_YEARS:
        chosen.add(min(
            available_years,
            key=lambda year: (abs(year - target), year),
        ))

    return sorted(chosen)


def find_largest_record_list(payload):
    """Find the largest list of dictionaries in an API response."""
    candidates = []

    def visit(value, path="$"):
        if isinstance(value, list):
            dict_rows = [x for x in value if isinstance(x, dict)]
            if dict_rows:
                candidates.append((len(dict_rows), path, dict_rows))

            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}")

    visit(payload)

    if not candidates:
        return [], None

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, path, rows = candidates[0]
    return rows, path


def find_total_count(payload):
    """Find totalCount or similar metadata anywhere in the response."""
    for item in walk_dicts(payload):
        for key in ("totalCount", "total_count", "total"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return None


def flatten_keys(value, prefix=""):
    """Return flattened field names for coverage analysis."""
    found = set()

    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            found.add(name)
            found.update(flatten_keys(child, name))

    elif isinstance(value, list):
        for child in value[:3]:
            found.update(flatten_keys(child, prefix))

    return found


def audit_module(year, season_info, module):
    url = f"{BASE_URL}/stats-info/{module}"
    params = {
        "count": COUNT,
        "ssgtUuid": season_info["ssgt_uuid"],
    }

    payload, request_info = request_json(url, params=params)

    result = {
        "season_year": year,
        "season_name": season_info.get("season_name"),
        "module": module,
        **request_info,
    }

    if payload is None:
        result.update({
            "success": False,
            "row_count": 0,
            "total_count": None,
            "truncated_or_incomplete": None,
            "fields": [],
            "field_coverage": {},
        })
        return result

    rows, rows_path = find_largest_record_list(payload)
    total_count = find_total_count(payload)

    field_counts = Counter()
    for row in rows:
        field_counts.update(flatten_keys(row))

    coverage = {
        field: round(count / len(rows) * 100, 1)
        for field, count in sorted(field_counts.items())
    }

    truncated = False
    if len(rows) >= COUNT:
        truncated = True
    if total_count is not None and len(rows) < total_count:
        truncated = True

    result.update({
        "success": True,
        "response_type": type(payload).__name__,
        "top_level_keys": (
            list(payload.keys())[:50]
            if isinstance(payload, dict)
            else None
        ),
        "rows_path": rows_path,
        "row_count": len(rows),
        "total_count": total_count,
        "truncated_or_incomplete": truncated,
        "fields": sorted(field_counts.keys()),
        "field_coverage": coverage,
    })

    return result


def main():
    started = datetime.now(timezone.utc).isoformat()

    print("Fetching SHL layout-info...")
    layout, layout_info = request_json(
        f"{BASE_URL}/layout-info"
    )

    if layout is None:
        raise RuntimeError(
            f"Could not retrieve SHL layout-info: {layout_info}"
        )

    catalog = get_season_catalog(layout)
    available_years = sorted(catalog)
    sample_years = choose_sample_years(available_years)

    print(f"Available SHL regular seasons: {len(available_years)}")
    print(f"Sampled seasons: {sample_years}")

    if not sample_years:
        raise RuntimeError(
            "No SHL regular seasons with ssgtUuid were found in layout-info"
        )

    results = []

    for year in sample_years:
        season_info = catalog[year]

        for module in MODULES:
            print(f"Auditing {year}: {module}")
            result = audit_module(year, season_info, module)
            results.append(result)

            if not result.get("success"):
                print(f"  FAILED: {result.get('error')}")
            else:
                print(
                    f"  rows={result.get('row_count')}, "
                    f"total={result.get('total_count')}, "
                    f"fields={len(result.get('fields', []))}, "
                    f"truncated={result.get('truncated_or_incomplete')}"
                )

            # Be gentle to the public API.
            time.sleep(1)

    success_count = sum(
        1 for result in results if result.get("success")
    )

    report = {
        "audit_started_utc": started,
        "audit_finished_utc": datetime.now(timezone.utc).isoformat(),
        "source": BASE_URL,
        "method": (
            "layout-info lookup followed by stats-info requests "
            "using ssgtUuid"
        ),
        "layout_request": layout_info,
        "available_regular_season_years": available_years,
        "sampled_seasons": [
            {
                "year": year,
                **catalog[year],
            }
            for year in sample_years
        ],
        "modules": MODULES,
        "request_count": len(results),
        "successful_requests": success_count,
        "failed_requests": len(results) - success_count,
        "results": results,
    }

    OUTPUT_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("SHL audit finished.")
    print(f"Successful requests: {success_count}/{len(results)}")
    print(f"Report saved to: {OUTPUT_FILE}")

    if success_count == 0:
        raise RuntimeError(
            "All stats-info requests failed. See report diagnostics."
        )


if __name__ == "__main__":
    main()
