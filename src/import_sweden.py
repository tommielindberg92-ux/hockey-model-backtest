
"""Importera HockeyAllsvenskan-resultat efter ordinarie tid (60 minuter)."""

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://hockeyallsvenskan.se/api/all-time-games-stats"
SOURCE = "HockeyAllsvenskan officiellt API"

COLUMNS = [
    "date", "league", "home_team", "away_team",
    "reg_home_goals", "reg_away_goals", "decided_in", "source",
]


def get_games(season, phase):
    # API:ets säsongsparameter använder startåret.
    params = urlencode({
        "season": season - 1,
        "phase": phase,
        "team_side": "all",
        "team": "all",
        "against_team": "all",
        "all_results": "all",
        "game_status": "played",
        "decided_in": "all",
    })

    request = Request(
        f"{BASE_URL}?{params}",
        headers={"User-Agent": "hockey-model-backtest/1.0"},
    )

    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "games", "results", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]

    raise ValueError("API-svaret hade ett oväntat format.")


def parse_period_scores(value):
    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, list) or len(value) < 3:
        raise ValueError("Saknade periodresultat för matchen.")

    home_goals = 0
    away_goals = 0

    for period in value[:3]:
        if isinstance(period, dict):
            home = period.get("Home", period.get("home"))
            away = period.get("Away", period.get("away"))
        elif isinstance(period, (list, tuple)) and len(period) >= 2:
            home, away = period[0], period[1]
        else:
            raise ValueError("Ogiltigt format på periodresultat.")

        if home is None or away is None:
            raise ValueError("Saknade mål i periodresultatet.")

        home_goals += int(home)
        away_goals += int(away)

    return home_goals, away_goals


def import_season(season, phase, output_file):
    games = get_games(season, phase)
    imported = []

    for game in games:
        decided_in = str(game.get("decided_in", "")).upper()

        if decided_in not in {"FT", "OT", "SO"}:
            continue

        try:
            home_goals, away_goals = parse_period_scores(
                game.get("period_scores")
            )

            if decided_in == "FT" and home_goals == away_goals:
                raise ValueError("FT-match oavgjord efter 60 minuter.")

            if decided_in in {"OT", "SO"} and home_goals != away_goals:
                raise ValueError("OT/SO-match ej oavgjord efter 60 minuter.")

            row = {
                "date": game["date"],
                "league": "HockeyAllsvenskan",
                "home_team": game["home_team_code"],
                "away_team": game["away_team_code"],
                "reg_home_goals": home_goals,
                "reg_away_goals": away_goals,
                "decided_in": decided_in,
                "source": SOURCE,
            }
            imported.append(row)

        except (KeyError, TypeError, ValueError) as exc:
            print(f"VARNING: match hoppades över: {exc}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if output_file.exists():
        with output_file.open(encoding="utf-8", newline="") as file:
            existing = list(csv.DictReader(file))

    combined = {
        (r["date"], r["league"], r["home_team"], r["away_team"]): r
        for r in existing
    }

    for row in imported:
        key = (row["date"], row["league"], row["home_team"], row["away_team"])
        combined[key] = row

    ordered = sorted(combined.values(), key=lambda r: r["date"])

    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)

    print(f"Importerade matcher: {len(imported)}")
    print(f"Totalt i filen: {len(ordered)}")
    print(f"Fil: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Importera HockeyAllsvenskan efter 60 minuter."
    )
    parser.add_argument("--season", type=int, required=True,
                        help="Säsongens slutår, exempelvis 2026")
    parser.add_argument("--phase", default="HA",
                        choices=["HA", "HAPlayoff", "HAkval"])
    parser.add_argument("--output", default="data/ha_matches.csv")
    args = parser.parse_args()

    import_season(args.season, args.phase, Path(args.output))


if __name__ == "__main__":
    main()
