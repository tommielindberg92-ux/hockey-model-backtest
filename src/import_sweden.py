
"""Importera HockeyAllsvenskan-resultat efter ordinarie tid (60 minuter)."""

import argparse
import csv
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = "https://hockeyallsvenskan.se/api/all-time-games-stats"
SOURCE = "HockeyAllsvenskan officiellt API"

COLUMNS = [
    "date", "league", "home_team", "away_team",
    "reg_home_goals", "reg_away_goals", "decided_in", "source",
]


def get_games(season, phase):
    """Hämta en säsongs matcher via API:ets POST-anrop."""
    body = {
        "league": "HA",
        "phase": phase,
        "team_side": "all",
        "team": "all",
        "against_team": "all",
        "all_results": "all",
        "game_status": "played",
        "decided_in": "all",
        "categories": "all",
        "page": 1,
        "page_size": 1000,
        # API:et använder säsongens startår.
        "season": str(season - 1),
    }

    request = Request(
        BASE_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "User-Agent": "hockey-model-backtest/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "games", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]

    raise ValueError("API-svaret hade ett oväntat format.")


def parse_period_scores(value):
    """Summera endast period 1–3. Ignorera eventuell OT."""
    if isinstance(value, str):
        value = json.loads(value)

    if not isinstance(value, list) or len(value) < 3:
        raise ValueError("Saknade periodresultat.")

    home_goals = 0
    away_goals = 0

    for period in value[:3]:
        if not isinstance(period, dict):
            raise ValueError("Ogiltigt format på periodresultat.")

        home = period.get("Home")
        away = period.get("Away")

        if home is None or away is None:
            raise ValueError("Saknade mål i periodresultatet.")

        home = int(home)
        away = int(away)

        if home < 0 or away < 0:
            raise ValueError("Negativt antal mål i periodresultatet.")

        home_goals += home
        away_goals += away

    return home_goals, away_goals


def import_season(season, phase, output_file):
    games = get_games(season, phase)
    imported = []

    for game in games:
        decided_in = str(game.get("decided_in", "")).upper()

        # Walkover och ofullständiga matcher tas inte med.
        if decided_in not in {"FT", "OT", "SO"}:
            continue

        try:
            period_scores = game.get(
                "period_scores", game.get("PeriodScores")
            )
            home_goals, away_goals = parse_period_scores(period_scores)

            # FT ska ha en vinnare efter 60 minuter.
            if decided_in == "FT" and home_goals == away_goals:
                raise ValueError("FT-match oavgjord efter 60 minuter.")

            # OT och SO ska vara oavgjorda efter 60 minuter.
            if decided_in in {"OT", "SO"} and home_goals != away_goals:
                raise ValueError("OT/SO-match ej oavgjord efter 60 minuter.")

            date_value = game.get("date", game.get("Date"))
            home_team = game.get(
                "home_team_code", game.get("HomeTeam")
            )
            away_team = game.get(
                "away_team_code", game.get("AwayTeam")
            )

            if not all((date_value, home_team, away_team)):
                raise ValueError("Matchen saknar datum eller lag.")

            imported.append({
                "date": str(date_value),
                "league": "HockeyAllsvenskan",
                "home_team": str(home_team),
                "away_team": str(away_team),
                "reg_home_goals": home_goals,
                "reg_away_goals": away_goals,
                "decided_in": decided_in,
                "source": SOURCE,
            })

        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
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
    parser.add_argument(
        "--season", type=int, required=True,
        help="Säsongens slutår, exempelvis 2026"
    )
    parser.add_argument(
        "--phase", default="HA",
        choices=["HA", "HAPlayoff", "HAkval"]
    )
    parser.add_argument(
        "--output", default="data/ha_matches.csv"
    )
    args = parser.parse_args()

    if args.season < 2006:
        parser.error("Historisk data finns från säsongen 2005/06.")

    import_season(args.season, args.phase, Path(args.output))


if __name__ == "__main__":
    main()
