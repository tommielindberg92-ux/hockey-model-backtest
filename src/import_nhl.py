"""Importera NHL-resultat efter ordinarie tid (60 minuter).

Hämtar avslutade matcher från NHL:s publika API.
Mål från period 4 och framåt räknas aldrig med.
"""

import argparse
import csv
import json
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://api-web.nhle.com/v1"
SOURCE = "NHL public API"
OUTPUT_COLUMNS = [
    "date", "league", "home_team", "away_team",
    "reg_home_goals", "reg_away_goals", "decided_in", "source",
]


def get_json(url):
    request = Request(url, headers={"User-Agent": "hockey-model-backtest/1.0"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Kunde inte hämta {url}: {exc}") from exc


def team_name(team):
    name = team.get("name") or team.get("commonName") or {}
    if isinstance(name, dict):
        return name.get("default", "")
    return str(name)

def get_regulation_result(game_id, home_id, away_id, decided_in):
    """Return goals after 60 minutes and the outcome type from the scoreboard."""
    if decided_in not in {"REG", "OT", "SO"}:
        raise ValueError(
            f"Okänd eller saknad sluttyp för match {game_id}: {decided_in}"
        )

    data = get_json(f"{BASE_URL}/gamecenter/{game_id}/play-by-play")

    home_goals = 0
    away_goals = 0

    for play in data.get("plays", []):
        period = play.get("periodDescriptor", {}).get("number")
        event_type = play.get("typeDescKey")

        if event_type != "goal" or not isinstance(period, int) or period > 3:
            continue

        scoring_team = play.get("details", {}).get("eventOwnerTeamId")
        if scoring_team == home_id:
            home_goals += 1
        elif scoring_team == away_id:
            away_goals += 1

    return home_goals, away_goals, decided_in



def import_nhl(start_date, end_date, output_file):
    results = []

    current = start_date
    while current <= end_date:
        url = f"{BASE_URL}/score/{current.isoformat()}"
        scoreboard = get_json(url)

        for game in scoreboard.get("games", []):
            if game.get("gameState") not in {"OFF", "FINAL"}:
                continue

            home = game.get("homeTeam", {})
            away = game.get("awayTeam", {})
            home_id = home.get("id")
            away_id = away.get("id")
            game_id = game.get("id")

            if not all((home_id, away_id, game_id)):
                print(f"Hoppar över match med ofullständig information: {game_id}")
                continue

                try:
                game_outcome = game.get("gameOutcome") or {}
                decided_in = game_outcome.get("lastPeriodType")

                home_goals, away_goals, decided_in = get_regulation_result(
                    game_id, home_id, away_id, decided_in
                )
            except (RuntimeError, ValueError, KeyError) as exc:
                print(f"VARNING: match {game_id} hoppades över: {exc}")
                continue

            results.append({
                "date": game.get("gameDate", current.isoformat()),
                "league": "NHL",
                "home_team": team_name(home),
                "away_team": team_name(away),
                "reg_home_goals": home_goals,
                "reg_away_goals": away_goals,
                "decided_in": decided_in,
                "source": SOURCE,
            })

            time.sleep(0.15)

        print(f"Kontrollerat datum: {current.isoformat()}")
        current += timedelta(days=1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if output_file.exists():
        with output_file.open("r", newline="", encoding="utf-8") as file:
            existing = list(csv.DictReader(file))

    combined = {
        (row["date"], row["league"], row["home_team"], row["away_team"]): row
        for row in existing
    }

    for row in results:
        key = (row["date"], row["league"], row["home_team"], row["away_team"])
        combined[key] = row

    ordered = sorted(combined.values(), key=lambda row: row["date"])

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)

    print(f"\nKlart! {len(results)} nya/importerade NHL-matcher.")
    print(f"Totalt i filen: {len(ordered)} matcher.")
    print(f"Fil: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Importera NHL-resultat efter 60 minuter."
    )
    parser.add_argument("--from-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--to-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output", default="data/nhl_matches.csv")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.from_date)
    end_date = date.fromisoformat(args.to_date)

    if end_date < start_date:
        parser.error("--to-date måste vara samma dag eller senare än --from-date")

    import_nhl(start_date, end_date, Path(args.output))


if __name__ == "__main__":
    main()
