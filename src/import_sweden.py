
"""Importera HockeyAllsvenskan- och SHL-resultat efter 60 minuter."""

import argparse
import csv
import json
import socket
import time
import urllib.error
from pathlib import Path
from urllib.request import Request, urlopen

HA_URL = "https://hockeyallsvenskan.se/api/all-time-games-stats"

SHL_BASE = "https://www.shl.se/api"
SHL_SCHEDULE_URL = f"{SHL_BASE}/sports-v2/game-schedule"
SHL_PLAY_BY_PLAY_URL = f"{SHL_BASE}/gameday/play-by-play"

SHL_SEASONS = {
    2025: "qeb-73bZkIm9A",
    2026: "xs4m9qupsi",
    2027: "ndcf81nlb3",
}

SHL_SERIES_UUID = "qQ9-bb0bzEWUk"
SHL_GAME_TYPE_UUID = "qQ9-af37Ti40B"

HEADERS = {
    "User-Agent": "Mozilla/5.0 hockey-model-backtest/1.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.shl.se/",
}

COLUMNS = [
    "date", "league", "home_team", "away_team",
    "reg_home_goals", "reg_away_goals", "decided_in", "source",
]


def request_json(url, data=None, headers=None, attempts=5, timeout=90):
    """Hämta JSON med återförsök vid tillfälliga nätverks- och serverfel."""

    request_headers = {
        "User-Agent": "hockey-model-backtest/1.0",
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)

    for attempt in range(1, attempts + 1):
        request = Request(
            url,
            data=data,
            headers=request_headers,
            method="POST" if data is not None else "GET",
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))

        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 425, 429, 500, 502, 503, 504}

            if not retryable or attempt == attempts:
                raise

            retry_after = exc.headers.get("Retry-After")
            try:
                delay = min(float(retry_after), 60) if retry_after else min(2 ** attempt, 30)
            except (TypeError, ValueError):
                delay = min(2 ** attempt, 30)

            print(
                f"HTTP {exc.code} vid hämtning. "
                f"Försök {attempt}/{attempts}. Väntar {delay:.0f} sekunder."
            )
            time.sleep(delay)

        except (
            TimeoutError,
            socket.timeout,
            urllib.error.URLError,
            ConnectionError,
        ) as exc:
            if attempt == attempts:
                print(f"API-anrop misslyckades efter {attempts} försök: {url}")
                raise

            delay = min(2 ** attempt, 30)
            print(
                f"Nätverksfel vid hämtning: {exc}. "
                f"Försök {attempt}/{attempts}. Väntar {delay} sekunder."
            )
            time.sleep(delay)

    raise RuntimeError(f"Kunde inte hämta data från {url}")


def merge_rows(output_file, new_rows):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if output_file.exists():
        with output_file.open(encoding="utf-8", newline="") as file:
            existing = list(csv.DictReader(file))

    combined = {
        (r["date"], r["league"], r["home_team"], r["away_team"]): r
        for r in existing
    }

    for row in new_rows:
        key = (row["date"], row["league"], row["home_team"], row["away_team"])
        combined[key] = row

    ordered = sorted(
        combined.values(),
        key=lambda row: (
            row["date"], row["league"], row["home_team"], row["away_team"]
        ),
    )

    with output_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(ordered)

    print(f"Totalt i filen: {len(ordered)}")
    print(f"Fil: {output_file}")


# ---------------- HockeyAllsvenskan ----------------

def get_ha_games(season, phase):
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
        "season": str(season - 1),
    }

    payload = request_json(
        HA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("data", "games", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]

    raise ValueError("HockeyAllsvenskans API-svar hade oväntat format.")


def parse_period_scores(value):
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
            raise ValueError("Negativt antal mål.")

        home_goals += home
        away_goals += away

    return home_goals, away_goals


def import_ha(season, phase):
    games = get_ha_games(season, phase)
    imported = []

    for game in games:
        decided_in = str(game.get("decided_in", "")).upper()

        if decided_in not in {"FT", "OT", "SO"}:
            continue

        try:
            period_scores = game.get(
                "period_scores", game.get("PeriodScores")
            )
            home_goals, away_goals = parse_period_scores(period_scores)

            if decided_in == "FT" and home_goals == away_goals:
                raise ValueError("FT-match oavgjord efter 60 minuter.")

            if decided_in in {"OT", "SO"} and home_goals != away_goals:
                raise ValueError("OT/SO-match ej oavgjord efter 60 minuter.")

            date_value = game.get("date", game.get("Date"))
            home_team = game.get("home_team_code", game.get("HomeTeam"))
            away_team = game.get("away_team_code", game.get("AwayTeam"))

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
                "source": "HockeyAllsvenskan officiellt API",
            })

        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            print(f"HA VARNING: match hoppades över: {exc}")

    print(f"HA importerade matcher: {len(imported)}")
    return imported


# ---------------- SHL ----------------

def event_time_seconds(event):
    value = event.get("time")
    if not isinstance(value, str) or ":" not in value:
        return -1

    try:
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return -1


def event_revision(event):
    try:
        return int(event.get("revision") or 0)
    except (TypeError, ValueError):
        return 0


def score_from_event(event):
    home = event.get("homeTeam") or {}
    away = event.get("awayTeam") or {}

    try:
        home_score = int(home["score"])
        away_score = int(away["score"])
    except (KeyError, TypeError, ValueError):
        return None

    if home_score < 0 or away_score < 0:
        return None

    return home_score, away_score


def period_number(event):
    try:
        return int(event.get("period"))
    except (TypeError, ValueError):
        return -1


def regulation_score_from_play_by_play(game_uuid):
    events = request_json(
        f"{SHL_PLAY_BY_PLAY_URL}/{game_uuid}",
        headers=HEADERS,
    )

    if not isinstance(events, list):
        raise ValueError("SHL play-by-play hade oväntat format.")

    period_three = [
        event for event in events
        if period_number(event) == 3
        and event_time_seconds(event) >= 0
        and score_from_event(event) is not None
    ]

    if not period_three:
        raise ValueError("Saknar giltig ställning efter period 3.")

    latest_event = max(
        period_three,
        key=lambda event: (
            event_time_seconds(event),
            event_revision(event),
        ),
    )

    score = score_from_event(latest_event)

    if score is None:
        raise ValueError("Kunde inte läsa period 3-resultatet.")

    return score


def team_name(team_info):
    names = team_info.get("names") or {}
    return names.get("long") or names.get("short")


def import_shl_season(end_year, season_uuid):
    params = (
        f"?seasonUuid={season_uuid}"
        f"&seriesUuid={SHL_SERIES_UUID}"
        f"&gameTypeUuid={SHL_GAME_TYPE_UUID}"
    )

    payload = request_json(
        SHL_SCHEDULE_URL + params,
        headers=HEADERS,
    )

    games = payload.get("gameInfo", [])
    imported = []
    skipped = 0
    errors = 0

    for game in games:
        home = game.get("homeTeamInfo") or {}
        away = game.get("awayTeamInfo") or {}

        home_team = team_name(home)
        away_team = team_name(away)
        game_uuid = game.get("uuid")
        date_value = game.get("startDateTime") or game.get("rawStartDateTime")

        if not all((home_team, away_team, game_uuid, date_value)):
            skipped += 1
            continue

        try:
            final_home = int(home["score"])
            final_away = int(away["score"])
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue

        if final_home < 0 or final_away < 0:
            skipped += 1
            continue

        overtime = bool(game.get("overtime"))
        shootout = bool(game.get("shootout"))

        if shootout:
            decided_in = "SO"
        elif overtime:
            decided_in = "OT"
        else:
            decided_in = "FT"

        try:
            if overtime or shootout:
                reg_home, reg_away = regulation_score_from_play_by_play(
                    game_uuid
                )

                if reg_home != reg_away:
                    raise ValueError(
                        "OT/SO-matchen var inte oavgjord efter period 3."
                    )
            else:
                reg_home, reg_away = final_home, final_away

                if reg_home == reg_away:
                    raise ValueError(
                        "FT-matchen var oavgjord efter 60 minuter."
                    )

            imported.append({
                "date": str(date_value),
                "league": "SHL",
                "home_team": str(home_team),
                "away_team": str(away_team),
                "reg_home_goals": reg_home,
                "reg_away_goals": reg_away,
                "decided_in": decided_in,
                "source": (
                    f"SHL officiellt API; game UUID: {game_uuid}"
                ),
            })

        except (KeyError, TypeError, ValueError) as exc:
            errors += 1
            print(
                f"SHL VARNING {home_team} - {away_team} "
                f"({game_uuid}): {exc}"
            )

    print(f"\nSHL säsong {end_year}:")
    print(f"Matcher i spelschemat: {len(games)}")
    print(f"Importerade matcher: {len(imported)}")
    print(f"Ej färdigspelade/utan resultat: {skipped}")
    print(f"Fel eller overifierade matcher: {errors}")

    if errors:
        raise RuntimeError(
            f"SHL-importen för {end_year} innehöll {errors} fel. "
            "Kontrollera varningarna innan datan används."
        )

    return imported


def import_shl():
    all_rows = []

    for end_year, season_uuid in SHL_SEASONS.items():
        print(f"\nHämtar SHL-säsongen {end_year - 1}/{end_year}...")
        all_rows.extend(import_shl_season(end_year, season_uuid))

    print(f"\nSHL totalt importerade matcher: {len(all_rows)}")
    return all_rows


# ---------------- Kommando ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Importera svenska hockeyresultat efter 60 minuter."
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="HockeyAllsvenskans slutår, exempelvis 2026",
    )
    parser.add_argument(
        "--phase",
        default="HA",
        choices=["HA", "HAPlayoff", "HAkval"],
    )
    parser.add_argument(
        "--output",
        default="data/ha_matches.csv",
    )
    parser.add_argument(
        "--include-shl",
        action="store_true",
        help="Importera även SHL-säsongerna 2024/25–2026/27.",
    )

    args = parser.parse_args()

    if args.season < 2006:
        parser.error("Historisk HockeyAllsvenskan-data finns från 2005/06.")

    output_file = Path(args.output)

    rows = import_ha(args.season, args.phase)

    if args.include_shl:
        rows.extend(import_shl())

    merge_rows(output_file, rows)


if __name__ == "__main__":
    main()
