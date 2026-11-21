
"""Validate regulation scores for every SHL overtime game in 2024/25."""

import requests

BASE = "https://www.shl.se/api"
SCHEDULE_URL = f"{BASE}/sports-v2/game-schedule"
PLAY_BY_PLAY_URL = f"{BASE}/gameday/play-by-play"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.shl.se/",
}

PARAMS = {
    "seasonUuid": "qeb-73bZkIm9A",
    "seriesUuid": "qQ9-bb0bzEWUk",
    "gameTypeUuid": "qQ9-af37Ti40B",
}


def event_time_seconds(event):
    value = event.get("time")
    if not isinstance(value, str) or ":" not in value:
        return -1

    try:
        minutes, seconds = value.split(":")
        return int(minutes) * 60 + int(seconds)
    except ValueError:
        return -1


def score_from_event(event):
    home = event.get("homeTeam") or {}
    away = event.get("awayTeam") or {}

    home_score = home.get("score")
    away_score = away.get("score")

    if home_score is None or away_score is None:
        return None

    try:
        return int(home_score), int(away_score)
    except (TypeError, ValueError):
        return None


def main():
    response = requests.get(
        SCHEDULE_URL,
        params=PARAMS,
        headers=HEADERS,
        timeout=30,
    )
    print("Schedule HTTP status:", response.status_code)
    response.raise_for_status()

    games = response.json().get("gameInfo", [])
    overtime_games = [
        game for game in games
        if game.get("overtime") and not game.get("shootout")
    ]

    print("Games in schedule:", len(games))
    print("Overtime games without shootout:", len(overtime_games))

    results = []

    for index, game in enumerate(overtime_games, start=1):
        game_uuid = game.get("uuid")
        home = game.get("homeTeamInfo") or {}
        away = game.get("awayTeamInfo") or {}

        home_name = (home.get("names") or {}).get("long", "Unknown")
        away_name = (away.get("names") or {}).get("long", "Unknown")

        final_score = (int(home["score"]), int(away["score"]))

        print(f"\n[{index}/{len(overtime_games)}] "
              f"{home_name} - {away_name}")
        print("Date:", game.get("startDateTime"))
        print("Final score:", final_score[0], "-", final_score[1])

        try:
            response = requests.get(
                f"{PLAY_BY_PLAY_URL}/{game_uuid}",
                headers=HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            events = response.json()
        except (requests.RequestException, ValueError) as error:
            print("FETCH ERROR:", error)
            results.append({
                "uuid": game_uuid,
                "status": "FETCH_ERROR",
            })
            continue

        period_three = [
            event for event in events
            if event.get("period") == 3
            and event_time_seconds(event) >= 0
            and score_from_event(event) is not None
        ]

        if not period_three:
            print("REGULATION SCORE: NOT FOUND")
            results.append({
                "uuid": game_uuid,
                "status": "NO_PERIOD_THREE_SCORE",
            })
            continue

        latest_event = max(
            period_three,
            key=lambda event: (
                event_time_seconds(event),
                event.get("revision", 0),
            ),
        )

        regulation_score = score_from_event(latest_event)
        differs = regulation_score != final_score

        print("Latest period 3 event:")
        print("  Time:", latest_event.get("time"))
        print("  Type:", latest_event.get("type"))
        print("  Revision:", latest_event.get("revision"))
        print("  Regulation score:",
              regulation_score[0], "-", regulation_score[1])
        print("  Final score:", final_score[0], "-", final_score[1])
        print("  Differs:", differs)

        results.append({
            "uuid": game_uuid,
            "date": game.get("startDateTime"),
            "home": home_name,
            "away": away_name,
            "status": "OK",
            "regulation_score": regulation_score,
            "final_score": final_score,
            "differs": differs,
            "time": latest_event.get("time"),
            "revision": latest_event.get("revision"),
        })

    successful = [item for item in results if item["status"] == "OK"]
    missing = [item for item in results if item["status"] != "OK"]
    differing = [item for item in successful if item["differs"]]
    identical = [item for item in successful if not item["differs"]]

    print("\n========== FINAL SUMMARY ==========")
    print("Overtime games expected:", len(overtime_games))
    print("Games inspected:", len(results))
    print("Successful score reads:", len(successful))
    print("Missing/errors:", len(missing))
    print("Regulation differs from final:", len(differing))
    print("Regulation equals final:", len(identical))

    if missing:
        print("\nPROBLEMS:")
        for item in missing:
            print(item)

    if identical:
        print("\nREVIEW: Regulation score equals final score:")
        for item in identical:
            print(item)

    print("\nValidation finished.")


if __name__ == "__main__":
    main()
