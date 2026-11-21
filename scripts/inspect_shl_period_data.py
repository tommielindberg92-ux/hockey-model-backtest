"""Inspect SHL play-by-play data from a real overtime game."""

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
    "seasonUuid": "qeb-73qtP8d19",
    "seriesUuid": "qQ9-bb0bzEWUk",
    "gameTypeUuid": "qQ9-af37Ti40B",
}

response = requests.get(
    SCHEDULE_URL,
    params=PARAMS,
    headers=HEADERS,
    timeout=30,
)
response.raise_for_status()

data = response.json()
games = data.get("gameInfo", [])

print("Games in schedule:", len(games))

overtime_games = [
    game for game in games
    if game.get("overtime") or game.get("shootout")
]

print("Overtime/shootout games:", len(overtime_games))

if not overtime_games:
    print("No overtime games found.")
    raise SystemExit(0)

game = overtime_games[0]
game_uuid = game["uuid"]

home = game.get("homeTeamInfo", {})
away = game.get("awayTeamInfo", {})

print("\nSelected game:")
print("UUID:", game_uuid)
print("Date:", game.get("startDateTime"))
print("Home:", home.get("names", {}).get("long"))
print("Away:", away.get("names", {}).get("long"))
print("Final score:", home.get("score"), "-", away.get("score"))
print("Overtime:", game.get("overtime"))
print("Shootout:", game.get("shootout"))

response = requests.get(
    f"{PLAY_BY_PLAY_URL}/{game_uuid}",
    headers=HEADERS,
    timeout=30,
)
response.raise_for_status()

events = response.json()

print("\nPlay-by-play event count:", len(events))

if events:
    print("Event fields:", list(events[0].keys()))

print("\nEvents from period 3 onward:")

for event in events:
    period = event.get("period", 0)

    if period >= 3:
        home_event = event.get("homeTeam", {})
        away_event = event.get("awayTeam", {})

        print({
            "period": period,
            "time": event.get("time"),
            "state": event.get("gameState"),
            "revision": event.get("revision"),
            "home_score": home_event.get("score"),
            "away_score": away_event.get("score"),
            "event_type": event.get("eventType"),
            "description": event.get("description"),
        })

print("\nOvertime inspection finished.")