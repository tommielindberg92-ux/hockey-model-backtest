
"""Inspect SHL play-by-play data from a real overtime-decided game."""

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

if not overtime_games:
    print("No overtime-decided games found.")
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
print("\nPlay-by-play HTTP status:", response.status_code)
response.raise_for_status()

events = response.json()
print("Play-by-play event count:", len(events))

period_three_events = [
    event for event in events
    if event.get("period") == 3
    and event.get("time") == "20:00"
    and event.get("homeTeam", {}).get("score") is not None
    and event.get("awayTeam", {}).get("score") is not None
]

print("\nPeriod 3, time 20:00 snapshots:")

for event in period_three_events:
    print({
        "time": event.get("time"),
        "state": event.get("gameState"),
        "revision": event.get("revision"),
        "home_score": event.get("homeTeam", {}).get("score"),
        "away_score": event.get("awayTeam", {}).get("score"),
        "homeGoals": event.get("homeGoals"),
        "awayGoals": event.get("awayGoals"),
        "type": event.get("type"),
    })

print("\nEvents in period 4:")

for event in events:
    if event.get("period") == 4:
        print({
            "time": event.get("time"),
            "state": event.get("gameState"),
            "revision": event.get("revision"),
            "home_score": event.get("homeTeam", {}).get("score"),
            "away_score": event.get("awayTeam", {}).get("score"),
            "homeGoals": event.get("homeGoals"),
            "awayGoals": event.get("awayGoals"),
            "type": event.get("type"),
        })

print("\nOvertime inspection finished.")
