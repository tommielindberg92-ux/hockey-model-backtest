
"""Inspect SHL period data for a specific historical game."""

import json
import requests
from collections import defaultdict

BASE_URL = "https://www.shl.se/api"
SEASON = 2025
GAME_ID = 20460  # Örebro - Luleå, 2025-03-11

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SHLPeriodInspector/1.0)",
    "Accept": "application/json, text/plain, */*",
}

session = requests.Session()
session.headers.update(HEADERS)


def get_json(url, params=None):
    response = session.get(url, params=params, timeout=45)
    response.raise_for_status()

    if not response.text.strip():
        raise ValueError(f"Empty response from {response.url}")

    return response.json()


def walk_dicts(value):
    """Recursively yield every dictionary in a JSON structure."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


print(f"Looking up SHL game ID {GAME_ID}, season {SEASON}")

# Step 1: Find the platform UUID from the season schedule.
schedule_url = f"{BASE_URL}/sports-v2/game-schedule"

schedule = get_json(
    schedule_url,
    params={"season": SEASON, "gameType": "regular"},
)

game_uuid = None

for item in walk_dicts(schedule):
    external_id = (
        item.get("extId")
        or item.get("gameId")
        or item.get("externalId")
    )

    if str(external_id) == str(GAME_ID):
        game_uuid = (
            item.get("uuid")
            or item.get("gameUuid")
            or item.get("game_uuid")
        )
        if game_uuid:
            break

if not game_uuid:
    print("Could not find the game UUID in the schedule.")
    print("Schedule top-level type:", type(schedule).__name__)

    if isinstance(schedule, dict):
        print("Schedule keys:", list(schedule.keys()))

    print("Schedule sample:")
    print(json.dumps(schedule, ensure_ascii=False)[:10000])
    raise SystemExit(1)

print(f"Game UUID: {game_uuid}")

# Step 2: Retrieve the game overview.
overview_url = f"{BASE_URL}/gameday/game-overview/{game_uuid}"
overview = get_json(overview_url)

print("\n--- GAME OVERVIEW ---")
print(json.dumps(overview, ensure_ascii=False, indent=2)[:12000])

# Step 3: Retrieve play-by-play events.
pbp_url = f"{BASE_URL}/gameday/play-by-play/{game_uuid}"
pbp = get_json(pbp_url)

# Locate goal events in the response.
goals = []

for item in walk_dicts(pbp):
    event_type = str(item.get("type", "")).lower()

    if event_type == "goal":
        period = item.get("period")
        team_id = item.get("team_id") or item.get("teamId")

        if period is not None and team_id:
            goals.append({
                "period": period,
                "time": item.get("time"),
                "team": team_id,
                "scorer": (
                    item.get("full_name")
                    or item.get("player_name")
                    or ""
                ),
                "event_id": item.get("event_id"),
            })

# Remove duplicate events if the API nests the same event more than once.
unique_goals = []
seen = set()

for goal in goals:
    key = (
        goal["period"],
        goal["time"],
        goal["team"],
        goal["event_id"],
    )
    if key not in seen:
        seen.add(key)
        unique_goals.append(goal)

goals = sorted(
    unique_goals,
    key=lambda item: (
        int(item["period"]),
        str(item["time"] or ""),
    ),
)

print("\n--- GOALS BY PERIOD ---")

period_scores = defaultdict(lambda: defaultdict(int))

for goal in goals:
    period = int(goal["period"])
    team = goal["team"]

    if 1 <= period <= 3:
        period_scores[period][team] += 1

    print(
        f"Period {period} | "
        f"{goal['time']} | "
        f"Team: {team} | "
        f"Scorer: {goal['scorer']}"
    )

print("\n--- REGULATION SCORE ---")

teams = sorted({
    goal["team"]
    for goal in goals
    if 1 <= int(goal["period"]) <= 3
})

for team in teams:
    total = sum(
        period_scores[period][team]
        for period in (1, 2, 3)
    )
    print(f"{team}: {total}")

print("\n--- PERIOD TOTALS ---")

for period in (1, 2, 3, 4, 5):
    scores = dict(period_scores[period])
    if scores:
        print(f"Period {period}: {scores}")

print("\nInspection completed.")
