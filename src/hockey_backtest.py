
"""Kronologiskt backtest för hockey 1X2 efter ordinarie tid."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date", "league", "home_team", "away_team",
    "reg_home_goals", "reg_away_goals",
}


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def regulation_probabilities(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 12,
) -> Tuple[float, float, float]:
    if home_lambda <= 0 or away_lambda <= 0:
        raise ValueError("Expected goal rates must be positive")

    home = [_poisson_pmf(i, home_lambda) for i in range(max_goals + 1)]
    away = [_poisson_pmf(i, away_lambda) for i in range(max_goals + 1)]

    p1 = px = p2 = 0.0
    for h, ph in enumerate(home):
        for a, pa in enumerate(away):
            probability = ph * pa
            if h > a:
                p1 += probability
            elif h == a:
                px += probability
            else:
                p2 += probability

    total = p1 + px + p2
    return p1 / total, px / total, p2 / total


@dataclass
class LeagueState:
    ratings: Dict[str, float] = field(default_factory=dict)
    home_goal_sum: float = 0.0
    away_goal_sum: float = 0.0
    games: int = 0

    def rating(self, team: str) -> float:
        return self.ratings.get(team, 1500.0)

    def goal_rates(self) -> Tuple[float, float]:
        if self.games < 10:
            return 2.8, 2.8

        return (
            max(0.5, self.home_goal_sum / self.games),
            max(0.5, self.away_goal_sum / self.games),
        )


class EloPoissonModel:
    def __init__(self, k_factor: float = 20.0, home_advantage: float = 55.0):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.states: Dict[str, LeagueState] = {}

    def _state(self, league: str) -> LeagueState:
        return self.states.setdefault(league, LeagueState())

    def predict(
        self, league: str, home_team: str, away_team: str
    ) -> Tuple[float, float, float]:
        state = self._state(league)

        home_elo = state.rating(home_team) + self.home_advantage
        away_elo = state.rating(away_team)

        elo_home_share = 1.0 / (
            1.0 + 10.0 ** ((away_elo - home_elo) / 400.0)
        )

        base_home, base_away = state.goal_rates()
        strength = (elo_home_share - 0.5) * 0.65

        home_lambda = max(0.5, base_home + strength)
        away_lambda = max(0.5, base_away - strength)

        return regulation_probabilities(home_lambda, away_lambda)

    def update(
        self,
        league: str,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
    ) -> None:
        state = self._state(league)

        home_rating = state.rating(home_team)
        away_rating = state.rating(away_team)

        expected = 1.0 / (
            1.0 + 10.0 ** (
                (away_rating - (home_rating + self.home_advantage)) / 400.0
            )
        )

        if home_goals > away_goals:
            actual = 1.0
        elif home_goals == away_goals:
            actual = 0.5
        else:
            actual = 0.0

        margin = abs(home_goals - away_goals)
        multiplier = math.log1p(margin) if margin else 1.0
        change = self.k_factor * multiplier * (actual - expected)

        state.ratings[home_team] = home_rating + change
        state.ratings[away_team] = away_rating - change

        state.home_goal_sum += home_goals
        state.away_goal_sum += away_goals
        state.games += 1


def backtest(matches: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(matches.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    data = matches.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["reg_home_goals"] = pd.to_numeric(
        data["reg_home_goals"], errors="raise"
    )
    data["reg_away_goals"] = pd.to_numeric(
        data["reg_away_goals"], errors="raise"
    )

    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Missing required match values")

    for column in ["reg_home_goals", "reg_away_goals"]:
        if (data[column] < 0).any() or (data[column] % 1 != 0).any():
            raise ValueError(f"Invalid goals in {column}")

    if (data["home_team"] == data["away_team"]).any():
        raise ValueError("A team cannot play against itself")

    data = data.sort_values("date", kind="stable").reset_index(drop=True)

    model = EloPoissonModel()
    rows = []

    # Process one complete date at a time.
    for match_date, day in data.groupby("date", sort=True):
        day_updates = []

        # First predict every match using only earlier dates.
        for row in day.itertuples(index=False):
            probabilities = model.predict(
                row.league, row.home_team, row.away_team
            )

            hg = int(row.reg_home_goals)
            ag = int(row.reg_away_goals)

            actual_index = 0 if hg > ag else (1 if hg == ag else 2)
            actual_outcome = np.zeros(3)
            actual_outcome[actual_index] = 1.0

            p = np.asarray(probabilities)
            rows.append({
                "date": match_date,
                "league": row.league,
                "home_team": row.home_team,
                "away_team": row.away_team,
                "p1": probabilities[0],
                "px": probabilities[1],
                "p2": probabilities[2],
                "actual": ("1", "X", "2")[actual_index],
                "brier": float(np.mean((p - actual_outcome) ** 2)),
                "log_loss": float(
                    -math.log(max(1e-15, p[actual_index]))
                ),
                "correct": int(np.argmax(p) == actual_index),
            })

            day_updates.append((
                row.league, row.home_team, row.away_team, hg, ag
            ))

        # Only after all predictions for the date are made, update.
        for update in day_updates:
            model.update(*update)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtesta hockeyprognoser efter 60 minuter."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="backtest_predictions.csv")
    args = parser.parse_args()

    results = backtest(pd.read_csv(args.input))
    results.to_csv(args.output, index=False)

    print(f"Matcher analyserade: {len(results)}")
    if len(results):
        print(f"Träffsäkerhet: {results.correct.mean():.3f}")
        print(f"Genomsnittlig Brier score: {results.brier.mean():.4f}")
        print(f"Genomsnittlig log loss: {results.log_loss.mean():.4f}")


if __name__ == "__main__":
    main()
