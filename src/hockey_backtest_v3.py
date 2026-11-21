
"""Version 3 – tidsviktad hockeymodell med separata hemma- och bortabaslinjer."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "date",
    "league",
    "home_team",
    "away_team",
    "reg_home_goals",
    "reg_away_goals",
}


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def regulation_probabilities(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = 12,
) -> Tuple[float, float, float]:
    if home_lambda <= 0 or away_lambda <= 0:
        raise ValueError("Expected goal rates must be positive")

    home = [poisson_pmf(i, home_lambda) for i in range(max_goals + 1)]
    away = [poisson_pmf(i, away_lambda) for i in range(max_goals + 1)]

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
    elo: Dict[str, float] = field(default_factory=dict)
    attack: Dict[str, float] = field(default_factory=dict)
    defence: Dict[str, float] = field(default_factory=dict)

    home_goal_sum: float = 0.0
    away_goal_sum: float = 0.0

    # Separata effektiva matchvikter för hemma- och bortamål.
    home_games: float = 0.0
    away_games: float = 0.0

    last_date: pd.Timestamp | None = None

    def rating(self, team: str) -> float:
        return self.elo.get(team, 1500.0)

    def attack_rating(self, team: str) -> float:
        return self.attack.get(team, 0.0)

    def defence_rating(self, team: str) -> float:
        return self.defence.get(team, 0.0)

    def goal_rates(self) -> Tuple[float, float]:
        if self.home_games < 5 or self.away_games < 5:
            return 2.8, 2.8

        return (
            max(1.5, self.home_goal_sum / self.home_games),
            max(1.5, self.away_goal_sum / self.away_games),
        )

    def decay(
        self,
        current_date: pd.Timestamp,
        half_life_days: float = 365.0,
    ) -> None:
        if self.last_date is None:
            self.last_date = current_date
            return

        days = max(0.0, (current_date - self.last_date).days)
        factor = 0.5 ** (days / half_life_days)

        # Lagstyrka och målhistorik blir gradvis mindre relevanta.
        for ratings in (self.attack, self.defence):
            for team in ratings:
                ratings[team] *= factor

        # Elo dras gradvis mot neutralt värde.
        for team in self.elo:
            self.elo[team] = (
                1500.0 + (self.elo[team] - 1500.0) * factor
            )

        self.home_goal_sum *= factor
        self.away_goal_sum *= factor
        self.home_games *= factor
        self.away_games *= factor

        self.last_date = current_date


class TimeWeightedModel:
    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 55.0,
        rating_learning_rate: float = 0.045,
        half_life_days: float = 365.0,
    ):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.rating_learning_rate = rating_learning_rate
        self.half_life_days = half_life_days
        self.states: Dict[str, LeagueState] = {}

    def state(self, league: str) -> LeagueState:
        return self.states.setdefault(league, LeagueState())

    def prepare_date(self, league: str, date: pd.Timestamp) -> None:
        self.state(league).decay(date, self.half_life_days)

    def expected_goals(
        self,
        league: str,
        home_team: str,
        away_team: str,
    ) -> Tuple[float, float]:
        state = self.state(league)
        base_home, base_away = state.goal_rates()

        elo_difference = (
            state.rating(home_team)
            - state.rating(away_team)
            + self.home_advantage
        )

        elo_effect = max(
            -0.25,
            min(0.25, elo_difference / 400.0 * 0.20),
        )

        home_strength = (
            state.attack_rating(home_team)
            + state.defence_rating(away_team)
        )
        away_strength = (
            state.attack_rating(away_team)
            + state.defence_rating(home_team)
        )

        home_lambda = base_home * math.exp(
            max(-0.65, min(0.65, home_strength + elo_effect))
        )
        away_lambda = base_away * math.exp(
            max(-0.65, min(0.65, away_strength - elo_effect))
        )

        return (
            max(0.5, min(6.0, home_lambda)),
            max(0.5, min(6.0, away_lambda)),
        )

    def predict(
        self,
        league: str,
        home_team: str,
        away_team: str,
    ) -> Tuple[float, float, float]:
        rates = self.expected_goals(league, home_team, away_team)
        return regulation_probabilities(*rates)

    def update(
        self,
        league: str,
        home_team: str,
        away_team: str,
        hg: int,
        ag: int,
    ) -> None:
        state = self.state(league)

        # Förväntade mål beräknas med information före matchen.
        home_lambda, away_lambda = self.expected_goals(
            league, home_team, away_team
        )

        home_rating = state.rating(home_team)
        away_rating = state.rating(away_team)

        expected_home = 1.0 / (
            1.0 + 10.0 ** (
                (away_rating - (home_rating + self.home_advantage)) / 400.0
            )
        )

        actual_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        margin = abs(hg - ag)
        multiplier = math.log1p(margin) if margin else 1.0

        change = self.k_factor * multiplier * (
            actual_home - expected_home
        )

        state.elo[home_team] = home_rating + change
        state.elo[away_team] = away_rating - change

        lr = self.rating_learning_rate

        home_residual = max(
            -0.8,
            min(0.8, math.log((hg + 0.5) / (home_lambda + 0.5))),
        )
        away_residual = max(
            -0.8,
            min(0.8, math.log((ag + 0.5) / (away_lambda + 0.5))),
        )

        # Positiv försvarsrating betyder att laget släpper in fler mål.
        state.attack[home_team] = max(
            -0.8,
            min(0.8, state.attack_rating(home_team) + lr * home_residual),
        )
        state.defence[away_team] = max(
            -0.8,
            min(0.8, state.defence_rating(away_team) + lr * home_residual),
        )

        state.attack[away_team] = max(
            -0.8,
            min(0.8, state.attack_rating(away_team) + lr * away_residual),
        )
        state.defence[home_team] = max(
            -0.8,
            min(0.8, state.defence_rating(home_team) + lr * away_residual),
        )

        state.home_goal_sum += hg
        state.away_goal_sum += ag
        state.home_games += 1.0
        state.away_games += 1.0


def backtest(matches: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(matches.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {', '.join(sorted(missing))}"
        )

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

    for column in ("reg_home_goals", "reg_away_goals"):
        if (data[column] < 0).any() or (data[column] % 1 != 0).any():
            raise ValueError(f"Invalid goals in {column}")

    if (data["home_team"] == data["away_team"]).any():
        raise ValueError("A team cannot play against itself")

    data = data.sort_values("date", kind="stable").reset_index(drop=True)

    model = TimeWeightedModel()
    rows = []

    # Alla matcher samma datum prognostiseras före dagens uppdateringar.
    for match_date, day in data.groupby("date", sort=True):
        day_updates = []

        for row in day.itertuples(index=False):
            league = str(row.league)
            home_team = str(row.home_team)
            away_team = str(row.away_team)

            model.prepare_date(league, match_date)

            probabilities = model.predict(
                league, home_team, away_team
            )

            hg = int(row.reg_home_goals)
            ag = int(row.reg_away_goals)

            actual_index = 0 if hg > ag else 1 if hg == ag else 2
            actual = np.zeros(3)
            actual[actual_index] = 1.0
            p = np.asarray(probabilities)

            rows.append({
                "date": match_date,
                "league": league,
                "home_team": home_team,
                "away_team": away_team,
                "p1": p[0],
                "px": p[1],
                "p2": p[2],
                "actual": ("1", "X", "2")[actual_index],
                "brier": float(np.mean((p - actual) ** 2)),
                "log_loss": float(
                    -math.log(max(1e-15, p[actual_index]))
                ),
                "correct": int(np.argmax(p) == actual_index),
            })

            day_updates.append((
                league, home_team, away_team, hg, ag
            ))

        for update in day_updates:
            model.update(*update)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtesta hockeyprognoser efter 60 minuter."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="v3_predictions.csv")
    args = parser.parse_args()

    results = backtest(pd.read_csv(args.input))
    results.to_csv(args.output, index=False)

    print(f"Matcher: {len(results)}")

    if len(results):
        print(f"Träffsäkerhet: {results.correct.mean():.4f}")
        print(f"Brier score: {results.brier.mean():.4f}")
        print(f"Log loss: {results.log_loss.mean():.4f}")

        print("\nModellens val:")
        print(
            results[["p1", "px", "p2"]]
            .idxmax(axis=1)
            .value_counts()
            .reindex(["p1", "px", "p2"], fill_value=0)
            .to_string()
        )

        print("\nGenomsnittliga sannolikheter:")
        print(results[["p1", "px", "p2"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
