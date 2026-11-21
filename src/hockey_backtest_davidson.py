
"""Davidson 1/X/2 model with league-specific Elo strengths.

Predictions are for regulation time (60 minutes). Parameters are fitted only
from matches completed before the current season; matches on the same date are
all predicted before any result from that date updates team ratings.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

REQUIRED = {
    "date", "league", "home_team", "away_team",
    "reg_home_goals", "reg_away_goals",
}

ELO_SCALE = math.log(10.0) / 400.0
DEFAULT_HOME_ADVANTAGE = 55.0
DEFAULT_LOG_NU = 0.0
MIN_FIT_MATCHES = 300
PARAMETER_SHRINKAGE = 500.0
K_FACTOR = 20.0


def season_key(date: pd.Timestamp) -> int:
    return date.year if date.month >= 8 else date.year - 1


def davidson_probabilities(
    elo_difference: float | np.ndarray,
    home_advantage: float,
    log_nu: float,
) -> np.ndarray:
    """Davidson probabilities: home win, draw, away win."""
    difference = np.asarray(elo_difference, dtype=float)
    eta = ELO_SCALE * (difference + home_advantage)
    log_draw_term = log_nu + 0.5 * eta

    log_denominator = np.logaddexp(
        np.logaddexp(eta, log_draw_term), 0.0
    )
    p1 = np.exp(eta - log_denominator)
    px = np.exp(log_draw_term - log_denominator)
    p2 = np.exp(-log_denominator)

    return np.stack((p1, px, p2), axis=-1)


def _mean_negative_log_likelihood(
    differences: np.ndarray,
    actual: np.ndarray,
    home_advantage: float,
    log_nu: float,
) -> float:
    probabilities = davidson_probabilities(
        differences, home_advantage, log_nu
    )
    selected = probabilities[np.arange(len(actual)), actual]
    return float(-np.log(np.clip(selected, 1e-15, 1.0)).mean())


def fit_davidson_parameters(
    history: list[tuple[float, int]]
) -> tuple[float, float]:
    """Fit home advantage and draw propensity using prior matches only."""
    if len(history) < MIN_FIT_MATCHES:
        return DEFAULT_HOME_ADVANTAGE, DEFAULT_LOG_NU

    differences = np.asarray(
        [item[0] for item in history], dtype=float
    )
    actual = np.asarray(
        [item[1] for item in history], dtype=int
    )

    home_advantage = DEFAULT_HOME_ADVANTAGE
    log_nu = DEFAULT_LOG_NU

    for _ in range(4):
        best_loss = float("inf")
        best_home_advantage = home_advantage

        for candidate in np.linspace(-40.0, 160.0, 101):
            loss = _mean_negative_log_likelihood(
                differences, actual, float(candidate), log_nu
            )
            if loss < best_loss:
                best_loss = loss
                best_home_advantage = float(candidate)

        home_advantage = best_home_advantage

        best_loss = float("inf")
        best_log_nu = log_nu

        for candidate in np.linspace(-1.5, 1.5, 121):
            loss = _mean_negative_log_likelihood(
                differences, actual, home_advantage, float(candidate)
            )
            if loss < best_loss:
                best_loss = loss
                best_log_nu = float(candidate)

        log_nu = best_log_nu

    weight = len(history) / (
        len(history) + PARAMETER_SHRINKAGE
    )

    home_advantage = (
        DEFAULT_HOME_ADVANTAGE
        + weight * (home_advantage - DEFAULT_HOME_ADVANTAGE)
    )
    log_nu = DEFAULT_LOG_NU + weight * (
        log_nu - DEFAULT_LOG_NU
    )

    return float(home_advantage), float(log_nu)


class DavidsonEloModel:
    def __init__(self) -> None:
        self.ratings: dict[tuple[str, str], float] = {}
        self.parameters: dict[str, tuple[float, float]] = {}
        self.seasons: dict[str, int] = {}
        self.history: dict[str, list[tuple[float, int]]] = {}

    def rating(self, league: str, team: str) -> float:
        return self.ratings.get((league, team), 1500.0)

    def prepare_season(self, league: str, season: int) -> None:
        if self.seasons.get(league) == season:
            return

        self.seasons[league] = season
        self.parameters[league] = fit_davidson_parameters(
            self.history.get(league, [])
        )

    def predict(self, league: str, home: str, away: str):
        home_advantage, log_nu = self.parameters.get(
            league,
            (DEFAULT_HOME_ADVANTAGE, DEFAULT_LOG_NU),
        )

        difference = (
            self.rating(league, home)
            - self.rating(league, away)
        )

        probabilities = davidson_probabilities(
            difference, home_advantage, log_nu
        )

        return difference, probabilities

    def update(
        self,
        league: str,
        home: str,
        away: str,
        difference: float,
        actual_index: int,
    ) -> None:
        home_advantage, log_nu = self.parameters.get(
            league,
            (DEFAULT_HOME_ADVANTAGE, DEFAULT_LOG_NU),
        )

        probabilities = davidson_probabilities(
            difference, home_advantage, log_nu
        )

        expected_home_score = float(
            probabilities[0] + 0.5 * probabilities[1]
        )

        actual_home_score = (
            1.0 if actual_index == 0
            else 0.5 if actual_index == 1
            else 0.0
        )

        change = K_FACTOR * (
            actual_home_score - expected_home_score
        )

        self.ratings[(league, home)] = (
            self.rating(league, home) + change
        )
        self.ratings[(league, away)] = (
            self.rating(league, away) - change
        )

        self.history.setdefault(league, []).append(
            (float(difference), int(actual_index))
        )


def backtest(matches: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(matches.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    data = matches.copy()
    data["date"] = pd.to_datetime(
        data["date"], errors="raise", format="mixed"
    )

    for column in ("reg_home_goals", "reg_away_goals"):
        data[column] = pd.to_numeric(
            data[column], errors="raise"
        )

        if data[column].isna().any() or (
            data[column] < 0
        ).any():
            raise ValueError(
                f"Invalid values in {column}"
            )

        if (data[column] % 1 != 0).any():
            raise ValueError(
                f"Non-integer goals in {column}"
            )

    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing required match values")

    if (data["home_team"] == data["away_team"]).any():
        raise ValueError(
            "A team cannot play against itself"
        )

    data["match_day"] = data["date"].dt.normalize()
    data = data.sort_values(
        ["date", "league", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    model = DavidsonEloModel()
    output = []

    for match_day, day in data.groupby(
        "match_day", sort=True
    ):
        day_records = []

        for row in day.itertuples(index=False):
            league = str(row.league)
            model.prepare_season(
                league, season_key(match_day)
            )

            home = str(row.home_team)
            away = str(row.away_team)

            difference, probabilities = model.predict(
                league, home, away
            )

            hg = int(row.reg_home_goals)
            ag = int(row.reg_away_goals)

            actual_index = (
                0 if hg > ag
                else 1 if hg == ag
                else 2
            )

            one_hot = np.eye(3)[actual_index]
            predicted_index = int(
                np.argmax(probabilities)
            )

            home_advantage, log_nu = (
                model.parameters[league]
            )

            day_records.append({
                "league": league,
                "home_team": home,
                "away_team": away,
                "date": row.date,
                "elo_difference": float(difference),
                "home_advantage": float(home_advantage),
                "draw_nu": float(math.exp(log_nu)),
                "p1": float(probabilities[0]),
                "px": float(probabilities[1]),
                "p2": float(probabilities[2]),
                "actual": ("1", "X", "2")[actual_index],
                "predicted": (
                    "1", "X", "2"
                )[predicted_index],
                "correct": int(
                    predicted_index == actual_index
                ),
                "brier": float(
                    np.mean(
                        (probabilities - one_hot) ** 2
                    )
                ),
                "log_loss": float(
                    -math.log(
                        max(
                            1e-15,
                            probabilities[actual_index],
                        )
                    )
                ),
                "_difference": float(difference),
                "_actual_index": int(actual_index),
            })

        # Results from this date are not used until all
        # predictions for the date have been generated.
        for record in day_records:
            model.update(
                record["league"],
                record["home_team"],
                record["away_team"],
                record["_difference"],
                record["_actual_index"],
            )

            record.pop("_difference")
            record.pop("_actual_index")
            output.append(record)

    return pd.DataFrame(output)


def print_metrics(
    results: pd.DataFrame,
    title: str,
) -> None:
    print(f"\n--- DAVIDSON: {title} ---")
    print(f"Matcher: {len(results)}")

    if results.empty:
        return

    print(
        f"Träffsäkerhet: {results.correct.mean():.4%}"
    )
    print(
        f"Brier score: {results.brier.mean():.6f}"
    )
    print(
        f"Log loss: {results.log_loss.mean():.6f}"
    )

    print("\nModellens val:")
    print(
        results["predicted"]
        .value_counts()
        .reindex(["1", "X", "2"], fill_value=0)
        .to_string()
    )

    print("\nGenomsnittliga sannolikheter:")
    print(
        results[["p1", "px", "p2"]]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nGenomsnittliga parametrar:")
    print(
        results[["home_advantage", "draw_nu"]]
        .mean()
        .round(4)
        .to_string()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backtesta Davidson 1/X/2-modellen "
            "efter 60 minuter."
        )
    )

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="davidson_predictions.csv",
    )

    args = parser.parse_args()

    results = backtest(pd.read_csv(args.input))
    results.to_csv(args.output, index=False)

    print_metrics(results, "SAMTLIGA LIGOR")

    for league in sorted(results["league"].unique()):
        print_metrics(
            results[results["league"] == league],
            league,
        )


if __name__ == "__main__":
    main()
