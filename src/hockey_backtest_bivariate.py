
"""Bivariate Poisson experiment using time-safe V7 goal predictions.

All predictions are for regulation time (60 minutes).
The shared-goal parameter is fitted separately by league, using only
previous matches. Draw calibration is also fitted using prior predictions.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from src.hockey_backtest_v7 import (
    backtest as run_v7,
    apply_draw_calibration,
    fit_draw_multiplier,
    season_key,
)

MIN_FIT_MATCHES = 300
SHRINKAGE = 500.0
MAX_ALPHA = 0.80
ALPHA_STEPS = 81
MAX_GOALS = 12


def poisson_pmf(mean: float, maximum: int = MAX_GOALS) -> np.ndarray:
    """Poisson probability mass from zero through maximum."""
    mean = max(float(mean), 1e-12)
    values = np.arange(maximum + 1)

    log_probs = (
        -mean
        + values * math.log(mean)
        - np.asarray([math.lgamma(int(x) + 1) for x in values])
    )

    return np.exp(log_probs)


def bivariate_poisson_matrix(
    home_mean: float,
    away_mean: float,
    alpha: float,
    maximum: int = MAX_GOALS,
) -> np.ndarray:
    """Joint score matrix for a bivariate Poisson distribution.

    X = U + W
    Y = V + W

    U, V and W are independent Poisson variables.
    W is the shared component.
    """
    home_mean = max(float(home_mean), 1e-9)
    away_mean = max(float(away_mean), 1e-9)

    shared_mean = min(
        max(0.0, float(alpha)) * min(home_mean, away_mean),
        min(home_mean, away_mean) * 0.95,
    )

    home_independent = max(home_mean - shared_mean, 1e-9)
    away_independent = max(away_mean - shared_mean, 1e-9)

    home_pmf = poisson_pmf(home_independent, maximum)
    away_pmf = poisson_pmf(away_independent, maximum)
    shared_pmf = poisson_pmf(shared_mean, maximum)

    matrix = np.zeros((maximum + 1, maximum + 1), dtype=float)

    for shared_goals in range(maximum + 1):
        probability = shared_pmf[shared_goals]

        if probability < 1e-15:
            continue

        size = maximum + 1 - shared_goals

        matrix[
            shared_goals:, shared_goals:
        ] += probability * np.outer(
            home_pmf[:size],
            away_pmf[:size],
        )

    total = matrix.sum()

    if total <= 0 or not np.isfinite(total):
        raise ValueError("Invalid bivariate probability matrix")

    return matrix / total


def bivariate_probabilities(
    home_mean: float,
    away_mean: float,
    alpha: float,
) -> np.ndarray:
    """Return probabilities for home win, draw and away win."""
    matrix = bivariate_poisson_matrix(
        home_mean,
        away_mean,
        alpha,
    )

    p1 = float(np.tril(matrix, k=-1).sum())
    px = float(np.trace(matrix))
    p2 = float(np.triu(matrix, k=1).sum())

    probabilities = np.asarray([p1, px, p2], dtype=float)
    probabilities = np.clip(probabilities, 1e-15, 1.0)

    return probabilities / probabilities.sum()


def observed_score_log_probability(
    home_goals: int,
    away_goals: int,
    home_mean: float,
    away_mean: float,
    alpha: float,
) -> float:
    """Exact log probability of an observed score."""
    shared_mean = (
        max(0.0, alpha)
        * min(home_mean, away_mean)
    )

    shared_mean = min(
        shared_mean,
        min(home_mean, away_mean) * 0.95,
    )

    home_independent = max(home_mean - shared_mean, 1e-12)
    away_independent = max(away_mean - shared_mean, 1e-12)
    shared_mean = max(shared_mean, 1e-12)

    terms = []

    for shared_goals in range(
        min(home_goals, away_goals) + 1
    ):
        home_remaining = home_goals - shared_goals
        away_remaining = away_goals - shared_goals

        log_term = (
            -home_independent
            + home_remaining * math.log(home_independent)
            - math.lgamma(home_remaining + 1)
            -away_independent
            + away_remaining * math.log(away_independent)
            - math.lgamma(away_remaining + 1)
            -shared_mean
            + shared_goals * math.log(shared_mean)
            - math.lgamma(shared_goals + 1)
        )

        terms.append(log_term)

    maximum = max(terms)

    return maximum + math.log(
        sum(math.exp(value - maximum) for value in terms)
    )


def fit_shared_parameter(history: list[tuple]) -> float:
    """Fit shared-goal proportion using previous matches only.

    History rows:
    home expected goals, away expected goals, actual home goals,
    actual away goals.
    """
    if len(history) < MIN_FIT_MATCHES:
        return 0.0

    candidates = np.linspace(
        0.0,
        MAX_ALPHA,
        ALPHA_STEPS,
    )

    best_alpha = 0.0
    best_loss = float("inf")

    for alpha in candidates:
        total_loss = 0.0

        for home_mean, away_mean, home_goals, away_goals in history:
            log_probability = observed_score_log_probability(
                int(home_goals),
                int(away_goals),
                float(home_mean),
                float(away_mean),
                float(alpha),
            )

            total_loss -= log_probability

        mean_loss = total_loss / len(history)

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_alpha = float(alpha)

    weight = len(history) / (len(history) + SHRINKAGE)

    return float(best_alpha * weight)


def backtest(matches: pd.DataFrame) -> pd.DataFrame:
    """Run V7 first, then evaluate bivariate probabilities."""
    v7 = run_v7(matches)

    if v7.empty:
        return v7

    data = v7.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["match_day"] = data["date"].dt.normalize()

    data = data.sort_values(
        ["date", "league", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    histories = {}
    current_seasons = {}
    current_alpha = {}

    calibration_probabilities = {}
    calibration_actuals = {}
    current_calibration_season = {}
    current_multiplier = {}

    output = []

    for match_day, day in data.groupby(
        "match_day",
        sort=True,
    ):
        day_records = []

        for row in day.itertuples(index=False):
            league = str(row.league)
            season = season_key(match_day)

            if league not in histories:
                histories[league] = []
                current_seasons[league] = None
                current_alpha[league] = 0.0

                calibration_probabilities[league] = []
                calibration_actuals[league] = []
                current_calibration_season[league] = None
                current_multiplier[league] = 1.0

            # Fit the shared parameter at the season boundary.
            if current_seasons[league] != season:
                current_seasons[league] = season

                current_alpha[league] = fit_shared_parameter(
                    histories[league]
                )

            # Fit draw calibration only on previous predictions.
            if current_calibration_season[league] != season:
                current_calibration_season[league] = season

                if (
                    len(calibration_actuals[league])
                    >= MIN_FIT_MATCHES
                ):
                    current_multiplier[league] = (
                        fit_draw_multiplier(
                            np.asarray(
                                calibration_probabilities[league],
                                dtype=float,
                            ),
                            np.asarray(
                                calibration_actuals[league],
                                dtype=int,
                            ),
                        )
                    )
                else:
                    current_multiplier[league] = 1.0

            home_mean = float(row.expected_home_goals)
            away_mean = float(row.expected_away_goals)

            alpha = current_alpha[league]

            raw_probabilities = bivariate_probabilities(
                home_mean,
                away_mean,
                alpha,
            )

            probabilities = apply_draw_calibration(
                raw_probabilities,
                current_multiplier[league],
            )

            actual_index = (
                0 if row.actual_home_goals > row.actual_away_goals
                else 1 if row.actual_home_goals == row.actual_away_goals
                else 2
            )

            one_hot = np.eye(3)[actual_index]
            predicted_index = int(np.argmax(probabilities))

            day_records.append({
                "date": row.date,
                "league": league,
                "home_team": row.home_team,
                "away_team": row.away_team,

                "home_mean": home_mean,
                "away_mean": away_mean,
                "shared_alpha": alpha,
                "shared_mean": (
                    alpha * min(home_mean, away_mean)
                ),

                "biv_raw_p1": float(raw_probabilities[0]),
                "biv_raw_px": float(raw_probabilities[1]),
                "biv_raw_p2": float(raw_probabilities[2]),

                "biv_p1": float(probabilities[0]),
                "biv_px": float(probabilities[1]),
                "biv_p2": float(probabilities[2]),

                "biv_calibration_k": float(
                    current_multiplier[league]
                ),

                "actual": ("1", "X", "2")[actual_index],
                "biv_predicted": (
                    "1", "X", "2"
                )[predicted_index],

                "biv_correct": int(
                    predicted_index == actual_index
                ),

                "biv_brier": float(
                    np.mean((probabilities - one_hot) ** 2)
                ),

                "biv_log_loss": float(
                    -math.log(
                        max(1e-15, probabilities[actual_index])
                    )
                ),

                "_history": (
                    home_mean,
                    away_mean,
                    int(row.actual_home_goals),
                    int(row.actual_away_goals),
                ),

                "_raw_probabilities": raw_probabilities,
                "_actual_index": actual_index,
            })

        # Update only after all matches on the date are predicted.
        for record in day_records:
            league = record["league"]

            histories[league].append(record["_history"])

            calibration_probabilities[league].append(
                record["_raw_probabilities"]
            )

            calibration_actuals[league].append(
                record["_actual_index"]
            )

            record.pop("_history")
            record.pop("_raw_probabilities")
            record.pop("_actual_index")

            output.append(record)

    result = pd.DataFrame(output)

    return result.merge(
        data[
            [
                "date",
                "league",
                "home_team",
                "away_team",
                "predicted",
                "correct",
                "brier",
                "log_loss",
            ]
        ],
        on=["date", "league", "home_team", "away_team"],
        how="left",
        validate="one_to_one",
    )


def print_metrics(results: pd.DataFrame, title: str) -> None:
    print(f"\n--- BIVARIATE POISSON: {title} ---")
    print(f"Matcher: {len(results)}")

    if results.empty:
        return

    print(
        f"Träffsäkerhet: {results.biv_correct.mean():.4%}"
    )
    print(
        f"Brier score: {results.biv_brier.mean():.6f}"
    )
    print(
        f"Log loss: {results.biv_log_loss.mean():.6f}"
    )

    print("\nModellens val:")
    print(
        results["biv_predicted"]
        .value_counts()
        .reindex(["1", "X", "2"], fill_value=0)
        .to_string()
    )

    print("\nGenomsnittliga sannolikheter:")
    print(
        results[["biv_p1", "biv_px", "biv_p2"]]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nGenomsnittlig delad målkomponent:")
    print(
        results[
            ["shared_alpha", "shared_mean"]
        ].mean().round(4).to_string()
    )

    print("\nJämförelse med V7:")
    print(
        f"V7 träffsäkerhet: {results.correct.mean():.4%}"
    )
    print(
        f"V7 Brier: {results.brier.mean():.6f}"
    )
    print(
        f"V7 log loss: {results.log_loss.mean():.6f}"
    )

    print("\nFörändring (Bivariate minus V7):")
    print(
        f"Träffsäkerhet: "
        f"{results.biv_correct.mean() - results.correct.mean():+.4%}"
    )
    print(
        f"Brier: "
        f"{results.biv_brier.mean() - results.brier.mean():+.6f}"
    )
    print(
        f"Log loss: "
        f"{results.biv_log_loss.mean() - results.log_loss.mean():+.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bivariate Poisson backtest using V7 goal predictions."
        )
    )

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="bivariate_predictions.csv",
    )

    args = parser.parse_args()

    results = backtest(pd.read_csv(args.input))
    results.to_csv(args.output, index=False)

    print_metrics(results, "SAMTLIGA LIGOR")

    for league in sorted(results["league"].unique()):
        print_metrics(
            results[results["league"] == league],
            str(league),
        )


if __name__ == "__main__":
    main()
