
"""Conditional draw experiment based on V7.

Tests whether V7's calibrated draw probability can be improved
using expected total goals and expected goal difference.

Predictions are for regulation time (60 minutes).
Training is league-specific and uses only earlier matches.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from src.hockey_backtest_v7 import backtest as run_v7
from src.hockey_backtest_v6 import season_key


MIN_TRAINING_MATCHES = 300
SHRINKAGE = 500.0
RIDGE = 2.0
MAX_ITERATIONS = 100
EPSILON = 1e-9


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-values))


def logit(probabilities):
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        EPSILON,
        1.0 - EPSILON,
    )
    return np.log(probabilities / (1.0 - probabilities))


def make_features(frame):
    """Build features from V7's pre-match predictions."""
    base_draw = logit(frame["px"].to_numpy(dtype=float))

    total_goals = frame[
        "expected_total_goals"
    ].to_numpy(dtype=float)

    goal_difference = np.abs(
        frame["expected_goal_difference"].to_numpy(dtype=float)
    )

    return np.column_stack([
        base_draw,
        total_goals,
        goal_difference,
    ])


def fit_logistic_model(features, targets):
    """Fit regularized logistic regression using NumPy Newton steps."""
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)

    if len(y) < MIN_TRAINING_MATCHES:
        return None

    if len(np.unique(y)) < 2:
        return None

    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)

    standardized = (x - means) / scales

    design = np.column_stack([
        np.ones(len(standardized)),
        standardized,
    ])

    coefficients = np.zeros(design.shape[1], dtype=float)

    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0

    for _ in range(MAX_ITERATIONS):
        probabilities = sigmoid(design @ coefficients)

        gradient = (
            design.T @ (probabilities - y)
            + penalty @ coefficients
        )

        weights = np.clip(
            probabilities * (1.0 - probabilities),
            1e-8,
            None,
        )

        hessian = (
            (design.T * weights) @ design
            + penalty
        )

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None

        coefficients -= step

        if np.max(np.abs(step)) < 1e-7:
            break

    return {
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "training_matches": len(y),
    }


def predict_draw_probability(model, features):
    x = np.asarray(features, dtype=float)

    standardized = (
        x - model["means"]
    ) / model["scales"]

    design = np.column_stack([
        np.ones(len(standardized)),
        standardized,
    ])

    return sigmoid(design @ model["coefficients"])


def adjust_probabilities(base_probabilities, new_draw_probability):
    """Change draw probability while preserving the 1-to-2 ratio."""
    base = np.asarray(base_probabilities, dtype=float)

    p1 = base[..., 0]
    px = base[..., 1]
    p2 = base[..., 2]

    new_px = np.clip(
        np.asarray(new_draw_probability, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    non_draw = np.maximum(1e-12, 1.0 - px)

    new_p1 = (p1 / non_draw) * (1.0 - new_px)
    new_p2 = (p2 / non_draw) * (1.0 - new_px)

    result = np.stack(
        [new_p1, new_px, new_p2],
        axis=-1,
    )

    result = np.clip(result, 1e-12, 1.0)
    return result / result.sum(axis=-1, keepdims=True)


def backtest(matches):
    """Run V7 and evaluate the conditional draw adjustment."""
    v7 = run_v7(matches)

    if v7.empty:
        return v7

    data = v7.copy()

    required = {
        "date",
        "league",
        "home_team",
        "away_team",
        "px",
        "p1",
        "p2",
        "expected_total_goals",
        "expected_goal_difference",
        "actual",
        "predicted",
        "correct",
        "brier",
        "log_loss",
    }

    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"V7 output is missing required columns: {sorted(missing)}"
        )

    data["date"] = pd.to_datetime(data["date"])
    data["match_day"] = data["date"].dt.normalize()
    data["season"] = data["date"].apply(season_key)

    data = data.sort_values(
        ["date", "league", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    features = make_features(data)

    data["_feature_0"] = features[:, 0]
    data["_feature_1"] = features[:, 1]
    data["_feature_2"] = features[:, 2]

    models = {}
    current_seasons = {}
    historical_features = {}
    historical_targets = {}

    output = []

    for match_day, day in data.groupby("match_day", sort=True):
        day_records = []

        # Fit at season boundaries, using earlier matches only.
        for _, row in day.iterrows():
            league = str(row["league"])
            season = int(row["season"])

            if league not in models:
                models[league] = None
                current_seasons[league] = None
                historical_features[league] = []
                historical_targets[league] = []

            if current_seasons[league] != season:
                current_seasons[league] = season

                training_count = len(
                    historical_targets[league]
                )

                if training_count >= MIN_TRAINING_MATCHES:
                    models[league] = fit_logistic_model(
                        np.asarray(
                            historical_features[league],
                            dtype=float,
                        ),
                        np.asarray(
                            historical_targets[league],
                            dtype=int,
                        ),
                    )
                else:
                    models[league] = None

        # Predict every match on this date before updating history.
        for _, row in day.iterrows():
            league = str(row["league"])

            base_probabilities = np.asarray(
                [
                    row["p1"],
                    row["px"],
                    row["p2"],
                ],
                dtype=float,
            )

            row_features = np.asarray([
                row["_feature_0"],
                row["_feature_1"],
                row["_feature_2"],
            ], dtype=float)

            model = models[league]

            if model is None:
                raw_model_draw = float(row["px"])
                shrinkage = 0.0
                final_draw = float(row["px"])
                training_count = len(
                    historical_targets[league]
                )
            else:
                raw_model_draw = float(
                    predict_draw_probability(
                        model,
                        row_features.reshape(1, -1),
                    )[0]
                )

                training_count = model["training_matches"]
                shrinkage = (
                    training_count
                    / (training_count + SHRINKAGE)
                )

                final_draw = (
                    shrinkage * raw_model_draw
                    + (1.0 - shrinkage) * float(row["px"])
                )

            probabilities = adjust_probabilities(
                base_probabilities,
                final_draw,
            )[0]

            actual = str(row["actual"])

            if actual not in ("1", "X", "2"):
                raise ValueError(
                    f"Unexpected actual outcome: {actual}"
                )

            actual_index = ("1", "X", "2").index(actual)
            one_hot = np.eye(3)[actual_index]
            predicted_index = int(np.argmax(probabilities))

            day_records.append({
                "date": row["date"],
                "league": league,
                "home_team": row["home_team"],
                "away_team": row["away_team"],

                "p1": float(probabilities[0]),
                "px": float(probabilities[1]),
                "p2": float(probabilities[2]),

                "conditional_raw_px": raw_model_draw,
                "conditional_shrinkage": shrinkage,
                "conditional_training_matches": training_count,

                "actual": actual,
                "predicted": ("1", "X", "2")[predicted_index],
                "correct": int(predicted_index == actual_index),

                "brier": float(
                    np.mean((probabilities - one_hot) ** 2)
                ),

                "log_loss": float(
                    -math.log(
                        max(1e-15, probabilities[actual_index])
                    )
                ),

                # V7 baseline for direct comparison.
                "v7_p1": float(row["p1"]),
                "v7_px": float(row["px"]),
                "v7_p2": float(row["p2"]),
                "v7_predicted": str(row["predicted"]),
                "v7_correct": int(row["correct"]),
                "v7_brier": float(row["brier"]),
                "v7_log_loss": float(row["log_loss"]),

                "_features": row_features,
                "_target": int(actual == "X"),
            })

        # Only after all same-day predictions are complete
        # may the day's results enter the training history.
        for record in day_records:
            league = record["league"]

            historical_features[league].append(
                record["_features"]
            )

            historical_targets[league].append(
                record["_target"]
            )

            record.pop("_features")
            record.pop("_target")

            output.append(record)

    return pd.DataFrame(output)


def print_metrics(results, title):
    print(f"\n--- CONDITIONAL DRAW: {title} ---")
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
    )

    print("\nGenomsnittliga sannolikheter:")
    print(
        results[["p1", "px", "p2"]]
        .mean()
        .round(4)
    )

    print("\nJämförelse mot V7:")
    print(
        f"V7 träffsäkerhet: {results.v7_correct.mean():.4%}"
    )
    print(
        f"V7 Brier: {results.v7_brier.mean():.6f}"
    )
    print(
        f"V7 log loss: {results.v7_log_loss.mean():.6f}"
    )

    print("\nFörändring (ny modell minus V7):")
    print(
        f"Träffsäkerhet: "
        f"{results.correct.mean() - results.v7_correct.mean():+.4%}"
    )
    print(
        f"Brier: "
        f"{results.brier.mean() - results.v7_brier.mean():+.6f}"
    )
    print(
        f"Log loss: "
        f"{results.log_loss.mean() - results.v7_log_loss.mean():+.6f}"
    )

    print("\nAntal kryssval:")
    print(
        f"V7: {(results.v7_predicted == 'X').sum()}"
    )
    print(
        f"Ny modell: {(results.predicted == 'X').sum()}"
    )

    print("\nAntal matcher med tränad justeringsmodell:")
    print(
        (results.conditional_training_matches > 0).sum()
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Conditional draw calibration based on V7 predictions."
        )
    )

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="conditional_draw_predictions.csv",
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
