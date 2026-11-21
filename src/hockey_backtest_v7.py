
"""V7: league-specific goal models with chronological draw calibration."""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.hockey_backtest_v3 import (
    TimeWeightedModel,
    regulation_probabilities,
)
from src.hockey_backtest_v6 import FeatureState, season_key


REQUIRED = {
    "date", "league", "home_team", "away_team",
    "reg_home_goals", "reg_away_goals",
}

CATEGORICAL = ["league", "home_team", "away_team"]

MIN_CALIBRATION_MATCHES = 300
CALIBRATION_SHRINKAGE = 500
MAX_ABS_LOG_MULTIPLIER = 0.5


def train_goal_models(features, home_goals, away_goals):
    x = pd.DataFrame(features)

    common = dict(
        loss_function="RMSE",
        iterations=700,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=6,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )

    home_model = CatBoostRegressor(
        **common,
        random_seed=71,
    )
    away_model = CatBoostRegressor(
        **common,
        random_seed=72,
    )

    home_model.fit(
        x,
        np.asarray(home_goals, dtype=float),
        cat_features=CATEGORICAL,
    )
    away_model.fit(
        x,
        np.asarray(away_goals, dtype=float),
        cat_features=CATEGORICAL,
    )

    return home_model, away_model


def apply_draw_calibration(probabilities, multiplier):
    """Adjust draw odds while preserving the ratio between 1 and 2."""
    probs = np.asarray(probabilities, dtype=float)

    p1 = probs[..., 0]
    px = probs[..., 1]
    p2 = probs[..., 2]

    denominator = 1.0 - px + multiplier * px
    calibrated_x = multiplier * px / denominator

    remaining = 1.0 - calibrated_x
    non_draw = np.maximum(1e-15, 1.0 - px)

    calibrated_1 = p1 / non_draw * remaining
    calibrated_2 = p2 / non_draw * remaining

    result = np.stack(
        [calibrated_1, calibrated_x, calibrated_2],
        axis=-1,
    )

    result = np.clip(result, 1e-15, 1.0)
    return result / result.sum(axis=-1, keepdims=True)


def fit_draw_multiplier(probabilities, actual_indices):
    """
    Fit a draw-odds multiplier using only earlier predictions.

    A deterministic grid search avoids adding an optimization dependency.
    The fitted value is shrunk toward 1 to reduce overfitting.
    """
    probs = np.asarray(probabilities, dtype=float)
    actual = np.asarray(actual_indices, dtype=int)

    count = len(actual)

    if count < MIN_CALIBRATION_MATCHES:
        return 1.0

    px = np.clip(probs[:, 1], 1e-12, 1.0 - 1e-12)
    p_actual = np.clip(
        probs[np.arange(count), actual],
        1e-12,
        1.0,
    )

    log_multipliers = np.linspace(
        -MAX_ABS_LOG_MULTIPLIER,
        MAX_ABS_LOG_MULTIPLIER,
        1001,
    )

    best_log_multiplier = 0.0
    best_loss = float("inf")

    is_draw = actual == 1

    for log_multiplier in log_multipliers:
        multiplier = math.exp(float(log_multiplier))

        calibrated_x = (
            multiplier * px
            / (1.0 - px + multiplier * px)
        )

        draw_loss = -np.log(
            np.clip(calibrated_x, 1e-12, 1.0)
        )

        non_draw_loss = (
            -np.log(p_actual)
            - np.log(np.clip(1.0 - calibrated_x, 1e-12, 1.0))
            + np.log(1.0 - px)
        )

        losses = np.where(
            is_draw,
            draw_loss,
            non_draw_loss,
        )

        mean_loss = float(np.mean(losses))

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_log_multiplier = float(log_multiplier)

    # Shrink toward no adjustment when the calibration sample is small.
    shrinkage = count / (
        count + CALIBRATION_SHRINKAGE
    )

    shrunk_log_multiplier = (
        best_log_multiplier * shrinkage
    )

    return float(math.exp(shrunk_log_multiplier))


def backtest(matches):
    missing = REQUIRED - set(matches.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = matches.copy()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="raise",
        format="mixed",
    )

    for column in ("reg_home_goals", "reg_away_goals"):
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

        if data[column].isna().any() or (data[column] < 0).any():
            raise ValueError(f"Invalid values in {column}")

        if (data[column] % 1 != 0).any():
            raise ValueError(f"Non-integer goals in {column}")

    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing required match values")

    if (data["home_team"] == data["away_team"]).any():
        raise ValueError("A team cannot play against itself")

    # Preserve kickoff timestamps but process all matches on a
    # calendar day before allowing any results from that day into history.
    data["match_day"] = data["date"].dt.normalize()

    data = data.sort_values(
        ["date", "league", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    state = FeatureState()
    baseline = TimeWeightedModel()

    feature_rows = {}
    home_targets = {}
    away_targets = {}
    current_season = {}
    league_models = {}

    # Calibration history contains only predictions and outcomes
    # from completed earlier matches.
    calibration_probabilities = {}
    calibration_actuals = {}
    calibration_season = {}
    calibration_multiplier = {}
    calibration_match_count = {}

    output = []

    for match_day, day in data.groupby("match_day", sort=True):
        day_records = []

        # Predict every match on this calendar day before updating
        # any model or calibration history with today's results.
        for row in day.itertuples(index=False):
            league = str(row.league)
            home = str(row.home_team)
            away = str(row.away_team)
            season = season_key(match_day)

            if league not in feature_rows:
                feature_rows[league] = []
                home_targets[league] = []
                away_targets[league] = []
                current_season[league] = None
                league_models[league] = None

                calibration_probabilities[league] = []
                calibration_actuals[league] = []
                calibration_season[league] = None
                calibration_multiplier[league] = 1.0
                calibration_match_count[league] = 0

            # Refit the draw calibration at the start of each season.
            # Only previous seasons' predictions and results are used.
            if season != calibration_season[league]:
                calibration_season[league] = season

                historical_count = len(
                    calibration_actuals[league]
                )

                calibration_match_count[league] = historical_count

                if historical_count >= MIN_CALIBRATION_MATCHES:
                    historical_probs = np.asarray(
                        calibration_probabilities[league],
                        dtype=float,
                    )
                    historical_actuals = np.asarray(
                        calibration_actuals[league],
                        dtype=int,
                    )

                    calibration_multiplier[league] = (
                        fit_draw_multiplier(
                            historical_probs,
                            historical_actuals,
                        )
                    )
                else:
                    calibration_multiplier[league] = 1.0

            # Retrain the league-specific goal models at season boundaries.
            if season != current_season[league]:
                current_season[league] = season
                league_models[league] = None

                if len(feature_rows[league]) >= 500:
                    league_models[league] = train_goal_models(
                        feature_rows[league],
                        home_targets[league],
                        away_targets[league],
                    )

            baseline.prepare_date(league, match_day)
            features = state.make_features(row, match_day)

            base_home, base_away = baseline.expected_goals(
                league,
                home,
                away,
            )

            models = league_models[league]

            if models is not None:
                home_model, away_model = models
                x = pd.DataFrame([features])

                ml_home = float(home_model.predict(x)[0])
                ml_away = float(away_model.predict(x)[0])

                expected_home = (
                    0.70 * ml_home + 0.30 * base_home
                )
                expected_away = (
                    0.70 * ml_away + 0.30 * base_away
                )
            else:
                expected_home = base_home
                expected_away = base_away

            expected_home = float(
                np.clip(expected_home, 0.5, 6.0)
            )
            expected_away = float(
                np.clip(expected_away, 0.5, 6.0)
            )

            raw_probs = np.asarray(
                regulation_probabilities(
                    expected_home,
                    expected_away,
                ),
                dtype=float,
            )
            raw_probs /= raw_probs.sum()

            multiplier = calibration_multiplier[league]

            probs = apply_draw_calibration(
                raw_probs,
                multiplier,
            )

            hg = int(row.reg_home_goals)
            ag = int(row.reg_away_goals)

            actual_index = (
                0 if hg > ag else 1 if hg == ag else 2
            )
            actual = ("1", "X", "2")[actual_index]

            one_hot = np.eye(3)[actual_index]
            predicted_index = int(np.argmax(probs))

            day_records.append({
                "row": row,
                "features": features,
                "home_target": hg,
                "away_target": ag,
                "raw_probs": raw_probs,
                "actual_index": actual_index,
                "output": {
                    "date": row.date,
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                    "expected_home_goals": expected_home,
                    "expected_away_goals": expected_away,
                    "raw_p1": float(raw_probs[0]),
                    "raw_px": float(raw_probs[1]),
                    "raw_p2": float(raw_probs[2]),
                    "p1": float(probs[0]),
                    "px": float(probs[1]),
                    "p2": float(probs[2]),
                    "calibration_k": float(multiplier),
                    "calibration_matches": int(
                        calibration_match_count[league]
                    ),
                    "actual": actual,
                    "predicted": ("1", "X", "2")[predicted_index],
                    "correct": int(predicted_index == actual_index),
                    "brier": float(
                        np.mean((probs - one_hot) ** 2)
                    ),
                    "log_loss": float(
                        -math.log(
                            max(1e-15, probs[actual_index])
                        )
                    ),
                },
            })

        # Update histories only after all matches of the day
        # have been predicted.
        for record in day_records:
            row = record["row"]
            league = str(row.league)

            state.update(row, match_day)

            baseline.update(
                league,
                str(row.home_team),
                str(row.away_team),
                int(row.reg_home_goals),
                int(row.reg_away_goals),
            )

            feature_rows[league].append(record["features"])
            home_targets[league].append(record["home_target"])
            away_targets[league].append(record["away_target"])

            # Store RAW probabilities. Never train the calibrator
            # on probabilities that it has already transformed.
            calibration_probabilities[league].append(
                record["raw_probs"]
            )
            calibration_actuals[league].append(
                record["actual_index"]
            )

            output.append(record["output"])

    return pd.DataFrame(output)


def print_metrics(results, title):
    print(f"\n--- {title} ---")
    print(f"Matcher: {len(results)}")

    if results.empty:
        print("Inga matcher att utvärdera.")
        return

    print(f"Träffsäkerhet: {results.correct.mean():.4%}")
    print(f"Brier score: {results.brier.mean():.5f}")
    print(f"Log loss: {results.log_loss.mean():.5f}")

    print("\nModellens val:")
    print(
        results["predicted"]
        .value_counts()
        .reindex(["1", "X", "2"], fill_value=0)
    )

    print("\nVerkliga utfall:")
    print(
        results["actual"]
        .value_counts(normalize=True)
        .reindex(["1", "X", "2"])
        .round(4)
    )

    print("\nGenomsnittliga sannolikheter:")
    print(results[["p1", "px", "p2"]].mean().round(4))

    print("\nGenomsnittliga råa sannolikheter:")
    print(
        results[["raw_p1", "raw_px", "raw_p2"]]
        .mean()
        .round(4)
    )

    print("\nKalibrering:")
    print(
        results[
            ["calibration_k", "calibration_matches"]
        ]
        .groupby("calibration_k")
        .first()
        .to_string()
    )

    print("\nGenomsnittligt förväntade mål:")
    print(
        results[
            ["expected_home_goals", "expected_away_goals"]
        ]
        .mean()
        .round(3)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="v7_predictions.csv",
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
