
"""V7: predict regulation goals first, then derive 1X2 probabilities."""

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


def train_goal_models(features, home_goals, away_goals):
    x = pd.DataFrame(features)

    common = dict(
        loss_function="RMSE",
        iterations=700,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=6,
        random_seed=71,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )

    home_model = CatBoostRegressor(**common)
    away_model = CatBoostRegressor(**{**common, "random_seed": 72})

    home_model.fit(
        x, np.asarray(home_goals, dtype=float),
        cat_features=CATEGORICAL,
    )
    away_model.fit(
        x, np.asarray(away_goals, dtype=float),
        cat_features=CATEGORICAL,
    )

    return home_model, away_model


def backtest(matches):
    missing = REQUIRED - set(matches.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = matches.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    for column in ("reg_home_goals", "reg_away_goals"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if data[column].isna().any() or (data[column] < 0).any():
            raise ValueError(f"Invalid values in {column}")

    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing required match values")

    if (data["home_team"] == data["away_team"]).any():
        raise ValueError("A team cannot play against itself")

    data = data.sort_values(
        ["date", "league", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    state = FeatureState()
    baseline = TimeWeightedModel()

    feature_rows = []
    home_targets = []
    away_targets = []
    output = []

    home_model = None
    away_model = None
    current_season = None

    for date, day in data.groupby("date", sort=True):
        season = season_key(date)

        # Train only on matches completed before this season.
        if season != current_season:
            current_season = season
            home_model = None
            away_model = None

            if len(feature_rows) >= 500:
                home_model, away_model = train_goal_models(
                    feature_rows,
                    home_targets,
                    away_targets,
                )

        day_records = []

        # Every match on this date is predicted before any result
        # from the same date is added to the model state.
        for row in day.itertuples(index=False):
            league = str(row.league)
            home = str(row.home_team)
            away = str(row.away_team)

            baseline.prepare_date(league, date)
            features = state.make_features(row, date)

            base_home, base_away = baseline.expected_goals(
                league, home, away
            )

            if home_model is not None:
                x = pd.DataFrame([features])

                ml_home = float(home_model.predict(x)[0])
                ml_away = float(away_model.predict(x)[0])

                # Blend learned goal estimates with the dynamic
                # V3 goal model to reduce extreme predictions.
                expected_home = 0.70 * ml_home + 0.30 * base_home
                expected_away = 0.70 * ml_away + 0.30 * base_away
            else:
                expected_home = base_home
                expected_away = base_away

            expected_home = float(np.clip(expected_home, 0.5, 6.0))
            expected_away = float(np.clip(expected_away, 0.5, 6.0))

            probs = np.asarray(
                regulation_probabilities(
                    expected_home,
                    expected_away,
                ),
                dtype=float,
            )
            probs /= probs.sum()

            hg = int(row.reg_home_goals)
            ag = int(row.reg_away_goals)

            actual_index = 0 if hg > ag else 1 if hg == ag else 2
            actual = ("1", "X", "2")[actual_index]
            one_hot = np.eye(3)[actual_index]
            predicted_index = int(np.argmax(probs))

            day_records.append({
                "row": row,
                "features": features,
                "home_target": hg,
                "away_target": ag,
                "output": {
                    "date": date,
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                    "expected_home_goals": expected_home,
                    "expected_away_goals": expected_away,
                    "p1": float(probs[0]),
                    "px": float(probs[1]),
                    "p2": float(probs[2]),
                    "actual": actual,
                    "predicted": ("1", "X", "2")[predicted_index],
                    "correct": int(predicted_index == actual_index),
                    "brier": float(np.mean((probs - one_hot) ** 2)),
                    "log_loss": float(
                        -math.log(max(1e-15, probs[actual_index]))
                    ),
                },
            })

        # Update only after all predictions for the date are complete.
        for record in day_records:
            row = record["row"]

            state.update(row, date)

            baseline.update(
                str(row.league),
                str(row.home_team),
                str(row.away_team),
                int(row.reg_home_goals),
                int(row.reg_away_goals),
            )

            feature_rows.append(record["features"])
            home_targets.append(record["home_target"])
            away_targets.append(record["away_target"])
            output.append(record["output"])

    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="v7_predictions.csv")
    args = parser.parse_args()

    results = backtest(pd.read_csv(args.input))
    results.to_csv(args.output, index=False)

    print(f"Matcher: {len(results)}")

    if len(results):
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

        print("\nGenomsnittligt förväntade mål:")
        print(
            results[
                ["expected_home_goals", "expected_away_goals"]
            ].mean().round(3)
        )


if __name__ == "__main__":
    main()
