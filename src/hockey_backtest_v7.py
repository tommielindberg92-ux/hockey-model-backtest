
"""V7: league-specific goal models, draw calibration and Dixon-Coles test."""

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

MIN_DC_MATCHES = 300
DC_SHRINKAGE = 500
DC_RHO_LIMIT = 0.15


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

    home_model = CatBoostRegressor(**common, random_seed=71)
    away_model = CatBoostRegressor(**common, random_seed=72)

    home_model.fit(
        x, np.asarray(home_goals, dtype=float),
        cat_features=CATEGORICAL,
    )
    away_model.fit(
        x, np.asarray(away_goals, dtype=float),
        cat_features=CATEGORICAL,
    )

    return home_model, away_model


def apply_draw_calibration(probabilities, multiplier):
    """Adjust draw odds while preserving the ratio between 1 and 2."""
    probs = np.asarray(probabilities, dtype=float)

    p1, px, p2 = probs[..., 0], probs[..., 1], probs[..., 2]
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
    """Fit draw calibration using earlier predictions only."""
    probs = np.asarray(probabilities, dtype=float)
    actual = np.asarray(actual_indices, dtype=int)
    count = len(actual)

    if count < MIN_CALIBRATION_MATCHES:
        return 1.0

    px = np.clip(probs[:, 1], 1e-12, 1.0 - 1e-12)
    p_actual = np.clip(
        probs[np.arange(count), actual], 1e-12, 1.0
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

        mean_loss = float(np.mean(
            np.where(is_draw, draw_loss, non_draw_loss)
        ))

        if mean_loss < best_loss:
            best_loss = mean_loss
            best_log_multiplier = float(log_multiplier)

    shrinkage = count / (count + CALIBRATION_SHRINKAGE)
    return float(math.exp(best_log_multiplier * shrinkage))


def poisson_probabilities(lam, max_goals=12):
    """Poisson probabilities for a single expected-goal value."""
    return np.asarray([
        math.exp(
            -lam + goals * math.log(lam)
            - math.lgamma(goals + 1)
        )
        for goals in range(max_goals + 1)
    ], dtype=float)


def dixon_coles_tau(home_goals, away_goals, home_lambda,
                    away_lambda, rho):
    """Dixon-Coles low-score correction factor."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_lambda * away_lambda * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_lambda * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_lambda * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_probabilities(home_lambda, away_lambda, rho,
                              max_goals=12):
    """Calculate normalized 1/X/2 probabilities with DC correction."""
    home = poisson_probabilities(home_lambda, max_goals)
    away = poisson_probabilities(away_lambda, max_goals)

    matrix = np.outer(home, away)

    matrix[0, 0] *= dixon_coles_tau(
        0, 0, home_lambda, away_lambda, rho
    )
    matrix[0, 1] *= dixon_coles_tau(
        0, 1, home_lambda, away_lambda, rho
    )
    matrix[1, 0] *= dixon_coles_tau(
        1, 0, home_lambda, away_lambda, rho
    )
    matrix[1, 1] *= dixon_coles_tau(
        1, 1, home_lambda, away_lambda, rho
    )

    matrix = np.clip(matrix, 0.0, None)
    total = matrix.sum()

    if total <= 0 or not np.isfinite(total):
        return np.asarray(
            regulation_probabilities(home_lambda, away_lambda),
            dtype=float,
        )

    matrix /= total

    p1 = float(np.tril(matrix, k=-1).sum())
    px = float(np.trace(matrix))
    p2 = float(np.triu(matrix, k=1).sum())

    result = np.asarray([p1, px, p2], dtype=float)
    return result / result.sum()


def fit_dixon_coles_rho(history):
    """
    Estimate rho from earlier matches only.

    Each history item contains:
    (predicted_home_lambda, predicted_away_lambda,
     actual_home_goals, actual_away_goals).

    The likelihood contribution is the standard DC tau correction.
    A shrinkage factor reduces instability in smaller samples.
    """
    if len(history) < MIN_DC_MATCHES:
        return 0.0

    candidates = np.linspace(
        -DC_RHO_LIMIT, DC_RHO_LIMIT, 1201
    )

    best_rho = 0.0
    best_log_likelihood = -float("inf")

    for rho in candidates:
        log_likelihood = 0.0
        valid = True

        for home_lambda, away_lambda, hg, ag in history:
            tau = dixon_coles_tau(
                hg, ag, home_lambda, away_lambda, rho
            )

            if tau <= 1e-10 or not np.isfinite(tau):
                valid = False
                break

            if hg <= 1 and ag <= 1:
                log_likelihood += math.log(tau)

        if valid and log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_rho = float(rho)

    shrinkage = len(history) / (
        len(history) + DC_SHRINKAGE
    )
    return float(best_rho * shrinkage)


def backtest(matches):
    missing = REQUIRED - set(matches.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = matches.copy()
    data["date"] = pd.to_datetime(
        data["date"], errors="raise", format="mixed"
    )

    for column in ("reg_home_goals", "reg_away_goals"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if data[column].isna().any() or (data[column] < 0).any():
            raise ValueError(f"Invalid values in {column}")
        if (data[column] % 1 != 0).any():
            raise ValueError(f"Non-integer goals in {column}")

    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing required match values")
    if (data["home_team"] == data["away_team"]).any():
        raise ValueError("A team cannot play against itself")

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

    calibration_probabilities = {}
    calibration_actuals = {}
    calibration_season = {}
    calibration_multiplier = {}
    calibration_match_count = {}

    dc_history = {}
    dc_rho = {}
    dc_calibration_probabilities = {}
    dc_calibration_actuals = {}
    dc_calibration_multiplier = {}
    dc_calibration_match_count = {}

    output = []

    for match_day, day in data.groupby("match_day", sort=True):
        day_records = []

        # All matches on the date are predicted before results are updated.
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

                dc_history[league] = []
                dc_rho[league] = 0.0
                dc_calibration_probabilities[league] = []
                dc_calibration_actuals[league] = []
                dc_calibration_multiplier[league] = 1.0
                dc_calibration_match_count[league] = 0

            # Existing V7 draw calibration.
            if season != calibration_season[league]:
                calibration_season[league] = season
                historical_count = len(calibration_actuals[league])
                calibration_match_count[league] = historical_count

                if historical_count >= MIN_CALIBRATION_MATCHES:
                    calibration_multiplier[league] = fit_draw_multiplier(
                        np.asarray(
                            calibration_probabilities[league],
                            dtype=float,
                        ),
                        np.asarray(
                            calibration_actuals[league], dtype=int
                        ),
                    )
                else:
                    calibration_multiplier[league] = 1.0

            # Dixon-Coles parameter and its own calibration.
            if season != getattr(
                backtest, "_dc_season_" + league, None
            ):
                setattr(backtest, "_dc_season_" + league, season)

                dc_rho[league] = fit_dixon_coles_rho(
                    dc_history[league]
                )

                dc_count = len(dc_calibration_actuals[league])
                dc_calibration_match_count[league] = dc_count

                if dc_count >= MIN_CALIBRATION_MATCHES:
                    dc_calibration_multiplier[league] = (
                        fit_draw_multiplier(
                            np.asarray(
                                dc_calibration_probabilities[league],
                                dtype=float,
                            ),
                            np.asarray(
                                dc_calibration_actuals[league],
                                dtype=int,
                            ),
                        )
                    )
                else:
                    dc_calibration_multiplier[league] = 1.0

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
                league, home, away
            )

            models = league_models[league]
            if models is not None:
                home_model, away_model = models
                x = pd.DataFrame([features])

                ml_home = float(home_model.predict(x)[0])
                ml_away = float(away_model.predict(x)[0])

                expected_home = 0.70 * ml_home + 0.30 * base_home
                expected_away = 0.70 * ml_away + 0.30 * base_away
            else:
                expected_home = base_home
                expected_away = base_away

            expected_home = float(np.clip(expected_home, 0.5, 6.0))
            expected_away = float(np.clip(expected_away, 0.5, 6.0))

            # Existing independent-Poisson V7 probabilities.
            raw_probs = np.asarray(
                regulation_probabilities(expected_home, expected_away),
                dtype=float,
            )
            raw_probs /= raw_probs.sum()

            multiplier = calibration_multiplier[league]
            probs = apply_draw_calibration(raw_probs, multiplier)

            # Dixon-Coles alternative using identical expected goals.
            rho = dc_rho[league]
            dc_raw_probs = dixon_coles_probabilities(
                expected_home, expected_away, rho
            )
            dc_multiplier = dc_calibration_multiplier[league]
            dc_probs = apply_draw_calibration(
                dc_raw_probs, dc_multiplier
            )

            hg = int(row.reg_home_goals)
            ag = int(row.reg_away_goals)
            actual_index = 0 if hg > ag else 1 if hg == ag else 2
            actual = ("1", "X", "2")[actual_index]
            one_hot = np.eye(3)[actual_index]

            predicted_index = int(np.argmax(probs))
            dc_predicted_index = int(np.argmax(dc_probs))

            day_records.append({
                "row": row,
                "features": features,
                "home_target": hg,
                "away_target": ag,
                "raw_probs": raw_probs,
                "dc_raw_probs": dc_raw_probs,
                "actual_index": actual_index,
                "expected_home": expected_home,
                "expected_away": expected_away,
                "output": {
                    "date": row.date,
                    "league": league,
                    "home_team": home,
                    "away_team": away,
                    "expected_home_goals": expected_home,
                    "expected_away_goals": expected_away,
                    "actual_home_goals": hg,
                    "actual_away_goals": ag,
                    "expected_total_goals": expected_home + expected_away,
                    "actual_total_goals": hg + ag,
                    "expected_goal_difference": expected_home - expected_away,
                    "actual_goal_difference": hg - ag,
                    "home_goal_error": expected_home - hg,
                    "away_goal_error": expected_away - ag,

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
                    "brier": float(np.mean((probs - one_hot) ** 2)),
                    "log_loss": float(
                        -math.log(max(1e-15, probs[actual_index]))
                    ),

                    "dc_rho": float(rho),
                    "dc_raw_p1": float(dc_raw_probs[0]),
                    "dc_raw_px": float(dc_raw_probs[1]),
                    "dc_raw_p2": float(dc_raw_probs[2]),
                    "dc_p1": float(dc_probs[0]),
                    "dc_px": float(dc_probs[1]),
                    "dc_p2": float(dc_probs[2]),
                    "dc_calibration_k": float(dc_multiplier),
                    "dc_calibration_matches": int(
                        dc_calibration_match_count[league]
                    ),
                    "dc_predicted": (
                        "1", "X", "2"
                    )[dc_predicted_index],
                    "dc_correct": int(
                        dc_predicted_index == actual_index
                    ),
                    "dc_brier": float(
                        np.mean((dc_probs - one_hot) ** 2)
                    ),
                    "dc_log_loss": float(
                        -math.log(
                            max(1e-15, dc_probs[actual_index])
                        )
                    ),
                },
            })

        # Only after all predictions for the date are made may results update.
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

            calibration_probabilities[league].append(
                record["raw_probs"]
            )
            calibration_actuals[league].append(
                record["actual_index"]
            )

            dc_calibration_probabilities[league].append(
                record["dc_raw_probs"]
            )
            dc_calibration_actuals[league].append(
                record["actual_index"]
            )

            dc_history[league].append((
                record["expected_home"],
                record["expected_away"],
                record["home_target"],
                record["away_target"],
            ))

            output.append(record["output"])

    # Avoid carrying season state between separate backtest invocations.
    for league in list(feature_rows):
        attribute = "_dc_season_" + league
        if hasattr(backtest, attribute):
            delattr(backtest, attribute)

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
    print(results["predicted"].value_counts().reindex(
        ["1", "X", "2"], fill_value=0
    ))

    print("\nVerkliga utfall:")
    print(results["actual"].value_counts(normalize=True).reindex(
        ["1", "X", "2"]
    ).round(4))

    print("\nGenomsnittliga sannolikheter:")
    print(results[["p1", "px", "p2"]].mean().round(4))

    print("\nGenomsnittliga råa sannolikheter:")
    print(results[["raw_p1", "raw_px", "raw_p2"]].mean().round(4))

    print("\nKalibrering:")
    print(results[
        ["calibration_k", "calibration_matches"]
    ].groupby("calibration_k").first().to_string())

    print("\nMÅLMODELL – prognos jämfört med verklighet:")
    goal_columns = [
        "expected_home_goals", "actual_home_goals",
        "expected_away_goals", "actual_away_goals",
        "expected_total_goals", "actual_total_goals",
    ]
    print(results[goal_columns].mean().round(3))

    for side, error_column in (
        ("Hemma", "home_goal_error"),
        ("Borta", "away_goal_error"),
    ):
        errors = results[error_column]
        print(
            f"{side}: MAE={errors.abs().mean():.4f}, "
            f"RMSE={np.sqrt(np.mean(errors ** 2)):.4f}, "
            f"bias={errors.mean():+.4f}"
        )

    total_error = (
        results["expected_total_goals"] - results["actual_total_goals"]
    )
    print(
        f"Totala mål: MAE={total_error.abs().mean():.4f}, "
        f"bias={total_error.mean():+.4f}"
    )

    correlation = results["home_goal_error"].corr(
        results["away_goal_error"]
    )
    print(f"Korrelation mellan hemma-/bortamålsfel: {correlation:.4f}")

    print("\nGenomsnittlig målskillnad:")
    print(
        "Prognos:",
        round(results["expected_goal_difference"].mean(), 3),
        "| Utfall:",
        round(results["actual_goal_difference"].mean(), 3),
    )


def print_dc_comparison(results, title):
    print(f"\n--- DIXON-COLES JÄMFÖRELSE: {title} ---")
    print(f"Matcher: {len(results)}")

    if results.empty:
        return

    print(
        f"V7 Brier: {results.brier.mean():.6f} | "
        f"DC Brier: {results.dc_brier.mean():.6f}"
    )
    print(
        f"V7 log loss: {results.log_loss.mean():.6f} | "
        f"DC log loss: {results.dc_log_loss.mean():.6f}"
    )
    print(
        f"V7 träffsäkerhet: {results.correct.mean():.4%} | "
        f"DC träffsäkerhet: {results.dc_correct.mean():.4%}"
    )

    print("\nGenomsnittliga sannolikheter:")
    print(
        "V7:",
        results[["p1", "px", "p2"]].mean().round(4).to_dict(),
    )
    print(
        "DC:",
        results[["dc_p1", "dc_px", "dc_p2"]]
        .mean().round(4).to_dict(),
    )

    print("\nAntal modellval:")
    print(
        "V7:",
        results["predicted"].value_counts()
        .reindex(["1", "X", "2"], fill_value=0).to_dict(),
    )
    print(
        "DC:",
        results["dc_predicted"].value_counts()
        .reindex(["1", "X", "2"], fill_value=0).to_dict(),
    )

    print("\nDixon-Coles rho:")
    print(results["dc_rho"].describe().round(4).to_string())

    print("\nFörändring (DC minus V7):")
    print(
        f"Brier: {results.dc_brier.mean() - results.brier.mean():+.6f}"
    )
    print(
        f"Log loss: "
        f"{results.dc_log_loss.mean() - results.log_loss.mean():+.6f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="v7_predictions.csv")
    args = parser.parse_args()

    results = backtest(pd.read_csv(args.input))
    results.to_csv(args.output, index=False)

    print_metrics(results, "SAMTLIGA LIGOR")
    print_dc_comparison(results, "SAMTLIGA LIGOR")

    for league in sorted(results["league"].unique()):
        group = results[results["league"] == league]
        print_metrics(group, league)
        print_dc_comparison(group, league)

    print("\n--- MÅLFEL PER SÄSONG ---")
    results["season"] = results["date"].apply(season_key)
    for (league, season), group in results.groupby(["league", "season"]):
        home_mae = group["home_goal_error"].abs().mean()
        away_mae = group["away_goal_error"].abs().mean()
        total_bias = (
            group["expected_total_goals"] - group["actual_total_goals"]
        ).mean()
        print(
            f"{league} {season}/{season + 1}: "
            f"matcher={len(group)}, "
            f"hemma_MAE={home_mae:.3f}, "
            f"borta_MAE={away_mae:.3f}, "
            f"total_bias={total_bias:+.3f}"
        )


if __name__ == "__main__":
    main()
