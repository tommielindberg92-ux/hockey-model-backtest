
"""V4: CatBoost + V3 ensemble for regulation-time hockey predictions."""

from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.hockey_backtest_v3 import TimeWeightedModel

REQUIRED = {
    "date", "league", "home_team", "away_team",
    "reg_home_goals", "reg_away_goals",
}

CATEGORICAL = ["league", "home_team", "away_team"]
LABELS = {"1": 0, "X": 1, "2": 2}


class FeatureState:
    def __init__(self):
        self.history = {}
        self.elo = {}
        self.last_date = {}
        self.global_home = []
        self.global_draw = []

    def rating(self, team):
        return self.elo.get(team, 1500.0)

    def team_features(self, team, date, venue):
        games = self.history.get(team, [])
        selected = [
            g for g in games
            if venue == "all" or g["venue"] == venue
        ]

        result = {}
        for window in (5, 10, 20):
            recent = selected[-window:]
            prefix = f"{team}_{venue}_{window}"

            if recent:
                result[f"{prefix}_points"] = (
                    sum(g["points"] for g in recent) / len(recent)
                )
                result[f"{prefix}_gf"] = (
                    sum(g["gf"] for g in recent) / len(recent)
                )
                result[f"{prefix}_ga"] = (
                    sum(g["ga"] for g in recent) / len(recent)
                )
                result[f"{prefix}_winrate"] = (
                    sum(g["points"] == 3 for g in recent)
                    / len(recent)
                )
            else:
                result[f"{prefix}_points"] = 1.0
                result[f"{prefix}_gf"] = 2.8
                result[f"{prefix}_ga"] = 2.8
                result[f"{prefix}_winrate"] = 0.33

        if games:
            days = max(0, (date - games[-1]["date"]).days)
            result[f"{team}_rest_days"] = min(days, 30)
            result[f"{team}_games_7d"] = sum(
                (date - g["date"]).days <= 7 for g in games
            )
            result[f"{team}_games_14d"] = sum(
                (date - g["date"]).days <= 14 for g in games
            )
        else:
            result[f"{team}_rest_days"] = 7
            result[f"{team}_games_7d"] = 0
            result[f"{team}_games_14d"] = 0

        result[f"{team}_elo"] = self.rating(team)
        return result

    def make_features(self, row, date):
        home = str(row.home_team)
        away = str(row.away_team)
        league = str(row.league)

        features = {
            "league": league,
            "home_team": home,
            "away_team": away,
            "elo_diff": self.rating(home) - self.rating(away),
            "elo_home": self.rating(home),
            "elo_away": self.rating(away),
            "previous_home_rate": (
                np.mean(self.global_home) if self.global_home else 0.47
            ),
            "previous_draw_rate": (
                np.mean(self.global_draw) if self.global_draw else 0.21
            ),
        }

        features.update(self.team_features(home, date, "all"))
        features.update(self.team_features(away, date, "all"))
        features.update(self.team_features(home, date, "home"))
        features.update(self.team_features(away, date, "away"))

        return features

    def update(self, row, date):
        home = str(row.home_team)
        away = str(row.away_team)
        hg = int(row.reg_home_goals)
        ag = int(row.reg_away_goals)

        rh = self.rating(home)
        ra = self.rating(away)
        expected = 1 / (1 + 10 ** ((ra - rh - 55) / 400))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        change = 20 * (actual - expected)

        self.elo[home] = rh + change
        self.elo[away] = ra - change

        home_points = 3 if hg > ag else 1 if hg == ag else 0
        away_points = 3 if ag > hg else 1 if hg == ag else 0

        self.history.setdefault(home, []).append({
            "date": date, "gf": hg, "ga": ag,
            "points": home_points, "venue": "home",
        })
        self.history.setdefault(away, []).append({
            "date": date, "gf": ag, "ga": hg,
            "points": away_points, "venue": "away",
        })

        self.global_home.append(int(hg > ag))
        self.global_draw.append(int(hg == ag))


def season_key(date):
    return date.year if date.month >= 8 else date.year - 1


def backtest(matches):
    missing = REQUIRED - set(matches.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = matches.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date", kind="stable").reset_index(drop=True)

    state = FeatureState()
    v3 = TimeWeightedModel()

    feature_rows = []
    target_rows = []
    output = []
    model = None
    current_season = None

    for date, day in data.groupby("date", sort=True):
        season = season_key(date)

        # Träna endast på avslutade tidigare säsonger.
        if season != current_season:
            current_season = season
            model = None

            if len(feature_rows) >= 500:
                train_x = pd.DataFrame(feature_rows)
                train_y = np.asarray(target_rows)

                if len(np.unique(train_y)) == 3:
                    model = CatBoostClassifier(
                        loss_function="MultiClass",
                        eval_metric="MultiClass",
                        iterations=500,
                        depth=6,
                        learning_rate=0.04,
                        l2_leaf_reg=5,
                        random_seed=42,
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=2,
                    )
                    model.fit(
                        train_x,
                        train_y,
                        cat_features=CATEGORICAL,
                    )

        day_updates = []
        day_features = []

        # Alla matcher samma datum får prognoser före uppdatering.
        for row in day.itertuples(index=False):
            features = state.make_features(row, date)
            day_features.append(features)

            base_probs = v3.predict(
                str(row.league),
                str(row.home_team),
                str(row.away_team),
            )

            if model is not None:
                x = pd.DataFrame([features])
                ml_probs = model.predict_proba(x)[0]

                # Ensemble: CatBoost 65 %, V3 35 %.
                probs = 0.65 * ml_probs + 0.35 * np.asarray(base_probs)
                probs = probs / probs.sum()
            else:
                probs = np.asarray(base_probs)

            hg = int(row.reg_home_goals)
            ag = int(row.reg_away_goals)
            actual_index = 0 if hg > ag else 1 if hg == ag else 2
            actual = ("1", "X", "2")[actual_index]

            output.append({
                "date": date,
                "league": str(row.league),
                "home_team": str(row.home_team),
                "away_team": str(row.away_team),
                "p1": probs[0],
                "px": probs[1],
                "p2": probs[2],
                "actual": actual,
                "correct": int(np.argmax(probs) == actual_index),
                "brier": float(np.mean(
                    (probs - np.eye(3)[actual_index]) ** 2
                )),
                "log_loss": float(
                    -math.log(max(1e-15, probs[actual_index]))
                ),
            })

            day_updates.append(row)

        # Först efter dagens samtliga prognoser uppdateras historiken.
        for row in day_updates:
            state.update(row, date)
            v3.update(
                str(row.league),
                str(row.home_team),
                str(row.away_team),
                int(row.reg_home_goals),
                int(row.reg_away_goals),
            )

            # Träningsraden blir tillgänglig först efter matchen.
            feature_rows.append(
                state.make_features(row, date)
            )
            hg = int(row.reg_home_goals)
            ag = int(row.reg_away_goals)
            target_rows.append(0 if hg > ag else 1 if hg == ag else 2)

    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="v4_predictions.csv")
    args = parser.parse_args()

    results = backtest(pd.read_csv(args.input))
    results.to_csv(args.output, index=False)

    print(f"Matcher: {len(results)}")
    print(f"Träffsäkerhet: {results.correct.mean():.4%}")
    print(f"Brier score: {results.brier.mean():.5f}")
    print(f"Log loss: {results.log_loss.mean():.5f}")
    print("\nModellens val:")
    print(
        results[["p1", "px", "p2"]].idxmax(axis=1)
        .value_counts()
        .reindex(["p1", "px", "p2"], fill_value=0)
    )
    print("\nGenomsnittliga sannolikheter:")
    print(results[["p1", "px", "p2"]].mean().round(4))


if __name__ == "__main__":
    main()
