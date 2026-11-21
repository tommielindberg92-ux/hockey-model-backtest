
"""V5: separat kryssmodell + separat hemma/borta-modell.

Alla prognoser avser 1/X/2 efter ordinarie tid.
Träning sker enbart på tidigare färdiga säsonger.
"""

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


def season_key(date):
    return date.year if date.month >= 8 else date.year - 1


class FeatureState:
    """Sparar endast information som fanns före matchen."""

    def __init__(self):
        self.history = {}
        self.elo = {}
        self.home_rates = {}
        self.draw_rates = {}

    def rating(self, league, team):
        return self.elo.get((league, team), 1500.0)

    def team_features(self, league, team, date, side, venue):
        games = self.history.get((league, team), [])

        selected = [
            game for game in games
            if venue == "all" or game["venue"] == venue
        ]

        result = {}

        for window in (5, 10, 20):
            recent = selected[-window:]
            prefix = f"{side}_{venue}_{window}"

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
            result[f"{side}_rest_days"] = min(days, 30)
            result[f"{side}_games_7d"] = sum(
                0 <= (date - g["date"]).days <= 7 for g in games
            )
            result[f"{side}_games_14d"] = sum(
                0 <= (date - g["date"]).days <= 14 for g in games
            )
        else:
            result[f"{side}_rest_days"] = 7
            result[f"{side}_games_7d"] = 0
            result[f"{side}_games_14d"] = 0

        result[f"{side}_elo"] = self.rating(league, team)
        return result

    def make_features(self, row, date):
        league = str(row.league)
        home = str(row.home_team)
        away = str(row.away_team)

        home_elo = self.rating(league, home)
        away_elo = self.rating(league, away)

        features = {
            "league": league,
            "home_team": home,
            "away_team": away,
            "elo_diff": home_elo - away_elo,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "previous_home_rate": (
                float(np.mean(self.home_rates[league]))
                if self.home_rates.get(league) else 0.47
            ),
            "previous_draw_rate": (
                float(np.mean(self.draw_rates[league]))
                if self.draw_rates.get(league) else 0.21
            ),
        }

        features.update(
            self.team_features(league, home, date, "home", "all")
        )
        features.update(
            self.team_features(league, away, date, "away", "all")
        )
        features.update(
            self.team_features(league, home, date, "home", "home")
        )
        features.update(
            self.team_features(league, away, date, "away", "away")
        )

        return features

    def update(self, row, date):
        league = str(row.league)
        home = str(row.home_team)
        away = str(row.away_team)
        hg = int(row.reg_home_goals)
        ag = int(row.reg_away_goals)

        rh = self.rating(league, home)
        ra = self.rating(league, away)

        expected = 1 / (1 + 10 ** ((ra - rh - 55) / 400))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        change = 20 * (actual - expected)

        self.elo[(league, home)] = rh + change
        self.elo[(league, away)] = ra - change

        home_points = 3 if hg > ag else 1 if hg == ag else 0
        away_points = 3 if ag > hg else 1 if hg == ag else 0

        self.history.setdefault((league, home), []).append({
            "date": date,
            "gf": hg,
            "ga": ag,
            "points": home_points,
            "venue": "home",
        })

        self.history.setdefault((league, away), []).append({
            "date": date,
            "gf": ag,
            "ga": hg,
            "points": away_points,
            "venue": "away",
        })

        self.home_rates.setdefault(league, []).append(int(hg > ag))
        self.draw_rates.setdefault(league, []).append(int(hg == ag))


def positive_probability(model, features, positive_class=1):
    """Hämtar sannolikheten för en bestämd klass."""
    probabilities = model.predict_proba(pd.DataFrame([features]))[0]

    for index, label in enumerate(model.classes_):
        if int(label) == positive_class:
            return float(probabilities[index])

    return 0.0


def train_models(feature_rows, targets):
    """Tränar kryssmodellen och hemma/borta-modellen separat."""

    x = pd.DataFrame(feature_rows)
    y = np.asarray(targets, dtype=int)

    draw_target = (y == 1).astype(int)
    draw_count = int(draw_target.sum())
    non_draw_count = int(len(draw_target) - draw_count)

    draw_model = None
    direction_model = None

    if draw_count >= 50 and non_draw_count >= 50:
        # Balansering hjälper modellen att lära sig kryss.
        # Sannolikheten korrigeras sedan tillbaka till ursprunglig
        # klassfördelning innan den används i slutprognosen.
        draw_weight = non_draw_count / draw_count

        draw_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=500,
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=6,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=2,
            class_weights=[1.0, float(draw_weight)],
        )

        draw_model.fit(
            x,
            draw_target,
            cat_features=CATEGORICAL,
        )

    # Riktning tränas enbart på matcher som inte slutade oavgjort.
    non_draw_mask = y != 1
    direction_y = (y[non_draw_mask] == 0).astype(int)

    if (
        int(non_draw_mask.sum()) >= 200
        and len(np.unique(direction_y)) == 2
    ):
        direction_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=500,
            depth=6,
            learning_rate=0.035,
            l2_leaf_reg=6,
            random_seed=43,
            verbose=False,
            allow_writing_files=False,
            thread_count=2,
        )

        direction_model.fit(
            x.loc[non_draw_mask].reset_index(drop=True),
            direction_y,
            cat_features=CATEGORICAL,
        )

    return draw_model, direction_model


def backtest(matches):
    missing = REQUIRED - set(matches.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = matches.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["reg_home_goals"] = pd.to_numeric(
        data["reg_home_goals"], errors="raise"
    )
    data["reg_away_goals"] = pd.to_numeric(
        data["reg_away_goals"], errors="raise"
    )

    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing required match values")

    if (
        (data["reg_home_goals"] < 0).any()
        or (data["reg_away_goals"] < 0).any()
    ):
        raise ValueError("Negative goals")

    if (data["home_team"] == data["away_team"]).any():
        raise ValueError("A team cannot play against itself")

    data = data.sort_values(
        ["date", "league", "home_team", "away_team"],
        kind="stable",
    ).reset_index(drop=True)

    state = FeatureState()
    v3 = TimeWeightedModel()

    feature_rows = []
    target_rows = []
    output = []

    draw_model = None
    direction_model = None
    current_season = None

    for date, day in data.groupby("date", sort=True):
        season = season_key(date)

        # Träna bara på matcher från tidigare avslutade säsonger.
        if season != current_season:
            current_season = season
            draw_model = None
            direction_model = None

            if len(feature_rows) >= 500:
                draw_model, direction_model = train_models(
                    feature_rows,
                    target_rows,
                )

        day_records = []

        # Alla matcher samma datum prognostiseras innan resultaten
        # från någon av dagens matcher uppdaterar modellen.
        for row in day.itertuples(index=False):
            league = str(row.league)

            v3.prepare_date(league, date)

            features = state.make_features(row, date)

            base_probs = np.asarray(
                v3.predict(
                    league,
                    str(row.home_team),
                    str(row.away_team),
                ),
                dtype=float,
            )

            base_draw = float(base_probs[1])
            base_home_conditional = float(
                base_probs[0] / max(1e-12, base_probs[0] + base_probs[2])
            )

            # Kryssmodellens balanserade sannolikhet korrigeras
            # för den ursprungliga andelen kryss i träningsdatan.
            if draw_model is not None:
                weighted_draw = positive_probability(
                    draw_model, features, positive_class=1
                )

                draw_count = sum(target_rows)
                non_draw_count = len(target_rows) - draw_count
                correction_weight = (
                    non_draw_count / max(1, draw_count)
                )

                corrected_draw = weighted_draw / (
                    correction_weight * (1.0 - weighted_draw)
                    + weighted_draw
                )

                p_draw = (
                    0.65 * corrected_draw
                    + 0.35 * base_draw
                )
            else:
                corrected_draw = base_draw
                p_draw = base_draw

            p_draw = float(np.clip(p_draw, 0.02, 0.55))

            # Riktning: sannolikheten för hemmaseger givet att
            # matchen inte slutar oavgjort.
            if direction_model is not None:
                ml_home_conditional = positive_probability(
                    direction_model, features, positive_class=1
                )

                p_home_conditional = (
                    0.65 * ml_home_conditional
                    + 0.35 * base_home_conditional
                )
            else:
                ml_home_conditional = base_home_conditional
                p_home_conditional = base_home_conditional

            p_home_conditional = float(
                np.clip(p_home_conditional, 0.02, 0.98)
            )

            p1 = (1.0 - p_draw) * p_home_conditional
            px = p_draw
            p2 = (1.0 - p_draw) * (1.0 - p_home_conditional)

            probs = np.asarray([p1, px, p2], dtype=float)
            probs = probs / probs.sum()

            hg = int(row.reg_home_goals)
            ag = int(row.reg_away_goals)

            actual_index = 0 if hg > ag else 1 if hg == ag else 2
            actual = ("1", "X", "2")[actual_index]
            one_hot = np.eye(3)[actual_index]

            day_records.append({
                "features": features,
                "target": actual_index,
                "row": row,
                "output": {
                    "date": date,
                    "league": league,
                    "home_team": str(row.home_team),
                    "away_team": str(row.away_team),
                    "p1": float(probs[0]),
                    "px": float(probs[1]),
                    "p2": float(probs[2]),
                    "p_draw_model": float(corrected_draw),
                    "p_home_given_no_draw": float(p_home_conditional),
                    "actual": actual,
                    "predicted": ("1", "X", "2")[
                        int(np.argmax(probs))
                    ],
                    "correct": int(np.argmax(probs) == actual_index),
                    "brier": float(np.mean((probs - one_hot) ** 2)),
                    "log_loss": float(
                        -math.log(max(1e-15, probs[actual_index]))
                    ),
                },
            })

        # Uppdatera historiken först när dagens prognoser är klara.
        for record in day_records:
            row = record["row"]

            state.update(row, date)

            v3.update(
                str(row.league),
                str(row.home_team),
                str(row.away_team),
                int(row.reg_home_goals),
                int(row.reg_away_goals),
            )

            feature_rows.append(record["features"])
            target_rows.append(record["target"])
            output.append(record["output"])

    return pd.DataFrame(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="v5_predictions.csv")
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

        print("\nVerklig utfallsfördelning:")
        print(results["actual"].value_counts(normalize=True)
              .reindex(["1", "X", "2"]).round(4))

        print("\nGenomsnittliga sannolikheter:")
        print(results[["p1", "px", "p2"]].mean().round(4))

        print("\nGenomsnittlig krysssannolikhet:")
        print(round(results["p_draw_model"].mean(), 4))


if __name__ == "__main__":
    main()
