
"""Kronologisk utvärdering av hockeymodellen."""

import argparse
import pandas as pd

from src.hockey_backtest import backtest


SPLITS = {
    "TRAIN": ("2005-01-01", "2018-06-30"),
    "VALIDATION": ("2018-07-01", "2022-06-30"),
    "TEST": ("2022-07-01", "2099-12-31"),
}


def evaluate_period(results, name, start, end):
    period = results[
        (results["date"] >= start)
        & (results["date"] <= end)
    ]

    if period.empty:
        print(f"\n{name}: INGA MATCHER")
        return

    print(f"\n--- {name} ---")
    print(f"Matcher: {len(period)}")
    print(f"Träffsäkerhet: {period['correct'].mean():.4f}")
    print(f"Brier score: {period['brier'].mean():.4f}")
    print(f"Log loss: {period['log_loss'].mean():.4f}")

    print("Modellens val:")
    print(
        period[["p1", "px", "p2"]]
        .idxmax(axis=1)
        .value_counts()
        .reindex(["p1", "px", "p2"], fill_value=0)
        .rename(index={"p1": "1", "px": "X", "p2": "2"})
        .to_string()
    )

    print("Genomsnittliga sannolikheter:")
    print(period[["p1", "px", "p2"]].mean().round(3).to_string())

    print("Verkliga utfall:")
    print(
        period["actual"]
        .value_counts(normalize=True)
        .reindex(["1", "X", "2"], fill_value=0)
        .round(3)
        .to_string()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    matches = pd.read_csv(args.input)

    print("Kör kronologiskt backtest...")
    results = backtest(matches)
    results["date"] = pd.to_datetime(results["date"])

    print(f"Totalt antal matcher: {len(results)}")

    for name, (start, end) in SPLITS.items():
        evaluate_period(
            results,
            name,
            pd.Timestamp(start),
            pd.Timestamp(end),
        )


if __name__ == "__main__":
    main()
