
"""Rättvis jämförelse mellan hockeymodell V2 och V3."""

import argparse
import pandas as pd

from src.hockey_backtest import backtest as backtest_v2
from src.hockey_backtest_v3 import backtest as backtest_v3


SPLITS = {
    "TRAIN": ("2005-01-01", "2018-06-30"),
    "VALIDATION": ("2018-07-01", "2022-06-30"),
    "TEST": ("2022-07-01", "2099-12-31"),
}

OUTCOMES = ["1", "X", "2"]


def evaluate(results, name):
    if results.empty:
        print(f"\n{name}: Inga matcher")
        return

    predictions = results[["p1", "px", "p2"]].idxmax(axis=1)
    predicted = predictions.map({"p1": "1", "px": "X", "p2": "2"})
    actual = results["actual"]

    print(f"\n{name}")
    print("-" * 45)
    print(f"Matcher: {len(results)}")
    print(f"Träffsäkerhet: {results['correct'].mean():.4%}")
    print(f"Brier score: {results['brier'].mean():.5f}")
    print(f"Log loss: {results['log_loss'].mean():.5f}")

    print("\nValda utfall:")
    print(predicted.value_counts().reindex(OUTCOMES, fill_value=0))

    print("\nVerkliga utfall:")
    print(actual.value_counts().reindex(OUTCOMES, fill_value=0))

    print("\nTräffsäkerhet per utfall (recall):")
    for outcome in OUTCOMES:
        mask = actual == outcome
        if mask.sum():
            score = (predicted[mask] == outcome).mean()
            print(f"{outcome}: {score:.2%}")

    print("\nGenomsnittliga sannolikheter:")
    print(results[["p1", "px", "p2"]].mean().round(4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    matches = pd.read_csv(args.input)

    print("Kör V2...")
    v2 = backtest_v2(matches)

    print("Kör V3...")
    v3 = backtest_v3(matches)

    keys = ["date", "league", "home_team", "away_team"]

    v2 = v2.sort_values(keys).reset_index(drop=True)
    v3 = v3.sort_values(keys).reset_index(drop=True)

    if len(v2) != len(v3):
        raise ValueError("Modellerna har olika antal matcher.")

    if not v2[keys].equals(v3[keys]):
        raise ValueError("Modellerna har inte samma matchunderlag.")

    if not v2["actual"].equals(v3["actual"]):
        raise ValueError("Modellerna har olika matchutfall.")

    print(f"\nGemensamt matchunderlag: {len(v2)} matcher")

    for name, (start, end) in SPLITS.items():
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        mask = (v2["date"] >= start) & (v2["date"] <= end)

        print("\n" + "=" * 55)
        print(f"PERIOD: {name}")
        print("=" * 55)

        evaluate(v2.loc[mask].copy(), "V2")
        evaluate(v3.loc[mask].copy(), "V3")

    print("\nJÄMFÖRELSE KLAR")


if __name__ == "__main__":
    main()
