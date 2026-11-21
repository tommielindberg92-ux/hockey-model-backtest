"""Convert period scores into verified 60-minute results."""
import argparse
from pathlib import Path

import pandas as pd

REQUIRED = {
    "date", "league", "home_team", "away_team",
    "p1_home", "p1_away", "p2_home", "p2_away",
    "p3_home", "p3_away", "decided_in", "source",
}


def convert_period_results(raw):
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = raw.copy()

    for col in ["date", "league", "home_team", "away_team",
                "decided_in", "source"]:
        data[col] = data[col].astype("string").str.strip()

    data["date"] = pd.to_datetime(data["date"], errors="raise")

    if data[list(REQUIRED)].isna().any().any():
        raise ValueError("Missing required values")

    data["decided_in"] = data["decided_in"].str.upper()

    if not set(data["decided_in"]).issubset({"REG", "OT", "SO"}):
        raise ValueError("decided_in must be REG, OT or SO")

    score_cols = [
        "p1_home", "p1_away", "p2_home",
        "p2_away", "p3_home", "p3_away"
    ]

    for col in score_cols:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if (data[col] < 0).any() or (data[col] % 1 != 0).any():
            raise ValueError(f"Invalid goals in {col}")

    keys = ["date", "league", "home_team", "away_team"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate matches found")

    result = pd.DataFrame({
        "date": data["date"],
        "league": data["league"],
        "home_team": data["home_team"],
        "away_team": data["away_team"],
        "reg_home_goals": data[
            ["p1_home", "p2_home", "p3_home"]
        ].sum(axis=1).astype(int),
        "reg_away_goals": data[
            ["p1_away", "p2_away", "p3_away"]
        ].sum(axis=1).astype(int),
        "decided_in": data["decided_in"],
        "source": data["source"],
    })

    return result.sort_values("date").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/matches.csv")
    args = parser.parse_args()

    converted = convert_period_results(pd.read_csv(args.input))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(output, index=False)

    print(f"Imported {len(converted)} matches into {output}")


if __name__ == "__main__":
    main()
