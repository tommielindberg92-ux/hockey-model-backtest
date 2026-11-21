import pandas as pd
import pytest

from src.hockey_backtest import backtest, regulation_probabilities


def test_probabilities_sum_to_one():
    probs = regulation_probabilities(2.8, 2.8)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] == pytest.approx(probs[2])


def test_home_advantage_creates_home_favouring_prediction():
    p1, px, p2 = regulation_probabilities(3.2, 2.4)
    assert p1 > p2


def test_backtest_returns_one_row_per_match_and_valid_probabilities():
    matches = pd.DataFrame([
        {"date": "2026-01-01", "league": "SHL", "home_team": "A",
         "away_team": "B", "reg_home_goals": 3, "reg_away_goals": 1},
        {"date": "2026-01-02", "league": "SHL", "home_team": "B",
         "away_team": "A", "reg_home_goals": 2, "reg_away_goals": 2},
    ])
    result = backtest(matches)
    assert len(result) == 2
    assert ((result[["p1", "px", "p2"]].sum(axis=1) - 1).abs() < 1e-9).all()


def test_missing_columns_raise_error():
    with pytest.raises(ValueError):
        backtest(pd.DataFrame([{"date": "2026-01-01"}]))

from src.import_historical import convert_period_results


def sample_period_match():
    return {
        "date": "2026-01-01",
        "league": "SHL",
        "home_team": "Team A",
        "away_team": "Team B",
        "p1_home": 1, "p1_away": 0,
        "p2_home": 0, "p2_away": 1,
        "p3_home": 1, "p3_away": 1,
        "decided_in": "OT",
        "source": "official match report",
        "ot_home": 1, "ot_away": 0,
        "shootout_home": 1, "shootout_away": 0,
    }


def test_importer_sums_only_three_regulation_periods():
    result = convert_period_results(pd.DataFrame([sample_period_match()]))
    match = result.iloc[0]
    assert match["reg_home_goals"] == 2
    assert match["reg_away_goals"] == 2
    assert match["decided_in"] == "OT"


def test_importer_rejects_duplicate_matches():
    row = sample_period_match()
    with pytest.raises(ValueError, match="Duplicate"):
        convert_period_results(pd.DataFrame([row, row]))


def test_importer_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing columns"):
        convert_period_results(pd.DataFrame([{"date": "2026-01-01"}]))


def test_importer_rejects_negative_period_goals():
    row = sample_period_match()
    row["p2_home"] = -1
    with pytest.raises(ValueError, match="Invalid goals"):
        convert_period_results(pd.DataFrame([row]))
from src.import_nhl import get_regulation_result


def test_nhl_regulation_result_excludes_overtime(monkeypatch):
    def fake_get_json(url):
        return {
            "plays": [
                {
                    "typeDescKey": "goal",
                    "periodDescriptor": {"number": 1},
                    "details": {"eventOwnerTeamId": 1},
                },
                {
                    "typeDescKey": "goal",
                    "periodDescriptor": {"number": 2},
                    "details": {"eventOwnerTeamId": 2},
                },
                {
                    "typeDescKey": "goal",
                    "periodDescriptor": {"number": 4, "periodType": "OT"},
                    "details": {"eventOwnerTeamId": 1},
                },
            ]
        }

    monkeypatch.setattr("src.import_nhl.get_json", fake_get_json)

    home, away, decided = get_regulation_result(123, 1, 2)

    assert home == 1
    assert away == 1
    assert decided == "OT"


def test_nhl_regulation_result_counts_three_periods():
    def fake_get_json(url):
        return {
            "plays": [
                {
                    "typeDescKey": "goal",
                    "periodDescriptor": {"number": 1},
                    "details": {"eventOwnerTeamId": 1},
                },
                {
                    "typeDescKey": "goal",
                    "periodDescriptor": {"number": 2},
                    "details": {"eventOwnerTeamId": 1},
                },
                {
                    "typeDescKey": "goal",
                    "periodDescriptor": {"number": 3},
                    "details": {"eventOwnerTeamId": 2},
                },
            ]
        }

    import src.import_nhl
    original = src.import_nhl.get_json
    src.import_nhl.get_json = fake_get_json
    try:
        home, away, decided = get_regulation_result(123, 1, 2)
    finally:
        src.import_nhl.get_json = original

    assert home == 2
    assert away == 1
    assert decided == "REG"