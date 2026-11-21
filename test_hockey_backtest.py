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
