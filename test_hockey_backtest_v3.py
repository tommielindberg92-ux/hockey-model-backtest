import pandas as pd
import pytest

from src.hockey_backtest_v3 import (
    LeagueState,
    regulation_probabilities,
    backtest,
)


def test_probabilities_sum_to_one():
    probs = regulation_probabilities(2.8, 2.8)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] == pytest.approx(probs[2])


def test_home_advantage():
    p1, px, p2 = regulation_probabilities(3.2, 2.4)
    assert p1 > p2


def test_decay_reduces_old_team_strength():
    state = LeagueState(
        attack={"Team A": 0.8},
        defence={"Team B": 0.6},
        last_date=pd.Timestamp("2025-01-01"),
    )

    state.decay(pd.Timestamp("2026-01-01"), 365)

    assert state.attack["Team A"] == pytest.approx(0.4, abs=0.002)
    assert state.defence["Team B"] == pytest.approx(0.3, abs=0.002)


def test_backtest_returns_valid_probabilities():
    matches = pd.DataFrame([
        {
            "date": "2026-01-01",
            "league": "SHL",
            "home_team": "A",
            "away_team": "B",
            "reg_home_goals": 3,
            "reg_away_goals": 1,
        },
        {
            "date": "2026-01-02",
            "league": "SHL",
            "home_team": "B",
            "away_team": "A",
            "reg_home_goals": 2,
            "reg_away_goals": 2,
        },
    ])

    result = backtest(matches)

    assert len(result) == 2
    assert (
        (result[["p1", "px", "p2"]].sum(axis=1) - 1).abs() < 1e-9
    ).all()


def test_same_day_predictions_do_not_use_same_day_results():
    matches = pd.DataFrame([
        {
            "date": "2026-01-01",
            "league": "SHL",
            "home_team": "A",
            "away_team": "B",
            "reg_home_goals": 5,
            "reg_away_goals": 0,
        },
        {
            "date": "2026-01-01",
            "league": "SHL",
            "home_team": "A",
            "away_team": "B",
            "reg_home_goals": 0,
            "reg_away_goals": 5,
        },
    ])

    result = backtest(matches)

    assert result.iloc[0]["p1"] == pytest.approx(result.iloc[1]["p1"])
    assert result.iloc[0]["px"] == pytest.approx(result.iloc[1]["px"])
    assert result.iloc[0]["p2"] == pytest.approx(result.iloc[1]["p2"])