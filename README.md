# Hockey model backtest

A chronological backtesting scaffold for regulation-time (60-minute) hockey 1X2 predictions.

## Scope
- Separate Elo ratings and goal averages by league.
- Predict probabilities for home win (1), draw (X), away win (2).
- Update team ratings only after each match has been predicted.
- Evaluate Brier score, log loss and accuracy.
- Never mix regulation results with overtime/shootout match-winner results.

## Data format
Create a CSV with these columns:

| Column | Meaning |
|---|---|
| `date` | Match date/time; sortable ISO format recommended |
| `league` | NHL, SHL or HockeyAllsvenskan |
| `home_team` | Home team |
| `away_team` | Away team |
| `reg_home_goals` | Home goals after 60 minutes |
| `reg_away_goals` | Away goals after 60 minutes |

Only use information available before each match as model inputs. The final regulation score is used only as the outcome and for updating the model after prediction.

## Run
```bash
python -m pip install -r requirements.txt
python -m src.hockey_backtest --input data/matches.csv
```

## Important
This is a baseline scaffold, not a validated profitable betting model. Historical odds must be separately sourced and verified for the exact 60-minute 1X2 market before calculating ROI or CLV.
