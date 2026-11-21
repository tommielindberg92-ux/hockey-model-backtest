Historical match data

This folder stores verified historical hockey match results for model development and chronological backtesting.

Supported leagues

* NHL
* SHL
* HockeyAllsvenskan

CSV format

The main dataset will be named matches.csv.

Required columns:

Column	Description
date	Match date and time
league	NHL, SHL or HockeyAllsvenskan
home_team	Home team
away_team	Away team
reg_home_goals	Home goals after 60 minutes
reg_away_goals	Away goals after 60 minutes

Data quality rules

1. Only verified official match results may be imported.
2. Regulation-time goals must be separated from overtime and shootout results.
3. Postponed, cancelled and incomplete matches must be excluded.
4. Duplicate matches must be detected and removed.
5. Team names must be standardized across seasons.
6. Every imported match must retain its original source reference.
7. Historical data must be sorted chronologically before backtesting.

Sources

Prefer official league APIs and official match reports.

* NHL: https://api-web.nhle.com/
* SHL: https://www.shl.se/

HockeyAllsvenskan source must be verified before importing historical data.

Important

Never estimate or invent missing regulation-time scores. Unverified matches must be excluded until they can be checked.
