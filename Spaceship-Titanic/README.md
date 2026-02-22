# Spaceship Titanic - Boosted Tree Improvements

`improved_bdt_pipeline.py` runs **without external ML packages** and produces `submission_improved.csv`.

## Feature and imputation behavior

- New engineered features include:
  - `SpentMoney` (alias of total spend),
  - `GroupSize` from `PassengerId` group counts,
  - `SpendPerPerson` (`TotalSpent / GroupSize`),
  - plus `TotalSpent`, `ZeroSpend`, `LogTotalSpent`, cabin/group/surname splits.
- Missing-data handling is rule/median based (not model-based):
  - spend columns default to `0`,
  - `CryoSleep` missing values are inferred from spending (`True` when total spend is `0`, else `False`),
  - age uses HomePlanet median fallback,
  - remaining numeric gaps are filled with global medians.

## Run

```bash
python improved_bdt_pipeline.py
```

Outputs:

- `submission_improved.csv`
