# Spaceship Titanic - Boosted Tree Improvements

`improved_bdt_pipeline.py` now runs **without external ML packages** and produces `submission_improved.csv` directly in this environment.

## What changed

- Uses a pure-Python boosted decision stump model (gradient boosting style) so no `numpy/pandas/sklearn/xgboost` install is required.
- Keeps useful structured feature engineering from the notebook (`Group_num`, cabin splits, spending aggregates, surname).
- Uses smoothed surname target encoding and threshold tuning before converting probabilities to labels.

## Run

```bash
python improved_bdt_pipeline.py
```

Outputs:

- `submission_improved.csv`
