# Project Part II: Predicting Housing Prices

The implementation predicts natural-log sale prices. The improvement plan is
implemented in `evaluate.py`: three baselines, random / property-grouped / purged
time validation, assessor-estimate ablation, quality checks and residual analysis.

## Files

- `proj.py`: feature engineering, model pipeline, validation, and export helpers.
- `evaluate.py`: grouped/time validation, baselines, ablation and diagnostics.
- `train_part2.py`: end-to-end validation, full-data training, and prediction export.
- `test_pipeline.py`: six regression and data-contract tests.
- `artifacts/`: compact reviewed metrics and charts; fitted models and prediction files are intentionally excluded.
- `IMPROVEMENT_REPORT.md`: results, limitations and reproducibility notes.

## Run

From this directory, run:

```powershell
pip install -r requirements.txt
python train_part2.py
```

Place the training CSV at
`cook_county_contest_data/cook_county_contest_train.csv` and the test CSV at
`cook_county_contest_test.csv`. Dataset archives, fitted models and predictions
are excluded from Git because this repository is intended to publish code and
reviewed aggregate evidence, not redistribute course data.

The script filters invalid sales below $500, reports holdout RMSE, trains a
fresh model on every valid training row, and writes the two submission files.

## Reproduce the improvement experiments

Use Python 3.12 and the pinned dependencies above. From this directory:

```powershell
python -m unittest -v test_pipeline
python evaluate.py --seeds 42 43 44 --output artifacts
```

For a final model that does not require assessor estimates:

```powershell
python evaluate.py --without-estimates --output artifacts-no-estimates
```

`--train`, `--test`, `--splits`, `--importance-rows` and `--output` are configurable.
The default experiment produces 42 model evaluations. Split membership is saved
as row-position arrays, linked to input hashes in `run_metadata.json`.
`artifacts` contains compact raw/summary metrics, grouped residuals, permutation
importance, data quality, split overlap checks and a comparison chart. Large
models, predictions, split arrays and logs are reproducible local outputs and are
ignored by Git.

## Validation results and boundaries

PIN-grouped HistGB RMSE across seeds 42/43/44 is $67,116 (sample SD $947),
versus $73,590 for Ridge and $163,923 for the median baseline. Removing estimates
raises grouped HistGB RMSE to $68,366. This is an ablation effect, not proof that
the estimates are free of leakage. See `IMPROVEMENT_REPORT.md` for the full results.

The latest-year holdout (2019), with overlapping PINs purged from earlier years,
gives HistGB RMSE $90,923 versus Ridge $80,313. The exported HistGB is a course
dataset model, not the recommended future-year deployment model.

`Most Recent Sale` is a retrospective flag and may be unavailable at prediction
time; assessor values also need an as-of availability audit. Time validation does
not establish that every feature was known before a sale. No online deployment
readiness or fairness claim is made. Ordinal codes in the existing HistGB preserve
the original baseline but introduce an arbitrary ordering for nominal categories.
Ridge uses one-hot categories and scaled numeric features with the same raw feature
set. Three fixed holdouts are stability checks, not independent confidence intervals.

The downloaded 55,311-row test CSV comes from the public Berkeley DS-100/fa22
course repository. Its identity with the local ECE course's grading test is
**unverified**; row count alone does not establish a match. Its labels are absent,
so no official test RMSE or rank can be reported. The previous statement that these
were verified official ECE contest predictions was too strong.

`proj.py` in this directory is the maintained implementation.
