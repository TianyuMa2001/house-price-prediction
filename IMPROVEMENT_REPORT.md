# Housing Price Prediction Improvement Report

This work implements the validation, baselines, ablation, data-quality checks, error analysis, and reproducibility requirements in the project improvement plan. The previous random-split result was reproduced precisely. New experiments show relatively stable performance on unseen properties but weaker performance across years. The main improvement is measuring these limitations rather than optimizing a single score.

## Previous State and Changes

The original `proj.py` trained HistGradientBoosting and exported predictions, but used only one random 80/20 split. It did not check whether the same property appeared on both sides and lacked an input contract, a complete evaluation report, and automated tests.

| Improvement | Implementation | Purpose |
|---|---|---|
| PIN-grouped validation | `evaluate.split_indices(..., 'group')` | Keep all transactions for one property identifier (PIN) on the same side |
| Temporal validation | Hold out 2019 and purge its PINs from the 2013–2018 training set | Evaluate future-year, unseen properties without including their historical transactions |
| Three baselines | Median, Ridge, HistGB | Measure the value of complexity relative to simpler methods |
| Assessor-estimate ablation | `include_estimates=False` | Exclude both raw and derived estimate features; prediction does not require estimate columns |
| Quality reporting | `data_quality.json`, `splits.json` | Check missingness, duplicate rows/PINs, invalid prices, unseen categories, and overlap |
| Error analysis | `grouped_errors.csv`, two importance CSVs | Diagnose errors by price band, property-class code, and town code |
| Engineering | Schema, CLI, logs, hashes, pinned versions, six tests | Fail early on invalid inputs and support traceable experiments and model reloading |

## Data and Split Evidence

The training set contains 138,217 rows, 62 columns, and 120,648 distinct PINs, including 17,569 repeated-PIN rows and seven fully duplicated rows. Every column in this already-filtered course dataset has zero missingness, and there are no nonfinite prices or prices below $500. This does not imply deployment data will have the same quality. Duplicates were retained to reproduce the original baseline; repeated PINs may represent legitimate repeat transactions.

| Split | Training rows | Validation rows | PIN overlap | Notes |
|---|---:|---:|---:|---|
| Random, seed 42 | 110,573 | 27,644 | 5,382 | Row positions do not overlap, but properties do |
| PIN-grouped, seed 42 | 110,596 | 27,621 | 0 | Property-level splitting need not produce exactly 20% of rows |
| Temporal, 2019 holdout | 113,898 | 20,199 | 0 | Purges 4,120 earlier transactions sharing holdout PINs |

Random and grouped experiments use seeds 42, 43, and 44. The temporal boundary is fixed and is evaluated once rather than presenting repeated identical splits as independent validation. Every split checks disjoint row positions. Only the random split retains PIN overlap as a comparison with the original method. Split positions and input SHA-256 hashes were saved during the full local evaluation; large position-array files are excluded from this GitHub repository.

## Baselines and Ablation

Random and grouped results below are means across three seeds; temporal results use one fixed-year experiment. RMSE and MAE are measured in US dollars. All models predict natural-log sale prices. Ridge uses one-hot encoding and numeric standardization; HistGB retains the original ordinal category encoding. The raw feature scope is consistent.

| Validation | Model | Estimates | RMSE | MAE | log-RMSE |
|---|---|---|---:|---:|---:|
| Random | Median | With/without | 164,301 | 123,110 | 0.7866 |
| Random | Ridge | With | 74,103 | 51,100 | 0.3611 |
| Random | HistGB | With | 67,208 | 47,137 | 0.3469 |
| Random | HistGB | Without | 68,630 | 47,819 | 0.3485 |
| PIN-grouped | Median | With/without | 163,923 | 122,518 | 0.7801 |
| PIN-grouped | Ridge | With | 73,590 | 50,847 | 0.3577 |
| PIN-grouped | Ridge | Without | 73,625 | 50,912 | 0.3576 |
| PIN-grouped | HistGB | With | 67,116 | 47,029 | 0.3438 |
| PIN-grouped | HistGB | Without | 68,366 | 47,698 | 0.3456 |
| Temporal | Median | With/without | 167,151 | 122,900 | 0.7198 |
| Temporal | Ridge | With | 80,313 | 58,779 | 0.3985 |
| Temporal | Ridge | Without | 81,016 | 59,576 | 0.4015 |
| Temporal | HistGB | With | 90,923 | 70,862 | 0.4923 |
| Temporal | HistGB | Without | 101,220 | 78,109 | 0.5169 |

All 42 evaluations, including run times and sample counts, are in `artifacts/metrics.csv`. The original seed-42 HistGB result was reproduced: log-RMSE = 0.3461350280 and dollar RMSE = 67,211.1439.

Grouped HistGB has approximately 8.8% lower mean RMSE than Ridge. Its sample standard deviation across three grouped splits is approximately $947. Removing estimates increases RMSE by approximately $1,251 (1.86%). Other features therefore retain useful signal, but this does not establish that estimates are leakage-free.

Ridge outperforms HistGB in the temporal experiment. Unseen future-year categories, limited tree extrapolation, and market distribution changes are plausible explanations, not established causal findings. HistGB remains the final course-dataset model. Future-year deployment would require model reselection and an audit of feature availability at prediction time.

![Model comparison](artifacts/model_comparison.png)

## Harder Samples

The following price-band results use grouped seed 42 and HistGB with estimates. Bands are determined by true labels for post-hoc diagnosis only; they are not fed back into training.

| Actual price | Samples | MAE | RMSE | log-RMSE |
|---|---:|---:|---:|---:|
| <100k | 5,133 | 28,511 | 42,558 | 0.5213 |
| 100k–250k | 11,793 | 39,258 | 52,081 | 0.3258 |
| 250k–500k | 8,375 | 55,997 | 74,510 | 0.2444 |
| 500k–1m | 2,320 | 102,573 | 132,490 | 0.2693 |

Higher-priced properties have larger dollar errors; lower-priced properties have larger relative errors. This validation set contains no properties above $1 million, so it does not support claims about luxury-property generalization. Town codes 73 (185 rows) and 10 (168 rows) have RMSE of approximately $118,352 and $113,808; property-class code 206 (719 rows) has RMSE of approximately $110,204. Codes are retained as source labels rather than assigning unverified geographic names. Different sample sizes and price distributions prevent treating these metrics as fairness conclusions.

Permutation importance samples 1,500 grouped-validation rows and repeats each shuffle three times, measuring the increase in log-RMSE. Leading features include building estimates, latitude, land estimates, `Most Recent Sale`, longitude, and sale year. A single-column shuffle measures fitted-model dependence. Correlated features can substitute for one another, so this is neither equivalent to retraining-based ablation nor a causal explanation.

## Limitations and Corrections

1. Assessor values need an independent as-of availability audit. A local dictionary describes prior-tax-year values, but that alone cannot guarantee every row is free of future information.
2. `Most Recent Sale` is retrospective, may depend on future transactions, and ranks highly in importance. The course model is not directly suitable for online prediction before a sale. The no-estimates ablation does not remove this flag.
3. Similar random and grouped scores do not imply repeated properties are always harmless. Three fixed splits provide stability evidence, not independent confidence intervals.
4. The 55,311 predictions use the public Berkeley DS-100/fa22 test dataset. Its identity with the local ECE course's official grading data remains unverified. Previously equating successful prediction export with verified official contest data was unsupported. Training and test share 12,325 PINs; test labels are absent, so no official test score or rank can be calculated.
5. Seven fully duplicated rows were retained for comparability. A future deduplication sensitivity analysis should be labeled as a separate experiment version.

## Reproduction and Outputs

Use Python 3.12:

```powershell
pip install -r requirements.txt
python -m unittest -v test_pipeline
python evaluate.py --seeds 42 43 44 --output artifacts
```

Supply the course CSVs at the paths described in the README, or use `--train` and `--test` to specify verified files. The final model fits all valid training rows and writes `artifacts/pipeline.joblib.gz` and `artifacts/predictions.csv`. Predictions remain log(Sale Price), with the course-format index column. `--without-estimates` makes the final model independent of estimate columns. `train_part2.py` remains a simpler training entry point.

Six automated tests cover PIN isolation, temporal purging, missing-column/nonfinite-value rejection, nonmutating feature engineering and fractional bathroom counts, unseen categories/missing values/no-estimates/serialization, and invalid targets and metrics. All tests pass. The GitHub test suite uses synthetic fixtures and does not require course data. Full local export checks validated prediction length, finite values, and identical predictions after reloading.

This repository publishes code and compact reviewed evidence. Dataset archives, fitted models, predictions, split arrays, and large logs are local reproducible outputs and are not uploaded.

An interview explanation can focus on the log-price target, grouped rather than random-only validation, Median and Ridge baselines, what ablation establishes, weaker future-year performance, and features unavailable online. The planned scope is complete; no web frontend, cloud deployment, or unsupported resume metrics were added.
