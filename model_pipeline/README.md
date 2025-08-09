## Model Training Pipeline – Design Document

### Objectives
- Build a simple, readable, modular pipeline to predict alert fatigue/medication errors.
- Baseline model: logistic regression via statsmodels (avoid manual one-hot encoding with patsy formulas).
- Strict separation of steps; extensible to CatBoost/XGBoost next.
- All code and artifacts live under `model_pipeline/`.

### Inputs and Assumptions
- Input: CSV from the data pipeline (e.g., `df_main_active_adult_renamed.csv` or `df_patients_level_data.csv`).
- Binary target column is provided by config (e.g., `target_column`).
- Feature list provided by config. Other columns ignored.
- Categorical features handled through statsmodels formula terms `C(col)` with frozen levels.
- Numeric imputation and scaling are optional and controlled via function inputs (see Preprocessing).

### High-Level Flow
1. Load data.
2. Predictive split into Train / Eval / Test by time order (percent allocations).
3. (Optional) Profile training set (HTML ProfileReport). Disabled by default.
4. Preprocess (fit on train; transform train/eval/test consistently; impute/scale optional).
5. Train logistic regression (statsmodels Logit or GLM Binomial).
6. Evaluate on eval set: predictions, PR curve, decile threshold metrics.
7. Report results: coefficients → odds ratios (OR) with CIs; forest plot; PR curve plot; threshold table.
8. Persist artifacts and a `run_summary.json` in a timestamped run folder.

### Predictive Split (Time-Based)
- Purpose: mimic real-world prediction by training on earlier data and evaluating on later periods.
- Inputs:
  - `date_column: str` – column with sortable datetime.
  - `train_frac: float`, `eval_frac: float`, `test_frac: float` – must sum to 1.0.
  - `ascending: bool` (default True) – earlier to later.
  - `stratify: bool` (default False) – optional label-aware temporal split; if True, we approximate stratification while preserving chronology (best-effort; will report drift if large).
- Behavior:
  - Sort by `date_column` (ascending by default).
  - Take first `train_frac` proportion as Train, next `eval_frac` as Eval, last `test_frac` as Test.
  - If `stratify=True`, perform label-aware allocation within contiguous time buckets to reduce class imbalance drift, while maintaining temporal order.
- Output: `train_df`, `eval_df`, `test_df` and a summary of class proportions across splits.

### Preprocessing (Optional Impute/Scale)
- Fit on Train only; apply to Eval/Test; persist parameters in-memory and to disk.
- Function inputs control behavior:
  - `impute_numeric: bool` (default True) – if True, impute numeric with Train medians.
  - `scale_numeric: bool` (default False) – if True, standardize numeric with Train mean/std.
  - `rare_category_threshold: float` (default 0.01) – bucket infrequent categories to "Other".
- Numeric:
  - Identify numeric columns (or accept explicit list).
  - If `impute_numeric=True`, fill missings with Train medians.
  - If `scale_numeric=True`, standardize using Train stats.
- Categorical:
  - Cast to `category`.
  - Freeze category levels from Train; unseen categories in Eval/Test map to "Other".
  - Pass frozen levels to patsy via `C(col, levels=[...])` to stabilize encoding across splits.

### Modeling (Statsmodels Logistic Regression)
- Wrapper builds a patsy formula from `feature_columns` and `target_column`:
  - Numeric features included directly.
  - Categorical features included as `C(col, levels=[...])` with Train-frozen levels.
- Fit `Logit` or `GLM(Binomial, logit)`; store fitted result.
- Predict probabilities on Eval/Test given preprocessed frames with the same schema.
- Extract coefficients, standard errors, and CIs for reporting; convert to ORs.

### Evaluation (on Eval Set)
- Compute predicted probabilities.
- Precision–Recall (PR) curve and AUC-PR.
- Threshold table at 10 percentiles (0.1–1.0): threshold, precision, recall, F1, specificity, accuracy, positive rate, TP/FP/TN/FN counts.
- Optional: pick a default threshold (0.5 or F1-max) for headline metrics.

### Reporting and Artifacts
- Training ProfileReport (HTML) via `ydata-profiling` (optional; enable via `generate_profile=True`).
- Coefficients → OR table (with 95% CI) saved as CSV.
- Forest plot of ORs (sorted by distance from 1.0) saved as PNG.
- PR curve plot saved as PNG.
- Threshold metrics table saved as CSV.
- Eval predictions CSV (ids optional if provided) for auditability.
- Timestamped run directory under `model_pipeline/outputs/{timestamp}/` containing:
  - `config_used.json`, `run_summary.json`
  - `train_profile_report.html`
  - `coefficients_or.csv`, `coefficients_forest.png`
  - `pr_curve.png`, `threshold_metrics.csv`, `eval_predictions.csv`

### Module Layout (all under `model_pipeline/`)
- `config.py` – dataclasses/defaults for configuration (paths, columns, seed, split, preprocessing flags).
- `data_loading.py` – CSV loading, dtype parsing, column validation.
- `split.py` – predictive time-based split; optional stratified-random split for comparison.
- `preprocess.py` – `Preprocessor` with optional numeric impute/scale, rare-category handling, schema freeze.
- `models/`
  - `base.py` – `BaseBinaryClassifier` interface.
  - `statsmodels_logit.py` – logistic regression wrapper.
- `evaluation/`
  - `metrics.py` – PR curve, AUC-PR, threshold table, summary metrics.
  - `plots.py` – PR curve and OR forest plots.
- `reporting/`
  - `profile.py` – training ProfileReport generation.
  - `coefficients.py` – OR table from fitted result.
  - `save.py` – run folder management and safe file writes.
- `pipeline.py` – `ModelTrainingPipeline` orchestrator.
- `run_training.py` – CLI entry point.

### Configuration (key fields)
```python
TrainingConfig(
  input_csv_path: str,
  date_column: str,                 # for predictive split
  target_column: str,
  feature_columns: list[str],
  train_frac: float = 0.7,
  eval_frac: float = 0.15,
  test_frac: float = 0.15,
  ascending: bool = True,
  stratify: bool = False,           # optional label-aware temporal split
  random_seed: int = 42,
  impute_numeric: bool = True,      # OPTIONAL via function inputs
  scale_numeric: bool = False,      # OPTIONAL via function inputs
  rare_category_threshold: float = 0.01,
  output_dir: str = 'model_pipeline/outputs',
  use_glm: bool = True,
)
```

### Minimal Public APIs (signatures)
- `split.predictive_time_split(df, date_column, train_frac, eval_frac, test_frac, ascending=True, stratify=False) -> (train_df, eval_df, test_df)`
- `preprocess.Preprocessor.fit(df, numeric_cols, categorical_cols, impute_numeric=True, scale_numeric=False, rare_category_threshold=0.01) -> None`
- `preprocess.Preprocessor.transform(df) -> pd.DataFrame`
- `models.statsmodels_logit.StatsmodelsLogitModel(formula).fit(df) -> None`
- `models.statsmodels_logit.StatsmodelsLogitModel.predict_proba(df) -> np.ndarray`
- `evaluation.metrics.compute_pr(y_true, y_score) -> dict`
- `evaluation.metrics.threshold_table(y_true, y_score, percentiles: list[float]) -> pd.DataFrame`
- `reporting.profile.generate_profile_report(df, output_path) -> str`
- `reporting.coefficients.coefficients_to_or(result) -> pd.DataFrame`
- `evaluation.plots.plot_or_forest(or_df, output_path) -> str`
- `evaluation.plots.plot_pr_curve(precision, recall, output_path) -> str`

### Extensibility & Simplicity
- Keep modules small and names explicit; minimize abstraction.
- Future models (CatBoost/XGBoost) can plug into `BaseBinaryClassifier` with the same preprocessing and evaluation.
- Preprocessing flags allow skipping imputation/scaling for tree-based models.

### Notes for Future Improvements
- Robust temporal stratification: experiment with target-conditional time binning to better balance labels without breaking chronology.
- Calibration curves and Brier score for probability quality.
- Group-aware splits (e.g., by patient or unit) to avoid leakage.
- Automatically detect and warn about label drift across splits.


