from __future__ import annotations

from typing import Dict, Tuple

import os
import pandas as pd

from .config import TrainingConfig
from .data_loading import load_csv, select_columns
from .split import predictive_time_split
from .preprocess import Preprocessor
from .reporting.profile import generate_profile_report
from .reporting.save import ensure_dir, make_run_dir, write_json


def run_preprocessing_only(config: TrainingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Run data loading, predictive split, preprocessing, and training profile.

    Returns (train_df, eval_df, test_df, artifacts_paths).
    """
    config.validate()

    # Prepare run directory
    ensure_dir(config.output_dir)
    run_dir = make_run_dir(config.output_dir)

    # Load data
    df = load_csv(config.input_csv_path, parse_dates=[config.date_column])

    # Keep only required columns: date, target, features
    required_cols = list({config.date_column, config.target_column, *config.feature_columns})
    df = select_columns(df, required_cols)

    # Predictive time-based split
    train_raw, eval_raw, test_raw = predictive_time_split(
        df=df,
        date_column=config.date_column,
        train_frac=config.train_frac,
        eval_frac=config.eval_frac,
        test_frac=config.test_frac,
        ascending=config.ascending,
        stratify=config.stratify,
        target_column=config.target_column,
    )

    # Preprocess: fit on train, transform all
    preprocessor = Preprocessor(
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
        impute_numeric=config.impute_numeric,
        scale_numeric=config.scale_numeric,
        rare_category_threshold=config.rare_category_threshold,
    ).fit(train_raw)

    train_df = preprocessor.transform(train_raw)
    eval_df = preprocessor.transform(eval_raw)
    test_df = preprocessor.transform(test_raw)

    # Profile training set (optional)
    profile_output = None
    if config.generate_profile:
        profile_path = os.path.join(run_dir, config.profile_report_filename)
        profile_output = generate_profile_report(train_df, profile_path)

    # Save run summary
    summary = {
        "input_csv_path": config.input_csv_path,
        "date_column": config.date_column,
        "target_column": config.target_column,
        "feature_columns": config.feature_columns,
        "train_frac": config.train_frac,
        "eval_frac": config.eval_frac,
        "test_frac": config.test_frac,
        "ascending": config.ascending,
        "stratify": config.stratify,
        "impute_numeric": config.impute_numeric,
        "scale_numeric": config.scale_numeric,
        "rare_category_threshold": config.rare_category_threshold,
        "run_dir": run_dir,
        "profile_report": profile_output,
        "rows": {
            "train": len(train_df),
            "eval": len(eval_df),
            "test": len(test_df),
        },
    }
    summary_path = os.path.join(run_dir, "run_summary.json")
    write_json(summary, summary_path)

    artifacts = {
        "run_dir": run_dir,
        "profile_report": profile_output,
        "run_summary": summary_path,
    }

    return train_df, eval_df, test_df, artifacts


