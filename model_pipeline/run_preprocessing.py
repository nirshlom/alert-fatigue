from __future__ import annotations

import argparse
import json
from typing import List

from .config import TrainingConfig
from .pipeline import run_preprocessing_only


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run preprocessing and predictive split")
    p.add_argument("--input-csv", required=True)
    p.add_argument("--date-column", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--features-json", required=True, help="JSON list of feature column names")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--eval-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--ascending", action="store_true", help="Sort time ascending (default)")
    p.add_argument("--descending", action="store_true", help="Sort time descending")
    p.add_argument("--stratify", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--impute-numeric", action="store_true")
    p.add_argument("--no-impute-numeric", action="store_true")
    p.add_argument("--scale-numeric", action="store_true")
    p.add_argument("--rare-cat-threshold", type=float, default=0.01)
    p.add_argument("--output-dir", default="model_pipeline/outputs")
    p.add_argument("--generate-profile", action="store_true", help="Generate ydata-profiling report on train set")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features: List[str] = json.loads(args.features_json)

    # Determine imputation flag precedence
    if args.no_impute_numeric:
        impute_numeric = False
    else:
        impute_numeric = args.impute_numeric

    cfg = TrainingConfig(
        input_csv_path=args.input_csv,
        date_column=args.date_column,
        target_column=args.target,
        feature_columns=features,
        train_frac=args.train_frac,
        eval_frac=args.eval_frac,
        test_frac=args.test_frac,
        ascending=not args.descending,
        stratify=args.stratify,
        random_seed=args.seed,
        impute_numeric=impute_numeric,
        scale_numeric=args.scale_numeric,
        rare_category_threshold=args.rare_cat_threshold,
        output_dir=args.output_dir,
        generate_profile=args.generate_profile,
    )

    train_df, eval_df, test_df, artifacts = run_preprocessing_only(cfg)

    print(f"Preprocessing completed. Rows -> train: {len(train_df)}, eval: {len(eval_df)}, test: {len(test_df)}")
    print(f"Artifacts: {artifacts}")


if __name__ == "__main__":
    main()


