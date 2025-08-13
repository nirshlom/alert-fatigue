from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for preprocessing and (future) training.

    This config focuses on preprocessing and data splitting to produce
    train/eval/test datasets and a training profile report.
    """

    input_csv_path: str
    date_column: str
    target_column: str
    feature_columns: List[str]

    # Predictive split fractions (must sum to 1.0)
    train_frac: float = 0.7
    eval_frac: float = 0.15
    test_frac: float = 0.15
    ascending: bool = True
    stratify: bool = False

    # Reproducibility
    random_seed: int = 42

    # Preprocessing flags
    impute_numeric: bool = True
    scale_numeric: bool = False
    rare_category_threshold: float = 0.01

    # Optional explicit dtype hints
    numeric_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None

    # Outputs
    output_dir: str = "model_pipeline/outputs"

    # Internals or advanced options
    profile_report_filename: str = "train_profile_report.html"
    generate_profile: bool = False
    
    # Model options
    use_glm: bool = True

    def validate(self) -> None:
        total = self.train_frac + self.eval_frac + self.test_frac
        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                f"train/eval/test fractions must sum to 1.0; got {total:.6f}"
            )
        if not (0 < self.train_frac < 1 and 0 < self.eval_frac < 1 and 0 < self.test_frac < 1):
            raise ValueError("All split fractions must be in (0, 1)")
        if self.rare_category_threshold < 0 or self.rare_category_threshold > 1:
            raise ValueError("rare_category_threshold must be in [0, 1]")


