from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    """Fit/transform preprocessing with optional numeric impute/scale and
    categorical rare-level bucketing. Fits on Train only; applies to Eval/Test.
    """

    numeric_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    impute_numeric: bool = True
    scale_numeric: bool = False
    rare_category_threshold: float = 0.01

    # Fitted attributes
    numeric_medians_: Dict[str, float] = field(default_factory=dict)
    numeric_means_: Dict[str, float] = field(default_factory=dict)
    numeric_stds_: Dict[str, float] = field(default_factory=dict)
    categorical_levels_: Dict[str, List[str]] = field(default_factory=dict)

    def _infer_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        if self.numeric_columns is not None and self.categorical_columns is not None:
            return self.numeric_columns, self.categorical_columns
        # Infer from dtypes
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        return numeric_cols, categorical_cols

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        numeric_cols, categorical_cols = self._infer_column_types(df)

        # Numeric statistics
        if self.impute_numeric:
            self.numeric_medians_ = {c: float(df[c].median()) for c in numeric_cols}
        if self.scale_numeric:
            self.numeric_means_ = {c: float(df[c].mean()) for c in numeric_cols}
            self.numeric_stds_ = {
                c: float(df[c].std(ddof=0)) if float(df[c].std(ddof=0)) != 0 else 1.0
                for c in numeric_cols
            }

        # Categorical: determine frequent levels and freeze
        self.categorical_levels_ = {}
        if categorical_cols:
            n = len(df)
            for c in categorical_cols:
                value_counts = df[c].astype("string").fillna("__MISSING__").value_counts(dropna=False)
                keep_levels = value_counts[value_counts / n >= self.rare_category_threshold].index.tolist()
                if "__MISSING__" not in keep_levels:
                    keep_levels.append("__MISSING__")
                self.categorical_levels_[c] = keep_levels

        # Persist the decision about which columns are considered numeric/categorical
        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.numeric_columns is None or self.categorical_columns is None:
            raise RuntimeError("Preprocessor must be fit before transform")

        out = df.copy()

        # Numeric
        for c in self.numeric_columns:
            if self.impute_numeric and c in self.numeric_medians_:
                out[c] = out[c].fillna(self.numeric_medians_[c])
            if self.scale_numeric and c in self.numeric_means_ and c in self.numeric_stds_:
                out[c] = (out[c] - self.numeric_means_[c]) / (self.numeric_stds_[c] or 1.0)

        # Categorical
        for c in self.categorical_columns:
            if c not in self.categorical_levels_:
                # If no levels were recorded (e.g., no categorical cols), skip
                continue
            levels = list(self.categorical_levels_[c])
            
            # Ensure "Other" is in the levels for handling unseen values
            if "Other" not in levels:
                levels.append("Other")
            
            # Transform the values
            out[c] = (
                out[c]
                .astype("string")
                .fillna("__MISSING__")
                .apply(lambda v: v if v in self.categorical_levels_[c] else "Other")
            )
            
            # Cast to category with the extended levels (including "Other")
            out[c] = out[c].astype(pd.CategoricalDtype(categories=levels))

        return out


