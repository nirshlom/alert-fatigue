from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings


@dataclass
class OneHotPreprocessor:
    """Fit/transform preprocessing with one-hot encoding for categorical features.
    
    This preprocessor applies one-hot encoding to categorical features with more than 2 levels,
    while binary categoricals (2 levels) are converted to 0/1 numeric columns.
    Numeric features can optionally be imputed and scaled.
    
    Fits on Train only; applies to Eval/Test consistently.
    
    Args:
        handle_unseen_categories: If True, map unseen categories to "Other" during transform.
            If False (default), raise ValueError when encountering unseen categories.
            This ensures data quality by catching unexpected values.
    """

    numeric_columns: Optional[List[str]] = None
    categorical_columns: Optional[List[str]] = None
    impute_numeric: bool = True
    scale_numeric: bool = False
    rare_category_threshold: float = 0.01
    # Optional mapping: feature name -> desired reference category name (for binary conversion)
    categorical_reference_levels: Optional[Dict[str, str]] = None
    # If True, map unseen categories to "Other"; if False, raise error on unseen categories
    handle_unseen_categories: bool = False

    # Fitted attributes
    numeric_medians_: Dict[str, float] = field(default_factory=dict)
    numeric_means_: Dict[str, float] = field(default_factory=dict)
    numeric_stds_: Dict[str, float] = field(default_factory=dict)
    categorical_levels_: Dict[str, List[str]] = field(default_factory=dict)
    categorical_missing_seen_: Dict[str, bool] = field(default_factory=dict)
    # Track which categoricals are binary vs multi-level
    categorical_is_binary_: Dict[str, bool] = field(default_factory=dict)
    # Track binary reference level (the level that becomes 1)
    binary_reference_levels_: Dict[str, str] = field(default_factory=dict)
    # Mapping from original feature names to new column names after one-hot encoding
    feature_mapping_: Dict[str, List[str]] = field(default_factory=dict)

    def _infer_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Infer numeric and categorical columns from DataFrame."""
        if self.numeric_columns is not None and self.categorical_columns is not None:
            return self.numeric_columns, self.categorical_columns
        # Infer from dtypes
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        return numeric_cols, categorical_cols

    def fit(self, df: pd.DataFrame) -> "OneHotPreprocessor":
        """Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame with features
            
        Returns:
            self: Fitted preprocessor instance
        """
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
        self.categorical_missing_seen_ = {}
        self.categorical_is_binary_ = {}
        self.binary_reference_levels_ = {}
        self.feature_mapping_ = {}
        
        if categorical_cols:
            n = len(df)
            for c in categorical_cols:
                series = df[c].astype("string")
                missing_seen = series.isna().any()
                self.categorical_missing_seen_[c] = bool(missing_seen)

                # Count without injecting a '__MISSING__' level unless actually present
                if missing_seen:
                    value_counts = series.fillna("__MISSING__").value_counts(dropna=False)
                else:
                    value_counts = series.value_counts(dropna=False)

                # Apply rare category threshold
                keep_levels = value_counts[value_counts / n >= self.rare_category_threshold].index.tolist()

                # Ensure '__MISSING__' is kept only if it truly exists in training
                if missing_seen and "__MISSING__" not in keep_levels:
                    keep_levels.append("__MISSING__")

                # Only add "Other" if we want to handle unseen categories
                if self.handle_unseen_categories and "Other" not in keep_levels:
                    keep_levels.append("Other")

                self.categorical_levels_[c] = keep_levels
                
                # Determine if binary (2 levels) or multi-level (>2 levels)
                num_levels = len(keep_levels)
                is_binary = num_levels == 2
                self.categorical_is_binary_[c] = is_binary
                
                # For binary features, determine reference level (the one that becomes 1)
                if is_binary:
                    # Use configured reference level if provided and valid
                    if (self.categorical_reference_levels and 
                        c in self.categorical_reference_levels):
                        desired_ref = self.categorical_reference_levels[c]
                        if desired_ref in keep_levels:
                            self.binary_reference_levels_[c] = desired_ref
                        else:
                            # Use first level as default
                            self.binary_reference_levels_[c] = keep_levels[0]
                            warnings.warn(
                                f"Requested reference '{desired_ref}' for binary feature '{c}' "
                                f"not in frozen levels. Using '{keep_levels[0]}' as reference."
                            )
                    else:
                        # Use first level as default reference
                        self.binary_reference_levels_[c] = keep_levels[0]
                else:
                    # For multi-level, prepare feature mapping for one-hot encoding
                    # Column names will be: {original_name}_{level}
                    onehot_cols = [f"{c}_{level}" for level in keep_levels]
                    self.feature_mapping_[c] = onehot_cols

        # Persist the decision about which columns are considered numeric/categorical
        self.numeric_columns = numeric_cols
        self.categorical_columns = categorical_cols
        
        # Build feature mapping for numeric columns (they keep their names)
        for col in numeric_cols:
            self.feature_mapping_[col] = [col]
        
        # Build feature mapping for binary categoricals (they keep their names)
        for col in categorical_cols:
            if self.categorical_is_binary_.get(col, False):
                self.feature_mapping_[col] = [col]
        
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using fitted preprocessor.
        
        Args:
            df: DataFrame to transform (train/eval/test)
            
        Returns:
            Transformed DataFrame with one-hot encoded categoricals
        """
        if self.numeric_columns is None or self.categorical_columns is None:
            raise RuntimeError("OneHotPreprocessor must be fit before transform")

        out = df.copy()
        result_dfs = []

        # Process numeric columns
        numeric_df = out[self.numeric_columns].copy()
        for c in self.numeric_columns:
            if self.impute_numeric and c in self.numeric_medians_:
                numeric_df[c] = numeric_df[c].fillna(self.numeric_medians_[c])
            if self.scale_numeric and c in self.numeric_means_ and c in self.numeric_stds_:
                numeric_df[c] = (numeric_df[c] - self.numeric_means_[c]) / (self.numeric_stds_[c] or 1.0)
        result_dfs.append(numeric_df)

        # Process categorical columns
        for c in self.categorical_columns:
            if c not in self.categorical_levels_:
                continue
            
            levels = list(self.categorical_levels_[c])
            col_series = out[c].astype("string")
            
            # Handle missing values
            if self.categorical_missing_seen_.get(c, False):
                col_series = col_series.fillna("__MISSING__")
            else:
                # If training had no missing, handle based on handle_unseen_categories flag
                if self.handle_unseen_categories:
                    col_series = col_series.fillna("Other")
                else:
                    # In strict mode, missing values are an error
                    if col_series.isna().any():
                        raise ValueError(
                            f"Found missing values in column '{c}' during transform, "
                            f"but this column had no missing values in training. "
                            f"Set handle_unseen_categories=True to allow this."
                        )
            
            # Handle unseen categories
            if self.handle_unseen_categories:
                # Map unseen categories to "Other"
                col_series = col_series.apply(
                    lambda v: v if v in levels else ("__MISSING__" if (v == "__MISSING__" and "__MISSING__" in levels) else "Other")
                )
            else:
                # Strict mode: check for unseen categories and raise error
                unseen_mask = ~col_series.isin(levels) & ~col_series.isna()
                if unseen_mask.any():
                    unseen_values = col_series[unseen_mask].unique().tolist()
                    raise ValueError(
                        f"Found unseen category values in column '{c}': {unseen_values}. "
                        f"Expected only values from training: {levels}. "
                        f"Set handle_unseen_categories=True to map unseen values to 'Other'."
                    )
                # All values are valid (in levels or NaN), no transformation needed
            
            is_binary = self.categorical_is_binary_.get(c, False)
            
            if is_binary:
                # Binary categorical: convert to 0/1 numeric
                reference_level = self.binary_reference_levels_[c]
                binary_col = (col_series == reference_level).astype(int)
                binary_df = pd.DataFrame({c: binary_col})
                result_dfs.append(binary_df)
            else:
                # Multi-level categorical: one-hot encode
                # Create one-hot encoded columns
                onehot_data = {}
                for level in levels:
                    col_name = f"{c}_{level}"
                    onehot_data[col_name] = (col_series == level).astype(int)
                
                onehot_df = pd.DataFrame(onehot_data)
                result_dfs.append(onehot_df)

        # Concatenate all processed columns
        result = pd.concat(result_dfs, axis=1)
        
        # Ensure column order matches feature_mapping_ order
        ordered_cols = []
        for orig_col in self.numeric_columns + self.categorical_columns:
            if orig_col in self.feature_mapping_:
                ordered_cols.extend(self.feature_mapping_[orig_col])
        
        # Only include columns that actually exist (in case of edge cases)
        ordered_cols = [col for col in ordered_cols if col in result.columns]
        result = result[ordered_cols]
        
        return result

    def get_feature_mapping(self) -> Dict[str, List[str]]:
        """Get mapping from original feature names to new column names.
        
        Returns:
            Dictionary mapping original feature name -> list of new column names
        """
        if not self.feature_mapping_:
            raise RuntimeError("Preprocessor must be fit before getting feature mapping")
        return self.feature_mapping_.copy()

    def get_transformed_feature_names(self) -> List[str]:
        """Get list of all feature names after transformation.
        
        Returns:
            List of feature column names in the transformed DataFrame
        """
        if not self.feature_mapping_:
            raise RuntimeError("Preprocessor must be fit before getting transformed feature names")
        
        all_features = []
        for orig_col in self.numeric_columns + self.categorical_columns:
            if orig_col in self.feature_mapping_:
                all_features.extend(self.feature_mapping_[orig_col])
        return all_features

