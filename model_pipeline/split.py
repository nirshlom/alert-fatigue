from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def predictive_time_split(
    df: pd.DataFrame,
    date_column: str,
    train_frac: float,
    eval_frac: float,
    test_frac: float,
    ascending: bool = True,
    stratify: bool = False,
    target_column: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/eval/test by time order (predictive split).

    Sorts the DataFrame by the `date_column` and slices by the provided
    fractions to produce chronological training, evaluation, and test sets.

    If `stratify=True` and `target_column` is provided, a best-effort label-aware
    split is performed by first bucketizing time into small consecutive bins
    and allocating whole bins into splits to reduce label drift while
    maintaining chronology.
    """
    total = train_frac + eval_frac + test_frac
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train/eval/test fractions must sum to 1.0; got {total:.6f}"
        )

    if date_column not in df.columns:
        raise KeyError(f"date_column '{date_column}' not found in DataFrame")

    df_sorted = df.sort_values(by=date_column, ascending=ascending).reset_index(drop=True)
    n = len(df_sorted)
    n_train = int(np.floor(train_frac * n))
    n_eval = int(np.floor(eval_frac * n))
    # Ensure at least 1 row in each split when possible
    n_train = max(min(n_train, n - 2), 1) if n >= 3 else max(n_train, 1)
    n_eval = max(min(n_eval, n - n_train - 1), 1) if n - n_train >= 2 else max(n_eval, 1)
    n_test = n - n_train - n_eval

    if not stratify or target_column is None or target_column not in df_sorted.columns:
        train_df = df_sorted.iloc[:n_train].copy()
        eval_df = df_sorted.iloc[n_train : n_train + n_eval].copy()
        test_df = df_sorted.iloc[n_train + n_eval :].copy()
        return train_df, eval_df, test_df

    # Best-effort temporal stratification: bin the timeline into equal-sized chunks
    # (by row count), compute label proportions per bin, then allocate bins to
    # train/eval/test to approximate requested fractions.
    num_bins = min(100, n)  # up to 100 bins or n rows
    bin_edges = np.linspace(0, n, num_bins + 1, dtype=int)
    bins = [df_sorted.iloc[bin_edges[i] : bin_edges[i + 1]] for i in range(num_bins)]

    # Compute cumulative sizes to assign bins by running total
    total_rows = n
    train_rows_target = int(np.floor(train_frac * total_rows))
    eval_rows_target = int(np.floor(eval_frac * total_rows))

    allocated = []
    rows_train = rows_eval = rows_test = 0
    for b in bins:
        if rows_train + len(b) <= train_rows_target:
            allocated.append(("train", b))
            rows_train += len(b)
        elif rows_eval + len(b) <= eval_rows_target:
            allocated.append(("eval", b))
            rows_eval += len(b)
        else:
            allocated.append(("test", b))
            rows_test += len(b)

    train_df = pd.concat([b for tag, b in allocated if tag == "train"], ignore_index=True)
    eval_df = pd.concat([b for tag, b in allocated if tag == "eval"], ignore_index=True)
    test_df = pd.concat([b for tag, b in allocated if tag == "test"], ignore_index=True)

    return train_df, eval_df, test_df


