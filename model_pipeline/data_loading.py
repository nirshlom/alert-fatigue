from __future__ import annotations

from typing import Optional

import pandas as pd


def load_csv(input_csv_path: str, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    """Load a CSV file with optional datetime parsing.

    Parameters:
        input_csv_path: Absolute or relative path to CSV.
        parse_dates: Optional list of columns to parse as dates.

    Returns:
        DataFrame with parsed data.
    """
    df = pd.read_csv(input_csv_path, low_memory=False)
    if parse_dates:
        for col in parse_dates:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a DataFrame with only the specified columns that exist in df.

    Raises a KeyError if any of the requested columns are missing, so
    failures are explicit and early.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input data: {missing}")
    return df[columns].copy()


