from __future__ import annotations

from typing import Optional

import pandas as pd


def generate_profile_report(df: pd.DataFrame, output_path: str) -> str:
    """Generate a ydata-profiling HTML report for the given DataFrame.

    Returns the output path. If ydata-profiling is unavailable, writes a
    minimal CSV head preview instead to ensure the pipeline doesn't break.
    """
    try:
        from ydata_profiling import ProfileReport  # type: ignore

        profile = ProfileReport(df, title="Training Set Profile", explorative=True)
        profile.to_file(output_path)
        return output_path
    except Exception:
        # Fallback: write a basic preview CSV
        fallback = output_path.replace(".html", "_fallback_preview.csv")
        df.head(200).to_csv(fallback, index=False)
        return fallback


