"""Reusable normalization helpers for cross-sport feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_zscore(series: pd.Series) -> pd.Series:
    """Return a zero-safe z-score series."""

    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    std = numeric.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(numeric)), index=series.index, dtype=float)
    mean = numeric.mean()
    return (numeric - mean) / std


def bounded_score(series: pd.Series, lower: float = 0.0, upper: float = 100.0) -> pd.Series:
    """Map a numeric series into a bounded range using min-max scaling."""

    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    minimum = numeric.min()
    maximum = numeric.max()
    if np.isclose(maximum, minimum):
        midpoint = (upper + lower) / 2
        return pd.Series(np.full(len(numeric), midpoint), index=series.index, dtype=float)
    scaled = (numeric - minimum) / (maximum - minimum)
    return lower + scaled * (upper - lower)


def normalize_feature_frame(
    frame: pd.DataFrame,
    columns: list[str],
    prefix: str = "norm_",
) -> pd.DataFrame:
    """Add normalized cross-sport columns into a copy of the given frame."""

    normalized = frame.copy()
    for column in columns:
        normalized[f"{prefix}{column}"] = safe_zscore(normalized[column])
    return normalized
