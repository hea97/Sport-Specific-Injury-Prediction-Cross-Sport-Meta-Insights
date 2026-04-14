"""Schema validation helpers for CSV-backed datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


@dataclass
class ValidationReport:
    """Collects row-level cleaning actions and validation notes."""

    dataset_name: str
    row_count_before: int
    row_count_after: int
    dropped_rows: int = 0
    notes: list[str] = field(default_factory=list)

    def add_note(self, message: str) -> None:
        """Append a human-readable validation note."""

        self.notes.append(message)


def ensure_required_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Raise a helpful error when mandatory columns are missing."""

    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing)}"
        )


def parse_datetime_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    report: ValidationReport,
) -> pd.DataFrame:
    """Convert date-like columns to pandas datetimes."""

    parsed = frame.copy()
    for column in columns:
        before_missing = parsed[column].isna().sum()
        parsed[column] = pd.to_datetime(parsed[column], errors="coerce")
        after_missing = parsed[column].isna().sum()
        introduced = after_missing - before_missing
        if introduced > 0:
            report.add_note(f"Coerced {introduced} invalid datetime values in '{column}'.")
    return parsed


def coerce_numeric_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    report: ValidationReport,
) -> pd.DataFrame:
    """Convert numeric-like object columns into numeric dtype."""

    coerced = frame.copy()
    for column in columns:
        before_missing = coerced[column].isna().sum()
        coerced[column] = pd.to_numeric(coerced[column], errors="coerce")
        after_missing = coerced[column].isna().sum()
        introduced = after_missing - before_missing
        if introduced > 0:
            report.add_note(f"Coerced {introduced} invalid numeric values in '{column}'.")
    return coerced


def drop_rows_missing_any(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    report: ValidationReport,
) -> pd.DataFrame:
    """Drop rows missing any required values and update the report."""

    cleaned = frame.dropna(subset=list(required_columns)).copy()
    dropped = len(frame) - len(cleaned)
    if dropped:
        report.dropped_rows += dropped
        report.add_note(
            f"Dropped {dropped} rows missing required values: {', '.join(required_columns)}."
        )
    report.row_count_after = len(cleaned)
    return cleaned
