"""Config-aware dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schemas import (
    ValidationReport,
    coerce_numeric_columns,
    drop_rows_missing_any,
    ensure_required_columns,
    parse_datetime_columns,
)


@dataclass
class LoadedDataset:
    """Container for a loaded dataset and its validation metadata."""

    dataset_name: str
    frame: pd.DataFrame
    report: ValidationReport


def load_dataset(dataset_name: str, dataset_path: Path) -> LoadedDataset:
    """Load a configured dataset and apply dataset-specific coercions."""

    normalized_name = dataset_name.lower()
    if normalized_name == "nba":
        return _load_nba(dataset_path)
    if normalized_name == "football":
        return _load_football(dataset_path)
    if normalized_name == "multimodal":
        return _load_multimodal(dataset_path)
    raise ValueError(f"Unsupported dataset loader: {dataset_name}")


def _read_csv(dataset_name: str, dataset_path: Path) -> tuple[pd.DataFrame, ValidationReport]:
    frame = pd.read_csv(dataset_path)
    report = ValidationReport(
        dataset_name=dataset_name,
        row_count_before=len(frame),
        row_count_after=len(frame),
    )
    return frame, report


def _strip_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    object_columns = cleaned.select_dtypes(include=["object"]).columns
    for column in object_columns:
        cleaned[column] = cleaned[column].apply(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return cleaned


def _load_nba(dataset_path: Path) -> LoadedDataset:
    frame, report = _read_csv("nba", dataset_path)
    ensure_required_columns(frame, ["Date", "Team", "Relinquished", "Notes"], "nba")
    frame = _strip_object_columns(frame)
    frame["Team"] = frame["Team"].fillna("Unknown")
    frame = parse_datetime_columns(frame, ["Date"], report)
    frame = drop_rows_missing_any(frame, ["Date", "Notes"], report)
    return LoadedDataset(dataset_name="nba", frame=frame, report=report)


def _load_football(dataset_path: Path) -> LoadedDataset:
    frame, report = _read_csv("football", dataset_path)
    ensure_required_columns(
        frame,
        [
            "Name",
            "Team Name",
            "Position",
            "Age",
            "FIFA rating",
            "Injury",
            "Date of Injury",
            "Date of return",
        ],
        "football",
    )
    frame = _strip_object_columns(frame)
    frame = parse_datetime_columns(frame, ["Date of Injury", "Date of return"], report)

    numeric_columns = ["Age", "FIFA rating"]
    for match_number in (1, 2, 3):
        numeric_columns.extend(
            [
                f"Match{match_number}_before_injury_GD",
                f"Match{match_number}_before_injury_Player_rating",
                f"Match{match_number}_missed_match_GD",
                f"Match{match_number}_after_injury_GD",
                f"Match{match_number}_after_injury_Player_rating",
            ]
        )

    existing_numeric_columns = [column for column in numeric_columns if column in frame.columns]
    frame = coerce_numeric_columns(frame, existing_numeric_columns, report)
    frame = drop_rows_missing_any(frame, ["Date of Injury", "Date of return", "Age", "FIFA rating"], report)
    return LoadedDataset(dataset_name="football", frame=frame, report=report)


def _load_multimodal(dataset_path: Path) -> LoadedDataset:
    frame, report = _read_csv("multimodal", dataset_path)
    ensure_required_columns(frame, ["injury_risk"], "multimodal")
    numeric_columns = list(frame.columns)
    frame = coerce_numeric_columns(frame, numeric_columns, report)
    frame = drop_rows_missing_any(frame, ["injury_risk"], report)
    return LoadedDataset(dataset_name="multimodal", frame=frame, report=report)
