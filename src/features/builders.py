"""Dataset-specific feature builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .normalization import bounded_score, normalize_feature_frame, safe_zscore


@dataclass
class BuiltDataset:
    """A model-ready dataset with engineered features and audit notes."""

    dataset_name: str
    frame: pd.DataFrame
    target_column: str
    numeric_features: list[str]
    categorical_features: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def feature_columns(self) -> list[str]:
        """Return all model input columns in order."""

        return [*self.numeric_features, *self.categorical_features]


def build_feature_dataset(frame: pd.DataFrame, config: Any) -> BuiltDataset:
    """Dispatch to the configured feature builder."""

    builder_name = config.features.builder.lower()
    if builder_name == "nba":
        return _build_nba_features(frame, config)
    if builder_name == "football":
        return _build_football_features(frame, config)
    if builder_name == "multimodal":
        return _build_multimodal_features(frame, config)
    raise ValueError(f"Unsupported feature builder: {config.features.builder}")


def _build_nba_features(frame: pd.DataFrame, config: Any) -> BuiltDataset:
    working = frame.copy().sort_values("Date").reset_index(drop=True)
    target = _keyword_target(
        working[config.label.source_column],
        positive_keywords=config.label.positive_keywords,
    )
    working[config.label.target_column] = target
    working["Month"] = working["Date"].dt.month.astype(int)
    working["Season_Phase"] = working["Month"].map(
        {
            10: 0,
            11: 0,
            12: 0,
            1: 1,
            2: 1,
            3: 2,
            4: 2,
            5: 3,
            6: 3,
            7: 3,
            8: 3,
            9: 3,
        }
    ).fillna(1)
    working["Days_Missed_Proxy"] = _weighted_keyword_score(
        working["Notes"],
        {
            "out indefinitely": 10.0,
            "out for season": 14.0,
            "surgery": 12.0,
            "fracture": 9.0,
            "broken": 8.0,
            "torn": 11.0,
            "strain": 5.0,
            "sprain": 4.0,
            "rest": 2.0,
            "sore": 1.0,
        },
    )
    working["Team_Injury_Burden_7d"] = _rolling_group_event_count(working, "Team", "Date", "7D")
    working["Team_Injury_Burden_30d"] = _rolling_group_event_count(working, "Team", "Date", "30D")
    working = normalize_feature_frame(
        working,
        ["Month", "Season_Phase", "Days_Missed_Proxy", "Team_Injury_Burden_7d", "Team_Injury_Burden_30d"],
        prefix="norm_",
    )
    load_signal = (
        0.35 * safe_zscore(working["Days_Missed_Proxy"])
        + 0.40 * safe_zscore(working["Team_Injury_Burden_7d"])
        + 0.15 * safe_zscore(working["Team_Injury_Burden_30d"])
        + 0.10 * safe_zscore(working["Season_Phase"])
    )
    working["Load_Score"] = bounded_score(load_signal)

    built = BuiltDataset(
        dataset_name="nba",
        frame=working,
        target_column=config.label.target_column,
        numeric_features=list(config.features.numeric_features),
        categorical_features=list(config.features.categorical_features),
        notes=[
            "NBA labels are derived from note keywords because the source file does not contain an explicit binary target.",
            "Load_Score is deterministic and based on team-level injury burden plus note-derived severity proxies.",
        ],
    )
    _ensure_feature_columns_exist(built)
    return built


def _build_football_features(frame: pd.DataFrame, config: Any) -> BuiltDataset:
    working = frame.copy().reset_index(drop=True)
    working["Days_Out"] = (working["Date of return"] - working["Date of Injury"]).dt.days
    invalid_days = (working["Days_Out"] < 0).sum()
    if invalid_days:
        working = working.loc[working["Days_Out"] >= 0].copy()

    working[config.label.target_column] = (
        working["Days_Out"] >= int(config.label.min_days)
    ).astype(int)
    working["Month"] = working["Date of Injury"].dt.month.astype(int)
    working["Injury_Type"] = working["Injury"].fillna("Unknown")
    working["Position_Group"] = working["Position"].fillna("Unknown")

    before_rating_columns = [
        column for column in working.columns if column.endswith("_before_injury_Player_rating")
    ]
    before_goal_diff_columns = [
        column for column in working.columns if column.endswith("_before_injury_GD")
    ]
    before_result_columns = [
        column for column in working.columns if column.endswith("_before_injury_Result")
    ]

    working["PreInjury_Player_Rating_Mean"] = working[before_rating_columns].mean(axis=1)
    working["PreInjury_Player_Rating_Std"] = working[before_rating_columns].std(axis=1).fillna(0.0)
    working["PreInjury_GD_Mean"] = working[before_goal_diff_columns].mean(axis=1)
    working["PreInjury_GD_AbsMean"] = working[before_goal_diff_columns].abs().mean(axis=1)
    working["PreInjury_Result_Points_Mean"] = working[before_result_columns].apply(
        _mean_result_points,
        axis=1,
    )
    working["Load_Score"] = bounded_score(
        0.45 * safe_zscore(working["PreInjury_GD_AbsMean"])
        + 0.35 * safe_zscore(working["PreInjury_Player_Rating_Std"])
        + 0.20 * safe_zscore(working["PreInjury_Result_Points_Mean"].fillna(0.0))
    )

    built = BuiltDataset(
        dataset_name="football",
        frame=working,
        target_column=config.label.target_column,
        numeric_features=list(config.features.numeric_features),
        categorical_features=list(config.features.categorical_features),
        notes=[
            f"Football labels represent serious injuries lasting at least {config.label.min_days} days.",
            f"Dropped {invalid_days} rows with negative injury duration.",
        ],
    )
    _ensure_feature_columns_exist(built)
    return built


def _build_multimodal_features(frame: pd.DataFrame, config: Any) -> BuiltDataset:
    working = frame.copy().reset_index(drop=True)
    working[config.label.target_column] = working[config.label.source_column].isin(
        config.label.positive_values
    ).astype(int)
    working["Physiology_Stress_Index"] = bounded_score(
        safe_zscore(working["heart_rate"])
        + safe_zscore(working["fatigue_index"])
        + safe_zscore(working["respiratory_rate"])
        + safe_zscore(working["gsr"])
        - safe_zscore(working["spo2"])
    )
    working["Biomechanical_Load_Index"] = bounded_score(
        safe_zscore(working["ground_reaction_force"])
        + safe_zscore(working["impact_force"])
        + safe_zscore(working["acc_rms"])
        - safe_zscore(working["gait_symmetry"])
    )
    working["Recovery_Debt_Index"] = bounded_score(
        safe_zscore(working["training_duration"])
        + safe_zscore(working["workload_intensity"])
        + safe_zscore(working["repetition_count"])
        - safe_zscore(working["rest_period"])
    )
    working["Load_Score"] = bounded_score(
        0.40 * safe_zscore(working["Biomechanical_Load_Index"])
        + 0.35 * safe_zscore(working["Recovery_Debt_Index"])
        + 0.25 * safe_zscore(working["Physiology_Stress_Index"])
    )

    built = BuiltDataset(
        dataset_name="multimodal",
        frame=working,
        target_column=config.label.target_column,
        numeric_features=list(config.features.numeric_features),
        categorical_features=list(config.features.categorical_features),
        notes=[
            "Multimodal labels are taken directly from the injury_risk column.",
            "Stress, load, and recovery composite indices are deterministic cross-sport normalized summaries.",
        ],
    )
    _ensure_feature_columns_exist(built)
    return built


def _ensure_feature_columns_exist(dataset: BuiltDataset) -> None:
    missing = [column for column in dataset.feature_columns if column not in dataset.frame.columns]
    if missing:
        raise ValueError(
            f"{dataset.dataset_name} feature builder did not create required columns: {missing}"
        )


def _keyword_target(series: pd.Series, positive_keywords: list[str]) -> pd.Series:
    pattern = "|".join(positive_keywords)
    return series.fillna("").str.contains(pattern, case=False, na=False, regex=True).astype(int)


def _weighted_keyword_score(series: pd.Series, weights: dict[str, float]) -> pd.Series:
    lowered = series.fillna("").str.lower()
    score = pd.Series(np.zeros(len(lowered)), index=series.index, dtype=float)
    for keyword, weight in weights.items():
        score += lowered.str.contains(keyword, regex=False).astype(float) * weight
    return score


def _rolling_group_event_count(
    frame: pd.DataFrame,
    group_column: str,
    date_column: str,
    window: str,
) -> pd.Series:
    working = frame[[group_column, date_column]].copy()
    working["_event"] = 1.0
    working["_row_id"] = np.arange(len(working))
    working = working.sort_values([group_column, date_column, "_row_id"])
    rolling = (
        working.set_index(date_column)
        .groupby(group_column)["_event"]
        .rolling(window=window, closed="both")
        .sum()
        .reset_index(level=0, drop=True)
        .to_numpy()
    )
    working["_burden"] = np.clip(rolling - 1.0, a_min=0.0, a_max=None)
    ordered = working.sort_values("_row_id")["_burden"].reset_index(drop=True)
    return ordered


def _result_to_points(result: Any) -> float:
    mapping = {"win": 3.0, "draw": 1.0, "lose": 0.0}
    if not isinstance(result, str):
        return np.nan
    return mapping.get(result.strip().lower(), np.nan)


def _mean_result_points(row: pd.Series) -> float:
    values = [value for value in (_result_to_points(item) for item in row) if not np.isnan(value)]
    if not values:
        return np.nan
    return float(np.mean(values))
