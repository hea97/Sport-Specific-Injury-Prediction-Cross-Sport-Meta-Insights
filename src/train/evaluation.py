"""Evaluation and thresholding helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def select_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    strategy: str = "fixed",
    threshold: float = 0.5,
    target_recall: float | None = None,
) -> float:
    """Select a classification threshold."""

    if strategy == "fixed":
        return float(threshold)

    if strategy != "target_recall":
        raise ValueError(f"Unsupported threshold strategy: {strategy}")

    if target_recall is None:
        raise ValueError("target_recall strategy requires a target recall value.")

    precision_values, recall_values, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return float(threshold)

    candidates: list[tuple[float, float, float]] = []
    for candidate_threshold, precision_value, recall_value in zip(
        thresholds,
        precision_values[:-1],
        recall_values[:-1],
    ):
        if recall_value >= target_recall:
            candidates.append((float(candidate_threshold), float(precision_value), float(recall_value)))

    if candidates:
        candidates.sort(key=lambda item: (item[1], item[2], -item[0]), reverse=True)
        return candidates[0][0]

    return float(threshold)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Compute standardized recall-first metrics."""

    y_true_array = np.asarray(y_true)
    y_score_array = np.asarray(y_score)
    y_pred = (y_score_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()
    return {
        "Recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "PR_AUC": float(average_precision_score(y_true_array, y_score_array)),
        "Threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def positive_class_scores(model: Any, features: Any) -> np.ndarray:
    """Return positive-class scores from a fitted classifier."""

    probabilities = model.predict_proba(features)
    if probabilities.ndim == 1:
        return probabilities
    return probabilities[:, 1]
