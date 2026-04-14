"""Backward-compatible model wrappers for legacy script usage."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.train.estimators import build_model_pipeline
from src.train.evaluation import compute_classification_metrics, positive_class_scores

LOGGER = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "random_forest": {"n_estimators": 300, "max_depth": 20},
    "xgboost": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1},
    "lightgbm": {"n_estimators": 300, "max_depth": 10, "learning_rate": 0.1},
    "mlp": {
        "hidden_dims": (128, 64, 32),
        "dropout": 0.3,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 128,
    },
}


def train_rf(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sport_name: str,
) -> tuple[float, Any]:
    """Legacy wrapper for RandomForest training."""

    return _legacy_train("random_forest", X_train, X_test, y_train, y_test, sport_name)


def train_xgb(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sport_name: str,
) -> tuple[float, Any]:
    """Legacy wrapper for XGBoost training."""

    return _legacy_train("xgboost", X_train, X_test, y_train, y_test, sport_name)


def train_lgb(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sport_name: str,
) -> tuple[float, Any]:
    """Legacy wrapper for LightGBM training."""

    return _legacy_train("lightgbm", X_train, X_test, y_train, y_test, sport_name)


def train_mlp(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sport_name: str,
) -> tuple[float, Any]:
    """Legacy wrapper for PyTorch MLP training."""

    return _legacy_train("mlp", X_train, X_test, y_train, y_test, sport_name)


def _legacy_train(
    model_type: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sport_name: str,
) -> tuple[float, Any]:
    pipeline = build_model_pipeline(
        model_type=model_type,
        model_params=DEFAULT_MODEL_PARAMS[model_type],
        numeric_features=list(X_train.columns),
        categorical_features=[],
        smote_enabled=True,
        smote_k_neighbors=5,
        seed=DEFAULT_SEED,
    )
    fitted_pipeline = pipeline.fit(X_train, y_train)
    scores = positive_class_scores(fitted_pipeline, X_test)
    metrics = compute_classification_metrics(y_test.to_numpy(), scores, threshold=0.5)
    LOGGER.info("[%s] %s Recall: %.4f", sport_name, model_type, metrics["Recall"])
    return metrics["Recall"], fitted_pipeline
