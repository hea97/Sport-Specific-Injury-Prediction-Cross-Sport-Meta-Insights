"""End-to-end training runner."""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

try:
    import torch
except ImportError:  # pragma: no cover - exercised in dependency-constrained environments
    torch = None

from src.data import load_dataset
from src.features import build_feature_dataset

from .config import ExperimentConfig, load_experiment_config
from .estimators import TorchMLPClassifier, build_model_pipeline
from .evaluation import compute_classification_metrics, positive_class_scores, select_threshold
from .reporting import refresh_reports

LOGGER = logging.getLogger(__name__)


def run_experiment(config_or_path: ExperimentConfig | str | Path) -> pd.DataFrame:
    """Run a full training job from a config object or path."""

    config = (
        config_or_path
        if isinstance(config_or_path, ExperimentConfig)
        else load_experiment_config(config_or_path)
    )
    _configure_logging()
    _set_global_seed(config.seed)

    loaded = load_dataset(
        dataset_name=config.dataset.loader,
        dataset_path=config.resolve_path(config.dataset.path),
    )
    built = build_feature_dataset(loaded.frame, config)

    data_frame = built.frame.copy()
    feature_columns = built.feature_columns
    target_column = built.target_column
    model_frame = data_frame[feature_columns + [target_column]].dropna(subset=[target_column]).copy()

    X = model_frame[feature_columns]
    y = model_frame[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.evaluation.test_size,
        random_state=config.seed,
        stratify=y,
    )

    artifact_root = config.resolve_path(config.artifacts.artifact_dir)
    model_root = artifact_root / "models"
    prediction_root = artifact_root / "predictions"
    artifact_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    for model_config in config.models:
        LOGGER.info("Training %s on %s", model_config.name, config.dataset.name)
        pipeline = build_model_pipeline(
            model_type=model_config.type,
            model_params=model_config.hyperparameters,
            numeric_features=built.numeric_features,
            categorical_features=built.categorical_features,
            smote_enabled=config.imbalance.smote,
            smote_k_neighbors=config.imbalance.k_neighbors,
            seed=config.seed,
        )

        cv_metrics = _run_cross_validation(config, pipeline, X, y)
        fitted_pipeline = pipeline.fit(X_train, y_train)
        train_scores = positive_class_scores(fitted_pipeline, X_train)
        test_scores = positive_class_scores(fitted_pipeline, X_test)
        threshold = select_threshold(
            y_true=y_train.to_numpy(),
            y_score=train_scores,
            strategy=config.evaluation.thresholding.strategy,
            threshold=config.evaluation.thresholding.threshold,
            target_recall=config.evaluation.thresholding.target_recall,
        )
        metrics = compute_classification_metrics(y_test.to_numpy(), test_scores, threshold)

        model_path = _save_model_artifact(
            fitted_pipeline=fitted_pipeline,
            model_name=model_config.name,
            model_root=model_root,
        )
        prediction_path = _save_predictions(
            X_test=X_test,
            y_test=y_test,
            y_score=test_scores,
            threshold=threshold,
            prediction_root=prediction_root,
            model_name=model_config.name,
        )
        feature_importance_path = _save_feature_importance(
            fitted_pipeline=fitted_pipeline,
            artifact_root=artifact_root,
            model_name=model_config.name,
        )

        result_rows.append(
            {
                "Dataset": config.dataset.name,
                "Model": model_config.name,
                **metrics,
                "Train_Rows": int(len(X_train)),
                "Test_Rows": int(len(X_test)),
                "Features": int(len(feature_columns)),
                "SMOTE": bool(config.imbalance.smote),
                "Seed": int(config.seed),
                "CV_Recall_Mean": cv_metrics.get("CV_Recall_Mean"),
                "CV_Recall_Std": cv_metrics.get("CV_Recall_Std"),
                "CV_PR_AUC_Mean": cv_metrics.get("CV_PR_AUC_Mean"),
                "Model_Artifact": str(model_path.relative_to(config.project_root)),
                "Predictions_Artifact": str(prediction_path.relative_to(config.project_root)),
                "Feature_Importance_Artifact": (
                    str(feature_importance_path.relative_to(config.project_root))
                    if feature_importance_path is not None
                    else ""
                ),
            }
        )

    results = pd.DataFrame(result_rows).sort_values("Recall", ascending=False).reset_index(drop=True)
    results_path = config.resolve_path(config.artifacts.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)

    _write_metadata(
        output_path=artifact_root / "run_metadata.json",
        config=config,
        loaded_notes=loaded.report.notes,
        built_notes=built.notes,
        feature_columns=feature_columns,
        row_count=len(model_frame),
        label_balance=float(y.mean()),
    )
    _promote_best_model_artifacts(
        project_root=config.project_root,
        dataset_name=config.dataset.name,
        results=results,
    )
    refresh_reports(
        project_root=config.project_root,
        high_risk_x=config.visualization.high_risk_x,
        high_risk_y=config.visualization.high_risk_y,
    )
    LOGGER.info("Completed %s training run.", config.dataset.name)
    return results


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _run_cross_validation(
    config: ExperimentConfig,
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    if not config.evaluation.cross_validation.enabled:
        return {}

    splitter = StratifiedKFold(
        n_splits=config.evaluation.cross_validation.n_splits,
        shuffle=True,
        random_state=config.seed,
    )
    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=splitter,
        scoring={
            "recall": "recall",
            "precision": "precision",
            "f1": "f1",
            "pr_auc": "average_precision",
        },
        n_jobs=1,
    )
    return {
        "CV_Recall_Mean": float(np.mean(scores["test_recall"])),
        "CV_Recall_Std": float(np.std(scores["test_recall"])),
        "CV_PR_AUC_Mean": float(np.mean(scores["test_pr_auc"])),
    }


def _save_model_artifact(fitted_pipeline: Any, model_name: str, model_root: Path) -> Path:
    model_step = fitted_pipeline.named_steps["model"]
    normalized_name = model_name.lower()

    if isinstance(model_step, TorchMLPClassifier):
        output_path = model_root / f"{normalized_name}.pt"
        checkpoint = model_step.export_checkpoint()
        checkpoint["preprocessor"] = fitted_pipeline.named_steps["preprocess"]
        checkpoint["classes"] = getattr(model_step, "classes_", np.array([0, 1])).tolist()
        torch.save(checkpoint, output_path)
        return output_path

    output_path = model_root / f"{normalized_name}.joblib"
    joblib.dump(fitted_pipeline, output_path)
    return output_path


def _save_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    prediction_root: Path,
    model_name: str,
) -> Path:
    output_path = prediction_root / f"{model_name.lower()}_predictions.csv"
    frame = X_test.copy()
    frame["y_true"] = y_test.to_numpy()
    frame["y_score"] = y_score
    frame["y_pred"] = (y_score >= threshold).astype(int)
    frame.to_csv(output_path, index=False)
    return output_path


def _save_feature_importance(
    fitted_pipeline: Any,
    artifact_root: Path,
    model_name: str,
) -> Path | None:
    model_step = fitted_pipeline.named_steps["model"]
    if not hasattr(model_step, "feature_importances_"):
        return None

    preprocessor = fitted_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    importance_frame = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": model_step.feature_importances_,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    output_path = artifact_root / f"{model_name.lower()}_feature_importance.csv"
    importance_frame.to_csv(output_path, index=False)
    return output_path


def _write_metadata(
    output_path: Path,
    config: ExperimentConfig,
    loaded_notes: list[str],
    built_notes: list[str],
    feature_columns: list[str],
    row_count: int,
    label_balance: float,
) -> None:
    config_payload = asdict(config)
    config_payload["config_path"] = str(config.config_path)
    config_payload["project_root"] = str(config.project_root)
    payload = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_payload,
        "row_count": row_count,
        "label_balance": label_balance,
        "feature_columns": feature_columns,
        "loader_notes": loaded_notes,
        "feature_notes": built_notes,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _promote_best_model_artifacts(project_root: Path, dataset_name: str, results: pd.DataFrame) -> None:
    dataset_artifact_root = project_root / "results" / "artifacts" / dataset_name
    best_row = results.sort_values("Recall", ascending=False).iloc[0]

    best_prediction_source = project_root / best_row["Predictions_Artifact"]
    best_prediction_target = dataset_artifact_root / "best_model_predictions.csv"
    shutil.copyfile(best_prediction_source, best_prediction_target)

    feature_rows = results[results["Feature_Importance_Artifact"].fillna("").astype(str).str.len() > 0]
    feature_artifact = ""
    if not feature_rows.empty:
        feature_artifact = str(
            feature_rows.sort_values("Recall", ascending=False).iloc[0]["Feature_Importance_Artifact"]
        ).strip()
    if feature_artifact:
        source = project_root / feature_artifact
        target = dataset_artifact_root / "feature_importance.csv"
        shutil.copyfile(source, target)
