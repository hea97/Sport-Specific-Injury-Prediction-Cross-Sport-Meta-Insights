"""Configuration parsing for reproducible experiment runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - exercised in dependency-constrained environments
    yaml = None


@dataclass
class DatasetConfig:
    """Dataset-level configuration."""

    name: str
    path: str
    loader: str


@dataclass
class LabelConfig:
    """Target-building configuration."""

    target_column: str
    strategy: str
    source_column: str | None = None
    positive_keywords: list[str] = field(default_factory=list)
    positive_values: list[int] = field(default_factory=lambda: [1])
    start_column: str | None = None
    end_column: str | None = None
    min_days: int = 0


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""

    builder: str
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Single model specification."""

    name: str
    type: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImbalanceConfig:
    """Class imbalance handling configuration."""

    smote: bool = True
    k_neighbors: int = 5


@dataclass
class ThresholdConfig:
    """Decision threshold configuration."""

    strategy: str = "fixed"
    threshold: float = 0.5
    target_recall: float | None = None


@dataclass
class CrossValidationConfig:
    """Optional cross-validation configuration."""

    enabled: bool = False
    n_splits: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    test_size: float = 0.2
    thresholding: ThresholdConfig = field(default_factory=ThresholdConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)


@dataclass
class ArtifactConfig:
    """Artifact path configuration."""

    results_csv: str
    artifact_dir: str


@dataclass
class VisualizationConfig:
    """Optional plot hints."""

    high_risk_x: str = "workload_intensity"
    high_risk_y: str = "rest_period"


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    dataset: DatasetConfig
    label: LabelConfig
    features: FeatureConfig
    models: list[ModelConfig]
    imbalance: ImbalanceConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactConfig
    visualization: VisualizationConfig
    seed: int
    config_path: Path
    project_root: Path

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve repo-relative paths from config values."""

        return (self.project_root / relative_path).resolve()


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    """Load and validate a dataset YAML configuration."""

    resolved_path = Path(config_path).resolve()
    project_root = resolved_path.parents[1]
    raw_text = resolved_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw_text) if yaml is not None else json.loads(raw_text)

    dataset = DatasetConfig(**payload["dataset"])
    label = LabelConfig(**payload["label"])
    features = FeatureConfig(**payload["features"])
    models = [ModelConfig(**item) for item in payload["models"]]
    imbalance = ImbalanceConfig(**payload.get("imbalance", {}))
    evaluation = _build_evaluation(payload.get("evaluation", {}))
    artifacts = ArtifactConfig(**payload["artifacts"])
    visualization = VisualizationConfig(**payload.get("visualization", {}))

    return ExperimentConfig(
        dataset=dataset,
        label=label,
        features=features,
        models=models,
        imbalance=imbalance,
        evaluation=evaluation,
        artifacts=artifacts,
        visualization=visualization,
        seed=int(payload.get("seed", 42)),
        config_path=resolved_path,
        project_root=project_root,
    )


def _build_evaluation(payload: dict[str, Any]) -> EvaluationConfig:
    thresholding = ThresholdConfig(**payload.get("thresholding", {}))
    cross_validation = CrossValidationConfig(**payload.get("cross_validation", {}))
    return EvaluationConfig(
        test_size=float(payload.get("test_size", 0.2)),
        thresholding=thresholding,
        cross_validation=cross_validation,
    )
