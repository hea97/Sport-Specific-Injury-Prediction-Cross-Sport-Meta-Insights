"""Model registry and CV-safe pipeline builders."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - exercised in dependency-constrained environments
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - exercised in dependency-constrained environments
    XGBClassifier = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised in dependency-constrained environments
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None


if nn is not None:

    class _TorchMLPModule(nn.Module):
        """Simple feed-forward network for tabular binary classification."""

        def __init__(self, input_size: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            previous_size = input_size
            for hidden_size in hidden_dims:
                layers.append(nn.Linear(previous_size, hidden_size))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                previous_size = hidden_size
            layers.append(nn.Linear(previous_size, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return self.network(features)

else:
    _TorchMLPModule = None


class TorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible PyTorch classifier for tabular binary classification."""

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 128,
        seed: int = 42,
        weight_decay: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.weight_decay = weight_decay
        self.verbose = verbose

    def fit(self, features: Any, target: Any) -> "TorchMLPClassifier":
        """Train the network using deterministic CPU execution."""

        if torch is None or _TorchMLPModule is None or optim is None:
            raise ImportError("torch is required to train the MLP model.")

        self._set_seed()
        feature_array = np.asarray(features, dtype=np.float32)
        target_array = np.asarray(target, dtype=np.float32).reshape(-1, 1)
        self.input_dim_ = feature_array.shape[1]
        self.classes_ = np.array([0, 1])
        self.model_ = _TorchMLPModule(self.input_dim_, self.hidden_dims, self.dropout)
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()
        dataset = TensorDataset(
            torch.from_numpy(feature_array),
            torch.from_numpy(target_array),
        )
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        batch_size = min(self.batch_size, len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        self.model_.train()
        for _ in range(self.epochs):
            for batch_features, batch_target in loader:
                optimizer.zero_grad()
                logits = self.model_(batch_features)
                loss = criterion(logits, batch_target)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, features: Any) -> np.ndarray:
        """Predict class probabilities."""

        if torch is None:
            raise ImportError("torch is required to score the MLP model.")
        feature_array = np.asarray(features, dtype=np.float32)
        feature_tensor = torch.from_numpy(feature_array)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(feature_tensor)
            positive_scores = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return np.column_stack([1.0 - positive_scores, positive_scores])

    def predict(self, features: Any) -> np.ndarray:
        """Predict hard labels at the default 0.5 threshold."""

        return (self.predict_proba(features)[:, 1] >= 0.5).astype(int)

    def export_checkpoint(self) -> dict[str, Any]:
        """Return a serializable checkpoint payload for torch.save."""

        if torch is None:
            raise ImportError("torch is required to export the MLP model.")
        return {
            "state_dict": self.model_.state_dict(),
            "input_dim": self.input_dim_,
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "weight_decay": self.weight_decay,
        }

    def _set_seed(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def build_model_pipeline(
    model_type: str,
    model_params: dict[str, Any],
    numeric_features: list[str],
    categorical_features: list[str],
    smote_enabled: bool,
    smote_k_neighbors: int,
    seed: int,
) -> Pipeline:
    """Build an imblearn Pipeline with preprocessing, SMOTE, and model steps."""

    estimator, scale_numeric = build_estimator(model_type, model_params, seed)
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )
    sampler: Any = "passthrough"
    if smote_enabled:
        sampler = SMOTE(random_state=seed, k_neighbors=smote_k_neighbors)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("sampler", sampler),
            ("model", estimator),
        ]
    )


def build_estimator(
    model_type: str,
    model_params: dict[str, Any],
    seed: int,
) -> tuple[Any, bool]:
    """Return an estimator plus whether numeric scaling should be applied."""

    normalized_type = model_type.lower()
    params = dict(model_params)

    if normalized_type == "random_forest":
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        return RandomForestClassifier(**params), False

    if normalized_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is required to train the XGBoost model.")
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("eval_metric", "logloss")
        return XGBClassifier(**params), False

    if normalized_type == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is required to train the LightGBM model.")
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("verbose", -1)
        return LGBMClassifier(**params), False

    if normalized_type == "mlp":
        params.setdefault("seed", seed)
        if "hidden_dims" in params:
            params["hidden_dims"] = tuple(params["hidden_dims"])
        return TorchMLPClassifier(**params), True

    raise ValueError(f"Unsupported model type: {model_type}")


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    """Build a shared tabular preprocessor."""

    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[str]]] = [
        ("numeric", SklearnPipeline(steps=numeric_steps), numeric_features),
    ]

    if categorical_features:
        categorical_pipeline = SklearnPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder across sklearn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - compatibility path for older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
