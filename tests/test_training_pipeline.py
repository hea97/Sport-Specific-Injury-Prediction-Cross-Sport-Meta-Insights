"""Smoke tests for model pipeline construction and output shape."""

from __future__ import annotations

import unittest

import pandas as pd

try:
    from src.train.estimators import build_model_pipeline
except ModuleNotFoundError:
    build_model_pipeline = None


@unittest.skipIf(build_model_pipeline is None, "scikit-learn/imblearn stack is not installed")
class TrainingPipelineTests(unittest.TestCase):
    """Verify the training pipeline fits and scores simple synthetic data."""

    def test_random_forest_pipeline_predicts_expected_shape(self) -> None:
        features = pd.DataFrame(
            {
                "numeric_a": [0.1, 0.2, 0.8, 0.9, 0.15, 0.85],
                "numeric_b": [1.0, 1.1, 0.1, 0.2, 0.9, 0.3],
                "category": ["A", "A", "B", "B", "A", "B"],
            }
        )
        target = pd.Series([0, 0, 1, 1, 0, 1], name="target")
        pipeline = build_model_pipeline(
            model_type="random_forest",
            model_params={"n_estimators": 20, "max_depth": 4},
            numeric_features=["numeric_a", "numeric_b"],
            categorical_features=["category"],
            smote_enabled=False,
            smote_k_neighbors=5,
            seed=42,
        )

        fitted = pipeline.fit(features, target)
        probabilities = fitted.predict_proba(features)
        self.assertEqual(probabilities.shape, (len(features), 2))


if __name__ == "__main__":
    unittest.main()
