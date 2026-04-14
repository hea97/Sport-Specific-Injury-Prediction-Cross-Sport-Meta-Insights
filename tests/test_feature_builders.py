"""Smoke tests for feature engineering."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.data import load_dataset
from src.features import build_feature_dataset
from src.train.config import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FeatureBuilderTests(unittest.TestCase):
    """Ensure configured feature builders create the requested columns."""

    def test_builders_create_requested_features(self) -> None:
        for config_name in ("nba", "football", "multimodal"):
            with self.subTest(config=config_name):
                config = load_experiment_config(PROJECT_ROOT / "configs" / f"{config_name}.yaml")
                loaded = load_dataset(
                    dataset_name=config.dataset.loader,
                    dataset_path=config.resolve_path(config.dataset.path),
                )
                built = build_feature_dataset(loaded.frame, config)
                for feature_name in built.feature_columns:
                    self.assertIn(feature_name, built.frame.columns)
                self.assertIn(built.target_column, built.frame.columns)


if __name__ == "__main__":
    unittest.main()
