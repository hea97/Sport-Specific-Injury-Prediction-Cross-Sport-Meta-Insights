"""Smoke tests for data loading."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.data import load_dataset
from src.train.config import load_experiment_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataLoadingTests(unittest.TestCase):
    """Ensure each configured loader returns non-empty data."""

    def test_loaders_return_rows(self) -> None:
        for config_name in ("nba", "football", "multimodal"):
            with self.subTest(config=config_name):
                config = load_experiment_config(PROJECT_ROOT / "configs" / f"{config_name}.yaml")
                loaded = load_dataset(
                    dataset_name=config.dataset.loader,
                    dataset_path=config.resolve_path(config.dataset.path),
                )
                self.assertGreater(len(loaded.frame), 0)
                self.assertGreaterEqual(loaded.report.row_count_before, loaded.report.row_count_after)


if __name__ == "__main__":
    unittest.main()
