"""Unit tests for metric computation."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from src.train.evaluation import compute_classification_metrics
except ModuleNotFoundError:
    compute_classification_metrics = None


@unittest.skipIf(compute_classification_metrics is None, "scikit-learn is not installed")
class MetricsTests(unittest.TestCase):
    """Verify confusion matrix accounting and summary metrics."""

    def test_metric_dictionary_contains_expected_values(self) -> None:
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.4, 0.8, 0.1])
        metrics = compute_classification_metrics(y_true=y_true, y_score=y_score, threshold=0.5)

        self.assertEqual(metrics["TP"], 1)
        self.assertEqual(metrics["FP"], 1)
        self.assertEqual(metrics["FN"], 1)
        self.assertEqual(metrics["TN"], 1)
        self.assertAlmostEqual(metrics["Recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
