"""Tests for report generation helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from src.train.reporting import _plot_model_comparison, _prepare_model_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReportingTests(unittest.TestCase):
    """Verify portfolio-facing comparison plots are generated predictably."""

    def setUp(self) -> None:
        self.combined = pd.DataFrame(
            [
                {"Dataset": "Football", "Model": "RF", "Recall": 0.46},
                {"Dataset": "Football", "Model": "XGB", "Recall": 0.50},
                {"Dataset": "Football", "Model": "LGB", "Recall": 0.44},
                {"Dataset": "Football", "Model": "MLP", "Recall": 0.56},
                {"Dataset": "Multimodal", "Model": "RF", "Recall": 0.33},
                {"Dataset": "Multimodal", "Model": "XGB", "Recall": 0.35},
                {"Dataset": "Multimodal", "Model": "LGB", "Recall": 0.31},
                {"Dataset": "Multimodal", "Model": "MLP", "Recall": 0.44},
                {"Dataset": "NBA", "Model": "RF", "Recall": 0.98},
                {"Dataset": "NBA", "Model": "XGB", "Recall": 0.97},
                {"Dataset": "NBA", "Model": "LGB", "Recall": 0.96},
                {"Dataset": "NBA", "Model": "MLP", "Recall": 0.95},
            ]
        )

    def test_prepare_model_comparison_uses_portfolio_order(self) -> None:
        pivoted, best_models, lower_bound = _prepare_model_comparison(self.combined)

        self.assertEqual(list(pivoted.index), ["NBA", "Football", "Multimodal"])
        self.assertEqual(list(pivoted.columns), ["RF", "XGB", "LGB", "MLP"])
        self.assertEqual(best_models.to_dict(), {"NBA": "RF", "Football": "MLP", "Multimodal": "MLP"})
        self.assertEqual(lower_bound, 0.25)

    def test_plot_model_comparison_writes_png(self) -> None:
        output_path = PROJECT_ROOT / "tests" / "_tmp_model_comparison.png"
        try:
            _plot_model_comparison(output_path, self.combined)

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
        finally:
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    unittest.main()
