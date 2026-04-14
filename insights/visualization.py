"""Regenerate insight plots from standardized result artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.config import load_experiment_config
from src.train.reporting import refresh_reports


def main() -> None:
    """Refresh the comparison markdown and plots."""

    multimodal_config = load_experiment_config(PROJECT_ROOT / "configs" / "multimodal.yaml")
    refresh_reports(
        project_root=PROJECT_ROOT,
        high_risk_x=multimodal_config.visualization.high_risk_x,
        high_risk_y=multimodal_config.visualization.high_risk_y,
    )


if __name__ == "__main__":
    main()
