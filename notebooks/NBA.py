"""Backward-compatible NBA experiment entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Run the NBA config-driven experiment."""

    try:
        from src.train.runner import run_experiment
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "Missing training dependencies. Install them with `python -m pip install -r requirements.txt`."
        ) from exc
    run_experiment(PROJECT_ROOT / "configs" / "nba.yaml")


if __name__ == "__main__":
    main()
