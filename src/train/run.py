"""CLI entrypoint for config-driven experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run a reproducible injury prediction experiment.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the dataset YAML config, for example configs/nba.yaml",
    )
    return parser.parse_args()


def main() -> None:
    """Run the configured experiment."""

    args = parse_args()
    try:
        from .runner import run_experiment
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "Missing training dependencies. Install them with `python -m pip install -r requirements.txt`."
        ) from exc
    run_experiment(Path(args.config))


if __name__ == "__main__":
    main()
