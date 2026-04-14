"""Partial dependence placeholder utilities."""

from __future__ import annotations

from pathlib import Path


def partial_dependence_placeholder(output_path: Path, reason: str | None = None) -> str:
    """Write a placeholder note for pending PDP support."""

    message = reason or "Partial dependence support is not implemented for this pipeline yet."
    output_path.write_text(message + "\n", encoding="utf-8")
    return message
