"""SHAP placeholder utilities."""

from __future__ import annotations

from pathlib import Path


def shap_summary_placeholder(output_path: Path, reason: str | None = None) -> str:
    """Write a small placeholder note when SHAP is unavailable or not configured."""

    message = reason or "SHAP support is not installed in this environment yet."
    output_path.write_text(message + "\n", encoding="utf-8")
    return message
