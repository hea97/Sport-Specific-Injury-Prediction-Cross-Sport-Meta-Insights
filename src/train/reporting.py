"""Reporting and visualization generation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger(__name__)


def refresh_reports(project_root: Path, high_risk_x: str, high_risk_y: str) -> None:
    """Regenerate comparison markdown and insights plots from latest artifacts."""

    results_dir = project_root / "results"
    insights_dir = project_root / "insights"
    insights_dir.mkdir(parents=True, exist_ok=True)

    combined = _collect_results(results_dir)
    if combined.empty:
        LOGGER.warning("No standardized results artifacts were found; skipping report generation.")
        return

    _write_model_comparison(results_dir / "MODEL_COMPARISON.md", combined)
    _plot_model_comparison(insights_dir / "Model_Comparison_Barplot.png", combined)
    _plot_nba_feature_importance(project_root, insights_dir / "NBA_Feature_Importance.png")
    _plot_multimodal_risk_zone(
        project_root,
        insights_dir / "Multimodal_HighRisk_Zone.png",
        high_risk_x=high_risk_x,
        high_risk_y=high_risk_y,
    )


def _collect_results(results_dir: Path) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for dataset_name in ("nba", "football", "multimodal"):
        results_path = results_dir / f"{dataset_name}_results.csv"
        if results_path.exists():
            table = pd.read_csv(results_path)
            required_columns = {"Dataset", "Model", "Recall", "Precision", "F1", "PR_AUC", "Threshold"}
            if not required_columns.issubset(table.columns):
                LOGGER.warning("Skipping legacy or incomplete results file: %s", results_path)
                continue
            tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _write_model_comparison(output_path: Path, combined: pd.DataFrame) -> None:
    best_rows = (
        combined.sort_values(["Dataset", "Recall"], ascending=[True, False])
        .groupby("Dataset")
        .head(1)
    )
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Model Comparison\n\n")
        handle.write("Generated from the latest standardized training artifacts.\n\n")
        handle.write("| Dataset | Model | Recall | Precision | F1 | PR-AUC | Threshold |\n")
        handle.write("| --- | --- | ---: | ---: | ---: | ---: | ---: |\n")
        for _, row in combined.sort_values(["Dataset", "Recall"], ascending=[True, False]).iterrows():
            handle.write(
                f"| {row['Dataset']} | {row['Model']} | {row['Recall']:.4f} | "
                f"{row['Precision']:.4f} | {row['F1']:.4f} | {row['PR_AUC']:.4f} | "
                f"{row['Threshold']:.2f} |\n"
            )

        handle.write("\n## Best Current Model Per Dataset\n\n")
        for _, row in best_rows.iterrows():
            handle.write(
                f"- **{row['Dataset']}**: {row['Model']} "
                f"(Recall {row['Recall']:.4f}, PR-AUC {row['PR_AUC']:.4f})\n"
            )


def _plot_model_comparison(output_path: Path, combined: pd.DataFrame) -> None:
    plt.figure(figsize=(11, 6))
    ordered = combined.sort_values(["Dataset", "Model"]).copy()
    labels = [f"{dataset}\n{model}" for dataset, model in zip(ordered["Dataset"], ordered["Model"])]
    plt.bar(labels, ordered["Recall"], color="#3e7cb1", alpha=0.9)
    plt.title("Multi-Sport Injury Prediction Recall by Dataset and Model")
    plt.ylabel("Recall")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_nba_feature_importance(project_root: Path, output_path: Path) -> None:
    feature_path = project_root / "results" / "artifacts" / "nba" / "feature_importance.csv"
    if not feature_path.exists():
        _save_placeholder_plot(
            output_path,
            "NBA Feature Importance",
            "Run the NBA pipeline to generate feature importance artifacts.",
        )
        return

    frame = pd.read_csv(feature_path).head(10)
    plt.figure(figsize=(9, 6))
    plt.barh(frame["Feature"], frame["Importance"], color="#c95d63")
    plt.gca().invert_yaxis()
    plt.title("NBA Feature Importance (Latest Best Tree Model)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _plot_multimodal_risk_zone(
    project_root: Path,
    output_path: Path,
    high_risk_x: str,
    high_risk_y: str,
) -> None:
    prediction_path = project_root / "results" / "artifacts" / "multimodal" / "best_model_predictions.csv"
    if not prediction_path.exists():
        _save_placeholder_plot(
            output_path,
            "Multimodal High-Risk Zone",
            "Run the multimodal pipeline to generate prediction artifacts.",
        )
        return

    frame = pd.read_csv(prediction_path)
    if high_risk_x not in frame.columns or high_risk_y not in frame.columns:
        _save_placeholder_plot(
            output_path,
            "Multimodal High-Risk Zone",
            f"Prediction artifacts did not include '{high_risk_x}' and '{high_risk_y}'.",
        )
        return

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        frame[high_risk_x],
        frame[high_risk_y],
        c=frame["y_true"],
        s=60 + frame["y_score"] * 140,
        alpha=0.75,
        cmap="coolwarm",
    )
    plt.colorbar(scatter, label="Observed Injury Risk")
    plt.xlabel(high_risk_x.replace("_", " ").title())
    plt.ylabel(high_risk_y.replace("_", " ").title())
    plt.title("Multimodal High-Risk Zone from Latest Predictions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _save_placeholder_plot(output_path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
