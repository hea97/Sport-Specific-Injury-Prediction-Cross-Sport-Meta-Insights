"""Reporting and visualization generation."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np

LOGGER = logging.getLogger(__name__)
PREFERRED_FONT_FAMILIES = (
    "Malgun Gothic",
    "AppleGothic",
    "NanumGothic",
    "Noto Sans CJK KR",
    "DejaVu Sans",
)
DATASET_ORDER = ["NBA", "Football", "Multimodal"]
MODEL_ORDER = ["RF", "XGB", "LGB", "MLP"]
MODEL_COLORS = {
    "RF": "#4F5D7A",
    "XGB": "#3B7080",
    "LGB": "#4D8B73",
    "MLP": "#9ABF68",
}
_FONT_CONFIGURED = False


def refresh_reports(project_root: Path, high_risk_x: str, high_risk_y: str) -> None:
    """Regenerate comparison markdown and insights plots from latest artifacts."""

    _configure_matplotlib_fonts()

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
    _configure_matplotlib_fonts()

    pivoted, best_models, lower_bound = _prepare_model_comparison(combined)
    if pivoted.empty:
        _save_placeholder_plot(
            output_path,
            "데이터셋별 모델 Recall 비교",
            "Run at least one standardized experiment to generate comparison data.",
        )
        return

    figure, axis = plt.subplots(figsize=(11.5, 6.8))
    positions = np.arange(len(pivoted.index), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, num=len(MODEL_ORDER))

    for model_name, offset in zip(MODEL_ORDER, offsets):
        recalls = pivoted[model_name].to_numpy(dtype=float)
        bar_positions = positions + offset
        colors = []
        edge_colors = []
        line_widths = []
        for dataset_name, recall_value in zip(pivoted.index, recalls):
            if np.isnan(recall_value):
                colors.append((0.0, 0.0, 0.0, 0.0))
                edge_colors.append((0.0, 0.0, 0.0, 0.0))
                line_widths.append(0.0)
                continue

            is_best_model = best_models.loc[dataset_name] == model_name
            if is_best_model:
                colors.append(MODEL_COLORS[model_name])
                edge_colors.append("#25313D")
                line_widths.append(1.3)
            else:
                colors.append(_blend_with_white(MODEL_COLORS[model_name], blend_ratio=0.72))
                edge_colors.append(_blend_with_white(MODEL_COLORS[model_name], blend_ratio=0.58))
                line_widths.append(0.8)

        bars = axis.bar(
            bar_positions,
            recalls,
            width=width,
            color=colors,
            edgecolor=edge_colors,
            linewidth=line_widths,
            zorder=3,
        )

        for dataset_name, recall_value, bar in zip(pivoted.index, recalls, bars):
            if np.isnan(recall_value) or best_models.loc[dataset_name] != model_name:
                continue
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                recall_value + 0.012,
                f"{recall_value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#25313D",
            )

    axis.set_title("데이터셋별 모델 Recall 비교", fontsize=17, pad=14, fontweight="bold")
    axis.set_ylabel("Recall", fontsize=11)
    axis.set_xticks(positions)
    axis.set_xticklabels(pivoted.index, fontsize=10)
    axis.set_ylim(lower_bound, 1.0)
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.9, alpha=0.9, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    legend_handles = [Patch(facecolor=MODEL_COLORS[model_name], edgecolor="none", label=model_name) for model_name in MODEL_ORDER]
    axis.legend(
        handles=legend_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
    )

    figure.text(
        0.5,
        0.02,
        "동일한 모델이라도 데이터셋 구조와 품질에 따라 성능 차이가 크게 나타났다.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#475467",
    )
    figure.tight_layout(rect=(0.0, 0.06, 0.88, 0.97))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


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
    _configure_matplotlib_fonts()
    plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _prepare_model_comparison(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    ordered = combined.copy()
    ordered["Dataset"] = pd.Categorical(ordered["Dataset"], categories=DATASET_ORDER, ordered=True)
    ordered["Model"] = pd.Categorical(ordered["Model"], categories=MODEL_ORDER, ordered=True)
    ordered = ordered.sort_values(["Dataset", "Model"]).dropna(subset=["Dataset", "Model"])

    pivoted = ordered.pivot_table(
        index="Dataset",
        columns="Model",
        values="Recall",
        aggfunc="max",
        observed=False,
    )
    pivoted = pivoted.reindex(DATASET_ORDER).dropna(how="all")
    pivoted = pivoted.reindex(columns=MODEL_ORDER)
    if pivoted.empty:
        return pivoted, pd.Series(dtype="object"), 0.0

    best_models = pivoted.idxmax(axis=1)
    lower_bound = _comparison_y_axis_lower_bound(pivoted)
    return pivoted, best_models, lower_bound


def _comparison_y_axis_lower_bound(pivoted: pd.DataFrame) -> float:
    min_recall = float(np.nanmin(pivoted.to_numpy(dtype=float)))
    if math.isnan(min_recall):
        return 0.25
    return 0.25


def _blend_with_white(color: str, blend_ratio: float) -> tuple[float, float, float]:
    red, green, blue = mcolors.to_rgb(color)
    return (
        red + (1.0 - red) * blend_ratio,
        green + (1.0 - green) * blend_ratio,
        blue + (1.0 - blue) * blend_ratio,
    )


def _configure_matplotlib_fonts() -> None:
    global _FONT_CONFIGURED
    if _FONT_CONFIGURED:
        return

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for family in PREFERRED_FONT_FAMILIES:
        if family in available_fonts:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_CONFIGURED = True
