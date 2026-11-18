"""
1. What this file does:
   Generates publication-quality figures and visual artifacts for CPU scheduling evaluations,
   including comparison bar charts, Gantt timelines, Decision Tree diagnostics, and fairness plots.
2. Overall design:
   Reads aggregate summary metrics, scenario breakdowns, and test split diagnostic outputs.
   Uses Matplotlib to render crisp, publication-ready figures at 200 DPI with consistent styling,
   clear labels, and standard error bands. Outputs all generated charts to results/figures/.
3. Elements in the file:
   - plot_comparison_bars: Multi-metric bar charts with 95% bootstrap error bars.
   - plot_gantt_fixtures: Gantt execution charts for the reference workload across all policies.
   - plot_dt_confusion_matrix: Annotated heatmap of the Decision Tree test confusion matrix.
   - plot_dt_feature_importance: Ranked horizontal bar chart of Decision Tree feature importances.
   - plot_fairness_starvation: Dual-panel comparison of Jain fairness and starvation rates.
   - plot_scenario_breakdown: Heatmap comparing policy performance across all 9 workload scenarios.
   - generate_all_plots: Master function generating all evaluation figures deterministically.
4. Final output of the file:
   Exports plotting routines and writes figure PNG files to results/figures/ for report integration.
"""

from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import config
from schedsim.domain.engine import SimResult, run
from schedsim.domain.timeline import CS_LABEL, IDLE_LABEL
from schedsim.ml.features import FEATURE_NAMES
from schedsim.policies.registry import get_policy
from schedsim.rl.agent import QLearningAgent
from schedsim.rl.policy import RLPolicy
from schedsim.workloads.presets import get_preset

PALETTE = {
    "P1": "#1f77b4",
    "P2": "#ff7f0e",
    "P3": "#2ca02c",
    "P4": "#d62728",
    "P5": "#9467bd",
    "CS": "#7f7f7f",
    "IDLE": "#c7c7c7",
}

POLICY_COLORS = {
    "FCFS": "#4c72b0",
    "SJF": "#55a868",
    "RR": "#c44e52",
    "SRTF": "#8172b3",
    "RL": "#ccb974",
    "DT": "#64b5cd",
}


def plot_comparison_bars(
    summary_df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> Path:
    """
    1. The use of function:
       Generates multi-metric comparison bar charts with 95% bootstrap error bars.
    2. How does the specific function work:
       Creates a 2x2 grid of subplots for cs_cost=0 and cs_cost=1 across Average Waiting Time
       and Average Response Time. Plots policy bars with symmetric bootstrap confidence intervals.
    3. Concepts used in the function:
       Subplot grid layout, categorical bar charts with empirical error bars, aesthetic styling.
    4. Places where this function is imported and used:
       Imported by generate_all_plots, evaluation runner, and CLI evaluate command.
    5. Final output of the function, before and after:
       Before: Summary DataFrame with metric means and half-CI values.
       After: Persisted results/figures/comparison_bars.png file.
    """
    out_file = (
        Path(output_path)
        if output_path is not None
        else (config.RESULTS_DIR / "figures" / "comparison_bars.png")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    metrics_to_plot = [
        ("avg_waiting", "Average Waiting Time (ticks)", "Waiting Time"),
        ("avg_response", "Average Response Time (ticks)", "Response Time"),
    ]

    costs = sorted(summary_df["cs_cost"].unique())
    if not costs:
        costs = [0, 1]

    for col_idx, cost in enumerate(costs[:2]):
        sub = summary_df[summary_df["cs_cost"] == cost]
        policies = list(sub["policy"])
        x = np.arange(len(policies))

        for row_idx, (metric_key, ylabel, title_prefix) in enumerate(metrics_to_plot):
            ax = axes[row_idx, col_idx]
            means = sub[f"{metric_key}_mean"].to_numpy()
            yerrs = sub[f"{metric_key}_ci_half"].to_numpy()
            colors = [POLICY_COLORS.get(p, "#333333") for p in policies]

            bars = ax.bar(
                x,
                means,
                yerr=yerrs,
                capsize=4,
                color=colors,
                edgecolor="black",
                linewidth=0.8,
                alpha=0.88,
            )

            for bar, val in zip(bars, means, strict=False):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() * 0.5,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color="white" if val > 2 else "black",
                    fontweight="bold",
                    fontsize=9,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(policies, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(
                f"{title_prefix} (cs_cost = {cost})",
                fontsize=12,
                fontweight="bold",
            )
            ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


def plot_gantt_fixtures(
    output_path: Path | str | None = None,
    fixture_name: str = "section_4_6",
    q_table_cs0_path: Path | str | None = None,
    q_table_cs1_path: Path | str | None = None,
) -> Path:
    """
    1. The use of function:
       Renders stacked Gantt timeline execution charts for the reference workload fixture.
    2. How does the specific function work:
       Loads the Section 4.6 reference fixture. Executes FCFS, SJF, RR(2), SRTF, and RL
       at cs_cost=0 and cs_cost=1. Draws horizontal broken bars corresponding to timeline segments,
       with distinct colors for PIDs, hatched patterns for context switches, and time tick markers.
    3. Concepts used in the function:
       Gantt chart visualization, timeline segmentation, execution interval representation.
    4. Places where this function is imported and used:
       Imported by generate_all_plots, evaluation runner, and report figure generator.
    5. Final output of the function, before and after:
       Before: Preset fixture name and model paths.
       After: Persisted results/figures/gantt_fixtures.png file.
    """
    out_file = (
        Path(output_path)
        if output_path is not None
        else (config.RESULTS_DIR / "figures" / "gantt_fixtures.png")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    procs = get_preset(fixture_name)
    p_cs0 = (
        Path(q_table_cs0_path)
        if q_table_cs0_path is not None
        else (config.MODELS_DIR / "q_table_cs0.pkl")
    )
    p_cs1 = (
        Path(q_table_cs1_path)
        if q_table_cs1_path is not None
        else (config.MODELS_DIR / "q_table_cs1.pkl")
    )

    agent_cs0 = QLearningAgent.load(p_cs0) if p_cs0.exists() else None
    agent_cs1 = QLearningAgent.load(p_cs1) if p_cs1.exists() else None

    policies_to_run = ["FCFS", "SJF", "RR", "SRTF"]
    if agent_cs0 is not None:
        policies_to_run.append("RL")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, cost, agent in [(ax0, 0, agent_cs0), (ax1, 1, agent_cs1)]:
        y_labels: list[str] = []
        for y_idx, pol_name in enumerate(reversed(policies_to_run)):
            y_labels.append(pol_name)
            if pol_name == "RL" and agent is not None:
                p = RLPolicy(agent, processes=procs, name="RLPolicy")
            elif pol_name == "RR":
                p = get_policy("rr", quantum=config.TIME_QUANTUM)
            else:
                p = get_policy(pol_name.lower())

            sim_res: SimResult = run(p, procs, cs_cost=cost)
            segments = sim_res.timeline.to_segments()

            for seg in segments:
                duration = seg.end - seg.start
                color = PALETTE.get(seg.label, "#999999")
                hatch = "///" if seg.label == CS_LABEL else None
                ax.barh(
                    y_idx,
                    duration,
                    left=seg.start,
                    height=0.6,
                    color=color,
                    edgecolor="black",
                    linewidth=0.8,
                    hatch=hatch,
                )
                if duration >= 1:
                    ax.text(
                        seg.start + duration / 2.0,
                        y_idx,
                        seg.label,
                        ha="center",
                        va="center",
                        color="white" if seg.label not in (IDLE_LABEL, CS_LABEL) else "black",
                        fontsize=9,
                        fontweight="bold",
                    )

        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontweight="bold", fontsize=11)
        ax.set_title(
            f"Execution Timeline on Reference Fixture (cs_cost = {cost})",
            fontweight="bold",
            fontsize=12,
        )
        ax.grid(axis="x", linestyle=":", alpha=0.6)

    ax1.set_xlabel("Simulation Time (discrete integer ticks)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


def plot_dt_confusion_matrix(
    dt_test_metrics: dict[str, Any],
    output_path: Path | str | None = None,
) -> Path:
    """
    1. The use of function:
       Renders an annotated heatmap of the Decision Tree confusion matrix on the test split.
    2. How does the specific function work:
       Extracts confusion matrix and class labels from metrics dictionary. Normalizes counts
       row-wise to obtain class-conditional percentages. Draws a color-mapped heatmap with cell
       annotations showing raw counts and row percentages.
    3. Concepts used in the function:
       Confusion matrix visualization, classification error analysis, color mapping.
    4. Places where this function is imported and used:
       Imported by generate_all_plots and supervised evaluation pipelines.
    5. Final output of the function, before and after:
       Before: Test metrics dictionary containing confusion matrix array.
       After: Persisted results/figures/dt_confusion_matrix.png file.
    """
    out_file = (
        Path(output_path)
        if output_path is not None
        else (config.RESULTS_DIR / "figures" / "dt_confusion_matrix.png")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cm = np.array(dt_test_metrics["confusion_matrix"], dtype=np.int64)
    classes = dt_test_metrics["classes"]

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Normalized Proportion", rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Decision Tree Confusion Matrix (Held-out Test Split)",
        ylabel="True Best Policy",
        xlabel="Predicted Policy",
    )

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100.0
            text = f"{val}\n({pct:.1f}%)"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontweight="bold",
                fontsize=10,
            )

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


def plot_dt_feature_importance(
    model_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """
    1. The use of function:
       Renders a ranked horizontal bar chart of Decision Tree Gini feature importances.
    2. How does the specific function work:
       Loads the trained DecisionTree model from model_path. Extracts feature importances and
       feature names from FEATURE_NAMES. Sorts features in ascending order of importance and
       draws horizontal bars with numerical percentage labels.
    3. Concepts used in the function:
       Feature importance ranking, tree explainability, horizontal bar chart styling.
    4. Places where this function is imported and used:
       Imported by generate_all_plots and ML diagnostic routines.
    5. Final output of the function, before and after:
       Before: Model checkpoint file path.
       After: Persisted results/figures/dt_feature_importance.png file.
    """
    out_file = (
        Path(output_path)
        if output_path is not None
        else (config.RESULTS_DIR / "figures" / "dt_feature_importance.png")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    m_path = (
        Path(model_path)
        if model_path is not None
        else (config.MODELS_DIR / "decision_tree.joblib")
    )
    model: DecisionTreeClassifier = joblib.load(m_path)

    importances = model.feature_importances_
    names = list(FEATURE_NAMES)

    indices = np.argsort(importances)
    sorted_names = [names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(
        range(len(sorted_names)),
        sorted_importances * 100.0,
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.7,
        alpha=0.85,
    )

    for bar, val in zip(bars, sorted_importances * 100.0, strict=False):
        if val > 0.5:
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:.1f}%",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel("Relative Importance (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Decision Tree Feature Importances (Gini Impurity)",
        fontweight="bold",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


def plot_fairness_starvation(
    summary_df: pd.DataFrame,
    output_path: Path | str | None = None,
) -> Path:
    """
    1. The use of function:
       Renders dual-panel bar charts comparing Jain Fairness Index and Starvation Rates.
    2. How does the specific function work:
       Extracts jain_fairness and starvation_rate_pct columns from summary_df. Plots two
       side-by-side bar charts across policies at cs_cost=1, contrasting fairness with starvation.
    3. Concepts used in the function:
       Fairness versus performance trade-off visualization, starvation incidence tracking.
    4. Places where this function is imported and used:
       Imported by generate_all_plots and evaluation report visualizers.
    5. Final output of the function, before and after:
       Before: Summary DataFrame with fairness and starvation statistics.
       After: Persisted results/figures/fairness_starvation.png file.
    """
    out_file = (
        Path(output_path)
        if output_path is not None
        else (config.RESULTS_DIR / "figures" / "fairness_starvation.png")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sub = summary_df[summary_df["cs_cost"] == 1]
    if sub.empty:
        sub = summary_df

    policies = list(sub["policy"])
    x = np.arange(len(policies))
    colors = [POLICY_COLORS.get(p, "#333333") for p in policies]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Jain Fairness Index
    jain_means = sub["jain_fairness_mean"].to_numpy()
    jain_errs = sub["jain_fairness_ci_half"].to_numpy()
    ax1.bar(
        x,
        jain_means,
        yerr=jain_errs,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies, fontweight="bold")
    ax1.set_ylabel("Jain's Fairness Index", fontsize=11)
    ax1.set_title("Slowdown Fairness Index (cs_cost = 1)", fontweight="bold", fontsize=12)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Starvation Rate %
    starve_vals = sub["starvation_rate_pct"].to_numpy()
    ax2.bar(
        x,
        starve_vals,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.85,
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, fontweight="bold")
    ax2.set_ylabel("Workloads with Starvation (%)", fontsize=11)
    ax2.set_title(
        "Starvation Incidence (Max Wait > 3x Avg Wait)",
        fontweight="bold",
        fontsize=12,
    )
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(ax2.patches, starve_vals, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


def plot_scenario_breakdown(
    scenario_df: pd.DataFrame,
    output_path: Path | str | None = None,
    metric: str = "avg_waiting",
) -> Path:
    """
    1. The use of function:
       Renders an annotated heatmap of policy performance across all 9 workload scenarios.
    2. How does the specific function work:
       Filters scenario_df for cs_cost=1. Pivots table with scenarios as rows and policies
       as columns. Renders a color-mapped heatmap annotated with mean values in each cell.
    3. Concepts used in the function:
       Scenario-stratified comparison heatmap, domain-specific behavior profiling.
    4. Places where this function is imported and used:
       Imported by generate_all_plots and report analysis builders.
    5. Final output of the function, before and after:
       Before: Grouped scenario DataFrame.
       After: Persisted results/figures/scenario_breakdown.png file.
    """
    out_file = (
        Path(output_path)
        if output_path is not None
        else (config.RESULTS_DIR / "figures" / "scenario_breakdown.png")
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)

    sub = scenario_df[scenario_df["cs_cost"] == 1]
    if sub.empty:
        sub = scenario_df

    pivot = sub.pivot(index="scenario", columns="policy", values=metric)
    scenarios = list(pivot.index)
    policies = list(pivot.columns)
    matrix = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(f"Mean {metric} (ticks)", rotation=-90, va="bottom")

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, fontweight="bold", fontsize=10)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=10)
    ax.set_title(
        f"Scenario-wise {metric} Comparison (cs_cost = 1)",
        fontweight="bold",
        fontsize=12,
    )

    thresh = (matrix.max() + matrix.min()) / 2.0
    for i in range(len(scenarios)):
        for j in range(len(policies)):
            val = matrix[i, j]
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
                fontsize=9,
            )

    plt.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    return out_file


def generate_all_plots(
    summary_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    dt_test_metrics: dict[str, Any],
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """
    1. The use of function:
       Orchestrates batch generation of all evaluation figures for final report and dashboard.
    2. How does the specific function work:
       Resolves output directory, invokes individual plotting functions in sequence,
       collects resulting file paths, and returns a dictionary of generated figure artifacts.
    3. Concepts used in the function:
       Batch visual artifact generation, publication pipeline management.
    4. Places where this function is imported and used:
       Imported by cli.py (evaluate command) and Stage 7 acceptance tests.
    5. Final output of the function, before and after:
       Before: Evaluation metrics DataFrames and dictionaries.
       After: Complete set of 5 publication-ready PNG figures in results/figures/.
    """
    out_d = Path(output_dir) if output_dir is not None else (config.RESULTS_DIR / "figures")
    out_d.mkdir(parents=True, exist_ok=True)

    generated: dict[str, Path] = {}
    generated["comparison_bars"] = plot_comparison_bars(
        summary_df, out_d / "comparison_bars.png"
    )
    generated["gantt_fixtures"] = plot_gantt_fixtures(
        out_d / "gantt_fixtures.png"
    )
    generated["dt_confusion_matrix"] = plot_dt_confusion_matrix(
        dt_test_metrics, out_d / "dt_confusion_matrix.png"
    )
    generated["dt_feature_importance"] = plot_dt_feature_importance(
        output_path=out_d / "dt_feature_importance.png"
    )
    generated["fairness_starvation"] = plot_fairness_starvation(
        summary_df, out_d / "fairness_starvation.png"
    )
    generated["scenario_breakdown"] = plot_scenario_breakdown(
        scenario_df, out_d / "scenario_breakdown.png"
    )

    return generated


__all__ = [
    "generate_all_plots",
    "plot_comparison_bars",
    "plot_dt_confusion_matrix",
    "plot_dt_feature_importance",
    "plot_fairness_starvation",
    "plot_gantt_fixtures",
    "plot_scenario_breakdown",
]
