"""
1. What this file does:
   Implements evaluation routines, metric benchmarking, and visualization plotting for the
   reinforcement learning scheduling agent.
2. Overall design:
   - evaluate_agent_on_workloads: Computes system-wide performance metrics (waiting, turnaround,
     response, makespan, context switches, unseen state rate) across arbitrary workload collections.
   - plot_learning_curves: Generates validation learning curve plots (mean +/- std across seeds)
     from training log CSVs, saving publication-ready figures in results/figures/.
   - plot_action_distribution: Analyzes state summary JSON records and visualizes rule action
     preference proportions across key queue states.
3. Elements in the file:
   - evaluate_agent_on_workloads: Detailed benchmark evaluator.
   - plot_learning_curves: Matplotlib learning curve generator.
   - plot_action_distribution: Policy action distribution visualizer.
4. Final output of the file:
   Produces results/figures/rl_learning_curve.png and exports evaluation functions.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config  # noqa: E402
from schedsim.domain.process import Process  # noqa: E402
from schedsim.rl.actions import ACTION_NAMES, ALL_ACTIONS  # noqa: E402
from schedsim.rl.agent import QLearningAgent  # noqa: E402
from schedsim.rl.env import Environment  # noqa: E402
from schedsim.workloads.io import Workload  # noqa: E402


def evaluate_agent_on_workloads(
    agent: QLearningAgent,
    workloads: Sequence[Workload | Sequence[Process]],
    cs_cost: int = 1,
) -> dict[str, float]:
    """
    1. The use of function:
       Computes detailed multi-metric scheduling statistics for a greedy agent across workloads.
    2. How does the specific function work:
       Resets evaluation tracking on agent. Iterates through each workload using Environment,
       stepping greedily via agent.best_action(record_unseen=True). Aggregates per-workload
       simulation metrics (avg_waiting, avg_turnaround, avg_response, makespan, context_switches),
       and computes mean statistics alongside unseen state rate.
    3. Concepts used in the function:
       Greedy policy evaluation, comprehensive metric aggregation, multi-workload benchmarking.
    4. Places where this function is imported and used:
       Called by evaluation scripts, comparison benchmarks, and test suites.
    5. Final output of the function, before and after:
       Before: QLearningAgent and list of workloads.
       After: Returns dictionary of float metrics.
    """
    agent.reset_eval_tracking()
    env = Environment(cs_cost=cs_cost)

    waitings: list[float] = []
    turnarounds: list[float] = []
    responses: list[float] = []
    switches: list[int] = []
    makespans: list[int] = []

    for item in workloads:
        procs = item.processes if isinstance(item, Workload) else item
        env.reset(procs)

        while not env.done:
            valid = env.valid_actions()
            assert env.state is not None
            action = agent.best_action(env.state, valid, record_unseen=True)
            env.step(action)

        res = env.result()
        m = res.simulation_metrics
        waitings.append(m.avg_waiting)
        turnarounds.append(m.avg_turnaround)
        responses.append(m.avg_response)
        switches.append(m.context_switches)
        makespans.append(m.makespan)

    n = len(workloads) if workloads else 1
    return {
        "avg_waiting": round(float(np.mean(waitings)), 4) if waitings else 0.0,
        "avg_turnaround": round(float(np.mean(turnarounds)), 4) if turnarounds else 0.0,
        "avg_response": round(float(np.mean(responses)), 4) if responses else 0.0,
        "avg_context_switches": round(float(np.mean(switches)), 4) if switches else 0.0,
        "avg_makespan": round(float(np.mean(makespans)), 4) if makespans else 0.0,
        "unseen_state_rate": round(agent.unseen_state_rate, 4),
        "total_workloads": float(n),
    }


def plot_learning_curves(
    logs_csv_path: Path | str | None = None,
    output_png_path: Path | str | None = None,
) -> Path:
    """
    1. The use of function:
       Renders publication-quality RL validation learning curve plots across training episodes.
    2. How does the specific function work:
       Loads rl_training_logs.csv containing multi-seed trajectory rows. Groups records by
       cs_cost and episode, computing mean and standard deviation of validation avg_waiting.
       Plots curves with shaded +/- 1 std error regions. Formats axes, legends, and styling,
       and saves image to results/figures/rl_learning_curve.png.
    3. Concepts used in the function:
       Learning curve visualization, empirical variance shading, headless matplotlib plotting.
    4. Places where this function is imported and used:
       Called after training runs and by CLI evaluation subcommands.
    5. Final output of the function, before and after:
       Before: Training logs on disk.
       After: Saves PNG figure and returns Path to generated image.
    """
    in_csv = Path(logs_csv_path) if logs_csv_path else config.DATA_DIR / "rl_training_logs.csv"
    if not in_csv.exists():
        raise FileNotFoundError(f"Training logs not found at: {in_csv}")

    df = pd.read_csv(in_csv)
    out_png = (
        Path(output_png_path)
        if output_png_path
        else config.RESULTS_DIR / "figures" / "rl_learning_curve.png"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)

    colors = {0: "#1f77b4", 1: "#d62728"}
    labels = {0: "Context Switch Cost = 0", 1: "Context Switch Cost = 1"}

    for cs_cost, group in df.groupby("cs_cost"):
        stats = group.groupby("episode")["avg_waiting"].agg(["mean", "std"]).reset_index()
        episodes = stats["episode"].values
        means = stats["mean"].values
        stds = stats["std"].fillna(0.0).values

        color = colors.get(int(cs_cost), "#333333")
        label = labels.get(int(cs_cost), f"cs_cost={cs_cost}")

        ax.plot(episodes, means, label=label, color=color, linewidth=2)
        ax.fill_between(
            episodes,
            means - stds,
            means + stds,
            color=color,
            alpha=0.18,
        )

    ax.set_title("Q-Learning Validation Learning Curve", fontsize=13, fontweight="bold")
    ax.set_xlabel("Training Episodes", fontsize=11)
    ax.set_ylabel("Validation Mean Waiting Time (ticks)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return out_png


def plot_action_distribution(
    summary_json_path: Path | str | None = None,
    output_png_path: Path | str | None = None,
) -> Path:
    """
    1. The use of function:
       Renders a bar chart visualizing optimal action preferences among top visited states.
    2. How does the specific function work:
       Loads q_table_summary.json, aggregates the frequency of best_action_name choices across
       the top visited states, normalizes into percentage shares, and renders a stylized bar chart.
    3. Concepts used in the function:
       Policy distribution analysis, categorical frequency visualization, bar charts.
    4. Places where this function is imported and used:
       Called during report preparation, figure export, and acceptance checks.
    5. Final output of the function, before and after:
       Before: State summary JSON file.
       After: Saves bar chart PNG to results/figures/rl_action_distribution.png and returns Path.
    """
    in_json = (
        Path(summary_json_path)
        if summary_json_path
        else config.MODELS_DIR / "q_table_summary.json"
    )
    if not in_json.exists():
        raise FileNotFoundError(f"Summary JSON not found at: {in_json}")

    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    top_states = data.get("top_states", [])
    action_counts = {ACTION_NAMES[a]: 0 for a in ALL_ACTIONS}
    for item in top_states:
        name = item.get("best_action_name")
        if name in action_counts:
            action_counts[name] += 1

    total = max(len(top_states), 1)
    action_pcts = {k: (v / total) * 100 for k, v in action_counts.items()}

    out_png = (
        Path(output_png_path)
        if output_png_path
        else config.RESULTS_DIR / "figures" / "rl_action_distribution.png"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    names = list(action_pcts.keys())
    pcts = [action_pcts[n] for n in names]
    bar_colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd"]

    bars = ax.bar(names, pcts, color=bar_colors[: len(names)], edgecolor="#333333", width=0.55)

    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.annotate(
            f"{pct:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title(
        f"Optimal Action Shares Across Top {len(top_states)} States",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Share of Top States (%)", fontsize=10)
    ax.set_ylim(0, max(pcts, default=10) * 1.25)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return out_png


__all__ = [
    "evaluate_agent_on_workloads",
    "plot_action_distribution",
    "plot_learning_curves",
]
