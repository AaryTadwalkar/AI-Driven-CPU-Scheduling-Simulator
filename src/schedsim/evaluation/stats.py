"""
1. What this file does:
   Implements rigorous statistical analysis, non-parametric bootstrap confidence intervals,
   pairwise algorithm win-rate matrices, fairness and starvation evaluations, and test-split
   Decision Tree regret assessments.
2. Overall design:
   Processes raw simulation records from evaluation_results.csv. Calculates mean ± 95% bootstrap
   confidence intervals across all performance metrics using 1,000 resamples. Computes per-scenario
   breakdowns, starvation rates (max waiting > 3x average waiting), Jain fairness distributions,
   and writes publication-ready summary tables in both CSV and formatted Markdown formats.
3. Elements in the file:
   - bootstrap_ci: Calculates non-parametric percentile bootstrap confidence intervals.
   - compute_summary_table: Generates aggregate table with means and 95% CIs per policy and cs_cost.
   - compute_win_rates: Computes pairwise head-to-head win, tie, and loss percentages.
   - compute_scenario_breakdown: Computes scenario-wise mean metrics across policies.
   - evaluate_decision_tree_test_split: Evaluates test accuracy, macro-F1, confusion matrix, regret.
   - export_summary_tables: Exports structured CSV and human-readable Markdown summary tables.
4. Final output of the file:
   Exports statistical computation and reporting functions for evaluation scripts and CLI runners.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier

import config
from schedsim.ml.evaluate_tree import compute_regret
from schedsim.ml.features import FEATURE_NAMES, extract_features
from schedsim.ml.labeling import CANDIDATE_POLICIES, compute_workload_label
from schedsim.workloads.io import Workload


def bootstrap_ci(
    data: Sequence[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    1. The use of function:
       Computes non-parametric percentile bootstrap confidence interval for the sample mean.
    2. How does the specific function work:
       Validates input data sequence. Draws n_resamples with replacement using a deterministic
       NumPy Generator. Calculates sample mean of each resample, finds (1-ci)/2 and 1-(1-ci)/2
       percentiles, and returns the empirical mean alongside lower and upper interval bounds.
    3. Concepts used in the function:
       Non-parametric bootstrap resampling, percentile interval estimation, deterministic RNG.
    4. Places where this function is imported and used:
       Imported by compute_summary_table, plotting modules, and evaluation test suites.
    5. Final output of the function, before and after:
       Before: Raw 1D sequence of float observations.
       After: Tuple of (sample_mean, ci_lower, ci_upper).
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        val = float(arr[0])
        return val, val, val

    rng = np.random.default_rng(seed)
    mean_val = float(np.mean(arr))
    indices = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    resample_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - ci
    ci_lower = float(np.percentile(resample_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(resample_means, 100.0 * (1.0 - alpha / 2.0)))

    return mean_val, ci_lower, ci_upper


def compute_summary_table(
    df: pd.DataFrame,
    metrics: Sequence[str] = (
        "avg_waiting",
        "avg_turnaround",
        "avg_response",
        "max_waiting",
        "makespan",
        "cpu_utilization",
        "context_switches",
        "jain_fairness",
        "composite_score",
    ),
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    1. The use of function:
       Aggregates simulation results into a summary table with bootstrap confidence intervals.
    2. How does the specific function work:
       Groups records by context switch cost and policy name. For every metric, calculates mean,
       lower CI, and upper CI via bootstrap_ci. Also computes starvation incidence rate.
       Constructs and returns a multi-metric summary DataFrame.
    3. Concepts used in the function:
       Dataframe grouping, statistical aggregation, confidence interval computation.
    4. Places where this function is imported and used:
       Imported by CLI evaluate runner and acceptance test suites.
    5. Final output of the function, before and after:
       Before: Raw per-workload execution DataFrame.
       After: Summary DataFrame with policy rows, metric means, and CI bounds.
    """
    rows: list[dict[str, Any]] = []
    costs = sorted(df["cs_cost"].unique())
    policies = list(df["policy"].unique())

    # Preferred presentation order
    canonical_order = ["FCFS", "SJF", "RR", "SRTF", "RL", "DT"]
    ordered_policies = [p for p in canonical_order if p in policies] + [
        p for p in policies if p not in canonical_order
    ]

    for cost in costs:
        cost_df = df[df["cs_cost"] == cost]
        for pol in ordered_policies:
            sub = cost_df[cost_df["policy"] == pol]
            if sub.empty:
                continue

            row: dict[str, Any] = {
                "cs_cost": cost,
                "policy": pol,
                "workload_count": len(sub),
            }

            for m in metrics:
                if m in sub.columns:
                    vals = sub[m].to_numpy(dtype=np.float64)
                    m_val, low, high = bootstrap_ci(
                        vals, n_resamples=n_resamples, seed=seed
                    )
                    row[f"{m}_mean"] = round(m_val, 3)
                    row[f"{m}_ci_lower"] = round(low, 3)
                    row[f"{m}_ci_upper"] = round(high, 3)
                    row[f"{m}_ci_half"] = round((high - low) / 2.0, 3)

            if "starvation" in sub.columns:
                starve_rate = float(sub["starvation"].mean() * 100.0)
                row["starvation_rate_pct"] = round(starve_rate, 2)

            rows.append(row)

    return pd.DataFrame(rows)


def compute_win_rates(
    df: pd.DataFrame,
    metric: str = "composite_score",
) -> pd.DataFrame:
    """
    1. The use of function:
       Computes pairwise head-to-head win, tie, and loss percentages across policies.
    2. How does the specific function work:
       Pivots the per-workload metric values by policy for each cs_cost. Computes pairwise
       comparison matrices (policy_A < policy_B for lower-is-better metrics).
    3. Concepts used in the function:
       Non-parametric pairwise tournament comparisons, tie handling.
    4. Places where this function is imported and used:
       Imported by report generator and summary documentation builders.
    5. Final output of the function, before and after:
       Before: Raw evaluation DataFrame.
       After: Pairwise win-rate DataFrame listing win, tie, and loss percentages.
    """
    records: list[dict[str, Any]] = []
    for cost in sorted(df["cs_cost"].unique()):
        cost_df = df[df["cs_cost"] == cost]
        pivot = cost_df.pivot(index="workload_id", columns="policy", values=metric)
        policies = list(pivot.columns)

        for p1 in policies:
            for p2 in policies:
                if p1 == p2:
                    continue
                valid = pivot[[p1, p2]].dropna()
                if valid.empty:
                    continue
                total = len(valid)
                wins = int((valid[p1] < valid[p2] - 1e-4).sum())
                ties = int((np.abs(valid[p1] - valid[p2]) <= 1e-4).sum())
                losses = int((valid[p1] > valid[p2] + 1e-4).sum())

                records.append({
                    "cs_cost": cost,
                    "policy_a": p1,
                    "policy_b": p2,
                    "metric": metric,
                    "win_pct": round(wins / total * 100.0, 2),
                    "tie_pct": round(ties / total * 100.0, 2),
                    "loss_pct": round(losses / total * 100.0, 2),
                })

    return pd.DataFrame(records)


def compute_scenario_breakdown(
    df: pd.DataFrame,
    metric: str = "avg_waiting",
) -> pd.DataFrame:
    """
    1. The use of function:
       Calculates mean metric performance broken down by workload scenario and policy.
    2. How does the specific function work:
       Groups evaluation data by context switch cost, scenario, and policy. Computes average
       metric values and pivots the table for scenario-wise algorithmic comparison.
    3. Concepts used in the function:
       Scenario-stratified benchmarking, performance profiling across workload regimes.
    4. Places where this function is imported and used:
       Imported by evaluation runner, plotting routines, and report notes.
    5. Final output of the function, before and after:
       Before: Flat evaluation DataFrame.
       After: Pivoted DataFrame with scenarios as rows and policies as columns.
    """
    grouped = (
        df.groupby(["cs_cost", "scenario", "policy"])[metric]
        .mean()
        .round(3)
        .reset_index()
    )
    return grouped


def evaluate_decision_tree_test_split(
    test_workloads: Sequence[Workload],
    model_path: Path | str | None = None,
    cs_cost: int = config.LABEL_CS_COST,
) -> dict[str, Any]:
    """
    1. The use of function:
       Conducts comprehensive supervised evaluation of the trained Decision Tree on test split.
    2. How does the specific function work:
       Computes ground-truth composite labels and extracts 12 workload features for test workloads.
       Evaluates accuracy, macro-averaged F1 score, confusion matrix, and average regret.
       Also computes regret for candidate baselines (Always-FCFS, Always-SJF, Always-RR).
    3. Concepts used in the function:
       Held-out test verification, multi-class classification metrics, decision regret analysis.
    4. Places where this function is imported and used:
       Imported by evaluation pipeline and acceptance tests.
    5. Final output of the function, before and after:
       Before: Sequence of test Workloads and trained model path.
       After: Structured dictionary with accuracy, macro_f1, confusion_matrix, and regret stats.
    """
    m_path = (
        Path(model_path)
        if model_path is not None
        else (config.MODELS_DIR / "decision_tree.joblib")
    )
    if not m_path.exists():
        raise FileNotFoundError(f"Model file not found: {m_path}")

    model: DecisionTreeClassifier = joblib.load(m_path)
    classes = list(model.classes_)

    x_list: list[list[float]] = []
    y_true: list[str] = []
    label_results = []

    for wl in test_workloads:
        feats = extract_features(wl.processes)
        x_list.append([feats[col] for col in FEATURE_NAMES])
        lbl_res = compute_workload_label(
            wl.processes,
            cs_cost=cs_cost,
            quantum=config.TIME_QUANTUM,
        )
        y_true.append(lbl_res.label)
        label_results.append(lbl_res)

    x_arr = np.array(x_list, dtype=np.float64)
    y_pred = [str(p) for p in model.predict(x_arr)]

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    # Calculate policy regret across test split
    df_scores = pd.DataFrame([
        {
            "score_fcfs": r.scores.get("FCFS", r.best_score),
            "score_sjf": r.scores.get("SJF", r.best_score),
            "score_rr": r.scores.get("RR", r.best_score),
        }
        for r in label_results
    ])
    dt_regret = compute_regret(df_scores, y_pred)
    baseline_regrets = {
        cand: compute_regret(df_scores, [cand] * len(df_scores))
        for cand in CANDIDATE_POLICIES
    }

    out = {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1, 4),
        "classes": classes,
        "confusion_matrix": cm,
        "regret": {
            "decision_tree": round(dt_regret, 4),
            "baselines": {k: round(v, 4) for k, v in baseline_regrets.items()},
        },
        "sample_count": len(test_workloads),
    }

    return out


def export_summary_tables(
    summary_df: pd.DataFrame,
    csv_path: Path | str | None = None,
    md_path: Path | str | None = None,
) -> None:
    """
    1. The use of function:
       Exports summary metrics to formatted CSV and publication-ready Markdown tables.
    2. How does the specific function work:
       Writes raw summary table to CSV. Generates a cleanly formatted Markdown table with
       mean ± half-CI notation for key metrics (waiting time, response time, turnaround time,
       context switches, and Jain index) segregated by context switch cost.
    3. Concepts used in the function:
       Markdown document generation, scientific result formatting.
    4. Places where this function is imported and used:
       Imported by cli.py evaluate and reporting utilities.
    5. Final output of the function, before and after:
       Before: Summary DataFrame and target file paths.
       After: Persisted results/summary_table.csv and results/summary_table.md.
    """
    c_path = (
        Path(csv_path) if csv_path is not None else (config.RESULTS_DIR / "summary_table.csv")
    )
    m_path = (
        Path(md_path) if md_path is not None else (config.RESULTS_DIR / "summary_table.md")
    )

    c_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(c_path, index=False)

    lines: list[str] = [
        "# Benchmark Summary: Classical, ML, and RL Schedulers",
        "",
        "Evaluation over held-out test split (1,000 workloads, 9 scenarios).",
        "Values reported as sample mean ± 95% bootstrap confidence interval (1,000 resamples).",
        "",
    ]

    for cost in sorted(summary_df["cs_cost"].unique()):
        lines.append(f"## Context Switch Cost = {cost}")
        lines.append("")
        lines.append(
            "| Policy | Avg Waiting (ticks) | Avg Response (ticks) | Avg Turnaround (ticks) "
            "| Context Switches | Jain Fairness | Starvation % |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|"
        )

        sub = summary_df[summary_df["cs_cost"] == cost]
        for _, row in sub.iterrows():
            pol = row["policy"]
            wait_str = f"{row['avg_waiting_mean']:.2f} ± {row['avg_waiting_ci_half']:.2f}"
            resp_str = f"{row['avg_response_mean']:.2f} ± {row['avg_response_ci_half']:.2f}"
            tat_str = f"{row['avg_turnaround_mean']:.2f} ± {row['avg_turnaround_ci_half']:.2f}"
            cs_str = (
                f"{row['context_switches_mean']:.1f} ± {row['context_switches_ci_half']:.1f}"
            )
            fair_str = (
                f"{row['jain_fairness_mean']:.3f} ± {row['jain_fairness_ci_half']:.3f}"
            )
            starve_str = f"{row.get('starvation_rate_pct', 0.0):.1f}%"

            lines.append(
                f"| **{pol}** | {wait_str} | {resp_str} | {tat_str} | {cs_str} "
                f"| {fair_str} | {starve_str} |"
            )
        lines.append("")

    with open(m_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


__all__ = [
    "bootstrap_ci",
    "compute_scenario_breakdown",
    "compute_summary_table",
    "compute_win_rates",
    "evaluate_decision_tree_test_split",
    "export_summary_tables",
]
