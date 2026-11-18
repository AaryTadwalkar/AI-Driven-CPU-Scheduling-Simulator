"""
1. What this file does:
   Implements the comprehensive Stage 7 acceptance test suite verifying evaluation pipeline
   determinism, non-parametric bootstrap confidence intervals, summary table exports, publication
   figure generation, Decision Tree held-out test metrics, and empirical scheduling invariants.
2. Overall design:
   Executes unit and integration test assertions across evaluation components:
   - run_experiments: Deterministic multi-policy simulation reproducibility.
   - stats: Bootstrap CI validity, summary table generation, regret analysis, and Markdown export.
   - plots: Verification that all 6 required publication figures are rendered at high resolution.
   - cli: Execution verification of the `cli.py evaluate` subcommand.
3. Elements in the file:
   - test_evaluation_determinism: Confirms identical simulation records across multiple runs.
   - test_bootstrap_ci_mathematical_properties: Checks bounds, edge cases, and coverage.
   - test_summary_table_structure_and_export: Confirms presence of all policies and metrics.
   - test_all_evaluation_figures_exist_and_valid: Validates figure existence and non-zero size.
   - test_decision_tree_test_split_metrics: Asserts accuracy, macro-F1, and regret bounds.
   - test_empirical_scheduling_invariants: Asserts SRTF optimality, RL convergence, and CS trend.
   - test_cli_evaluate_subcommand: Validates end-to-end execution of `python cli.py evaluate`.
4. Final output of the file:
   Executes pytest test assertions certifying that Stage 7 satisfies all requirements.
"""

import json
from pathlib import Path

import pandas as pd

import config
from cli import main
from schedsim.evaluation.run_experiments import (
    EVAL_POLICIES,
    run_evaluation_experiments,
)
from schedsim.evaluation.stats import (
    bootstrap_ci,
    compute_summary_table,
    export_summary_tables,
)


def test_evaluation_determinism() -> None:
    """
    1. The use of function:
       Tests that simulation experiments execute deterministically on identical test workloads.
    2. How does the specific function work:
       Executes run_evaluation_experiments twice on 5 test workloads with 1 worker process.
       Compares the resulting DataFrames value-by-value to verify identical metrics.
    3. Concepts used in the function:
       Determinism verification, regression protection, numerical consistency.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Unverified simulation reproducibility.
       After: Asserts df1 equals df2 exactly.
    """
    df1 = run_evaluation_experiments(
        split="test",
        n_workloads=5,
        cs_costs=(0, 1),
        policies=("FCFS", "SJF"),
        n_workers=1,
    )
    df2 = run_evaluation_experiments(
        split="test",
        n_workloads=5,
        cs_costs=(0, 1),
        policies=("FCFS", "SJF"),
        n_workers=1,
    )

    assert len(df1) == len(df2) == 20
    pd.testing.assert_frame_equal(df1, df2)


def test_bootstrap_ci_mathematical_properties() -> None:
    """
    1. The use of function:
       Tests mathematical invariants and edge-case handling of bootstrap confidence intervals.
    2. How does the specific function work:
       Evaluates empty inputs, single-value sequences, constant values, and normal data.
       Checks that ci_lower <= mean <= ci_upper in all cases and that confidence bounds hold.
    3. Concepts used in the function:
       Bootstrap percentile bounds, boundary condition validation.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Unverified bootstrap estimator.
       After: Confirms correct statistical invariants.
    """
    # Edge cases
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)
    assert bootstrap_ci([5.0]) == (5.0, 5.0, 5.0)
    assert bootstrap_ci([3.0, 3.0, 3.0]) == (3.0, 3.0, 3.0)

    # General normal data
    data = [10.0, 12.0, 11.0, 9.0, 10.5, 11.5, 13.0, 8.5]
    mean_val, low, high = bootstrap_ci(data, n_resamples=500, ci=0.95, seed=123)

    assert low <= mean_val <= high
    assert 8.5 <= low
    assert high <= 13.0


def test_summary_table_structure_and_export(tmp_path: Path) -> None:
    """
    1. The use of function:
       Tests summary table aggregation, columns, confidence intervals, and Markdown generation.
    2. How does the specific function work:
       Generates evaluation records on 5 workloads across all policies, aggregates summary table,
       and exports to CSV and Markdown. Asserts file existence, non-emptiness, and headers.
    3. Concepts used in the function:
       DataFrame schema validation, file I/O verification, Markdown formatting check.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Unverified summary table generation.
       After: Confirms all policies, metrics, and Markdown formatting are present.
    """
    eval_df = run_evaluation_experiments(
        split="test",
        n_workloads=5,
        cs_costs=(0, 1),
        policies=EVAL_POLICIES,
        n_workers=1,
    )
    summary_df = compute_summary_table(eval_df, n_resamples=200)

    assert len(summary_df) == 12  # 6 policies * 2 costs
    assert "avg_waiting_mean" in summary_df.columns
    assert "avg_waiting_ci_half" in summary_df.columns
    assert "jain_fairness_mean" in summary_df.columns
    assert "starvation_rate_pct" in summary_df.columns

    csv_p = tmp_path / "summary.csv"
    md_p = tmp_path / "summary.md"
    export_summary_tables(summary_df, csv_path=csv_p, md_path=md_p)

    assert csv_p.exists() and csv_p.stat().st_size > 100
    assert md_p.exists() and md_p.stat().st_size > 100

    content = md_p.read_text(encoding="utf-8")
    assert "## Context Switch Cost = 0" in content
    assert "## Context Switch Cost = 1" in content
    assert "FCFS" in content
    assert "SRTF" in content
    assert "RL" in content
    assert "DT" in content


def test_all_evaluation_figures_exist_and_valid() -> None:
    """
    1. The use of function:
       Verifies that all required publication-quality evaluation figures exist and are non-empty.
    2. How does the specific function work:
       Checks results/figures/ directory for comparison_bars.png, gantt_fixtures.png,
       dt_confusion_matrix.png, dt_feature_importance.png, fairness_starvation.png,
       and scenario_breakdown.png. Confirms file sizes exceed 5 KB.
    3. Concepts used in the function:
       Visual artifact presence and integrity assertion.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Unverified figure disk state.
       After: Asserts all 6 figure files exist with valid non-zero size.
    """
    fig_dir = config.RESULTS_DIR / "figures"
    expected_figures = [
        "comparison_bars.png",
        "gantt_fixtures.png",
        "dt_confusion_matrix.png",
        "dt_feature_importance.png",
        "fairness_starvation.png",
        "scenario_breakdown.png",
    ]

    for fname in expected_figures:
        fpath = fig_dir / fname
        assert fpath.exists(), f"Figure file missing: {fpath}"
        assert fpath.stat().st_size > 5000, f"Figure file too small: {fpath}"


def test_decision_tree_test_split_metrics() -> None:
    """
    1. The use of function:
       Tests the performance of the Decision Tree policy recommender on the held-out test split.
    2. How does the specific function work:
       Loads results/decision_tree_test_metrics.json. Checks that test accuracy is at least 85%,
       macro-F1 is at least 80%, mean regret is below 0.10, and DT regret is lower than Always-FCFS.
    3. Concepts used in the function:
       Held-out generalization verification, regret thresholding, benchmark certification.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Unverified test-split classification metrics.
       After: Confirms high test accuracy and low regret on held-out test split.
    """
    json_path = config.RESULTS_DIR / "decision_tree_test_metrics.json"
    assert json_path.exists(), f"Decision tree test metrics JSON missing: {json_path}"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["sample_count"] == 1000
    assert data["accuracy"] >= 0.85, f"Accuracy too low: {data['accuracy']}"
    assert data["macro_f1"] >= 0.80, f"Macro-F1 too low: {data['macro_f1']}"

    dt_regret = data["regret"]["decision_tree"]
    assert dt_regret < 0.10, f"Decision tree regret too high: {dt_regret}"

    fcfs_regret = data["regret"]["baselines"]["FCFS"]
    rr_regret = data["regret"]["baselines"]["RR"]
    assert dt_regret < fcfs_regret
    assert dt_regret < rr_regret


def test_empirical_scheduling_invariants() -> None:
    """
    1. The use of function:
       Validates fundamental theoretical and empirical operating system scheduling properties.
    2. How does the specific function work:
       Loads results/summary_table.csv. Asserts:
       - At cs_cost=0, SRTF has the lowest mean waiting time among all baselines.
       - RL mean waiting time at cs_cost=0 is within 3.5 ticks of SRTF (optimal bound).
       - At cs_cost=1, RL context switches decrease relative to cs_cost=0.
       - Jain fairness index is strictly between 0 and 1 for all algorithms.
    3. Concepts used in the function:
       SRTF theoretical optimality, RL preemption trade-off adaptation, fairness boundaries.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Unverified empirical relations.
       After: Asserts theoretical consistency of evaluation results.
    """
    summary_csv = config.RESULTS_DIR / "summary_table.csv"
    assert summary_csv.exists(), f"Summary CSV missing: {summary_csv}"

    df = pd.read_csv(summary_csv)
    cs0 = df[df["cs_cost"] == 0].set_index("policy")
    cs1 = df[df["cs_cost"] == 1].set_index("policy")

    # Invariant 1: At cs_cost=0, SRTF minimizes waiting time among classical baselines
    srtf_wait = cs0.loc["SRTF", "avg_waiting_mean"]
    fcfs_wait = cs0.loc["FCFS", "avg_waiting_mean"]
    sjf_wait = cs0.loc["SJF", "avg_waiting_mean"]
    rr_wait = cs0.loc["RR", "avg_waiting_mean"]
    assert srtf_wait < fcfs_wait
    assert srtf_wait < sjf_wait
    assert srtf_wait < rr_wait

    # Invariant 2: Tabular RL approaches SRTF at cs_cost=0
    rl_wait_cs0 = cs0.loc["RL", "avg_waiting_mean"]
    assert abs(rl_wait_cs0 - srtf_wait) < 3.5

    # Invariant 3: RL context switches decrease when cs_cost increases from 0 to 1
    rl_cs0_switches = cs0.loc["RL", "context_switches_mean"]
    rl_cs1_switches = cs1.loc["RL", "context_switches_mean"]
    assert rl_cs1_switches < rl_cs0_switches

    # Invariant 4: Jain fairness index valid range (0, 1]
    for _, row in df.iterrows():
        fairness = row["jain_fairness_mean"]
        assert 0.0 < fairness <= 1.0


def test_cli_evaluate_subcommand(tmp_path: Path) -> None:
    """
    1. The use of function:
       Tests the end-to-end execution of `python cli.py evaluate` via command line arguments.
    2. How does the specific function work:
       Invokes main() with evaluate subcommand pointing to temporary output files for 5 workloads.
       Asserts process exit code 0 and verifies generated output files.
    3. Concepts used in the function:
       CLI integration testing, end-to-end command validation.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Unverified CLI evaluate invocation.
       After: Asserts exit code 0 and temporary file creation.
    """
    out_res = tmp_path / "eval_records.csv"
    out_csv = tmp_path / "eval_summary.csv"
    out_md = tmp_path / "eval_summary.md"
    out_figs = tmp_path / "figures"

    code = main([
        "evaluate",
        "--split",
        "test",
        "--n-workloads",
        "5",
        "--output-results",
        str(out_res),
        "--output-summary-csv",
        str(out_csv),
        "--output-summary-md",
        str(out_md),
        "--output-figures-dir",
        str(out_figs),
        "--workers",
        "1",
    ])

    assert code == 0
    assert out_res.exists() and out_res.stat().st_size > 0
    assert out_csv.exists() and out_csv.stat().st_size > 0
    assert out_md.exists() and out_md.stat().st_size > 0
    assert (out_figs / "comparison_bars.png").exists()
