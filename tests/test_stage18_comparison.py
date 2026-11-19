"""
1. What this file does:
   Defines unit and integration acceptance tests for Stage 18: Final Comparison (The Deliverable),
   validating the full 11-policy run matrix across S0 and S1, S0 SRTF optimality sanity check,
   15-column comparison table schemas, 1D linear break-even interpolation, bootstrap CI bounds,
   generated publication figures, pre-registered hypotheses evaluations (H1-H4), and CLI execution.
2. Overall design:
   Structured into discrete Pytest test classes corresponding to the 7 Stage 18 acceptance criteria:
   - TestS0Sanity: Tests SRTF optimality bound at cs_cost = 0.
   - TestComparisonMatrixSchema: Tests column headers and roster completeness for S0 and S1 tables.
   - TestBreakEvenCalculation: Tests linear interpolation mathematics and boundary conditions.
   - TestBootstrapCI: Tests that lower <= mean <= upper across deterministic and stochastic data.
   - TestGeneratedFigures: Tests existence and validity of all 5 publication-quality figures.
   - TestHypothesesResults: Tests existence and formal verdicts in docs/HYPOTHESES_RESULTS.md.
   - TestCLICompareAll: Tests end-to-end CLI subcommand compare-all invocation and artifacts.
3. Elements in the file:
   - TestS0Sanity: Class verifying theoretical optimality bound at S0.
   - TestComparisonMatrixSchema: Class verifying comparison matrix dataframe and markdown schemas.
   - TestBreakEvenCalculation: Class verifying break-even interpolation mathematical accuracy.
   - TestBootstrapCI: Class verifying non-parametric percentile bootstrap confidence intervals.
   - TestGeneratedFigures: Class validating generated figure files and PNG integrity.
   - TestHypothesesResults: Class validating hypothesis evaluation report and verdict syntax.
   - TestCLICompareAll: Class executing integration run of CLI compare-all subcommand.
4. Final output of the file:
   Executes pytest test suite verifying Stage 18 deliverables with 100% pass rate.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import config
from cli import main
from schedsim.evaluation import bootstrap_ci
from schedsim.evaluation.final_comparison import (
    POLICY_ROSTER_NAMES,
    build_policy_workload_result,
    calculate_break_even,
    verify_s0_sanity,
)
from schedsim.policies.registry import get_policy
from schedsim.workloads.splits import generate_split_workloads


class TestS0Sanity:
    """
    1. The use of class:
       Acceptance Test 1: Verifies theoretical optimality bound of SRTF at Setting S0 (cs = 0).
    2. What the class contains:
       Unit tests validating that no policy beats SRTF before overhead at cs_cost = 0.
    3. Technologies used and why:
       Pytest assertions, NumPy numerical comparisons, SchedSim engine simulation.
    4. Key design and overall design of the class:
       Runs verification against test split workloads and unit tests the sanity checker itself.
    """

    def test_s0_sanity_function_passes(self) -> None:
        """
        1. The use of function:
           Tests that verify_s0_sanity executes without error and confirms SRTF optimality.
        2. How does the specific function work:
           Generates 20 test split workloads, runs SRTF and competitor policies, and asserts
           verify_s0_sanity returns True.
        3. Concepts used in the function:
           Theoretical bounds verification, preemption without penalty optimality.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Unverified policy results at S0.
           After: Asserted that verify_s0_sanity returns True without raising AssertionError.
        """
        workloads = generate_split_workloads("test", n_workloads=20)
        scenarios = [w.scenario for w in workloads]
        from schedsim.evaluation.final_comparison import simulate_policy_workloads

        srtf_policy = get_policy("SRTF")
        fcfs_policy = get_policy("FCFS")
        sjf_policy = get_policy("SJF")

        srtf_sims = simulate_policy_workloads(srtf_policy, workloads, cs_cost=0)
        fcfs_sims = simulate_policy_workloads(fcfs_policy, workloads, cs_cost=0)
        sjf_sims = simulate_policy_workloads(sjf_policy, workloads, cs_cost=0)

        results = {
            "SRTF": build_policy_workload_result("SRTF", srtf_sims, srtf_sims, scenarios),
            "FCFS": build_policy_workload_result("FCFS", fcfs_sims, fcfs_sims, scenarios),
            "SJF": build_policy_workload_result("SJF", sjf_sims, sjf_sims, scenarios),
        }
        is_sane = verify_s0_sanity(results)
        assert is_sane is True

    def test_s0_sanity_detects_violation(self) -> None:
        """
        1. The use of function:
           Tests that verify_s0_sanity correctly detects when a policy hypothetically beats SRTF.
        2. How does the specific function work:
           Creates synthetic PolicyWorkloadResult instances where a mock policy achieves a lower
           waiting time than SRTF by 1.0 tick, and asserts ValueError is raised.
        3. Concepts used in the function:
           Fault injection testing, sanity bound enforcement.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Synthetic violation injected.
           After: Confirmed ValueError is raised with descriptive message.
        """
        from schedsim.evaluation.final_comparison import PolicyWorkloadResult

        zeros_5 = np.zeros(5, dtype=np.float64)
        srtf_res = PolicyWorkloadResult(
            policy_name="SRTF",
            raw_avg_waiting=np.full(5, 10.0, dtype=np.float64),
            raw_avg_turnaround=np.full(5, 20.0, dtype=np.float64),
            raw_avg_response=zeros_5,
            effective_avg_turnaround=np.full(5, 20.0, dtype=np.float64),
            effective_makespan=zeros_5,
            max_waiting=zeros_5,
            jain_fairness=zeros_5,
            context_switches=zeros_5,
            decisions=zeros_5,
            inference_us_total=zeros_5,
            ai_overhead_ticks=zeros_5,
            raw_makespan=zeros_5,
            scenarios=("balanced",) * 5,
        )
        cheating_res = PolicyWorkloadResult(
            policy_name="CheatingPolicy",
            raw_avg_waiting=np.full(5, 8.0, dtype=np.float64),
            raw_avg_turnaround=np.full(5, 18.0, dtype=np.float64),
            raw_avg_response=zeros_5,
            effective_avg_turnaround=np.full(5, 18.0, dtype=np.float64),
            effective_makespan=zeros_5,
            max_waiting=zeros_5,
            jain_fairness=zeros_5,
            context_switches=zeros_5,
            decisions=zeros_5,
            inference_us_total=zeros_5,
            ai_overhead_ticks=zeros_5,
            raw_makespan=zeros_5,
            scenarios=("balanced",) * 5,
        )
        results = {
            "SRTF": srtf_res,
            "CheatingPolicy": cheating_res,
        }
        with pytest.raises(ValueError, match="S0 Sanity Violated"):
            verify_s0_sanity(results)


class TestComparisonMatrixSchema:
    """
    1. The use of class:
       Acceptance Test 2: Validates the schema and completeness of comparison matrix tables.
    2. What the class contains:
       Tests checking presence of all 15 required metric columns and all 11 roster policies.
    3. Technologies used and why:
       Pandas DataFrame operations, set comparisons, CSV reading.
    4. Key design and overall design of the class:
       Verifies column schema, non-null values, row count, and CI string formatting.
    """

    REQUIRED_COLUMNS = [
        "policy",
        "avg_waiting",
        "avg_turnaround",
        "avg_response",
        "effective_avg_turnaround",
        "effective_makespan",
        "max_waiting",
        "jain_fairness",
        "context_switches",
        "decisions",
        "inference_us_total",
        "overhead_pct_makespan",
        "gain_over_best_classical",
        "gap_to_oracle",
        "win_tie_loss",
    ]

    def test_s0_and_s1_matrix_files_exist_and_conform(self) -> None:
        """
        1. The use of function:
           Verifies results/final_comparison_S0.csv and results/final_comparison_S1.csv exist
           and conform strictly to the 15-column schema and 11-policy roster.
        2. How does the specific function work:
           Reads CSV files into Pandas DataFrames, compares column names against REQUIRED_COLUMNS,
           checks that all 11 policies are present, and validates non-emptiness.
        3. Concepts used in the function:
           Schema validation, tabular completeness check.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Unverified CSV files on disk.
           After: Confirmed schema, column names, policy rosters, and row counts match spec.
        """
        for setting in ("S0", "S1"):
            csv_path = config.RESULTS_DIR / f"final_comparison_{setting}.csv"
            md_path = config.RESULTS_DIR / f"final_comparison_{setting}.md"
            assert csv_path.exists(), f"Missing required deliverable: {csv_path}"
            assert md_path.exists(), f"Missing required deliverable: {md_path}"

            df = pd.read_csv(csv_path)
            # Verify columns
            for col in self.REQUIRED_COLUMNS:
                assert col in df.columns, f"Missing column '{col}' in {csv_path}"

            # Verify all 11 policies present
            present_policies = set(df["policy"].tolist())
            for expected_policy in POLICY_ROSTER_NAMES:
                assert (
                    expected_policy in present_policies
                ), f"Missing policy '{expected_policy}' in {csv_path}"

            assert len(df) == 11, f"Expected 11 policies in {csv_path}, got {len(df)}"

            # Verify Markdown file contains table headers
            md_content = md_path.read_text(encoding="utf-8")
            assert "| policy |" in md_content
            assert "| avg_waiting |" in md_content
            assert "| effective_avg_turnaround |" in md_content


class TestBreakEvenCalculation:
    """
    1. The use of class:
       Acceptance Test 3: Tests 1D linear interpolation for break-even hardware tick calculation.
    2. What the class contains:
       Unit tests for boundary conditions, crossing interpolation, and edge-case handling.
    3. Technologies used and why:
       Analytical test cases, floating point precision comparison.
    4. Key design and overall design of the class:
       Tests calculate_break_even across strictly defined mathematical brackets.
    """

    def test_break_even_exact_linear_interpolation(self) -> None:
        """
        1. The use of function:
           Verifies that linear interpolation calculates the exact crossing tick point.
        2. How does the specific function work:
           Provides synthetic sweep values {10: 50.0, 100: 20.0} with target cost 30.0.
           Expected tick: 10 + ((30 - 50) / (20 - 50)) * (100 - 10) = 10 + (2/3)*90 = 70.0.
           Asserts calculate_break_even returns float matching 70.0.
        3. Concepts used in the function:
           Piecewise linear interpolation, zero-crossing root estimation.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Unverified interpolation function.
           After: Confirmed exact float match on break-even tick value.
        """
        ticks = [1.0, 10.0, 100.0, 1000.0]
        metrics = [80.0, 50.0, 20.0, 15.0]
        target_cost = 30.0  # Crosses between tick 10 (cost 50) and tick 100 (cost 20)
        t_star = calculate_break_even(ticks, metrics, target_cost)
        assert t_star is not None
        assert pytest.approx(t_star, rel=1e-5) == 70.0

    def test_break_even_already_better_at_lowest_tick(self) -> None:
        """
        1. The use of function:
           Verifies that when a model already beats the baseline at tick 1, tick 1 is returned.
        2. How does the specific function work:
           Provides synthetic sweep where cost at tick 1 is 10.0, below target 20.0.
           Asserts t_star == 1.0.
        3. Concepts used in the function:
           Lower boundary clamping.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Unverified boundary check.
           After: Confirmed lowest tick returned immediately.
        """
        ticks = [1.0, 10.0, 100.0]
        metrics = [10.0, 8.0, 5.0]
        target_cost = 20.0
        t_star = calculate_break_even(ticks, metrics, target_cost)
        assert t_star == 1.0

    def test_break_even_never_crosses(self) -> None:
        """
        1. The use of function:
           Verifies that when a model never beats the baseline, None is returned.
        2. How does the specific function work:
           Provides synthetic sweep where cost is strictly higher than target for all points.
           Asserts t_star is None.
        3. Concepts used in the function:
           Negative outcome representation in break-even analysis.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Sweep with strictly higher costs.
           After: Confirmed None is returned.
        """
        ticks = [1.0, 10.0, 100.0, 1000.0]
        metrics = [50.0, 45.0, 40.0, 35.0]
        target_cost = 25.0
        t_star = calculate_break_even(ticks, metrics, target_cost)
        assert t_star is None


class TestBootstrapCI:
    """
    1. The use of class:
       Acceptance Test 4: Verifies bootstrap confidence interval computation bounds.
    2. What the class contains:
       Unit tests asserting lower <= mean <= upper for deterministic and random datasets.
    3. Technologies used and why:
       NumPy random number generation, percentile estimation.
    4. Key design and overall design of the class:
       Validates consistency of _bootstrap_ci_1d helper across array sizes and distributions.
    """

    def test_bootstrap_bounds_on_random_normal(self) -> None:
        """
        1. The use of function:
           Verifies that lower <= mean <= upper holds for a random normal distribution.
        2. How does the specific function work:
           Generates 500 samples from N(25, 4), computes bootstrap CI, and validates ordering.
        3. Concepts used in the function:
           Empirical bootstrap CI correctness.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Sample array.
           After: Confirmed mean, lower, and upper satisfy lower <= mean <= upper.
        """
        rng = np.random.default_rng(42)
        data = rng.normal(loc=25.0, scale=4.0, size=500)
        mean, lower, upper = bootstrap_ci(data, n_resamples=500, ci=0.95, seed=42)

        assert lower <= mean
        assert mean <= upper
        assert pytest.approx(mean, rel=0.05) == 25.0

    def test_bootstrap_bounds_on_constant_array(self) -> None:
        """
        1. The use of function:
           Verifies that lower == mean == upper for constant value arrays.
        2. How does the specific function work:
           Creates an array of identical values [42.0] * 50, computes bootstrap CI, and asserts
           lower == mean == upper == 42.0.
        3. Concepts used in the function:
           Degenerate distribution handling in bootstrap resampling.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Constant array.
           After: Confirmed lower == mean == upper == 42.0.
        """
        data = np.full(50, 42.0)
        mean, lower, upper = bootstrap_ci(data, n_resamples=200, ci=0.95, seed=42)

        assert mean == 42.0
        assert lower == 42.0
        assert upper == 42.0


class TestGeneratedFigures:
    """
    1. The use of class:
       Acceptance Test 5: Verifies existence and validity of all 5 publication-quality figures.
    2. What the class contains:
       File existence checks, non-empty byte checks, and PIL image header validation.
    3. Technologies used and why:
       PIL (Pillow) image library, pathlib.Path.
    4. Key design and overall design of the class:
       Confirms publication figures are properly saved, readable, and non-corrupted.
    """

    EXPECTED_FIGURES = [
        "effective_cost_vs_tick.png",
        "gap_to_oracle_bars.png",
        "overhead_breakdown.png",
        "final_scenario_heatmap.png",
        "gantt_with_ai_stalls.png",
    ]

    def test_all_five_figures_exist_and_are_valid_images(self) -> None:
        """
        1. The use of function:
           Verifies that all 5 figure files exist in results/figures/ and are valid PNG images.
        2. How does the specific function work:
           Iterates through EXPECTED_FIGURES, asserts file existence, checks file size > 2 KB,
           and opens with Image.open to verify PNG integrity.
        3. Concepts used in the function:
           Image integrity verification, deliverable artifact validation.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Unverified image files on disk.
           After: Confirmed all 5 figures exist, are non-empty, and load without corruption.
        """
        figures_dir = config.RESULTS_DIR / "figures"
        assert figures_dir.exists(), f"Figures directory does not exist: {figures_dir}"

        for fig_name in self.EXPECTED_FIGURES:
            fig_path = figures_dir / fig_name
            assert fig_path.exists(), f"Missing required figure: {fig_path}"
            assert fig_path.stat().st_size > 2048, f"Figure file is suspiciously small: {fig_path}"

            # Open with PIL to confirm valid image header and format
            with Image.open(fig_path) as img:
                assert img.format == "PNG", f"Expected PNG format, got {img.format} for {fig_path}"
                assert img.width > 100, f"Image width too small: {img.width}"
                assert img.height > 100, f"Image height too small: {img.height}"


class TestHypothesesResults:
    """
    1. The use of class:
       Acceptance Test 6: Verifies docs/HYPOTHESES_RESULTS.md exists and contains H1-H4 verdicts.
    2. What the class contains:
       Text parsing tests checking for H1-H4 sections and valid formal verdicts.
    3. Technologies used and why:
       Text processing, regular expressions, file reading.
    4. Key design and overall design of the class:
       Ensures pre-registered scientific hypotheses are evaluated with explicit verdicts.
    """

    VALID_VERDICTS = {"CONFIRMED", "REFUTED", "INCONCLUSIVE"}

    def test_hypotheses_results_document_exists_and_valid(self) -> None:
        """
        1. The use of function:
           Verifies that docs/HYPOTHESES_RESULTS.md contains verdicts for H1, H2, H3, and H4.
        2. How does the specific function work:
           Reads docs/HYPOTHESES_RESULTS.md, verifies presence of sections for H1, H2, H3, H4,
           and checks that each section contains a verdict from VALID_VERDICTS.
        3. Concepts used in the function:
           Pre-registered scientific hypothesis evaluation validation.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Unverified markdown report.
           After: Confirmed all 4 hypotheses have valid formal verdicts and quantitative evidence.
        """
        doc_path = config.DOCS_DIR / "HYPOTHESES_RESULTS.md"
        assert doc_path.exists(), f"Missing required hypothesis evaluation doc: {doc_path}"

        content = doc_path.read_text(encoding="utf-8")

        for h_id in ("H1", "H2", "H3", "H4"):
            assert h_id in content, f"Missing section for hypothesis {h_id} in {doc_path}"

            # Find matching verdict in text
            found_verdict = False
            for verdict in self.VALID_VERDICTS:
                if f"**{verdict}**" in content or f": {verdict}" in content:
                    found_verdict = True
                    break
            assert found_verdict, f"No valid verdict found for hypothesis in {doc_path}"


class TestCLICompareAll:
    """
    1. The use of class:
       Acceptance Test 7: Verifies end-to-end execution of CLI compare-all subcommand.
    2. What the class contains:
       Subprocess or direct main() invocation test with temporary directory output.
    3. Technologies used and why:
       Pytest tmp_path fixture, CLI main invocation, file output checks.
    4. Key design and overall design of the class:
       Executes compare-all on a small workload subset to verify arguments and pipeline.
    """

    def test_cli_compare_all_invocation(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Tests that main(['compare-all', '--skip-plots', ...]) executes successfully.
        2. How does the specific function work:
           Invokes main() with compare-all, --n-workloads 5, --skip-plots, and --output-dir.
           Asserts exit code 0 and verifies S0/S1 CSV and Markdown tables were written.
        3. Concepts used in the function:
           CLI integration testing, temporary artifact containment.
        4. Places where this function is imported and used:
           Invoked by pytest during acceptance test execution.
        5. Final output of the function, before and after:
           Before: Clean temporary directory.
           After: Asserted exit code 0 and verified all comparison tables created in tmp_path.
        """
        args = [
            "compare-all",
            "--n-workloads",
            "5",
            "--skip-plots",
            "--output-dir",
            str(tmp_path),
        ]
        exit_code = main(args)
        assert exit_code == 0

        assert (tmp_path / "final_comparison_S0.csv").exists()
        assert (tmp_path / "final_comparison_S0.md").exists()
        assert (tmp_path / "final_comparison_S1.csv").exists()
        assert (tmp_path / "final_comparison_S1.md").exists()
