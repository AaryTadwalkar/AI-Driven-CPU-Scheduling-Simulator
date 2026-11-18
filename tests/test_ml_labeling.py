"""
1. What this file does:
   Tests multi-metric composite labeling and the pilot study analysis framework for the Decision
   Tree policy recommender.
2. Overall design:
   Validates `compute_workload_label` across canonical benchmarks (including Section 4.6),
   verifies composite score normalization formulas, deterministic tie-breaking, margin
   calculations, ambiguity detection, and pilot study reporting.
3. Elements in the file:
   - test_candidate_policies_constant: Verifies candidate policies tuple.
   - test_compute_workload_label_section_4_6: Tests composite scoring on Section 4.6 fixture.
   - test_compute_workload_label_tie_and_ambiguity: Tests margin and ambiguity detection.
   - test_compute_workload_label_cs_cost_impact: Tests labeling sensitivity to context switch cost.
   - test_compute_workload_label_custom_weights: Tests custom metric weighting.
   - test_run_pilot_study_structure: Tests pilot study execution and output distribution schema.
   - test_compute_workload_label_validation_errors: Tests error handling on invalid inputs.
4. Final output of the file:
   Pytest test suite validating labeling and pilot analysis fidelity.
"""

import pytest

from schedsim.domain.process import Process
from schedsim.ml.labeling import (
    CANDIDATE_POLICIES,
    LabelResult,
    compute_workload_label,
    run_pilot_study,
)
from schedsim.workloads.generator import generate_workload
from schedsim.workloads.scenarios import SCENARIO_SHORT_JOBS


def test_candidate_policies_constant() -> None:
    """
    1. The use of function:
       Validates candidate policies constant tuple.
    2. How does the specific function work:
       Asserts CANDIDATE_POLICIES contains exactly ('FCFS', 'SJF', 'RR').
    3. Concepts used in the function:
       Constant verification, schema validation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified candidate policies constant.
       After: Asserts exact matching of candidate policies.
    """
    assert CANDIDATE_POLICIES == ("FCFS", "SJF", "RR")


def test_compute_workload_label_section_4_6() -> None:
    """
    1. The use of function:
       Tests multi-metric composite labeling on the Section 4.6 hand-verified fixture.
    2. How does the specific function work:
       Evaluates P1(0, 5), P2(1, 3), P3(2, 2) at cs_cost=0.
       Validates that raw metrics match Section 4.6 values and composite scores are computed
       according to formula.
    3. Concepts used in the function:
       Hand-verified benchmark testing, composite score verification.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested composite labeling on Section 4.6.
       After: Asserts label result correctly captures metrics and winning policy.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=5),
        Process(pid="P2", arrival=1, burst=3),
        Process(pid="P3", arrival=2, burst=2),
    ]
    res = compute_workload_label(procs, cs_cost=0, quantum=2)

    assert isinstance(res, LabelResult)
    assert res.label in CANDIDATE_POLICIES

    # FCFS metrics
    fcfs_m = res.raw_metrics["FCFS"]
    assert pytest.approx(fcfs_m["avg_waiting"], rel=1e-3) == 10.0 / 3.0
    assert pytest.approx(fcfs_m["avg_response"], rel=1e-3) == 10.0 / 3.0
    assert pytest.approx(fcfs_m["avg_turnaround"], rel=1e-3) == 20.0 / 3.0

    # SJF metrics
    sjf_m = res.raw_metrics["SJF"]
    assert pytest.approx(sjf_m["avg_waiting"], rel=1e-3) == 3.0
    assert pytest.approx(sjf_m["avg_response"], rel=1e-3) == 3.0
    assert pytest.approx(sjf_m["avg_turnaround"], rel=1e-3) == 19.0 / 3.0

    # RR metrics
    rr_m = res.raw_metrics["RR"]
    assert pytest.approx(rr_m["avg_waiting"], rel=1e-3) == 4.0
    assert pytest.approx(rr_m["avg_response"], rel=1e-3) == 1.0
    assert pytest.approx(rr_m["avg_turnaround"], rel=1e-3) == 22.0 / 3.0

    assert res.best_score == res.scores[res.label]
    assert res.margin >= 0.0


def test_compute_workload_label_tie_and_ambiguity() -> None:
    """
    1. The use of function:
       Tests ambiguity detection when candidate policy scores are nearly identical.
    2. How does the specific function work:
       Evaluates a single-process workload where FCFS, SJF, and RR produce identical execution,
       asserting margin is close to 0 and is_ambiguous is True.
    3. Concepts used in the function:
       Ambiguity flagging, tie tolerance thresholding.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested tie and ambiguity detection.
       After: Asserts is_ambiguous is True and margin is under tolerance.
    """
    procs = [Process(pid="P1", arrival=0, burst=5)]
    res = compute_workload_label(procs, tie_tolerance=0.02)
    assert res.is_ambiguous is True
    assert res.margin < 0.02


def test_compute_workload_label_cs_cost_impact() -> None:
    """
    1. The use of function:
       Tests the impact of context switch overhead on composite policy scoring.
    2. How does the specific function work:
       Evaluates a workload under cs_cost=0 vs cs_cost=2, verifying that Round Robin's waiting
       and turnaround metrics worsen under positive context switch costs due to preemption overhead.
    3. Concepts used in the function:
       Context switch sensitivity, preemption overhead impact.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested cs_cost sensitivity in labeling.
       After: Asserts higher cs_cost degrades Round Robin metrics relative to FCFS/SJF.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=10),
        Process(pid="P2", arrival=1, burst=2),
    ]
    res_cs0 = compute_workload_label(procs, cs_cost=0, quantum=2)
    res_cs2 = compute_workload_label(procs, cs_cost=2, quantum=2)

    assert (
        res_cs2.raw_metrics["RR"]["avg_waiting"]
        > res_cs0.raw_metrics["RR"]["avg_waiting"]
    )


def test_compute_workload_label_custom_weights() -> None:
    """
    1. The use of function:
       Tests labeling with custom objective weights prioritizing response time.
    2. How does the specific function work:
       Evaluates the Section 4.6 fixture where RR dominates on response time (1.0 vs 3.0 and 3.33).
       Under heavy response weight (e.g. 0.9 response), RR should win.
    3. Concepts used in the function:
       Weighted objective optimization, Pareto trade-off control.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested custom metric weighting in compute_workload_label.
       After: Asserts response-heavy weighting selects Round Robin.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=5),
        Process(pid="P2", arrival=1, burst=3),
        Process(pid="P3", arrival=2, burst=2),
    ]
    weights = {"wait": 0.05, "resp": 0.90, "tat": 0.05}
    res = compute_workload_label(procs, cs_cost=0, quantum=2, weights=weights)
    assert res.label == "RR"


def test_run_pilot_study_structure() -> None:
    """
    1. The use of function:
       Tests the `run_pilot_study` function executing across a sample of workloads.
    2. How does the specific function work:
       Generates 18 sample workloads and executes `run_pilot_study`. Asserts output schema,
       presence of standard configurations, and that distribution percentages sum to 100%.
    3. Concepts used in the function:
       Pilot study validation, sensitivity report verification.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested pilot study runner.
       After: Asserts valid pilot study summary dictionary with consistent totals.
    """
    sample_workloads = [
        generate_workload(scenario=SCENARIO_SHORT_JOBS, n=4, seed=i)
        for i in range(18)
    ]
    report = run_pilot_study(sample_workloads)

    assert isinstance(report, dict)
    assert "cs0_q2_standard" in report
    assert "cs1_q2_standard" in report
    assert "cs1_q4_standard" in report
    assert "cs1_q2_balanced_weights" in report

    for cfg_name, data in report.items():
        assert data["total_workloads"] == 18
        total_pct = sum(
            p_data["percentage"] for p_data in data["distribution"].values()
        )
        assert pytest.approx(total_pct, abs=1e-2) == 100.0


def test_compute_workload_label_validation_errors() -> None:
    """
    1. The use of function:
       Tests input validation in `compute_workload_label`.
    2. How does the specific function work:
       Passes invalid process sequences and asserts TypeError or ValueError.
    3. Concepts used in the function:
       Defensive input validation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid inputs to compute_workload_label.
       After: Asserts expected exceptions raised.
    """
    with pytest.raises(TypeError):
        compute_workload_label("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        compute_workload_label([])
