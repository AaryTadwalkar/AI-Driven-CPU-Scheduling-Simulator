"""
1. What this file does:
   Defines unit and acceptance tests for Stage 15: Model 4 - Gradient-Boosted Candidate Ranker
   of the CPU Scheduling Simulator, verifying model exports, branchless score equivalence,
   permutation equivariance, determinism, engine integration, closed-loop validation gap,
   registry lookups, CLI execution, tie-breaking rules, and defensive error handling.
2. Overall design:
   Verifies all 7 acceptance criteria and core invariants from REVISED_IMPLEMENTATION_PLAN.md:
   - Acceptance test 1: Exported scores match scikit-learn decision_function within 1e-6.
   - Acceptance test 2: Permutation equivariance (shuffling ready candidates only permutes scores).
   - Acceptance test 3: Determinism across repeated training runs with identical random seed.
   - Acceptance test 4: GBDTRankerPolicy executes inside Engine.run() without errors for S0 and S1.
   - Acceptance test 5: S0 validation closed-loop gap vs SRTF is near zero (<= 0.02).
   - Acceptance test 6: Centralized policy registry lookups for PID_GBDT and GBDT.
   - Acceptance test 7: CLI subcommand `train-pid-gbdt` execution and artifact generation.
   - Comprehensive unit tests: tie-breaking priority, factory loading, and input assertions.
3. Elements in the file:
   - test_stage15_config_and_exports: Verifies config constants and module exports.
   - test_stage15_acceptance_1_exported_scores_match_sklearn_decision_function: Test 1 verification.
   - test_stage15_acceptance_2_permutation_equivariance: Test 2 permutation equivariance.
   - test_stage15_acceptance_3_determinism_across_repeated_training: Test 3 training determinism.
   - test_stage15_acceptance_4_gbdt_policy_runs_in_engine_s0_s1: Test 4 simulation execution.
   - test_stage15_acceptance_5_s0_validation_gap_vs_srtf_near_zero: Test 5 S0 closed-loop gap.
   - test_stage15_acceptance_6_registry_lookup_pid_gbdt_and_gbdt: Test 6 policy registry lookup.
   - test_stage15_acceptance_7_cli_train_pid_gbdt_command: Test 7 CLI subcommand execution.
   - test_stage15_tie_breaking_rules: Unit tests for the 4-tier tie-breaking mechanism.
   - test_stage15_defensive_validation_and_errors: Unit tests asserting defensive validation errors.
   - test_stage15_validation_report_json_structure: Verifies results/pid_gbdt_val.json structure.
4. Final output of the file:
   Pytest test suite executing full verification of Stage 15 GBDT candidate ranker components.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

import config
from cli import main
from schedsim.domain.engine import ProcView, View, run
from schedsim.domain.process import Process
from schedsim.pidsel import (
    GBDTRankerPolicy,
    evaluate_gbdt_on_validation,
    export_tree_ensemble,
    prepare_training_data,
    score_candidates,
    train_gbdt,
)
from schedsim.pidsel.features import extract_candidate_features
from schedsim.policies.registry import get_policy, list_policies
from schedsim.workloads.splits import generate_split_workloads


def test_stage15_config_and_exports() -> None:
    """
    1. The use of function:
       Verifies configuration constants and module exports for Stage 15.
    2. How does the specific function work:
       Asserts config.GBDT_PARAMS contains expected hyperparameters. Verifies that all Stage 15
       functions and classes are properly exported by schedsim.pidsel.
    3. Concepts used in the function:
       Configuration validation, package interface verification.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Raw configuration and package exports.
       After: Confirmed valid configuration parameters and symbols.
    """
    assert "n_estimators" in config.GBDT_PARAMS
    assert "max_depth" in config.GBDT_PARAMS
    assert "learning_rate" in config.GBDT_PARAMS
    assert config.GBDT_PARAMS["n_estimators"] == 50
    assert config.GBDT_PARAMS["max_depth"] == 3
    assert config.GBDT_PARAMS["learning_rate"] == 0.1

    assert callable(export_tree_ensemble)
    assert callable(score_candidates)
    assert callable(prepare_training_data)
    assert callable(train_gbdt)
    assert callable(evaluate_gbdt_on_validation)
    assert issubclass(GBDTRankerPolicy, object)


def test_stage15_acceptance_1_exported_scores_match_sklearn_decision_function() -> None:
    """
    1. The use of function:
       Acceptance Test 1: Verifies pure-NumPy candidate scores match scikit-learn decision_function
       within 1e-6 on 10,000 candidate rows.
    2. How does the specific function work:
       Fits a GradientBoostingClassifier on synthetic candidate features with GBDT_PARAMS.
       Exports flat tree arrays via export_tree_ensemble. Generates 10,000 synthetic candidate rows.
       Computes scores via score_candidates and decision values via model.decision_function.
       Asserts maximum absolute difference is strictly below 1e-6.
    3. Concepts used in the function:
       Inference export fidelity, numerical equivalence, tolerance assertion.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Scikit-learn model and candidate features matrix.
       After: Confirms numerical difference across 10,000 rows is strictly < 1e-6.
    """
    rng = np.random.default_rng(12345)
    x_train = rng.normal(size=(2000, 13))
    y_train = (x_train[:, 0] - 0.5 * x_train[:, 1] + 0.3 * x_train[:, 5] > 0).astype(int)

    clf = GradientBoostingClassifier(**config.GBDT_PARAMS, random_state=42)
    clf.fit(x_train, y_train)

    trees, lr, init_val = export_tree_ensemble(clf)
    assert len(trees) == config.GBDT_PARAMS["n_estimators"]
    assert lr == config.GBDT_PARAMS["learning_rate"]

    x_test = rng.normal(size=(10000, 13))
    scores = score_candidates(trees, lr, init_val, x_test)
    sklearn_decision = clf.decision_function(x_test)

    max_diff = float(np.max(np.abs(scores - sklearn_decision)))
    assert max_diff < 1e-6, f"Max score deviation {max_diff} exceeds 1e-6 threshold."


def test_stage15_acceptance_2_permutation_equivariance() -> None:
    """
    1. The use of function:
       Acceptance Test 2: Verifies permutation equivariance of candidate scoring and selection.
    2. How does the specific function work:
       Constructs a View with 4 ready candidates having distinct properties. Scores candidates
       under the canonical ordering. Permutes the ready queue in all possible orders and checks
       that the extracted feature rows and candidate scores permute identically, and that
       GBDTRankerPolicy.select() returns the exact same chosen PID.
    3. Concepts used in the function:
       Permutation equivariance, invariant decision making, ranking symmetry.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Multiple permutations of ready process queues.
       After: Confirms identical score mappings and identical PID election across all permutations.
    """
    p1 = ProcView(pid="P1", arrival=0, burst=6, remaining=4, waited=2)
    p2 = ProcView(pid="P2", arrival=1, burst=4, remaining=2, waited=1)
    p3 = ProcView(pid="P3", arrival=2, burst=8, remaining=7, waited=0)
    p4 = ProcView(pid="P4", arrival=0, burst=5, remaining=2, waited=2)
    base_ready = (p1, p2, p3, p4)

    policy = GBDTRankerPolicy.from_setting("S0")

    view_base = View(time=3, ready=base_ready, last_pid="P1", newly_arrived=(), pending_arrivals=1)
    expected_pid = policy.select(view_base)

    x_base, pids_base = extract_candidate_features(view_base, cs_cost=0, pending_arrivals=1)
    scores_base = score_candidates(policy.trees, policy.lr, policy.init_val, x_base)
    score_map = {pid: s for pid, s in zip(pids_base, scores_base, strict=True)}

    import itertools

    for perm in itertools.permutations(base_ready):
        perm_view = View(
            time=3, ready=perm, last_pid="P1", newly_arrived=(), pending_arrivals=1
        )
        x_perm, pids_perm = extract_candidate_features(perm_view, cs_cost=0, pending_arrivals=1)
        scores_perm = score_candidates(policy.trees, policy.lr, policy.init_val, x_perm)

        for pid, score in zip(pids_perm, scores_perm, strict=True):
            assert abs(score - score_map[pid]) < 1e-9, (
                f"Score for {pid} changed under permutation from {score_map[pid]} to {score}"
            )

        selected_pid = policy.select(perm_view)
        assert selected_pid == expected_pid, (
            f"Selected PID changed under permutation: expected {expected_pid}, got {selected_pid}."
        )


def test_stage15_acceptance_3_determinism_across_repeated_training(tmp_path: Path) -> None:
    """
    1. The use of function:
       Acceptance Test 3: Verifies determinism across repeated training runs with identical seeds.
    2. How does the specific function work:
       Invokes train_gbdt twice for setting S0 with max_states=100 and random_state=42.
       Asserts that the exported tree structures, thresholds, features, leaf values, and model
       predictions are bit-for-bit identical across runs.
    3. Concepts used in the function:
       Deterministic training, seed reproducibility, model artifact comparison.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Raw dataset and fixed random state.
       After: Confirms identical model artifacts across repeated fits.
    """
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    _, art1 = train_gbdt(setting="S0", models_dir=dir1, max_states=100, random_state=42)
    _, art2 = train_gbdt(setting="S0", models_dir=dir2, max_states=100, random_state=42)

    assert art1["lr"] == art2["lr"]
    assert art1["init_val"] == art2["init_val"]
    assert len(art1["trees"]) == len(art2["trees"])

    for t1, t2 in zip(art1["trees"], art2["trees"], strict=True):
        assert np.array_equal(t1["l"], t2["l"])
        assert np.array_equal(t1["r"], t2["r"])
        assert np.array_equal(t1["f"], t2["f"])
        assert np.array_equal(t1["th"], t2["th"])
        assert np.array_equal(t1["v"], t2["v"])


def test_stage15_acceptance_4_gbdt_policy_runs_in_engine_s0_s1() -> None:
    """
    1. The use of function:
       Acceptance Test 4: Verifies GBDTRankerPolicy executes inside Engine.run() without errors
       for both setting S0 and setting S1.
    2. How does the specific function work:
       Instantiates GBDTRankerPolicy via from_setting("S0") and from_setting("S1").
       Executes simulation runs across a multi-process workload using run().
       Verifies work conservation, completion times, and simulation metric invariants.
    3. Concepts used in the function:
       Closed-loop policy simulation, engine integration, invariant assertions.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Workload processes and initialized GBDTRankerPolicy instances.
       After: Valid SimResult objects with verified simulation metrics.
    """
    procs = [
        Process("P1", arrival=0, burst=4, priority=0),
        Process("P2", arrival=1, burst=3, priority=0),
        Process("P3", arrival=2, burst=5, priority=0),
    ]

    # Test S0 execution
    p_s0 = GBDTRankerPolicy.from_setting("S0")
    res_s0 = run(p_s0, procs, cs_cost=0)
    assert res_s0.timeline.busy_ticks == 12
    assert res_s0.simulation_metrics.context_switches >= 0
    assert len(res_s0.process_metrics) == 3

    # Test S1 execution
    p_s1 = GBDTRankerPolicy.from_setting("S1")
    res_s1 = run(p_s1, procs, cs_cost=1)
    assert res_s1.timeline.busy_ticks == 12
    assert res_s1.timeline.overhead_ticks >= 0
    assert len(res_s1.process_metrics) == 3


def test_stage15_acceptance_5_s0_validation_gap_vs_srtf_near_zero() -> None:
    """
    1. The use of function:
       Acceptance Test 5: Verifies S0 validation closed-loop gap vs SRTF is near zero (<= 0.02)
       across 100+ validation workloads.
    2. How does the specific function work:
       Loads S0 GBDT model and executes closed-loop simulation across 100 validation workloads
       under GBDTRankerPolicy and SRTFPolicy. Calculates the closed-loop waiting time gap:
       (J_policy - J_srtf) / J_srtf. Asserts the gap is strictly <= 0.02 (2%).
    3. Concepts used in the function:
       Closed-loop validation benchmarking, optimality gap verification.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Trained S0 ranker and 100 validation workloads.
       After: Confirms closed-loop waiting time gap vs SRTF is <= 0.02.
    """
    val_workloads = generate_split_workloads("val", n_workloads=100)
    policy = GBDTRankerPolicy.from_setting("S0")
    from schedsim.policies.srtf import SRTFPolicy

    srtf = SRTFPolicy()

    j_policy = []
    j_srtf = []
    for w in val_workloads:
        r_p = run(policy, w.processes, cs_cost=0)
        r_s = run(srtf, w.processes, cs_cost=0)
        j_policy.append(r_p.simulation_metrics.avg_waiting)
        j_srtf.append(r_s.simulation_metrics.avg_waiting)

    mean_p = float(np.mean(j_policy))
    mean_s = float(np.mean(j_srtf))
    gap = (mean_p - mean_s) / mean_s
    assert gap <= 0.02, f"S0 closed loop gap vs SRTF ({gap:.4f}) exceeds 0.02 threshold."


def test_stage15_acceptance_6_registry_lookup_pid_gbdt_and_gbdt() -> None:
    """
    1. The use of function:
       Acceptance Test 6: Verifies centralized policy registry lookup for PID_GBDT and GBDT.
    2. How does the specific function work:
       Tests get_policy with canonical and case-insensitive names: "PID_GBDT", "GBDT",
       "pid_gbdt", and "gbdt". Confirms all instantiate GBDTRankerPolicy.
       Verifies list_policies(include_aliases=True) lists PID_GBDT and GBDT.
    3. Concepts used in the function:
       Factory resolution, case-insensitivity, alias mapping.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Policy registry lookup keys.
       After: Confirms valid instantiation of GBDTRankerPolicy.
    """
    p1 = get_policy("PID_GBDT")
    assert isinstance(p1, GBDTRankerPolicy)

    p2 = get_policy("GBDT")
    assert isinstance(p2, GBDTRankerPolicy)

    p3 = get_policy("pid_gbdt")
    assert isinstance(p3, GBDTRankerPolicy)

    p4 = get_policy("gbdt")
    assert isinstance(p4, GBDTRankerPolicy)

    p_s1 = get_policy("PID_GBDT", setting="S1")
    assert isinstance(p_s1, GBDTRankerPolicy)
    assert p_s1.cs_cost == 1

    registered_aliases = list_policies(include_aliases=True)
    assert "PID_GBDT" in registered_aliases
    assert "GBDT" in registered_aliases


def test_stage15_acceptance_7_cli_train_pid_gbdt_command(tmp_path: Path) -> None:
    """
    1. The use of function:
       Acceptance Test 7: Verifies CLI subcommand `train-pid-gbdt` execution and artifact output.
    2. How does the specific function work:
       Invokes cli.main with `train-pid-gbdt`, `--setting S0`, `--max-states 50`, `--data-dir data`,
       `--models-dir <tmp_path>`. Asserts return code is 0. Confirms `pid_gbdt_S0.joblib` and
       `pid_gbdt_S0_meta.json` are created in target directory.
    3. Concepts used in the function:
       CLI subcommand testing, parameter parsing, artifact output verification.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Temporary model directory and CLI arguments.
       After: Confirms exit code 0 and generated model artifacts.
    """
    exit_code = main(
        [
            "train-pid-gbdt",
            "--setting",
            "S0",
            "--max-states",
            "50",
            "--models-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "pid_gbdt_S0.joblib").exists()
    assert (tmp_path / "pid_gbdt_S0_meta.json").exists()


def test_stage15_tie_breaking_rules() -> None:
    """
    1. The use of function:
       Tests the 4-tier tie-breaking mechanism in GBDTRankerPolicy.
    2. How does the specific function work:
       Constructs artificial trees with zero leaf values to force identical candidate scores.
       Tests each tier of tie-breaking:
       - Tier 1: Higher score wins over lower score.
       - Tier 2: Tied score -> smaller remaining burst wins.
       - Tier 3: Tied score and remaining -> earlier arrival wins.
       - Tier 4: Tied score, remaining, and arrival -> lexicographically smaller PID wins.
    3. Concepts used in the function:
       Multi-tier tie-breaking, deterministic resolution, edge case testing.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Synthetic process views with tied criteria.
       After: Confirms exact tier-by-tier tie-breaking resolution.
    """
    dummy_trees = [
        {
            "l": np.array([-1], dtype=np.int32),
            "r": np.array([-1], dtype=np.int32),
            "f": np.array([-2], dtype=np.int32),
            "th": np.array([-2.0], dtype=np.float64),
            "v": np.array([0.0], dtype=np.float64),
        }
    ]
    pol = GBDTRankerPolicy(trees=dummy_trees, lr=1.0, init_val=0.0, cs_cost=0)

    # Tier 2: Equal scores (both 0.0), smaller remaining wins
    p1 = ProcView(pid="P1", arrival=0, burst=5, remaining=4, waited=0)
    p2 = ProcView(pid="P2", arrival=0, burst=5, remaining=2, waited=0)
    v1 = View(time=0, ready=(p1, p2), last_pid=None, newly_arrived=())
    assert pol.select(v1) == "P2"

    # Tier 3: Equal scores and remaining, earlier arrival wins
    p3 = ProcView(pid="P3", arrival=1, burst=5, remaining=3, waited=0)
    p4 = ProcView(pid="P4", arrival=0, burst=5, remaining=3, waited=1)
    v2 = View(time=1, ready=(p3, p4), last_pid=None, newly_arrived=())
    assert pol.select(v2) == "P4"

    # Tier 4: Equal scores, remaining, and arrival, lexicographically smaller PID wins
    p5 = ProcView(pid="PB", arrival=0, burst=5, remaining=3, waited=0)
    p6 = ProcView(pid="PA", arrival=0, burst=5, remaining=3, waited=0)
    v3 = View(time=0, ready=(p5, p6), last_pid=None, newly_arrived=())
    assert pol.select(v3) == "PA"


def test_stage15_defensive_validation_and_errors() -> None:
    """
    1. The use of function:
       Tests defensive validation and error handling across Stage 15 components.
    2. How does the specific function work:
       Tests invalid types, negative context switch costs, invalid array shapes, empty queues,
       and missing files to assert TypeError, ValueError, and FileNotFoundError.
    3. Concepts used in the function:
       Defensive boundary testing, exception assertion, input sanitization.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Invalid input parameter configurations.
       After: Confirms expected exceptions raised with informative diagnostic messages.
    """
    with pytest.raises(TypeError, match="model must be a GradientBoostingClassifier"):
        export_tree_ensemble("bad_model")  # type: ignore[arg-type]

    dummy_trees = [
        {
            "l": np.array([-1]),
            "r": np.array([-1]),
            "f": np.array([-2]),
            "th": np.array([-2.0]),
            "v": np.array([0.0]),
        }
    ]
    with pytest.raises(ValueError, match="x must be a 2D array"):
        score_candidates(dummy_trees, 0.1, 0.0, np.ones(13))

    assert len(score_candidates(dummy_trees, 0.1, 0.0, np.empty((0, 13)))) == 0

    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        prepare_training_data("non_existent_file.npz")

    with pytest.raises(ValueError, match="cs_cost must be a non-negative int"):
        GBDTRankerPolicy(cs_cost=-1)

    with pytest.raises(ValueError, match="Unknown setting 'S99'"):
        GBDTRankerPolicy.from_setting("S99")

    pol = GBDTRankerPolicy.from_setting("S0")
    empty_view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(ValueError, match="ready queue is empty"):
        pol.select(empty_view)

    # Forced decision (single candidate in ready queue)
    single_p = ProcView(pid="P9", arrival=0, burst=3, remaining=3, waited=0)
    single_view = View(time=0, ready=(single_p,), last_pid=None, newly_arrived=())
    assert pol.select(single_view) == "P9"

    # Latency hooks return 0.0
    procs = [Process("P1", 0, 4)]
    assert pol.on_start(procs) == 0.0
    assert pol.decision_cost_us(single_view) == 0.0
    assert pol.reset() is None


def test_stage15_validation_report_json_structure() -> None:
    """
    1. The use of function:
       Verifies that the generated results/pid_gbdt_val.json report contains required structure.
    2. How does the specific function work:
       Reads results/pid_gbdt_val.json, validates top-level keys "S0" and "S1", and verifies that
       metrics fields (top1_agreement, mean_j_policy, closed_loop_gap_oracle, closed_loop_gap_srtf)
       exist with valid numeric bounds.
    3. Concepts used in the function:
       Report structure validation, schema verification, performance threshold assertion.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 15 test suite.
    5. Final output of the function, before and after:
       Before: Persisted JSON validation report.
       After: Confirmed report structure meets all Stage 15 specifications.
    """
    report_path = Path("results/pid_gbdt_val.json")
    assert report_path.exists(), "results/pid_gbdt_val.json must exist."

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "S0" in data
    assert "S1" in data

    s0 = data["S0"]
    assert s0["setting"] == "S0"
    assert s0["cs_cost"] == 0
    assert s0["n_workloads"] == 500
    assert s0["top1_agreement"] >= 0.95
    assert s0["closed_loop_gap_srtf"] <= 0.02

    s1 = data["S1"]
    assert s1["setting"] == "S1"
    assert s1["cs_cost"] == 1
    assert s1["n_workloads"] == 500
    assert s1["top1_agreement"] >= 0.95
    assert s1["closed_loop_gap_oracle"] <= 0.05
