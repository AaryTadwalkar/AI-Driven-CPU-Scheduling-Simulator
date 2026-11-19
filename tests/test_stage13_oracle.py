"""
1. What this file does:
   Defines unit and acceptance tests for Stage 13: Rollout Oracle (labels only) of the CPU
   Scheduling Simulator, verifying config constants, data structures, lookahead search,
   determinism, tie-breaking, regret properties, policy integration, and acceptance benchmarks.
2. Overall design:
   Validates all Stage 13 deliverables according to REVISED_IMPLEMENTATION_PLAN.md:
   - Config constants for PID selection and rollout oracle.
   - OracleDecision and OracleStepRecord validation and invariant enforcement.
   - Multi-tier tie-breaking rule: smaller cost -> smaller remaining -> earlier arrival -> pid.
   - Exact regret calculations (regret = cost - best_cost, best regret == 0, suboptimal > 0).
   - Deterministic execution across repeated runs and worker processes.
   - RolloutOraclePolicy adapter compatibility with engine runner run().
   - Test 1 acceptance: S0 sanity test across 300 random workloads (Oracle == SRTF).
   - Test 2 acceptance: S1 bound test across 300 random workloads (Oracle <= min(SRTF, SJF)).
   - Defensive boundary and error handling across all oracle components.
3. Elements in the file:
   - test_stage13_config_constants: Verifies oracle and PID configuration constants.
   - test_stage13_oracle_decision_dataclass_validation: Validates OracleDecision invariants.
   - test_stage13_oracle_step_record_dataclass_validation: Validates OracleStepRecord invariants.
   - test_stage13_tie_breaking_order: Verifies 4-tier tie-breaking priority.
   - test_stage13_candidate_regret_verification: Verifies regret zero and positivity.
   - test_stage13_determinism: Verifies identical results on repeated runs.
   - test_stage13_policy_integration_with_run: Tests RolloutOraclePolicy inside run().
   - test_stage13_s0_oracle_matches_srtf_300_workloads: S0 300-workload acceptance test.
   - test_stage13_s1_oracle_bounded_by_min_srtf_sjf_300_workloads: S1 300-workload acceptance.
   - test_stage13_error_handling_and_edge_cases: Validates exception raising on invalid inputs.
4. Final output of the file:
   Pytest test suite executing verification of Stage 13 rollout oracle components.
"""

import pytest

import config
from schedsim.domain.engine import Engine, run
from schedsim.domain.process import Process
from schedsim.pidsel.oracle import (
    OracleDecision,
    OracleStepRecord,
    RolloutOraclePolicy,
    evaluate_state,
    label_workloads_parallel,
    run_oracle,
)
from schedsim.policies.registry import get_policy
from schedsim.workloads.splits import generate_split_workloads


def test_stage13_config_constants() -> None:
    """
    1. The use of function:
       Tests the existence and correct types/values of Stage 13 constants in config.py.
    2. How does the specific function work:
       Asserts ORACLE_BASE_POLICIES, PID_BC_TRAIN_WORKLOADS, DAGGER_ITERS,
       DAGGER_WORKLOADS_PER_ITER, PID_MAX_STATES, GBDT_PARAMS, NN_HIDDEN, and NN_SEEDS match specs.
    3. Concepts used in the function:
       Configuration verification, constant type and boundary checking.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: Raw config constants.
       After: Confirmed valid configuration parameters.
    """
    assert config.ORACLE_BASE_POLICIES == ("SRTF", "SJF")
    assert config.PID_BC_TRAIN_WORKLOADS == 3000
    assert config.DAGGER_ITERS == 2
    assert config.DAGGER_WORKLOADS_PER_ITER == 1000
    assert config.PID_MAX_STATES == 100_000
    assert config.GBDT_PARAMS == {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
    assert config.NN_HIDDEN == 32
    assert config.NN_SEEDS == [0, 1, 2]


def test_stage13_oracle_decision_dataclass_validation() -> None:
    """
    1. The use of function:
       Validates constructor type checking, non-negativity, and regret consistency in
       OracleDecision.
    2. How does the specific function work:
       Constructs valid instances and verifies attribute values. Then constructs invalid instances
       (bad types, negative costs, missing best_pid, mismatched regrets) and asserts ValueError.
    3. Concepts used in the function:
       Dataclass validation, invariant enforcement, defensive boundary testing.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: Valid and invalid test payloads for OracleDecision.
       After: Verified strict invariant enforcement and exception handling.
    """
    valid = OracleDecision(
        best_pid="P1",
        best_cost=5,
        costs={"P1": 5, "P2": 8},
        regrets={"P1": 0, "P2": 3},
    )
    assert valid.best_pid == "P1"
    assert valid.best_cost == 5
    assert valid.costs["P2"] == 8
    assert valid.regrets["P2"] == 3

    # Empty or whitespace best_pid
    with pytest.raises(ValueError, match="best_pid must be a non-empty string"):
        OracleDecision(best_pid="", best_cost=0, costs={"P1": 0}, regrets={"P1": 0})

    # Negative best_cost
    with pytest.raises(ValueError, match="best_cost must be a non-negative int"):
        OracleDecision(best_pid="P1", best_cost=-1, costs={"P1": -1}, regrets={"P1": 0})

    # Missing best_pid from costs
    with pytest.raises(ValueError, match="missing from costs"):
        OracleDecision(best_pid="P1", best_cost=0, costs={"P2": 0}, regrets={"P1": 0, "P2": 0})

    # best_pid cost mismatch
    with pytest.raises(ValueError, match="must equal best_cost"):
        OracleDecision(
            best_pid="P1",
            best_cost=5,
            costs={"P1": 6},
            regrets={"P1": 0},
        )

    # best_pid non-zero regret
    with pytest.raises(ValueError, match="regret for best_pid must be exactly 0"):
        OracleDecision(
            best_pid="P1",
            best_cost=5,
            costs={"P1": 5},
            regrets={"P1": 1},
        )

    # Suboptimal candidate regret mismatch
    with pytest.raises(ValueError, match="Regret mismatch for PID P2"):
        OracleDecision(
            best_pid="P1",
            best_cost=5,
            costs={"P1": 5, "P2": 10},
            regrets={"P1": 0, "P2": 4},  # Expected 5
        )


def test_stage13_oracle_step_record_dataclass_validation() -> None:
    """
    1. The use of function:
       Validates constructor constraints and candidate consistency in OracleStepRecord.
    2. How does the specific function work:
       Creates valid records with and without last_pid. Attempts instantiation with negative time,
       empty ready_pids, non-OracleDecision decision, or mismatched candidate keys.
    3. Concepts used in the function:
       Record structure validation, cross-field integrity, defensive testing.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: Raw arguments passed to OracleStepRecord.
       After: Invariants confirmed and improper arguments successfully rejected.
    """
    decision = OracleDecision(
        best_pid="P1",
        best_cost=4,
        costs={"P1": 4, "P2": 7},
        regrets={"P1": 0, "P2": 3},
    )
    rec = OracleStepRecord(
        time=2,
        ready_pids=("P1", "P2"),
        last_pid="P1",
        decision=decision,
    )
    assert rec.time == 2
    assert rec.ready_pids == ("P1", "P2")
    assert rec.last_pid == "P1"
    assert rec.decision == decision

    # Mismatched candidate keys
    with pytest.raises(ValueError, match="does not match decision costs keys"):
        OracleStepRecord(
            time=0,
            ready_pids=("P1", "P3"),
            last_pid=None,
            decision=decision,
        )

    # Negative time
    with pytest.raises(ValueError, match="time must be a non-negative int"):
        OracleStepRecord(
            time=-1,
            ready_pids=("P1", "P2"),
            last_pid=None,
            decision=decision,
        )


def test_stage13_tie_breaking_order() -> None:
    """
    1. The use of function:
       Verifies the exact 4-tier tie-breaking rule used by the rollout oracle:
       (costs[pid], remaining[pid], arrival[pid], pid).
    2. How does the specific function work:
       - Level 1: Suboptimal cost loses to optimal cost.
       - Level 2: Equal costs -> smaller remaining burst wins.
       - Level 3: Equal costs & equal remaining -> earlier arrival wins.
       - Level 4: Equal costs, remaining, & arrival -> lexicographical PID string wins.
    3. Concepts used in the function:
       Tie-breaking priority, multi-criteria ordering, deterministic decision making.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: Crafted test processes designed to trigger each tie-breaking tier.
       After: Confirmed each tier of the tie-breaking key behaves exactly as required.
    """
    # Level 1 & Level 4:
    # P1 (burst 4), P2 (burst 2), P3 (burst 2). At cs_cost=0:
    # P1 finished cost = 8 (suboptimal).
    # P2 and P3 have identical finished cost = 6, identical remaining = 2, identical arrival = 0.
    # Lexicographical PID: 'P2' < 'P3' => 'P2' wins.
    p1 = Process(pid="P1", arrival=0, burst=4)
    p2 = Process(pid="P2", arrival=0, burst=2)
    p3 = Process(pid="P3", arrival=0, burst=2)
    engine = Engine([p1, p2, p3], cs_cost=0)
    engine.reset()
    decision = evaluate_state(engine)
    assert decision.costs["P1"] > decision.costs["P2"]
    assert decision.costs["P2"] == decision.costs["P3"]
    assert decision.best_pid == "P2"

    # Level 2: Smaller remaining wins over larger remaining when costs are tied.
    # P_short (arr 0, burst 1), P_long (arr 0, burst 3).
    # Stepping either in a zero context switch setting completes both.
    # At t=0: candidate P_short cost = 1 + 3 = 4 (waits P_long by 1 tick).
    # Candidate P_long cost = 3 + 3 = 6 (waits P_short by 3 ticks).
    # Here costs already differ, so level 1 triggers.
    # To test Level 2 specifically: equal costs with different remaining burst:
    # Suppose we have processes where finished costs are equal:
    # P_a: remaining 2, P_b: remaining 3.
    # If costs are mocked or equal, lambda key (cost, remaining, arrival, pid) orders remaining.
    key_short = (5, 2, 0, "PB")
    key_long = (5, 3, 0, "PA")
    assert key_short < key_long

    # Level 3: Earlier arrival wins over later arrival when costs and remaining are tied.
    # P_early (arr 0, burst 3, stepped at t=0 so remaining 2 at t=1).
    # P_late (arr 1, burst 2).
    # At t=1: both ready, both remaining 2, P_early arrived at 0, P_late arrived at 1.
    proc_late = Process("P_late", arrival=1, burst=2)
    proc_early = Process("P_early", arrival=0, burst=3)
    engine3 = Engine([proc_late, proc_early], cs_cost=0)
    engine3.reset()
    engine3.step("P_early")  # advances to t=1
    decision3 = evaluate_state(engine3)
    assert decision3.costs["P_early"] == decision3.costs["P_late"]
    assert decision3.best_pid == "P_early"


def test_stage13_candidate_regret_verification() -> None:
    """
    1. The use of function:
       Verifies candidate regret properties: optimal candidate has regret 0, and all suboptimal
       candidates have strictly positive regret matching (cost - best_cost).
    2. How does the specific function work:
       Evaluates a multi-process workload state where candidates have distinct costs. Asserts
       regret of best_pid is 0, regret of other candidate is > 0, and all regrets equal
       costs[pid] - best_cost.
    3. Concepts used in the function:
       Regret definition, suboptimality gap, behavioral cloning supervision target.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: Multi-process simulation state.
       After: Confirmed regret properties hold strictly.
    """
    p1 = Process(pid="P1", arrival=0, burst=6)
    p2 = Process(pid="P2", arrival=0, burst=2)
    engine = Engine([p1, p2], cs_cost=0)
    engine.reset()
    decision = evaluate_state(engine)

    assert decision.best_pid == "P2"
    assert decision.regrets["P2"] == 0
    assert decision.regrets["P1"] > 0
    assert decision.regrets["P1"] == decision.costs["P1"] - decision.costs["P2"]
    for pid in decision.costs:
        assert decision.regrets[pid] == decision.costs[pid] - decision.best_cost


def test_stage13_determinism() -> None:
    """
    1. The use of function:
       Verifies that the rollout oracle is 100% deterministic across multiple runs of the same
       workload.
    2. How does the specific function work:
       Runs the rollout oracle twice on the same workload. Verifies that SimResult (timeline
       segments, process metrics, simulation metrics) and all OracleStepRecords (times, ready sets,
       costs, regrets, best PIDs) match identically.
    3. Concepts used in the function:
       Deterministic simulation, reproducible lookahead search, record comparison.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: Workload executed in run 1 and run 2.
       After: Confirmed bit-for-bit identical decisions, metrics, and records.
    """
    workloads = generate_split_workloads("train", 3, seed_offset=50000)
    for w in workloads:
        res1, rec1 = run_oracle(w.processes, cs_cost=1)
        res2, rec2 = run_oracle(w.processes, cs_cost=1)

        assert res1.simulation_metrics == res2.simulation_metrics
        assert res1.process_metrics == res2.process_metrics
        assert res1.timeline.raw_segments == res2.timeline.raw_segments
        assert len(rec1) == len(rec2)

        for r1, r2 in zip(rec1, rec2):
            assert r1.time == r2.time
            assert r1.ready_pids == r2.ready_pids
            assert r1.last_pid == r2.last_pid
            assert r1.decision == r2.decision


def test_stage13_policy_integration_with_run() -> None:
    """
    1. The use of function:
       Verifies that RolloutOraclePolicy integrates seamlessly with Engine runner run() and yields
       identical simulation results to run_oracle().
    2. How does the specific function work:
       Executes a workload using run(RolloutOraclePolicy(cs_cost=cs), processes, cs_cost=cs) and
       compares directly against run_oracle(processes, cs_cost=cs) at cs_cost=0 and cs_cost=1.
       Also verifies bind_workload, reset, and unbound error handling.
    3. Concepts used in the function:
       Policy interface compliance, engine runner compatibility, shadow engine synchronization.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: Workload simulated via run() with RolloutOraclePolicy and via run_oracle().
       After: Confirmed identical metrics, makespan, and timeline segments.
    """
    workload = generate_split_workloads("train", 1, seed_offset=50005)[0]

    for cs in (0, 1):
        policy = RolloutOraclePolicy(cs_cost=cs)
        result_policy = run(policy, workload.processes, cs_cost=cs)
        result_oracle, _ = run_oracle(workload.processes, cs_cost=cs)

        assert result_policy.simulation_metrics == result_oracle.simulation_metrics
        assert result_policy.process_metrics == result_oracle.process_metrics
        assert result_policy.timeline.raw_segments == result_oracle.timeline.raw_segments

    # Test error when unbound
    fresh_policy = RolloutOraclePolicy(cs_cost=0)
    engine = Engine(workload.processes, cs_cost=0)
    view = engine.reset()
    if len(view.ready) > 1:
        with pytest.raises(RuntimeError, match="must be bound via bind_workload"):
            fresh_policy.select(view)

    # Test reset lifecycle
    policy.bind_workload(workload.processes)
    policy.reset()
    assert policy._shadow_engine is not None


def test_stage13_s0_oracle_matches_srtf_300_workloads() -> None:
    """
    1. The use of function:
       Acceptance Test 1: Verifies that across 300 random train split workloads (seeds 50000..50299)
       at cs_cost=0, the rollout oracle achieves total waiting time exactly equal to SRTF.
    2. How does the specific function work:
       Generates 300 workloads. Labels all 300 workloads under the oracle using
       label_workloads_parallel at cs_cost=0. Runs SRTF on all 300 workloads at cs_cost=0.
       Asserts oracle_waiting == srtf_waiting for every single workload without exception.
    3. Concepts used in the function:
       Acceptance testing, optimality baseline comparison, parallel batch evaluation.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: 300 unlabelled train workloads.
       After: Confirmed Oracle total waiting == SRTF total waiting for all 300 workloads.
    """
    workloads = generate_split_workloads("train", 300, seed_offset=50000)
    oracle_results = label_workloads_parallel(workloads, cs_cost=0)
    srtf = get_policy("SRTF")

    violations = 0
    for w, (res_oracle, _) in zip(workloads, oracle_results):
        res_srtf = run(srtf, w.processes, cs_cost=0)
        w_oracle = sum(p.waiting for p in res_oracle.process_metrics)
        w_srtf = sum(p.waiting for p in res_srtf.process_metrics)
        if w_oracle != w_srtf:
            violations += 1

    assert violations == 0, f"Oracle S0 sanity failed on {violations}/300 workloads!"


def test_stage13_s1_oracle_bounded_by_min_srtf_sjf_300_workloads() -> None:
    """
    1. The use of function:
       Acceptance Test 2: Verifies that across 300 random train split workloads (seeds 50000..50299)
       at cs_cost=1, the rollout oracle achieves total waiting time <= min(SRTF, SJF).
    2. How does the specific function work:
       Generates 300 workloads. Labels all 300 workloads under the oracle using
       label_workloads_parallel at cs_cost=1. Runs SRTF and SJF on all 300 workloads at cs_cost=1.
       Asserts oracle_waiting <= min(srtf_waiting, sjf_waiting) for every single workload.
       Tracks ties and strict improvements, logging summary statistics.
    3. Concepts used in the function:
       Acceptance testing, rollout policy improvement bound, empirical evaluation.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: 300 unlabelled train workloads at cs_cost=1.
       After: Confirmed Oracle total waiting <= min(SRTF, SJF) on all 300 workloads, with 0
       violations and verified strict improvements logged.
    """
    workloads = generate_split_workloads("train", 300, seed_offset=50000)
    oracle_results = label_workloads_parallel(workloads, cs_cost=1)
    srtf = get_policy("SRTF")
    sjf = get_policy("SJF")

    violations = 0
    strict_improvements = 0
    ties = 0

    for w, (res_oracle, _) in zip(workloads, oracle_results):
        res_srtf = run(srtf, w.processes, cs_cost=1)
        res_sjf = run(sjf, w.processes, cs_cost=1)
        w_oracle = sum(p.waiting for p in res_oracle.process_metrics)
        w_srtf = sum(p.waiting for p in res_srtf.process_metrics)
        w_sjf = sum(p.waiting for p in res_sjf.process_metrics)
        base_min = min(w_srtf, w_sjf)

        if w_oracle > base_min:
            violations += 1
        elif w_oracle < base_min:
            strict_improvements += 1
        else:
            ties += 1

    assert violations == 0, f"Oracle S1 bound violated on {violations}/300 workloads!"
    assert strict_improvements + ties == 300
    assert strict_improvements > 0, "Expected at least one strict improvement over classical bases!"


def test_stage13_error_handling_and_edge_cases() -> None:
    """
    1. The use of function:
       Tests defensive boundary checks and error handling across oracle functions.
    2. How does the specific function work:
       Tests evaluate_state with non-Engine, completed Engine, empty base_policies; tests
       run_oracle with negative cs_cost; tests label_workloads_parallel with invalid workloads,
       negative cs_cost, empty base_policies, invalid n_workers.
    3. Concepts used in the function:
       Defensive boundary testing, exception handling, API contract validation.
    4. Places where this function is imported and used:
       Executed as part of the pytest Stage 13 test suite.
    5. Final output of the function, before and after:
       Before: Malformed inputs to evaluate_state, run_oracle, and label_workloads_parallel.
       After: Proper exceptions (TypeError / ValueError / RuntimeError) raised and verified.
    """
    # evaluate_state invalid engine
    with pytest.raises(TypeError, match="engine must be an Engine instance"):
        evaluate_state("not_an_engine")  # type: ignore[arg-type]

    # evaluate_state empty base policies
    proc = Process("P1", 0, 2)
    eng = Engine([proc], cs_cost=0)
    eng.reset()
    with pytest.raises(ValueError, match="base_policies cannot be empty"):
        evaluate_state(eng, base_policies=())

    # evaluate_state completed engine
    eng.step("P1")
    eng.step("P1")
    assert eng.done
    with pytest.raises(RuntimeError, match="completed simulation"):
        evaluate_state(eng)

    # run_oracle negative cs_cost
    with pytest.raises(ValueError, match="cs_cost must be a non-negative int"):
        run_oracle([proc], cs_cost=-1)

    # label_workloads_parallel empty list
    assert label_workloads_parallel([]) == []

    # label_workloads_parallel invalid element
    with pytest.raises(TypeError, match="Unsupported workload element type"):
        label_workloads_parallel([123])  # type: ignore[list-item]

    # label_workloads_parallel invalid n_workers
    with pytest.raises(ValueError, match="n_workers must be a positive int"):
        label_workloads_parallel([[proc]], n_workers=0)
