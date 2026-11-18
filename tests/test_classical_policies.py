"""
1. What this file does:
   Comprehensive Stage 2 acceptance test suite validating classical scheduling algorithms
   (FCFS, SJF, SRTF, Round Robin), registry factory integration, domain invariants,
   context switch accounting at cs_cost = 1, and SRTF optimality across 200 random workloads.
2. Overall design:
   Validates Section 4.6 hand-verified fixtures for all 4 policies, the Round Robin arrival tie
   convention, 200 seeded random workloads testing SRTF average waiting time optimality, manual
   context switch counts with cs_cost = 1, named preset acceptance, cost monotonicity, and
   registry factory lookups.
3. Elements in the file:
   - test_section_4_6_hand_verified_fixtures_all_policies: Tests all policies on 4.6 fixture.
   - test_round_robin_tie_convention: Tests quantum expiry vs arrival ordering at tick 2.
   - test_property_tests_200_seeded_random_workloads: Tests 200 workloads for SRTF optimality.
   - test_context_switch_counts_cs_cost_1_manual_fixture: Tests CS counts at cs_cost = 1.
   - test_named_presets_acceptance_across_policies: Tests all presets on all classical policies.
   - test_context_switch_cost_monotonicity: Tests makespan and wait monotonicity with cs_cost.
   - test_registry_policy_instantiation_and_execution: Tests factory lookup and execution.
4. Final output of the file:
   Pytest test suite serving as the final validation gate for Stage 2 acceptance criteria.
"""

import numpy as np
import pytest

from schedsim.domain.engine import Engine, run
from schedsim.domain.process import Process
from schedsim.policies import (
    FCFSPolicy,
    RoundRobinPolicy,
    SJFPolicy,
    SRTFPolicy,
    get_policy,
)
from schedsim.workloads.presets import get_preset, list_presets


def test_section_4_6_hand_verified_fixtures_all_policies() -> None:
    """
    1. The use of function:
       Validates Section 4.6 hand-verified benchmark table for FCFS, SJF, SRTF, and Round Robin.
    2. How does the specific function work:
       Loads workload P1 (arr 0, burst 5), P2 (arr 1, burst 3), P3 (arr 2, burst 2) at cs_cost = 0.
       Runs each policy through the unit-step engine. Asserts exact completions, waiting times,
       and average wait, turnaround, and response times against the Section 4.6 specification.
    3. Concepts used in the function:
       Canonical OS scheduling benchmarks, deterministic scheduling verification, metric validation.
    4. Places where this function is imported and used:
       Executed by pytest test runner as the primary Stage 2 classical policy acceptance test.
    5. Final output of the function, before and after:
       Before: Specification table and workload processes.
       After: Confirms all 4 classical policies match Section 4.6 metrics within 1e-3 tolerance.
    """
    expected_benchmarks = {
        "FCFS": {
            "policy": FCFSPolicy(),
            "completions": {"P1": 5, "P2": 8, "P3": 10},
            "waiting": {"P1": 0, "P2": 4, "P3": 6},
            "avg_wait": 10.0 / 3.0,
            "avg_tat": 20.0 / 3.0,
            "avg_resp": 10.0 / 3.0,
            "segments": [(0, 5, "P1"), (5, 8, "P2"), (8, 10, "P3")],
        },
        "SJF": {
            "policy": SJFPolicy(),
            "completions": {"P1": 5, "P2": 10, "P3": 7},
            "waiting": {"P1": 0, "P2": 6, "P3": 3},
            "avg_wait": 9.0 / 3.0,
            "avg_tat": 19.0 / 3.0,
            "avg_resp": 9.0 / 3.0,
            "segments": [(0, 5, "P1"), (5, 7, "P3"), (7, 10, "P2")],
        },
        "SRTF": {
            "policy": SRTFPolicy(),
            "completions": {"P1": 10, "P2": 4, "P3": 6},
            "waiting": {"P1": 5, "P2": 0, "P3": 2},
            "avg_wait": 7.0 / 3.0,
            "avg_tat": 17.0 / 3.0,
            "avg_resp": 2.0 / 3.0,
            "segments": [(0, 1, "P1"), (1, 4, "P2"), (4, 6, "P3"), (6, 10, "P1")],
        },
        "RR": {
            "policy": RoundRobinPolicy(quantum=2),
            "completions": {"P1": 10, "P2": 9, "P3": 6},
            "waiting": {"P1": 5, "P2": 5, "P3": 2},
            "avg_wait": 12.0 / 3.0,
            "avg_tat": 22.0 / 3.0,
            "avg_resp": 3.0 / 3.0,
            "segments": [
                (0, 2, "P1"),
                (2, 4, "P2"),
                (4, 6, "P3"),
                (6, 8, "P1"),
                (8, 9, "P2"),
                (9, 10, "P1"),
            ],
        },
    }

    for name, bench in expected_benchmarks.items():
        procs = [
            Process("P1", arrival=0, burst=5),
            Process("P2", arrival=1, burst=3),
            Process("P3", arrival=2, burst=2),
        ]
        res = run(bench["policy"], procs, cs_cost=0)
        p_map = {m.pid: m for m in res.process_metrics}

        for pid, exp_comp in bench["completions"].items():
            assert p_map[pid].completion == exp_comp, f"{name} {pid} completion mismatch"
        for pid, exp_wait in bench["waiting"].items():
            assert p_map[pid].waiting == exp_wait, f"{name} {pid} waiting mismatch"

        sim = res.simulation_metrics
        assert sim.avg_waiting == pytest.approx(bench["avg_wait"], abs=1e-3)
        assert sim.avg_turnaround == pytest.approx(bench["avg_tat"], abs=1e-3)
        assert sim.avg_response == pytest.approx(bench["avg_resp"], abs=1e-3)

        actual_segs = [(s.start, s.end, s.label) for s in res.timeline.to_segments()]
        assert actual_segs == bench["segments"], f"{name} timeline segments mismatch"


def test_round_robin_tie_convention() -> None:
    """
    1. The use of function:
       Validates Round Robin arrival tie-breaking convention when quantum expiration coincides
       with arrival.
    2. How does the specific function work:
       Configures P1 (arr 0, burst 4) and P2 (arr 2, burst 2) with quantum = 2.
       Monitors tick 2 where P1's quantum expires and P2 arrives. Verifies newly arrived P2 is
       queued ahead of preempted P1. Confirms P2 executes ticks 2..4 to completion before P1
       resumes at tick 4 to complete at tick 6.
    3. Concepts used in the function:
       Round Robin queue order, simultaneous event tie-breaking, preemption ordering semantics.
    4. Places where this function is imported and used:
       Executed by pytest runner as a dedicated acceptance test for Section 4.4 Round Robin tie.
    5. Final output of the function, before and after:
       Before: Workload exhibiting quantum expiration at arrival tick.
       After: Confirms P2 completes at tick 4 and P1 completes at tick 6 with exact queue ordering.
    """
    procs = [
        Process("P1", arrival=0, burst=4),
        Process("P2", arrival=2, burst=2),
    ]
    rr = RoundRobinPolicy(quantum=2)
    engine = Engine(processes=procs, cs_cost=0)

    # Tick 0: P1 arrives and is dispatched
    v0 = engine.reset()
    assert rr.select(v0) == "P1"
    info1 = engine.step("P1")

    # Tick 1: P1 continues for second quantum tick
    assert rr.select(info1.view) == "P1"
    info2 = engine.step("P1")

    # Tick 2: P1's quantum of 2 expires, P2 arrives exactly at tick 2
    assert info2.view.time == 2
    assert info2.view.newly_arrived == ("P2",)
    selected_at_2 = rr.select(info2.view)

    # Tie-breaking convention: new arrival P2 placed ahead of preempted P1
    assert selected_at_2 == "P2"
    assert rr.queue == ("P1",)

    # Step the selected process (P2) and complete simulation
    curr_info = engine.step(selected_at_2)
    while not engine.done:
        pid = rr.select(curr_info.view)
        curr_info = engine.step(pid)

    res = engine.result()
    p_map = {m.pid: m for m in res.process_metrics}

    # Verify P2 completes strictly before P1
    assert p_map["P2"].completion == 4
    assert p_map["P1"].completion == 6
    assert p_map["P2"].completion < p_map["P1"].completion

    # Verify timeline execution sequence
    segs = [(s.start, s.end, s.label) for s in res.timeline.to_segments()]
    assert segs == [(0, 2, "P1"), (2, 4, "P2"), (4, 6, "P1")]

    # Verify individual process waiting times
    assert p_map["P2"].waiting == 0
    assert p_map["P1"].waiting == 2

    # Also verify that end-to-end run() helper produces identical results
    fresh_procs = [
        Process("P1", arrival=0, burst=4),
        Process("P2", arrival=2, burst=2),
    ]
    e2e_res = run(RoundRobinPolicy(quantum=2), fresh_procs, cs_cost=0)
    e2e_map = {m.pid: m for m in e2e_res.process_metrics}
    assert e2e_map["P2"].completion == 4
    assert e2e_map["P1"].completion == 6
    assert [(s.start, s.end, s.label) for s in e2e_res.timeline.to_segments()] == [
        (0, 2, "P1"),
        (2, 4, "P2"),
        (4, 6, "P1"),
    ]


def test_property_tests_200_seeded_random_workloads() -> None:
    """
    1. The use of function:
       Validates domain invariants and SRTF waiting-time optimality across 200 random workloads.
    2. How does the specific function work:
       Iterates through seeds 0 to 199. For each seed, generates 3 to 8 processes with random
       arrivals [0, 10] and bursts [1, 10] using numpy.random.default_rng. Executes each workload
       with FCFS, SJF, SRTF, and RR (q=2) at cs_cost = 0. Asserts SRTF optimality
       (avg_waiting <= other + 1e-9), full completion, and work conservation.
    3. Concepts used in the function:
       Property-based testing, SRTF theoretical optimality theorem, work conservation invariant.
    4. Places where this function is imported and used:
       Executed by pytest runner as the property testing verification gate of Stage 2.
    5. Final output of the function, before and after:
       Before: Pseudo-random process workloads generated across 200 distinct RNG seeds.
       After: Proves all 800 simulation runs satisfy completeness, conservation, and SRTF.
    """
    for seed in range(200):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 9))
        arrivals = rng.integers(0, 11, size=n)
        bursts = rng.integers(1, 11, size=n)
        procs = [
            Process(f"P{i + 1}", int(arr), int(b))
            for i, (arr, b) in enumerate(zip(arrivals, bursts))
        ]
        total_burst = sum(p.burst for p in procs)

        fcfs_res = run(FCFSPolicy(), procs, cs_cost=0)
        sjf_res = run(SJFPolicy(), procs, cs_cost=0)
        srtf_res = run(SRTFPolicy(), procs, cs_cost=0)
        rr_res = run(RoundRobinPolicy(quantum=2), procs, cs_cost=0)

        all_results = [
            ("FCFS", fcfs_res),
            ("SJF", sjf_res),
            ("SRTF", srtf_res),
            ("RR", rr_res),
        ]

        srtf_avg_wait = srtf_res.simulation_metrics.avg_waiting

        for pol_name, res in all_results:
            # Completeness: all processes completed
            assert len(res.process_metrics) == n, f"Seed {seed} {pol_name}: incomplete completion"

            # Conservation of work: busy ticks equal sum of burst times
            assert res.timeline.busy_ticks == total_burst, (
                f"Seed {seed} {pol_name}: work conservation violated "
                f"({res.timeline.busy_ticks} != {total_burst})"
            )

            # Timeline partition: makespan == busy_ticks + idle_ticks
            assert res.simulation_metrics.makespan == (
                res.timeline.busy_ticks + res.timeline.idle_ticks
            ), f"Seed {seed} {pol_name}: timeline partition mismatch"

            # Metric invariants for each completed process
            for pm in res.process_metrics:
                assert pm.turnaround == pm.completion - pm.arrival
                assert pm.waiting == pm.turnaround - pm.burst
                assert pm.response >= 0

            # SRTF Optimality: SRTF average waiting time is <= every other policy
            assert srtf_avg_wait <= res.simulation_metrics.avg_waiting + 1e-9, (
                f"Seed {seed}: SRTF optimality violated vs {pol_name} "
                f"(SRTF={srtf_avg_wait:.4f}, {pol_name}={res.simulation_metrics.avg_waiting:.4f})"
            )


def test_context_switch_counts_cs_cost_1_manual_fixture() -> None:
    """
    1. The use of function:
       Validates context switch counting and timeline partition invariants at cs_cost = 1.
    2. How does the specific function work:
       Runs Section 4.6 fixture with cs_cost = 1. Verifies FCFS performs exactly 2 context
       switches (makespan 12, overhead 2) and RR (q=2) performs exactly 5 context switches
       (makespan 15, overhead 5). Verifies step penalties equal process waiting times and
       timeline partitions without gaps or overlaps.
    3. Concepts used in the function:
       Context switch overhead, penalty-waiting equivalence, timeline segment continuity.
    4. Places where this function is imported and used:
       Executed by pytest runner as the acceptance test for Section 4.3 context switch semantics.
    5. Final output of the function, before and after:
       Before: Workload executed with non-zero context switch overhead penalty.
       After: Confirms exact context switch counts, overhead ticks, makespans, and penalty sums.
    """
    # 1. Test FCFS with cs_cost = 1
    fcfs_procs = [
        Process("P1", arrival=0, burst=5),
        Process("P2", arrival=1, burst=3),
        Process("P3", arrival=2, burst=2),
    ]
    fcfs_eng = Engine(processes=fcfs_procs, cs_cost=1)
    fcfs_pol = FCFSPolicy()
    fcfs_view = fcfs_eng.reset()
    fcfs_penalties = 0

    while not fcfs_eng.done:
        pid = fcfs_pol.select(fcfs_view)
        step_info = fcfs_eng.step(pid)
        fcfs_penalties += step_info.penalty
        fcfs_view = step_info.view

    fcfs_res = fcfs_eng.result()
    assert fcfs_res.simulation_metrics.context_switches == 2
    assert fcfs_res.simulation_metrics.overhead_ticks == 2
    assert fcfs_res.simulation_metrics.makespan == 12

    # Hand-verified FCFS timeline segments:
    # P1: [0, 5) -> CS: [5, 6) -> P2: [6, 9) -> CS: [9, 10) -> P3: [10, 12)
    expected_fcfs_segs = [
        (0, 5, "P1"),
        (5, 6, "CS"),
        (6, 9, "P2"),
        (9, 10, "CS"),
        (10, 12, "P3"),
    ]
    actual_fcfs_segs = [(s.start, s.end, s.label) for s in fcfs_res.timeline.to_segments()]
    assert actual_fcfs_segs == expected_fcfs_segs

    # Verify per-process metrics and step penalty equivalence for FCFS
    fcfs_p_map = {m.pid: m for m in fcfs_res.process_metrics}
    assert (
        fcfs_p_map["P1"].completion,
        fcfs_p_map["P2"].completion,
        fcfs_p_map["P3"].completion,
    ) == (5, 9, 12)
    assert (
        fcfs_p_map["P1"].waiting,
        fcfs_p_map["P2"].waiting,
        fcfs_p_map["P3"].waiting,
    ) == (0, 5, 8)
    assert fcfs_penalties == sum(m.waiting for m in fcfs_res.process_metrics)
    assert fcfs_penalties == 13

    # 2. Test Round Robin (q=2) with cs_cost = 1
    rr_procs = [
        Process("P1", arrival=0, burst=5),
        Process("P2", arrival=1, burst=3),
        Process("P3", arrival=2, burst=2),
    ]
    rr_eng = Engine(processes=rr_procs, cs_cost=1)
    rr_pol = RoundRobinPolicy(quantum=2)
    rr_view = rr_eng.reset()
    rr_penalties = 0

    while not rr_eng.done:
        pid = rr_pol.select(rr_view)
        step_info = rr_eng.step(pid)
        rr_penalties += step_info.penalty
        rr_view = step_info.view

    rr_res = rr_eng.result()
    assert rr_res.simulation_metrics.context_switches == 5
    assert rr_res.simulation_metrics.overhead_ticks == 5
    assert rr_res.simulation_metrics.makespan == 15

    # Hand-verified RR timeline segments:
    # P1: [0, 2) -> CS: [2, 3) -> P2: [3, 5) -> CS: [5, 6) -> P3: [6, 8) ->
    # CS: [8, 9) -> P1: [9, 11) -> CS: [11, 12) -> P2: [12, 13) -> CS: [13, 14) -> P1: [14, 15)
    expected_rr_segs = [
        (0, 2, "P1"),
        (2, 3, "CS"),
        (3, 5, "P2"),
        (5, 6, "CS"),
        (6, 8, "P3"),
        (8, 9, "CS"),
        (9, 11, "P1"),
        (11, 12, "CS"),
        (12, 13, "P2"),
        (13, 14, "CS"),
        (14, 15, "P1"),
    ]
    actual_rr_segs = [(s.start, s.end, s.label) for s in rr_res.timeline.to_segments()]
    assert actual_rr_segs == expected_rr_segs

    # Verify per-process metrics and step penalty equivalence for RR
    rr_p_map = {m.pid: m for m in rr_res.process_metrics}
    assert (
        rr_p_map["P1"].completion,
        rr_p_map["P2"].completion,
        rr_p_map["P3"].completion,
    ) == (15, 13, 8)
    assert (
        rr_p_map["P1"].waiting,
        rr_p_map["P2"].waiting,
        rr_p_map["P3"].waiting,
    ) == (10, 9, 4)
    assert rr_penalties == sum(m.waiting for m in rr_res.process_metrics)
    assert rr_penalties == 23

    # Invariants check for both runs
    for res in (fcfs_res, rr_res):
        segs = res.timeline.to_segments()
        assert segs[0].start == 0
        assert segs[-1].end == res.simulation_metrics.makespan
        for i in range(len(segs) - 1):
            assert segs[i].end == segs[i + 1].start
        assert res.timeline.busy_ticks == 10


def test_named_presets_acceptance_across_policies() -> None:
    """
    1. The use of function:
       Validates all standard named preset workloads across all four classical scheduling policies.
    2. How does the specific function work:
       Iterates through all presets from list_presets(). For each preset, instantiates fresh
       Process instances and executes FCFS, SJF, SRTF, and RR (q=2). Asserts clean completion,
       work conservation, valid timeline segmentation without gaps or overlaps, and metric rules.
    3. Concepts used in the function:
       Preset workload validation, domain invariant testing, cross-policy acceptance testing.
    4. Places where this function is imported and used:
       Executed by pytest runner to verify simulator robustness on canonical scenarios.
    5. Final output of the function, before and after:
       Before: Canonical named preset definitions.
       After: Verifies every preset executes to completion across all classical policies.
    """
    available_presets = list(list_presets().keys())
    assert len(available_presets) >= 7

    classical_policies = [
        ("FCFS", lambda: FCFSPolicy()),
        ("SJF", lambda: SJFPolicy()),
        ("SRTF", lambda: SRTFPolicy()),
        ("RR", lambda: RoundRobinPolicy(quantum=2)),
    ]

    for preset_name in available_presets:
        procs = get_preset(preset_name)
        total_burst = sum(p.burst for p in procs)
        n_procs = len(procs)

        for pol_name, pol_factory in classical_policies:
            policy = pol_factory()
            res = run(policy, procs, cs_cost=0)

            # Completeness: all processes finished
            assert len(res.process_metrics) == n_procs, (
                f"Preset '{preset_name}' under {pol_name}: expected {n_procs} completions, "
                f"got {len(res.process_metrics)}"
            )

            # Work conservation: busy ticks match cumulative bursts
            assert res.timeline.busy_ticks == total_burst, (
                f"Preset '{preset_name}' under {pol_name}: busy ticks "
                f"({res.timeline.busy_ticks}) != burst sum ({total_burst})"
            )

            # Timeline partition validity
            segs = res.timeline.to_segments()
            assert segs, f"Preset '{preset_name}' under {pol_name} produced empty timeline"
            assert segs[0].start == 0
            assert segs[-1].end == res.simulation_metrics.makespan
            for i in range(len(segs) - 1):
                assert segs[i].end == segs[i + 1].start, (
                    f"Preset '{preset_name}' under {pol_name}: segment gap or overlap at "
                    f"index {i} ({segs[i].end} != {segs[i + 1].start})"
                )

            # Metric algebraic invariants
            for pm in res.process_metrics:
                assert pm.turnaround == pm.completion - pm.arrival
                assert pm.waiting == pm.turnaround - pm.burst
                assert pm.response >= 0


def test_context_switch_cost_monotonicity() -> None:
    """
    1. The use of function:
       Verifies non-zero context switch cost monotonically increases makespan and waiting times.
    2. How does the specific function work:
       Executes Section 4.6 and Convoy presets with cs_cost = 0 and cs_cost = 1 for all classical
       policies. Asserts makespan(cs_cost=1) >= makespan(cs_cost=0) and avg_waiting(cs_cost=1) >=
       avg_waiting(cs_cost=0), and confirms overhead_ticks == cs_cost * context_switches.
    3. Concepts used in the function:
       Monotonicity under scheduling overhead, context switch penalty accounting.
    4. Places where this function is imported and used:
       Executed by pytest runner as an invariant test for context switch cost scaling.
    5. Final output of the function, before and after:
       Before: Identical workloads evaluated under cs_cost = 0 and cs_cost = 1.
       After: Proves non-negative overhead scaling across all policies.
    """
    test_presets = ["section_4_6", "convoy"]
    classical_factories = [
        ("FCFS", lambda: FCFSPolicy()),
        ("SJF", lambda: SJFPolicy()),
        ("SRTF", lambda: SRTFPolicy()),
        ("RR", lambda: RoundRobinPolicy(quantum=2)),
    ]

    for p_name in test_presets:
        procs = get_preset(p_name)
        for pol_name, pol_factory in classical_factories:
            r0 = run(pol_factory(), procs, cs_cost=0)
            r1 = run(pol_factory(), procs, cs_cost=1)

            # Non-zero context switch cost must not decrease makespan
            assert r1.simulation_metrics.makespan >= r0.simulation_metrics.makespan, (
                f"{pol_name} on {p_name}: makespan decreased with cs_cost=1 "
                f"({r1.simulation_metrics.makespan} < {r0.simulation_metrics.makespan})"
            )

            # Non-zero context switch cost must not decrease average waiting time
            assert r1.simulation_metrics.avg_waiting >= r0.simulation_metrics.avg_waiting, (
                f"{pol_name} on {p_name}: avg_waiting decreased with cs_cost=1 "
                f"({r1.simulation_metrics.avg_waiting} < {r0.simulation_metrics.avg_waiting})"
            )

            # Overhead ticks must equal context_switches * cs_cost
            assert (
                r1.simulation_metrics.overhead_ticks == r1.simulation_metrics.context_switches * 1
            )
            assert r0.simulation_metrics.overhead_ticks == 0


def test_registry_policy_instantiation_and_execution() -> None:
    """
    1. The use of function:
       Verifies that registered classical policy names and aliases instantiate and execute.
    2. How does the specific function work:
       Queries get_policy with canonical names and aliases ('fcfs', 'FCFS', 'sjf', 'SJF', 'srtf',
       'SRTF', 'rr', 'RR', 'roundrobin', 'RoundRobin'). Runs each instantiated policy on the Section
       4.6 workload and verifies execution success.
    3. Concepts used in the function:
       Registry lookup, case-insensitivity, factory instantiation, end-to-end policy execution.
    4. Places where this function is imported and used:
       Executed by pytest runner to verify registry integration across all classical policies.
    5. Final output of the function, before and after:
       Before: String identifiers for classical scheduling policies.
       After: Confirms successful retrieval, instantiation, and execution for all registered names.
    """
    test_queries = [
        ("fcfs", {}),
        ("FCFS", {}),
        ("sjf", {}),
        ("SJF", {}),
        ("srtf", {}),
        ("SRTF", {}),
        ("rr", {"quantum": 2}),
        ("RR", {"quantum": 2}),
        ("roundrobin", {"quantum": 2}),
        ("RoundRobin", {"quantum": 2}),
    ]

    for name, kwargs in test_queries:
        policy = get_policy(name, **kwargs)
        procs = get_preset("section_4_6")
        res = run(policy, procs, cs_cost=0)
        assert len(res.process_metrics) == 3
        assert res.timeline.busy_ticks == 10
        assert res.simulation_metrics.makespan == 10
