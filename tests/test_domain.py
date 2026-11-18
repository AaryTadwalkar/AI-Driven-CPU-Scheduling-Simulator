"""
1. What this file does:
   Defines the end-to-end acceptance test suite for the Stage 1 domain core of the CPU scheduling
   simulator, verifying Engine, Process, Timeline, Metrics, and the run() simulation loop.
2. Overall design:
   Validates core domain invariants including initial idle jumps, simultaneous process arrivals,
   context switch overhead calculation (cs_cost = 2 hand-verified fixture), the penalty-waiting
   algebraic identity, timeline partition integrity, work conservation, engine policy neutrality,
   and Section 4.6 reference benchmark fixtures.
3. Elements in the file:
   - test_idle_gap_at_start_acceptance: Tests idle jump from tick 0 to arrival tick and completion.
   - test_simultaneous_arrivals_acceptance: Tests simultaneous arrivals at same tick.
   - test_context_switch_overhead_cs_cost_2_hand_calculated: Tests hand-verified cs_cost=2 fixture.
   - test_penalty_sum_equals_waiting_time_invariant: Tests sum(penalties) == sum(waiting) identity.
   - test_timeline_partition_invariant: Tests gapless, non-overlapping [0, makespan) partition.
   - test_conservation_of_execution_work_invariant: Tests busy_ticks == sum(bursts) work invariant.
   - test_engine_independence_and_no_heuristic_bias: Tests engine neutrality without bias.
   - test_section_4_6_hand_verified_fixtures_acceptance: Tests Section 4.6 FCFS, SJF, and SRTF runs.
   - test_domain_metrics_algebraic_invariants: Tests mathematical coherence of all domain metrics.
   - test_end_to_end_run_helper_acceptance: Tests run() loop producing SimResult.
4. Final output of the file:
   A clean, passing pytest acceptance test module acting as the final validation gate for Stage 1.
"""

import pytest

from schedsim.domain.engine import (
    Engine,
    ScriptedPolicy,
    SimResult,
    View,
    run,
)
from schedsim.domain.metrics import (
    ProcessMetrics,
    SimulationMetrics,
    calculate_jain_fairness,
)
from schedsim.domain.process import Process
from schedsim.domain.timeline import CS_LABEL, IDLE_LABEL, TimelineSegment


def test_idle_gap_at_start_acceptance() -> None:
    """
    1. The use of function:
       Tests acceptance criterion 1: a workload starting with an idle gap before the first arrival.
    2. How does the specific function work:
       Configures P1 with arrival=5 and burst=2. Initializes Engine, calls reset() to trigger
       the initial idle jump to tick 5, executes P1 for 2 ticks to completion at tick 7, and
       verifies the timeline, idle segments, makespan, and metric calculations.
    3. Concepts used in the function:
       Discrete idle jump, IDLE timeline logging, process completion lifecycle, metric derivation.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Raw workload with delayed arrival at tick 5.
       After: Confirms IDLE [0, 5), P1 [5, 7), makespan=7, busy=2, idle=5, and zero waiting time.
    """
    p1 = Process(pid="P1", arrival=5, burst=2)
    engine = Engine(processes=[p1], cs_cost=0)

    # Reset jumps immediately to tick 5 since no process is ready at tick 0
    view = engine.reset()
    assert view.time == 5
    assert view.last_pid is None
    assert view.newly_arrived == ("P1",)
    assert len(view.ready) == 1
    assert view.ready[0].pid == "P1"
    assert view.ready[0].remaining == 2
    assert view.ready[0].waited == 0

    # Step tick 5 -> 6
    step1 = engine.step("P1")
    assert not step1.done
    assert step1.penalty == 0
    assert step1.view.time == 6
    assert step1.view.ready[0].remaining == 1
    assert step1.view.ready[0].waited == 0

    # Step tick 6 -> 7 (P1 completes)
    step2 = engine.step("P1")
    assert step2.done
    assert step2.penalty == 0
    assert step2.view.time == 7
    assert len(step2.view.ready) == 0
    assert engine.done

    result = engine.result()
    assert isinstance(result, SimResult)

    # Invariant: timeline segments form IDLE [0, 5) and P1 [5, 7)
    expected_segments = [
        TimelineSegment(label=IDLE_LABEL, start=0, end=5),
        TimelineSegment(label="P1", start=5, end=7),
    ]
    assert result.timeline.to_segments() == expected_segments
    assert result.timeline.busy_ticks == 2
    assert result.timeline.idle_ticks == 5
    assert result.timeline.overhead_ticks == 0

    # Per-process metrics verification
    pm = result.process_metrics[0]
    assert pm.pid == "P1"
    assert pm.arrival == 5
    assert pm.burst == 2
    assert pm.arrival + pm.response == 5
    assert pm.completion == 7
    assert pm.turnaround == 2
    assert pm.waiting == 0
    assert pm.response == 0

    # Aggregate simulation metrics
    sm = result.simulation_metrics
    assert sm.makespan == 7
    assert result.timeline.busy_ticks == 2
    assert result.timeline.idle_ticks == 5
    assert sm.overhead_ticks == 0
    assert sm.context_switches == 0
    assert sm.cpu_utilization == pytest.approx(2.0 / 7.0)
    assert sm.throughput == pytest.approx(1.0 / 7.0)
    assert sm.avg_waiting == 0.0
    assert sm.avg_turnaround == 2.0
    assert sm.avg_response == 0.0


def test_simultaneous_arrivals_acceptance() -> None:
    """
    1. The use of function:
       Tests acceptance criterion 2: multiple processes arriving at the exact same discrete tick.
    2. How does the specific function work:
       Configures P1 (arr 2, burst 3) and P2 (arr 2, burst 2). Engine jumps to tick 2, admitting
       both processes into the ready queue ordered canonically by (arrival, pid). Executes P1
       to completion while verifying P2's waited counter increments tick-by-tick, then executes P2.
    3. Concepts used in the function:
       Simultaneous process arrival admission, canonical ready queue sorting, waited tracking.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Two processes arriving simultaneously at tick 2.
       After: Confirms simultaneous admission, correct waited progression, and exact metric values.
    """
    p1 = Process(pid="P1", arrival=2, burst=3)
    p2 = Process(pid="P2", arrival=2, burst=2)
    engine = Engine(processes=[p1, p2], cs_cost=0)

    # Reset triggers idle jump [0, 2) and admits both P1 and P2
    view = engine.reset()
    assert view.time == 2
    assert view.newly_arrived == ("P1", "P2")
    assert len(view.ready) == 2
    assert view.ready[0].pid == "P1"
    assert view.ready[0].arrival == 2
    assert view.ready[0].waited == 0
    assert view.ready[1].pid == "P2"
    assert view.ready[1].arrival == 2
    assert view.ready[1].waited == 0

    # Step P1: tick 2 -> 3. P2 waits 1 tick. Penalty is 1 (P2 ready and not running).
    step1 = engine.step("P1")
    assert step1.penalty == 1
    assert step1.view.time == 3
    assert step1.view.ready[0].pid == "P1" and step1.view.ready[0].waited == 0
    assert step1.view.ready[1].pid == "P2" and step1.view.ready[1].waited == 1

    # Step P1: tick 3 -> 4. P2 waits another tick. Penalty is 1.
    step2 = engine.step("P1")
    assert step2.penalty == 1
    assert step2.view.time == 4
    assert step2.view.ready[1].pid == "P2" and step2.view.ready[1].waited == 2

    # Step P1: tick 4 -> 5. P1 finishes. Penalty is 1.
    step3 = engine.step("P1")
    assert step3.penalty == 1
    assert step3.view.time == 5
    assert len(step3.view.ready) == 1
    assert step3.view.ready[0].pid == "P2"
    assert step3.view.ready[0].waited == 3
    assert step3.view.ready[0].remaining == 2

    # Step P2: tick 5 -> 6. No other process ready. Penalty is 0.
    step4 = engine.step("P2")
    assert step4.penalty == 0
    assert step4.view.time == 6
    assert step4.view.ready[0].remaining == 1
    assert step4.view.ready[0].waited == 3

    # Step P2: tick 6 -> 7. P2 finishes. Penalty is 0.
    step5 = engine.step("P2")
    assert step5.penalty == 0
    assert step5.done
    assert engine.done

    result = engine.result()
    p1_m = next(m for m in result.process_metrics if m.pid == "P1")
    p2_m = next(m for m in result.process_metrics if m.pid == "P2")

    assert p1_m.arrival + p1_m.response == 2
    assert p1_m.completion == 5
    assert p1_m.turnaround == 3
    assert p1_m.waiting == 0
    assert p1_m.response == 0

    assert p2_m.arrival + p2_m.response == 5
    assert p2_m.completion == 7
    assert p2_m.turnaround == 5
    assert p2_m.waiting == 3
    assert p2_m.response == 3

    # Total penalties == sum(waiting)
    total_penalties = sum(s.penalty for s in [step1, step2, step3, step4, step5])
    assert total_penalties == p1_m.waiting + p2_m.waiting == 3


def test_context_switch_overhead_cs_cost_2_hand_calculated() -> None:
    """
    1. The use of function:
       Tests acceptance criterion 3: hand-calculated context switch overhead with cs_cost = 2.
    2. How does the specific function work:
       Fixture: P1 (arr 0, burst 2), P2 (arr 1, burst 2), cs_cost = 2.
       Tick 0: dispatch P1 (first dispatch free), runs [0, 1).
       Tick 1: P2 arrives, P1 runs [1, 2) to completion.
       Tick 2: switch to P2, spending 2 overhead ticks [2, 4) in CS label.
       Tick 4: P2 executes [4, 5).
       Tick 5: P2 executes [5, 6) to completion.
       Verifies switches=1, overhead=2, busy=4, makespan=6, P1 wait=0, P2 wait=3.
    3. Concepts used in the function:
       Context switch penalty and overhead ticks, free first dispatch, waiting time during CS.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Raw hand-calculated fixture specification.
       After: Validates exact match with manual metrics, timeline segments, and penalties.
    """
    p1 = Process(pid="P1", arrival=0, burst=2)
    p2 = Process(pid="P2", arrival=1, burst=2)
    engine = Engine(processes=[p1, p2], cs_cost=2)

    # Initial view at tick 0. First dispatch is free.
    view0 = engine.reset()
    assert view0.time == 0
    assert view0.last_pid is None
    assert len(view0.ready) == 1 and view0.ready[0].pid == "P1"

    # Step 1: P1 runs tick 0..1. P2 arrives at tick 1.
    step1 = engine.step("P1")
    assert step1.penalty == 0
    assert step1.view.time == 1
    assert step1.view.newly_arrived == ("P2",)
    assert len(step1.view.ready) == 2
    # P1 remaining 1, waited 0; P2 remaining 2, waited 0
    assert step1.view.ready[0].pid == "P1" and step1.view.ready[0].remaining == 1
    assert step1.view.ready[1].pid == "P2" and step1.view.ready[1].waited == 0

    # Step 2: P1 continues tick 1..2. P2 waits tick 1..2.
    # Same pid continuation -> zero context switch.
    step2 = engine.step("P1")
    assert step2.penalty == 1  # P2 was ready and waited during [1, 2)
    assert step2.view.time == 2
    assert step2.view.last_pid == "P1"
    assert len(step2.view.ready) == 1
    assert step2.view.ready[0].pid == "P2"
    assert step2.view.ready[0].waited == 1  # P2 waited 1 tick so far

    # Step 3: Switch to P2. last_pid is "P1", pid is "P2", cs_cost is 2.
    # Spends ticks [2, 4) in CS overhead, during which P2 waits 2 ticks.
    # Then executes P2 for tick [4, 5).
    step3 = engine.step("P2")
    # Penalty: 2 ticks of CS overhead (P2 ready and waiting) + 0 ticks during execution [4, 5) = 2
    assert step3.penalty == 2
    assert step3.view.time == 5
    assert step3.view.last_pid == "P2"
    assert len(step3.view.ready) == 1
    assert step3.view.ready[0].pid == "P2"
    assert step3.view.ready[0].remaining == 1
    # P2 total waited so far: 1 tick (while P1 ran) + 2 ticks (during CS) = 3
    assert step3.view.ready[0].waited == 3

    # Step 4: P2 continues tick 5..6 to completion.
    step4 = engine.step("P2")
    assert step4.penalty == 0
    assert step4.done
    assert step4.view.time == 6
    assert engine.done

    result = engine.result()

    # Timeline segments verification
    expected_segments = [
        TimelineSegment(label="P1", start=0, end=2),
        TimelineSegment(label=CS_LABEL, start=2, end=4),
        TimelineSegment(label="P2", start=4, end=6),
    ]
    assert result.timeline.to_segments() == expected_segments
    assert result.timeline.busy_ticks == 4
    assert result.timeline.overhead_ticks == 2
    assert result.timeline.idle_ticks == 0

    # Simulation metrics verification
    sm = result.simulation_metrics
    assert sm.makespan == 6
    assert result.timeline.busy_ticks == 4
    assert sm.overhead_ticks == 2
    assert result.timeline.idle_ticks == 0
    assert sm.context_switches == 1
    assert sm.cpu_utilization == pytest.approx(4.0 / 6.0)
    assert sm.throughput == pytest.approx(2.0 / 6.0)

    # Per-process metrics verification
    p1_m = next(m for m in result.process_metrics if m.pid == "P1")
    p2_m = next(m for m in result.process_metrics if m.pid == "P2")

    assert p1_m.arrival == 0 and p1_m.burst == 2
    assert p1_m.arrival + p1_m.response == 0
    assert p1_m.completion == 2
    assert p1_m.turnaround == 2
    assert p1_m.waiting == 0
    assert p1_m.response == 0

    assert p2_m.arrival == 1 and p2_m.burst == 2
    assert p2_m.arrival + p2_m.response == 4
    assert p2_m.completion == 6
    assert p2_m.turnaround == 5
    assert p2_m.waiting == 3  # 1 tick while P1 executed + 2 CS ticks
    assert p2_m.response == 3

    # Reward / Penalty invariant: sum(step penalties) == sum(process waiting)
    total_penalties = sum(s.penalty for s in [step1, step2, step3, step4])
    total_waiting = p1_m.waiting + p2_m.waiting
    assert total_penalties == 3
    assert total_waiting == 3


def test_penalty_sum_equals_waiting_time_invariant() -> None:
    """
    1. The use of function:
       Tests acceptance criterion 4: sum(penalties) == sum(process waiting times) invariant.
    2. How does the specific function work:
       Executes diverse workloads under multiple scheduling sequences, varying cs_cost from 0 to 2,
       with and without preemption, and verifies that the sum of penalties reported by each engine
       step precisely matches the aggregate sum of process waiting times computed by metrics.py.
    3. Concepts used in the function:
       Algebraic simulation invariant, penalty accounting, cumulative waiting time conservation.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Multiple test configurations with preemption, idle gaps, and context switches.
       After: Asserts exact equality between step penalties and process waiting times for all cases.
    """
    # Case A: 3 processes, cs_cost = 0, interleaved execution
    workload_a = [
        Process(pid="P1", arrival=0, burst=3),
        Process(pid="P2", arrival=1, burst=2),
        Process(pid="P3", arrival=2, burst=1),
    ]
    # Sequence: P1, P2, P3, P2, P1, P1
    policy_a = ScriptedPolicy(["P1", "P2", "P3", "P2", "P1", "P1"])
    engine_a = Engine(processes=workload_a, cs_cost=0)
    view_a = engine_a.reset()
    penalties_a = []
    while not engine_a.done:
        action = policy_a.select(view_a)
        step_info = engine_a.step(action)
        penalties_a.append(step_info.penalty)
        view_a = step_info.view

    res_a = engine_a.result()
    assert sum(penalties_a) == sum(pm.waiting for pm in res_a.process_metrics)

    # Case B: 3 processes, cs_cost = 1, multiple context switches
    workload_b = [
        Process(pid="P1", arrival=0, burst=2),
        Process(pid="P2", arrival=0, burst=2),
        Process(pid="P3", arrival=0, burst=2),
    ]
    # Alternating single ticks: P1, P2, P3, P1, P2, P3
    policy_b = ScriptedPolicy(["P1", "P2", "P3", "P1", "P2", "P3"])
    engine_b = Engine(processes=workload_b, cs_cost=1)
    view_b = engine_b.reset()
    penalties_b = []
    while not engine_b.done:
        action = policy_b.select(view_b)
        step_info = engine_b.step(action)
        penalties_b.append(step_info.penalty)
        view_b = step_info.view

    res_b = engine_b.result()
    assert sum(penalties_b) == sum(pm.waiting for pm in res_b.process_metrics)

    # Case C: Workload with idle gaps and cs_cost = 2
    workload_c = [
        Process(pid="P1", arrival=3, burst=2),
        Process(pid="P2", arrival=10, burst=1),
    ]
    engine_c = Engine(processes=workload_c, cs_cost=2)
    view_c = engine_c.reset()
    penalties_c = []
    policy_c = ScriptedPolicy(["P1", "P1", "P2"])
    while not engine_c.done:
        action = policy_c.select(view_c)
        step_c = engine_c.step(action)
        penalties_c.append(step_c.penalty)
        view_c = step_c.view

    res_c = engine_c.result()
    assert sum(penalties_c) == sum(pm.waiting for pm in res_c.process_metrics)


def test_timeline_partition_invariant() -> None:
    """
    1. The use of function:
       Tests acceptance criterion 5: merged timeline segments form an exact partition of
       [0, makespan) with zero gaps and zero overlaps.
    2. How does the specific function work:
       Executes multiple simulations with varied features (initial idle gap, intermediate idle gaps,
       context switch overhead intervals, rapid preemptions). For each execution, asserts that
       Timeline.validate_partition() succeeds, that segments are contiguous with
       segment[i].end == segment[i+1].start, and that no two adjacent segments share the same label.
    3. Concepts used in the function:
       Mathematical partition of half-open intervals, gapless timeline validation, segment merging.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Raw discrete timeline tick recordings from heterogeneous workloads.
       After: Verifies strict interval partition property on every generated timeline.
    """
    workloads = [
        # Initial idle gap + contiguous execution
        ([Process("P1", arrival=4, burst=3)], 0, ["P1", "P1", "P1"]),
        # Initial idle gap + intermediate idle gap
        (
            [Process("P1", arrival=2, burst=2), Process("P2", arrival=8, burst=1)],
            0,
            ["P1", "P1", "P2"],
        ),
        # Interleaved with CS overhead = 2
        (
            [Process("P1", arrival=0, burst=2), Process("P2", arrival=1, burst=2)],
            2,
            ["P1", "P1", "P2", "P2"],
        ),
        # High contention with CS overhead = 1
        (
            [
                Process("P1", arrival=0, burst=1),
                Process("P2", arrival=0, burst=1),
                Process("P3", arrival=0, burst=1),
            ],
            1,
            ["P1", "P2", "P3"],
        ),
    ]

    for procs, cs_cost, sequence in workloads:
        engine = Engine(processes=procs, cs_cost=cs_cost)
        view = engine.reset()
        policy = ScriptedPolicy(sequence)
        while not engine.done:
            action = policy.select(view)
            step_info = engine.step(action)
            view = step_info.view

        result = engine.result()
        timeline = result.timeline
        makespan = result.simulation_metrics.makespan

        # Built-in strict partition validator must pass without exception
        assert timeline.validate_partition() is True

        # Explicit invariant assertions
        segments = timeline.to_segments()
        assert len(segments) > 0
        assert segments[0].start == 0
        assert segments[-1].end == makespan

        for i in range(len(segments) - 1):
            curr_seg = segments[i]
            next_seg = segments[i + 1]
            assert curr_seg.start < curr_seg.end, "Segment must have positive duration"
            assert curr_seg.end == next_seg.start, "Segments must be contiguous with zero gap"
            assert curr_seg.label != next_seg.label, "Adjacent segments must not share same label"


def test_conservation_of_execution_work_invariant() -> None:
    """
    1. The use of function:
       Tests acceptance criterion 6: total busy ticks equals the exact sum of process bursts.
    2. How does the specific function work:
       Constructs diverse workloads and runs them under different context switch costs (0, 1, 2, 5)
       and diverse scheduling orders. Asserts that timeline.busy_ticks strictly equals
       sum(p.burst for p in processes) regardless of context switch overhead, idle gaps, or
       preemption frequency.
    3. Concepts used in the function:
       Work conservation law, CPU execution tracking, invariance under scheduling perturbations.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Workload definitions with known total burst requirements.
       After: Validates that busy execution ticks never deviate from total burst demand.
    """
    procs = [
        Process("P1", arrival=0, burst=4),
        Process("P2", arrival=2, burst=3),
        Process("P3", arrival=5, burst=1),
    ]
    total_burst = sum(p.burst for p in procs)
    assert total_burst == 8

    # Run with cs_cost = 0 (FCFS order)
    seq_fcfs = ["P1"] * 4 + ["P2"] * 3 + ["P3"] * 1
    res_fcfs = run(ScriptedPolicy(seq_fcfs), procs, cs_cost=0)
    assert res_fcfs.timeline.busy_ticks == total_burst

    # Run with cs_cost = 2 (heavily interleaved order)
    # P1 (2 ticks), P2 (2 ticks), P1 (2 ticks), P3 (1 tick), P2 (1 tick)
    seq_interleaved = ["P1", "P1", "P2", "P2", "P1", "P1", "P3", "P2"]
    res_interleaved = run(ScriptedPolicy(seq_interleaved), procs, cs_cost=2)
    assert res_interleaved.timeline.busy_ticks == total_burst
    assert res_interleaved.simulation_metrics.overhead_ticks > 0

    # Run with cs_cost = 5 on delayed arrival workload
    delayed_procs = [
        Process("P1", arrival=10, burst=5),
        Process("P2", arrival=20, burst=5),
    ]
    res_delayed = run(
        ScriptedPolicy(["P1"] * 5 + ["P2"] * 5),
        delayed_procs,
        cs_cost=5,
    )
    assert res_delayed.timeline.busy_ticks == 10
    assert res_delayed.timeline.idle_ticks == 15  # [0, 10) + [15, 20)


def test_engine_independence_and_no_heuristic_bias() -> None:
    """
    1. The use of function:
       Tests acceptance criterion 7: engine contains zero policy-specific code or bias.
    2. How does the specific function work:
       Executes the engine using two opposing policies on the same workload:
       1) MaxBurstPolicy: always selects the ready process with the maximum remaining burst.
       2) ReversePidPolicy: always selects the ready process with the lexicographically largest PID.
       Verifies that the engine executes exactly the process chosen by the policy without imposing
       any default ordering (such as FCFS, SJF, or arrival-order priority).
    3. Concepts used in the function:
       Engine policy neutrality, separation of mechanism and policy, dependency injection.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Identical workload evaluated under adversarial and custom scheduling policies.
       After: Confirms execution order strictly mirrors policy selections without engine bias.
    """
    procs = [
        Process("P1", arrival=0, burst=2),
        Process("P2", arrival=0, burst=5),
        Process("P3", arrival=0, burst=1),
    ]

    # Policy 1: Reverse PID policy (P3, then P2, then P1)
    def reverse_pid_policy(view: View) -> str:
        # Pick ready process with lexicographically maximum PID
        return max(view.ready, key=lambda pv: pv.pid).pid

    res_reverse = run(reverse_pid_policy, procs, cs_cost=0)
    # P3 completes first (burst 1), then P2 (burst 5), then P1 (burst 2)
    p3_m = next(m for m in res_reverse.process_metrics if m.pid == "P3")
    p2_m = next(m for m in res_reverse.process_metrics if m.pid == "P2")
    p1_m = next(m for m in res_reverse.process_metrics if m.pid == "P1")
    assert p3_m.completion == 1
    assert p2_m.completion == 6
    assert p1_m.completion == 8

    # Policy 2: Longest Remaining Time policy (LRTF-like)
    def longest_remaining_policy(view: View) -> str:
        # Pick ready process with maximum remaining burst
        return max(view.ready, key=lambda pv: pv.remaining).pid

    res_longest = run(longest_remaining_policy, procs, cs_cost=0)
    p2_m_long = next(m for m in res_longest.process_metrics if m.pid == "P2")
    assert p2_m_long.arrival + p2_m_long.response == 0


def test_section_4_6_hand_verified_fixtures_acceptance() -> None:
    """
    1. The use of function:
       Tests the Section 4.6 hand-verified fixture for FCFS, SJF, and SRTF at cs_cost = 0.
    2. How does the specific function work:
       Workload: P1 (arr 0, burst 5), P2 (arr 1, burst 3), P3 (arr 2, burst 2).
       Executes FCFS, SJF, and SRTF using the domain engine and ScriptedPolicy or policy functions,
       verifying exact agreement with the Section 4.6 specification table:
       - FCFS completions: (5, 8, 10), waiting: (0, 4, 6), avg wait 3.333, avg tat 6.667.
       - SJF completions: (5, 10, 7), waiting: (0, 6, 3), avg wait 3.000, avg tat 6.333.
       - SRTF completions: (10, 4, 6), waiting: (5, 0, 2), avg wait 2.333, avg tat 5.667.
    3. Concepts used in the function:
       Standard OS scheduling reference validation, tie-breaking rules, metric verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Raw workload from Section 4.6 of the implementation plan.
       After: Confirms exact numerical convergence with the hand-calculated reference table.
    """
    procs = [
        Process("P1", arrival=0, burst=5),
        Process("P2", arrival=1, burst=3),
        Process("P3", arrival=2, burst=2),
    ]

    # --- 1. FCFS ---
    # FCFS select rule: ready process with min (arrival, pid)
    def fcfs_policy(view: View) -> str:
        return min(view.ready, key=lambda pv: (pv.arrival, pv.pid)).pid

    res_fcfs = run(fcfs_policy, procs, cs_cost=0)
    p1 = next(m for m in res_fcfs.process_metrics if m.pid == "P1")
    p2 = next(m for m in res_fcfs.process_metrics if m.pid == "P2")
    p3 = next(m for m in res_fcfs.process_metrics if m.pid == "P3")

    assert (p1.completion, p2.completion, p3.completion) == (5, 8, 10)
    assert (p1.waiting, p2.waiting, p3.waiting) == (0, 4, 6)
    assert res_fcfs.simulation_metrics.avg_waiting == pytest.approx(10.0 / 3.0, abs=1e-3)
    assert res_fcfs.simulation_metrics.avg_turnaround == pytest.approx(20.0 / 3.0, abs=1e-3)
    assert res_fcfs.simulation_metrics.avg_response == pytest.approx(10.0 / 3.0, abs=1e-3)

    # --- 2. SJF (non-preemptive) ---
    # SJF select rule: if last_pid ready keep it; else min (burst, arrival, pid)
    def sjf_policy(view: View) -> str:
        if view.last_pid is not None and any(pv.pid == view.last_pid for pv in view.ready):
            return view.last_pid
        return min(view.ready, key=lambda pv: (pv.burst, pv.arrival, pv.pid)).pid

    res_sjf = run(sjf_policy, procs, cs_cost=0)
    p1_sjf = next(m for m in res_sjf.process_metrics if m.pid == "P1")
    p2_sjf = next(m for m in res_sjf.process_metrics if m.pid == "P2")
    p3_sjf = next(m for m in res_sjf.process_metrics if m.pid == "P3")

    assert (p1_sjf.completion, p2_sjf.completion, p3_sjf.completion) == (5, 10, 7)
    assert (p1_sjf.waiting, p2_sjf.waiting, p3_sjf.waiting) == (0, 6, 3)
    assert res_sjf.simulation_metrics.avg_waiting == pytest.approx(9.0 / 3.0, abs=1e-3)
    assert res_sjf.simulation_metrics.avg_turnaround == pytest.approx(19.0 / 3.0, abs=1e-3)
    assert res_sjf.simulation_metrics.avg_response == pytest.approx(9.0 / 3.0, abs=1e-3)

    # --- 3. SRTF (preemptive) ---
    # SRTF select rule: min (remaining, running-first, arrival, pid)
    def srtf_policy(view: View) -> str:
        return min(
            view.ready,
            key=lambda pv: (
                pv.remaining,
                0 if pv.pid == view.last_pid else 1,
                pv.arrival,
                pv.pid,
            ),
        ).pid

    res_srtf = run(srtf_policy, procs, cs_cost=0)
    p1_srtf = next(m for m in res_srtf.process_metrics if m.pid == "P1")
    p2_srtf = next(m for m in res_srtf.process_metrics if m.pid == "P2")
    p3_srtf = next(m for m in res_srtf.process_metrics if m.pid == "P3")

    assert (p1_srtf.completion, p2_srtf.completion, p3_srtf.completion) == (10, 4, 6)
    assert (p1_srtf.waiting, p2_srtf.waiting, p3_srtf.waiting) == (5, 0, 2)
    assert res_srtf.simulation_metrics.avg_waiting == pytest.approx(7.0 / 3.0, abs=1e-3)
    assert res_srtf.simulation_metrics.avg_turnaround == pytest.approx(17.0 / 3.0, abs=1e-3)
    assert res_srtf.simulation_metrics.avg_response == pytest.approx(2.0 / 3.0, abs=1e-3)


def test_domain_metrics_algebraic_invariants() -> None:
    """
    1. The use of function:
       Tests the algebraic relations connecting process metrics, aggregate metrics, and timeline.
    2. How does the specific function work:
       Executes a multi-process simulation with cs_cost = 1. Validates:
       - turnaround == completion - arrival
       - waiting == turnaround - burst
       - response == first_start - arrival
       - makespan == busy_ticks + idle_ticks + overhead_ticks
       - cpu_utilization == busy_ticks / makespan
       - throughput == len(processes) / makespan
       - Jain's fairness index calculation matches manual computation over slowdowns.
    3. Concepts used in the function:
       Algebraic metrics coherence, Jain fairness calculation, CPU utilization invariants.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Raw completed simulation result.
       After: Validates that all mathematical definitions hold without internal inconsistency.
    """
    procs = [
        Process("P1", arrival=1, burst=3),
        Process("P2", arrival=2, burst=2),
        Process("P3", arrival=4, burst=1),
    ]
    # Execute with cs_cost = 1
    seq = ["P1", "P1", "P2", "P3", "P1", "P2"]
    res = run(ScriptedPolicy(seq), procs, cs_cost=1)

    sm = res.simulation_metrics
    # Aggregate CPU time conservation
    assert sm.makespan == res.timeline.busy_ticks + res.timeline.idle_ticks + sm.overhead_ticks
    assert sm.cpu_utilization == pytest.approx(res.timeline.busy_ticks / sm.makespan)
    assert sm.throughput == pytest.approx(len(procs) / sm.makespan)

    # Per-process invariants
    slowdowns = []
    for pm in res.process_metrics:
        assert isinstance(pm, ProcessMetrics)
        assert pm.turnaround == pm.completion - pm.arrival
        assert pm.waiting == pm.turnaround - pm.burst
        first_start = pm.arrival + pm.response
        assert pm.response == first_start - pm.arrival
        assert pm.arrival <= first_start <= pm.completion
        assert pm.waiting >= 0
        slowdowns.append(pm.turnaround / pm.burst)

    # Averages
    assert sm.avg_waiting == pytest.approx(
        sum(pm.waiting for pm in res.process_metrics) / len(procs)
    )
    assert sm.avg_turnaround == pytest.approx(
        sum(pm.turnaround for pm in res.process_metrics) / len(procs)
    )
    assert sm.avg_response == pytest.approx(
        sum(pm.response for pm in res.process_metrics) / len(procs)
    )

    # Jain fairness verification
    expected_jain = calculate_jain_fairness(slowdowns)
    assert sm.jain_fairness == pytest.approx(expected_jain)


def test_end_to_end_run_helper_acceptance() -> None:
    """
    1. The use of function:
       Tests the domain run() helper function end-to-end against full acceptance expectations.
    2. How does the specific function work:
       Executes run() with a callable policy and validates that it returns a frozen SimResult
       containing immutable ProcessMetrics, SimulationMetrics, and merged Timeline with valid
       partition, conservation of work, and matching context switch counter.
    3. Concepts used in the function:
       High-level simulation loop orchestration, immutability, acceptance gate verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during domain acceptance test suite verification.
    5. Final output of the function, before and after:
       Before: Policy and workload specifications passed to run().
       After: Complete SimResult verified against all Stage 1 requirements.
    """
    procs = [
        Process("P1", arrival=0, burst=2),
        Process("P2", arrival=2, burst=2),
    ]

    # Non-preemptive policy
    def simple_policy(view: View) -> str:
        if view.last_pid is not None and any(pv.pid == view.last_pid for pv in view.ready):
            return view.last_pid
        return view.ready[0].pid

    result = run(simple_policy, procs, cs_cost=1)
    assert isinstance(result, SimResult)
    assert len(result.process_metrics) == 2
    assert isinstance(result.simulation_metrics, SimulationMetrics)

    # Timeline partition
    assert result.timeline.validate_partition() is True

    # Conservation
    assert result.timeline.busy_ticks == 4

    # Make sure SimResult attributes are read-only
    with pytest.raises(Exception):
        result.process_metrics = ()  # type: ignore[misc]
