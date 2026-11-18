"""
1. What this file does:
   Comprehensive test suite validating the Policy abstract base class, classical scheduling
   algorithms (FCFS, SJF, SRTF, Round Robin), and the centralized policy registry.
2. Overall design:
   Validates policy instantiation, parameter boundaries, unit-step select() decisions, tie-breaking
   rules, preemption and non-preemption behavior, queue management, reset hooks, end-to-end
   execution against Section 4.6 benchmarks, policy factory lookup, case-insensitivity, kwargs
   forwarding, error handling, and dynamic policy registration.
3. Elements in the file:
   - test_policy_base_contract: Validates Policy ABC, __call__ delegation, and default reset.
   - test_fcfs_policy_unit_and_e2e: Validates FCFS selection, tie-breaking, and 4.6 fixture.
   - test_sjf_policy_unit_and_e2e: Validates non-preemptive SJF selection, retention, and 4.6.
   - test_srtf_policy_unit_and_e2e: Validates preemptive SRTF selection, tie-breaking, and 4.6.
   - test_round_robin_policy_validation_and_properties: Tests RR parameters, properties, and reset.
   - test_round_robin_hand_verified_4_6_fixture: Validates RR q=2 on Section 4.6 fixture.
   - test_round_robin_arrival_tie_convention: Validates new arrivals take priority over expired q.
   - test_round_robin_early_completion_and_single_process: Tests completion before quantum expiry.
   - test_classical_policies_work_conservation_and_invariants: Tests invariants on workloads.
   - test_round_robin_fallback_when_deque_empty: Tests fallback selection when queue is empty.
   - test_registry_factory_lookup_and_case_insensitivity: Tests policy factory and name matching.
   - test_registry_kwargs_forwarding: Tests passing initialization kwargs through factory.
   - test_registry_unknown_and_validation_errors: Tests error handling on unknown or bad inputs.
   - test_registry_dynamic_registration_and_listing: Tests custom registration and listing.
4. Final output of the file:
   Pytest test module providing full test coverage for schedsim.policies package.
"""

import pytest

from schedsim.domain.engine import Engine, ProcView, View, run
from schedsim.domain.process import Process
from schedsim.policies import (
    FCFSPolicy,
    Policy,
    RoundRobinPolicy,
    SJFPolicy,
    SRTFPolicy,
    get_policy,
    list_policies,
    register_policy,
)
from schedsim.policies.registry import _reset_registry


def test_policy_base_contract() -> None:
    """
    1. The use of function:
       Tests Policy abstract base class enforcement, parameter validation, and callable delegation.
    2. How does the specific function work:
       Verifies that direct instantiation of Policy raises TypeError. Tests concrete subclass
       instantiation with default and custom names, invalid name types/values, reset() default
       no-op, and __call__ delegation to select().
    3. Concepts used in the function:
       Abstract base class enforcement, callable protocol, input boundary testing.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Policy ABC and dummy subclass definitions.
       After: Confirms full adherence to the Policy contract and error raising on violations.
    """
    # Cannot instantiate Policy directly
    with pytest.raises(TypeError):
        Policy()  # type: ignore[abstract]

    class DummyPolicy(Policy):
        """
        1. The use of class: Minimal test fixture implementing Policy ABC.
        2. What the class contain: Simple select() implementation returning first ready PID.
        3. Technologies used and why: Python class inheritance for interface testing.
        4. Key design and overall design of the class: Concrete implementation of abstract Policy.
        """

        def select(self, view: View) -> str:
            return view.ready[0].pid

    # Default name and custom name
    p_default = DummyPolicy()
    assert p_default.name == "Policy"

    p_custom = DummyPolicy(name="CustomDummy")
    assert p_custom.name == "CustomDummy"

    # Invalid names
    with pytest.raises(TypeError, match="Policy name must be a str"):
        DummyPolicy(name=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot be empty"):
        DummyPolicy(name="   ")

    # Callable delegation and reset
    pv = ProcView(pid="P1", arrival=0, burst=5, remaining=5, waited=0)
    v = View(time=0, ready=(pv,), last_pid=None, newly_arrived=("P1",))
    assert p_default(v) == "P1"
    assert p_default.reset() is None


def test_fcfs_policy_unit_and_e2e() -> None:
    """
    1. The use of function:
       Tests FCFSPolicy unit-step selection, tie-breaking rules, and Section 4.6 benchmark results.
    2. How does the specific function work:
       Creates views with multiple ready processes with varying arrival times and identical arrival
       times (testing lexicographical pid tie-breaking). Runs the Section 4.6 benchmark:
       P1 (arr 0, burst 5), P2 (arr 1, burst 3), P3 (arr 2, burst 2). Verifies completions
       (5, 8, 10), waiting times (0, 4, 6), and aggregate averages.
    3. Concepts used in the function:
       First-Come First-Served scheduling, canonical sorting, non-preemption.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Raw process workloads and synthetic views.
       After: Confirms exact numerical convergence with Section 4.6 FCFS reference metrics.
    """
    policy = FCFSPolicy()
    assert policy.name == "FCFS"

    # Empty ready queue raises ValueError
    empty_view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(ValueError, match="ready queue is empty"):
        policy.select(empty_view)

    # Unit selection: earlier arrival wins
    pv1 = ProcView(pid="P1", arrival=3, burst=4, remaining=4, waited=0)
    pv2 = ProcView(pid="P2", arrival=1, burst=4, remaining=4, waited=2)
    view = View(time=3, ready=(pv1, pv2), last_pid=None, newly_arrived=())
    assert policy.select(view) == "P2"

    # Tie-breaking: identical arrival breaks lexicographically by pid
    pv_b = ProcView(pid="PB", arrival=2, burst=3, remaining=3, waited=1)
    pv_a = ProcView(pid="PA", arrival=2, burst=3, remaining=3, waited=1)
    view_tie = View(time=3, ready=(pv_b, pv_a), last_pid=None, newly_arrived=())
    assert policy.select(view_tie) == "PA"

    # End-to-end Section 4.6 fixture
    procs = [
        Process("P1", arrival=0, burst=5),
        Process("P2", arrival=1, burst=3),
        Process("P3", arrival=2, burst=2),
    ]
    res = run(policy, procs, cs_cost=0)
    p_map = {m.pid: m for m in res.process_metrics}

    assert (p_map["P1"].completion, p_map["P2"].completion, p_map["P3"].completion) == (5, 8, 10)
    assert (p_map["P1"].waiting, p_map["P2"].waiting, p_map["P3"].waiting) == (0, 4, 6)
    assert res.simulation_metrics.avg_waiting == pytest.approx(10.0 / 3.0, abs=1e-3)
    assert res.simulation_metrics.avg_turnaround == pytest.approx(20.0 / 3.0, abs=1e-3)
    assert res.simulation_metrics.avg_response == pytest.approx(10.0 / 3.0, abs=1e-3)


def test_sjf_policy_unit_and_e2e() -> None:
    """
    1. The use of function:
       Tests SJFPolicy unit-step selection, non-preemption retention, and Section 4.6 benchmarks.
    2. How does the specific function work:
       Verifies that SJFPolicy retains view.last_pid if it is still ready, even if a shorter burst
       job is present. When last_pid is not ready, it picks minimum (burst, arrival, pid).
       Executes Section 4.6 fixture and verifies completions (5, 10, 7), waiting (0, 6, 3), and
       aggregates.
    3. Concepts used in the function:
       Shortest Job First scheduling, non-preemptive CPU retention, greedy burst optimization.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Raw process workloads and synthetic views.
       After: Confirms exact agreement with Section 4.6 SJF reference metrics.
    """
    policy = SJFPolicy()
    assert policy.name == "SJF"

    # Empty ready queue raises ValueError
    empty_view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(ValueError, match="ready queue is empty"):
        policy.select(empty_view)

    # Non-preemptive retention: last_pid is ready, so it stays
    pv1 = ProcView(pid="P1", arrival=0, burst=5, remaining=3, waited=0)
    pv2 = ProcView(pid="P2", arrival=2, burst=1, remaining=1, waited=0)
    view_retain = View(time=2, ready=(pv1, pv2), last_pid="P1", newly_arrived=("P2",))
    assert policy.select(view_retain) == "P1"

    # Selection when last_pid is None or finished: shortest burst wins
    view_pick = View(time=5, ready=(pv1, pv2), last_pid=None, newly_arrived=())
    assert policy.select(view_pick) == "P2"

    # Tie-breaking: burst equal -> arrival wins; arrival equal -> pid wins
    pv_x = ProcView(pid="PX", arrival=3, burst=4, remaining=4, waited=1)
    pv_y = ProcView(pid="PY", arrival=1, burst=4, remaining=4, waited=3)
    view_tie_arr = View(time=4, ready=(pv_x, pv_y), last_pid=None, newly_arrived=())
    assert policy.select(view_tie_arr) == "PY"

    pv_k1 = ProcView(pid="P2", arrival=1, burst=4, remaining=4, waited=3)
    pv_k2 = ProcView(pid="P1", arrival=1, burst=4, remaining=4, waited=3)
    view_tie_pid = View(time=4, ready=(pv_k1, pv_k2), last_pid=None, newly_arrived=())
    assert policy.select(view_tie_pid) == "P1"

    # End-to-end Section 4.6 fixture
    procs = [
        Process("P1", arrival=0, burst=5),
        Process("P2", arrival=1, burst=3),
        Process("P3", arrival=2, burst=2),
    ]
    res = run(policy, procs, cs_cost=0)
    p_map = {m.pid: m for m in res.process_metrics}

    assert (p_map["P1"].completion, p_map["P2"].completion, p_map["P3"].completion) == (5, 10, 7)
    assert (p_map["P1"].waiting, p_map["P2"].waiting, p_map["P3"].waiting) == (0, 6, 3)
    assert res.simulation_metrics.avg_waiting == pytest.approx(9.0 / 3.0, abs=1e-3)
    assert res.simulation_metrics.avg_turnaround == pytest.approx(19.0 / 3.0, abs=1e-3)
    assert res.simulation_metrics.avg_response == pytest.approx(9.0 / 3.0, abs=1e-3)


def test_srtf_policy_unit_and_e2e() -> None:
    """
    1. The use of function:
       Tests SRTF preemptive selection, running-first tie-breaking, and Section 4.6 benchmarks.
    2. How does the specific function work:
       Tests that a process with strictly shorter remaining burst immediately preempts the running
       process. Verifies that when remaining bursts are equal, the currently running process wins
       the tie to avoid useless context switches. Executes Section 4.6 fixture and verifies
       completions (10, 4, 6), waiting (5, 0, 2), and aggregates.
    3. Concepts used in the function:
       Shortest Remaining Time First scheduling, preemption, anti-churn tie breaking.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Raw process workloads and synthetic views.
       After: Confirms exact agreement with Section 4.6 SRTF reference metrics.
    """
    policy = SRTFPolicy()
    assert policy.name == "SRTF"

    # Empty ready queue raises ValueError
    empty_view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(ValueError, match="ready queue is empty"):
        policy.select(empty_view)

    # Strictly shorter remaining preempts running process
    pv_curr = ProcView(pid="P1", arrival=0, burst=5, remaining=4, waited=0)
    pv_new = ProcView(pid="P2", arrival=1, burst=3, remaining=3, waited=0)
    view_preempt = View(time=1, ready=(pv_curr, pv_new), last_pid="P1", newly_arrived=("P2",))
    assert policy.select(view_preempt) == "P2"

    # Running process wins ties when remaining times are identical
    pv_same1 = ProcView(pid="P1", arrival=0, burst=5, remaining=3, waited=0)
    pv_same2 = ProcView(pid="P2", arrival=1, burst=3, remaining=3, waited=0)
    view_tie_running = View(time=2, ready=(pv_same1, pv_same2), last_pid="P1", newly_arrived=())
    assert policy.select(view_tie_running) == "P1"

    # When last_pid is neither, tie breaks by arrival, then pid
    view_tie_neither = View(time=2, ready=(pv_same2, pv_same1), last_pid=None, newly_arrived=())
    assert policy.select(view_tie_neither) == "P1"

    # End-to-end Section 4.6 fixture
    procs = [
        Process("P1", arrival=0, burst=5),
        Process("P2", arrival=1, burst=3),
        Process("P3", arrival=2, burst=2),
    ]
    res = run(policy, procs, cs_cost=0)
    p_map = {m.pid: m for m in res.process_metrics}

    assert (p_map["P1"].completion, p_map["P2"].completion, p_map["P3"].completion) == (10, 4, 6)
    assert (p_map["P1"].waiting, p_map["P2"].waiting, p_map["P3"].waiting) == (5, 0, 2)
    assert res.simulation_metrics.avg_waiting == pytest.approx(7.0 / 3.0, abs=1e-3)
    assert res.simulation_metrics.avg_turnaround == pytest.approx(17.0 / 3.0, abs=1e-3)
    assert res.simulation_metrics.avg_response == pytest.approx(2.0 / 3.0, abs=1e-3)


def test_round_robin_policy_validation_and_properties() -> None:
    """
    1. The use of function:
       Tests RoundRobinPolicy parameter boundaries, name formatting, accessors, and reset hook.
    2. How does the specific function work:
       Asserts error handling for invalid quantum types (bool, float, str) and values (0, -1).
       Tests property getters (quantum, current_pid, quantum_ticks_consumed, queue), name
       conventions (RR vs RR(q=k) vs custom), and reset() clearing internal state.
    3. Concepts used in the function:
       Defensive validation, state encapsulation, lifecycle management.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Raw RoundRobinPolicy configurations.
       After: Confirms valid parameter enforcement and state inspection properties.
    """
    # Valid parameter defaults and names
    rr_default = RoundRobinPolicy()
    assert rr_default.quantum == 2
    assert rr_default.name == "RR"

    rr_q3 = RoundRobinPolicy(quantum=3)
    assert rr_q3.quantum == 3
    assert rr_q3.name == "RR(q=3)"

    rr_custom = RoundRobinPolicy(quantum=4, name="CustomRR")
    assert rr_custom.quantum == 4
    assert rr_custom.name == "CustomRR"

    # Boundary checks for quantum
    with pytest.raises(TypeError, match="quantum must be an int"):
        RoundRobinPolicy(quantum=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="quantum must be an int"):
        RoundRobinPolicy(quantum="2")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quantum must be >= 1"):
        RoundRobinPolicy(quantum=0)
    with pytest.raises(ValueError, match="quantum must be >= 1"):
        RoundRobinPolicy(quantum=-2)

    # Empty ready queue raises ValueError
    empty_view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(ValueError, match="ready queue is empty"):
        rr_default.select(empty_view)

    # State properties and reset()
    pv1 = ProcView(pid="P1", arrival=0, burst=3, remaining=3, waited=0)
    v1 = View(time=0, ready=(pv1,), last_pid=None, newly_arrived=("P1",))
    selected = rr_default.select(v1)
    assert selected == "P1"
    assert rr_default.current_pid == "P1"
    assert rr_default.quantum_ticks_consumed == 1
    assert rr_default.queue == ()

    rr_default.reset()
    assert rr_default.current_pid is None
    assert rr_default.quantum_ticks_consumed == 0
    assert rr_default.queue == ()


def test_round_robin_hand_verified_4_6_fixture() -> None:
    """
    1. The use of function:
       Tests RoundRobinPolicy (quantum=2) on the Section 4.6 hand-verified fixture.
    2. How does the specific function work:
       Workload: P1 (arr 0, burst 5), P2 (arr 1, burst 3), P3 (arr 2, burst 2).
       Executes simulation with cs_cost = 0. Asserts exact completions (10, 9, 6),
       waiting times (5, 5, 2), avg wait 4.000, avg turnaround 7.333, and avg response 1.000.
    3. Concepts used in the function:
       Round Robin time slicing, Section 4.6 specification compliance, deterministic execution.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Raw Section 4.6 fixture processes.
       After: Confirms exact numerical convergence with Section 4.6 RR reference metrics.
    """
    procs = [
        Process("P1", arrival=0, burst=5),
        Process("P2", arrival=1, burst=3),
        Process("P3", arrival=2, burst=2),
    ]
    rr = RoundRobinPolicy(quantum=2)
    res = run(rr, procs, cs_cost=0)
    p_map = {m.pid: m for m in res.process_metrics}

    assert (p_map["P1"].completion, p_map["P2"].completion, p_map["P3"].completion) == (10, 9, 6)
    assert (p_map["P1"].waiting, p_map["P2"].waiting, p_map["P3"].waiting) == (5, 5, 2)
    assert res.simulation_metrics.avg_waiting == pytest.approx(4.000, abs=1e-3)
    assert res.simulation_metrics.avg_turnaround == pytest.approx(22.0 / 3.0, abs=1e-3)
    assert res.simulation_metrics.avg_response == pytest.approx(1.000, abs=1e-3)

    # Execution: P1 [0, 2), P2 [2, 4), P3 [4, 6), P1 [6, 8), P2 [8, 9), P1 [9, 10)
    seg_labels = [(seg.start, seg.end, seg.label) for seg in res.timeline.to_segments()]
    expected_segments = [
        (0, 2, "P1"),
        (2, 4, "P2"),
        (4, 6, "P3"),
        (6, 8, "P1"),
        (8, 9, "P2"),
        (9, 10, "P1"),
    ]
    assert seg_labels == expected_segments


def test_round_robin_arrival_tie_convention() -> None:
    """
    1. The use of function:
       Tests the Round Robin arrival tie convention: a process arriving at the exact tick a
       quantum expires is placed ahead of the preempted process in the queue.
    2. How does the specific function work:
       Configures P1 (arr 0, burst 4) with quantum = 2. At tick 2, P1's quantum expires and P2
       arrives (arr 2, burst 2). Engine presents newly_arrived = ('P2',). Confirms that P2 is
       scheduled at tick 2 ahead of P1.
    3. Concepts used in the function:
       Simultaneous quantum expiration and process arrival tie-breaking convention.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Workload where arrival coincides with quantum expiration.
       After: Confirms new arrival runs ahead of preempted process.
    """
    procs = [
        Process("P1", arrival=0, burst=4),
        Process("P2", arrival=2, burst=2),
    ]
    rr = RoundRobinPolicy(quantum=2)
    engine = Engine(processes=procs, cs_cost=0)
    v0 = engine.reset()

    # Tick 0: P1 selected, consumed = 1
    assert rr.select(v0) == "P1"
    info1 = engine.step("P1")

    # Tick 1: P1 selected, consumed = 2
    assert rr.select(info1.view) == "P1"
    info2 = engine.step("P1")

    # Tick 2: P1 quantum expires, P2 arrives
    assert info2.view.newly_arrived == ("P2",)
    selected_at_2 = rr.select(info2.view)
    assert selected_at_2 == "P2"
    assert rr.queue == ("P1",)


def test_round_robin_early_completion_and_single_process() -> None:
    """
    1. The use of function:
       Tests Round Robin handling of early process completion and single-process execution.
    2. How does the specific function work:
       Test 1: Single process with burst=5 and quantum=2 completes at tick 5 without stall.
       Test 2: P1 with burst=1 and quantum=3 finishes at tick 1; next ready process P2 is scheduled
       immediately at tick 1 without waiting for quantum 3.
    3. Concepts used in the function:
       Early completion handling, single process execution loop, quantum resetting.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Workloads with early completions and single jobs.
       After: Confirms seamless dispatch without deadlocks or wasted quantum ticks.
    """
    # Single process
    p_single = [Process("P1", arrival=0, burst=5)]
    res_single = run(RoundRobinPolicy(quantum=2), p_single, cs_cost=0)
    assert res_single.process_metrics[0].completion == 5
    assert res_single.simulation_metrics.makespan == 5

    # Early completion
    p_early = [
        Process("P1", arrival=0, burst=1),
        Process("P2", arrival=0, burst=3),
    ]
    rr = RoundRobinPolicy(quantum=3)
    res_early = run(rr, p_early, cs_cost=0)
    p_map = {m.pid: m for m in res_early.process_metrics}
    assert p_map["P1"].completion == 1
    assert p_map["P2"].completion == 4
    assert res_early.simulation_metrics.makespan == 4


def test_classical_policies_work_conservation_and_invariants() -> None:
    """
    1. The use of function:
       Tests domain invariants across all four classical policies on diverse workloads.
    2. How does the specific function work:
       Executes FCFS, SJF, SRTF, and RR on workloads with delays, equal bursts, context switches.
       Verifies:
       - busy_ticks == sum(process bursts) (conservation of work).
       - makespan == busy_ticks + idle_ticks + overhead_ticks (timeline partition).
       - At cs_cost = 0, SRTF average waiting time <= FCFS, SJF, RR average waiting time.
    3. Concepts used in the function:
       Conservation of execution work, timeline partitioning, SRTF theoretical optimality.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Workload definitions with variable arrivals and bursts.
       After: Confirms all invariant assertions hold across all four classical policies.
    """
    workloads = [
        # Idle start + staggered arrivals
        [
            Process("P1", arrival=3, burst=4),
            Process("P2", arrival=5, burst=2),
            Process("P3", arrival=6, burst=1),
        ],
        # Equal arrivals + equal bursts
        [
            Process("P1", arrival=0, burst=3),
            Process("P2", arrival=0, burst=3),
            Process("P3", arrival=0, burst=3),
        ],
        # Convoy scenario
        [
            Process("P1", arrival=0, burst=12),
            Process("P2", arrival=1, burst=1),
            Process("P3", arrival=1, burst=2),
            Process("P4", arrival=2, burst=1),
        ],
    ]

    policies: list[Policy] = [
        FCFSPolicy(),
        SJFPolicy(),
        SRTFPolicy(),
        RoundRobinPolicy(quantum=2),
    ]

    for procs in workloads:
        total_burst = sum(p.burst for p in procs)
        results = {}

        for pol in policies:
            pol.reset()
            res = run(pol, procs, cs_cost=0)
            results[pol.name] = res.simulation_metrics.avg_waiting

            # Conservation invariant
            assert res.timeline.busy_ticks == total_burst
            # Timeline partition invariant
            assert (
                res.timeline.makespan
                == res.timeline.busy_ticks
                + res.timeline.idle_ticks
                + res.timeline.overhead_ticks
            )
            # All processes completed
            assert len(res.process_metrics) == len(procs)
            for pm in res.process_metrics:
                assert pm.completion >= pm.arrival + pm.burst

        # SRTF optimality check at cs_cost = 0
        srtf_avg_wait = results["SRTF"]
        for name, avg_wait in results.items():
            assert srtf_avg_wait <= avg_wait + 1e-9, (
                f"SRTF ({srtf_avg_wait}) must be <= {name} ({avg_wait}) at cs_cost=0"
            )


def test_round_robin_fallback_when_deque_empty() -> None:
    """
    1. The use of function:
       Tests Round Robin fallback path when internal FIFO queue is empty but ready queue has jobs.
    2. How does the specific function work:
       Constructs a View with two ready processes (P1, P2) but empty newly_arrived.
       Asserts that select() invokes the fallback logic, sorting ready processes canonically by
       (arrival, pid), electing P1 as current_pid, and enqueuing remaining ready process P2.
    3. Concepts used in the function:
       Fallback handling, defensive queue population, canonical sorting.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Empty Round Robin internal queue with un-notified ready processes.
       After: Correctly dispatches P1 and enqueues P2 for future execution.
    """
    rr = RoundRobinPolicy(quantum=2)
    pv1 = ProcView(pid="P1", arrival=0, burst=3, remaining=3, waited=0)
    pv2 = ProcView(pid="P2", arrival=1, burst=3, remaining=3, waited=0)
    view = View(time=0, ready=(pv2, pv1), last_pid=None, newly_arrived=())

    assert rr.select(view) == "P1"
    assert rr.current_pid == "P1"
    assert rr.queue == ("P2",)


def test_registry_factory_lookup_and_case_insensitivity() -> None:
    """
    1. The use of function:
       Tests get_policy factory instantiation and case-insensitive policy lookup.
    2. How does the specific function work:
       Validates that canonical names ("FCFS", "SJF", "SRTF", "RR", "RoundRobin") instantiate
       the correct policy types with default arguments. Tests case-insensitivity variants
       ("fcfs", "fCfS", "  sjf  ", "srtf", "rr", "roundrobin", "ROUNDROBIN") and confirms they
       instantiate the expected policy classes.
    3. Concepts used in the function:
       Factory pattern, case-insensitive string resolution, whitespace normalization.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Raw policy name strings in various cases.
       After: Correctly typed and initialized Policy instances.
    """
    # Canonical name lookups
    p_fcfs = get_policy("FCFS")
    assert isinstance(p_fcfs, FCFSPolicy)
    assert p_fcfs.name == "FCFS"

    p_sjf = get_policy("SJF")
    assert isinstance(p_sjf, SJFPolicy)
    assert p_sjf.name == "SJF"

    p_srtf = get_policy("SRTF")
    assert isinstance(p_srtf, SRTFPolicy)
    assert p_srtf.name == "SRTF"

    p_rr = get_policy("RR")
    assert isinstance(p_rr, RoundRobinPolicy)
    assert p_rr.name == "RR"
    assert p_rr.quantum == 2

    p_roundrobin = get_policy("RoundRobin")
    assert isinstance(p_roundrobin, RoundRobinPolicy)
    assert p_roundrobin.name == "RR"
    assert p_roundrobin.quantum == 2

    # Case-insensitivity and whitespace stripping
    assert isinstance(get_policy("fcfs"), FCFSPolicy)
    assert isinstance(get_policy("fCfS"), FCFSPolicy)
    assert isinstance(get_policy("  sjf  "), SJFPolicy)
    assert isinstance(get_policy("SjF"), SJFPolicy)
    assert isinstance(get_policy("srtf"), SRTFPolicy)
    assert isinstance(get_policy("  SRTF "), SRTFPolicy)
    assert isinstance(get_policy("rr"), RoundRobinPolicy)
    assert isinstance(get_policy(" Rr "), RoundRobinPolicy)
    assert isinstance(get_policy("roundrobin"), RoundRobinPolicy)
    assert isinstance(get_policy("ROUNDROBIN"), RoundRobinPolicy)
    assert isinstance(get_policy("  roundRobin  "), RoundRobinPolicy)


def test_registry_kwargs_forwarding() -> None:
    """
    1. The use of function:
       Tests keyword argument forwarding from get_policy to policy constructors.
    2. How does the specific function work:
       Tests passing quantum and name arguments to RoundRobinPolicy via "RR" and "RoundRobin"
       keys. Tests passing custom name argument to FCFSPolicy, SJFPolicy, and SRTFPolicy.
       Confirms that the instantiated policies reflect the passed keyword arguments.
    3. Concepts used in the function:
       Variadic kwargs forwarding, dynamic policy configuration.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Kwargs intended for policy constructors.
       After: Policy instances initialized with the customized attributes.
    """
    # Round Robin with quantum
    p_rr_q5 = get_policy("RR", quantum=5)
    assert isinstance(p_rr_q5, RoundRobinPolicy)
    assert p_rr_q5.quantum == 5
    assert p_rr_q5.name == "RR(q=5)"

    p_rr_custom = get_policy("RoundRobin", quantum=3, name="CustomRR")
    assert isinstance(p_rr_custom, RoundRobinPolicy)
    assert p_rr_custom.quantum == 3
    assert p_rr_custom.name == "CustomRR"

    # Classical policies with custom names
    p_fcfs = get_policy("FCFS", name="MyFCFS")
    assert isinstance(p_fcfs, FCFSPolicy)
    assert p_fcfs.name == "MyFCFS"

    p_sjf = get_policy("SJF", name="MySJF")
    assert isinstance(p_sjf, SJFPolicy)
    assert p_sjf.name == "MySJF"

    p_srtf = get_policy("SRTF", name="MySRTF")
    assert isinstance(p_srtf, SRTFPolicy)
    assert p_srtf.name == "MySRTF"


def test_registry_unknown_and_validation_errors() -> None:
    """
    1. The use of function:
       Tests error handling and diagnostics for unknown policies and invalid arguments.
    2. How does the specific function work:
       Verifies that unknown policy names raise KeyError with a diagnostic message containing
       the available policy list. Asserts that empty strings or whitespace raise KeyError.
       Asserts that non-string policy names raise TypeError.
    3. Concepts used in the function:
       Defensive error raising, diagnostic messaging, input validation.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Invalid or non-existent policy lookup invocations.
       After: Verified KeyError and TypeError exceptions with informative messages.
    """
    # Unknown policy name raises KeyError with available policies in message
    with pytest.raises(KeyError, match=r"Unknown policy 'NonExistent'"):
        get_policy("NonExistent")

    try:
        get_policy("NoSuchPolicy")
    except KeyError as exc:
        msg = str(exc)
        for expected in ("FCFS", "RR", "RoundRobin", "SJF", "SRTF"):
            assert expected in msg

    # Empty and whitespace-only names
    with pytest.raises(KeyError, match=r"Unknown policy ''"):
        get_policy("")
    with pytest.raises(KeyError, match=r"Unknown policy '   '"):
        get_policy("   ")

    # Non-string names raise TypeError
    with pytest.raises(TypeError, match="Policy name must be a str"):
        get_policy(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Policy name must be a str"):
        get_policy(None)  # type: ignore[arg-type]


def test_registry_dynamic_registration_and_listing() -> None:
    """
    1. The use of function:
       Tests dynamic policy registration, registration validation, and list_policies listing.
    2. How does the specific function work:
       Verifies initial list_policies returns the sorted canonical names. Validates register_policy
       input checks (name type, empty name, policy_cls subclassing Policy). Registers a custom
       policy class, verifies its appearance in list_policies, instantiates it via get_policy with
       case-insensitivity, and cleans up registry state using _reset_registry.
    3. Concepts used in the function:
       Dynamic extension, registry lifecycle management, test fixture cleanup.
    4. Places where this function is imported and used:
       Executed by pytest runner during policy suite verification.
    5. Final output of the function, before and after:
       Before: Default registry state.
       After: Confirms dynamic registration works and resets to clean baseline.
    """
    initial_policies = ["FCFS", "RR", "RoundRobin", "SJF", "SRTF"]
    assert list_policies() == initial_policies

    # Validation errors on register_policy
    with pytest.raises(TypeError, match="Policy name must be a str"):
        register_policy(123, FCFSPolicy)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Policy name cannot be empty"):
        register_policy("", FCFSPolicy)
    with pytest.raises(ValueError, match="Policy name cannot be empty"):
        register_policy("   ", FCFSPolicy)
    with pytest.raises(TypeError, match="policy_cls must be a subclass of Policy"):
        register_policy("BadCls", int)  # type: ignore[arg-type]

    class CustomPriorityPolicy(Policy):
        """
        1. The use of class: Test fixture verifying dynamic registration of user-defined policies.
        2. What the class contain: Constructor forwarding custom name and dummy select() method.
        3. Technologies used and why: Subclass of Policy for runtime registry validation.
        4. Key design and overall design of the class: Concrete dummy policy for registry tests.
        """

        def __init__(self, name: str = "Priority") -> None:
            super().__init__(name=name)

        def select(self, view: View) -> str:
            return view.ready[0].pid

    try:
        register_policy("Priority", CustomPriorityPolicy)
        expected_with_custom = ["FCFS", "Priority", "RR", "RoundRobin", "SJF", "SRTF"]
        assert list_policies() == expected_with_custom

        # Case-insensitive lookup of newly registered policy
        p_retrieved = get_policy("priority")
        assert isinstance(p_retrieved, CustomPriorityPolicy)
        assert p_retrieved.name == "Priority"

        p_custom_name = get_policy("PRIORITY", name="CustomName")
        assert isinstance(p_custom_name, CustomPriorityPolicy)
        assert p_custom_name.name == "CustomName"
    finally:
        _reset_registry()

    assert list_policies() == initial_policies

