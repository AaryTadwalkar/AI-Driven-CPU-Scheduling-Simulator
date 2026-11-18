"""
1. What this file does:
   Provides comprehensive unit tests for Engine, ProcView, View, StepInfo, SimResult,
   ScriptedPolicy, and the run() helper in schedsim.domain.engine.
2. Overall design:
   Verifies unit-step simulation execution, discrete time progression, context switch overhead,
   idle time jumping, ready queue sorting, newly arrived tracking, step penalty and waiting time
   algebraic equivalence, error handling, and end-to-end policy execution.
3. Elements in the file:
   - test_proc_view_valid_creation: Tests ProcView initialization and property access.
   - test_proc_view_immutability: Tests frozen immutability of ProcView.
   - test_proc_view_type_and_range_validation: Tests ProcView post-init validation checks.
   - test_view_valid_creation: Tests View initialization and field values.
   - test_view_immutability: Tests frozen immutability of View.
   - test_view_validation: Tests View post-init validation errors.
   - test_step_info_validation: Tests StepInfo creation, immutability, and validation.
   - test_sim_result_validation: Tests SimResult creation, immutability, and validation.
   - test_scripted_policy_sequence_execution: Tests sequence-based ScriptedPolicy.
   - test_scripted_policy_callable_execution: Tests callable-based ScriptedPolicy.
   - test_scripted_policy_validation_and_exhaustion: Tests sequence exhaustion and input errors.
   - test_engine_initialization_and_validation: Tests Engine constructor checks.
   - test_engine_reset_reinitialization: Tests engine reset restoring pristine state.
   - test_single_process_execution_lifecycle: Tests end-to-end execution of a single task.
   - test_initial_idle_jump: Tests initial idle jump when earliest arrival is > 0.
   - test_intermediate_idle_jump: Tests idle jump between disjoint process arrival spans.
   - test_multiple_idle_jumps: Tests multiple disjoint idle gaps.
   - test_context_switch_zero_cost: Tests context switch tracking when cs_cost == 0.
   - test_context_switch_nonzero_cost_and_overhead_ticks: Tests CS ticks when cs_cost > 0.
   - test_context_switch_same_pid_no_switch: Tests that continuing same PID has zero switch.
   - test_context_switch_away_from_completed_process: Tests switch away from finished process.
   - test_idle_gap_preserves_last_pid_for_context_switch: Tests last_pid persistence across idle.
   - test_arrivals_during_context_switch_ticks: Tests process arrival admission during CS ticks.
   - test_ready_queue_and_newly_arrived_ordering: Tests canonical (arrival, pid) ordering.
   - test_penalty_step_and_cumulative_waiting_equivalence: Tests penalty == waiting equivalence.
   - test_run_helper_with_scripted_policy: Tests run() helper with ScriptedPolicy sequence.
   - test_run_helper_with_callable_and_lambda: Tests run() helper with callable functions.
   - test_run_helper_invalid_policy: Tests run() error on non-callable invalid policy.
   - test_engine_step_when_done_raises_runtime_error: Tests RuntimeError on stepping finished run.
   - test_engine_step_invalid_pid_raises_errors: Tests ValueError/TypeError on invalid step PIDs.
   - test_engine_result_before_done_raises_runtime_error: Tests result() on incomplete run.
   - test_section_4_6_fcfs_full_simulation: Tests full FCFS execution from Section 4.6.
   - test_section_4_6_srtf_full_simulation: Tests full SRTF execution from Section 4.6.
4. Final output of the file:
   Pytest suite verifying Engine correctness, invariants, and simulation semantics.
"""

from dataclasses import FrozenInstanceError

import pytest

from schedsim.domain.engine import (
    Engine,
    ProcView,
    ScriptedPolicy,
    SimResult,
    StepInfo,
    View,
    run,
)
from schedsim.domain.metrics import (
    ProcessMetrics,
    SimulationMetrics,
)
from schedsim.domain.process import Process
from schedsim.domain.timeline import CS_LABEL, IDLE_LABEL, Timeline, TimelineSegment


def test_proc_view_valid_creation() -> None:
    """
    1. The use of function:
       Tests successful instantiation and field access of ProcView.
    2. How does the specific function work:
       Constructs a ProcView with valid attributes and verifies each property.
    3. Concepts used in the function:
       Dataclass instantiation, attribute verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Raw valid process view parameters.
       After: Asserts fields match expected values.
    """
    pv = ProcView(
        pid="P1",
        arrival=0,
        burst=5,
        remaining=3,
        waited=2,
        last_run_end=4,
    )
    assert pv.pid == "P1"
    assert pv.arrival == 0
    assert pv.burst == 5
    assert pv.remaining == 3
    assert pv.waited == 2
    assert pv.last_run_end == 4


def test_proc_view_immutability() -> None:
    """
    1. The use of function:
       Verifies that ProcView instances are strictly immutable.
    2. How does the specific function work:
       Attempts attribute modification on a ProcView instance and catches FrozenInstanceError.
    3. Concepts used in the function:
       Dataclass immutability, frozen instance protection.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Immutable ProcView instance.
       After: Confirms FrozenInstanceError is raised upon modification attempt.
    """
    pv = ProcView(pid="P1", arrival=0, burst=4, remaining=4, waited=0)
    with pytest.raises(FrozenInstanceError):
        pv.remaining = 3  # type: ignore[misc]


def test_proc_view_type_and_range_validation() -> None:
    """
    1. The use of function:
       Tests error handling and post-init validation for invalid ProcView parameters.
    2. How does the specific function work:
       Passes invalid types (bool, str, negative numbers, out-of-range remaining) and asserts
       TypeError or ValueError.
    3. Concepts used in the function:
       Boundary validation, type checking, defensive programming.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Malformed ProcView parameter sets.
       After: Confirms appropriate exceptions are raised.
    """
    with pytest.raises(TypeError, match="ProcView pid must be a str"):
        ProcView(pid=123, arrival=0, burst=4, remaining=4, waited=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ProcView pid cannot be empty"):
        ProcView(pid="   ", arrival=0, burst=4, remaining=4, waited=0)

    with pytest.raises(TypeError, match="ProcView arrival must be an int"):
        ProcView(pid="P1", arrival=True, burst=4, remaining=4, waited=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ProcView arrival must be >= 0"):
        ProcView(pid="P1", arrival=-1, burst=4, remaining=4, waited=0)

    with pytest.raises(ValueError, match="ProcView burst must be >= 1"):
        ProcView(pid="P1", arrival=0, burst=0, remaining=0, waited=0)

    with pytest.raises(ValueError, match="must be in range"):
        ProcView(pid="P1", arrival=0, burst=4, remaining=5, waited=0)

    with pytest.raises(ValueError, match="ProcView waited must be >= 0"):
        ProcView(pid="P1", arrival=0, burst=4, remaining=4, waited=-1)

    with pytest.raises(TypeError, match="last_run_end must be int or None"):
        ProcView(
            pid="P1",
            arrival=0,
            burst=4,
            remaining=4,
            waited=0,
            last_run_end="2",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="last_run_end must be >= 0"):
        ProcView(
            pid="P1", arrival=0, burst=4, remaining=4, waited=0, last_run_end=-1
        )


def test_view_valid_creation() -> None:
    """
    1. The use of function:
       Tests successful instantiation and attribute access of View.
    2. How does the specific function work:
       Constructs a View instance with ready ProcViews and asserts all properties match.
    3. Concepts used in the function:
       Dataclass instantiation, nested tuple attributes.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Raw view parameters.
       After: Confirms attributes match expected inputs.
    """
    pv1 = ProcView("P1", 0, 4, 4, 0)
    pv2 = ProcView("P2", 1, 2, 2, 0)
    view = View(time=1, ready=(pv1, pv2), last_pid="P1", newly_arrived=("P2",))

    assert view.time == 1
    assert len(view.ready) == 2
    assert view.last_pid == "P1"
    assert view.newly_arrived == ("P2",)


def test_view_immutability() -> None:
    """
    1. The use of function:
       Verifies that View instances are strictly immutable.
    2. How does the specific function work:
       Attempts attribute modification on a View instance and asserts FrozenInstanceError.
    3. Concepts used in the function:
       Frozen dataclass immutability.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Immutable View instance.
       After: Raises FrozenInstanceError.
    """
    view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(FrozenInstanceError):
        view.time = 1  # type: ignore[misc]


def test_view_validation() -> None:
    """
    1. The use of function:
       Tests post-init validation for invalid View parameters.
    2. How does the specific function work:
       Passes invalid time, ready items, last_pid, and newly_arrived collections to assert errors.
    3. Concepts used in the function:
       Defensive validation, collection type checking.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Malformed View parameter sets.
       After: Confirms appropriate TypeError or ValueError is raised.
    """
    with pytest.raises(TypeError, match="View time must be an int"):
        View(time=True, ready=(), last_pid=None, newly_arrived=())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="View time must be >= 0"):
        View(time=-1, ready=(), last_pid=None, newly_arrived=())

    with pytest.raises(TypeError, match="View ready must be a tuple"):
        View(time=0, ready=[], last_pid=None, newly_arrived=())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="View ready item must be ProcView"):
        View(time=0, ready=("bad",), last_pid=None, newly_arrived=())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="View last_pid must be str or None"):
        View(time=0, ready=(), last_pid=123, newly_arrived=())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="View last_pid cannot be empty"):
        View(time=0, ready=(), last_pid="  ", newly_arrived=())

    with pytest.raises(TypeError, match="View newly_arrived must be a tuple"):
        View(time=0, ready=(), last_pid=None, newly_arrived=[])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="newly_arrived item must be a non-empty string"):
        View(time=0, ready=(), last_pid=None, newly_arrived=("",))


def test_step_info_validation() -> None:
    """
    1. The use of function:
       Tests StepInfo creation, immutability, and attribute validation.
    2. How does the specific function work:
       Constructs valid StepInfo, verifies immutability, and tests rejection of invalid inputs.
    3. Concepts used in the function:
       Dataclass verification, post-init error checking.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Valid and invalid StepInfo parameters.
       After: Confirms valid creation and expected exception raising.
    """
    view = View(time=1, ready=(), last_pid="P1", newly_arrived=())
    si = StepInfo(penalty=2, view=view, done=False)
    assert si.penalty == 2
    assert si.view == view
    assert si.done is False

    with pytest.raises(FrozenInstanceError):
        si.penalty = 3  # type: ignore[misc]

    with pytest.raises(TypeError, match="StepInfo penalty must be an int"):
        StepInfo(penalty="bad", view=view, done=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="StepInfo penalty must be >= 0"):
        StepInfo(penalty=-1, view=view, done=False)

    with pytest.raises(TypeError, match="StepInfo view must be View"):
        StepInfo(penalty=0, view="bad", done=False)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="StepInfo done must be a bool"):
        StepInfo(penalty=0, view=view, done=1)  # type: ignore[arg-type]


def test_sim_result_validation() -> None:
    """
    1. The use of function:
       Tests SimResult creation, immutability, and composite object validation.
    2. How does the specific function work:
       Constructs valid SimResult, verifies fields, and verifies type checks on each attribute.
    3. Concepts used in the function:
       Result encapsulation, composite dataclass validation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Valid and invalid SimResult parameters.
       After: Confirms valid instantiation and expected exception raising.
    """
    pm = ProcessMetrics("P1", 0, 2, 2, 2, 0, 0)
    sm = SimulationMetrics(0.0, 2.0, 0.0, 0, 2, 1.0, 0, 0.5, 0, 1.0)
    tl = Timeline([TimelineSegment("P1", 0, 2)])

    res = SimResult(process_metrics=[pm], simulation_metrics=sm, timeline=tl)
    assert len(res.process_metrics) == 1
    assert res.simulation_metrics == sm
    assert res.timeline == tl

    with pytest.raises(FrozenInstanceError):
        res.process_metrics = []  # type: ignore[misc]

    with pytest.raises(TypeError, match="process_metrics must be a list"):
        SimResult(
            process_metrics=(pm,),  # type: ignore[arg-type]
            simulation_metrics=sm,
            timeline=tl,
        )

    with pytest.raises(TypeError, match="must be ProcessMetrics"):
        SimResult(
            process_metrics=["bad"],  # type: ignore[list-item]
            simulation_metrics=sm,
            timeline=tl,
        )

    with pytest.raises(TypeError, match="simulation_metrics must be SimulationMetrics"):
        SimResult(
            process_metrics=[pm],
            simulation_metrics="bad",  # type: ignore[arg-type]
            timeline=tl,
        )

    with pytest.raises(TypeError, match="timeline must be Timeline"):
        SimResult(
            process_metrics=[pm],
            simulation_metrics=sm,
            timeline="bad",  # type: ignore[arg-type]
        )


def test_scripted_policy_sequence_execution() -> None:
    """
    1. The use of function:
       Tests ScriptedPolicy initialization with a sequence of PIDs and sequential replay.
    2. How does the specific function work:
       Passes list of PIDs to ScriptedPolicy and asserts sequential return on select() and call.
    3. Concepts used in the function:
       Policy replay, scripted test driver.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Script sequence ["P1", "P1", "P2"].
       After: Confirms sequential decision output matching the script.
    """
    policy = ScriptedPolicy(["P1", "P1", "P2"])
    view = View(time=0, ready=(), last_pid=None, newly_arrived=())

    assert policy.select(view) == "P1"
    assert policy(view) == "P1"
    assert policy.select(view) == "P2"


def test_scripted_policy_callable_execution() -> None:
    """
    1. The use of function:
       Tests ScriptedPolicy initialization with a decision callable.
    2. How does the specific function work:
       Passes a lambda selecting the first ready PID to ScriptedPolicy and tests decision output.
    3. Concepts used in the function:
       Callable policy adapter, dynamic decision execution.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Callable decision function.
       After: Confirms decision matches callable logic.
    """
    policy = ScriptedPolicy(lambda v: v.ready[0].pid)
    pv = ProcView("P1", 0, 3, 3, 0)
    view = View(time=0, ready=(pv,), last_pid=None, newly_arrived=("P1",))

    assert policy.select(view) == "P1"
    assert policy(view) == "P1"


def test_scripted_policy_validation_and_exhaustion() -> None:
    """
    1. The use of function:
       Tests ScriptedPolicy input validation and exhaustion handling.
    2. How does the specific function work:
       Asserts TypeError for invalid script types, ValueError for bad PID items, and IndexError
       when exhausting a finite sequence.
    3. Concepts used in the function:
       Exhaustion exception handling, defensive type checks.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Invalid or exhausted policy script.
       After: Confirms appropriate exceptions raised.
    """
    with pytest.raises(TypeError, match="must be a Sequence"):
        ScriptedPolicy(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="must be a non-empty str"):
        ScriptedPolicy(["P1", ""])

    policy = ScriptedPolicy(["P1"])
    view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    assert policy.select(view) == "P1"

    with pytest.raises(IndexError, match="exhausted"):
        policy.select(view)

    with pytest.raises(TypeError, match="view must be a View instance"):
        policy.select("not_a_view")  # type: ignore[arg-type]


def test_engine_initialization_and_validation() -> None:
    """
    1. The use of function:
       Tests Engine constructor parameter validation and initial state.
    2. How does the specific function work:
       Passes invalid processes, negative cs_cost, non-integer cs_cost, and asserts errors.
       Verifies initial properties of a valid engine.
    3. Concepts used in the function:
       Constructor validation, initial simulation state.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Valid and invalid Engine configuration parameters.
       After: Confirms exceptions raised on invalid parameters and pristine initial state.
    """
    with pytest.raises(ValueError, match="cannot be empty"):
        Engine([])

    p1 = Process("P1", 0, 3)
    with pytest.raises(TypeError, match="cs_cost must be an int"):
        Engine([p1], cs_cost=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cs_cost must be >= 0"):
        Engine([p1], cs_cost=-1)

    eng = Engine([p1], cs_cost=1)
    assert eng.time == 0
    assert eng.done is False
    assert eng.context_switches == 0
    assert eng.timeline is not None


def test_engine_reset_reinitialization() -> None:
    """
    1. The use of function:
       Tests that calling Engine.reset() cleanly restores initial state after partial run.
    2. How does the specific function work:
       Executes one step of simulation, calls reset(), and verifies time=0, timeline reset,
       and fresh initial View.
    3. Concepts used in the function:
       Simulator state reinitialization, reproducible reset.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Partially executed Engine instance.
       After: Pristine initial View and zeroed simulation counters.
    """
    p1 = Process("P1", 0, 3)
    p2 = Process("P2", 0, 2)
    eng = Engine([p1, p2])

    v1 = eng.step("P1").view
    assert v1.time == 1

    v_reset = eng.reset()
    assert v_reset.time == 0
    assert len(v_reset.ready) == 2
    assert v_reset.last_pid is None
    assert v_reset.newly_arrived == ("P1", "P2")
    assert eng.time == 0
    assert eng.context_switches == 0


def test_single_process_execution_lifecycle() -> None:
    """
    1. The use of function:
       Tests full unit-step execution of a single process workload to completion.
    2. How does the specific function work:
       Runs 3 unit steps for P1(arrival=0, burst=3). Verifies step penalties, remaining bursts,
       completion, timeline, and final result metrics.
    3. Concepts used in the function:
       Unit-step progression, process completion, timeline partition verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Unexecuted single process workload.
       After: Fully executed process with makespan=3, zero waiting, and valid partition.
    """
    p1 = Process("P1", 0, 3)
    eng = Engine([p1])

    init_view = eng.reset()
    assert init_view.time == 0
    assert len(init_view.ready) == 1
    assert init_view.ready[0].pid == "P1"
    assert init_view.ready[0].remaining == 3
    assert init_view.ready[0].last_run_end is None

    # Step 1
    s1 = eng.step("P1")
    assert s1.penalty == 0
    assert s1.done is False
    assert s1.view.time == 1
    assert s1.view.last_pid == "P1"
    assert s1.view.ready[0].remaining == 2
    assert s1.view.ready[0].last_run_end == 1
    assert s1.view.newly_arrived == ()

    # Step 2
    s2 = eng.step("P1")
    assert s2.penalty == 0
    assert s2.done is False
    assert s2.view.time == 2
    assert s2.view.ready[0].remaining == 1
    assert s2.view.ready[0].last_run_end == 2

    # Step 3
    s3 = eng.step("P1")
    assert s3.penalty == 0
    assert s3.done is True
    assert eng.done is True
    assert s3.view.time == 3
    assert s3.view.ready == ()

    res = eng.result()
    assert len(res.process_metrics) == 1
    pm = res.process_metrics[0]
    assert pm.pid == "P1"
    assert pm.completion == 3
    assert pm.turnaround == 3
    assert pm.waiting == 0
    assert pm.response == 0

    assert res.simulation_metrics.makespan == 3
    assert res.timeline.busy_ticks == 3
    assert res.simulation_metrics.overhead_ticks == 0
    assert res.simulation_metrics.context_switches == 0
    assert res.timeline.validate_partition() is True


def test_initial_idle_jump() -> None:
    """
    1. The use of function:
       Tests initial idle jumping when the earliest process arrival is strictly greater than 0.
    2. How does the specific function work:
       Sets up P1 arriving at tick 4. Asserts reset() jumps time to 4, records IDLE [0, 4),
       and verifies first step execution starts at tick 4 with free dispatch.
    3. Concepts used in the function:
       Initial idle jump, timeline IDLE recording, first-dispatch free rule.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Workload arriving at tick 4.
       After: Engine advances directly to tick 4 on reset, records IDLE [0, 4), and executes
              cleanly.
    """
    p1 = Process("P1", arrival=4, burst=2)
    eng = Engine([p1], cs_cost=1)

    init_view = eng.reset()
    assert init_view.time == 4
    assert len(init_view.ready) == 1
    assert init_view.ready[0].pid == "P1"
    assert init_view.last_pid is None
    assert init_view.newly_arrived == ("P1",)

    # First dispatch is free even with cs_cost > 0
    s1 = eng.step("P1")
    assert s1.penalty == 0
    assert s1.view.time == 5
    assert eng.context_switches == 0

    s2 = eng.step("P1")
    assert s2.done is True
    assert s2.view.time == 6

    res = eng.result()
    assert res.simulation_metrics.makespan == 6
    assert res.timeline.busy_ticks == 2
    assert res.simulation_metrics.overhead_ticks == 0
    assert res.simulation_metrics.context_switches == 0

    segments = res.timeline.to_segments()
    assert len(segments) == 2
    assert segments[0] == TimelineSegment(IDLE_LABEL, 0, 4)
    assert segments[1] == TimelineSegment("P1", 4, 6)
    assert res.timeline.validate_partition() is True


def test_intermediate_idle_jump() -> None:
    """
    1. The use of function:
       Tests idle jumping when the ready queue empties before subsequent arrivals.
    2. How does the specific function work:
       Runs P1 (arrival=0, burst=2) and P2 (arrival=6, burst=2) with cs_cost=1.
       Verifies idle jump [2, 6), persistence of last_pid across idle gap, and context switch
       overhead charged upon dispatching P2.
    3. Concepts used in the function:
       Intermediate idle jump, last_pid persistence across idle gaps, context switch overhead.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Disjoint process workloads.
       After: Correctly jumps to tick 6, charges CS overhead, and partitions timeline [0, 9).
    """
    p1 = Process("P1", 0, 2)
    p2 = Process("P2", 6, 2)
    eng = Engine([p1, p2], cs_cost=1)

    eng.step("P1")
    # P1 finishes at tick 2; ready queue empty; jumps to tick 6
    s2 = eng.step("P1")
    assert s2.done is False
    assert s2.view.time == 6
    assert s2.view.last_pid == "P1"
    assert s2.view.newly_arrived == ("P2",)
    assert len(s2.view.ready) == 1
    assert s2.view.ready[0].pid == "P2"

    # Next step: switch from P1 to P2 with cs_cost=1
    # CS tick: [6, 7), P2 waits (waited=1), penalty=1
    # Exec tick: [7, 8), P2 runs (remaining=1), penalty=0
    # Total step penalty = 1
    s3 = eng.step("P2")
    assert s3.penalty == 1
    assert s3.view.time == 8
    assert eng.context_switches == 1

    s4 = eng.step("P2")
    assert s4.done is True
    assert s4.view.time == 9

    res = eng.result()
    assert res.simulation_metrics.makespan == 9
    assert res.timeline.busy_ticks == 4
    assert res.simulation_metrics.overhead_ticks == 1
    assert res.simulation_metrics.context_switches == 1

    p2_metrics = res.process_metrics[1]
    assert p2_metrics.arrival == 6
    assert p2_metrics.completion == 9
    assert p2_metrics.turnaround == 3
    assert p2_metrics.waiting == 1
    assert p2_metrics.response == 1

    segments = res.timeline.to_segments()
    assert segments == [
        TimelineSegment("P1", 0, 2),
        TimelineSegment(IDLE_LABEL, 2, 6),
        TimelineSegment(CS_LABEL, 6, 7),
        TimelineSegment("P2", 7, 9),
    ]
    assert res.timeline.validate_partition() is True


def test_multiple_idle_jumps() -> None:
    """
    1. The use of function:
       Tests multiple disjoint idle gaps across three distinct arrivals.
    2. How does the specific function work:
       Workload P1(2, 1), P2(5, 1), P3(8, 1). Jumps [0, 2), [3, 5), [6, 8).
    3. Concepts used in the function:
       Successive idle jumps, timeline partition coverage.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Three isolated single-tick workloads.
       After: Full makespan=9 with 6 idle ticks and 3 busy ticks.
    """
    procs = [Process("P1", 2, 1), Process("P2", 5, 1), Process("P3", 8, 1)]
    eng = Engine(procs)

    v0 = eng.reset()
    assert v0.time == 2

    s1 = eng.step("P1")
    assert s1.view.time == 5

    s2 = eng.step("P2")
    assert s2.view.time == 8

    s3 = eng.step("P3")
    assert s3.done is True
    assert s3.view.time == 9

    res = eng.result()
    assert res.simulation_metrics.makespan == 9
    assert res.timeline.idle_ticks == 6
    assert res.timeline.busy_ticks == 3
    assert res.timeline.validate_partition() is True


def test_context_switch_zero_cost() -> None:
    """
    1. The use of function:
       Tests that context_switches counter increments when pid != last_pid even if cs_cost == 0.
    2. How does the specific function work:
       Alternates execution between P1 and P2 with cs_cost=0. Verifies context_switches count
       increments without overhead ticks on the timeline.
    3. Concepts used in the function:
       Zero-cost context switch counting, invariant verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: cs_cost = 0 with alternating execution.
       After: context_switches increments, overhead_ticks remains 0.
    """
    p1 = Process("P1", 0, 2)
    p2 = Process("P2", 0, 2)
    eng = Engine([p1, p2], cs_cost=0)

    # Dispatch P1 (first dispatch, free)
    eng.step("P1")
    assert eng.context_switches == 0

    # Switch to P2 (counts as context switch, 0 overhead ticks)
    eng.step("P2")
    assert eng.context_switches == 1

    # Switch back to P1
    eng.step("P1")
    assert eng.context_switches == 2

    # Switch back to P2
    s4 = eng.step("P2")
    assert eng.context_switches == 3
    assert s4.done is True

    res = eng.result()
    assert res.simulation_metrics.context_switches == 3
    assert res.simulation_metrics.overhead_ticks == 0
    assert res.simulation_metrics.makespan == 4


def test_context_switch_nonzero_cost_and_overhead_ticks() -> None:
    """
    1. The use of function:
       Tests context switch overhead ticks and timeline CS recording when cs_cost > 0.
    2. How does the specific function work:
       Executes P1 then switches to P2 with cs_cost=2. Verifies 2 CS ticks recorded on timeline
       and ready processes accumulate wait time during CS ticks.
    3. Concepts used in the function:
       Multi-tick context switch overhead, waiting time accumulation during CS.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: P1 running followed by switch to P2 with cs_cost=2.
       After: Exactly 2 CS ticks recorded, both ready processes accumulate wait time.
    """
    p1 = Process("P1", 0, 2)
    p2 = Process("P2", 0, 2)
    eng = Engine([p1, p2], cs_cost=2)

    # Step 1: P1 runs [0, 1). P2 waits (waited=1). Penalty=1.
    s1 = eng.step("P1")
    assert s1.penalty == 1
    assert s1.view.time == 1

    # Step 2: Switch to P2 with cs_cost=2.
    # CS tick 1 [1, 2): P1 waits (0->1), P2 waits (1->2). Penalty += 2.
    # CS tick 2 [2, 3): P1 waits (1->2), P2 waits (2->3). Penalty += 2.
    # Exec tick [3, 4): P2 runs, P1 waits (2->3). Penalty += 1.
    # Total step penalty = 2 + 2 + 1 = 5.
    s2 = eng.step("P2")
    assert s2.penalty == 5
    assert s2.view.time == 4
    assert eng.context_switches == 1

    res = eng.timeline.to_segments()
    assert res[0] == TimelineSegment("P1", 0, 1)
    assert res[1] == TimelineSegment(CS_LABEL, 1, 3)
    assert res[2] == TimelineSegment("P2", 3, 4)


def test_context_switch_same_pid_no_switch() -> None:
    """
    1. The use of function:
       Verifies that continuing execution of the same process never triggers a context switch.
    2. How does the specific function work:
       Executes P1 for consecutive steps with cs_cost=2 and asserts context_switches remains 0.
    3. Concepts used in the function:
       Process continuation semantics.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: P1 executed consecutively.
       After: context_switches remains 0 and no CS overhead is incurred.
    """
    p1 = Process("P1", 0, 3)
    p2 = Process("P2", 0, 2)
    eng = Engine([p1, p2], cs_cost=2)

    eng.step("P1")
    assert eng.context_switches == 0
    eng.step("P1")
    assert eng.context_switches == 0
    eng.step("P1")
    assert eng.context_switches == 0


def test_context_switch_away_from_completed_process() -> None:
    """
    1. The use of function:
       Verifies that switching away from a completed process counts as a context switch.
    2. How does the specific function work:
       Runs P1(0, 1) to completion, then dispatches P2(0, 1). Asserts context_switches increments
       to 1 and CS overhead is spent if cs_cost > 0.
    3. Concepts used in the function:
       Context switch semantics on process completion.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: P1 completed in preceding tick.
       After: Dispatch of P2 increments context_switches and applies cs_cost overhead.
    """
    p1 = Process("P1", 0, 1)
    p2 = Process("P2", 0, 1)
    eng = Engine([p1, p2], cs_cost=1)

    eng.step("P1")
    assert eng.context_switches == 0

    s2 = eng.step("P2")
    assert eng.context_switches == 1
    assert s2.view.time == 3  # 1 (P1) + 1 (CS) + 1 (P2)
    assert s2.done is True


def test_idle_gap_preserves_last_pid_for_context_switch() -> None:
    """
    1. The use of function:
       Verifies that idle gaps do not reset last_pid, preserving context switch semantics.
    2. How does the specific function work:
       Runs P1, allows idle jump to P2, then dispatches P2. Asserts context switch is counted.
    3. Concepts used in the function:
       last_pid retention across idle intervals.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Idle gap separating P1 completion and P2 dispatch.
       After: P2 dispatch registers context switch from P1.
    """
    p1 = Process("P1", 0, 1)
    p2 = Process("P2", 5, 1)
    eng = Engine([p1, p2], cs_cost=0)

    s1 = eng.step("P1")
    assert s1.view.last_pid == "P1"
    assert s1.view.time == 5

    eng.step("P2")
    assert eng.context_switches == 1


def test_arrivals_during_context_switch_ticks() -> None:
    """
    1. The use of function:
       Tests that processes arriving during multi-tick context switches are admitted immediately
       and accumulate wait time for subsequent overhead ticks.
    2. How does the specific function work:
       Sets up P1(0, 1), P2(0, 3), P3(arrival=2, burst=1) with cs_cost=2.
       P1 finishes at tick 1. Switch to P2 starts CS ticks [1, 2) and [2, 3).
       At tick 2, P3 arrives and is admitted. During [2, 3), both P2 and P3 wait.
       During [3, 4), P2 runs and P3 waits.
    3. Concepts used in the function:
       Arrival admission during overhead ticks, wait time conservation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: P3 arrives at tick 2 during a 2-tick CS interval [1, 3).
       After: P3 is admitted at tick 2, waits 1 tick during CS and 1 tick during P2 execution.
    """
    p1 = Process("P1", 0, 1)
    p2 = Process("P2", 0, 3)
    p3 = Process("P3", 2, 1)
    eng = Engine([p1, p2, p3], cs_cost=2)

    eng.step("P1")  # t=0..1, P1 finishes

    # Step to P2:
    # CS tick 1 [1, 2): P2 waits (waited: 1 -> 2). Penalty += 1.
    # At t=2: P3 arrives! Admitted into ready set.
    # CS tick 2 [2, 3): P2 waits (waited: 2 -> 3), P3 waits (waited: 0 -> 1). Penalty += 2.
    # Exec tick [3, 4): P2 runs, P3 waits (waited: 1 -> 2). Penalty += 1.
    # Total step penalty = 1 + 2 + 1 = 4.
    s2 = eng.step("P2")
    assert s2.penalty == 4
    assert s2.view.time == 4
    assert s2.view.newly_arrived == ("P3",)

    # Check P3 view in ready queue
    p3_view = next(p for p in s2.view.ready if p.pid == "P3")
    assert p3_view.waited == 2


def test_ready_queue_and_newly_arrived_ordering() -> None:
    """
    1. The use of function:
       Verifies canonical sorting of ready queue and newly_arrived by (arrival, pid).
    2. How does the specific function work:
       Provides processes with same and different arrival times out of alphabetical order.
       Asserts View.ready and View.newly_arrived are sorted deterministically.
    3. Concepts used in the function:
       Deterministic tie-breaking, canonical queue representation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Workload passed in unsorted order.
       After: Confirms ready views and newly_arrived are ordered by (arrival, pid).
    """
    p_c = Process("PC", 0, 2)
    p_a = Process("PA", 0, 2)
    p_b = Process("PB", 0, 2)
    eng = Engine([p_c, p_a, p_b])

    view = eng.reset()
    assert [p.pid for p in view.ready] == ["PA", "PB", "PC"]
    assert view.newly_arrived == ("PA", "PB", "PC")

    s1 = eng.step("PA")
    assert s1.view.newly_arrived == ()
    assert [p.pid for p in s1.view.ready] == ["PA", "PB", "PC"]


def test_penalty_step_and_cumulative_waiting_equivalence() -> None:
    """
    1. The use of function:
       Verifies that the sum of step penalties strictly equals the sum of waiting times across
       all completed processes.
    2. How does the specific function work:
       Executes a multi-process workload with interleaved executions and context switches.
       Tracks cumulative penalties and asserts exact equality with total process waiting time.
    3. Concepts used in the function:
       Exact RL reward equivalence (return == -sum(waiting times)), conservation law.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Dynamic simulation run with varying context switches.
       After: Confirms sum(penalties) == sum(process waiting times) down to exact integer equality.
    """
    p1 = Process("P1", 0, 3)
    p2 = Process("P2", 1, 2)
    p3 = Process("P3", 2, 2)
    eng = Engine([p1, p2, p3], cs_cost=1)

    policy = ScriptedPolicy(["P1", "P2", "P1", "P3", "P2", "P1", "P3"])
    cumulative_penalty = 0

    view = eng.reset()
    while not eng.done:
        pid = policy.select(view)
        s = eng.step(pid)
        cumulative_penalty += s.penalty
        view = s.view

    res = eng.result()
    total_waiting = sum(m.waiting for m in res.process_metrics)
    assert cumulative_penalty == total_waiting


def test_run_helper_with_scripted_policy() -> None:
    """
    1. The use of function:
       Tests the run() helper executing a sequence-based ScriptedPolicy to completion.
    2. How does the specific function work:
       Supplies processes and ScriptedPolicy to run() and asserts SimResult validity.
    3. Concepts used in the function:
       Simulation runner, end-to-end execution.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Workload and scripted sequence.
       After: Complete valid SimResult returned.
    """
    procs = [Process("P1", 0, 2), Process("P2", 0, 1)]
    policy = ScriptedPolicy(["P1", "P1", "P2"])
    res = run(policy=policy, processes=procs, cs_cost=1)

    assert len(res.process_metrics) == 2
    assert res.simulation_metrics.makespan == 4
    assert res.simulation_metrics.context_switches == 1
    assert res.timeline.validate_partition() is True


def test_run_helper_with_callable_and_lambda() -> None:
    """
    1. The use of function:
       Tests the run() helper executing with a plain lambda function policy.
    2. How does the specific function work:
       Passes a lambda choosing the ready process with shortest remaining burst.
    3. Concepts used in the function:
       Callable policy execution, duck typing.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Raw lambda policy.
       After: Completes simulation and returns SimResult.
    """
    procs = [Process("P1", 0, 2), Process("P2", 0, 1)]
    # Greedy shortest remaining policy
    res = run(
        policy=lambda v: min(v.ready, key=lambda p: (p.remaining, p.arrival, p.pid)).pid,
        processes=procs,
    )
    assert len(res.process_metrics) == 2
    assert res.simulation_metrics.makespan == 3


def test_run_helper_invalid_policy() -> None:
    """
    1. The use of function:
       Verifies TypeError when passing an invalid non-callable policy to run().
    2. How does the specific function work:
       Passes an integer as policy and asserts TypeError.
    3. Concepts used in the function:
       Defensive input validation in simulation runner.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Non-callable policy object.
       After: Raises TypeError.
    """
    procs = [Process("P1", 0, 2)]
    with pytest.raises(TypeError, match="Policy must provide a callable"):
        run(policy=12345, processes=procs)


def test_engine_step_when_done_raises_runtime_error() -> None:
    """
    1. The use of function:
       Verifies RuntimeError is raised when calling step() after simulation is complete.
    2. How does the specific function work:
       Executes single process to completion and attempts an extra step() call.
    3. Concepts used in the function:
       State machine guard, terminated engine protection.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Engine in completed state.
       After: Calling step() raises RuntimeError.
    """
    eng = Engine([Process("P1", 0, 1)])
    eng.step("P1")
    assert eng.done is True

    with pytest.raises(RuntimeError, match="Simulation is already complete"):
        eng.step("P1")


def test_engine_step_invalid_pid_raises_errors() -> None:
    """
    1. The use of function:
       Tests ValueError and TypeError when stepping with invalid or non-ready PIDs.
    2. How does the specific function work:
       Steps with non-string PID, unadmitted PID, and already completed PID.
    3. Concepts used in the function:
       Dispatch validation, ready-queue constraint enforcement.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Engine ready for step.
       After: Raises TypeError or ValueError on invalid PID dispatch.
    """
    p1 = Process("P1", 0, 1)
    p2 = Process("P2", 10, 1)
    eng = Engine([p1, p2])

    with pytest.raises(TypeError, match="pid must be a str"):
        eng.step(123)  # type: ignore[arg-type]

    # P2 is not yet ready at tick 0
    with pytest.raises(ValueError, match="is not in ready queue"):
        eng.step("P2")

    eng.step("P1")
    # P1 is now completed, no longer ready
    with pytest.raises(ValueError, match="is not in ready queue"):
        eng.step("P1")


def test_engine_result_before_done_raises_runtime_error() -> None:
    """
    1. The use of function:
       Verifies that calling Engine.result() before completion raises RuntimeError.
    2. How does the specific function work:
       Instantiates engine and immediately calls result() while unfinished.
    3. Concepts used in the function:
       Lifecycle state check, premature result generation protection.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Unfinished Engine instance.
       After: Calling result() raises RuntimeError.
    """
    eng = Engine([Process("P1", 0, 2)])
    with pytest.raises(RuntimeError, match="Cannot compute SimResult"):
        eng.result()


def test_section_4_6_fcfs_full_simulation() -> None:
    """
    1. The use of function:
       Validates full FCFS execution of the Section 4.6 specification fixture.
    2. How does the specific function work:
       Workload: P1(0, 5), P2(1, 3), P3(2, 2).
       Under FCFS with cs_cost=0:
       - P1 executes [0, 5)
       - P2 executes [5, 8)
       - P3 executes [8, 10)
       Verifies individual metrics, aggregate metrics, and timeline against Section 4.6 math.
    3. Concepts used in the function:
       Section 4.6 FCFS verification, end-to-end simulation comparison.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Unexecuted Section 4.6 workload.
       After: Confirms exact numerical values for FCFS metrics and timeline segments.
    """
    procs = [Process("P1", 0, 5), Process("P2", 1, 3), Process("P3", 2, 2)]
    fcfs_script = ["P1"] * 5 + ["P2"] * 3 + ["P3"] * 2

    res = run(policy=ScriptedPolicy(fcfs_script), processes=procs, cs_cost=0)

    # Process metrics verification
    m1, m2, m3 = res.process_metrics
    assert (m1.completion, m1.turnaround, m1.waiting, m1.response) == (5, 5, 0, 0)
    assert (m2.completion, m2.turnaround, m2.waiting, m2.response) == (8, 7, 4, 4)
    assert (m3.completion, m3.turnaround, m3.waiting, m3.response) == (10, 8, 6, 6)

    # Aggregate simulation metrics verification
    sim = res.simulation_metrics
    assert sim.makespan == 10
    assert sim.avg_waiting == pytest.approx(10.0 / 3.0)
    assert sim.avg_turnaround == pytest.approx(20.0 / 3.0)
    assert sim.avg_response == pytest.approx(10.0 / 3.0)
    assert sim.max_waiting == 6
    assert sim.cpu_utilization == 1.0
    assert sim.overhead_ticks == 0
    assert sim.context_switches == 2
    assert sim.jain_fairness == pytest.approx(242.0 / 303.0)

    # Timeline verification
    assert res.timeline.to_segments() == [
        TimelineSegment("P1", 0, 5),
        TimelineSegment("P2", 5, 8),
        TimelineSegment("P3", 8, 10),
    ]
    assert res.timeline.validate_partition() is True


def test_section_4_6_srtf_full_simulation() -> None:
    """
    1. The use of function:
       Validates full SRTF execution of the Section 4.6 specification fixture.
    2. How does the specific function work:
       Workload: P1(0, 5), P2(1, 3), P3(2, 2).
       Under SRTF with cs_cost=0:
       - t=0: P1 runs [0, 1) (rem: 4)
       - t=1: P2 arrives (burst 3), preempts P1, runs [1, 4) (completion 4)
       - t=4: P3 (burst 2) < P1 (rem 4), P3 runs [4, 6) (completion 6)
       - t=6: P1 resumes [6, 10) (completion 10)
       Verifies metrics and timeline against Section 4.6 math.
    3. Concepts used in the function:
       Section 4.6 SRTF verification, preemption handling, reference baseline validation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Unexecuted Section 4.6 workload.
       After: Confirms exact numerical values for SRTF metrics and timeline segments.
    """
    procs = [Process("P1", 0, 5), Process("P2", 1, 3), Process("P3", 2, 2)]
    srtf_script = ["P1"] + ["P2"] * 3 + ["P3"] * 2 + ["P1"] * 4

    res = run(policy=ScriptedPolicy(srtf_script), processes=procs, cs_cost=0)

    m1, m2, m3 = res.process_metrics
    assert (m1.completion, m1.turnaround, m1.waiting, m1.response) == (10, 10, 5, 0)
    assert (m2.completion, m2.turnaround, m2.waiting, m2.response) == (4, 3, 0, 0)
    assert (m3.completion, m3.turnaround, m3.waiting, m3.response) == (6, 4, 2, 2)

    sim = res.simulation_metrics
    assert sim.makespan == 10
    assert sim.avg_waiting == pytest.approx(7.0 / 3.0)
    assert sim.avg_turnaround == pytest.approx(17.0 / 3.0)
    assert sim.avg_response == pytest.approx(2.0 / 3.0)
    assert sim.max_waiting == 5
    assert sim.cpu_utilization == 1.0
    assert sim.overhead_ticks == 0
    assert sim.context_switches == 3

    assert res.timeline.to_segments() == [
        TimelineSegment("P1", 0, 1),
        TimelineSegment("P2", 1, 4),
        TimelineSegment("P3", 4, 6),
        TimelineSegment("P1", 6, 10),
    ]
    assert res.timeline.validate_partition() is True
