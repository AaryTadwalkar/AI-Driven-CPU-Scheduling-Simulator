"""
1. What this file does:
   Implements comprehensive unit and regression tests for Stage 11: Engine AI inference stalls,
   non-forced decision counting, fractional debt accumulation, and state cloning / checkpointing.
2. Overall design:
   Verifies backwards compatibility with Part A execution semantics, validates the hand-calculated
   Stage 11 reference fixture, verifies fractional debt conversion, validates forced-decision
   cost exemption, verifies engine clone and restore isolation, tests timeline partitioning with
   AI intervals, checks one-time on_start overhead charging, and enforces reward penalty identity.
3. Elements in the file:
   - MockCostPolicy: Mock scheduling policy with configurable upfront and per-decision costs.
   - test_stage11_part_a_backwards_compatibility: Confirms identical results when tick_us is None.
   - test_stage11_hand_verified_fixture: Confirms exact hand-calculated completion/waiting values.
   - test_stage11_fractional_debt_accumulator: Tests 0.25 tick charges accumulating to 2 ticks.
   - test_stage11_forced_decision_free: Asserts decisions with single ready candidate are free.
   - test_stage11_engine_clone_and_restore: Verifies independent divergent steps and restoration.
   - test_stage11_timeline_partition_with_all_labels: Verifies PID | CS | AI | IDLE partition.
   - test_stage11_on_start_charged_once: Verifies upfront workload-level model inference latency.
   - test_stage11_reward_identity_preservation: Asserts sum of step penalties equals sum of waits.
   - test_stage11_engine_properties_and_input_validation: Asserts parameter boundaries and types.
   - test_stage11_simulation_metrics_effective_defaults: Asserts metric defaults and overrides.
   - test_stage11_config_settings_and_constants: Asserts Setting dataclass and config definitions.
4. Final output of the file:
   Pytest test suite validating all Stage 11 engine, metrics, timeline, and policy requirements.
"""

from collections.abc import Sequence
from typing import Any

import pytest

from config import (
    DEFAULT_TICK_US,
    FORCED_DECISION_FREE,
    SETTINGS,
    TICK_US_SWEEP,
    Setting,
)
from schedsim.domain.engine import Engine, View, run
from schedsim.domain.metrics import (
    ProcessMetrics,
    calculate_simulation_metrics,
)
from schedsim.domain.process import Process
from schedsim.domain.timeline import AI_LABEL, CS_LABEL, IDLE_LABEL
from schedsim.policies import (
    FCFSPolicy,
    Policy,
    RoundRobinPolicy,
    SJFPolicy,
    SRTFPolicy,
)


class MockCostPolicy(Policy):
    """
    1. The use of class:
       Mock scheduling policy providing configurable upfront on_start and per-decision costs.
    2. What the class contains:
       - _inner_policy: Delegated Policy instance determining process selection order.
       - _decision_cost_us: Configured latency in microseconds returned by decision_cost_us().
       - _start_cost_us: Configured latency in microseconds returned by on_start().
       - select: Selects the next PID by delegating to _inner_policy.select().
       - on_start: Returns configured upfront latency in microseconds.
       - decision_cost_us: Returns configured per-decision latency in microseconds.
       - reset: Resets inner policy state.
    3. Technologies used and why:
       - Subclasses Policy to verify compliance with scheduling policy interface contract.
       - Strict static typing for clean deterministic testing.
    4. Key design and overall design of the class:
       Provides an isolated test double instrumenting latency values without external models.
    """

    def __init__(
        self,
        inner_policy: Policy | None = None,
        decision_cost_us: float = 0.0,
        start_cost_us: float = 0.0,
        name: str = "MockCostPolicy",
    ) -> None:
        """
        1. The use of function:
           Initializes the mock cost policy with underlying selector and latency costs.
        2. How does the specific function work:
           Stores inner_policy (defaulting to FCFSPolicy), decision_cost_us, and start_cost_us.
        3. Concepts used in the function:
           Constructor initialization, policy delegation, test instrumentation.
        4. Places where this function is imported and used:
           Instantiated in Stage 11 unit test fixtures.
        5. Final output of the function, before and after:
           Before: Uninitialized mock policy.
           After: Initialized MockCostPolicy instance with configured latency returns.
        """
        super().__init__(name=name)
        self._inner_policy: Policy = inner_policy if inner_policy is not None else FCFSPolicy()
        self._decision_cost_us: float = decision_cost_us
        self._start_cost_us: float = start_cost_us

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the next PID to execute for the given View.
        2. How does the specific function work:
           Delegates selection directly to the configured inner scheduling policy.
        3. Concepts used in the function:
           Delegation pattern, polymorphic dispatch.
        4. Places where this function is imported and used:
           Invoked by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Current simulation View.
           After: Returns PID selected by inner policy.
        """
        return self._inner_policy.select(view)

    def on_start(self, processes: Sequence[Any]) -> float:
        """
        1. The use of function:
           Returns the configured upfront workload inference latency in microseconds.
        2. How does the specific function work:
           Returns stored float self._start_cost_us.
        3. Concepts used in the function:
           Upfront latency hook, test double return.
        4. Places where this function is imported and used:
           Invoked by run() prior to simulation stepping loop.
        5. Final output of the function, before and after:
           Before: Sequence of workload processes.
           After: Returns non-negative float start cost in microseconds.
        """
        return self._start_cost_us

    def decision_cost_us(self, view: View) -> float:
        """
        1. The use of function:
           Returns the configured per-decision model latency in microseconds.
        2. How does the specific function work:
           Returns stored float self._decision_cost_us.
        3. Concepts used in the function:
           Per-decision latency hook, test double return.
        4. Places where this function is imported and used:
           Invoked by run() or Engine.step() on each scheduling decision.
        5. Final output of the function, before and after:
           Before: Current simulation View.
           After: Returns non-negative float decision cost in microseconds.
        """
        return self._decision_cost_us

    def reset(self) -> None:
        """
        1. The use of function:
           Resets internal state of the inner policy.
        2. How does the specific function work:
           Invokes self._inner_policy.reset().
        3. Concepts used in the function:
           State reset delegation.
        4. Places where this function is imported and used:
           Invoked when resetting schedulers between simulation episodes.
        5. Final output of the function, before and after:
           Before: Inner policy in current state.
           After: Inner policy reset.
        """
        self._inner_policy.reset()


def test_stage11_part_a_backwards_compatibility() -> None:
    """
    1. The use of function:
       Verifies that running with tick_us=None reproduces Part A baseline metrics exactly.
    2. How does the specific function work:
       Executes FCFS, SJF, SRTF, and RR on the standard Section 4.6 3-process hand fixture
       with tick_us=None. Asserts zero AI stall ticks, zero decisions overhead, and confirms
       waiting times match known Part A reference values.
    3. Concepts used in the function:
       Backwards compatibility testing, regression defense, zero-overhead invariance.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Modified Engine with AI stall capabilities.
       After: Asserts exact equality with Part A classical baseline metrics.
    """
    processes = [
        Process(pid="P1", arrival=0, burst=5),
        Process(pid="P2", arrival=1, burst=3),
        Process(pid="P3", arrival=2, burst=2),
    ]

    # FCFS baseline check
    res_fcfs = run(FCFSPolicy(), processes, cs_cost=0, tick_us=None)
    assert res_fcfs.simulation_metrics.ai_overhead_ticks == 0
    assert res_fcfs.simulation_metrics.inference_us_total == 0.0
    assert res_fcfs.simulation_metrics.effective_makespan == res_fcfs.simulation_metrics.makespan
    assert (
        res_fcfs.simulation_metrics.effective_turnaround
        == res_fcfs.simulation_metrics.avg_turnaround
    )
    assert pytest.approx(res_fcfs.simulation_metrics.avg_waiting, rel=1e-3) == 3.333

    # SJF baseline check
    res_sjf = run(SJFPolicy(), processes, cs_cost=0, tick_us=None)
    assert res_sjf.simulation_metrics.ai_overhead_ticks == 0
    assert pytest.approx(res_sjf.simulation_metrics.avg_waiting, rel=1e-3) == 3.000

    # SRTF baseline check
    res_srtf = run(SRTFPolicy(), processes, cs_cost=0, tick_us=None)
    assert res_srtf.simulation_metrics.ai_overhead_ticks == 0
    assert pytest.approx(res_srtf.simulation_metrics.avg_waiting, rel=1e-3) == 2.333

    # RR baseline check (quantum=2)
    res_rr = run(RoundRobinPolicy(quantum=2), processes, cs_cost=0, tick_us=None)
    assert res_rr.simulation_metrics.ai_overhead_ticks == 0
    assert pytest.approx(res_rr.simulation_metrics.avg_waiting, rel=1e-3) == 4.000


def test_stage11_hand_verified_fixture() -> None:
    """
    1. The use of function:
       Validates the exact hand-calculated fixture from Section 5 Stage 11 of the revised plan.
    2. How does the specific function work:
       Constructs workload P1 (arr 0, burst 2) and P2 (arr 0, burst 1). Runs FCFS-like policy
       with decision_cost_us=1000.0, cs_cost=0, tick_us=1000.0. Asserts non-forced decisions at
       t=0 and t=2, exactly 2 AI ticks, completions P1=4, P2=5, waiting 2 and 4, average waiting
       3.0, and effective_makespan=5.
    3. Concepts used in the function:
       Deterministic hand-calculated verification, discrete event trace validation.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Engine running hand-calculated fixture.
       After: All hand-verified assertions verified green.
    """
    processes = [
        Process(pid="P1", arrival=0, burst=2),
        Process(pid="P2", arrival=0, burst=1),
    ]
    policy = MockCostPolicy(inner_policy=FCFSPolicy(), decision_cost_us=1000.0)

    res = run(policy, processes, cs_cost=0, tick_us=1000.0)

    # Decisions check: t=0 (P1, P2 ready => non-forced), t=2 (P1, P2 ready => non-forced),
    # t=4 (only P2 ready => forced decision, free).
    assert res.simulation_metrics.decisions == 2
    assert res.simulation_metrics.ai_overhead_ticks == 2
    assert res.simulation_metrics.inference_us_total == 2000.0
    assert res.simulation_metrics.makespan == 5
    assert res.simulation_metrics.effective_makespan == 5

    pm_map = {m.pid: m for m in res.process_metrics}
    assert pm_map["P1"].completion == 4
    assert pm_map["P1"].waiting == 2
    assert pm_map["P1"].turnaround == 4

    assert pm_map["P2"].completion == 5
    assert pm_map["P2"].waiting == 4
    assert pm_map["P2"].turnaround == 5

    assert pytest.approx(res.simulation_metrics.avg_waiting, rel=1e-6) == 3.0
    assert pytest.approx(res.simulation_metrics.avg_turnaround, rel=1e-6) == 4.5
    assert pytest.approx(res.simulation_metrics.effective_turnaround, rel=1e-6) == 4.5

    # Check timeline segments
    segments = res.timeline.to_segments()
    expected_segments = [
        ("AI", 0, 1),
        ("P1", 1, 2),
        ("AI", 2, 3),
        ("P1", 3, 4),
        ("P2", 4, 5),
    ]
    actual_segments = [(s.label, s.start, s.end) for s in segments]
    assert actual_segments == expected_segments


def test_stage11_fractional_debt_accumulator() -> None:
    """
    1. The use of function:
       Verifies fractional debt accumulation (0.25 tick per decision yielding 2 AI stall ticks).
    2. How does the specific function work:
       Workload with 2 long-running processes (burst 10 each). Decision cost is 250 µs with
       tick_us=1000 µs (0.25 tick per decision). Steps 8 non-forced decisions. Asserts exactly
       2 AI stall ticks are charged over 8 decisions without floating-point accumulation drift.
    3. Concepts used in the function:
       Fractional debt accumulation, discrete stall tick conversion, arithmetic stability.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Engine with fractional debt accumulator.
       After: Exactly 2 AI ticks charged over 8 non-forced decisions.
    """
    processes = [
        Process(pid="P1", arrival=0, burst=10),
        Process(pid="P2", arrival=0, burst=10),
    ]
    engine = Engine(processes=processes, cs_cost=0, tick_us=1000.0)

    # 8 non-forced decisions with 250 us cost each (= 0.25 ticks per decision)
    # Decisions 1-3: debt 0.25, 0.50, 0.75 -> 0 stall ticks
    # Decision 4: debt 1.00 -> 1 stall tick (debt resets to 0.0)
    # Decisions 5-7: debt 0.25, 0.50, 0.75 -> 0 stall ticks
    # Decision 8: debt 1.00 -> 1 stall tick (debt resets to 0.0)
    for _ in range(4):
        engine.step("P1", decision_cost_us=250.0)
    assert engine.decisions == 4
    assert engine.ai_overhead_ticks == 1

    for _ in range(4):
        engine.step("P2", decision_cost_us=250.0)
    assert engine.decisions == 8
    assert engine.ai_overhead_ticks == 2
    assert engine.inference_us_total == 2000.0


def test_stage11_forced_decision_free() -> None:
    """
    1. The use of function:
       Verifies that single-candidate decisions are free and do not increment decisions counter.
    2. How does the specific function work:
       Constructs workload with solitary ready process P1 (arr 0, burst 4). Calls engine.step()
       with large decision_cost_us=5000.0 at tick_us=1000.0. Asserts 0 AI ticks and 0 decisions.
    3. Concepts used in the function:
       Forced decision rule, cost exemption, ready queue cardinality check.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Engine running solitary ready process.
       After: Decisions counter remains 0, 0 stall ticks incurred.
    """
    processes = [Process(pid="P1", arrival=0, burst=4)]
    engine = Engine(processes=processes, cs_cost=0, tick_us=1000.0)

    for _ in range(4):
        engine.step("P1", decision_cost_us=5000.0)

    assert engine.decisions == 0
    assert engine.ai_overhead_ticks == 0
    assert engine.inference_us_total == 0.0
    assert engine.timeline.makespan == 4
    assert engine.timeline.ai_ticks == 0


def test_stage11_engine_clone_and_restore() -> None:
    """
    1. The use of function:
       Validates engine clone isolation and checkpoint restoration across divergent branches.
    2. How does the specific function work:
       Initializes engine, steps 2 ticks. Clones engine checkpoint. Steps original with P1 and
       checkpoint with P2. Verifies complete state isolation. Restores original from checkpoint
       and verifies exact state match.
    3. Concepts used in the function:
       Deep state cloning, checkpoint rollback, tree search branching support.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Divergent execution branches on original and clone.
       After: Confirmed state independence and successful restoration.
    """
    processes = [
        Process(pid="P1", arrival=0, burst=5),
        Process(pid="P2", arrival=0, burst=5),
    ]
    engine = Engine(processes=processes, cs_cost=1, tick_us=1000.0)

    # Step twice with P1
    engine.step("P1", decision_cost_us=500.0)
    engine.step("P1", decision_cost_us=500.0)

    assert engine.time == 3  # 1 AI tick from 1000 us total + 2 burst ticks
    assert engine.decisions == 2
    assert engine.ai_overhead_ticks == 1

    # Clone checkpoint
    checkpoint = engine.clone()

    # Divergent steps: engine continues with P1, checkpoint steps with P2
    engine.step("P1", decision_cost_us=0.0)
    checkpoint.step("P2", decision_cost_us=0.0)  # Causes context switch on checkpoint

    # Checkpoint should have context switch cost, engine should not
    assert engine.context_switches == 0
    assert checkpoint.context_switches == 1
    assert engine.time == 4
    assert checkpoint.time == 5  # 3 + 1 CS + 1 execution

    # Restore engine from checkpoint
    engine.restore(checkpoint)
    assert engine.context_switches == 1
    assert engine.time == 5
    assert engine.timeline.to_segments() == checkpoint.timeline.to_segments()


def test_stage11_timeline_partition_with_all_labels() -> None:
    """
    1. The use of function:
       Verifies validate_partition() on a timeline containing PID, CS, AI, and IDLE segments.
    2. How does the specific function work:
       Constructs workload with arrival at t=2 (inducing IDLE 0..2). Runs with cs_cost=1 and
       AI decision cost 1000 µs (inducing AI and CS ticks). Validates partition integrity.
    3. Concepts used in the function:
       Timeline partitioning, label exhaustiveness, gapless interval verification.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Timeline with PID, CS, AI, and IDLE intervals.
       After: validate_partition() returns True, total makespan conservation verified.
    """
    processes = [
        Process(pid="P1", arrival=2, burst=2),
        Process(pid="P2", arrival=2, burst=2),
    ]
    policy = MockCostPolicy(inner_policy=RoundRobinPolicy(quantum=1), decision_cost_us=1000.0)

    res = run(policy, processes, cs_cost=1, tick_us=1000.0)
    tl = res.timeline

    # Must contain IDLE, AI, CS, and process labels
    labels = {s.label for s in tl.to_segments()}
    assert IDLE_LABEL in labels
    assert AI_LABEL in labels
    assert CS_LABEL in labels
    assert "P1" in labels
    assert "P2" in labels

    assert tl.validate_partition() is True
    assert tl.makespan == tl.busy_ticks + tl.overhead_ticks + tl.ai_ticks + tl.idle_ticks
    assert tl.busy_ticks == 4


def test_stage11_on_start_charged_once() -> None:
    """
    1. The use of function:
       Verifies that on_start() upfront latency is charged exactly once at first dispatch.
    2. How does the specific function work:
       Uses MockCostPolicy with start_cost_us=2000.0 and decision_cost_us=0.0. Runs workload at
       tick_us=1000.0. Asserts 2 AI stall ticks appear at start, and no further AI ticks occur.
    3. Concepts used in the function:
       Category 1 upfront model latency accounting, single-charge invariant.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Workload executing with Category 1 start overhead.
       After: Upfront 2 AI ticks charged once, total inference_us_total equals start_cost_us.
    """
    processes = [
        Process(pid="P1", arrival=0, burst=2),
        Process(pid="P2", arrival=0, burst=1),
    ]
    policy = MockCostPolicy(
        inner_policy=FCFSPolicy(),
        start_cost_us=2000.0,
        decision_cost_us=0.0,
    )

    res = run(policy, processes, cs_cost=0, tick_us=1000.0)

    assert res.simulation_metrics.ai_overhead_ticks == 2
    assert res.simulation_metrics.inference_us_total == 2000.0
    # First segment must be AI of duration 2
    first_seg = res.timeline.to_segments()[0]
    assert first_seg.label == AI_LABEL
    assert first_seg.start == 0
    assert first_seg.end == 2


def test_stage11_reward_identity_preservation() -> None:
    """
    1. The use of function:
       Verifies that sum of step penalties equals sum of process waiting times under AI stalls.
    2. How does the specific function work:
       Executes engine step-by-step with AI stalls and context switches, accumulating penalties.
       Asserts total step penalties match sum(waiting) across all completed processes.
    3. Concepts used in the function:
       Reward conservation, Markov decision process penalty identity (reward = -waiting).
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Discrete simulation steps with AI stalls.
       After: Mathematical equality confirmed between penalties and waiting times.
    """
    processes = [
        Process(pid="P1", arrival=0, burst=3),
        Process(pid="P2", arrival=1, burst=2),
        Process(pid="P3", arrival=2, burst=1),
    ]
    engine = Engine(processes=processes, cs_cost=1, tick_us=1000.0)

    total_penalty = 0
    while not engine.done:
        view = engine.current_view
        assert view is not None
        pid = view.ready[0].pid
        step_info = engine.step(pid, decision_cost_us=500.0)
        total_penalty += step_info.penalty

    result = engine.result()
    total_waiting = sum(pm.waiting for pm in result.process_metrics)
    assert total_penalty == total_waiting


def test_stage11_engine_properties_and_input_validation() -> None:
    """
    1. The use of function:
       Validates property accessors and defensive error handling for Engine parameters.
    2. How does the specific function work:
       Tests invalid tick_us, invalid decision_cost_us, invalid start cost, and invalid restore
       arguments. Tests property returns on Engine.
    3. Concepts used in the function:
       Defensive validation, boundary condition checks, type safety.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Engine instances subject to valid and invalid inputs.
       After: Correct properties returned or TypeError / ValueError raised on violations.
    """
    processes = [Process(pid="P1", arrival=0, burst=2)]

    # Valid properties
    engine = Engine(processes=processes, cs_cost=0, tick_us=500.0)
    assert engine.tick_us == 500.0
    assert engine.ai_overhead_ticks == 0
    assert engine.decisions == 0
    assert engine.inference_us_total == 0.0

    # Invalid tick_us
    with pytest.raises(ValueError, match="strictly positive"):
        Engine(processes=processes, cs_cost=0, tick_us=0)
    with pytest.raises(ValueError, match="strictly positive"):
        Engine(processes=processes, cs_cost=0, tick_us=-10.0)
    with pytest.raises(TypeError, match="tick_us must be a float"):
        Engine(processes=processes, cs_cost=0, tick_us="1000")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tick_us must be a float"):
        Engine(processes=processes, cs_cost=0, tick_us=True)  # type: ignore[arg-type]

    # Invalid decision_cost_us
    with pytest.raises(ValueError, match="cannot be negative"):
        engine.step("P1", decision_cost_us=-1.0)
    with pytest.raises(TypeError, match="must be numeric"):
        engine.step("P1", decision_cost_us="fast")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be numeric"):
        engine.step("P1", decision_cost_us=True)  # type: ignore[arg-type]

    # Invalid start cost
    with pytest.raises(ValueError, match="cannot be negative"):
        engine.charge_start_overhead(-5.0)
    with pytest.raises(TypeError, match="must be numeric"):
        engine.charge_start_overhead(True)  # type: ignore[arg-type]

    # Invalid restore argument
    with pytest.raises(TypeError, match="must be an Engine instance"):
        engine.restore("not_an_engine")  # type: ignore[arg-type]


def test_stage11_simulation_metrics_effective_defaults() -> None:
    """
    1. The use of function:
       Validates SimulationMetrics post-init defaulting and explicit overriding behavior.
    2. How does the specific function work:
       Instantiates SimulationMetrics with default (0 / 0.0) effective metrics, verifying they
       default to makespan and avg_turnaround. Instantiates with explicit values and checks they
       are preserved. Tests calculate_simulation_metrics optional arguments.
    3. Concepts used in the function:
       Dataclass default resolution, backwards-compatible metric computation.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Raw SimulationMetrics initializations.
       After: Correct effective makespan and turnaround assertions verified.
    """
    pm = ProcessMetrics(
        pid="P1",
        arrival=0,
        burst=3,
        completion=5,
        turnaround=5,
        waiting=2,
        response=0,
    )

    # Defaults when not provided
    sm = calculate_simulation_metrics(
        proc_metrics=[pm],
        makespan=5,
        busy_ticks=3,
        overhead_ticks=0,
        context_switches=0,
    )
    assert sm.effective_makespan == 5
    assert sm.effective_turnaround == 5.0
    assert sm.ai_overhead_ticks == 0
    assert sm.decisions == 0
    assert sm.inference_us_total == 0.0

    # Explicit values provided
    sm_explicit = calculate_simulation_metrics(
        proc_metrics=[pm],
        makespan=5,
        busy_ticks=3,
        overhead_ticks=0,
        context_switches=0,
        ai_overhead_ticks=2,
        decisions=1,
        inference_us_total=1000.0,
        effective_makespan=7,
        effective_turnaround=7.0,
    )
    assert sm_explicit.effective_makespan == 7
    assert sm_explicit.effective_turnaround == 7.0
    assert sm_explicit.ai_overhead_ticks == 2
    assert sm_explicit.decisions == 1
    assert sm_explicit.inference_us_total == 1000.0

    # Type validation on calculate_simulation_metrics
    with pytest.raises(TypeError, match="ai_overhead_ticks must be an int"):
        calculate_simulation_metrics(
            [pm], 5, 3, 0, 0, ai_overhead_ticks="0"  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="inference_us_total must be numeric"):
        calculate_simulation_metrics(
            [pm], 5, 3, 0, 0, inference_us_total=True  # type: ignore[arg-type]
        )


def test_stage11_config_settings_and_constants() -> None:
    """
    1. The use of function:
       Validates global Setting configuration definitions and constants in config.py.
    2. How does the specific function work:
       Asserts that Setting is a frozen dataclass, SETTINGS contains S0 and S1 with correct
       cs_cost values, TICK_US_SWEEP matches specification, DEFAULT_TICK_US is 1000, and
       FORCED_DECISION_FREE is True.
    3. Concepts used in the function:
       Configuration contract assertion, specification alignment.
    4. Places where this function is imported and used:
       Executed by pytest test runner.
    5. Final output of the function, before and after:
       Before: Global config definitions.
       After: All configuration constants verified against specification.
    """
    assert isinstance(SETTINGS, dict)
    assert "S0" in SETTINGS
    assert "S1" in SETTINGS

    s0 = SETTINGS["S0"]
    assert isinstance(s0, Setting)
    assert s0.name == "S0"
    assert s0.cs_cost == 0

    s1 = SETTINGS["S1"]
    assert isinstance(s1, Setting)
    assert s1.name == "S1"
    assert s1.cs_cost == 1

    # Setting is immutable
    with pytest.raises(Exception):
        s0.cs_cost = 5  # type: ignore[misc]

    assert TICK_US_SWEEP == [1, 10, 100, 1000, 10000]
    assert DEFAULT_TICK_US == 1000
    assert FORCED_DECISION_FREE is True
