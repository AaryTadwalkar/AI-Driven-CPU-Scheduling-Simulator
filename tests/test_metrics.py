"""
1. What this file does:
   Defines unit tests for ProcessMetrics, SimulationMetrics, and metric calculation functions
   in schedsim.domain.metrics.
2. Overall design:
   Validates per-process metrics, aggregate simulation metrics, Jain's fairness index calculations,
   algebraic invariants, division by zero safeguards, type validations, and immutability.
3. Elements in the file:
   - test_process_metrics_valid_creation: Tests ProcessMetrics creation and attribute access.
   - test_process_metrics_immutability: Tests frozen immutability of ProcessMetrics.
   - test_process_metrics_invalid_types: Tests type validation in ProcessMetrics post-init.
   - test_process_metrics_invalid_bounds: Tests algebraic bounds and invariant checks.
   - test_calculate_process_metrics_standard: Tests standard per-process metric calculation.
   - test_calculate_process_metrics_delayed_and_preempted: Tests delayed and preempted processes.
   - test_calculate_process_metrics_invalid_inputs: Tests input validation on calculation function.
   - test_calculate_jain_fairness_empty: Tests Jain fairness returning 1.0 for empty input.
   - test_calculate_jain_fairness_identical: Tests Jain fairness returning 1.0 for identical inputs.
   - test_calculate_jain_fairness_single: Tests single-item fairness calculation.
   - test_calculate_jain_fairness_asymmetric: Tests asymmetric slowdown values against manual math.
   - test_calculate_jain_fairness_invalid_inputs: Tests type and boundary errors on Jain fairness.
   - test_simulation_metrics_immutability_and_coercion: Tests coercion and frozen immutability.
   - test_calculate_simulation_metrics_fixture_fcfs: Tests aggregate metrics against FCFS fixture.
   - test_calculate_simulation_metrics_fixture_srtf: Tests aggregate metrics against SRTF fixture.
   - test_calculate_simulation_metrics_with_overhead_and_idle: Tests CS overhead and idle gaps.
   - test_calculate_simulation_metrics_zero_makespan_and_empty: Tests zero-makespan safeguards.
   - test_calculate_simulation_metrics_invalid_inputs: Tests input validation on aggregate metric.
4. Final output of the file:
   Pytest assertions confirming correct scheduling metrics calculation and rigorous error checks.
"""

from dataclasses import FrozenInstanceError

import pytest

from schedsim.domain.metrics import (
    ProcessMetrics,
    SimulationMetrics,
    calculate_jain_fairness,
    calculate_process_metrics,
    calculate_simulation_metrics,
)
from schedsim.domain.process import Process


def test_process_metrics_valid_creation() -> None:
    """
    1. The use of function:
       Tests successful instantiation of ProcessMetrics with valid attributes.
    2. How does the specific function work:
       Directly constructs a ProcessMetrics instance and verifies each field value.
    3. Concepts used in the function:
       Dataclass instantiation, attribute verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Raw valid metric parameters.
       After: Confirms attributes match expected inputs.
    """
    m = ProcessMetrics(
        pid="P1",
        arrival=0,
        burst=5,
        completion=5,
        turnaround=5,
        waiting=0,
        response=0,
    )
    assert m.pid == "P1"
    assert m.arrival == 0
    assert m.burst == 5
    assert m.completion == 5
    assert m.turnaround == 5
    assert m.waiting == 0
    assert m.response == 0


def test_process_metrics_immutability() -> None:
    """
    1. The use of function:
       Verifies that ProcessMetrics instances are frozen and immutable.
    2. How does the specific function work:
       Attempts attribute modification on a ProcessMetrics instance and expects FrozenInstanceError.
    3. Concepts used in the function:
       Frozen dataclass, immutability, mutation protection.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Valid immutable ProcessMetrics instance.
       After: Raises FrozenInstanceError when attempting attribute reassignment.
    """
    m = ProcessMetrics(
        pid="P1",
        arrival=0,
        burst=5,
        completion=5,
        turnaround=5,
        waiting=0,
        response=0,
    )
    with pytest.raises(FrozenInstanceError):
        m.completion = 10  # type: ignore[misc]


def test_process_metrics_invalid_types() -> None:
    """
    1. The use of function:
       Verifies that ProcessMetrics rejects invalid attribute types.
    2. How does the specific function work:
       Passes booleans, floats, non-strings, and empty strings into ProcessMetrics and asserts
       TypeError or ValueError.
    3. Concepts used in the function:
       Strict type enforcement, rejection of booleans disguised as integers.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Incompatible type parameters.
       After: Raises TypeError or ValueError.
    """
    with pytest.raises(TypeError, match="pid must be a str"):
        ProcessMetrics(
            pid=123,  # type: ignore[arg-type]
            arrival=0,
            burst=5,
            completion=5,
            turnaround=5,
            waiting=0,
            response=0,
        )

    with pytest.raises(ValueError, match="pid cannot be empty"):
        ProcessMetrics(
            pid="   ",
            arrival=0,
            burst=5,
            completion=5,
            turnaround=5,
            waiting=0,
            response=0,
        )

    with pytest.raises(TypeError, match="arrival must be an int"):
        ProcessMetrics(
            pid="P1",
            arrival=True,  # type: ignore[arg-type]
            burst=5,
            completion=5,
            turnaround=5,
            waiting=0,
            response=0,
        )

    with pytest.raises(TypeError, match="burst must be an int"):
        ProcessMetrics(
            pid="P1",
            arrival=0,
            burst="5",  # type: ignore[arg-type]
            completion=5,
            turnaround=5,
            waiting=0,
            response=0,
        )


def test_process_metrics_invalid_bounds() -> None:
    """
    1. The use of function:
       Tests algebraic boundary enforcement and invariant checks in ProcessMetrics.
    2. How does the specific function work:
       Tests negative arrival, burst < 1, completion < arrival + burst, turnaround mismatch,
       waiting mismatch, and negative response.
    3. Concepts used in the function:
       Domain invariant checking, metric relationships (TAT = CT - AT, WT = TAT - BT).
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Numerically inconsistent or out-of-boundary values.
       After: Raises ValueError with descriptive messages.
    """
    with pytest.raises(ValueError, match="arrival must be >= 0"):
        ProcessMetrics(
            pid="P1",
            arrival=-1,
            burst=5,
            completion=5,
            turnaround=6,
            waiting=1,
            response=0,
        )

    with pytest.raises(ValueError, match="burst must be >= 1"):
        ProcessMetrics(
            pid="P1",
            arrival=0,
            burst=0,
            completion=5,
            turnaround=5,
            waiting=5,
            response=0,
        )

    with pytest.raises(ValueError, match="cannot be less than arrival \\+ burst"):
        ProcessMetrics(
            pid="P1",
            arrival=2,
            burst=5,
            completion=6,
            turnaround=4,
            waiting=-1,
            response=0,
        )

    with pytest.raises(ValueError, match="turnaround .* must equal completion - arrival"):
        ProcessMetrics(
            pid="P1",
            arrival=0,
            burst=5,
            completion=5,
            turnaround=99,
            waiting=0,
            response=0,
        )

    with pytest.raises(ValueError, match="waiting .* must equal turnaround - burst"):
        ProcessMetrics(
            pid="P1",
            arrival=0,
            burst=5,
            completion=5,
            turnaround=5,
            waiting=99,
            response=0,
        )

    with pytest.raises(ValueError, match="response must be >= 0"):
        ProcessMetrics(
            pid="P1",
            arrival=0,
            burst=5,
            completion=5,
            turnaround=5,
            waiting=0,
            response=-1,
        )


def test_calculate_process_metrics_standard() -> None:
    """
    1. The use of function:
       Tests calculate_process_metrics on a standard non-preempted process starting at arrival.
    2. How does the specific function work:
       Passes Process(P1, arrival=0, burst=5), completion=5, first_start=0.
       Verifies derived turnaround=5, waiting=0, response=0.
    3. Concepts used in the function:
       Immediate dispatch, perfect zero waiting/response times.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Raw process definition and timestamps.
       After: Accurately computed ProcessMetrics instance.
    """
    p = Process(pid="P1", arrival=0, burst=5)
    metrics = calculate_process_metrics(p, completion=5, first_start=0)
    assert metrics.pid == "P1"
    assert metrics.arrival == 0
    assert metrics.burst == 5
    assert metrics.completion == 5
    assert metrics.turnaround == 5
    assert metrics.waiting == 0
    assert metrics.response == 0


def test_calculate_process_metrics_delayed_and_preempted() -> None:
    """
    1. The use of function:
       Tests calculate_process_metrics with delayed dispatch and preemption scenarios.
    2. How does the specific function work:
       Evaluates P2 (arrival 1, burst 3, starts at 5, completes at 8) and
       P3 (arrival 2, burst 2, starts at 2, completes at 6).
    3. Concepts used in the function:
       Turnaround = completion - arrival, waiting = turnaround - burst, response = start - arrival.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Process specifications with delayed start and preemption gaps.
       After: Correctly calculated turnaround, waiting, and response metrics.
    """
    # P2: arrived at 1, waited until 5 to start, finished at 8
    p2 = Process(pid="P2", arrival=1, burst=3)
    m2 = calculate_process_metrics(p2, completion=8, first_start=5)
    assert m2.turnaround == 7  # 8 - 1
    assert m2.waiting == 4     # 7 - 3
    assert m2.response == 4    # 5 - 1

    # P3: arrived at 2, started immediately at 2, preempted, finished at 6
    p3 = Process(pid="P3", arrival=2, burst=2)
    m3 = calculate_process_metrics(p3, completion=6, first_start=2)
    assert m3.turnaround == 4  # 6 - 2
    assert m3.waiting == 2     # 4 - 2
    assert m3.response == 0    # 2 - 2


def test_calculate_process_metrics_invalid_inputs() -> None:
    """
    1. The use of function:
       Verifies input validation and error raising in calculate_process_metrics.
    2. How does the specific function work:
       Tests invalid types for process, completion, and first_start, as well as timestamps
       violating arrival and burst constraints.
    3. Concepts used in the function:
       Defensive validation, temporal order verification (first_start >= arrival).
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Invalid function parameters.
       After: Raises TypeError or ValueError.
    """
    p = Process(pid="P1", arrival=2, burst=3)

    with pytest.raises(TypeError, match="Expected Process instance"):
        calculate_process_metrics("not_a_proc", 5, 2)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="completion must be an int"):
        calculate_process_metrics(p, "5", 2)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="first_start must be an int"):
        calculate_process_metrics(p, 5, True)  # type: ignore[arg-type]

    # first_start before arrival
    with pytest.raises(ValueError, match="cannot be earlier than process arrival"):
        calculate_process_metrics(p, completion=6, first_start=1)

    # completion earlier than first_start + burst
    with pytest.raises(ValueError, match="cannot be less than first_start \\+ burst"):
        calculate_process_metrics(p, completion=4, first_start=2)


def test_calculate_jain_fairness_empty() -> None:
    """
    1. The use of function:
       Verifies that calculate_jain_fairness returns 1.0 when given an empty sequence.
    2. How does the specific function work:
       Passes [] and () and asserts return value is exactly 1.0.
    3. Concepts used in the function:
       Jain's index definition for empty workload sets.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Empty sequence of slowdowns.
       After: Asserts result equals 1.0.
    """
    assert calculate_jain_fairness([]) == 1.0
    assert calculate_jain_fairness(()) == 1.0


def test_calculate_jain_fairness_identical() -> None:
    """
    1. The use of function:
       Verifies that calculate_jain_fairness returns 1.0 for homogeneous slowdown values.
    2. How does the specific function work:
       Tests lists of identical floats, identical integers, and all zeros.
    3. Concepts used in the function:
       Maximal fairness invariant: when all entities experience equal treatment, index is 1.0.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Homogeneous sequence of slowdowns.
       After: Asserts result equals 1.0.
    """
    assert calculate_jain_fairness([1.0, 1.0, 1.0]) == 1.0
    assert calculate_jain_fairness([2.5, 2.5]) == 1.0
    assert calculate_jain_fairness([0.0, 0.0]) == 1.0
    assert calculate_jain_fairness([4, 4, 4, 4]) == 1.0


def test_calculate_jain_fairness_single() -> None:
    """
    1. The use of function:
       Verifies that calculate_jain_fairness returns 1.0 for a single entity workload.
    2. How does the specific function work:
       Passes a single-element list and asserts the fairness index is 1.0.
    3. Concepts used in the function:
       Single process fairness boundary condition.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Sequence with one slowdown value.
       After: Asserts result equals 1.0.
    """
    assert calculate_jain_fairness([1.5]) == 1.0
    assert calculate_jain_fairness([10.0]) == 1.0


def test_calculate_jain_fairness_asymmetric() -> None:
    """
    1. The use of function:
       Tests calculate_jain_fairness against known mathematical hand calculations.
    2. How does the specific function work:
       - [1.0, 2.0]: (3)^2 / (2 * 5) = 9 / 10 = 0.9.
       - [2.0, 1.0, 2.0]: (5)^2 / (3 * 9) = 25 / 27 ≈ 0.9259259.
       - Section 4.6 FCFS fixture: [1.0, 7/3, 4.0]: (22/3)^2 / (3 * 202/9) = 242 / 303 ≈ 0.79868.
    3. Concepts used in the function:
       Jain's fairness index formula: (sum(x))^2 / (n * sum(x^2)).
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Sequences with asymmetric slowdowns.
       After: Asserts results match expected formulas within pytest.approx tolerances.
    """
    res1 = calculate_jain_fairness([1.0, 2.0])
    assert res1 == pytest.approx(0.9)

    res2 = calculate_jain_fairness([2.0, 1.0, 2.0])
    assert res2 == pytest.approx(25.0 / 27.0)

    slowdowns_fixture = [1.0, 7.0 / 3.0, 4.0]
    res3 = calculate_jain_fairness(slowdowns_fixture)
    assert res3 == pytest.approx(242.0 / 303.0)


def test_calculate_jain_fairness_invalid_inputs() -> None:
    """
    1. The use of function:
       Tests input validation and error raising for calculate_jain_fairness.
    2. How does the specific function work:
       Passes non-sequence objects, non-numeric elements, and negative values to assert errors.
    3. Concepts used in the function:
       Defensive validation, type checking, numerical range verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Invalid slowdown inputs.
       After: Raises TypeError or ValueError.
    """
    with pytest.raises(TypeError, match="Expected Sequence for slowdowns"):
        calculate_jain_fairness(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be numeric"):
        calculate_jain_fairness([1.0, "two", 3.0])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="must be numeric"):
        calculate_jain_fairness([1.0, True])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_jain_fairness([1.0, -0.5, 2.0])


def test_simulation_metrics_immutability_and_coercion() -> None:
    """
    1. The use of function:
       Tests type coercion, validation, and frozen immutability of SimulationMetrics.
    2. How does the specific function work:
       Supplies integers to float fields and verifies coercion. Attempts attribute reassignment
       to assert FrozenInstanceError. Tests negative metric bounds and jain_fairness > 1.0.
    3. Concepts used in the function:
       Dynamic float coercion, frozen dataclass immutability, boundary validation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Valid and invalid parameter configurations for SimulationMetrics.
       After: Confirms immutability, successful type coercion, and error signaling on invalid data.
    """
    metrics = SimulationMetrics(
        avg_waiting=0,
        avg_turnaround=10,
        avg_response=2,
        max_waiting=4,
        makespan=15,
        cpu_utilization=1,
        overhead_ticks=0,
        throughput=0,
        context_switches=1,
        jain_fairness=1,
    )
    assert isinstance(metrics.avg_waiting, float)
    assert isinstance(metrics.avg_turnaround, float)
    assert isinstance(metrics.cpu_utilization, float)
    assert isinstance(metrics.throughput, float)
    assert isinstance(metrics.jain_fairness, float)

    with pytest.raises(FrozenInstanceError):
        metrics.makespan = 20  # type: ignore[misc]

    with pytest.raises(TypeError, match="must be a float"):
        SimulationMetrics(
            avg_waiting="invalid",  # type: ignore[arg-type]
            avg_turnaround=10.0,
            avg_response=2.0,
            max_waiting=4,
            makespan=15,
            cpu_utilization=1.0,
            overhead_ticks=0,
            throughput=0.2,
            context_switches=1,
            jain_fairness=1.0,
        )

    with pytest.raises(TypeError, match="must be an int"):
        SimulationMetrics(
            avg_waiting=0.0,
            avg_turnaround=10.0,
            avg_response=2.0,
            max_waiting="invalid",  # type: ignore[arg-type]
            makespan=15,
            cpu_utilization=1.0,
            overhead_ticks=0,
            throughput=0.2,
            context_switches=1,
            jain_fairness=1.0,
        )

    with pytest.raises(ValueError, match="must be >= 0.0"):
        SimulationMetrics(
            avg_waiting=-1.0,
            avg_turnaround=10.0,
            avg_response=2.0,
            max_waiting=4,
            makespan=15,
            cpu_utilization=1.0,
            overhead_ticks=0,
            throughput=0.2,
            context_switches=1,
            jain_fairness=1.0,
        )

    with pytest.raises(ValueError, match="must be >= 0"):
        SimulationMetrics(
            avg_waiting=0.0,
            avg_turnaround=10.0,
            avg_response=2.0,
            max_waiting=-1,
            makespan=15,
            cpu_utilization=1.0,
            overhead_ticks=0,
            throughput=0.2,
            context_switches=1,
            jain_fairness=1.0,
        )

    with pytest.raises(ValueError, match="jain_fairness must be <= 1.0"):
        SimulationMetrics(
            avg_waiting=0.0,
            avg_turnaround=10.0,
            avg_response=2.0,
            max_waiting=4,
            makespan=15,
            cpu_utilization=1.0,
            overhead_ticks=0,
            throughput=0.2,
            context_switches=1,
            jain_fairness=1.5,
        )

    clamped = SimulationMetrics(
        avg_waiting=0.0,
        avg_turnaround=10.0,
        avg_response=2.0,
        max_waiting=4,
        makespan=15,
        cpu_utilization=1.0,
        overhead_ticks=0,
        throughput=0.2,
        context_switches=1,
        jain_fairness=1.0 + 1e-12,
    )
    assert clamped.jain_fairness == 1.0


def test_calculate_simulation_metrics_fixture_fcfs() -> None:
    """
    1. The use of function:
       Validates calculate_simulation_metrics against the Section 4.6 hand-verified FCFS fixture.
    2. How does the specific function work:
       Sets up P1 (arr 0, burst 5), P2 (arr 1, burst 3), P3 (arr 2, burst 2) under FCFS:
       - P1: completion 5, first_start 0 (wait 0, tat 5, resp 0)
       - P2: completion 8, first_start 5 (wait 4, tat 7, resp 4)
       - P3: completion 10, first_start 8 (wait 6, tat 8, resp 6)
       Runs calculate_simulation_metrics with makespan=10, busy=10, overhead=0, switches=2.
    3. Concepts used in the function:
       System-level aggregate benchmarks, FCFS scheduling verification, Section 4.6 specification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: ProcessMetrics list corresponding to Section 4.6 FCFS run.
       After: Confirms exact numerical match for all aggregate fields.
    """
    p1 = Process("P1", 0, 5)
    p2 = Process("P2", 1, 3)
    p3 = Process("P3", 2, 2)

    m1 = calculate_process_metrics(p1, completion=5, first_start=0)
    m2 = calculate_process_metrics(p2, completion=8, first_start=5)
    m3 = calculate_process_metrics(p3, completion=10, first_start=8)

    sim = calculate_simulation_metrics(
        proc_metrics=[m1, m2, m3],
        makespan=10,
        busy_ticks=10,
        overhead_ticks=0,
        context_switches=2,
    )

    assert sim.avg_waiting == pytest.approx(10.0 / 3.0)
    assert sim.avg_turnaround == pytest.approx(20.0 / 3.0)
    assert sim.avg_response == pytest.approx(10.0 / 3.0)
    assert sim.max_waiting == 6
    assert sim.makespan == 10
    assert sim.cpu_utilization == 1.0
    assert sim.overhead_ticks == 0
    assert sim.throughput == pytest.approx(0.3)
    assert sim.context_switches == 2
    assert sim.jain_fairness == pytest.approx(242.0 / 303.0)


def test_calculate_simulation_metrics_fixture_srtf() -> None:
    """
    1. The use of function:
       Validates calculate_simulation_metrics against the Section 4.6 hand-verified SRTF fixture.
    2. How does the specific function work:
       Sets up P1, P2, P3 under SRTF:
       - P1: completion 10, first_start 0 (wait 5, tat 10, resp 0)
       - P2: completion 4, first_start 1 (wait 0, tat 3, resp 0)
       - P3: completion 6, first_start 4 (wait 2, tat 4, resp 2)
       Runs calculate_simulation_metrics with makespan=10, busy=10, overhead=0, switches=2.
    3. Concepts used in the function:
       Preemptive scheduling aggregate metrics, SRTF verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: ProcessMetrics list corresponding to Section 4.6 SRTF run.
       After: Confirms exact numerical match for SRTF averages and Jain fairness.
    """
    p1 = Process("P1", 0, 5)
    p2 = Process("P2", 1, 3)
    p3 = Process("P3", 2, 2)

    m1 = calculate_process_metrics(p1, completion=10, first_start=0)
    m2 = calculate_process_metrics(p2, completion=4, first_start=1)
    m3 = calculate_process_metrics(p3, completion=6, first_start=4)

    sim = calculate_simulation_metrics(
        proc_metrics=[m1, m2, m3],
        makespan=10,
        busy_ticks=10,
        overhead_ticks=0,
        context_switches=2,
    )

    assert sim.avg_waiting == pytest.approx(7.0 / 3.0)
    assert sim.avg_turnaround == pytest.approx(17.0 / 3.0)
    assert sim.avg_response == pytest.approx(2.0 / 3.0)
    assert sim.max_waiting == 5
    assert sim.makespan == 10
    assert sim.cpu_utilization == 1.0
    assert sim.throughput == pytest.approx(0.3)
    # slowdowns: P1=2.0, P2=1.0, P3=2.0 -> fairness = 25 / 27
    assert sim.jain_fairness == pytest.approx(25.0 / 27.0)


def test_calculate_simulation_metrics_with_overhead_and_idle() -> None:
    """
    1. The use of function:
       Tests calculate_simulation_metrics with context switch overhead and idle periods.
    2. How does the specific function work:
       Simulates a scenario with makespan=15, busy_ticks=10, overhead_ticks=2, idle ticks=3.
       Verifies cpu_utilization is 10/15 and throughput is 2/15.
    3. Concepts used in the function:
       CPU utilization accounting, throughput calculation under non-busy periods.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Raw counters with overhead and idle time.
       After: Correctly calculated fractional utilization and throughput.
    """
    p1 = Process("P1", 0, 5)
    p2 = Process("P2", 0, 5)
    m1 = calculate_process_metrics(p1, completion=5, first_start=0)
    m2 = calculate_process_metrics(p2, completion=15, first_start=8)

    sim = calculate_simulation_metrics(
        proc_metrics=[m1, m2],
        makespan=15,
        busy_ticks=10,
        overhead_ticks=2,
        context_switches=1,
    )

    assert sim.makespan == 15
    assert sim.overhead_ticks == 2
    assert sim.cpu_utilization == pytest.approx(10.0 / 15.0)
    assert sim.throughput == pytest.approx(2.0 / 15.0)


def test_calculate_simulation_metrics_zero_makespan_and_empty() -> None:
    """
    1. The use of function:
       Tests division by zero safeguards when proc_metrics is empty and makespan is 0.
    2. How does the specific function work:
       Calls calculate_simulation_metrics with empty sequence and zero counters.
       Asserts all averages and rates are 0.0, and jain_fairness is 1.0.
    3. Concepts used in the function:
       Defensive zero-division protection for empty simulation runs.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Empty process list and zero-tick makespan.
       After: Returns SimulationMetrics with 0.0 values without ZeroDivisionError.
    """
    sim = calculate_simulation_metrics(
        proc_metrics=[],
        makespan=0,
        busy_ticks=0,
        overhead_ticks=0,
        context_switches=0,
    )

    assert sim.avg_waiting == 0.0
    assert sim.avg_turnaround == 0.0
    assert sim.avg_response == 0.0
    assert sim.max_waiting == 0
    assert sim.makespan == 0
    assert sim.cpu_utilization == 0.0
    assert sim.overhead_ticks == 0
    assert sim.throughput == 0.0
    assert sim.context_switches == 0
    assert sim.jain_fairness == 1.0


def test_calculate_simulation_metrics_invalid_inputs() -> None:
    """
    1. The use of function:
       Verifies input validation and error raising for calculate_simulation_metrics.
    2. How does the specific function work:
       Tests non-sequence proc_metrics, sequence containing non-ProcessMetrics items, and
       negative integer counters.
    3. Concepts used in the function:
       Type and boundary validation on aggregate metrics input parameters.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Invalid parameters for calculate_simulation_metrics.
       After: Raises TypeError or ValueError.
    """
    with pytest.raises(TypeError, match="Expected Sequence for proc_metrics"):
        calculate_simulation_metrics(
            proc_metrics="invalid",  # type: ignore[arg-type]
            makespan=10,
            busy_ticks=5,
            overhead_ticks=0,
            context_switches=0,
        )

    with pytest.raises(TypeError, match="must be ProcessMetrics"):
        calculate_simulation_metrics(
            proc_metrics=["not_a_metric"],  # type: ignore[list-item]
            makespan=10,
            busy_ticks=5,
            overhead_ticks=0,
            context_switches=0,
        )

    with pytest.raises(TypeError, match="makespan must be an int"):
        calculate_simulation_metrics(
            proc_metrics=[],
            makespan=True,  # type: ignore[arg-type]
            busy_ticks=0,
            overhead_ticks=0,
            context_switches=0,
        )

    with pytest.raises(ValueError, match="makespan cannot be negative"):
        calculate_simulation_metrics(
            proc_metrics=[],
            makespan=-1,
            busy_ticks=0,
            overhead_ticks=0,
            context_switches=0,
        )

    with pytest.raises(ValueError, match="busy_ticks cannot be negative"):
        calculate_simulation_metrics(
            proc_metrics=[],
            makespan=10,
            busy_ticks=-5,
            overhead_ticks=0,
            context_switches=0,
        )

    with pytest.raises(ValueError, match="overhead_ticks cannot be negative"):
        calculate_simulation_metrics(
            proc_metrics=[],
            makespan=10,
            busy_ticks=5,
            overhead_ticks=-1,
            context_switches=0,
        )

    with pytest.raises(ValueError, match="context_switches cannot be negative"):
        calculate_simulation_metrics(
            proc_metrics=[],
            makespan=10,
            busy_ticks=5,
            overhead_ticks=0,
            context_switches=-1,
        )
