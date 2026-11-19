"""
1. What this file does:
   Defines metrics data structures and calculation routines for CPU scheduling simulations,
   providing both per-process and aggregate simulation performance evaluations.
2. Overall design:
   Contains immutable value objects (ProcessMetrics, SimulationMetrics) and deterministic
   pure functions to compute standard operating systems scheduling metrics (turnaround, waiting,
   response, makespan, CPU utilization, throughput) and Jain's fairness index over slowdowns.
3. Elements in the file:
   - ProcessMetrics: Frozen dataclass encapsulating per-process lifecycle time metrics.
   - SimulationMetrics: Frozen dataclass encapsulating system-wide aggregate benchmark metrics.
   - calculate_jain_fairness: Standalone helper evaluating Jain's fairness index over slowdowns.
   - calculate_process_metrics: Derives per-process turnaround, waiting, and response times.
   - calculate_simulation_metrics: Computes aggregate averages, utilization, and fairness.
4. Final output of the file:
   Exports domain metric entities and evaluation functions for engine result reporting,
   baseline algorithm comparison, and reinforcement learning reward evaluation.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from schedsim.domain.process import Process


@dataclass(frozen=True)
class ProcessMetrics:
    """
    1. The use of class:
       Encapsulates immutable per-process scheduling metrics calculated at process completion.
    2. What the class contains:
       - pid: Identifier string matching the associated Process.
       - arrival: Arrival tick when the process entered the system.
       - burst: Total CPU computation duration required by the process.
       - completion: Tick timestamp when the process completed execution.
       - turnaround: Total elapsed time from arrival to completion (completion - arrival).
       - waiting: Total time spent ready but not executing on CPU (turnaround - burst).
       - response: Initial delay between process arrival and first CPU dispatch.
       - __post_init__: Integrity validator enforcing non-negative values and metric invariants.
    3. Technologies used and why:
       - Python dataclasses with frozen=True for immutability, thread-safety, and hashing.
       - Strict type and algebraic invariant assertions to guarantee simulation correctness.
    4. Key design and overall design of the class:
       Acts as a domain value object representing completed process performance. Invariants
       ensure mathematical consistency across all scheduling algorithms.
    """

    pid: str
    arrival: int
    burst: int
    completion: int
    turnaround: int
    waiting: int
    response: int

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates field types, non-negativity, and algebraic invariants upon instantiation.
        2. How does the specific function work:
           Verifies that pid is a non-empty string and all timing metrics are non-negative
           integers. Confirms turnaround == completion - arrival, waiting == turnaround - burst,
           and response >= 0.
        3. Concepts used in the function:
           Dataclass post-init hook, invariant validation, defensive programming.
        4. Places where this function is imported and used:
           Invoked automatically during ProcessMetrics instantiation.
        5. Final output of the function, before and after:
           Before: Unverified raw metric field values.
           After: Verified immutable instance or raises TypeError / ValueError on violation.
        """
        if not isinstance(self.pid, str):
            raise TypeError(f"ProcessMetrics pid must be a str, got {type(self.pid).__name__}.")
        if not self.pid.strip():
            raise ValueError("ProcessMetrics pid cannot be empty or whitespace only.")

        int_fields = {
            "arrival": self.arrival,
            "burst": self.burst,
            "completion": self.completion,
            "turnaround": self.turnaround,
            "waiting": self.waiting,
            "response": self.response,
        }
        for name, val in int_fields.items():
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"ProcessMetrics {name} must be an int, got {type(val).__name__}.")

        if self.arrival < 0:
            raise ValueError(f"ProcessMetrics arrival must be >= 0, got {self.arrival}.")
        if self.burst < 1:
            raise ValueError(f"ProcessMetrics burst must be >= 1, got {self.burst}.")
        if self.completion < self.arrival + self.burst:
            raise ValueError(
                f"ProcessMetrics completion ({self.completion}) cannot be less than "
                f"arrival + burst ({self.arrival + self.burst})."
            )
        if self.turnaround != self.completion - self.arrival:
            raise ValueError(
                f"ProcessMetrics turnaround ({self.turnaround}) must equal "
                f"completion - arrival ({self.completion - self.arrival})."
            )
        if self.waiting != self.turnaround - self.burst:
            raise ValueError(
                f"ProcessMetrics waiting ({self.waiting}) must equal "
                f"turnaround - burst ({self.turnaround - self.burst})."
            )
        if self.response < 0:
            raise ValueError(f"ProcessMetrics response must be >= 0, got {self.response}.")


@dataclass(frozen=True)
class SimulationMetrics:
    """
    1. The use of class:
       Encapsulates system-level aggregate scheduling metrics for an entire simulation run,
       incorporating AI model inference stalls and effective throughput measures.
    2. What the class contains:
       - avg_waiting: Mean waiting time across all finished processes.
       - avg_turnaround: Mean turnaround time across all finished processes.
       - avg_response: Mean response time across all finished processes.
       - max_waiting: Maximum waiting time observed among any finished process.
       - makespan: Total simulation ticks from tick 0 to full workload completion.
       - cpu_utilization: Fraction of makespan spent executing user process bursts.
       - overhead_ticks: Cumulative ticks spent on context switch penalties.
       - throughput: Completed processes per unit simulation time (n / makespan).
       - context_switches: Total count of process preemptions and context dispatches.
       - jain_fairness: Jain's fairness index computed over per-process slowdowns (T / B).
       - ai_overhead_ticks: Total ticks consumed by AI inference stalls ('AI').
       - decisions: Total count of non-forced scheduling decisions made.
       - inference_us_total: Total raw model inference latency accumulated in microseconds.
       - effective_makespan: Total makespan including AI stall overhead ticks.
       - effective_turnaround: Average turnaround time including AI stall delays.
       - __post_init__: Integrity validator checking types, defaults, and non-negative boundaries.
    3. Technologies used and why:
       - Frozen dataclass for immutability, thread-safety, and clean equality comparisons.
       - Automatic float conversion of numerical attributes for robust interoperability.
    4. Key design and overall design of the class:
       Serves as the primary evaluation entity summarizing simulation performance, utilized
       for comparing classical policies, benchmarking RL agents, and overhead break-even analysis.
    """

    avg_waiting: float
    avg_turnaround: float
    avg_response: float
    max_waiting: int
    makespan: int
    cpu_utilization: float
    overhead_ticks: int
    throughput: float
    context_switches: int
    jain_fairness: float
    ai_overhead_ticks: int = 0
    decisions: int = 0
    inference_us_total: float = 0.0
    effective_makespan: int = 0
    effective_turnaround: float = 0.0

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates metric types, non-negativity, and defaults upon instantiation.
        2. How does the specific function work:
           Ensures float metrics are valid non-boolean numerical values (coercing int to float if
           needed), verifies int metrics are non-negative, sets effective_makespan to makespan if 0,
           sets effective_turnaround to avg_turnaround if 0.0, and checks jain_fairness bounds.
        3. Concepts used in the function:
           Dataclass post-init hook, numerical bounds checking, dynamic attribute coercion.
        4. Places where this function is imported and used:
           Invoked automatically during SimulationMetrics instantiation.
        5. Final output of the function, before and after:
           Before: Unverified raw metric inputs.
           After: Verified immutable instance with initialized effective metrics.
        """
        float_fields = (
            "avg_waiting",
            "avg_turnaround",
            "avg_response",
            "cpu_utilization",
            "throughput",
            "jain_fairness",
            "inference_us_total",
            "effective_turnaround",
        )
        for name in float_fields:
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise TypeError(
                    f"SimulationMetrics {name} must be a float, got {type(val).__name__}."
                )
            if not isinstance(val, float):
                object.__setattr__(self, name, float(val))

        int_fields = {
            "max_waiting": self.max_waiting,
            "makespan": self.makespan,
            "overhead_ticks": self.overhead_ticks,
            "context_switches": self.context_switches,
            "ai_overhead_ticks": self.ai_overhead_ticks,
            "decisions": self.decisions,
            "effective_makespan": self.effective_makespan,
        }
        for name, ival in int_fields.items():
            if isinstance(ival, bool) or not isinstance(ival, int):
                raise TypeError(
                    f"SimulationMetrics {name} must be an int, got {type(ival).__name__}."
                )
            if ival < 0:
                raise ValueError(f"SimulationMetrics {name} must be >= 0, got {ival}.")

        for name in float_fields:
            fval = getattr(self, name)
            if fval < 0.0:
                raise ValueError(f"SimulationMetrics {name} must be >= 0.0, got {fval}.")

        if self.effective_makespan == 0:
            object.__setattr__(self, "effective_makespan", self.makespan)
        if self.effective_turnaround == 0.0:
            object.__setattr__(self, "effective_turnaround", self.avg_turnaround)

        if self.jain_fairness > 1.0 + 1e-9:
            raise ValueError(
                f"SimulationMetrics jain_fairness must be <= 1.0, got {self.jain_fairness}."
            )
        if self.jain_fairness > 1.0:
            object.__setattr__(self, "jain_fairness", 1.0)


def calculate_jain_fairness(slowdowns: Sequence[float]) -> float:
    """
    1. The use of function:
       Computes Jain's fairness index over a sequence of per-process slowdown values.
    2. How does the specific function work:
       Checks input sequence integrity and non-negativity. If slowdowns is empty or all elements
       are identical, returns 1.0. Otherwise computes (sum(x))^2 / (n * sum(x^2)) and clamps
       the result to [0.0, 1.0].
    3. Concepts used in the function:
       Jain's fairness index (Raj Jain et al., 1984), slowdown metric (turnaround / burst),
       algebraic edge-case protection against division by zero.
    4. Places where this function is imported and used:
       Imported by calculate_simulation_metrics, benchmark evaluation scripts, and test suites.
    5. Final output of the function, before and after:
       Before: Sequence of float/int slowdown values.
       After: Float fairness score in the range [0.0, 1.0].
    """
    if isinstance(slowdowns, (str, bytes)) or not isinstance(slowdowns, Sequence):
        raise TypeError(
            f"Expected Sequence for slowdowns, got {type(slowdowns).__name__}."
        )

    if len(slowdowns) == 0:
        return 1.0

    float_values: list[float] = []
    for idx, val in enumerate(slowdowns):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(
                f"Slowdown item at index {idx} must be numeric, got {type(val).__name__}."
            )
        fval = float(val)
        if fval < 0.0:
            raise ValueError(
                f"Slowdown item at index {idx} cannot be negative, got {fval}."
            )
        float_values.append(fval)

    first_val = float_values[0]
    if all(val == first_val for val in float_values):
        return 1.0

    sum_x = sum(float_values)
    sum_sq = sum(val * val for val in float_values)
    n = len(float_values)
    denominator = n * sum_sq

    raw_fairness = (sum_x * sum_x) / denominator
    return min(1.0, max(0.0, float(raw_fairness)))


def calculate_process_metrics(
    process: Process, completion: int, first_start: int
) -> ProcessMetrics:
    """
    1. The use of function:
       Computes per-process scheduling metrics (turnaround, waiting, response) for a finished
       Process instance given its completion and initial execution timestamps.
    2. How does the specific function work:
       Validates process, completion, and first_start inputs. Asserts that first_start is at or
       after process.arrival, and completion is at least first_start + process.burst. Computes
       turnaround = completion - arrival, waiting = turnaround - burst, and response =
       first_start - arrival, returning a frozen ProcessMetrics instance.
    3. Concepts used in the function:
       Turnaround Time (TAT = CT - AT), Waiting Time (WT = TAT - BT), Response Time (RT =
       First Start - AT), Operating Systems scheduling metrics.
    4. Places where this function is imported and used:
       Imported by Engine.result(), workload evaluators, simulation post-processors, and tests.
    5. Final output of the function, before and after:
       Before: Process definition with completion and first_start tick timestamps.
       After: Immutable ProcessMetrics instance populated with validated metric values.
    """
    if not isinstance(process, Process):
        raise TypeError(f"Expected Process instance, got {type(process).__name__}.")

    if isinstance(completion, bool) or not isinstance(completion, int):
        raise TypeError(f"completion must be an int, got {type(completion).__name__}.")
    if isinstance(first_start, bool) or not isinstance(first_start, int):
        raise TypeError(f"first_start must be an int, got {type(first_start).__name__}.")

    if first_start < process.arrival:
        raise ValueError(
            f"first_start ({first_start}) cannot be earlier than process arrival "
            f"({process.arrival})."
        )
    if completion < first_start + process.burst:
        raise ValueError(
            f"completion ({completion}) cannot be less than first_start + burst "
            f"({first_start + process.burst})."
        )

    turnaround = completion - process.arrival
    waiting = turnaround - process.burst
    response = first_start - process.arrival

    return ProcessMetrics(
        pid=process.pid,
        arrival=process.arrival,
        burst=process.burst,
        completion=completion,
        turnaround=turnaround,
        waiting=waiting,
        response=response,
    )


def calculate_simulation_metrics(
    proc_metrics: Sequence[ProcessMetrics],
    makespan: int,
    busy_ticks: int,
    overhead_ticks: int,
    context_switches: int,
    ai_overhead_ticks: int = 0,
    decisions: int = 0,
    inference_us_total: float = 0.0,
    effective_makespan: int | None = None,
    effective_turnaround: float | None = None,
) -> SimulationMetrics:
    """
    1. The use of function:
       Computes aggregate simulation performance metrics across all completed processes,
       engine execution counters, and AI overhead measurements.
    2. How does the specific function work:
       Validates input collections, counters, and optional AI overhead metrics. Computes
       average waiting, turnaround, response, maximum waiting, CPU utilization, throughput,
       and Jain's fairness index over process slowdowns. Populates effective makespan and
       effective turnaround defaults, returning an immutable SimulationMetrics instance.
    3. Concepts used in the function:
       Aggregate performance evaluation, CPU utilization fraction, system throughput,
       Jain's fairness index, AI stall accounting, defensive validation.
    4. Places where this function is imported and used:
       Imported by Engine.result(), simulation experiment runners, RL evaluation wrappers,
       and unit tests.
    5. Final output of the function, before and after:
       Before: Sequence of ProcessMetrics and aggregate engine execution tick counters.
       After: Immutable SimulationMetrics instance summarizing overall system performance.
    """
    if isinstance(proc_metrics, (str, bytes)) or not isinstance(proc_metrics, Sequence):
        raise TypeError(
            f"Expected Sequence for proc_metrics, got {type(proc_metrics).__name__}."
        )

    for idx, item in enumerate(proc_metrics):
        if not isinstance(item, ProcessMetrics):
            raise TypeError(
                f"Item at index {idx} in proc_metrics must be ProcessMetrics, "
                f"got {type(item).__name__}."
            )

    counters = {
        "makespan": makespan,
        "busy_ticks": busy_ticks,
        "overhead_ticks": overhead_ticks,
        "context_switches": context_switches,
        "ai_overhead_ticks": ai_overhead_ticks,
        "decisions": decisions,
    }
    for name, cval in counters.items():
        if isinstance(cval, bool) or not isinstance(cval, int):
            raise TypeError(f"{name} must be an int, got {type(cval).__name__}.")
        if cval < 0:
            raise ValueError(f"{name} cannot be negative, got {cval}.")

    if isinstance(inference_us_total, bool) or not isinstance(
        inference_us_total, (int, float)
    ):
        raise TypeError(
            f"inference_us_total must be numeric, got {type(inference_us_total).__name__}."
        )
    if inference_us_total < 0:
        raise ValueError(
            f"inference_us_total cannot be negative, got {inference_us_total}."
        )

    if effective_makespan is not None:
        if isinstance(effective_makespan, bool) or not isinstance(effective_makespan, int):
            raise TypeError(
                f"effective_makespan must be an int, got {type(effective_makespan).__name__}."
            )
        if effective_makespan < 0:
            raise ValueError(
                f"effective_makespan cannot be negative, got {effective_makespan}."
            )

    if effective_turnaround is not None:
        if isinstance(effective_turnaround, bool) or not isinstance(
            effective_turnaround, (int, float)
        ):
            raise TypeError(
                f"effective_turnaround must be numeric, got "
                f"{type(effective_turnaround).__name__}."
            )
        if effective_turnaround < 0:
            raise ValueError(
                f"effective_turnaround cannot be negative, got {effective_turnaround}."
            )

    n = len(proc_metrics)
    if n > 0:
        avg_waiting = sum(m.waiting for m in proc_metrics) / n
        avg_turnaround = sum(m.turnaround for m in proc_metrics) / n
        avg_response = sum(m.response for m in proc_metrics) / n
        max_waiting = max(m.waiting for m in proc_metrics)
        slowdowns = [m.turnaround / m.burst for m in proc_metrics]
        jain_fairness = calculate_jain_fairness(slowdowns)
    else:
        avg_waiting = 0.0
        avg_turnaround = 0.0
        avg_response = 0.0
        max_waiting = 0
        jain_fairness = 1.0

    cpu_utilization = float(busy_ticks / makespan) if makespan > 0 else 0.0
    throughput = float(n / makespan) if makespan > 0 else 0.0

    eff_makespan = makespan if effective_makespan is None else effective_makespan
    eff_turnaround = (
        avg_turnaround if effective_turnaround is None else float(effective_turnaround)
    )

    return SimulationMetrics(
        avg_waiting=avg_waiting,
        avg_turnaround=avg_turnaround,
        avg_response=avg_response,
        max_waiting=max_waiting,
        makespan=makespan,
        cpu_utilization=cpu_utilization,
        overhead_ticks=overhead_ticks,
        throughput=throughput,
        context_switches=context_switches,
        jain_fairness=jain_fairness,
        ai_overhead_ticks=ai_overhead_ticks,
        decisions=decisions,
        inference_us_total=float(inference_us_total),
        effective_makespan=eff_makespan,
        effective_turnaround=eff_turnaround,
    )
