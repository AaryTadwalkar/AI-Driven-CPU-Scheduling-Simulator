"""
1. What this file does:
   Implements the core discrete unit-step CPU scheduling simulation engine, state views,
   step execution logic, AI model inference stall overhead accounting, engine cloning,
   scripted testing policy, and execution runner.
2. Overall design:
   Provides a deterministic discrete-event simulation engine advancing integer tick intervals
   [t, t+1). Features exact context switch overhead accounting, idle time jumping, ready queue
   admission tracking, process lifecycle recording, AI model latency conversion to stall ticks
   via a fractional debt accumulator, and step penalty calculation matching reinforcement
   learning reward semantics.
3. Elements in the file:
   - ProcView: Frozen dataclass representing an immutable process snapshot for policies.
   - View: Frozen dataclass representing the observable simulation state at a discrete tick.
   - StepInfo: Frozen dataclass returning penalty, resulting view, and completion flag per step.
   - SimResult: Frozen dataclass encapsulating process metrics, simulation metrics, and timeline.
   - ScriptedPolicy: Policy driven by a predetermined sequence of PIDs or a custom callable.
   - Engine: Central stateful simulation engine driving unit-step execution with AI stalls.
   - run: Simulation loop helper executing a policy over a process workload to completion.
4. Final output of the file:
   Exports Engine, ProcView, View, StepInfo, SimResult, ScriptedPolicy, and run for domain,
   classical algorithms, ML/RL agents, testing fixtures, and evaluation suites.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from schedsim.domain.metrics import (
    ProcessMetrics,
    SimulationMetrics,
    calculate_process_metrics,
    calculate_simulation_metrics,
)
from schedsim.domain.process import Process, validate_processes
from schedsim.domain.timeline import AI_LABEL, CS_LABEL, IDLE_LABEL, Timeline


@dataclass(frozen=True)
class ProcView:
    """
    1. The use of class:
       Provides an immutable read-only snapshot of a process's state exposed to scheduling policies.
    2. What the class contains:
       - pid: Process identifier string.
       - arrival: Arrival tick when the process entered the system.
       - burst: Total CPU computation duration required by the process.
       - remaining: Remaining CPU computation ticks needed for completion.
       - waited: Cumulative ticks spent in the ready queue without CPU execution.
       - last_run_end: Tick timestamp when the process last ran on CPU, or None if never run.
       - __post_init__: Integrity validator enforcing attribute types, non-negativity, and ranges.
    3. Technologies used and why:
       - Frozen dataclass for immutability, thread-safety, and structural equality.
       - Python 3.11 type hints with strict runtime type and boundary assertions.
    4. Key design and overall design of the class:
       Acts as an isolated domain view object shielding internal mutable engine state from
       policy mutation while supplying all necessary scheduling decision features.
    """

    pid: str
    arrival: int
    burst: int
    remaining: int
    waited: int
    last_run_end: int | None = None

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates ProcView attribute types, non-negativity, and mathematical ranges.
        2. How does the specific function work:
           Verifies pid is a non-empty string, arrival, burst, remaining, and waited are integers,
           and remaining is within [0, burst]. Confirms last_run_end is non-negative int or None.
        3. Concepts used in the function:
           Dataclass post-init hook, invariant validation, defensive boundary verification.
        4. Places where this function is imported and used:
           Invoked automatically during ProcView instantiation.
        5. Final output of the function, before and after:
           Before: Unverified raw field attributes.
           After: Confirmed valid invariant state or raises TypeError / ValueError.
        """
        if not isinstance(self.pid, str):
            raise TypeError(f"ProcView pid must be a str, got {type(self.pid).__name__}.")
        if not self.pid.strip():
            raise ValueError("ProcView pid cannot be empty or whitespace only.")

        int_fields = {
            "arrival": self.arrival,
            "burst": self.burst,
            "remaining": self.remaining,
            "waited": self.waited,
        }
        for name, val in int_fields.items():
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"ProcView {name} must be an int, got {type(val).__name__}.")

        if self.arrival < 0:
            raise ValueError(f"ProcView arrival must be >= 0, got {self.arrival}.")
        if self.burst < 1:
            raise ValueError(f"ProcView burst must be >= 1, got {self.burst}.")
        if self.remaining < 0 or self.remaining > self.burst:
            raise ValueError(
                f"ProcView remaining ({self.remaining}) must be in range [0, {self.burst}]."
            )
        if self.waited < 0:
            raise ValueError(f"ProcView waited must be >= 0, got {self.waited}.")

        if self.last_run_end is not None:
            if isinstance(self.last_run_end, bool) or not isinstance(self.last_run_end, int):
                raise TypeError(
                    f"ProcView last_run_end must be int or None, "
                    f"got {type(self.last_run_end).__name__}."
                )
            if self.last_run_end < 0:
                raise ValueError(
                    f"ProcView last_run_end must be >= 0, got {self.last_run_end}."
                )


@dataclass(frozen=True)
class View:
    """
    1. The use of class:
       Represents the observable system state presented to a scheduling policy at each step.
    2. What the class contains:
       - time: Current integer tick timestamp.
       - ready: Tuple of ProcView instances currently awaiting CPU execution.
       - last_pid: Identifier of the process that ran in the preceding execution tick, or None.
       - newly_arrived: Tuple of PIDs admitted since the previous View was produced.
       - pending_arrivals: Non-negative integer count of unadmitted processes remaining.
       - __post_init__: Type and integrity validation for all view attributes.
    3. Technologies used and why:
       - Frozen dataclass with immutable nested tuples for leak-free policy querying.
       - Python 3.11 type hints with strict runtime type checks.
    4. Key design and overall design of the class:
       Provides a clean, immutable Markovian decision state for both classical policies and RL
       state feature extractors, decoupled from internal simulator mechanics.
    """

    time: int
    ready: tuple[ProcView, ...]
    last_pid: str | None
    newly_arrived: tuple[str, ...]
    pending_arrivals: int = 0

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates View attribute types and collection integrity upon instantiation.
        2. How does the specific function work:
           Checks time is a non-negative integer, ready is a tuple of ProcViews, last_pid is
           a non-empty string or None, newly_arrived is a tuple of non-empty strings, and
           pending_arrivals is a non-negative integer.
        3. Concepts used in the function:
           Dataclass post-init hook, collection integrity checking, defensive programming.
        4. Places where this function is imported and used:
           Invoked automatically during View instantiation.
        5. Final output of the function, before and after:
           Before: Unverified raw view attributes.
           After: Confirmed valid invariant state or raises TypeError / ValueError.
        """
        if isinstance(self.time, bool) or not isinstance(self.time, int):
            raise TypeError(f"View time must be an int, got {type(self.time).__name__}.")
        if self.time < 0:
            raise ValueError(f"View time must be >= 0, got {self.time}.")

        if not isinstance(self.ready, tuple):
            raise TypeError(f"View ready must be a tuple, got {type(self.ready).__name__}.")
        for item in self.ready:
            if not isinstance(item, ProcView):
                raise TypeError(
                    f"View ready item must be ProcView, got {type(item).__name__}."
                )

        if self.last_pid is not None:
            if not isinstance(self.last_pid, str):
                raise TypeError(
                    f"View last_pid must be str or None, got {type(self.last_pid).__name__}."
                )
            if not self.last_pid.strip():
                raise ValueError("View last_pid cannot be empty or whitespace only.")

        if not isinstance(self.newly_arrived, tuple):
            raise TypeError(
                f"View newly_arrived must be a tuple, got {type(self.newly_arrived).__name__}."
            )
        for pid in self.newly_arrived:
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError("View newly_arrived item must be a non-empty string.")

        if (
            isinstance(self.pending_arrivals, bool)
            or not isinstance(self.pending_arrivals, int)
            or self.pending_arrivals < 0
        ):
            raise ValueError(
                f"View pending_arrivals must be a non-negative int, "
                f"got {self.pending_arrivals}."
            )


@dataclass(frozen=True)
class StepInfo:
    """
    1. The use of class:
       Encapsulates the outcome of a single Engine step execution.
    2. What the class contains:
       - penalty: Non-negative integer cost representing ready process wait ticks in this step.
       - view: The resulting View snapshot after step execution.
       - done: Boolean flag indicating whether all processes in the workload have completed.
       - __post_init__: Integrity validator enforcing attribute types and non-negativity.
    3. Technologies used and why:
       - Frozen dataclass for immutability and compatibility with RL step interfaces (gym-like).
       - Python 3.11 type hints with strict runtime type and boundary assertions.
    4. Key design and overall design of the class:
       Mimics standard reinforcement learning transition tuples (reward = -penalty, next_obs = view,
       terminated = done) ensuring exact algebraic equivalence with total waiting time.
    """

    penalty: int
    view: View
    done: bool

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates StepInfo attributes upon creation.
        2. How does the specific function work:
           Verifies penalty is a non-negative int, view is a View instance, and done is a bool.
        3. Concepts used in the function:
           Dataclass post-init hook, invariant validation.
        4. Places where this function is imported and used:
           Invoked automatically during StepInfo instantiation.
        5. Final output of the function, before and after:
           Before: Unverified raw step outcome attributes.
           After: Confirmed valid invariant state or raises TypeError / ValueError.
        """
        if isinstance(self.penalty, bool) or not isinstance(self.penalty, int):
            raise TypeError(f"StepInfo penalty must be an int, got {type(self.penalty).__name__}.")
        if self.penalty < 0:
            raise ValueError(f"StepInfo penalty must be >= 0, got {self.penalty}.")
        if not isinstance(self.view, View):
            raise TypeError(f"StepInfo view must be View, got {type(self.view).__name__}.")
        if not isinstance(self.done, bool):
            raise TypeError(f"StepInfo done must be a bool, got {type(self.done).__name__}.")


@dataclass(frozen=True)
class SimResult:
    """
    1. The use of class:
       Encapsulates the complete final result of a scheduling simulation run.
    2. What the class contains:
       - process_metrics: List of ProcessMetrics for each process in the workload.
       - simulation_metrics: System-wide aggregate SimulationMetrics summary.
       - timeline: Timeline execution history tracker.
       - __post_init__: Type and integrity validation for all result attributes.
    3. Technologies used and why:
       - Frozen dataclass aggregating domain value objects for reporting, plotting, and analysis.
       - Python 3.11 type hints with strict runtime type checks.
    4. Key design and overall design of the class:
       Acts as the comprehensive summary contract returned by Engine.result() and run(),
       consumed by visualizers, statistical benchmarks, and API responses.
    """

    process_metrics: list[ProcessMetrics]
    simulation_metrics: SimulationMetrics
    timeline: Timeline

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates SimResult attribute types and collection integrity upon creation.
        2. How does the specific function work:
           Verifies process_metrics is a list of ProcessMetrics, simulation_metrics is
           SimulationMetrics, and timeline is Timeline.
        3. Concepts used in the function:
           Dataclass post-init hook, composite object validation.
        4. Places where this function is imported and used:
           Invoked automatically during SimResult instantiation.
        5. Final output of the function, before and after:
           Before: Unverified raw simulation result fields.
           After: Confirmed valid invariant state or raises TypeError.
        """
        if not isinstance(self.process_metrics, list):
            raise TypeError(
                f"SimResult process_metrics must be a list, "
                f"got {type(self.process_metrics).__name__}."
            )
        for idx, pm in enumerate(self.process_metrics):
            if not isinstance(pm, ProcessMetrics):
                raise TypeError(
                    f"SimResult process_metrics item at index {idx} must be ProcessMetrics, "
                    f"got {type(pm).__name__}."
                )
        if not isinstance(self.simulation_metrics, SimulationMetrics):
            raise TypeError(
                f"SimResult simulation_metrics must be SimulationMetrics, "
                f"got {type(self.simulation_metrics).__name__}."
            )
        if not isinstance(self.timeline, Timeline):
            raise TypeError(
                f"SimResult timeline must be Timeline, got {type(self.timeline).__name__}."
            )


class ScriptedPolicy:
    """
    1. The use of class:
       Provides a deterministic scripted policy driven by a sequence of PIDs or a custom callable.
    2. What the class contains:
       - _callable: Optional decision function accepting View and returning selected PID.
       - _sequence: Optional list of pre-scheduled PIDs.
       - _index: Current pointer position in the pre-scheduled sequence.
       - select: Selects the next PID to execute for the given simulation View.
       - __call__: Callable alias invoking select for flexible policy invocation.
    3. Technologies used and why:
       - Python standard typing (Sequence, Callable) for dual sequence/callable polymorphism.
       - Strict boundary validation ensuring deterministic testing of scheduling algorithms.
    4. Key design and overall design of the class:
       Decouples test orchestration from policy logic, enabling repeatable scripted test runs
       and edge-case reproduction.
    """

    def __init__(self, script: Sequence[str] | Callable[[View], str]) -> None:
        """
        1. The use of function:
           Initializes a ScriptedPolicy with a sequence of action PIDs or a decision callable.
        2. How does the specific function work:
           Checks if script is callable; if so, stores it. If script is a sequence of non-empty
           strings, copies it to an internal list. Otherwise raises TypeError or ValueError.
        3. Concepts used in the function:
           Polymorphic initialization, defensive input validation.
        4. Places where this function is imported and used:
           Instantiated in test suites and deterministic verification scripts.
        5. Final output of the function, before and after:
           Before: Uninitialized ScriptedPolicy reference.
           After: Initialized ScriptedPolicy ready for select() calls.
        """
        if callable(script):
            self._callable: Callable[[View], str] | None = script
            self._sequence: list[str] | None = None
            self._index: int = 0
        elif isinstance(script, Sequence) and not isinstance(script, (str, bytes)):
            for idx, item in enumerate(script):
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(
                        f"ScriptedPolicy script item at index {idx} must be a non-empty str."
                    )
            self._callable = None
            self._sequence = list(script)
            self._index = 0
        else:
            raise TypeError(
                f"script must be a Sequence[str] or Callable[[View], str], "
                f"got {type(script).__name__}."
            )

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Returns the next scheduled PID decision for the provided simulation View.
        2. How does the specific function work:
           Validates view type. If callable is configured, executes callable(view). Otherwise
           retrieves next PID from pre-scheduled list and advances index. Raises IndexError
           if sequence is exhausted.
        3. Concepts used in the function:
           Script replay, boundary enforcement, exhausted-sequence handling.
        4. Places where this function is imported and used:
           Called by Engine simulation loops and testing harnesses.
        5. Final output of the function, before and after:
           Before: Internal sequence index points to current decision.
           After: Returns selected PID string and advances internal index by 1.
        """
        if not isinstance(view, View):
            raise TypeError(f"view must be a View instance, got {type(view).__name__}.")
        if self._callable is not None:
            return self._callable(view)
        assert self._sequence is not None
        if self._index >= len(self._sequence):
            raise IndexError(
                f"ScriptedPolicy script exhausted: sequence of {len(self._sequence)} "
                f"actions has been depleted."
            )
        pid = self._sequence[self._index]
        self._index += 1
        return pid

    def __call__(self, view: View) -> str:
        """
        1. The use of function:
           Callable protocol alias delegating directly to select(view).
        2. How does the specific function work:
           Forwards the provided View directly to self.select(view).
        3. Concepts used in the function:
           Python __call__ magic method, method forwarding.
        4. Places where this function is imported and used:
           Allows ScriptedPolicy to be invoked as a standard function.
        5. Final output of the function, before and after:
           Before: View passed to policy instance call.
           After: Returns PID string from self.select(view).
        """
        return self.select(view)


class Engine:
    """
    1. The use of class:
       Central discrete unit-step CPU scheduling simulation engine supporting context switches,
       idle time jumps, AI model inference stalls, and state cloning.
    2. What the class contains:
       - _processes: Tuple of Process domain objects defining the workload.
       - _proc_map: Dictionary mapping PID strings to Process instances.
       - _cs_cost: Non-negative integer tick cost for context switches.
       - _tick_us: Microseconds of real time represented by one simulation tick, or None.
       - _time: Current simulation integer tick timestamp.
       - _last_pid: PID of the process dispatched in the previous execution tick, or None.
       - _timeline: Timeline recorder capturing process, context switch, AI stall, and idle ticks.
       - _context_switches: Integer counter tracking total context switch occurrences.
       - _remaining: Dictionary tracking remaining burst ticks per process.
       - _waited: Dictionary tracking accumulated ready-queue wait ticks per process.
       - _first_start: Dictionary tracking tick timestamp of first CPU dispatch per process.
       - _completion: Dictionary tracking tick timestamp of completion per process.
       - _last_run_end: Dictionary tracking tick timestamp of most recent CPU run per process.
       - _admitted: Set of PIDs that have been admitted into the simulation.
       - _ready_set: Set of PIDs currently ready and awaiting CPU execution.
       - _completed: Set of PIDs that have completed execution.
       - _current_view: Current View instance representing available ready tasks.
       - _ai_debt: Accumulated fractional AI inference stall tick debt.
       - _ai_overhead_ticks: Total integer AI inference stall ticks charged.
       - _decisions: Total count of non-forced scheduling decisions made.
       - _inference_us_total: Total raw model inference latency accumulated in microseconds.
       - _start_cost_us: One-time upfront model cost in microseconds awaiting first dispatch.
       - reset: Resets engine state, executes initial idle jump if needed, and returns initial View.
       - charge_start_overhead: Records one-time upfront model inference latency for the workload.
       - step: Executes one unit step of simulation for the chosen PID and returns StepInfo.
       - clone: Creates an independent isolated deep copy of the engine state.
       - restore: Restores engine execution state from a checkpoint.
       - done: Property indicating whether all processes have completed execution.
       - time: Property returning the current simulation time.
       - timeline: Property returning the Timeline recorder instance.
       - context_switches: Property returning total context switches dispatched.
       - current_view: Property returning current observable View snapshot.
       - has_pending_arrivals: Property indicating whether any unadmitted processes remain.
       - ai_overhead_ticks: Property returning total AI stall ticks charged.
       - decisions: Property returning non-forced decisions count.
       - inference_us_total: Property returning cumulative raw inference latency in microseconds.
       - tick_us: Property returning microseconds per simulation tick or None.
       - result: Packages process metrics, aggregate simulation metrics, and execution timeline.
    3. Technologies used and why:
       - Discrete-event unit-step simulation modeling interval [t, t+1).
       - Fractional debt accumulator converting microsecond latencies to integer stall ticks.
       - Conservation of time and waiting penalties matching reinforcement learning rewards.
    4. Key design and overall design of the class:
       Implements a strict, reproducible simulation contract across classical, heuristic,
       and reinforcement learning policies with exact overhead accounting and idle jumping.
    """

    def __init__(
        self,
        processes: Sequence[Process],
        cs_cost: int = 0,
        tick_us: float | None = None,
    ) -> None:
        """
        1. The use of function:
           Initializes a new Engine with workload processes, context switch cost, and tick scale.
        2. How does the specific function work:
           Validates processes and cs_cost >= 0, validates tick_us > 0 if provided, initializes
           internal tracking structures, and triggers reset() to establish the initial state.
        3. Concepts used in the function:
           Defensive parameter validation, immutable collection copying, initial state setup.
        4. Places where this function is imported and used:
           Instantiated by run(), benchmark scripts, test suites, and RL environments.
        5. Final output of the function, before and after:
           Before: Uninitialized Engine reference.
           After: Fully initialized Engine reset to initial tick with ready queue populated.
        """
        validate_processes(processes)
        if isinstance(cs_cost, bool) or not isinstance(cs_cost, int):
            raise TypeError(f"cs_cost must be an int, got {type(cs_cost).__name__}.")
        if cs_cost < 0:
            raise ValueError(f"cs_cost must be >= 0, got {cs_cost}.")

        if tick_us is not None:
            if isinstance(tick_us, bool) or not isinstance(tick_us, (int, float)):
                raise TypeError(
                    f"tick_us must be a float, int or None, got {type(tick_us).__name__}."
                )
            if tick_us <= 0:
                raise ValueError(f"tick_us must be strictly positive, got {tick_us}.")
            self._tick_us: float | None = float(tick_us)
        else:
            self._tick_us = None

        self._processes: tuple[Process, ...] = tuple(processes)
        self._proc_map: dict[str, Process] = {p.pid: p for p in self._processes}
        self._cs_cost: int = cs_cost

        self._time: int = 0
        self._last_pid: str | None = None
        self._timeline: Timeline = Timeline()
        self._context_switches: int = 0

        self._remaining: dict[str, int] = {}
        self._waited: dict[str, int] = {}
        self._first_start: dict[str, int | None] = {}
        self._completion: dict[str, int | None] = {}
        self._last_run_end: dict[str, int | None] = {}

        self._admitted: set[str] = set()
        self._ready_set: set[str] = set()
        self._completed: set[str] = set()
        self._current_view: View | None = None

        self._ai_debt: float = 0.0
        self._ai_overhead_ticks: int = 0
        self._decisions: int = 0
        self._inference_us_total: float = 0.0
        self._start_cost_us: float = 0.0

        self.reset()

    def _admit_new_arrivals(self) -> list[str]:
        """
        1. The use of function:
           Admits unadmitted processes whose arrival timestamp is <= current simulation time.
        2. How does the specific function work:
           Iterates through all workload processes. For any unadmitted process with arrival <= time,
           adds PID to admitted and ready sets, and records PID in admitted list.
        3. Concepts used in the function:
           Process admission, arrival queueing, deterministic ordering.
        4. Places where this function is imported and used:
           Internal helper invoked by reset() and step().
        5. Final output of the function, before and after:
           Before: Eligible processes unadmitted.
           After: Eligible processes added to ready set; returns list of admitted PIDs.
        """
        admitted_pids: list[str] = []
        for p in self._processes:
            if p.pid not in self._admitted and p.arrival <= self._time:
                self._admitted.add(p.pid)
                self._ready_set.add(p.pid)
                admitted_pids.append(p.pid)
        return admitted_pids

    def _build_view(self, newly_arrived: Sequence[str]) -> View:
        """
        1. The use of function:
           Constructs an immutable View snapshot representing the current simulation state.
        2. How does the specific function work:
           Sorts ready PIDs deterministically by (arrival, pid), instantiates ProcView records,
           sorts newly_arrived PIDs by (arrival, pid), and returns a frozen View instance.
        3. Concepts used in the function:
           State snapshotting, canonical ordering, immutable value object projection.
        4. Places where this function is imported and used:
           Internal helper invoked by reset() and step().
        5. Final output of the function, before and after:
           Before: Internal engine state at current tick.
           After: Returns immutable View instance reflecting current state.
        """
        sorted_ready_pids = sorted(
            self._ready_set,
            key=lambda pid: (self._proc_map[pid].arrival, pid),
        )
        ready_views = tuple(
            ProcView(
                pid=pid,
                arrival=self._proc_map[pid].arrival,
                burst=self._proc_map[pid].burst,
                remaining=self._remaining[pid],
                waited=self._waited[pid],
                last_run_end=self._last_run_end[pid],
            )
            for pid in sorted_ready_pids
        )
        sorted_newly_arrived = tuple(
            sorted(
                newly_arrived,
                key=lambda pid: (self._proc_map[pid].arrival, pid),
            )
        )
        pending_arrivals = len(self._processes) - len(self._admitted)
        return View(
            time=self._time,
            ready=ready_views,
            last_pid=self._last_pid,
            newly_arrived=sorted_newly_arrived,
            pending_arrivals=pending_arrivals,
        )

    def reset(self) -> View:
        """
        1. The use of function:
           Resets the Engine state to the initial tick and returns the initial View.
        2. How does the specific function work:
           Clears counters, resets timeline, reinitializes remaining bursts and wait times.
           Admits arrivals at tick 0. If no process arrived at tick 0, executes an idle jump
           to the earliest arrival tick, records IDLE interval, admits arrivals, and returns View.
        3. Concepts used in the function:
           Simulation reset, initial idle jump, ready queue population.
        4. Places where this function is imported and used:
           Called in __init__, by run(), in test setups, and at start of RL episodes.
        5. Final output of the function, before and after:
           Before: Engine in uninitialized or post-run state.
           After: Engine reset to initial tick with ready queue populated; returns initial View.
        """
        self._time = 0
        self._last_pid = None
        self._timeline = Timeline()
        self._context_switches = 0

        self._remaining = {p.pid: p.burst for p in self._processes}
        self._waited = {p.pid: 0 for p in self._processes}
        self._first_start = {p.pid: None for p in self._processes}
        self._completion = {p.pid: None for p in self._processes}
        self._last_run_end = {p.pid: None for p in self._processes}

        self._admitted = set()
        self._ready_set = set()
        self._completed = set()

        self._ai_debt = 0.0
        self._ai_overhead_ticks = 0
        self._decisions = 0
        self._inference_us_total = 0.0
        self._start_cost_us = 0.0

        admitted = self._admit_new_arrivals()

        if not self._ready_set:
            earliest_arrival = min(p.arrival for p in self._processes)
            self._timeline.record_segment(0, earliest_arrival, IDLE_LABEL)
            self._time = earliest_arrival
            admitted = self._admit_new_arrivals()

        self._current_view = self._build_view(newly_arrived=admitted)
        return self._current_view

    def charge_start_overhead(self, cost_us: float) -> None:
        """
        1. The use of function:
           Records a one-time upfront inference/feature-extraction overhead cost for the workload.
        2. How does the specific function work:
           Validates cost_us >= 0.0 and accumulates it into _start_cost_us to be converted to
           AI stall ticks at the first dispatch.
        3. Concepts used in the function:
           One-time workload-level model inference charging, Category 1 ML model accounting.
        4. Places where this function is imported and used:
           Invoked by run() or external benchmark harnesses prior to stepping the engine.
        5. Final output of the function, before and after:
           Before: _start_cost_us has previous accumulated start cost.
           After: _start_cost_us increased by float(cost_us).
        """
        if isinstance(cost_us, bool) or not isinstance(cost_us, (int, float)):
            raise TypeError(f"cost_us must be numeric, got {type(cost_us).__name__}.")
        if cost_us < 0:
            raise ValueError(f"cost_us cannot be negative, got {cost_us}.")
        self._start_cost_us += float(cost_us)

    def step(self, pid: str, decision_cost_us: float = 0.0) -> StepInfo:
        """
        1. The use of function:
           Executes a single discrete unit step of simulation for the chosen PID, accounting for
           AI inference overhead, context switch costs, and execution progress.
        2. How does the specific function work:
           Validates inputs. Identifies non-forced decisions (ready_set > 1) to increment
           decisions counter and accumulate raw inference latency. Adds any one-time start cost
           on first dispatch. Converts chargeable microseconds into discrete AI stall ticks via
           the fractional debt accumulator. Records any context switches. Executes 1 CPU tick for
           pid, updates completion, admits arrivals, handles idle jumping, and returns StepInfo.
        3. Concepts used in the function:
           Unit-step execution, non-forced decision identification, fractional debt accumulation,
           AI stall overhead, context switch penalty, idle jumping, reward preservation.
        4. Places where this function is imported and used:
           Called on every simulation step by run(), RL environments, and tests.
        5. Final output of the function, before and after:
           Before: Engine at current tick awaiting dispatch decision.
           After: Advances engine time, updates timeline and process states; returns StepInfo.
        """
        if self.done:
            raise RuntimeError("Simulation is already complete; cannot call step().")
        if not isinstance(pid, str):
            raise TypeError(f"pid must be a str, got {type(pid).__name__}.")
        if pid not in self._ready_set:
            raise ValueError(
                f"Process '{pid}' is not in ready queue. "
                f"Available ready PIDs: {sorted(self._ready_set)}."
            )
        if isinstance(decision_cost_us, bool) or not isinstance(
            decision_cost_us, (int, float)
        ):
            raise TypeError(
                f"decision_cost_us must be numeric, got {type(decision_cost_us).__name__}."
            )
        if decision_cost_us < 0:
            raise ValueError(
                f"decision_cost_us cannot be negative, got {decision_cost_us}."
            )

        step_penalty = 0
        step_newly_arrived: list[str] = []

        is_non_forced = len(self._ready_set) > 1
        if is_non_forced:
            self._decisions += 1
            self._inference_us_total += float(decision_cost_us)
            cost_to_charge_us = float(decision_cost_us)
        else:
            cost_to_charge_us = 0.0

        if self._start_cost_us > 0.0:
            self._inference_us_total += self._start_cost_us
            cost_to_charge_us += self._start_cost_us
            self._start_cost_us = 0.0

        if self._tick_us is not None and self._tick_us > 0.0 and cost_to_charge_us > 0.0:
            self._ai_debt += cost_to_charge_us / self._tick_us
            stall_ticks = int(self._ai_debt + 1e-9)
            if stall_ticks > 0:
                self._ai_debt -= stall_ticks
                self._ai_overhead_ticks += stall_ticks
                remaining_stalls = stall_ticks
                while remaining_stalls > 0:
                    if self.has_pending_arrivals:
                        next_arr = min(
                            p.arrival for p in self._processes if p.pid not in self._admitted
                        )
                        chunk = min(remaining_stalls, max(1, next_arr - self._time))
                    else:
                        chunk = remaining_stalls

                    self._timeline.record_segment(self._time, self._time + chunk, AI_LABEL)
                    for r_pid in self._ready_set:
                        self._waited[r_pid] += chunk
                    step_penalty += chunk * len(self._ready_set)
                    self._time += chunk
                    remaining_stalls -= chunk
                    admitted_now = self._admit_new_arrivals()
                    step_newly_arrived.extend(admitted_now)

        if self._last_pid is not None and pid != self._last_pid:
            self._context_switches += 1
            if self._cs_cost > 0:
                for _ in range(self._cs_cost):
                    self._timeline.record_tick(self._time, CS_LABEL)
                    for r_pid in self._ready_set:
                        self._waited[r_pid] += 1
                    step_penalty += len(self._ready_set)
                    self._time += 1
                    admitted_now = self._admit_new_arrivals()
                    step_newly_arrived.extend(admitted_now)

        if self._first_start[pid] is None:
            self._first_start[pid] = self._time

        self._timeline.record_tick(self._time, pid)
        self._remaining[pid] -= 1
        self._last_run_end[pid] = self._time + 1

        for r_pid in self._ready_set:
            if r_pid != pid:
                self._waited[r_pid] += 1
        step_penalty += len(self._ready_set) - 1

        self._time += 1

        if self._remaining[pid] == 0:
            self._completion[pid] = self._time
            self._ready_set.remove(pid)
            self._completed.add(pid)

        admitted_now = self._admit_new_arrivals()
        step_newly_arrived.extend(admitted_now)

        if not self.done and not self._ready_set:
            next_arrival = min(
                p.arrival for p in self._processes if p.pid not in self._admitted
            )
            self._timeline.record_segment(self._time, next_arrival, IDLE_LABEL)
            self._time = next_arrival
            admitted_now = self._admit_new_arrivals()
            step_newly_arrived.extend(admitted_now)

        self._last_pid = pid

        new_view = self._build_view(newly_arrived=step_newly_arrived)
        self._current_view = new_view
        return StepInfo(penalty=step_penalty, view=new_view, done=self.done)

    @property
    def done(self) -> bool:
        """
        1. The use of function:
           Indicates whether all workload processes have completed execution.
        2. How does the specific function work:
           Compares the count of completed processes against the total workload count.
        3. Concepts used in the function:
           Termination checking, completion invariant, property decorator.
        4. Places where this function is imported and used:
           Used by simulation loops, RL termination checks, and test assertions.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns True if all processes finished, False otherwise.
        """
        return len(self._completed) == len(self._processes)

    @property
    def time(self) -> int:
        """
        1. The use of function:
           Returns the current simulation tick timestamp.
        2. How does the specific function work:
           Returns integer value of internal _time attribute.
        3. Concepts used in the function:
           State inspection, discrete time property.
        4. Places where this function is imported and used:
           Used in debugging, test assertions, and audit logging.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns current integer tick.
        """
        return self._time

    @property
    def timeline(self) -> Timeline:
        """
        1. The use of function:
           Returns the Timeline instance recording execution history.
        2. How does the specific function work:
           Returns internal _timeline reference.
        3. Concepts used in the function:
           Execution logging, encapsulation.
        4. Places where this function is imported and used:
           Used by result visualizers, partition verifiers, and unit tests.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns Timeline recorder instance.
        """
        return self._timeline

    @property
    def context_switches(self) -> int:
        """
        1. The use of function:
           Returns the total number of context switches dispatched during simulation.
        2. How does the specific function work:
           Returns integer value of internal _context_switches counter.
        3. Concepts used in the function:
           Context switch counter, overhead tracking.
        4. Places where this function is imported and used:
           Used in metrics derivation and test assertions.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns non-negative integer count of context switches.
        """
        return self._context_switches

    @property
    def current_view(self) -> View | None:
        """
        1. The use of function:
           Returns the most recent observable View snapshot of the simulation.
        2. How does the specific function work:
           Returns internal _current_view reference representing ready tasks and state.
        3. Concepts used in the function:
           Read-only state exposure, property decorator, encapsulation.
        4. Places where this function is imported and used:
           Used by RL environments, test verifiers, and decision policies.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns View instance or None if not initialized.
        """
        return self._current_view

    @property
    def has_pending_arrivals(self) -> bool:
        """
        1. The use of function:
           Indicates whether any workload processes remain unadmitted in the future.
        2. How does the specific function work:
           Compares count of admitted processes against total count of workload processes.
        3. Concepts used in the function:
           Arrival tracking, admission completion check, property decorator.
        4. Places where this function is imported and used:
           Used by RL state encoders and environment wrappers to flag pending arrivals.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns True if unadmitted processes remain, False otherwise.
        """
        return len(self._admitted) < len(self._processes)

    @property
    def ai_overhead_ticks(self) -> int:
        """
        1. The use of function:
           Returns the total number of AI stall ticks incurred by model inference.
        2. How does the specific function work:
           Returns the integer counter _ai_overhead_ticks.
        3. Concepts used in the function:
           AI overhead tracking, property accessor.
        4. Places where this function is imported and used:
           Used by simulation metric calculators, audit logging, and unit tests.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns integer count of AI stall ticks.
        """
        return self._ai_overhead_ticks

    @property
    def decisions(self) -> int:
        """
        1. The use of function:
           Returns the total number of non-forced scheduling decisions made during simulation.
        2. How does the specific function work:
           Returns the integer counter _decisions.
        3. Concepts used in the function:
           Decision counting, non-forced decision identification, property accessor.
        4. Places where this function is imported and used:
           Used by metrics calculations, test verifications, and comparison tables.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns integer count of non-forced decisions.
        """
        return self._decisions

    @property
    def inference_us_total(self) -> float:
        """
        1. The use of function:
           Returns the cumulative raw AI model inference latency in microseconds.
        2. How does the specific function work:
           Returns the float counter _inference_us_total.
        3. Concepts used in the function:
           Microsecond latency accounting, property accessor.
        4. Places where this function is imported and used:
           Used by metrics calculations, latency profiles, and cost model evaluation.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns total raw inference microseconds as a float.
        """
        return self._inference_us_total

    @property
    def tick_us(self) -> float | None:
        """
        1. The use of function:
           Returns the simulation tick time constant in microseconds per tick.
        2. How does the specific function work:
           Returns the configured _tick_us float or None if unconstrained.
        3. Concepts used in the function:
           Time scale conversion factor, property accessor.
        4. Places where this function is imported and used:
           Used by latency-to-tick conversion, test verifications, and experiment runners.
        5. Final output of the function, before and after:
           Before: Property accessed on Engine instance.
           After: Returns configured tick_us float or None.
        """
        return self._tick_us

    def clone(self) -> "Engine":
        """
        1. The use of function:
           Creates an independent deep copy of the engine's mutable execution state.
        2. How does the specific function work:
           Allocates a new Engine instance without running __init__, copies immutable process
           references and settings, duplicates all mutable collections (remaining, waited,
           completion, timeline, ready/completed sets), and returns the isolated clone.
        3. Concepts used in the function:
           Object cloning, state isolation, defensive copying, MCTS/rollout branch exploration.
        4. Places where this function is imported and used:
           Used by rollout oracles (Stage 13), lookahead search heuristics, and state checkpoints.
        5. Final output of the function, before and after:
           Before: Original Engine instance at current simulation state.
           After: Returns a new Engine instance with identical state; stepping either instance
           does not affect the other.
        """
        cloned = Engine.__new__(Engine)
        cloned._processes = self._processes
        cloned._proc_map = self._proc_map
        cloned._cs_cost = self._cs_cost
        cloned._tick_us = self._tick_us

        cloned._time = self._time
        cloned._last_pid = self._last_pid
        cloned._timeline = Timeline(segments=self._timeline.raw_segments)
        cloned._context_switches = self._context_switches

        cloned._remaining = self._remaining.copy()
        cloned._waited = self._waited.copy()
        cloned._first_start = self._first_start.copy()
        cloned._completion = self._completion.copy()
        cloned._last_run_end = self._last_run_end.copy()

        cloned._admitted = self._admitted.copy()
        cloned._ready_set = self._ready_set.copy()
        cloned._completed = self._completed.copy()
        cloned._current_view = self._current_view

        cloned._ai_debt = self._ai_debt
        cloned._ai_overhead_ticks = self._ai_overhead_ticks
        cloned._decisions = self._decisions
        cloned._inference_us_total = self._inference_us_total
        cloned._start_cost_us = self._start_cost_us

        return cloned

    def restore(self, checkpoint: "Engine") -> None:
        """
        1. The use of function:
           Restores the engine's execution state from a previously cloned Engine checkpoint.
        2. How does the specific function work:
           Validates checkpoint type. Overwrites mutable execution state, collections, timeline,
           and overhead counters with fresh copies from the checkpoint instance.
        3. Concepts used in the function:
           State restoration, checkpoint/rollback pattern, rollout tree search.
        4. Places where this function is imported and used:
           Used by rollout oracle branches (Stage 13) to reset engine state without re-allocating.
        5. Final output of the function, before and after:
           Before: Engine at current or diverged execution state.
           After: Engine state matches the checkpoint exactly.
        """
        if not isinstance(checkpoint, Engine):
            raise TypeError(
                f"checkpoint must be an Engine instance, got {type(checkpoint).__name__}."
            )
        self._processes = checkpoint._processes
        self._proc_map = checkpoint._proc_map
        self._cs_cost = checkpoint._cs_cost
        self._tick_us = checkpoint._tick_us

        self._time = checkpoint._time
        self._last_pid = checkpoint._last_pid
        self._timeline = Timeline(segments=checkpoint._timeline.raw_segments)
        self._context_switches = checkpoint._context_switches

        self._remaining = checkpoint._remaining.copy()
        self._waited = checkpoint._waited.copy()
        self._first_start = checkpoint._first_start.copy()
        self._completion = checkpoint._completion.copy()
        self._last_run_end = checkpoint._last_run_end.copy()

        self._admitted = checkpoint._admitted.copy()
        self._ready_set = checkpoint._ready_set.copy()
        self._completed = checkpoint._completed.copy()
        self._current_view = checkpoint._current_view

        self._ai_debt = checkpoint._ai_debt
        self._ai_overhead_ticks = checkpoint._ai_overhead_ticks
        self._decisions = checkpoint._decisions
        self._inference_us_total = checkpoint._inference_us_total
        self._start_cost_us = checkpoint._start_cost_us

    def result(self) -> SimResult:
        """
        1. The use of function:
           Compiles and returns the comprehensive simulation result (SimResult).
        2. How does the specific function work:
           Verifies simulation is done. Computes ProcessMetrics for each workload process in
           original input order. Derives aggregate SimulationMetrics from process metrics and
           timeline counters, passing AI overhead and effective metrics. Packages everything
           into an immutable SimResult.
        3. Concepts used in the function:
           Result compilation, metric derivation, immutable reporting.
        4. Places where this function is imported and used:
           Called at simulation completion by run(), evaluation scripts, and API endpoints.
        5. Final output of the function, before and after:
           Before: Completed Engine simulation state.
           After: Returns SimResult containing process metrics, simulation metrics, and timeline.
        """
        if not self.done:
            raise RuntimeError("Cannot compute SimResult: simulation is not yet complete.")

        proc_metrics = [
            calculate_process_metrics(
                process=p,
                completion=self._completion[p.pid],  # type: ignore[arg-type]
                first_start=self._first_start[p.pid],  # type: ignore[arg-type]
            )
            for p in self._processes
        ]

        avg_turnaround = (
            sum(m.turnaround for m in proc_metrics) / len(proc_metrics)
            if proc_metrics
            else 0.0
        )

        sim_metrics = calculate_simulation_metrics(
            proc_metrics=proc_metrics,
            makespan=self._timeline.makespan,
            busy_ticks=self._timeline.busy_ticks,
            overhead_ticks=self._timeline.overhead_ticks,
            context_switches=self._context_switches,
            ai_overhead_ticks=self._ai_overhead_ticks,
            decisions=self._decisions,
            inference_us_total=self._inference_us_total,
            effective_makespan=self._timeline.makespan,
            effective_turnaround=avg_turnaround,
        )

        return SimResult(
            process_metrics=proc_metrics,
            simulation_metrics=sim_metrics,
            timeline=self._timeline,
        )


def run(
    policy: Any,
    processes: Sequence[Process],
    cs_cost: int = 0,
    tick_us: float | None = None,
) -> SimResult:
    """
    1. The use of function:
       Runs a complete CPU scheduling simulation using the provided policy and processes,
       incorporating optional AI model latency overhead accounting.
    2. How does the specific function work:
       Instantiates Engine with tick_us, resets to obtain initial View. Calls on_start() if
       supported by policy to charge one-time upfront costs. Loops while not engine.done:
       queries policy for next PID, queries decision_cost_us(view) if supported, steps engine,
       and updates view. Finally returns compiled SimResult.
    3. Concepts used in the function:
       Simulation runner, policy dispatch loop, AI latency hooks, deterministic execution.
    4. Places where this function is imported and used:
       Imported by benchmark scripts, test suites, ML/RL evaluation pipelines, and REST API.
    5. Final output of the function, before and after:
       Before: Workload process definitions, policy specification, and optional tick_us.
       After: Complete SimResult instance containing process metrics, aggregate metrics, timeline.
    """
    engine = Engine(processes=processes, cs_cost=cs_cost, tick_us=tick_us)
    view = engine.reset()

    if hasattr(policy, "bind_workload") and callable(policy.bind_workload):
        policy.bind_workload(processes)
    elif hasattr(policy, "reset") and callable(policy.reset):
        policy.reset()

    if hasattr(policy, "on_start") and callable(policy.on_start):
        start_cost = float(policy.on_start(processes))
        if start_cost > 0.0:
            engine.charge_start_overhead(start_cost)

    while not engine.done:
        if hasattr(policy, "select") and callable(policy.select):
            pid = policy.select(view)
        elif callable(policy):
            pid = policy(view)
        else:
            raise TypeError(
                f"Policy must provide a callable .select(view) method or be callable, "
                f"got {type(policy).__name__}."
            )

        cost_us = 0.0
        if hasattr(policy, "decision_cost_us") and callable(policy.decision_cost_us):
            cost_us = float(policy.decision_cost_us(view))

        step_info = engine.step(pid, decision_cost_us=cost_us)
        view = step_info.view

    return engine.result()
