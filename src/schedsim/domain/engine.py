"""
1. What this file does:
   Implements the core discrete unit-step CPU scheduling simulation engine, state views,
   step execution logic, scripted testing policy, and execution runner.
2. Overall design:
   Provides a deterministic discrete-event simulation engine advancing integer tick intervals
   [t, t+1). Features exact context switch overhead accounting, idle time jumping, ready queue
   admission tracking, process lifecycle recording, and step penalty calculation matching
   reinforcement learning reward semantics.
3. Elements in the file:
   - ProcView: Frozen dataclass representing an immutable process snapshot for policies.
   - View: Frozen dataclass representing the observable simulation state at a discrete tick.
   - StepInfo: Frozen dataclass returning penalty, resulting view, and completion flag per step.
   - SimResult: Frozen dataclass encapsulating process metrics, simulation metrics, and timeline.
   - ScriptedPolicy: Policy driven by a predetermined sequence of PIDs or a custom callable.
   - Engine: Central stateful simulation engine driving unit-step scheduling execution.
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
from schedsim.domain.timeline import CS_LABEL, IDLE_LABEL, Timeline


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

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates View attribute types and collection integrity upon instantiation.
        2. How does the specific function work:
           Checks time is a non-negative integer, ready is a tuple of ProcViews, last_pid is
           a non-empty string or None, and newly_arrived is a tuple of non-empty strings.
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
       Central discrete unit-step CPU scheduling simulation engine.
    2. What the class contains:
       - _processes: Tuple of Process domain objects defining the workload.
       - _proc_map: Dictionary mapping PID strings to Process instances.
       - _cs_cost: Non-negative integer tick cost for context switches.
       - _time: Current simulation integer tick timestamp.
       - _last_pid: PID of the process dispatched in the previous execution tick, or None.
       - _timeline: Timeline recorder capturing process, context switch, and idle intervals.
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
       - reset: Resets engine state, executes initial idle jump if needed, and returns initial View.
       - step: Executes one unit step of simulation for the chosen PID and returns StepInfo.
       - done: Property indicating whether all processes have completed execution.
       - time: Property returning the current simulation time.
       - timeline: Property returning the Timeline recorder instance.
       - context_switches: Property returning total context switches dispatched.
       - current_view: Property returning current observable View snapshot.
       - has_pending_arrivals: Property indicating whether any unadmitted processes remain.
       - result: Packages process metrics, aggregate simulation metrics, and execution timeline.
    3. Technologies used and why:
       - Discrete-event unit-step simulation modeling interval [t, t+1).
       - Conservation of time and waiting penalties matching reinforcement learning rewards.
    4. Key design and overall design of the class:
       Implements a strict, reproducible simulation contract across classical, heuristic,
       and reinforcement learning policies with exact overhead accounting and idle jumping.
    """

    def __init__(self, processes: Sequence[Process], cs_cost: int = 0) -> None:
        """
        1. The use of function:
           Initializes a new Engine with workload processes and context switch cost.
        2. How does the specific function work:
           Validates processes via validate_processes, validates cs_cost >= 0, initializes internal
           tracking structures, and triggers reset() to establish the initial state.
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
        return View(
            time=self._time,
            ready=ready_views,
            last_pid=self._last_pid,
            newly_arrived=sorted_newly_arrived,
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

        admitted = self._admit_new_arrivals()

        if not self._ready_set:
            earliest_arrival = min(p.arrival for p in self._processes)
            self._timeline.record_segment(0, earliest_arrival, IDLE_LABEL)
            self._time = earliest_arrival
            admitted = self._admit_new_arrivals()

        self._current_view = self._build_view(newly_arrived=admitted)
        return self._current_view

    def step(self, pid: str) -> StepInfo:
        """
        1. The use of function:
           Executes a single discrete unit step of simulation for the chosen PID.
        2. How does the specific function work:
           Validates engine is not done and pid is in ready set. If switching from a different
           last_pid with cs_cost > 0, records cs_cost 'CS' ticks, admitting arrivals and charging
           wait penalties. Runs 1 execution tick for pid, updates remaining burst, wait times,
           and timeline. Handles completion, admits new arrivals, performs idle jump if ready queue
           is empty, updates last_pid, and returns StepInfo.
        3. Concepts used in the function:
           Unit-step execution, context switch overhead, idle jumping, penalty accounting.
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

        step_penalty = 0
        step_newly_arrived: list[str] = []

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

    def result(self) -> SimResult:
        """
        1. The use of function:
           Compiles and returns the comprehensive simulation result (SimResult).
        2. How does the specific function work:
           Verifies simulation is done. Computes ProcessMetrics for each workload process in
           original input order. Derives aggregate SimulationMetrics from process metrics and
           timeline counters. Packages everything into an immutable SimResult.
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

        sim_metrics = calculate_simulation_metrics(
            proc_metrics=proc_metrics,
            makespan=self._timeline.makespan,
            busy_ticks=self._timeline.busy_ticks,
            overhead_ticks=self._timeline.overhead_ticks,
            context_switches=self._context_switches,
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
) -> SimResult:
    """
    1. The use of function:
       Runs a complete CPU scheduling simulation using the provided policy and processes.
    2. How does the specific function work:
       Instantiates Engine, resets to obtain initial View, and loops while not engine.done.
       In each iteration, requests PID selection from policy (via .select(view) or __call__(view)),
       steps the engine, and updates view. Finally returns engine.result().
    3. Concepts used in the function:
       Simulation runner, policy dispatch loop, deterministic execution.
    4. Places where this function is imported and used:
       Imported by benchmark scripts, test suites, ML/RL evaluation pipelines, and REST API.
    5. Final output of the function, before and after:
       Before: Workload process definitions and scheduling policy specification.
       After: Complete SimResult instance containing process metrics, aggregate metrics, timeline.
    """
    engine = Engine(processes=processes, cs_cost=cs_cost)
    view = engine.reset()

    if hasattr(policy, "bind_workload") and callable(policy.bind_workload):
        policy.bind_workload(processes)
    elif hasattr(policy, "reset") and callable(policy.reset):
        policy.reset()

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
        step_info = engine.step(pid)
        view = step_info.view

    return engine.result()
