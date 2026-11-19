"""
1. What this file does:
   Implements the Round Robin (RR) classical CPU scheduling algorithm with configurable quantum.
2. Overall design:
   Inherits from the Policy base class. Maintains a FIFO deque of ready process IDs, tracking the
   currently executing process and consumed execution quantum ticks. Newly arrived processes are
   enqueued in arrival order. When a running process exhausts its quantum, it yields the CPU and
   is re-enqueued after new arrivals, ensuring strictly fair cyclic time-sharing.
3. Elements in the file:
   - RoundRobinPolicy: Stateful scheduling policy implementing Round Robin with time quantum.
4. Final output of the file:
   Exports RoundRobinPolicy for simulation runs, interactive comparisons, and benchmarking.
"""

from collections import deque

from schedsim.domain.engine import View
from schedsim.policies.base import Policy


class RoundRobinPolicy(Policy):
    """
    1. The use of class:
       Executes Round Robin (RR) time-sliced CPU scheduling with configurable quantum.
    2. What the class contains:
       - quantum: Configurable execution slice duration in discrete ticks (default 2, >= 1).
       - _deque: FIFO queue maintaining ready process IDs awaiting execution.
       - _current_pid: Identifier of the process currently occupying the CPU, or None.
       - _quantum_ticks_consumed: Execution ticks consumed by _current_pid in current quantum.
       - __init__: Initializes policy with quantum and optional custom name.
       - quantum (property): Read-only accessor for the configured time quantum.
       - current_pid (property): Read-only accessor for currently running process ID.
       - quantum_ticks_consumed (property): Read-only accessor for consumed quantum ticks.
       - queue (property): Read-only tuple snapshot of the internal FIFO queue.
       - reset: Resets FIFO deque and quantum execution counters.
       - select: Selects the next process PID to execute based on FIFO ordering and quantum slices.
    3. Technologies used and why:
       - collections.deque for O(1) FIFO push and popleft operations.
       - Subclasses Policy ABC for seamless interoperability with Engine and simulation runner.
    4. Key design and overall design of the class:
       Stateful preemptive scheduling heuristic. Ensures quantum counts execution ticks only,
       places newly admitted processes ahead of quantum-expired processes arriving at the same
       tick, and handles process completion and idle intervals cleanly.
    """

    def __init__(self, quantum: int = 2, name: str | None = None) -> None:
        """
        1. The use of function:
           Initializes a RoundRobinPolicy instance with a time quantum and policy name.
        2. How does the specific function work:
           Validates that quantum is an integer >= 1. Configures the policy name (defaulting to
           "RR" for quantum=2 or "RR(q=<quantum>)" otherwise), initializes an empty FIFO deque,
           and sets current_pid to None with 0 consumed quantum ticks.
        3. Concepts used in the function:
           Defensive parameter validation, constructor chaining, state initialization.
        4. Places where this function is imported and used:
           Instantiated in policy registry, CLI, simulation scripts, and test suites.
        5. Final output of the function, before and after:
           Before: Uninitialized RoundRobinPolicy instance.
           After: Fully configured RoundRobinPolicy ready for simulation execution.
        """
        if isinstance(quantum, bool) or not isinstance(quantum, int):
            raise TypeError(
                f"RoundRobinPolicy quantum must be an int, got {type(quantum).__name__}."
            )
        if quantum < 1:
            raise ValueError(f"RoundRobinPolicy quantum must be >= 1, got {quantum}.")

        policy_name = name if name is not None else ("RR" if quantum == 2 else f"RR(q={quantum})")
        super().__init__(name=policy_name)

        self._quantum: int = quantum
        self._deque: deque[str] = deque()
        self._current_pid: str | None = None
        self._quantum_ticks_consumed: int = 0

    @property
    def quantum(self) -> int:
        """
        1. The use of function:
           Returns the configured time quantum for this Round Robin policy.
        2. How does the specific function work:
           Returns the private integer attribute self._quantum.
        3. Concepts used in the function:
           Read-only property accessor, encapsulation.
        4. Places where this function is imported and used:
           Queried by CLI, evaluation benchmarks, and test assertions.
        5. Final output of the function, before and after:
           Before: Access to private _quantum.
           After: Returns configured integer time quantum.
        """
        return self._quantum

    @property
    def current_pid(self) -> str | None:
        """
        1. The use of function:
           Returns the identifier of the process currently occupying the CPU slice, if any.
        2. How does the specific function work:
           Returns the private attribute self._current_pid.
        3. Concepts used in the function:
           Read-only property accessor, introspection.
        4. Places where this function is imported and used:
           Queried during state inspection, debugging, and test assertions.
        5. Final output of the function, before and after:
           Before: Access to private _current_pid.
           After: Returns PID string or None.
        """
        return self._current_pid

    @property
    def quantum_ticks_consumed(self) -> int:
        """
        1. The use of function:
           Returns the number of execution ticks consumed by the current process in this quantum.
        2. How does the specific function work:
           Returns the private integer attribute self._quantum_ticks_consumed.
        3. Concepts used in the function:
           Read-only property accessor, state tracking.
        4. Places where this function is imported and used:
           Used for step-level verification and test inspection.
        5. Final output of the function, before and after:
           Before: Access to private _quantum_ticks_consumed.
           After: Returns integer count of consumed execution ticks.
        """
        return self._quantum_ticks_consumed

    @property
    def queue(self) -> tuple[str, ...]:
        """
        1. The use of function:
           Provides an immutable snapshot tuple of process IDs currently queued in FIFO order.
        2. How does the specific function work:
           Converts internal self._deque into an immutable tuple and returns it.
        3. Concepts used in the function:
           Immutable snapshot projection, defensive copying.
        4. Places where this function is imported and used:
           Inspected by test assertions and visualization components.
        5. Final output of the function, before and after:
           Before: Mutable internal deque state.
           After: Returns immutable tuple of queued PID strings.
        """
        return tuple(self._deque)

    def decision_cost_us(self, view: View) -> float:
        """
        1. The use of function:
           Returns the per-decision latency in microseconds for classical Round Robin.
        2. How does the specific function work:
           Returns 0.0 µs as classical baseline overhead is neglected in simulation, while
           measured latencies are preserved in models/latency_profile.json.
        3. Concepts used in the function:
           Classical baseline overhead exemption, hardware profile reference.
        4. Places where this function is imported and used:
           Invoked by Engine runner on non-forced scheduling steps.
        5. Final output of the function, before and after:
           Before: Observable View instance representing ready processes.
           After: Returns 0.0 µs float latency.
        """
        return 0.0

    def reset(self) -> None:
        """
        1. The use of function:
           Resets RoundRobinPolicy internal FIFO queue and quantum tracking state.
        2. How does the specific function work:
           Clears self._deque, resets self._current_pid to None, and sets
           self._quantum_ticks_consumed to 0.
        3. Concepts used in the function:
           State clearing, lifecycle management hook.
        4. Places where this function is imported and used:
           Invoked prior to running a new simulation workload or during test resets.
        5. Final output of the function, before and after:
           Before: Stateful FIFO queue and quantum progress from previous simulation.
           After: Fully cleared internal queue and zeroed quantum counters.
        """
        self._deque.clear()
        self._current_pid = None
        self._quantum_ticks_consumed = 0

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the next process PID to run according to Round Robin time-slicing rules.
        2. How does the specific function work:
           1. Verifies view.ready is non-empty; extracts ready PIDs set.
           2. Prunes deque of any PIDs no longer in ready set.
           3. Appends view.newly_arrived PIDs to the right of the deque in arrival order.
           4. If current_pid is still ready and has reached quantum ticks, re-enqueues it at the
              tail (after new arrivals), resetting current_pid and quantum_ticks_consumed.
           5. If current_pid is None or not in ready set, pops the next PID from the deque (or
              falls back to canonical ready queue ordering if deque is empty).
           6. Increments quantum_ticks_consumed by 1 and returns current_pid.
        3. Concepts used in the function:
           Round Robin scheduling, time slicing, FIFO queue dispatching, arrival priority ties.
        4. Places where this function is imported and used:
           Invoked on each discrete simulation step by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Observable simulation View at current tick.
           After: Returns selected PID string or raises ValueError if ready queue is empty.
        """
        if not view.ready:
            raise ValueError("Cannot select process: ready queue is empty.")

        ready_pids = {p.pid for p in view.ready}

        # 1. Prune deque of any PIDs no longer in the ready queue
        self._deque = deque(pid for pid in self._deque if pid in ready_pids)

        # 2. Append newly arrived processes to the right of the deque in arrival order
        for pid in view.newly_arrived:
            if pid in ready_pids and pid not in self._deque and pid != self._current_pid:
                self._deque.append(pid)

        # 3. If current process is still ready and has exhausted its quantum:
        #    Re-enqueue at tail AFTER newly arrived processes, and reset current process
        if self._current_pid is not None and self._current_pid in ready_pids:
            if self._quantum_ticks_consumed >= self._quantum:
                self._deque.append(self._current_pid)
                self._current_pid = None
                self._quantum_ticks_consumed = 0

        # 4. If no current process or current process is no longer ready (e.g. completed):
        #    Pop the next process from the head of the FIFO deque
        if self._current_pid is None or self._current_pid not in ready_pids:
            if self._deque:
                self._current_pid = self._deque.popleft()
                self._quantum_ticks_consumed = 0
            else:
                # Fallback: pick first available ready PID canonically by (arrival, pid)
                sorted_ready = sorted(view.ready, key=lambda p: (p.arrival, p.pid))
                self._current_pid = sorted_ready[0].pid
                self._quantum_ticks_consumed = 0
                for p in sorted_ready:
                    if p.pid != self._current_pid and p.pid not in self._deque:
                        self._deque.append(p.pid)

        # 5. Increment consumed execution ticks for the selected process
        self._quantum_ticks_consumed += 1

        # 6. Return selected process identifier
        return self._current_pid
