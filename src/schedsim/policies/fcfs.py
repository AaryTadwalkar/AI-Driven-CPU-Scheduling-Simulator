"""
1. What this file does:
   Implements the First-Come, First-Served (FCFS) classical CPU scheduling algorithm.
2. Overall design:
   Inherits from the Policy base class. At each discrete tick, selects the ready process
   with the minimum (arrival, pid) tuple. Because the running process already holds the minimal
   arrival time, FCFS is inherently non-preemptive. Deterministic tie-breaking uses lexicographical
   PID order.
3. Elements in the file:
   - FCFSPolicy: Class implementing non-preemptive First-Come, First-Served scheduling.
4. Final output of the file:
   Exports FCFSPolicy for simulation execution, benchmarking, and policy comparison.
"""

from schedsim.domain.engine import View
from schedsim.policies.base import Policy


class FCFSPolicy(Policy):
    """
    1. The use of class:
       Executes First-Come, First-Served (FCFS) process dispatching in the CPU scheduling simulator.
    2. What the class contains:
       - name: Identifier string "FCFS".
       - __init__: Initializes the policy with name "FCFS".
       - select: Selects the ready process having the minimum (arrival, pid).
    3. Technologies used and why:
       - Subclasses Policy ABC for polymorphic engine compatibility.
       - Deterministic min() key selection with composite tuple (arrival, pid).
    4. Key design and overall design of the class:
       Stateless heuristic scheduling based strictly on arrival chronology. Guarantees
       non-preemptive execution without requiring internal mutable state.
    """

    def __init__(self, name: str = "FCFS") -> None:
        """
        1. The use of function:
           Initializes FCFSPolicy instance with canonical name "FCFS" or a custom name.
        2. How does the specific function work:
           Calls super().__init__(name=name) to configure the policy identifier.
        3. Concepts used in the function:
           Constructor chaining, default parameterization.
        4. Places where this function is imported and used:
           Instantiated in policy registry, simulation pipelines, CLI, and test suites.
        5. Final output of the function, before and after:
           Before: Uninitialized FCFSPolicy.
           After: Configured FCFSPolicy instance ready for select() calls.
        """
        super().__init__(name=name)

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the ready process with the earliest arrival time.
        2. How does the specific function work:
           Verifies that view.ready is non-empty. Computes the ready process minimizing the tuple
           (arrival, pid) and returns its PID string.
        3. Concepts used in the function:
           Canonical ordering, deterministic tie-breaking, non-preemption.
        4. Places where this function is imported and used:
           Invoked on each discrete simulation step by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Observable simulation View at current tick.
           After: Returns selected PID string or raises ValueError if ready queue is empty.
        """
        if not view.ready:
            raise ValueError("Cannot select process: ready queue is empty.")
        best = min(view.ready, key=lambda p: (p.arrival, p.pid))
        return best.pid
