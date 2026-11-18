"""
1. What this file does:
   Implements the Shortest Job First (SJF) non-preemptive classical CPU scheduling algorithm.
2. Overall design:
   Inherits from the Policy base class. Operates in non-preemptive mode: if the previously running
   process (view.last_pid) is still in the ready queue, it retains the CPU. When CPU allocation is
   open (last_pid finished or None), it selects the ready process with the minimum
   (burst, arrival, pid) tuple.
3. Elements in the file:
   - SJFPolicy: Class implementing non-preemptive Shortest Job First scheduling.
4. Final output of the file:
   Exports SJFPolicy for simulation execution, benchmarking, and policy comparison.
"""

from schedsim.domain.engine import View
from schedsim.policies.base import Policy


class SJFPolicy(Policy):
    """
    1. The use of class:
       Executes non-preemptive Shortest Job First (SJF) dispatching in the simulation engine.
    2. What the class contains:
       - name: Identifier string "SJF".
       - __init__: Initializes the policy with name "SJF".
       - select: Retains view.last_pid if still ready, else selects minimum (burst, arrival, pid).
    3. Technologies used and why:
       - Subclasses Policy ABC for uniform simulation execution.
       - Leverages view.last_pid to enforce non-preemption without local state mutation.
    4. Key design and overall design of the class:
       Pure functional non-preemptive scheduling heuristic. Guarantees that running jobs run to
       completion before dispatching the shortest total burst job.
    """

    def __init__(self, name: str = "SJF") -> None:
        """
        1. The use of function:
           Initializes SJFPolicy instance with canonical name "SJF" or a custom name.
        2. How does the specific function work:
           Calls super().__init__(name=name) to configure the policy identifier.
        3. Concepts used in the function:
           Constructor chaining, default parameterization.
        4. Places where this function is imported and used:
           Instantiated in policy registry, simulation pipelines, CLI, and test suites.
        5. Final output of the function, before and after:
           Before: Uninitialized SJFPolicy.
           After: Configured SJFPolicy instance ready for select() calls.
        """
        super().__init__(name=name)

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the next process PID to run according to non-preemptive SJF rules.
        2. How does the specific function work:
           Verifies that view.ready is non-empty. If view.last_pid is non-None and present in
           view.ready, retains view.last_pid (non-preemptive execution). Otherwise, selects the
           ready process with minimum (burst, arrival, pid) and returns its PID.
        3. Concepts used in the function:
           Non-preemptive dispatch, multi-criteria tie-breaking, greedy optimization.
        4. Places where this function is imported and used:
           Invoked on each discrete simulation step by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Observable simulation View at current tick.
           After: Returns selected PID string or raises ValueError if ready queue is empty.
        """
        if not view.ready:
            raise ValueError("Cannot select process: ready queue is empty.")

        if view.last_pid is not None and any(p.pid == view.last_pid for p in view.ready):
            return view.last_pid

        best = min(view.ready, key=lambda p: (p.burst, p.arrival, p.pid))
        return best.pid
