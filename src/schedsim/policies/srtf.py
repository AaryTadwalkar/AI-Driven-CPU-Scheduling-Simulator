"""
1. What this file does:
   Implements Shortest Remaining Time First (SRTF) preemptive CPU scheduling algorithm.
2. Overall design:
   Inherits from the Policy base class. At each discrete tick, selects the ready process
   with the minimum (remaining, 0 if pid == last_pid else 1, arrival, pid) tuple. The currently
   running process wins ties to prevent unnecessary preemption and context switch churn.
3. Elements in the file:
   - SRTFPolicy: Class implementing preemptive Shortest Remaining Time First scheduling.
4. Final output of the file:
   Exports SRTFPolicy for simulation execution, benchmarking, and reference baseline evaluation.
"""

from schedsim.domain.engine import View
from schedsim.policies.base import Policy


class SRTFPolicy(Policy):
    """
    1. The use of class:
       Executes preemptive Shortest Remaining Time First (SRTF) scheduling in the simulation engine.
    2. What the class contains:
       - name: Identifier string "SRTF".
       - __init__: Initializes policy with name "SRTF".
       - select: Selects process with minimum (remaining, running-first, arrival, pid).
    3. Technologies used and why:
       - Subclasses Policy ABC for polymorphic integration.
       - Dynamic tie-breaking tuple prioritizing remaining burst, running continuity, arrival,
         and lexicographical pid.
    4. Key design and overall design of the class:
       Optimal single-core waiting time baseline at zero context switch cost. Preempts immediately
       when a strictly shorter remaining burst process arrives, but avoids thrashing by favoring
       the running process during equal-remaining-burst ties.
    """

    def __init__(self, name: str = "SRTF") -> None:
        """
        1. The use of function:
           Initializes SRTFPolicy instance with canonical name "SRTF" or a custom name.
        2. How does the specific function work:
           Calls super().__init__(name=name) to configure the policy identifier.
        3. Concepts used in the function:
           Constructor chaining, default parameterization.
        4. Places where this function is imported and used:
           Instantiated in policy registry, simulation pipelines, CLI, and test suites.
        5. Final output of the function, before and after:
           Before: Uninitialized SRTFPolicy.
           After: Configured SRTFPolicy instance ready for select() calls.
        """
        super().__init__(name=name)

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the next process PID to run according to preemptive SRTF rules.
        2. How does the specific function work:
           Verifies that view.ready is non-empty. Evaluates each ready process using the key
           (remaining, 0 if pid == last_pid else 1, arrival, pid) and returns the winning PID.
        3. Concepts used in the function:
           Preemption, remaining-time optimization, thrashing prevention, tie-breaking.
        4. Places where this function is imported and used:
           Invoked on each discrete simulation step by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Observable simulation View at current tick.
           After: Returns selected PID string or raises ValueError if ready queue is empty.
        """
        if not view.ready:
            raise ValueError("Cannot select process: ready queue is empty.")

        best = min(
            view.ready,
            key=lambda p: (
                p.remaining,
                0 if p.pid == view.last_pid else 1,
                p.arrival,
                p.pid,
            ),
        )
        return best.pid
