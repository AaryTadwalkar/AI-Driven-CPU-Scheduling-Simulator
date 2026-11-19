"""
1. What this file does:
   Defines the abstract base class and interface contract for all CPU scheduling policies,
   including lifecycle hooks for upfront and per-decision model inference latency accounting.
2. Overall design:
   Establishes a uniform protocol for discrete unit-step scheduling algorithms. Policies must
   implement the abstract select() method to pick the next PID from the current simulation View,
   provide a reset() hook for stateful algorithms, implement __call__ for callable delegation,
   and provide on_start() / decision_cost_us() hooks for AI overhead modeling.
3. Elements in the file:
   - Policy: Abstract base class providing the standardized interface for scheduling policies.
4. Final output of the file:
   Exports the Policy abstract base class for classical and learned scheduling algorithms.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from schedsim.domain.engine import View


class Policy(ABC):
    """
    1. The use of class:
       Serves as the abstract base class defining the contract for CPU scheduling policies.
    2. What the class contains:
       - name: Human-readable policy identifier string.
       - __init__: Initializes policy instance with a descriptive name.
       - select: Abstract method choosing the next process PID to run given a simulation View.
       - reset: Method resetting internal state between simulation runs (default no-op).
       - on_start: Hook returning one-time upfront model inference latency in microseconds.
       - decision_cost_us: Hook returning per-decision model inference latency in microseconds.
       - __call__: Magic method delegating callable invocation directly to select(view).
    3. Technologies used and why:
       - abc.ABC and @abstractmethod to enforce uniform interface implementation across policies.
       - Python 3.11 type annotations for static typing and safe polymorphic dispatch.
    4. Key design and overall design of the class:
       Decouples the core simulation engine from dispatch heuristics. Any policy conforming to
       Policy can be plugged interchangeably into Engine and run().
    """

    name: str = "Policy"

    def __init__(self, name: str = "Policy") -> None:
        """
        1. The use of function:
           Initializes a Policy instance with an identifier name.
        2. How does the specific function work:
           Validates that name is a non-empty string and stores it in self.name.
        3. Concepts used in the function:
           Defensive parameter checking, object initialization.
        4. Places where this function is imported and used:
           Invoked during instantiation of derived policy subclasses.
        5. Final output of the function, before and after:
           Before: Uninitialized Policy instance.
           After: Policy instance with configured name attribute or raises TypeError/ValueError.
        """
        if not isinstance(name, str):
            raise TypeError(f"Policy name must be a str, got {type(name).__name__}.")
        if not name.strip():
            raise ValueError("Policy name cannot be empty or whitespace only.")
        self.name = name

    @abstractmethod
    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the process ID of the next process to execute on the CPU for the current tick.
        2. How does the specific function work:
           Abstract method to be implemented by concrete scheduling policy subclasses.
        3. Concepts used in the function:
           Polymorphic dispatch, abstract interface specification.
        4. Places where this function is imported and used:
           Invoked on each discrete simulation step by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Current simulation state View.
           After: Returns selected PID string to execute.
        """
        pass  # pragma: no cover

    def reset(self) -> None:
        """
        1. The use of function:
           Resets internal policy state before or between simulation runs.
        2. How does the specific function work:
           Default implementation is a no-op; stateful policies (e.g. Round Robin) override this.
        3. Concepts used in the function:
           Lifecycle management hook, stateless/stateful policy abstraction.
        4. Places where this function is imported and used:
           Called before beginning a new simulation episode or when resetting stateful schedulers.
        5. Final output of the function, before and after:
           Before: Policy state from prior execution or initialization.
           After: Cleared internal state ready for a fresh simulation run.
        """
        pass

    def on_start(self, processes: Sequence[Any]) -> float:
        """
        1. The use of function:
           Hook invoked once at workload start, returning upfront inference/feature latency in µs.
        2. How does the specific function work:
           Default implementation returns 0.0 µs (no overhead). Category 1 ML models override
           this to return measured feature extraction and prediction time over the workload.
        3. Concepts used in the function:
           Lifecycle hook, one-time upfront model cost accounting, default no-op.
        4. Places where this function is imported and used:
           Invoked by Engine runner (run()) prior to the discrete simulation stepping loop.
        5. Final output of the function, before and after:
           Before: Sequence of workload Process instances.
           After: Returns non-negative float inference cost in microseconds (default 0.0).
        """
        return 0.0

    def decision_cost_us(self, view: View) -> float:
        """
        1. The use of function:
           Returns the estimated or measured latency in microseconds for a non-forced decision.
        2. How does the specific function work:
           Default implementation returns 0.0 µs (no overhead). Learned per-decision policies
           (e.g., Q-learning, GBDT, Neural Scorer) override this based on ready queue size.
        3. Concepts used in the function:
           Per-decision overhead accounting, latency model lookup, default no-op.
        4. Places where this function is imported and used:
           Invoked by Engine runner (run()) or test harnesses on each non-forced scheduling step.
        5. Final output of the function, before and after:
           Before: Observable View instance representing ready processes.
           After: Returns non-negative float latency in microseconds (default 0.0).
        """
        return 0.0

    def __call__(self, view: View) -> str:
        """
        1. The use of function:
           Enables direct callable invocation of the policy instance with a View.
        2. How does the specific function work:
           Delegates directly to self.select(view) and returns the chosen PID.
        3. Concepts used in the function:
           Python callable protocol (__call__), delegation pattern.
        4. Places where this function is imported and used:
           Used when policies are treated as callables in functional simulation loops and tests.
        5. Final output of the function, before and after:
           Before: View instance passed as callable argument.
           After: Returns PID string selected by self.select(view).
        """
        return self.select(view)
