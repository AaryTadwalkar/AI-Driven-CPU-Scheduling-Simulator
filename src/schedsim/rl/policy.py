"""
1. What this file does:
   Implements the RLPolicy adapter allowing trained reinforcement learning agents to execute
   as standard scheduling policies within the unified unit-step Engine.
2. Overall design:
   Inherits from the abstract Policy base class. Translates continuous simulation View instances
   into compact 7-tuple RLState representations, queries the greedy QLearningAgent for optimal
   rule-based action selection, and resolves the action into a concrete process PID via canonical
   tie-breaking.
3. Elements in the file:
   - RLPolicy: Policy subclass wrapping a trained QLearningAgent.
4. Final output of the file:
   Exports RLPolicy for simulation runs, evaluation suites, CLI commands, and REST APIs.
"""

from collections.abc import Sequence

from schedsim.domain.engine import View
from schedsim.domain.process import Process
from schedsim.policies.base import Policy
from schedsim.rl.actions import action_to_pid, get_valid_actions
from schedsim.rl.agent import QLearningAgent
from schedsim.rl.state import encode_state


class RLPolicy(Policy):
    """
    1. The use of class:
       Wraps a trained QLearningAgent to provide a standard Policy interface for simulation.
    2. What the class contains:
       - name: Policy identifier string (defaults to 'RLPolicy' or configured name).
       - agent: Trained QLearningAgent instance evaluating actions greedily.
       - _processes: Optional cached tuple of Process instances for exact arrival tracking.
       - __init__: Initializes policy with agent, optional processes, and identifier name.
       - bind_workload: Binds a specific process sequence to track pending arrivals.
       - reset: Resets policy state between simulation episodes.
       - select: Encodes View into RLState, selects greedy action, and returns target PID.
    3. Technologies used and why:
       - Polymorphic Policy inheritance ensuring seamless integration into Engine and run().
       - Decoupled inference: evaluates purely greedily without modifying Q-values.
    4. Key design and overall design of the class:
       Acts as the single bridge between reinforcement learning tabular policies and the
       classical simulation harness, allowing unified comparison under identical metrics.
    """

    name: str = "RLPolicy"

    def __init__(
        self,
        agent: QLearningAgent,
        processes: Sequence[Process] | None = None,
        name: str = "RLPolicy",
    ) -> None:
        """
        1. The use of function:
           Initializes an RLPolicy instance wrapping a trained QLearningAgent.
        2. How does the specific function work:
           Validates agent is an instance of QLearningAgent. Stores agent reference, optionally
           converts processes into an immutable tuple, and passes name to super().__init__().
        3. Concepts used in the function:
           Constructor delegation, defensive parameter assertion, immutable collection storage.
        4. Places where this function is imported and used:
           Instantiated by evaluation scripts, CLI commands, and unit tests.
        5. Final output of the function, before and after:
           Before: Uninitialized RLPolicy reference.
           After: Fully initialized RLPolicy instance ready for select() calls.
        """
        if not isinstance(agent, QLearningAgent):
            raise TypeError(
                f"agent must be a QLearningAgent instance, got {type(agent).__name__}."
            )
        super().__init__(name=name)
        self.agent: QLearningAgent = agent
        self._processes: tuple[Process, ...] | None = (
            tuple(processes) if processes is not None else None
        )

    def bind_workload(self, processes: Sequence[Process]) -> "RLPolicy":
        """
        1. The use of function:
           Binds a process workload sequence to the policy to enable exact pending arrival checks.
        2. How does the specific function work:
           Converts input processes sequence into an immutable tuple and caches it on the instance.
           Returns self for fluent method chaining.
        3. Concepts used in the function:
           Workload binding, fluent interface, arrival state tracking.
        4. Places where this function is imported and used:
           Called automatically by run() or explicitly prior to simulation runs.
        5. Final output of the function, before and after:
           Before: Policy with previous or unconfigured workload reference.
           After: Policy updated with active workload processes; returns self.
        """
        self._processes = tuple(processes)
        return self

    def reset(self) -> None:
        """
        1. The use of function:
           Resets internal state between simulation runs.
        2. How does the specific function work:
           Currently a no-op since QLearningAgent evaluation is stateless and greedy.
        3. Concepts used in the function:
           Policy reset hook, interface compliance.
        4. Places where this function is imported and used:
           Invoked by simulation runners between episodes.
        5. Final output of the function, before and after:
           Before: Policy state prior to reset.
           After: Policy ready for next episode.
        """
        pass

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the next process PID to dispatch on CPU based on the trained Q-table.
        2. How does the specific function work:
           Computes valid actions from current View. Determines if any unadmitted processes remain
           in the future (if workload was bound, else False). Encodes current View into compact
           7-tuple RLState. Queries agent.best_action() for the greedy decision. Resolves action
           into concrete PID string via action_to_pid() and returns it.
        3. Concepts used in the function:
           Greedy policy dispatch, state feature projection, canonical tie-breaking.
        4. Places where this function is imported and used:
           Invoked on every simulation execution tick by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Observable simulation View.
           After: Returns selected ready process PID string.
        """
        if not isinstance(view, View):
            raise TypeError(f"view must be a View instance, got {type(view).__name__}.")

        valid_actions = get_valid_actions(view)
        if not valid_actions:
            raise ValueError("Cannot select process: no valid actions available in current View.")

        if self._processes is not None:
            more_pending = any(p.arrival > view.time for p in self._processes)
        else:
            more_pending = False

        state = encode_state(view, more_arrivals_pending=more_pending)
        action = self.agent.best_action(state, valid_actions, record_unseen=False)
        return action_to_pid(view, action)


__all__ = ["RLPolicy"]
