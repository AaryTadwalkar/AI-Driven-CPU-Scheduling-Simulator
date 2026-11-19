"""
1. What this file does:
   Implements the RLPolicy adapter allowing trained reinforcement learning agents to execute
   as standard scheduling policies within the unified unit-step Engine, supporting per-setting
   Q-table loading and explicit decision cost lifecycle hooks.
2. Overall design:
   Inherits from abstract Policy base class. Translates continuous simulation View instances
   into compact 7-tuple RLState representations, queries the greedy QLearningAgent for optimal
   rule-based action selection, and resolves the action into a concrete process PID via canonical
   tie-breaking. Provides factory constructors loading trained Q-tables for benchmark settings.
3. Elements in the file:
   - RLPolicy: Policy subclass wrapping a trained QLearningAgent.
4. Final output of the file:
   Exports RLPolicy for simulation runs, evaluation suites, CLI commands, and REST APIs.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Optional

import config
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
       - from_cs_cost: Factory instantiating policy with Q-table corresponding to cs_cost.
       - from_setting: Factory instantiating policy with Q-table for benchmark setting (S0/S1).
       - bind_workload: Binds a specific process sequence to track pending arrivals.
       - on_start: Binds active workload and returns 0.0 µs upfront cost.
       - decision_cost_us: Returns 0.0 µs baseline latency per non-forced decision.
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

    @classmethod
    def from_cs_cost(
        cls,
        cs_cost: int = 1,
        processes: Optional[Sequence[Process]] = None,
        models_dir: Path | str = config.MODELS_DIR,
        name: Optional[str] = None,
    ) -> "RLPolicy":
        """
        1. The use of function:
           Factory method instantiating RLPolicy with the trained Q-table matching cs_cost.
        2. How does the specific function work:
           Resolves model file path based on cs_cost (q_table_cs0.pkl or q_table_cs1.pkl /
           q_table.pkl), loads QLearningAgent via QLearningAgent.load(), and constructs RLPolicy.
        3. Concepts used in the function:
           Classmethod factory pattern, per-setting model selection, defensive file validation.
        4. Places where this function is imported and used:
           Invoked by benchmark evaluation suites, CLI commands, and Stage 12 acceptance tests.
        5. Final output of the function, before and after:
           Before: cs_cost integer and optional parameters.
           After: Configured RLPolicy instance wrapping the trained agent.
        """
        dir_path = Path(models_dir)
        target_file = dir_path / f"q_table_cs{cs_cost}.pkl"
        if not target_file.exists():
            fallback_file = dir_path / "q_table.pkl"
            if fallback_file.exists():
                target_file = fallback_file
            else:
                raise FileNotFoundError(
                    f"Trained Q-table for cs_cost={cs_cost} not found at {target_file}."
                )

        agent = QLearningAgent.load(target_file)
        policy_name = name or f"RLPolicy_cs{cs_cost}"
        return cls(agent=agent, processes=processes, name=policy_name)

    @classmethod
    def from_setting(
        cls,
        setting: str = "S1",
        processes: Optional[Sequence[Process]] = None,
        models_dir: Path | str = config.MODELS_DIR,
        name: Optional[str] = None,
    ) -> "RLPolicy":
        """
        1. The use of function:
           Factory method instantiating RLPolicy with the trained Q-table for a benchmark setting.
        2. How does the specific function work:
           Validates setting name against config.SETTINGS, extracts corresponding cs_cost penalty,
           and delegates to cls.from_cs_cost().
        3. Concepts used in the function:
           Setting resolution, factory delegation, benchmark parameter encapsulation.
        4. Places where this function is imported and used:
           Used by multi-setting benchmark runners and Stage 12 unit/acceptance tests.
        5. Final output of the function, before and after:
           Before: Setting identifier string and optional parameters.
           After: Configured RLPolicy instance wrapping the setting's trained Q-table.
        """
        setting_key = setting.upper().strip()
        if setting_key not in config.SETTINGS:
            valid = ", ".join(config.SETTINGS.keys())
            raise ValueError(f"Unknown setting '{setting}'. Must be one of: {valid}.")
        cs_cost = config.SETTINGS[setting_key].cs_cost
        policy_name = name or f"RLPolicy_{setting_key}"
        return cls.from_cs_cost(
            cs_cost=cs_cost,
            processes=processes,
            models_dir=models_dir,
            name=policy_name,
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
           Called automatically by on_start() or explicitly prior to simulation runs.
        5. Final output of the function, before and after:
           Before: Policy with previous or unconfigured workload reference.
           After: Policy updated with active workload processes; returns self.
        """
        self._processes = tuple(processes)
        return self

    def on_start(self, processes: Sequence[Process]) -> float:
        """
        1. The use of function:
           Lifecycle hook invoked once at simulation start to bind active workload processes.
        2. How does the specific function work:
           Caches the process sequence via bind_workload() and returns 0.0 µs upfront cost.
        3. Concepts used in the function:
           Lifecycle hook, automatic workload binding, zero upfront inference cost.
        4. Places where this function is imported and used:
           Invoked by Engine runner (run()) before the simulation stepping loop begins.
        5. Final output of the function, before and after:
           Before: Unbound or previously bound workload processes.
           After: Workload processes cached on policy instance; returns 0.0 µs.
        """
        self.bind_workload(processes)
        return 0.0

    def decision_cost_us(self, view: View) -> float:
        """
        1. The use of function:
           Returns the estimated decision latency in microseconds from the latency profile.
        2. How does the specific function work:
           Returns 0.0 if view is None or ready count is <= 1 (forced decision). Otherwise
           queries get_interpolated_cost_us("Q-learning", len(view.ready)).
        3. Concepts used in the function:
           Per-decision overhead accounting, latency profile lookup, interpolation.
        4. Places where this function is imported and used:
           Invoked by Engine during discrete simulation stepping loop when tick_us is configured.
        5. Final output of the function, before and after:
           Before: Observable View instance representing ready processes.
           After: Returns float latency in microseconds.
        """
        if view is None or (config.FORCED_DECISION_FREE and len(view.ready) <= 1):
            return 0.0
        from schedsim.overhead import get_interpolated_cost_us

        return get_interpolated_cost_us("Q-learning", len(view.ready))

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
