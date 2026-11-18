"""
1. What this file does:
   Implements the reinforcement learning Environment wrapping the discrete unit-step CPU
   scheduling Engine into a standardized MDP interface.
2. Overall design:
   Provides a gym-like step-by-step reinforcement learning environment:
   - reset: Initializes the simulation Engine with a workload and returns the initial RLState.
   - valid_actions: Returns unmasked discrete action indices for the current simulation View.
   - step: Dispatches a chosen rule-based action index, advances the simulation by one execution
     tick (and any preceding context switch overhead), computes the negative waiting penalty
     reward, and returns (next_state, reward, done, info).
   - result: Exposes final SimResult metrics and execution timeline upon episode completion.
3. Elements in the file:
   - Environment: MDP environment class wrapping Engine, action dispatch, and state encoding.
4. Final output of the file:
   Exports Environment for Q-learning training loops, evaluation agents, and acceptance tests.
"""

from collections.abc import Sequence
from typing import Any

from schedsim.domain.engine import Engine, SimResult
from schedsim.domain.process import Process, validate_processes
from schedsim.rl.actions import action_to_pid, get_valid_actions
from schedsim.rl.state import TERMINAL_STATE, RLState, encode_state


class Environment:
    """
    1. The use of class:
       Provides a standardized MDP environment wrapping the unified unit-step CPU scheduling
       Engine for reinforcement learning agents.
    2. What the class contains:
       - _cs_cost: Integer context switch overhead tick cost.
       - _workload: Tuple of Process domain objects representing the current episode workload.
       - _engine: Internal Engine instance executing unit ticks.
       - _current_state: Most recent encoded 7-tuple RLState.
       - reset: Resets environment state for a workload and returns initial RLState.
       - valid_actions: Computes valid, unmasked action indices for the current state.
       - step: Executes chosen action index, returns (next_state, reward, done, info).
       - cs_cost: Property returning configured context switch cost.
       - engine: Property returning internal Engine instance.
       - done: Property indicating whether episode has terminated.
       - time: Property returning current simulation tick.
       - state: Property returning current RLState.
       - workload: Property returning active workload processes.
       - result: Returns final SimResult from the Engine upon completion.
    3. Technologies used and why:
       - Wraps Engine directly ensuring 100% semantic identity with classical baselines.
       - Strict type validation and action masking ensuring robust MDP rollouts.
    4. Key design and overall design of the class:
       Decouples Q-learning agent from low-level process tracking by exposing discrete rule
       actions (0..3) and compact permutation-invariant aggregate state tuples.
    """

    def __init__(
        self,
        cs_cost: int = 0,
        workload: Sequence[Process] | None = None,
    ) -> None:
        """
        1. The use of function:
           Initializes an Environment instance with context switch cost and optional workload.
        2. How does the specific function work:
           Validates cs_cost is a non-negative integer. If workload is supplied, validates and
           stores it, then immediately resets the environment to establish initial state.
        3. Concepts used in the function:
           Environment initialization, defensive parameter assertion, optional initial setup.
        4. Places where this function is imported and used:
           Instantiated by RL training loops, evaluation runners, and unit test suites.
        5. Final output of the function, before and after:
           Before: Uninitialized Environment reference.
           After: Fully initialized Environment ready for step() or reset() calls.
        """
        if isinstance(cs_cost, bool) or not isinstance(cs_cost, int):
            raise TypeError(f"cs_cost must be an int, got {type(cs_cost).__name__}.")
        if cs_cost < 0:
            raise ValueError(f"cs_cost must be >= 0, got {cs_cost}.")

        self._cs_cost: int = cs_cost
        self._workload: tuple[Process, ...] | None = None
        self._engine: Engine | None = None
        self._current_state: RLState | None = None

        if workload is not None:
            self.reset(workload)

    def reset(self, workload: Sequence[Process] | None = None) -> RLState:
        """
        1. The use of function:
           Resets the environment for an episode and returns the initial encoded RLState.
        2. How does the specific function work:
           If workload is provided, validates processes and stores them as an immutable tuple.
           If workload is omitted, uses the previously configured workload. Instantiates and resets
           the simulation Engine. If Engine is done immediately, returns TERMINAL_STATE; otherwise,
           encodes the initial View into a compact 7-tuple RLState and caches it.
        3. Concepts used in the function:
           Episode initialization, simulation reset, state feature extraction and binning.
        4. Places where this function is imported and used:
           Called at the start of each RL training episode, evaluation rollout, and test.
        5. Final output of the function, before and after:
           Before: Environment in previous or uninitialized state.
           After: Returns initial RLState tuple and readies environment for step() calls.
        """
        if workload is not None:
            validate_processes(workload)
            self._workload = tuple(workload)
        elif self._workload is None:
            raise ValueError("Workload must be provided on first reset() call.")

        self._engine = Engine(self._workload, cs_cost=self._cs_cost)
        initial_view = self._engine.current_view

        if self._engine.done or initial_view is None or not initial_view.ready:
            self._current_state = TERMINAL_STATE
            return self._current_state

        more_pending = self._engine.has_pending_arrivals
        self._current_state = encode_state(initial_view, more_arrivals_pending=more_pending)
        return self._current_state

    def valid_actions(self) -> list[int]:
        """
        1. The use of function:
           Returns the list of valid, unmasked action indices available in the current state.
        2. How does the specific function work:
           If engine is not initialized or is already done, returns an empty list. Otherwise
           retrieves current View from Engine and delegates to get_valid_actions(view).
        3. Concepts used in the function:
           Action masking, valid action pruning, MDP constraints.
        4. Places where this function is imported and used:
           Called prior to action selection by Q-learning epsilon-greedy policy and tests.
        5. Final output of the function, before and after:
           Before: Environment state inspected.
           After: Returns list of available integer action indices (e.g. [0, 1, 2] or [0, 1, 2, 3]).
        """
        if self._engine is None or self._engine.done:
            return []

        view = self._engine.current_view
        if view is None or not view.ready:
            return []

        return get_valid_actions(view)

    def step(self, action: int) -> tuple[RLState, float, bool, dict[str, Any]]:
        """
        1. The use of function:
           Advances the simulation environment by one step using the chosen rule-based action.
        2. How does the specific function work:
           Validates that the environment has been reset and is not terminated. Asserts action is
           in valid_actions(). Resolves action index to target PID via action_to_pid(). Steps the
           underlying Engine, computes reward = -float(step_info.penalty), extracts next_state
           (or TERMINAL_STATE if done), updates cached state, and returns transition tuple.
        3. Concepts used in the function:
           Markov decision process step, reward assignment, penalty conservation, termination.
        4. Places where this function is imported and used:
           Called inside RL training loops, rollout trajectories, and acceptance tests.
        5. Final output of the function, before and after:
           Before: Environment at current simulation tick.
           After: Advances engine by 1 tick (+CS), returns (next_state, reward, done, info).
        """
        if self._engine is None:
            raise RuntimeError("Environment must be reset with a workload before calling step().")
        if self._engine.done:
            raise RuntimeError("Cannot call step() on a completed environment. Call reset().")

        valid = self.valid_actions()
        if action not in valid:
            raise ValueError(
                f"Action {action} is invalid or masked. Valid actions: {valid}."
            )

        current_view = self._engine.current_view
        if current_view is None or not current_view.ready:
            raise RuntimeError("Cannot execute step: Engine view has no ready processes.")

        pid = action_to_pid(current_view, action)
        step_info = self._engine.step(pid)

        reward = -float(step_info.penalty)
        done = step_info.done

        if done:
            next_state = TERMINAL_STATE
        else:
            more_pending = self._engine.has_pending_arrivals
            next_state = encode_state(step_info.view, more_arrivals_pending=more_pending)

        self._current_state = next_state
        info: dict[str, Any] = {
            "pid": pid,
            "penalty": step_info.penalty,
            "time": self._engine.time,
        }
        return next_state, reward, done, info

    @property
    def cs_cost(self) -> int:
        """
        1. The use of function:
           Returns the configured context switch cost in ticks.
        2. How does the specific function work:
           Returns internal _cs_cost integer attribute.
        3. Concepts used in the function:
           Property accessor, configuration inspection.
        4. Places where this function is imported and used:
           Used by agents, test assertions, and evaluation reporting.
        5. Final output of the function, before and after:
           Before: Property accessed on Environment instance.
           After: Returns integer context switch cost.
        """
        return self._cs_cost

    @property
    def engine(self) -> Engine | None:
        """
        1. The use of function:
           Returns the underlying simulation Engine instance.
        2. How does the specific function work:
           Returns internal _engine reference or None if not reset.
        3. Concepts used in the function:
           Property accessor, engine inspection.
        4. Places where this function is imported and used:
           Used by evaluators, visualizers, and audit inspectors.
        5. Final output of the function, before and after:
           Before: Property accessed on Environment instance.
           After: Returns Engine instance or None.
        """
        return self._engine

    @property
    def done(self) -> bool:
        """
        1. The use of function:
           Indicates whether the current simulation episode has finished.
        2. How does the specific function work:
           If engine is None, returns True. Otherwise delegates to self._engine.done.
        3. Concepts used in the function:
           Property accessor, episode termination check.
        4. Places where this function is imported and used:
           Used by training loop conditions, agent rollout loops, and test assertions.
        5. Final output of the function, before and after:
           Before: Property accessed on Environment instance.
           After: Returns True if episode finished, False otherwise.
        """
        if self._engine is None:
            return True
        return self._engine.done

    @property
    def time(self) -> int:
        """
        1. The use of function:
           Returns the current simulation tick timestamp.
        2. How does the specific function work:
           If engine is None, returns 0. Otherwise delegates to self._engine.time.
        3. Concepts used in the function:
           Property accessor, discrete time query.
        4. Places where this function is imported and used:
           Used for step tracking, logging, and debugging.
        5. Final output of the function, before and after:
           Before: Property accessed on Environment instance.
           After: Returns current integer tick.
        """
        if self._engine is None:
            return 0
        return self._engine.time

    @property
    def state(self) -> RLState | None:
        """
        1. The use of function:
           Returns the most recent encoded RLState tuple.
        2. How does the specific function work:
           Returns internal _current_state attribute.
        3. Concepts used in the function:
           Property accessor, MDP state query.
        4. Places where this function is imported and used:
           Used by Q-learning agents to inspect current state without stepping.
        5. Final output of the function, before and after:
           Before: Property accessed on Environment instance.
           After: Returns RLState tuple or None if uninitialized.
        """
        return self._current_state

    @property
    def workload(self) -> tuple[Process, ...] | None:
        """
        1. The use of function:
           Returns the active workload processes for the current episode.
        2. How does the specific function work:
           Returns internal _workload tuple or None if uninitialized.
        3. Concepts used in the function:
           Property accessor, workload inspection.
        4. Places where this function is imported and used:
           Used by evaluation logging and reproducibility verification.
        5. Final output of the function, before and after:
           Before: Property accessed on Environment instance.
           After: Returns tuple of Process instances or None.
        """
        return self._workload

    def result(self) -> SimResult:
        """
        1. The use of function:
           Compiles and returns the comprehensive SimResult from the underlying Engine.
        2. How does the specific function work:
           If engine is None, raises RuntimeError. Otherwise forwards call to self._engine.result().
        3. Concepts used in the function:
           Result delegation, final evaluation metric extraction.
        4. Places where this function is imported and used:
           Called after episode completion by agents, evaluation runners, and tests.
        5. Final output of the function, before and after:
           Before: Completed Environment instance.
           After: Returns SimResult containing process metrics, simulation metrics, and timeline.
        """
        if self._engine is None:
            raise RuntimeError("Environment has not been initialized with reset().")
        return self._engine.result()

    def __repr__(self) -> str:
        """
        1. The use of function:
           Returns a human-readable string representation of the Environment.
        2. How does the specific function work:
           Formats class name, cs_cost, done status, and current tick timestamp.
        3. Concepts used in the function:
           String representation, debugging utility.
        4. Places where this function is imported and used:
           Used in debugging and logging outputs.
        5. Final output of the function, before and after:
           Before: Environment instance inspected.
           After: Formatted string summary.
        """
        return (
            f"Environment(cs_cost={self._cs_cost}, done={self.done}, "
            f"time={self.time})"
        )


__all__ = ["Environment"]
