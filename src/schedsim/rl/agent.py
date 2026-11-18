"""
1. What this file does:
   Implements the tabular Q-learning agent for learning optimal CPU scheduling heuristics under
   arbitrary context switch costs.
2. Overall design:
   Provides a tabular Q-learning implementation over the discrete 7-tuple aggregate state space:
   - Uses a sparse dictionary Q-table mapping (state, action) pairs to expected discounted return.
   - Implements epsilon-greedy exploration over dynamically valid action subsets with epsilon
     decay down to a configured minimum exploration floor.
   - Enforces canonical tie-breaking by lowest action index when multiple actions share maximal Q.
   - Tracks state visitation frequencies and records unseen state fallbacks during evaluation.
   - Implements model serialization with metadata and human-readable JSON state summary exports.
3. Elements in the file:
   - QLearningAgent: Core tabular Q-learning reinforcement learning agent class.
4. Final output of the file:
   Exports QLearningAgent for training loops, policy adapters, evaluation pipelines, and tests.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

import config
from schedsim.rl.actions import (
    ACTION_EARLIEST_ARRIVAL,
    ACTION_NAMES,
    ALL_ACTIONS,
)
from schedsim.rl.state import RLState


class QLearningAgent:
    """
    1. The use of class:
       Maintains tabular Q-values and executes temporal difference learning for CPU scheduling.
    2. What the class contains:
       - alpha: Learning rate parameter for Bellman temporal difference updates.
       - gamma: Discount factor parameter weighting future returns.
       - epsilon: Current probability of selecting a random exploratory valid action.
       - eps_min: Lower bound floor for epsilon decay.
       - eps_decay: Multiplicative decay factor applied to epsilon after each episode.
       - _q_table: Dictionary mapping (RLState, int) tuples to float Q-value estimates.
       - _state_visits: Counter tracking total visits per RLState.
       - _action_visits: Counter tracking selections per (RLState, int) pair.
       - _unseen_state_eval_count: Counter of fallback decisions triggered on unseen states.
       - _total_eval_decisions: Counter of total evaluation action decisions made.
       - metadata: Dictionary storing training provenance, hyperparameters, and environment info.
       - get_q, set_q: Q-value getter and setter methods.
       - best_action: Greedy action selector breaking ties by lowest action index.
       - select_action: Epsilon-greedy action selector restricted to valid actions.
       - update: Standard Bellman TD update Q(s,a) <- Q(s,a) + alpha * [r + gamma*max Q(s',a') - Q].
       - decay_epsilon: Decays exploration probability by eps_decay down to eps_min.
       - reset_eval_tracking, unseen_state_rate: Evaluation tracking inspection helpers.
       - get_summary: Generates structured inspection summary for top visited states.
       - save, load: Model checkpoint persistence and restoration utilities.
    3. Technologies used and why:
       - Tabular sparse dictionary: Highly memory-efficient and fast for <= 3,200 discrete states.
       - joblib serialization: Fast and standardized model checkpointing matching scikit-learn.
    4. Key design and overall design of the class:
       Decoupled from simulation environment mechanics; operates purely on abstract integer states
       and valid action subsets, supporting reproducible learning and neutral fallback handling.
    """

    def __init__(
        self,
        alpha: float = config.ALPHA,
        gamma: float = config.GAMMA,
        eps_start: float = config.EPS_START,
        eps_min: float = config.EPS_MIN,
        eps_decay: float = config.EPS_DECAY,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        1. The use of function:
           Initializes a new QLearningAgent instance with configured hyperparameters.
        2. How does the specific function work:
           Validates hyperparameter ranges (alpha in (0, 1], gamma in [0, 1], eps_start in [0, 1],
           eps_min in [0, 1], eps_decay in (0, 1]). Initializes internal tracking dictionaries,
           exploration state, and evaluation counters.
        3. Concepts used in the function:
           Defensive parameter assertion, hyperparameter initialization, tabular storage setup.
        4. Places where this function is imported and used:
           Instantiated by training scripts, evaluation harnesses, and unit tests.
        5. Final output of the function, before and after:
           Before: Uninitialized QLearningAgent reference.
           After: Fully initialized QLearningAgent with empty Q-table and fresh counters.
        """
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not (0.0 < alpha <= 1.0)
        ):
            raise ValueError(f"alpha must be in (0.0, 1.0], got {alpha}.")
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, (int, float))
            or not (0.0 <= gamma <= 1.0)
        ):
            raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}.")
        if (
            isinstance(eps_start, bool)
            or not isinstance(eps_start, (int, float))
            or not (0.0 <= eps_start <= 1.0)
        ):
            raise ValueError(f"eps_start must be in [0.0, 1.0], got {eps_start}.")
        if (
            isinstance(eps_min, bool)
            or not isinstance(eps_min, (int, float))
            or not (0.0 <= eps_min <= 1.0)
        ):
            raise ValueError(f"eps_min must be in [0.0, 1.0], got {eps_min}.")
        if (
            isinstance(eps_decay, bool)
            or not isinstance(eps_decay, (int, float))
            or not (0.0 < eps_decay <= 1.0)
        ):
            raise ValueError(f"eps_decay must be in (0.0, 1.0], got {eps_decay}.")

        self.alpha: float = float(alpha)
        self.gamma: float = float(gamma)
        self.epsilon: float = float(eps_start)
        self.eps_min: float = float(eps_min)
        self.eps_decay: float = float(eps_decay)

        self._q_table: dict[tuple[RLState, int], float] = {}
        self._state_visits: dict[RLState, int] = {}
        self._action_visits: dict[tuple[RLState, int], int] = {}

        self._unseen_state_eval_count: int = 0
        self._total_eval_decisions: int = 0
        self.metadata: dict[str, Any] = metadata or {}

    def get_q(self, state: RLState, action: int) -> float:
        """
        1. The use of function:
           Returns the estimated Q-value for a given state-action pair.
        2. How does the specific function work:
           Queries internal _q_table mapping; returns stored float value or default 0.0 if unseen.
        3. Concepts used in the function:
           Sparse table query, default zero initialization.
        4. Places where this function is imported and used:
           Used by action selectors, Bellman updates, inspect tools, and test verifiers.
        5. Final output of the function, before and after:
           Before: Query state and action index.
           After: Returns estimated float Q-value.
        """
        return self._q_table.get((state, action), 0.0)

    def set_q(self, state: RLState, action: int, value: float) -> None:
        """
        1. The use of function:
           Assigns a new Q-value estimate to a specific state-action pair.
        2. How does the specific function work:
           Inserts or overwrites the float value in internal _q_table.
        3. Concepts used in the function:
           Q-value assignment, table mutation.
        4. Places where this function is imported and used:
           Called internally by update() and during test fixture preparation.
        5. Final output of the function, before and after:
           Before: Previous Q-value in table.
           After: Updated Q-value in table for (state, action).
        """
        self._q_table[(state, action)] = float(value)

    def best_action(
        self,
        state: RLState,
        valid_actions: Sequence[int],
        record_unseen: bool = False,
    ) -> int:
        """
        1. The use of function:
           Selects the greedy action maximizing estimated Q-value among valid actions.
        2. How does the specific function work:
           Validates valid_actions is non-empty. If state has never been visited during training,
           falls back to neutral ACTION_EARLIEST_ARRIVAL (if valid, else valid_actions[0]) and
           optionally increments unseen evaluation counter. If state was seen, computes Q-values
           for all valid actions and chooses argmax, breaking ties strictly by lowest action index.
        3. Concepts used in the function:
           Greedy action selection, neutral unseen-state fallback, deterministic tie-breaking.
        4. Places where this function is imported and used:
           Called during policy inference, validation evaluation, RLPolicy execution, and tests.
        5. Final output of the function, before and after:
           Before: Current RLState and valid action candidates.
           After: Returns integer action index representing optimal greedy decision.
        """
        if not valid_actions:
            raise ValueError("Cannot select action: valid_actions sequence is empty.")

        if record_unseen:
            self._total_eval_decisions += 1

        is_unseen = state not in self._state_visits
        if is_unseen:
            if record_unseen:
                self._unseen_state_eval_count += 1
            if ACTION_EARLIEST_ARRIVAL in valid_actions:
                return ACTION_EARLIEST_ARRIVAL
            return valid_actions[0]

        best_a = min(valid_actions, key=lambda a: (-self.get_q(state, a), a))
        return best_a

    def select_action(
        self,
        state: RLState,
        valid_actions: Sequence[int],
        rng: np.random.Generator,
        training: bool = True,
    ) -> int:
        """
        1. The use of function:
           Selects an action using epsilon-greedy exploration over valid actions.
        2. How does the specific function work:
           Validates valid_actions. If training is True and a random float is strictly less than
           self.epsilon, uniformly samples an action from valid_actions. Otherwise delegates
           to best_action() for greedy exploitation.
        3. Concepts used in the function:
           Epsilon-greedy exploration, exploration-exploitation trade-off, action masking.
        4. Places where this function is imported and used:
           Called on every step of training episode rollouts.
        5. Final output of the function, before and after:
           Before: State, valid actions, RNG, and training mode flag.
           After: Returns selected integer action index.
        """
        if not valid_actions:
            raise ValueError("Cannot select action: valid_actions sequence is empty.")

        if training and rng.random() < self.epsilon:
            return int(rng.choice(valid_actions))

        return self.best_action(state, valid_actions, record_unseen=not training)

    def update(
        self,
        state: RLState,
        action: int,
        reward: float,
        next_state: RLState,
        next_valid_actions: Sequence[int],
        done: bool,
    ) -> float:
        """
        1. The use of function:
           Performs a single tabular Q-learning Bellman temporal difference update.
        2. How does the specific function work:
           Computes target return: if done is True, target = reward; otherwise target =
           reward + gamma * max_{a' in next_valid_actions} Q(next_state, a'). Computes TD error
           = target - Q(state, action), updates Q(state, action) by alpha * TD error, increments
           state and action visit counters, and returns TD error.
        3. Concepts used in the function:
           Temporal difference learning, Bellman optimality equation, off-policy Q-learning.
        4. Places where this function is imported and used:
           Called after every transition step in training episode loops.
        5. Final output of the function, before and after:
           Before: Q-table in pre-transition state.
           After: Q-table updated with new estimate; returns float temporal difference error.
        """
        if done:
            target = float(reward)
        else:
            if next_valid_actions:
                max_next_q = max(self.get_q(next_state, a) for a in next_valid_actions)
            else:
                max_next_q = 0.0
            target = float(reward) + self.gamma * max_next_q

        current_q = self.get_q(state, action)
        td_error = target - current_q
        new_q = current_q + self.alpha * td_error

        self.set_q(state, action, new_q)
        self._state_visits[state] = self._state_visits.get(state, 0) + 1
        self._action_visits[(state, action)] = self._action_visits.get((state, action), 0) + 1

        return td_error

    def decay_epsilon(self) -> float:
        """
        1. The use of function:
           Decays the exploration probability epsilon after an episode.
        2. How does the specific function work:
           Multiplies self.epsilon by self.eps_decay, capping from below at self.eps_min.
        3. Concepts used in the function:
           Exponential exploration decay, exploration scheduling.
        4. Places where this function is imported and used:
           Called at the end of each training episode.
        5. Final output of the function, before and after:
           Before: Exploration rate epsilon prior to episode completion.
           After: Decayed epsilon value bounded by eps_min.
        """
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        return self.epsilon

    def reset_eval_tracking(self) -> None:
        """
        1. The use of function:
           Resets evaluation counters tracking unseen state occurrences and total decisions.
        2. How does the specific function work:
           Zeroes out _unseen_state_eval_count and _total_eval_decisions.
        3. Concepts used in the function:
           Evaluation metrics isolation, counter reset.
        4. Places where this function is imported and used:
           Called prior to running greedy evaluation batches over validation workloads.
        5. Final output of the function, before and after:
           Before: Counters in previous state.
           After: Both evaluation counters reset to 0.
        """
        self._unseen_state_eval_count = 0
        self._total_eval_decisions = 0

    @property
    def unseen_state_rate(self) -> float:
        """
        1. The use of function:
           Calculates the fraction of evaluation decisions that encountered an unseen state.
        2. How does the specific function work:
           Divides _unseen_state_eval_count by _total_eval_decisions if total > 0, else 0.0.
        3. Concepts used in the function:
           Generalization diagnostic, state coverage metric.
        4. Places where this function is imported and used:
           Queried during validation evaluation, convergence logging, and acceptance tests.
        5. Final output of the function, before and after:
           Before: Property accessed on agent.
           After: Returns float rate in [0.0, 1.0].
        """
        if self._total_eval_decisions == 0:
            return 0.0
        return self._unseen_state_eval_count / self._total_eval_decisions

    @property
    def q_table(self) -> dict[tuple[RLState, int], float]:
        """
        1. The use of function:
           Returns a copy of the internal Q-table dictionary.
        2. How does the specific function work:
           Returns shallow copy of _q_table.
        3. Concepts used in the function:
           Property accessor, encapsulation.
        4. Places where this function is imported and used:
           Used by visualizers, test inspectors, and summary generators.
        5. Final output of the function, before and after:
           Before: Property accessed.
           After: Returns dict of (RLState, int) -> float.
        """
        return dict(self._q_table)

    @property
    def state_visits(self) -> dict[RLState, int]:
        """
        1. The use of function:
           Returns a copy of the state visitation count dictionary.
        2. How does the specific function work:
           Returns shallow copy of _state_visits.
        3. Concepts used in the function:
           Property accessor, visitation distribution.
        4. Places where this function is imported and used:
           Used by summary builders and coverage analytics.
        5. Final output of the function, before and after:
           Before: Property accessed.
           After: Returns dict of RLState -> int.
        """
        return dict(self._state_visits)

    @property
    def q_table_size(self) -> int:
        """
        1. The use of function:
           Returns the total number of state-action entries stored in the Q-table.
        2. How does the specific function work:
           Returns len(self._q_table).
        3. Concepts used in the function:
           Table capacity, memory monitoring.
        4. Places where this function is imported and used:
           Used in training logs, validation checkpoints, and CLI summaries.
        5. Final output of the function, before and after:
           Before: Property accessed.
           After: Returns non-negative integer count of Q-entries.
        """
        return len(self._q_table)

    @property
    def unique_states_visited(self) -> int:
        """
        1. The use of function:
           Returns the total count of distinct RLStates visited during training.
        2. How does the specific function work:
           Returns len(self._state_visits).
        3. Concepts used in the function:
           State space coverage metric.
        4. Places where this function is imported and used:
           Used in diagnostic logging, convergence tracking, and acceptance tests.
        5. Final output of the function, before and after:
           Before: Property accessed.
           After: Returns non-negative integer count of visited states.
        """
        return len(self._state_visits)

    def get_summary(self, top_n: int = 500) -> dict[str, Any]:
        """
        1. The use of function:
           Builds a human-readable JSON-serializable summary of the Q-table.
        2. How does the specific function work:
           Sorts visited states by visit count in descending order. For up to top_n states,
           records state tuple string, visit count, optimal action, and per-action Q-values
           and visit counts. Packages top-level capacity metrics and metadata.
        3. Concepts used in the function:
           Model interpretability, policy inspection, structured JSON serialization.
        4. Places where this function is imported and used:
           Called when writing models/q_table_summary.json.
        5. Final output of the function, before and after:
           Before: Trained Q-table.
           After: Returns dictionary ready for json.dump().
        """
        sorted_states = sorted(
            self._state_visits.keys(),
            key=lambda s: self._state_visits[s],
            reverse=True,
        )

        state_records: list[dict[str, Any]] = []
        for s in sorted_states[:top_n]:
            best_a = self.best_action(s, ALL_ACTIONS)
            record = {
                "state": list(s),
                "state_str": str(s),
                "visits": self._state_visits[s],
                "best_action": best_a,
                "best_action_name": ACTION_NAMES[best_a],
                "q_values": {
                    ACTION_NAMES[a]: round(self.get_q(s, a), 4) for a in ALL_ACTIONS
                },
                "action_visits": {
                    ACTION_NAMES[a]: self._action_visits.get((s, a), 0) for a in ALL_ACTIONS
                },
            }
            state_records.append(record)

        return {
            "metadata": self.metadata,
            "q_table_size": self.q_table_size,
            "unique_states_visited": self.unique_states_visited,
            "top_states_count": len(state_records),
            "top_states": state_records,
        }

    def save(self, path: Path | str, metadata: dict[str, Any] | None = None) -> None:
        """
        1. The use of function:
           Persists the trained agent, Q-table, visit statistics, and metadata to disk.
        2. How does the specific function work:
           Combines internal state attributes and caller metadata into a dictionary.
           Creates parent directories if necessary and serializes using joblib.dump().
        3. Concepts used in the function:
           Model persistence, joblib serialization, directory assurance.
        4. Places where this function is imported and used:
           Called at the end of training runs and model checkpointing.
        5. Final output of the function, before and after:
           Before: In-memory agent instance.
           After: Serialized file written to target path on disk.
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        merged_meta = dict(self.metadata)
        if metadata:
            merged_meta.update(metadata)

        payload = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "eps_min": self.eps_min,
            "eps_decay": self.eps_decay,
            "q_table": self._q_table,
            "state_visits": self._state_visits,
            "action_visits": self._action_visits,
            "metadata": merged_meta,
        }
        joblib.dump(payload, target_path)

    @classmethod
    def load(cls, path: Path | str) -> "QLearningAgent":
        """
        1. The use of function:
           Loads a trained QLearningAgent from a saved joblib checkpoint file.
        2. How does the specific function work:
           Reads serialized payload via joblib.load(). Re-instantiates QLearningAgent with
           saved hyperparameters, restores internal Q-table and visit counts, and returns agent.
        3. Concepts used in the function:
           Model deserialization, factory class method, state restoration.
        4. Places where this function is imported and used:
           Called during policy inference, CLI simulation, evaluation, and REST API startup.
        5. Final output of the function, before and after:
           Before: Saved checkpoint file on disk.
           After: Returns fully restored QLearningAgent ready for evaluation or inference.
        """
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at: {target_path}")

        payload = joblib.load(target_path)
        agent = cls(
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            eps_start=payload["epsilon"],
            eps_min=payload["eps_min"],
            eps_decay=payload["eps_decay"],
            metadata=payload.get("metadata", {}),
        )
        agent._q_table = dict(payload["q_table"])
        agent._state_visits = dict(payload["state_visits"])
        agent._action_visits = dict(payload["action_visits"])
        return agent

    def __repr__(self) -> str:
        """
        1. The use of function:
           Returns a human-readable string representation of the QLearningAgent.
        2. How does the specific function work:
           Formats class name, hyperparameters, Q-table size, and unique visited states.
        3. Concepts used in the function:
           String representation, debugging utility.
        4. Places where this function is imported and used:
           Used in logging and debug outputs.
        5. Final output of the function, before and after:
           Before: QLearningAgent instance inspected.
           After: Returns formatted summary string.
        """
        return (
            f"QLearningAgent(alpha={self.alpha}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.4f}, q_entries={self.q_table_size}, "
            f"unique_states={self.unique_states_visited})"
        )


__all__ = ["QLearningAgent"]
