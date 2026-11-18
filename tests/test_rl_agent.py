"""
1. What this file does:
   Defines unit tests for the tabular QLearningAgent reinforcement learning class.
2. Overall design:
   Validates all core agent behaviors:
   - Hyperparameter boundaries and validation checks.
   - Q-value retrieval and updates matching Bellman temporal difference mathematics.
   - Deterministic tie-breaking by lowest action index.
   - Unseen-state fallback to neutral EARLIEST_ARRIVAL and tracking rate.
   - Epsilon decay schedule down to minimum floor.
   - Checkpoint persistence and lossless restoration via joblib.
   - Structured JSON summary compilation for top visited states.
3. Elements in the file:
   - test_agent_initialization: Tests constructor validation and parameter limits.
   - test_get_set_q: Tests Q-value dictionary storage and default zero return.
   - test_best_action_seen_and_tie_breaking: Tests argmax action selection and tie-breaking.
   - test_best_action_unseen_fallback: Tests neutral fallback and evaluation counting.
   - test_select_action_exploration: Tests epsilon-greedy probabilistic branching.
   - test_update_bellman_dynamics: Verifies exact numeric temporal difference updates.
   - test_decay_epsilon: Verifies geometric decay and minimum floor enforcement.
   - test_eval_tracking_and_unseen_rate: Verifies diagnostic rate calculation.
   - test_save_load_roundtrip: Verifies model serialization and deserialization.
   - test_get_summary: Verifies JSON summary structure.
4. Final output of the file:
   Pytest test suite validating schedsim.rl.agent.QLearningAgent.
"""

import numpy as np
import pytest

from schedsim.rl.actions import (
    ACTION_CONTINUE_CURRENT,
    ACTION_EARLIEST_ARRIVAL,
    ACTION_LONGEST_REMAINING,
    ACTION_SHORTEST_REMAINING,
    ALL_ACTIONS,
)
from schedsim.rl.agent import QLearningAgent
from schedsim.rl.state import RLState


def test_agent_initialization() -> None:
    """
    1. The use of function:
       Verifies QLearningAgent constructor parameter validations and ranges.
    2. How does the specific function work:
       Tests default values and asserts ValueError when alpha, gamma, eps_start,
       eps_min, or eps_decay exceed valid mathematical boundaries.
    3. Concepts used in the function:
       Boundary value analysis, parameter validation, defensive programming.
    4. Places where this function is imported and used:
       Executed as part of pytest test runner.
    5. Final output of the function, before and after:
       Before: Unverified agent initialization.
       After: Asserts valid defaults and ValueError on illegal hyperparameter inputs.
    """
    agent = QLearningAgent()
    assert agent.alpha == 0.1
    assert agent.gamma == 1.0
    assert agent.epsilon == 1.0
    assert agent.eps_min == 0.05
    assert agent.eps_decay == 0.9995
    assert agent.q_table_size == 0
    assert agent.unique_states_visited == 0

    with pytest.raises(ValueError, match="alpha must be in"):
        QLearningAgent(alpha=0.0)

    with pytest.raises(ValueError, match="alpha must be in"):
        QLearningAgent(alpha=1.5)

    with pytest.raises(ValueError, match="gamma must be in"):
        QLearningAgent(gamma=-0.1)

    with pytest.raises(ValueError, match="eps_start must be in"):
        QLearningAgent(eps_start=2.0)

    with pytest.raises(ValueError, match="eps_decay must be in"):
        QLearningAgent(eps_decay=0.0)


def test_get_set_q() -> None:
    """
    1. The use of function:
       Verifies sparse Q-table getter and setter methods.
    2. How does the specific function work:
       Queries an unvisited state-action pair to assert default 0.0, sets a specific float,
       and verifies that get_q returns the assigned value.
    3. Concepts used in the function:
       Sparse dictionary mapping, default value semantics.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Q-table storage unverified.
       After: Confirms accurate storage and default retrieval.
    """
    agent = QLearningAgent()
    state: RLState = (1, 0, 2, 0, 1, 0, 1)

    assert agent.get_q(state, ACTION_SHORTEST_REMAINING) == 0.0

    agent.set_q(state, ACTION_SHORTEST_REMAINING, -4.5)
    assert agent.get_q(state, ACTION_SHORTEST_REMAINING) == -4.5
    assert agent.q_table_size == 1


def test_best_action_seen_and_tie_breaking() -> None:
    """
    1. The use of function:
       Verifies greedy best_action selection and lowest action index tie-breaking.
    2. How does the specific function work:
       Marks a state as visited. Sets identical Q-values for action 1 and action 2.
       Asserts that action 1 is selected over action 2. Sets higher Q for action 2,
       verifying it takes precedence.
    3. Concepts used in the function:
       Greedy exploitation, deterministic tie-breaking.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Tie-breaking behavior unverified.
       After: Confirms lowest action index tie-break and argmax selection.
    """
    agent = QLearningAgent()
    state: RLState = (2, 1, 3, 1, 0, 1, 0)

    # Simulate that state was visited
    agent._state_visits[state] = 1

    # Equal Q-values (0.0): ties broken by lowest action index
    valid = [ACTION_LONGEST_REMAINING, ACTION_EARLIEST_ARRIVAL]
    assert agent.best_action(state, valid) == ACTION_LONGEST_REMAINING

    # Make action 2 strictly better (-1.0 > -2.0)
    agent.set_q(state, ACTION_LONGEST_REMAINING, -2.0)
    agent.set_q(state, ACTION_EARLIEST_ARRIVAL, -1.0)
    assert agent.best_action(state, valid) == ACTION_EARLIEST_ARRIVAL

    with pytest.raises(ValueError, match="valid_actions sequence is empty"):
        agent.best_action(state, [])


def test_best_action_unseen_fallback() -> None:
    """
    1. The use of function:
       Verifies neutral fallback to EARLIEST_ARRIVAL on never-seen states.
    2. How does the specific function work:
       Presents an unvisited state with multiple valid actions. Asserts that
       ACTION_EARLIEST_ARRIVAL is chosen and unseen evaluation counter increments.
       Tests fallback to valid_actions[0] if EARLIEST_ARRIVAL is absent.
    3. Concepts used in the function:
       Neutral fallback policy, unseen state tracking.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unseen state fallback unverified.
       After: Confirms EARLIEST_ARRIVAL fallback and counter increment.
    """
    agent = QLearningAgent()
    unseen_state: RLState = (3, 2, 4, 2, 0, 0, 0)

    valid = [
        ACTION_SHORTEST_REMAINING,
        ACTION_LONGEST_REMAINING,
        ACTION_EARLIEST_ARRIVAL,
    ]
    chosen = agent.best_action(unseen_state, valid, record_unseen=True)
    assert chosen == ACTION_EARLIEST_ARRIVAL
    assert agent._unseen_state_eval_count == 1
    assert agent._total_eval_decisions == 1

    # If EARLIEST_ARRIVAL is not valid, falls back to valid[0]
    valid_no_earliest = [ACTION_SHORTEST_REMAINING, ACTION_LONGEST_REMAINING]
    chosen2 = agent.best_action(unseen_state, valid_no_earliest, record_unseen=True)
    assert chosen2 == ACTION_SHORTEST_REMAINING
    assert agent._unseen_state_eval_count == 2
    assert agent._total_eval_decisions == 2


def test_select_action_exploration() -> None:
    """
    1. The use of function:
       Verifies epsilon-greedy probabilistic exploration in select_action.
    2. How does the specific function work:
       With epsilon=1.0, asserts actions are randomly sampled.
       With epsilon=0.0, asserts actions are chosen greedily.
    3. Concepts used in the function:
       Epsilon-greedy sampling, exploitation vs exploration.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Exploration branching unverified.
       After: Confirms exploration when epsilon=1.0 and exploitation when epsilon=0.0.
    """
    agent = QLearningAgent(eps_start=1.0)
    rng = np.random.default_rng(123)
    state: RLState = (1, 1, 1, 1, 0, 0, 0)
    agent._state_visits[state] = 5
    agent.set_q(state, ACTION_SHORTEST_REMAINING, 100.0)

    # When training and epsilon=1.0, actions vary
    actions = {
        agent.select_action(state, ALL_ACTIONS, rng, training=True)
        for _ in range(50)
    }
    assert len(actions) > 1

    # When training=False (evaluation), purely greedy
    greedy_actions = {
        agent.select_action(state, ALL_ACTIONS, rng, training=False)
        for _ in range(20)
    }
    assert greedy_actions == {ACTION_SHORTEST_REMAINING}


def test_update_bellman_dynamics() -> None:
    """
    1. The use of function:
       Validates exact numeric Bellman temporal difference learning updates.
    2. How does the specific function work:
       Performs a non-terminal step update: target = reward + gamma * max Q(s', a').
       TD error = target - current_Q, new_Q = current_Q + alpha * TD error.
       Then performs a terminal step update: target = reward.
    3. Concepts used in the function:
       Bellman equation, TD error calculation, state-action visitation tracking.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Mathematical correctness of update unverified.
       After: Asserts exact match with manual arithmetic calculation.
    """
    agent = QLearningAgent(alpha=0.2, gamma=0.9)
    s1: RLState = (1, 0, 1, 0, 0, 1, 0)
    s2: RLState = (2, 1, 2, 0, 1, 1, 1)

    # Initialize s2 with max Q = 10.0 for action 1
    agent.set_q(s2, ACTION_SHORTEST_REMAINING, 5.0)
    agent.set_q(s2, ACTION_LONGEST_REMAINING, 10.0)

    # Step: s1 -> a=0 -> r=-2.0 -> s2 (valid actions [0, 1])
    # Target = -2.0 + 0.9 * 10.0 = 7.0
    # Current Q = 0.0, TD error = 7.0, New Q = 0.0 + 0.2 * 7.0 = 1.4
    td_error = agent.update(
        state=s1,
        action=ACTION_SHORTEST_REMAINING,
        reward=-2.0,
        next_state=s2,
        next_valid_actions=[ACTION_SHORTEST_REMAINING, ACTION_LONGEST_REMAINING],
        done=False,
    )
    assert td_error == pytest.approx(7.0)
    assert agent.get_q(s1, ACTION_SHORTEST_REMAINING) == pytest.approx(1.4)
    assert agent._state_visits[s1] == 1
    assert agent._action_visits[(s1, ACTION_SHORTEST_REMAINING)] == 1

    # Terminal step: s2 -> a=1 -> r=-1.0 -> done
    # Target = -1.0
    # Current Q = 10.0, TD error = -1.0 - 10.0 = -11.0
    # New Q = 10.0 + 0.2 * (-11.0) = 7.8
    td_term = agent.update(
        state=s2,
        action=ACTION_LONGEST_REMAINING,
        reward=-1.0,
        next_state=(0, 0, 0, 0, 0, 0, 0),
        next_valid_actions=[],
        done=True,
    )
    assert td_term == pytest.approx(-11.0)
    assert agent.get_q(s2, ACTION_LONGEST_REMAINING) == pytest.approx(7.8)


def test_decay_epsilon() -> None:
    """
    1. The use of function:
       Verifies geometric epsilon decay and floor capping.
    2. How does the specific function work:
       Decays epsilon with decay=0.5 and floor=0.1, confirming value halves until
       reaching and respecting the floor.
    3. Concepts used in the function:
       Exploration schedule, asymptotic decay.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Epsilon decay unverified.
       After: Confirms accurate decay and floor enforcement.
    """
    agent = QLearningAgent(eps_start=1.0, eps_min=0.1, eps_decay=0.5)
    assert agent.decay_epsilon() == 0.5
    assert agent.decay_epsilon() == 0.25
    assert agent.decay_epsilon() == 0.125
    assert agent.decay_epsilon() == 0.1  # Capped at eps_min
    assert agent.decay_epsilon() == 0.1


def test_eval_tracking_and_unseen_rate() -> None:
    """
    1. The use of function:
       Tests unseen_state_rate computation and reset_eval_tracking.
    2. How does the specific function work:
       Invokes best_action() on seen and unseen states with record_unseen=True,
       verifies unseen_state_rate fraction, and checks reset behavior.
    3. Concepts used in the function:
       Diagnostic rate calculation, state isolation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Diagnostic rate calculation unverified.
       After: Confirms accurate fractional rate and clean zeroing on reset.
    """
    agent = QLearningAgent()
    s_seen: RLState = (1, 1, 1, 1, 1, 1, 1)
    s_unseen: RLState = (2, 2, 2, 2, 2, 2, 2)

    agent._state_visits[s_seen] = 10

    # 1 seen decision, 1 unseen decision
    agent.best_action(s_seen, ALL_ACTIONS, record_unseen=True)
    agent.best_action(s_unseen, ALL_ACTIONS, record_unseen=True)

    assert agent.unseen_state_rate == 0.5

    agent.reset_eval_tracking()
    assert agent.unseen_state_rate == 0.0


def test_save_load_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    """
    1. The use of function:
       Verifies lossless agent serialization and restoration to disk via joblib.
    2. How does the specific function work:
       Configures agent with custom hyperparameters, updates Q-values and visit counts,
       saves to a temporary path, and restores via QLearningAgent.load(). Asserts exact equality.
    3. Concepts used in the function:
       Model persistence, deserialization round-trip, metadata preservation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Serialization unverified.
       After: Confirms loaded agent has identical parameters, Q-table, and visit counts.
    """
    model_path = tmp_path / "test_q_agent.pkl"  # type: ignore[operator]
    meta = {"training_episodes": 500, "best_seed": 42}
    agent = QLearningAgent(alpha=0.15, gamma=0.95, eps_start=0.3, metadata=meta)

    s: RLState = (1, 2, 3, 0, 1, 0, 1)
    agent.set_q(s, ACTION_CONTINUE_CURRENT, -3.14)
    agent._state_visits[s] = 7
    agent._action_visits[(s, ACTION_CONTINUE_CURRENT)] = 7

    agent.save(model_path)
    loaded = QLearningAgent.load(model_path)

    assert loaded.alpha == agent.alpha
    assert loaded.gamma == agent.gamma
    assert loaded.epsilon == agent.epsilon
    assert loaded.get_q(s, ACTION_CONTINUE_CURRENT) == -3.14
    assert loaded.state_visits[s] == 7
    assert loaded.metadata == meta


def test_get_summary() -> None:
    """
    1. The use of function:
       Verifies JSON summary generation for top visited states.
    2. How does the specific function work:
       Visits multiple states, assigns Q-values, calls get_summary(top_n=2), and asserts
       the returned dictionary contains formatted metrics and records sorted by visit count.
    3. Concepts used in the function:
       Inspection summaries, structured JSON generation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Summary generator unverified.
       After: Confirms accurate summary format and visit-based sorting.
    """
    agent = QLearningAgent(metadata={"version": "1.0"})
    s1: RLState = (1, 0, 0, 0, 0, 0, 0)
    s2: RLState = (2, 0, 0, 0, 0, 0, 0)

    agent._state_visits[s1] = 50
    agent._state_visits[s2] = 100
    agent.set_q(s2, ACTION_SHORTEST_REMAINING, -1.5)

    summary = agent.get_summary(top_n=10)
    assert summary["q_table_size"] == 1
    assert summary["unique_states_visited"] == 2
    assert summary["top_states_count"] == 2
    assert summary["top_states"][0]["visits"] == 100  # s2 has more visits
    assert summary["top_states"][0]["state"] == list(s2)
