"""
1. What this file does:
   Initializes the reinforcement learning package for tabular Q-learning scheduling.
2. Overall design:
   Exposes the core MDP formulation symbols, agent, policy adapter, and training routines:
   - Rule-based discrete action definitions and dispatch resolvers.
   - Permutation-invariant 7-tuple aggregate state representation and bucket quantization.
   - Standardized gym-like simulation Environment wrapping the unit-step Engine.
   - Tabular QLearningAgent with epsilon decay and summary serialization.
   - RLPolicy adapter allowing trained agents to execute inside Engine.run().
   - Training loops, validation evaluators, hyperparameter search, and figure plotters.
3. Elements in the file:
   - Action constants: ACTION_SHORTEST_REMAINING, ACTION_LONGEST_REMAINING,
     ACTION_EARLIEST_ARRIVAL, ACTION_CONTINUE_CURRENT, ALL_ACTIONS, ACTION_NAMES.
   - Action helpers: get_valid_actions, action_to_pid.
   - State definitions: RLState, STATE_SHAPE, TOTAL_POSSIBLE_STATES, TERMINAL_STATE.
   - State helpers: digitize_value, encode_state.
   - Environment: MDP environment wrapping Engine.
   - QLearningAgent: Core tabular Q-learning agent.
   - RLPolicy: Policy wrapper executing trained Q-learning policy.
   - Training & evaluation routines: evaluate_on_validation, train_q_learning,
     run_hparam_search, train_final_agents, evaluate_agent_on_workloads,
     plot_learning_curves, plot_action_distribution.
4. Final output of the file:
   Enables high-level imports from schedsim.rl for agents, training loops, and test suites.
"""

from schedsim.rl.actions import (
    ACTION_CONTINUE_CURRENT,
    ACTION_EARLIEST_ARRIVAL,
    ACTION_LONGEST_REMAINING,
    ACTION_NAMES,
    ACTION_SHORTEST_REMAINING,
    ALL_ACTIONS,
    action_to_pid,
    get_valid_actions,
)
from schedsim.rl.agent import QLearningAgent
from schedsim.rl.env import Environment
from schedsim.rl.evaluate import (
    evaluate_agent_on_workloads,
    plot_action_distribution,
    plot_learning_curves,
)
from schedsim.rl.policy import RLPolicy
from schedsim.rl.state import (
    STATE_SHAPE,
    TERMINAL_STATE,
    TOTAL_POSSIBLE_STATES,
    RLState,
    digitize_value,
    encode_state,
)
from schedsim.rl.train import (
    evaluate_on_validation,
    run_hparam_search,
    train_final_agents,
    train_q_learning,
)

__all__ = [
    "ACTION_CONTINUE_CURRENT",
    "ACTION_EARLIEST_ARRIVAL",
    "ACTION_LONGEST_REMAINING",
    "ACTION_NAMES",
    "ACTION_SHORTEST_REMAINING",
    "ALL_ACTIONS",
    "Environment",
    "QLearningAgent",
    "RLPolicy",
    "RLState",
    "STATE_SHAPE",
    "TERMINAL_STATE",
    "TOTAL_POSSIBLE_STATES",
    "action_to_pid",
    "digitize_value",
    "encode_state",
    "evaluate_agent_on_workloads",
    "evaluate_on_validation",
    "get_valid_actions",
    "plot_action_distribution",
    "plot_learning_curves",
    "run_hparam_search",
    "train_final_agents",
    "train_q_learning",
]
