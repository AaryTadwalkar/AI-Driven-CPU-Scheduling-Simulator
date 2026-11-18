"""
1. What this file does:
   Defines the discrete rule-based action space and action dispatch mappings for the reinforcement
   learning scheduling agent.
2. Overall design:
   Provides 4 high-level scheduling heuristic actions:
   - 0: SHORTEST_REMAINING (preempts toward the shortest remaining burst; running process wins ties)
   - 1: LONGEST_REMAINING (favors compute-heavy jobs with running process tie advantage)
   - 2: EARLIEST_ARRIVAL (dispatches the earliest arrived ready process; pure FCFS ordering)
   - 3: CONTINUE_CURRENT (maintains execution continuity of the currently running process; masked
        when the last process completed or was idle).
   Implements action masking and maps abstract action indices into concrete process PIDs using
   the canonical tie-breaking protocol defined in Section 4.4.
3. Elements in the file:
   - ACTION_SHORTEST_REMAINING: Integer constant 0.
   - ACTION_LONGEST_REMAINING: Integer constant 1.
   - ACTION_EARLIEST_ARRIVAL: Integer constant 2.
   - ACTION_CONTINUE_CURRENT: Integer constant 3.
   - ALL_ACTIONS: Canonical tuple of all 4 integer action indices.
   - ACTION_NAMES: Dictionary mapping action indices to human-readable names.
   - get_valid_actions: Computes list of unmasked action indices for a given View.
   - action_to_pid: Maps an action index to a chosen ready process PID string.
4. Final output of the file:
   Exports action constants and resolution functions for RL environment, agent, and evaluation.
"""

from schedsim.domain.engine import View

ACTION_SHORTEST_REMAINING: int = 0
ACTION_LONGEST_REMAINING: int = 1
ACTION_EARLIEST_ARRIVAL: int = 2
ACTION_CONTINUE_CURRENT: int = 3

ALL_ACTIONS: tuple[int, ...] = (
    ACTION_SHORTEST_REMAINING,
    ACTION_LONGEST_REMAINING,
    ACTION_EARLIEST_ARRIVAL,
    ACTION_CONTINUE_CURRENT,
)

ACTION_NAMES: dict[int, str] = {
    ACTION_SHORTEST_REMAINING: "SHORTEST_REMAINING",
    ACTION_LONGEST_REMAINING: "LONGEST_REMAINING",
    ACTION_EARLIEST_ARRIVAL: "EARLIEST_ARRIVAL",
    ACTION_CONTINUE_CURRENT: "CONTINUE_CURRENT",
}


def get_valid_actions(view: View) -> list[int]:
    """
    1. The use of function:
       Computes the list of valid, unmasked action indices available in the current View.
    2. How does the specific function work:
       Checks if `view.ready` is non-empty. If empty, returns an empty list. Otherwise, includes
       actions 0 (SHORTEST_REMAINING), 1 (LONGEST_REMAINING), and 2 (EARLIEST_ARRIVAL). Action 3
       (CONTINUE_CURRENT) is appended if and only if `view.last_pid` is not None and matches a
       process currently present in `view.ready`.
    3. Concepts used in the function:
       Action masking, invalid action pruning, MDP transition constraints.
    4. Places where this function is imported and used:
       Called by Environment.valid_actions(), Q-learning action selection, and test suites.
    5. Final output of the function, before and after:
       Before: Current simulation View.
       After: List of integer action indices [0, 1, 2] or [0, 1, 2, 3].
    """
    if not isinstance(view, View):
        raise TypeError(f"view must be a View instance, got {type(view).__name__}.")

    if not view.ready:
        return []

    valid = [
        ACTION_SHORTEST_REMAINING,
        ACTION_LONGEST_REMAINING,
        ACTION_EARLIEST_ARRIVAL,
    ]

    if view.last_pid is not None and any(p.pid == view.last_pid for p in view.ready):
        valid.append(ACTION_CONTINUE_CURRENT)

    return valid


def action_to_pid(
    view: View,
    action: int,
    fallback_to_earliest: bool = False,
) -> str:
    """
    1. The use of function:
       Maps an abstract rule-based action index into a concrete ready process PID.
    2. How does the specific function work:
       Validates that view.ready is non-empty and action is an int.
       - Action 0: returns pid with min (remaining, running_first, arrival, pid).
       - Action 1: returns pid with min (-remaining, running_first, arrival, pid).
       - Action 2: returns pid with min (arrival, pid).
       - Action 3: returns view.last_pid if ready; if masked and fallback_to_earliest is True,
         delegates to Action 2, otherwise raises ValueError.
    3. Concepts used in the function:
       Heuristic action dispatch, canonical tie-breaking, continuity preservation.
    4. Places where this function is imported and used:
       Called by Environment.step(), scripted baseline agents, and test suites.
    5. Final output of the function, before and after:
       Before: View and action index.
       After: Selected process PID string or raises ValueError on empty queue or masked action.
    """
    if not isinstance(view, View):
        raise TypeError(f"view must be a View instance, got {type(view).__name__}.")
    if isinstance(action, bool) or not isinstance(action, int):
        raise TypeError(f"action must be an int, got {type(action).__name__}.")
    if not view.ready:
        raise ValueError("Cannot select process: ready queue is empty.")

    if action == ACTION_SHORTEST_REMAINING:
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

    elif action == ACTION_LONGEST_REMAINING:
        best = min(
            view.ready,
            key=lambda p: (
                -p.remaining,
                0 if p.pid == view.last_pid else 1,
                p.arrival,
                p.pid,
            ),
        )
        return best.pid

    elif action == ACTION_EARLIEST_ARRIVAL:
        best = min(
            view.ready,
            key=lambda p: (
                p.arrival,
                p.pid,
            ),
        )
        return best.pid

    elif action == ACTION_CONTINUE_CURRENT:
        if view.last_pid is not None and any(p.pid == view.last_pid for p in view.ready):
            return view.last_pid
        if fallback_to_earliest:
            return action_to_pid(view, ACTION_EARLIEST_ARRIVAL)
        raise ValueError(
            f"Action CONTINUE_CURRENT is masked: last_pid '{view.last_pid}' is not ready."
        )

    else:
        raise ValueError(
            f"Unknown action {action}. Valid actions are {ALL_ACTIONS}."
        )


__all__ = [
    "ACTION_CONTINUE_CURRENT",
    "ACTION_EARLIEST_ARRIVAL",
    "ACTION_LONGEST_REMAINING",
    "ACTION_NAMES",
    "ACTION_SHORTEST_REMAINING",
    "ALL_ACTIONS",
    "action_to_pid",
    "get_valid_actions",
]
