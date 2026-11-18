"""
1. What this file does:
   Defines the permutation-invariant, compact reinforcement learning state space representation
   for the CPU scheduling MDP.
2. Overall design:
   Transforms observable simulation View snapshots into an immutable 7-tuple of small integers
   representing discretized queue characteristics:
   - ready count (4 buckets)
   - shortest remaining burst among ready (5 buckets)
   - longest remaining burst among ready (5 buckets)
   - max waited ticks among ready (4 buckets)
   - last process still ready flag (0/1)
   - more arrivals pending flag (0/1)
   - is last process the shortest remaining (0/1).
   Eliminates all process IDs and absolute timestamps to ensure permutation invariance and
   compact tabular state capacity (at most 3,200 raw states).
3. Elements in the file:
   - RLState: Type alias for the 7-element state tuple.
   - STATE_SHAPE: Canonical dimensions tuple (4, 5, 5, 4, 2, 2, 2).
   - TOTAL_POSSIBLE_STATES: Integer constant 3200.
   - TERMINAL_STATE: Dedicated terminal sentinel state tuple (0, 0, 0, 0, 0, 0, 0).
   - digitize_value: Discretizes an integer value into a zero-indexed bucket given upper bounds.
   - encode_state: Extracts and discretizes view features into the canonical 7-tuple state.
4. Final output of the file:
   Exports state encoding functions and constants for the RL environment, Q-table, and agent.
"""

from typing import Tuple

import config
from schedsim.domain.engine import View

RLState = Tuple[int, int, int, int, int, int, int]

STATE_SHAPE: tuple[int, ...] = (4, 5, 5, 4, 2, 2, 2)
TOTAL_POSSIBLE_STATES: int = 4 * 5 * 5 * 4 * 2 * 2 * 2  # 3200
TERMINAL_STATE: RLState = (0, 0, 0, 0, 0, 0, 0)


def digitize_value(val: int, edges: tuple[int, ...]) -> int:
    """
    1. The use of function:
       Maps an integer value to a zero-indexed discrete bucket index using upper-bound edges.
    2. How does the specific function work:
       Iterates over the sorted sequence of edge thresholds. Returns the index of the first edge
       where val <= edge. If val exceeds all thresholds, returns the final bucket index len(edges).
    3. Concepts used in the function:
       Bucket discretization, binning, scalar quantization.
    4. Places where this function is imported and used:
       Called internally by `encode_state` to discretize queue lengths, bursts, and waited times.
    5. Final output of the function, before and after:
       Before: Raw continuous or integer metric value and edge tuple.
       After: Integer bucket index in [0, len(edges)].
    """
    for idx, edge in enumerate(edges):
        if val <= edge:
            return idx
    return len(edges)


def encode_state(
    view: View,
    more_arrivals_pending: bool = False,
) -> RLState:
    """
    1. The use of function:
       Extracts and discretizes observable queue features into a permutation-invariant state tuple.
    2. How does the specific function work:
       Validates that view.ready is non-empty. Computes ready count, min/max remaining burst,
       and max waited time among ready processes. Digitizes each feature against bucket edges
       from config.py. Determines whether the last dispatched process is still ready, flags
       pending arrivals, and checks if the last process holds the shortest remaining burst.
    3. Concepts used in the function:
       Markovian state construction, permutation invariance, state space compression.
    4. Places where this function is imported and used:
       Called by Environment.reset(), Environment.step(), Q-learning agent, and test suites.
    5. Final output of the function, before and after:
       Before: Observable View object and boolean arrivals flag.
       After: Immutable 7-tuple of small integers representing the compact MDP state.
    """
    if not isinstance(view, View):
        raise TypeError(f"view must be a View instance, got {type(view).__name__}.")
    if not view.ready:
        raise ValueError("Cannot encode state: ready queue is empty.")

    ready_count = len(view.ready)
    remaining_bursts = [p.remaining for p in view.ready]
    waited_times = [p.waited for p in view.ready]

    min_remaining = min(remaining_bursts)
    max_remaining = max(remaining_bursts)
    max_waited = max(waited_times)

    # 1. Ready count (4 buckets: 1, 2, 3, 4+)
    b_ready_count = digitize_value(ready_count, config.STATE_READY_COUNT_EDGES)

    # 2. Shortest remaining among ready (5 buckets: 1, 2, 3-4, 5-8, 9+)
    b_shortest_rem = digitize_value(
        min_remaining, config.STATE_SHORTEST_REMAINING_EDGES
    )

    # 3. Longest remaining among ready (5 buckets: 1-2, 3-4, 5-8, 9-16, 17+)
    b_longest_rem = digitize_value(
        max_remaining, config.STATE_LONGEST_REMAINING_EDGES
    )

    # 4. Max waited among ready (4 buckets: 0, 1-2, 3-5, 6+)
    b_max_waited = digitize_value(max_waited, config.STATE_MAX_WAITED_EDGES)

    # 5. Last process still ready? (0 or 1)
    last_is_ready = int(
        view.last_pid is not None
        and any(p.pid == view.last_pid for p in view.ready)
    )

    # 6. More arrivals pending? (0 or 1)
    b_more_arrivals = int(bool(more_arrivals_pending))

    # 7. Is last process shortest-remaining? (0 or 1, only if ready)
    if last_is_ready:
        last_proc = next(p for p in view.ready if p.pid == view.last_pid)
        is_last_shortest = int(last_proc.remaining == min_remaining)
    else:
        is_last_shortest = 0

    state: RLState = (
        b_ready_count,
        b_shortest_rem,
        b_longest_rem,
        b_max_waited,
        last_is_ready,
        b_more_arrivals,
        is_last_shortest,
    )
    return state


__all__ = [
    "RLState",
    "STATE_SHAPE",
    "TERMINAL_STATE",
    "TOTAL_POSSIBLE_STATES",
    "digitize_value",
    "encode_state",
]
