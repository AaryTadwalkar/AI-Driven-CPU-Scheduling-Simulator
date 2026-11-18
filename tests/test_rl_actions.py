"""
1. What this file does:
   Tests the reinforcement learning action space, validity masking, and heuristic PID resolution
   routines.
2. Overall design:
   Validates `get_valid_actions` and `action_to_pid` in `schedsim.rl.actions` using constructed
   `View` and `ProcView` fixtures. Asserts canonical tie-breaking order, running process
   continuity, masking constraints, and exception handling.
3. Elements in the file:
   - test_action_constants: Validates all action indices and name mappings.
   - test_get_valid_actions_masking: Verifies action 3 is offered iff last_pid is ready.
   - test_action_shortest_remaining: Verifies action 0 selects shortest remaining and breaks ties.
   - test_action_longest_remaining: Verifies action 1 selects longest remaining and breaks ties.
   - test_action_earliest_arrival: Verifies action 2 selects earliest arrival and breaks ties.
   - test_action_continue_current: Verifies action 3 dispatches last_pid and supports fallback.
   - test_action_to_pid_validation_errors: Verifies error handling on empty queue or bad inputs.
4. Final output of the file:
   Pytest test suite validating RL action mechanics and dispatch rules.
"""

import pytest

from schedsim.domain.engine import ProcView, View
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


def test_action_constants() -> None:
    """
    1. The use of function:
       Validates canonical action constants, tuples, and string mappings.
    2. How does the specific function work:
       Asserts ALL_ACTIONS has length 4 and ACTION_NAMES maps 0..3 to uppercase names.
    3. Concepts used in the function:
       Constant verification, contract adherence.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified action constants.
       After: Asserts exact presence and mapping of all 4 actions.
    """
    assert ALL_ACTIONS == (0, 1, 2, 3)
    assert ACTION_SHORTEST_REMAINING == 0
    assert ACTION_LONGEST_REMAINING == 1
    assert ACTION_EARLIEST_ARRIVAL == 2
    assert ACTION_CONTINUE_CURRENT == 3

    assert ACTION_NAMES[0] == "SHORTEST_REMAINING"
    assert ACTION_NAMES[1] == "LONGEST_REMAINING"
    assert ACTION_NAMES[2] == "EARLIEST_ARRIVAL"
    assert ACTION_NAMES[3] == "CONTINUE_CURRENT"


def test_get_valid_actions_masking() -> None:
    """
    1. The use of function:
       Tests action validity masking depending on whether last_pid is ready.
    2. How does the specific function work:
       Constructs Views where last_pid is None, finished, ready, or ready queue is empty.
       Asserts [0, 1, 2] vs [0, 1, 2, 3] vs [].
    3. Concepts used in the function:
       Action masking, invalid action pruning.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested action masking logic.
       After: Asserts action 3 is offered iff last_pid is ready.
    """
    p1 = ProcView(pid="P1", arrival=0, burst=5, remaining=3, waited=0)
    p2 = ProcView(pid="P2", arrival=1, burst=4, remaining=4, waited=1)

    # 1. last_pid is None -> only [0, 1, 2]
    v_none = View(time=0, ready=(p1, p2), last_pid=None, newly_arrived=())
    assert get_valid_actions(v_none) == [0, 1, 2]

    # 2. last_pid is P1 (ready) -> [0, 1, 2, 3]
    v_p1 = View(time=1, ready=(p1, p2), last_pid="P1", newly_arrived=())
    assert get_valid_actions(v_p1) == [0, 1, 2, 3]

    # 3. last_pid is P3 (not in ready) -> only [0, 1, 2]
    v_p3 = View(time=2, ready=(p1, p2), last_pid="P3", newly_arrived=())
    assert get_valid_actions(v_p3) == [0, 1, 2]

    # 4. ready is empty -> []
    v_empty = View(time=3, ready=(), last_pid="P1", newly_arrived=())
    assert get_valid_actions(v_empty) == []


def test_action_shortest_remaining() -> None:
    """
    1. The use of function:
       Tests action 0 selecting process with shortest remaining time and tie-breaking.
    2. How does the specific function work:
       Tests strictly shortest remaining, ties broken by running process, then arrival, then PID.
    3. Concepts used in the function:
       Heuristic action resolution, tie-breaking protocol.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested action 0 dispatch.
       After: Asserts correct PID selected across all tie scenarios.
    """
    p1 = ProcView(pid="P1", arrival=0, burst=5, remaining=3, waited=1)
    p2 = ProcView(pid="P2", arrival=1, burst=4, remaining=2, waited=0)
    v1 = View(time=2, ready=(p1, p2), last_pid=None, newly_arrived=())
    assert action_to_pid(v1, ACTION_SHORTEST_REMAINING) == "P2"

    # Tie on remaining: running process wins
    p3 = ProcView(pid="P3", arrival=0, burst=3, remaining=2, waited=1)
    p4 = ProcView(pid="P4", arrival=1, burst=2, remaining=2, waited=0)
    v2 = View(time=2, ready=(p3, p4), last_pid="P4", newly_arrived=())
    assert action_to_pid(v2, ACTION_SHORTEST_REMAINING) == "P4"

    # Tie on remaining when neither is running: earlier arrival wins
    v3 = View(time=2, ready=(p3, p4), last_pid=None, newly_arrived=())
    assert action_to_pid(v3, ACTION_SHORTEST_REMAINING) == "P3"


def test_action_longest_remaining() -> None:
    """
    1. The use of function:
       Tests action 1 selecting process with longest remaining time and tie-breaking.
    2. How does the specific function work:
       Tests strictly longest remaining, running process advantage on ties, and earlier arrival.
    3. Concepts used in the function:
       Longest remaining dispatch, tie-break verification.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested action 1 dispatch.
       After: Asserts correct PID selected for longest remaining.
    """
    p1 = ProcView(pid="P1", arrival=0, burst=5, remaining=5, waited=0)
    p2 = ProcView(pid="P2", arrival=1, burst=8, remaining=7, waited=0)
    v1 = View(time=1, ready=(p1, p2), last_pid=None, newly_arrived=())
    assert action_to_pid(v1, ACTION_LONGEST_REMAINING) == "P2"

    # Tie: running process wins
    p3 = ProcView(pid="P3", arrival=0, burst=7, remaining=7, waited=0)
    v2 = View(time=2, ready=(p2, p3), last_pid="P3", newly_arrived=())
    assert action_to_pid(v2, ACTION_LONGEST_REMAINING) == "P3"


def test_action_earliest_arrival() -> None:
    """
    1. The use of function:
       Tests action 2 selecting process with earliest arrival time (FCFS).
    2. How does the specific function work:
       Tests arrival ordering and lexicographical PID tie-breaking.
    3. Concepts used in the function:
       FCFS dispatch, arrival tie-breaking.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested action 2 dispatch.
       After: Asserts earliest arrival selected correctly.
    """
    p1 = ProcView(pid="P1", arrival=2, burst=3, remaining=3, waited=0)
    p2 = ProcView(pid="P2", arrival=0, burst=5, remaining=5, waited=2)
    v = View(time=2, ready=(p1, p2), last_pid=None, newly_arrived=())
    assert action_to_pid(v, ACTION_EARLIEST_ARRIVAL) == "P2"

    # Simultaneous arrival tie: lexicographical PID
    p3 = ProcView(pid="P3", arrival=0, burst=2, remaining=2, waited=2)
    v_tie = View(time=2, ready=(p3, p2), last_pid=None, newly_arrived=())
    assert action_to_pid(v_tie, ACTION_EARLIEST_ARRIVAL) == "P2"


def test_action_continue_current() -> None:
    """
    1. The use of function:
       Tests action 3 (CONTINUE_CURRENT) behavior when valid, masked, and with fallback.
    2. How does the specific function work:
       Asserts last_pid returned when ready. Asserts ValueError when masked without fallback.
       Asserts earliest arrival returned when masked with fallback_to_earliest=True.
    3. Concepts used in the function:
       Action continuation, masked exception handling, fallback policy dispatch.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested action 3 dispatch.
       After: Asserts continuation and fallback behavior.
    """
    p1 = ProcView(pid="P1", arrival=0, burst=5, remaining=3, waited=0)
    p2 = ProcView(pid="P2", arrival=1, burst=4, remaining=4, waited=1)

    # Valid continuation
    v_valid = View(time=2, ready=(p1, p2), last_pid="P1", newly_arrived=())
    assert action_to_pid(v_valid, ACTION_CONTINUE_CURRENT) == "P1"

    # Masked continuation without fallback
    v_masked = View(time=3, ready=(p1, p2), last_pid="P0", newly_arrived=())
    with pytest.raises(ValueError, match="is masked: last_pid 'P0' is not ready"):
        action_to_pid(v_masked, ACTION_CONTINUE_CURRENT, fallback_to_earliest=False)

    # Masked continuation with fallback -> selects earliest arrival (P1)
    assert (
        action_to_pid(v_masked, ACTION_CONTINUE_CURRENT, fallback_to_earliest=True)
        == "P1"
    )


def test_action_to_pid_validation_errors() -> None:
    """
    1. The use of function:
       Tests defensive validation and error handling in action_to_pid and get_valid_actions.
    2. How does the specific function work:
       Tests invalid types, empty ready queues, and unknown action integers.
    3. Concepts used in the function:
       Defensive validation, boundary checks.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid inputs.
       After: Asserts TypeError or ValueError raised.
    """
    with pytest.raises(TypeError, match="view must be a View instance"):
        get_valid_actions("not_a_view")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="view must be a View instance"):
        action_to_pid("not_a_view", 0)  # type: ignore[arg-type]

    p1 = ProcView(pid="P1", arrival=0, burst=4, remaining=4, waited=0)
    v = View(time=0, ready=(p1,), last_pid=None, newly_arrived=())

    with pytest.raises(TypeError, match="action must be an int"):
        action_to_pid(v, "0")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown action 99"):
        action_to_pid(v, 99)

    v_empty = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(ValueError, match="ready queue is empty"):
        action_to_pid(v_empty, 0)
