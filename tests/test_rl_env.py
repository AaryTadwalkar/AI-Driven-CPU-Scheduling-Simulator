"""
1. What this file does:
   Defines unit test suites for the reinforcement learning Environment class.
2. Overall design:
   Validates MDP environment behavior:
   - Proper initialization, input validation, and reset behavior.
   - Action masking and valid actions computation across simulation steps.
   - Transition semantics, reward computation (reward == -penalty), and info dict content.
   - Error handling on uninitialized, masked, or completed environment steps.
   - SimResult retrieval upon episode completion.
3. Elements in the file:
   - test_env_initialization: Verifies parameter validation and initial defaults.
   - test_env_reset: Verifies reset with and without re-supplying workloads.
   - test_env_valid_actions: Checks action masking and availability across steps.
   - test_env_step_transitions: Checks step output contracts, state transitions, and rewards.
   - test_env_step_masked_action_raises: Asserts ValueError on illegal action dispatches.
   - test_env_step_before_reset_raises: Asserts RuntimeError when stepping uninitialized env.
   - test_env_step_after_done_raises: Asserts RuntimeError when stepping completed env.
   - test_env_result_handling: Tests result() availability only upon completion.
   - test_env_repr: Tests human-readable string representation.
4. Final output of the file:
   Pytest test suite validating schedsim.rl.env.Environment.
"""

import pytest

from schedsim.domain.engine import SimResult
from schedsim.domain.process import Process
from schedsim.rl.actions import (
    ACTION_CONTINUE_CURRENT,
    ACTION_EARLIEST_ARRIVAL,
    ACTION_LONGEST_REMAINING,
    ACTION_SHORTEST_REMAINING,
)
from schedsim.rl.env import Environment
from schedsim.rl.state import TERMINAL_STATE


def test_env_initialization() -> None:
    """
    1. The use of function:
       Verifies Environment initialization and parameter validation.
    2. How does the specific function work:
       Tests default cs_cost=0, valid positive cs_cost, and rejection of negative or
       non-integer context switch costs.
    3. Concepts used in the function:
       Constructor validation, boundary testing, type checking.
    4. Places where this function is imported and used:
       Executed as part of pytest test runner.
    5. Final output of the function, before and after:
       Before: Unverified constructor logic.
       After: Asserts proper instantiation and exception raising.
    """
    env = Environment()
    assert env.cs_cost == 0
    assert env.engine is None
    assert env.done is True
    assert env.time == 0
    assert env.state is None
    assert env.workload is None
    assert env.valid_actions() == []

    env2 = Environment(cs_cost=2)
    assert env2.cs_cost == 2

    with pytest.raises(TypeError, match="cs_cost must be an int"):
        Environment(cs_cost="1")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="cs_cost must be an int"):
        Environment(cs_cost=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cs_cost must be >= 0"):
        Environment(cs_cost=-1)


def test_env_reset() -> None:
    """
    1. The use of function:
       Tests Environment.reset() with workloads and re-resetting.
    2. How does the specific function work:
       Supplies a 2-process workload to reset(), confirms returned state is a 7-tuple,
       engine is initialized, time is set, and re-resetting without args uses cached workload.
    3. Concepts used in the function:
       Reset contract, workload caching, state encoding.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unreset environment.
       After: Confirms initial state valid 7-tuple and workload recorded.
    """
    p1 = Process(pid="P1", arrival=0, burst=4)
    p2 = Process(pid="P2", arrival=2, burst=6)
    workload = [p1, p2]

    env = Environment(cs_cost=1)
    with pytest.raises(ValueError, match="Workload must be provided on first reset"):
        env.reset()

    state = env.reset(workload)
    assert isinstance(state, tuple)
    assert len(state) == 7
    assert env.engine is not None
    assert env.done is False
    assert env.time == 0
    assert env.workload == tuple(workload)

    # Re-reset without passing workload uses cached workload
    state2 = env.reset()
    assert state2 == state


def test_env_valid_actions() -> None:
    """
    1. The use of function:
       Verifies valid_actions() behavior across execution phases.
    2. How does the specific function work:
       Tests that at tick 0 before any dispatch, action 3 (CONTINUE_CURRENT) is masked.
       After dispatching action 0, action 3 becomes valid if that process remains ready.
    3. Concepts used in the function:
       Action masking, dynamic validity inspection.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified action masking in Environment.
       After: Confirms action 3 dynamically enabled/disabled.
    """
    p1 = Process(pid="P1", arrival=0, burst=3)
    p2 = Process(pid="P2", arrival=0, burst=5)
    env = Environment(cs_cost=0, workload=[p1, p2])

    # Initially last_pid is None, so action 3 is masked
    actions = env.valid_actions()
    assert ACTION_CONTINUE_CURRENT not in actions
    assert set(actions) == {
        ACTION_SHORTEST_REMAINING,
        ACTION_LONGEST_REMAINING,
        ACTION_EARLIEST_ARRIVAL,
    }

    # Step action 0 (dispatches P1)
    next_state, reward, done, info = env.step(ACTION_SHORTEST_REMAINING)
    assert info["pid"] == "P1"
    assert not done

    # Now P1 is last_pid and still ready (remaining 2), so action 3 must be valid
    actions_after = env.valid_actions()
    assert ACTION_CONTINUE_CURRENT in actions_after
    assert len(actions_after) == 4


def test_env_step_transitions() -> None:
    """
    1. The use of function:
       Tests Environment.step() return contracts, transitions, and reward identity.
    2. How does the specific function work:
       Steps through a simple 1-process workload, verifying transition tuple, reward,
       and terminal state return.
    3. Concepts used in the function:
       Step return contract, penalty conversion, episode termination.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Step transitions unverified.
       After: Confirms valid step output contracts and terminal state transition.
    """
    p1 = Process(pid="P1", arrival=0, burst=2)
    env = Environment(cs_cost=0, workload=[p1])

    # Step 1: P1 runs tick [0, 1). Ready queue had only P1, so penalty=0, reward=0.0
    state1, reward1, done1, info1 = env.step(ACTION_EARLIEST_ARRIVAL)
    assert isinstance(state1, tuple)
    assert len(state1) == 7
    assert reward1 == 0.0
    assert not done1
    assert info1["pid"] == "P1"
    assert info1["penalty"] == 0
    assert info1["time"] == 1

    # Step 2: P1 completes at tick 2. Terminal state reached
    state2, reward2, done2, info2 = env.step(ACTION_EARLIEST_ARRIVAL)
    assert state2 == TERMINAL_STATE
    assert reward2 == 0.0
    assert done2 is True
    assert info2["pid"] == "P1"
    assert info2["time"] == 2


def test_env_step_masked_action_raises() -> None:
    """
    1. The use of function:
       Verifies ValueError when dispatching an invalid or masked action.
    2. How does the specific function work:
       Attempts to execute action 3 (CONTINUE_CURRENT) when last_pid is None, and
       attempts to execute an out-of-range action index.
    3. Concepts used in the function:
       Defensive error raising, action validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Action error boundaries unverified.
       After: Confirms ValueError on illegal action indices.
    """
    p1 = Process(pid="P1", arrival=0, burst=2)
    env = Environment(cs_cost=0, workload=[p1])

    with pytest.raises(ValueError, match="Action 3 is invalid or masked"):
        env.step(ACTION_CONTINUE_CURRENT)

    with pytest.raises(ValueError, match="Action 99 is invalid or masked"):
        env.step(99)


def test_env_step_before_reset_raises() -> None:
    """
    1. The use of function:
       Verifies RuntimeError when step() is called before reset().
    2. How does the specific function work:
       Creates uninitialized Environment and calls step(0).
    3. Concepts used in the function:
       State machine invariants, lifecycle enforcement.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Uninitialized step vulnerability.
       After: Confirms RuntimeError raised.
    """
    env = Environment(cs_cost=0)
    with pytest.raises(RuntimeError, match="Environment must be reset"):
        env.step(0)


def test_env_step_after_done_raises() -> None:
    """
    1. The use of function:
       Verifies RuntimeError when step() is called after episode completion.
    2. How does the specific function work:
       Runs a 1-tick process to completion, then attempts an additional step().
    3. Concepts used in the function:
       Post-completion guard, state validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Post-done step behavior unverified.
       After: Confirms RuntimeError raised on extraneous step.
    """
    p1 = Process(pid="P1", arrival=0, burst=1)
    env = Environment(cs_cost=0, workload=[p1])
    env.step(ACTION_EARLIEST_ARRIVAL)
    assert env.done is True

    with pytest.raises(RuntimeError, match="Cannot call step\\(\\) on a completed environment"):
        env.step(0)


def test_env_result_handling() -> None:
    """
    1. The use of function:
       Tests result() retrieval before and after episode termination.
    2. How does the specific function work:
       Asserts RuntimeError before completion, then completes simulation and verifies
       that result() returns a valid SimResult.
    3. Concepts used in the function:
       Result compilation, engine delegation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Result retrieval unverified.
       After: Confirms proper error before done and valid SimResult after done.
    """
    env_unreset = Environment()
    with pytest.raises(RuntimeError, match="Environment has not been initialized"):
        env_unreset.result()

    p1 = Process(pid="P1", arrival=0, burst=2)
    env = Environment(cs_cost=0, workload=[p1])
    with pytest.raises(RuntimeError, match="simulation is not yet complete"):
        env.result()

    env.step(0)
    env.step(0)
    assert env.done is True

    res = env.result()
    assert isinstance(res, SimResult)
    assert len(res.process_metrics) == 1
    assert res.process_metrics[0].pid == "P1"


def test_env_repr() -> None:
    """
    1. The use of function:
       Tests the string representation of Environment.
    2. How does the specific function work:
       Instantiates Environment and checks repr formatting.
    3. Concepts used in the function:
       String representation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Repr unverified.
       After: Confirms accurate formatted string.
    """
    env = Environment(cs_cost=1)
    rep = repr(env)
    assert "Environment" in rep
    assert "cs_cost=1" in rep
