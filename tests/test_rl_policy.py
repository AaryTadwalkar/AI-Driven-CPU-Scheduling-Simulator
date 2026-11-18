"""
1. What this file does:
   Defines unit tests for the RLPolicy scheduling policy adapter class.
2. Overall design:
   Verifies that the RLPolicy wrapper integrates with the domain simulation engine:
   - Proper construction, parameter validation, and workload binding.
   - Dispatch behavior via select() and __call__() protocols.
   - Seamless end-to-end execution within the unified run() simulation harness.
   - Equivalence to classical baselines when Q-values favor specific heuristic actions.
3. Elements in the file:
   - test_rl_policy_initialization: Tests constructor validation and defaults.
   - test_rl_policy_select_and_bind: Tests select() PID resolution and bind_workload().
   - test_rl_policy_run_integration: Tests end-to-end execution via run().
   - test_rl_policy_srtf_equivalence_when_favoring_action_0: Asserts equivalence to SRTF.
4. Final output of the file:
   Pytest test suite validating schedsim.policies.rl_policy.RLPolicy.
"""

import pytest

from schedsim.domain.engine import SimResult, run
from schedsim.domain.process import Process
from schedsim.policies.srtf import SRTFPolicy
from schedsim.rl.actions import ACTION_SHORTEST_REMAINING, ALL_ACTIONS
from schedsim.rl.agent import QLearningAgent
from schedsim.rl.env import Environment
from schedsim.rl.policy import RLPolicy
from schedsim.workloads.presets import get_preset


def test_rl_policy_initialization() -> None:
    """
    1. The use of function:
       Verifies RLPolicy constructor parameter validation.
    2. How does the specific function work:
       Asserts TypeError if agent is not a QLearningAgent, and checks default name attribute.
    3. Concepts used in the function:
       Type validation, constructor assertion.
    4. Places where this function is imported and used:
       Executed as part of pytest test runner.
    5. Final output of the function, before and after:
       Before: RLPolicy initialization unverified.
       After: Confirms valid instantiation or raises TypeError.
    """
    agent = QLearningAgent()
    policy = RLPolicy(agent)
    assert policy.name == "RLPolicy"
    assert policy.agent is agent

    with pytest.raises(TypeError, match="agent must be a QLearningAgent instance"):
        RLPolicy("not_an_agent")  # type: ignore[arg-type]


def test_rl_policy_select_and_bind() -> None:
    """
    1. The use of function:
       Tests select() process dispatch and bind_workload() fluent chaining.
    2. How does the specific function work:
       Creates a 2-process workload and an agent with high Q-value for action 0.
       Binds workload to policy, resets environment to get View, calls policy.select(view),
       and verifies that the shortest process PID is returned.
    3. Concepts used in the function:
       View dispatch, state encoding, fluent method chaining.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: select() behavior unverified.
       After: Confirms accurate PID resolution.
    """
    p1 = Process(pid="P1", arrival=0, burst=5)
    p2 = Process(pid="P2", arrival=0, burst=2)
    workload = [p1, p2]

    agent = QLearningAgent()
    policy = RLPolicy(agent)
    chained = policy.bind_workload(workload)
    assert chained is policy

    env = Environment(cs_cost=0, workload=workload)
    initial_view = env.engine.current_view  # type: ignore[union-attr]

    # Pre-train state with preference for SHORTEST_REMAINING
    state = env.state
    assert state is not None
    agent._state_visits[state] = 1
    agent.set_q(state, ACTION_SHORTEST_REMAINING, 10.0)

    chosen_pid = policy.select(initial_view)
    assert chosen_pid == "P2"  # P2 has burst 2 vs P1 burst 5


def test_rl_policy_run_integration() -> None:
    """
    1. The use of function:
       Verifies that RLPolicy runs seamlessly through the unified run() engine harness.
    2. How does the specific function work:
       Loads the canonical Section 4.6 preset workload, runs it through run(policy, workload),
       and confirms that a complete, valid SimResult is produced.
    3. Concepts used in the function:
       Harness integration, domain decoupling, simulation execution.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: run() execution with RLPolicy unverified.
       After: Confirms valid SimResult generated without exceptions.
    """
    workload = get_preset("section_4_6")
    agent = QLearningAgent()
    policy = RLPolicy(agent)

    result = run(policy, workload, cs_cost=1)
    assert isinstance(result, SimResult)
    assert len(result.process_metrics) == len(workload)
    assert result.simulation_metrics.makespan > 0


def test_rl_policy_srtf_equivalence_when_favoring_action_0() -> None:
    """
    1. The use of function:
       Verifies that an agent configured to favor SHORTEST_REMAINING reproduces SRTFPolicy.
    2. How does the specific function work:
       Instantiates an agent with Q-values favoring action 0 for all actions. Runs a preset
       workload under both SRTFPolicy and RLPolicy at cs_cost=0. Asserts identical metrics.
    3. Concepts used in the function:
       Behavioral equivalence, baseline validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Behavioral equivalence unverified.
       After: Asserts identical makespan, waiting time, and turnaround time.
    """
    workload = get_preset("convoy")
    agent = QLearningAgent()

    # Pre-populate all reachable states with high Q for ACTION_SHORTEST_REMAINING
    env = Environment(cs_cost=0, workload=workload)
    while not env.done:
        s = env.state
        if s is not None:
            agent._state_visits[s] = 1
            agent.set_q(s, ACTION_SHORTEST_REMAINING, 100.0)
            for a in ALL_ACTIONS:
                if a != ACTION_SHORTEST_REMAINING:
                    agent.set_q(s, a, 0.0)
        env.step(ACTION_SHORTEST_REMAINING)

    rl_policy = RLPolicy(agent, workload)
    rl_result = run(rl_policy, workload, cs_cost=0)

    srtf_policy = SRTFPolicy()
    srtf_result = run(srtf_policy, workload, cs_cost=0)

    assert rl_result.simulation_metrics.makespan == srtf_result.simulation_metrics.makespan
    assert rl_result.simulation_metrics.avg_waiting == pytest.approx(
        srtf_result.simulation_metrics.avg_waiting
    )
    assert rl_result.simulation_metrics.avg_turnaround == pytest.approx(
        srtf_result.simulation_metrics.avg_turnaround
    )
