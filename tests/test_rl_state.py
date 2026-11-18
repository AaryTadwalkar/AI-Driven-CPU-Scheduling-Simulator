"""
1. What this file does:
   Tests the reinforcement learning state space representation, bucket discretization, and
   permutation-invariance properties.
2. Overall design:
   Validates `digitize_value` and `encode_state` in `schedsim.rl.state`. Asserts exact bucket
   indices across boundary conditions, hashability, dictionary key usage, and permutation
   invariance under PID reordering.
3. Elements in the file:
   - test_state_shape_and_constants: Validates state dimensions and total possible states.
   - test_digitize_value_boundaries: Tests boundary edges across all bucket configurations.
   - test_encode_state_known_values: Validates 7-tuple feature extraction against analytical values.
   - test_encode_state_permutation_invariance: Tests identical state tuples under PID permutation.
   - test_encode_state_hashability: Verifies state tuples can be used as dictionary keys.
   - test_encode_state_validation_errors: Verifies error handling on empty queue or invalid types.
4. Final output of the file:
   Pytest test suite certifying RL state encoding integrity and invariance properties.
"""

import pytest

import config
from schedsim.domain.engine import ProcView, View
from schedsim.rl.state import (
    STATE_SHAPE,
    TERMINAL_STATE,
    TOTAL_POSSIBLE_STATES,
    RLState,
    digitize_value,
    encode_state,
)


def test_state_shape_and_constants() -> None:
    """
    1. The use of function:
       Validates state dimensions, terminal state, and total possible states constant.
    2. How does the specific function work:
       Asserts STATE_SHAPE is (4, 5, 5, 4, 2, 2, 2) and product is 3200.
    3. Concepts used in the function:
       State space size validation, combinatorial capacity.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified state constants.
       After: Asserts exact match with Section 5 Stage 5 specifications.
    """
    assert STATE_SHAPE == (4, 5, 5, 4, 2, 2, 2)
    assert TOTAL_POSSIBLE_STATES == 3200
    assert TERMINAL_STATE == (0, 0, 0, 0, 0, 0, 0)
    prod = 1
    for dim in STATE_SHAPE:
        prod *= dim
    assert prod == 3200


def test_digitize_value_boundaries() -> None:
    """
    1. The use of function:
       Tests `digitize_value` across boundary edges for all configured discretization features.
    2. How does the specific function work:
       Tests ready count (1, 2, 3, 4+), shortest remaining (1, 2, 3-4, 5-8, 9+), longest remaining
       (1-2, 3-4, 5-8, 9-16, 17+), and max waited (0, 1-2, 3-5, 6+).
    3. Concepts used in the function:
       Boundary value analysis, discretization bucket verification.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested bucket threshold logic.
       After: Asserts exact bucket index returned for all boundary cases.
    """
    # 1. Ready count: (1, 2, 3) -> 4 buckets
    rc_edges = config.STATE_READY_COUNT_EDGES
    assert digitize_value(1, rc_edges) == 0
    assert digitize_value(2, rc_edges) == 1
    assert digitize_value(3, rc_edges) == 2
    assert digitize_value(4, rc_edges) == 3
    assert digitize_value(12, rc_edges) == 3

    # 2. Shortest remaining: (1, 2, 4, 8) -> 5 buckets
    sr_edges = config.STATE_SHORTEST_REMAINING_EDGES
    assert digitize_value(1, sr_edges) == 0
    assert digitize_value(2, sr_edges) == 1
    assert digitize_value(3, sr_edges) == 2
    assert digitize_value(4, sr_edges) == 2
    assert digitize_value(5, sr_edges) == 3
    assert digitize_value(8, sr_edges) == 3
    assert digitize_value(9, sr_edges) == 4
    assert digitize_value(25, sr_edges) == 4

    # 3. Longest remaining: (2, 4, 8, 16) -> 5 buckets
    lr_edges = config.STATE_LONGEST_REMAINING_EDGES
    assert digitize_value(1, lr_edges) == 0
    assert digitize_value(2, lr_edges) == 0
    assert digitize_value(3, lr_edges) == 1
    assert digitize_value(4, lr_edges) == 1
    assert digitize_value(5, lr_edges) == 2
    assert digitize_value(8, lr_edges) == 2
    assert digitize_value(9, lr_edges) == 3
    assert digitize_value(16, lr_edges) == 3
    assert digitize_value(17, lr_edges) == 4
    assert digitize_value(30, lr_edges) == 4

    # 4. Max waited: (0, 2, 5) -> 4 buckets
    mw_edges = config.STATE_MAX_WAITED_EDGES
    assert digitize_value(0, mw_edges) == 0
    assert digitize_value(1, mw_edges) == 1
    assert digitize_value(2, mw_edges) == 1
    assert digitize_value(3, mw_edges) == 2
    assert digitize_value(5, mw_edges) == 2
    assert digitize_value(6, mw_edges) == 3
    assert digitize_value(50, mw_edges) == 3


def test_encode_state_known_values() -> None:
    """
    1. The use of function:
       Tests `encode_state` on a constructed View with known expected feature values.
    2. How does the specific function work:
       Constructs View with P1(remaining=2, waited=4), P2(remaining=6, waited=1).
       last_pid="P1", more_arrivals=True.
       Evaluates expected bucket values and asserts exact matching tuple.
    3. Concepts used in the function:
       State encoding, feature composition, boolean flag assertion.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested state encoding on reference View.
       After: Asserts exact matching 7-tuple state.
    """
    p1 = ProcView(pid="P1", arrival=0, burst=5, remaining=2, waited=4)
    p2 = ProcView(pid="P2", arrival=1, burst=8, remaining=6, waited=1)
    view = View(time=3, ready=(p1, p2), last_pid="P1", newly_arrived=())

    state = encode_state(view, more_arrivals_pending=True)
    assert isinstance(state, tuple)
    assert len(state) == 7

    # 1. ready_count = 2 -> bucket 1
    assert state[0] == 1
    # 2. shortest_rem = 2 -> bucket 1
    assert state[1] == 1
    # 3. longest_rem = 6 -> bucket 2 (in 5-8)
    assert state[2] == 2
    # 4. max_waited = 4 -> bucket 2 (in 3-5)
    assert state[3] == 2
    # 5. last_pid is P1, which is ready -> 1
    assert state[4] == 1
    # 6. more_arrivals_pending = True -> 1
    assert state[5] == 1
    # 7. is_last_shortest: P1 remaining=2 is the minimum (2) -> 1
    assert state[6] == 1


def test_encode_state_permutation_invariance() -> None:
    """
    1. The use of function:
       Tests strict permutation invariance of state encoding when PIDs are swapped or renamed.
    2. How does the specific function work:
       Constructs two Views with identical process parameters but permuted and renamed PIDs
       (P1, P2 vs P_X, P_Y). Asserts that `encode_state()` produces identical tuples.
    3. Concepts used in the function:
       Permutation invariance, PID-agnostic state abstraction.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified permutation invariance.
       After: Asserts identical state tuples for symmetrically permuted Views.
    """
    # View A: P1 remaining 3, P2 remaining 7
    pa1 = ProcView(pid="P1", arrival=0, burst=5, remaining=3, waited=2)
    pa2 = ProcView(pid="P2", arrival=1, burst=8, remaining=7, waited=0)
    va = View(time=2, ready=(pa1, pa2), last_pid="P1", newly_arrived=())

    # View B: P_ALPHA remaining 7, P_BETA remaining 3, last_pid is P_BETA
    pb1 = ProcView(pid="P_ALPHA", arrival=1, burst=8, remaining=7, waited=0)
    pb2 = ProcView(pid="P_BETA", arrival=0, burst=5, remaining=3, waited=2)
    vb = View(time=2, ready=(pb1, pb2), last_pid="P_BETA", newly_arrived=())

    sa = encode_state(va, more_arrivals_pending=False)
    sb = encode_state(vb, more_arrivals_pending=False)

    assert sa == sb, f"Permuted views produced divergent states: {sa} vs {sb}."


def test_encode_state_hashability() -> None:
    """
    1. The use of function:
       Tests that encoded state tuples are hashable and valid as dictionary keys for Q-tables.
    2. How does the specific function work:
       Encodes a state, creates a dictionary mapping `(state, action)` to Q-values, and verifies
       lookup.
    3. Concepts used in the function:
       Hashability, tabular Q-table indexing.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested hashability in tabular structures.
       After: Asserts successful dictionary hashing and retrieval.
    """
    p1 = ProcView(pid="P1", arrival=0, burst=4, remaining=4, waited=0)
    v = View(time=0, ready=(p1,), last_pid=None, newly_arrived=())
    s: RLState = encode_state(v, more_arrivals_pending=False)

    q_table: dict[tuple[RLState, int], float] = {}
    q_table[(s, 0)] = 2.5
    assert (s, 0) in q_table
    assert q_table[(s, 0)] == 2.5


def test_encode_state_validation_errors() -> None:
    """
    1. The use of function:
       Tests defensive validation and error handling in encode_state.
    2. How does the specific function work:
       Passes invalid types or empty ready queue and asserts TypeError or ValueError.
    3. Concepts used in the function:
       Defensive programming, exception verification.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid inputs to encode_state.
       After: Asserts TypeError or ValueError raised.
    """
    with pytest.raises(TypeError, match="view must be a View instance"):
        encode_state("not_a_view")  # type: ignore[arg-type]

    v_empty = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(ValueError, match="ready queue is empty"):
        encode_state(v_empty)
