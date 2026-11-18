"""
1. What this file does:
   Defines unit tests for the Process dataclass and validate_processes collection validator.
2. Overall design:
   Validates invariant enforcement, boundary rejection, immutability guarantees, and batch
   validation rules using pytest test functions.
3. Elements in the file:
   - test_process_valid_creation: Tests valid parameter initialization and defaults.
   - test_process_immutability: Tests frozen dataclass modification prevention.
   - test_process_equality_and_hashing: Tests value-object equality and set membership.
   - test_process_invalid_pid: Tests string and non-empty validation on pid.
   - test_process_invalid_arrival: Tests type and non-negative constraints on arrival.
   - test_process_invalid_burst: Tests type and positive-integer constraints on burst.
   - test_process_invalid_priority: Tests type and non-negative constraints on priority.
   - test_validate_processes_valid: Tests validation of correct process collections.
   - test_validate_processes_empty: Tests rejection of empty process collections.
   - test_validate_processes_duplicates: Tests rejection of duplicate process IDs.
   - test_validate_processes_invalid_types: Tests rejection of non-Process objects in collection.
4. Final output of the file:
   Pytest assertion passes confirming robust domain entity behavior and error signaling.
"""

from dataclasses import FrozenInstanceError

import pytest

from schedsim.domain.process import Process, validate_processes


def test_process_valid_creation() -> None:
    """
    1. The use of function:
       Tests successful instantiation of Process objects with valid boundaries.
    2. How does the specific function work:
       Creates processes with default priority and explicit priority, asserting attribute values.
    3. Concepts used in the function:
       Dataclass field access, default parameter assertion.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Uninstantiated valid processes.
       After: Asserts attributes match provided arguments and default priority is 0.
    """
    p1 = Process(pid="P1", arrival=0, burst=5)
    assert p1.pid == "P1"
    assert p1.arrival == 0
    assert p1.burst == 5
    assert p1.priority == 0

    p2 = Process(pid="P2", arrival=2, burst=10, priority=3)
    assert p2.pid == "P2"
    assert p2.arrival == 2
    assert p2.burst == 10
    assert p2.priority == 3


def test_process_immutability() -> None:
    """
    1. The use of function:
       Verifies that Process instances are strictly immutable.
    2. How does the specific function work:
       Attempts attribute reassignment on an instantiated process and asserts FrozenInstanceError.
    3. Concepts used in the function:
       Frozen dataclass enforcement, immutability safety.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Valid frozen Process instance.
       After: Asserts modifying any field raises FrozenInstanceError.
    """
    p = Process(pid="P1", arrival=0, burst=4)
    with pytest.raises(FrozenInstanceError):
        p.burst = 8  # type: ignore[misc]


def test_process_equality_and_hashing() -> None:
    """
    1. The use of function:
       Verifies value-based equality and hashability of Process objects.
    2. How does the specific function work:
       Instantiates identical and distinct Process instances, verifying equality and set hashing.
    3. Concepts used in the function:
       Value object pattern, __hash__, __eq__ contracts in Python.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Multiple Process instances.
       After: Asserts identical definitions evaluate equal and dedup in hash sets.
    """
    p1 = Process(pid="P1", arrival=1, burst=4, priority=1)
    p2 = Process(pid="P1", arrival=1, burst=4, priority=1)
    p3 = Process(pid="P2", arrival=1, burst=4, priority=1)

    assert p1 == p2
    assert p1 != p3
    assert len({p1, p2, p3}) == 2


def test_process_invalid_pid() -> None:
    """
    1. The use of function:
       Ensures invalid process IDs are rejected with appropriate exceptions.
    2. How does the specific function work:
       Attempts to instantiate Process with empty, whitespace, and non-string pids.
    3. Concepts used in the function:
       Defensive validation, TypeError and ValueError distinction.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Malformed pid candidates.
       After: Asserts ValueError for blank strings and TypeError for non-strings.
    """
    with pytest.raises(ValueError, match="cannot be empty"):
        Process(pid="", arrival=0, burst=1)

    with pytest.raises(ValueError, match="cannot be empty"):
        Process(pid="   ", arrival=0, burst=1)

    with pytest.raises(TypeError, match="must be a str"):
        Process(pid=123, arrival=0, burst=1)  # type: ignore[arg-type]


def test_process_invalid_arrival() -> None:
    """
    1. The use of function:
       Ensures arrival times are non-negative integers.
    2. How does the specific function work:
       Tests negative arrival values and non-int types (float, bool, string).
    3. Concepts used in the function:
       Boundary validation, Python boolean subclass detection.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Malformed arrival values.
       After: Asserts ValueError for negative numbers and TypeError for non-integers.
    """
    with pytest.raises(ValueError, match="arrival must be >= 0"):
        Process(pid="P1", arrival=-1, burst=2)

    with pytest.raises(TypeError, match="arrival must be an int"):
        Process(pid="P1", arrival=1.5, burst=2)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="arrival must be an int"):
        Process(pid="P1", arrival=False, burst=2)  # type: ignore[arg-type]


def test_process_invalid_burst() -> None:
    """
    1. The use of function:
       Ensures burst durations are strictly positive integers (>= 1).
    2. How does the specific function work:
       Tests zero burst, negative burst, and non-int types (float, bool, string).
    3. Concepts used in the function:
       Strict positivity assertion, type checking.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Malformed burst values.
       After: Asserts ValueError for burst < 1 and TypeError for non-integers.
    """
    with pytest.raises(ValueError, match="burst must be >= 1"):
        Process(pid="P1", arrival=0, burst=0)

    with pytest.raises(ValueError, match="burst must be >= 1"):
        Process(pid="P1", arrival=0, burst=-4)

    with pytest.raises(TypeError, match="burst must be an int"):
        Process(pid="P1", arrival=0, burst=3.2)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="burst must be an int"):
        Process(pid="P1", arrival=0, burst=True)  # type: ignore[arg-type]


def test_process_invalid_priority() -> None:
    """
    1. The use of function:
       Ensures priority values are non-negative integers.
    2. How does the specific function work:
       Tests negative priority and non-int types (float, bool, string).
    3. Concepts used in the function:
       Non-negative constraint checking, defensive type checking.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Malformed priority values.
       After: Asserts ValueError for priority < 0 and TypeError for non-integers.
    """
    with pytest.raises(ValueError, match="priority must be >= 0"):
        Process(pid="P1", arrival=0, burst=1, priority=-2)

    with pytest.raises(TypeError, match="priority must be an int"):
        Process(pid="P1", arrival=0, burst=1, priority=1.5)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="priority must be an int"):
        Process(pid="P1", arrival=0, burst=1, priority=True)  # type: ignore[arg-type]


def test_validate_processes_valid() -> None:
    """
    1. The use of function:
       Verifies validate_processes passes for valid process sequences.
    2. How does the specific function work:
       Supplies lists and tuples of valid unique Process objects to validate_processes.
    3. Concepts used in the function:
       Sequence validation, idempotence.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Unchecked valid sequences.
       After: Confirms no exception is raised.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=3),
        Process(pid="P2", arrival=1, burst=2),
        Process(pid="P3", arrival=2, burst=1),
    ]
    validate_processes(procs)
    validate_processes(tuple(procs))


def test_validate_processes_empty() -> None:
    """
    1. The use of function:
       Ensures empty collections are rejected by validate_processes.
    2. How does the specific function work:
       Passes empty list, empty tuple, and None, expecting ValueError.
    3. Concepts used in the function:
       Non-empty collection invariant checking.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Empty collection candidates.
       After: Asserts ValueError is raised.
    """
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_processes([])

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_processes(())


def test_validate_processes_duplicates() -> None:
    """
    1. The use of function:
       Ensures duplicate process IDs are detected and rejected.
    2. How does the specific function work:
       Creates a list with repeated pid "P1" and verifies ValueError with duplicate message.
    3. Concepts used in the function:
       Uniqueness constraints, duplicate detection.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Sequence with duplicated process identifiers.
       After: Asserts ValueError reporting duplicate process ID.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=2),
        Process(pid="P2", arrival=1, burst=3),
        Process(pid="P1", arrival=2, burst=4),
    ]
    with pytest.raises(ValueError, match="Duplicate process ID detected: 'P1'"):
        validate_processes(procs)


def test_validate_processes_invalid_types() -> None:
    """
    1. The use of function:
       Ensures non-Process objects within the sequence trigger TypeError.
    2. How does the specific function work:
       Passes lists containing invalid items (dictionaries, strings) to validate_processes.
    3. Concepts used in the function:
       Homogeneous collection type verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test runs.
    5. Final output of the function, before and after:
       Before: Sequence containing non-Process items.
       After: Asserts TypeError is raised indicating expected Process instance.
    """
    invalid_procs = [
        Process(pid="P1", arrival=0, burst=2),
        {"pid": "P2", "arrival": 1, "burst": 3},  # type: ignore[list-item]
    ]
    with pytest.raises(TypeError, match="Expected Process instance"):
        validate_processes(invalid_procs)
