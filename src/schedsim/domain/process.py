"""
1. What this file does:
   Defines the immutable Process dataclass and workload sequence validation functions for the
   CPU scheduling simulation system.
2. Overall design:
   Acts as the fundamental domain entity representing executable tasks. Ensures strict data
   integrity at construction time and collection-level invariants (non-empty, unique pids)
   before scheduling runs.
3. Elements in the file:
   - Process: Frozen dataclass encapsulating pid, arrival, burst, and priority with validation.
   - validate_processes: Enforces collection-level constraints on sequences of Process instances.
4. Final output of the file:
   Exports verified Process definitions and batch validation utilities for domain, workload,
   policy, engine, and API layers.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Process:
    """
    1. The use of class:
       Represents an immutable CPU process domain entity with validated scheduling parameters.
    2. What the class contains:
       - pid: Unique process identifier string.
       - arrival: Non-negative integer tick indicating when the process arrives.
       - burst: Positive integer tick duration required for CPU computation.
       - priority: Non-negative integer priority value (default 0).
       - __post_init__: Validation routine executing immediately after dataclass instantiation.
    3. Technologies used and why:
       - Python dataclasses with frozen=True for immutability, thread-safety, and automatic
         generation of __eq__, __hash__, and __repr__.
       - Strict type and range validation to prevent malformed simulation inputs early.
    4. Key design and overall design of the class:
       Adheres to Domain-Driven Design (DDD) value object semantics. Immutability ensures that
       simulation engines and policies cannot inadvertently mutate workload definitions during
       execution.
    """

    pid: str
    arrival: int
    burst: int
    priority: int = 0

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates the integrity, types, and boundaries of process attributes upon creation.
        2. How does the specific function work:
           Inspects pid, arrival, burst, and priority. Raises TypeError for invalid types and
           ValueError for out-of-boundary values (empty pid, negative arrival/priority, burst < 1).
        3. Concepts used in the function:
           Dataclass post-initialization hook, defensive programming, boundary validation.
        4. Places where this function is imported and used:
           Invoked automatically by the dataclass __init__ whenever a Process is instantiated.
        5. Final output of the function, before and after:
           Before: Unverified raw field values assigned by dataclass constructor.
           After: Confirmed valid invariant state or raises TypeError / ValueError on failure.
        """
        if not isinstance(self.pid, str):
            raise TypeError(f"Process pid must be a str, got {type(self.pid).__name__}.")
        if not self.pid.strip():
            raise ValueError("Process pid cannot be empty or whitespace only.")

        if isinstance(self.arrival, bool) or not isinstance(self.arrival, int):
            raise TypeError(
                f"Process arrival must be an int, got {type(self.arrival).__name__}."
            )
        if self.arrival < 0:
            raise ValueError(f"Process arrival must be >= 0, got {self.arrival}.")

        if isinstance(self.burst, bool) or not isinstance(self.burst, int):
            raise TypeError(
                f"Process burst must be an int, got {type(self.burst).__name__}."
            )
        if self.burst < 1:
            raise ValueError(f"Process burst must be >= 1, got {self.burst}.")

        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError(
                f"Process priority must be an int, got {type(self.priority).__name__}."
            )
        if self.priority < 0:
            raise ValueError(f"Process priority must be >= 0, got {self.priority}.")


def validate_processes(processes: Sequence[Process]) -> None:
    """
    1. The use of function:
       Validates a collection of Process objects for engine execution, ensuring non-emptiness
       and unique process identifiers.
    2. How does the specific function work:
       Checks that processes is non-empty, verifies each element is a Process instance, and checks
       for duplicate pids using a seen-set. Raises ValueError or TypeError on violations.
    3. Concepts used in the function:
       Collection invariants, set-based uniqueness checking, defensive parameter validation.
    4. Places where this function is imported and used:
       Imported by simulation engines, workload generators, API endpoints, and test suites.
    5. Final output of the function, before and after:
       Before: Arbitrary sequence of processes with unknown validity.
       After: Returns None if sequence satisfies all invariants, otherwise raises
       ValueError/TypeError.
    """
    if processes is None or len(processes) == 0:
        raise ValueError("Process collection cannot be empty.")

    seen_pids: set[str] = set()
    for proc in processes:
        if not isinstance(proc, Process):
            raise TypeError(
                f"Expected Process instance, got {type(proc).__name__}."
            )
        if proc.pid in seen_pids:
            raise ValueError(f"Duplicate process ID detected: '{proc.pid}'.")
        seen_pids.add(proc.pid)
