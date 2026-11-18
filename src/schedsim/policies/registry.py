"""
1. What this file does:
   Implements the centralized policy registry and factory for CPU scheduling policies.
2. Overall design:
   Maintains a registry mapping case-insensitive policy names and aliases to their respective
   Policy subclass implementations. Provides factory instantiation with kwargs forwarding,
   dynamic policy registration, canonical policy listing, and descriptive error handling.
3. Elements in the file:
   - _REGISTRY: Private dictionary storing normalized lookup keys and (canonical_name, policy_cls).
   - register_policy: Function to register a Policy subclass under a given name.
   - get_policy: Factory function to retrieve and instantiate a Policy instance by name.
   - list_policies: Function returning sorted unique canonical policy names.
   - _reset_registry: Helper function restoring default classical policy registrations.
4. Final output of the file:
   Exports get_policy, list_policies, and register_policy for CLI, runner, and simulation modules.
"""

from typing import Any

from schedsim.policies.base import Policy
from schedsim.policies.fcfs import FCFSPolicy
from schedsim.policies.round_robin import RoundRobinPolicy
from schedsim.policies.sjf import SJFPolicy
from schedsim.policies.srtf import SRTFPolicy

_REGISTRY: dict[str, tuple[str, type[Policy]]] = {}


def register_policy(name: str, policy_cls: type[Policy]) -> None:
    """
    1. The use of function:
       Registers a Policy subclass in the global registry under a given name.
    2. How does the specific function work:
       Validates that name is a non-empty string and policy_cls is a subclass of Policy.
       Normalizes the lookup key by stripping whitespace and converting to lowercase.
       Stores a tuple of (stripped_name, policy_cls) in the internal registry.
    3. Concepts used in the function:
       Type verification (issubclass, isinstance), string normalization, registry pattern.
    4. Places where this function is imported and used:
       Exported in schedsim.policies, used on module load to register classical algorithms,
       and called by user code or plugins to register custom scheduling algorithms.
    5. Final output of the function, before and after:
       Before: Registry without the mapping for name.
       After: Registry containing the normalized mapping to (stripped_name, policy_cls).
    """
    if not isinstance(name, str):
        raise TypeError(f"Policy name must be a str, got {type(name).__name__}")
    stripped = name.strip()
    if not stripped:
        raise ValueError("Policy name cannot be empty")
    if not isinstance(policy_cls, type) or not issubclass(policy_cls, Policy):
        raise TypeError("policy_cls must be a subclass of Policy")

    _REGISTRY[stripped.lower()] = (stripped, policy_cls)


def get_policy(name: str, /, **kwargs: Any) -> Policy:
    """
    1. The use of function:
       Retrieves and instantiates a registered scheduling policy by name with optional kwargs.
    2. How does the specific function work:
       Validates that name is a string. Normalizes the name to lowercase stripped format and looks
       it up in the registry. If not found, raises a KeyError listing all currently available
       policies. If found, instantiates and returns the corresponding Policy class with kwargs.
    3. Concepts used in the function:
       Factory pattern, case-insensitive string matching, error diagnostic formatting, kwargs,
       positional-only parameter syntax.
    4. Places where this function is imported and used:
       Exported in schedsim.policies, used by the CLI simulation subcommand, benchmark runners,
       API endpoints, and unit test suites.
    5. Final output of the function, before and after:
       Before: Policy name string and optional constructor keyword arguments.
       After: Instantiated and configured Policy instance ready for simulation execution.
    """
    if not isinstance(name, str):
        raise TypeError(f"Policy name must be a str, got {type(name).__name__}")
    stripped = name.strip()
    key = stripped.lower()
    if not key or key not in _REGISTRY:
        avail = ", ".join(list_policies())
        raise KeyError(f"Unknown policy '{name}'. Available policies: {avail}")

    _, policy_cls = _REGISTRY[key]
    return policy_cls(**kwargs)


def list_policies() -> list[str]:
    """
    1. The use of function:
       Returns a sorted list of unique canonical names of all registered scheduling policies.
    2. How does the specific function work:
       Iterates over the registered entries, extracts canonical names, deduplicates them, and
       returns them sorted lexicographically.
    3. Concepts used in the function:
       Registry introspection, lexicographical sorting, collection deduplication.
    4. Places where this function is imported and used:
       Exported in schedsim.policies, used by CLI choices, diagnostic error messages in
       get_policy, and unit test suites.
    5. Final output of the function, before and after:
       Before: Current internal registry mappings.
       After: Sorted list of strings containing canonical policy names.
    """
    return sorted({canonical for canonical, _ in _REGISTRY.values()})


def _reset_registry() -> None:
    """
    1. The use of function:
       Resets the internal policy registry to standard baseline classical scheduling policies.
    2. How does the specific function work:
       Clears the registry dictionary and registers the default classical algorithms: FCFS,
       SJF, SRTF, RR, and RoundRobin.
    3. Concepts used in the function:
       State re-initialization, test isolation, fixture cleanup.
    4. Places where this function is imported and used:
       Called at module import time in registry.py, and invoked during test teardown in
       tests/test_policies.py to guarantee isolated test execution.
    5. Final output of the function, before and after:
       Before: Potentially mutated or extended registry dictionary.
       After: Registry restored to initial baseline with default classical policies.
    """
    _REGISTRY.clear()
    register_policy("FCFS", FCFSPolicy)
    register_policy("SJF", SJFPolicy)
    register_policy("SRTF", SRTFPolicy)
    register_policy("RR", RoundRobinPolicy)
    register_policy("RoundRobin", RoundRobinPolicy)


# Pre-register default policies at import time
_reset_registry()
