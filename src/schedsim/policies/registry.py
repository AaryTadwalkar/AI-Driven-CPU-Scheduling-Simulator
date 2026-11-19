"""
1. What this file does:
   Implements the centralized policy registry and factory for CPU scheduling policies, including
   registration of Category 1 SelectorPolicy aliases ('DT' and 'RF').
2. Overall design:
   Maintains a primary registry mapping case-insensitive policy names to Policy subclasses and an
   alias table supporting parameterized factory resolution for Category 1 ML selectors. Provides
   factory instantiation with kwargs forwarding, dynamic policy registration, canonical policy
   listing, and descriptive error handling.
3. Elements in the file:
   - _REGISTRY: Dictionary storing primary canonical policy classes.
   - _ALIASES: Dictionary storing parameterized alias mappings.
   - register_policy: Registers a Policy subclass under a given name.
   - register_alias: Registers a parameterized alias pointing to a Policy subclass.
   - get_policy: Factory function retrieving and instantiating a policy or alias by name.
   - list_policies: Returns sorted unique canonical policy names.
   - _reset_registry: Helper function restoring baseline classical policies and default aliases.
4. Final output of the file:
   Exports get_policy, list_policies, register_policy, and register_alias for simulations.
"""

from typing import Any

import config
from schedsim.policies.base import Policy
from schedsim.policies.fcfs import FCFSPolicy
from schedsim.policies.round_robin import RoundRobinPolicy
from schedsim.policies.sjf import SJFPolicy
from schedsim.policies.srtf import SRTFPolicy

_REGISTRY: dict[str, tuple[str, type[Policy]]] = {}
_ALIASES: dict[str, tuple[str, type[Policy], dict[str, Any]]] = {}


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


def register_alias(name: str, policy_cls: type[Policy], **default_kwargs: Any) -> None:
    """
    1. The use of function:
       Registers a parameterized policy alias in the registry.
    2. How does the specific function work:
       Validates name and policy_cls, normalizes name to lowercase, and stores a tuple of
       (canonical_name, policy_cls, default_kwargs) in _ALIASES.
    3. Concepts used in the function:
       Alias mapping, factory parameter preset, case-insensitive indexing.
    4. Places where this function is imported and used:
       Used to register DT and RF aliases pointing to SelectorPolicy.
    5. Final output of the function, before and after:
       Before: _ALIASES without the specified mapping.
       After: _ALIASES updated with the registered alias mapping.
    """
    if not isinstance(name, str):
        raise TypeError(f"Policy alias name must be a str, got {type(name).__name__}")
    stripped = name.strip()
    if not stripped:
        raise ValueError("Policy alias name cannot be empty")
    if not isinstance(policy_cls, type) or not issubclass(policy_cls, Policy):
        raise TypeError("policy_cls must be a subclass of Policy")

    _ALIASES[stripped.lower()] = (stripped, policy_cls, default_kwargs)


def get_policy(name: str, /, **kwargs: Any) -> Policy:
    """
    1. The use of function:
       Retrieves and instantiates a registered scheduling policy by name with optional kwargs.
    2. How does the specific function work:
       Validates that name is a string. Normalizes the name to lowercase stripped format.
       Checks _REGISTRY first; if present, instantiates policy_cls with kwargs.
       Next checks _ALIASES; if present, merges default kwargs with user kwargs and instantiates.
       If not found in either, raises KeyError with available policies.
    3. Concepts used in the function:
       Factory pattern, alias resolution, case-insensitive string matching, kwargs merging.
    4. Places where this function is imported and used:
       Exported in schedsim.policies, used by CLI simulation subcommand, benchmark runners,
       API endpoints, and unit test suites.
    5. Final output of the function, before and after:
       Before: Policy name string and optional constructor keyword arguments.
       After: Instantiated and configured Policy instance ready for simulation execution.
    """
    if not isinstance(name, str):
        raise TypeError(f"Policy name must be a str, got {type(name).__name__}")
    stripped = name.strip()
    key = stripped.lower()
    if key in _REGISTRY:
        _, policy_cls = _REGISTRY[key]
        return policy_cls(**kwargs)

    if key in ("dt", "rf", "selectorpolicy"):
        from schedsim.policies.selector import SelectorPolicy
        if key == "dt":
            return SelectorPolicy(model="dt", **kwargs)
        if key == "rf":
            return SelectorPolicy(model="rf", **kwargs)
        return SelectorPolicy(**kwargs)

    if key in ("qlearning", "rl", "rlpolicy", "q-learning", "q_learning"):
        from schedsim.rl.policy import RLPolicy
        if "agent" in kwargs:
            agent = kwargs.pop("agent")
            return RLPolicy(agent=agent, **kwargs)
        cs = kwargs.pop("cs_cost", 1)
        models_dir = kwargs.pop("models_dir", config.MODELS_DIR)
        return RLPolicy.from_cs_cost(cs_cost=cs, models_dir=models_dir, **kwargs)

    if key in ("pid_gbdt", "gbdt", "gbdtrankerpolicy"):
        from schedsim.pidsel.gbdt_ranker import GBDTRankerPolicy
        models_dir = kwargs.pop("models_dir", config.MODELS_DIR)
        return GBDTRankerPolicy(models_dir=models_dir, **kwargs)

    if key in ("pid_nn", "pid_neural", "neuralcandidatepolicy"):
        from schedsim.pidsel.scorer_net import NeuralCandidatePolicy
        models_dir = kwargs.pop("models_dir", config.MODELS_DIR)
        return NeuralCandidatePolicy(models_dir=models_dir, **kwargs)

    if key in _ALIASES:
        _, policy_cls, def_kwargs = _ALIASES[key]
        merged = {**def_kwargs, **kwargs}
        return policy_cls(**merged)

    avail = ", ".join(list_policies(include_aliases=True))
    raise KeyError(f"Unknown policy '{name}'. Available policies: {avail}")


def list_policies(include_aliases: bool = False) -> list[str]:
    """
    1. The use of function:
       Returns a sorted list of unique canonical names of registered scheduling policies.
    2. How does the specific function work:
       Iterates over the registered entries, extracts canonical names, deduplicates them,
       optionally includes registered aliases if include_aliases is True, and returns them sorted.
    3. Concepts used in the function:
       Registry introspection, lexicographical sorting, collection deduplication.
    4. Places where this function is imported and used:
       Exported in schedsim.policies, used by CLI choices, diagnostic error messages in
       get_policy, and unit test suites.
    5. Final output of the function, before and after:
       Before: Current internal registry mappings and include_aliases flag.
       After: Sorted list of strings containing canonical policy names.
    """
    names = {canonical for canonical, _ in _REGISTRY.values()}
    if include_aliases:
        names |= {canonical for canonical, _, _ in _ALIASES.values()}
    return sorted(names)


def _reset_registry() -> None:
    """
    1. The use of function:
       Resets the internal policy registry to standard baseline classical scheduling policies.
    2. How does the specific function work:
       Clears the registry dictionary and registers the default classical algorithms: FCFS,
       SJF, SRTF, RR, and RoundRobin. Also resets Category 1 SelectorPolicy aliases.
    3. Concepts used in the function:
       State re-initialization, test isolation, fixture cleanup.
    4. Places where this function is imported and used:
       Called at module import time in registry.py, and invoked during test teardown in
       tests/test_policies.py to guarantee isolated test execution.
    5. Final output of the function, before and after:
       Before: Potentially mutated or extended registry dictionary.
       After: Registry restored to initial baseline with default classical policies and aliases.
    """
    _REGISTRY.clear()
    register_policy("FCFS", FCFSPolicy)
    register_policy("SJF", SJFPolicy)
    register_policy("SRTF", SRTFPolicy)
    register_policy("RR", RoundRobinPolicy)
    register_policy("RoundRobin", RoundRobinPolicy)

    _ALIASES.clear()
    try:
        from schedsim.pidsel.gbdt_ranker import GBDTRankerPolicy

        register_alias("PID_GBDT", GBDTRankerPolicy)
        register_alias("GBDT", GBDTRankerPolicy)
    except ImportError:
        pass
    try:
        from schedsim.pidsel.scorer_net import NeuralCandidatePolicy

        register_alias("PID_NN", NeuralCandidatePolicy)
        register_alias("PID_NEURAL", NeuralCandidatePolicy)
    except ImportError:
        pass


# Pre-register default policies at import time
_reset_registry()

__all__ = [
    "get_policy",
    "list_policies",
    "register_alias",
    "register_policy",
]
