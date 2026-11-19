"""
1. What this file does:
   Implements the SelectorPolicy meta-policy for Category 1 ML models (Decision Tree and
   Random Forest), selecting a specialized classical scheduling algorithm once at workload start
   and delegating discrete stepping execution to the chosen delegate policy.
2. Overall design:
   Inherits from abstract Policy base class. At simulation initialization (on_start hook), extracts
   features and queries the Category 1 recommender API for the designated model type ('dt' or 'rf')
   and benchmark setting ('S0' or 'S1'). Instantiates the selected classical policy (FCFS, SJF, RR)
   via the central policy factory, delegating subsequent per-tick dispatch decisions with zero
   per-decision latency overhead.
3. Elements in the file:
   - SelectorPolicy: Base meta-policy wrapping Decision Tree or Random Forest recommendation.
   - DTPolicy: Concrete convenience subclass pre-configured for Decision Tree selection.
   - RFPolicy: Concrete convenience subclass pre-configured for Random Forest selection.
4. Final output of the file:
   Exports SelectorPolicy, DTPolicy, and RFPolicy for policy registry, simulations, and tests.
"""

from collections.abc import Sequence
from typing import Any, Optional

import config
from schedsim.domain.engine import View
from schedsim.domain.process import Process, validate_processes
from schedsim.policies.base import Policy


class SelectorPolicy(Policy):
    """
    1. The use of class:
       Provides a unified Category 1 policy adapter that queries Decision Tree or Random Forest
       recommenders upfront at workload start and delegates execution to the predicted policy.
    2. What the class contains:
       - model: String identifier of the recommender model family ('dt' or 'rf').
       - setting: String identifier of the evaluation setting ('S0' or 'S1').
       - chosen_algorithm: String name of the winning algorithm selected on_start (or None).
       - recommendation: Dictionary containing predicted algorithm, probabilities, and features.
       - _delegate: Instantiated Policy object executing discrete per-tick scheduling.
       - __init__: Initializes model and setting configurations.
       - on_start: Runs upfront recommender inference and configures the delegate policy.
       - select: Delegates process selection to the active delegate policy.
       - decision_cost_us: Returns 0.0 µs as Category 1 inference is charged only once at start.
       - reset: Clears delegate and internal state between simulation runs.
    3. Technologies used and why:
       - Polymorphic Policy inheritance ensuring seamless compatibility with Engine and run().
       - Delegation pattern to decouple one-time upfront ML inference from unit-step dispatch.
    4. Key design and overall design of the class:
       Implements Section 1 Category 1 semantics: charged once at start, free per decision,
       seamlessly wrapping classical schedulers according to learned classification boundaries.
    """

    def __init__(
        self,
        model: str = "dt",
        setting: str = "S1",
        name: Optional[str] = None,
    ) -> None:
        """
        1. The use of function:
           Initializes a SelectorPolicy instance with specified model type and benchmark setting.
        2. How does the specific function work:
           Validates that model is 'dt' or 'rf' and setting is a registered key in config.SETTINGS.
           Constructs a default human-readable name if omitted, and initializes internal state.
        3. Concepts used in the function:
           Constructor initialization, parameter validation, defensive defaults.
        4. Places where this function is imported and used:
           Directly instantiated by callers or dynamically constructed via policy registry factory.
        5. Final output of the function, before and after:
           Before: Uninitialized SelectorPolicy instance.
           After: Configured SelectorPolicy instance ready for on_start() and simulation execution.
        """
        if not isinstance(model, str):
            raise TypeError(f"model must be a str, got {type(model).__name__}.")
        norm_model = model.lower().strip()
        if norm_model in ("dt", "decision_tree"):
            self.model: str = "dt"
        elif norm_model in ("rf", "random_forest"):
            self.model = "rf"
        else:
            raise ValueError(f"Unknown model '{model}'. Must be 'dt' or 'rf'.")

        if not isinstance(setting, str):
            raise TypeError(f"setting must be a str, got {type(setting).__name__}.")
        norm_setting = setting.upper().strip()
        if norm_setting not in config.SETTINGS:
            valid_settings = ", ".join(config.SETTINGS.keys())
            raise ValueError(
                f"Unknown setting '{setting}'. Must be one of: {valid_settings}."
            )
        self.setting: str = norm_setting

        policy_name = name or f"Selector_{self.model.upper()}_{self.setting}"
        super().__init__(name=policy_name)

        self.chosen_algorithm: Optional[str] = None
        self.recommendation: Optional[dict[str, Any]] = None
        self._delegate: Optional[Policy] = None

    def on_start(self, processes: Sequence[Process]) -> float:
        """
        1. The use of function:
           Runs upfront ML recommendation, selects the target algorithm, and builds the delegate.
        2. How does the specific function work:
           Validates the process sequence, calls recommend() with self.model and self.setting,
           stores the resulting algorithm and probabilities, instantiates the delegate policy
           via get_policy(), propagates on_start to the delegate, and returns upfront latency.
        3. Concepts used in the function:
           Upfront recommender inference, factory delegation, lifecycle hook execution.
        4. Places where this function is imported and used:
           Invoked by Engine runner (run()) prior to stepping discrete simulation timeline.
        5. Final output of the function, before and after:
           Before: Unset delegate and unselected policy state.
           After: Delegate instantiated and ready for select(); returns upfront µs (0.0).
        """
        if not isinstance(processes, Sequence) or isinstance(processes, (str, bytes)):
            raise TypeError(
                f"processes must be a Sequence[Process], got {type(processes).__name__}."
            )
        proc_list = list(processes)
        validate_processes(proc_list)

        from schedsim.ml.evaluate_tree import recommend
        from schedsim.policies.registry import get_policy

        rec = recommend(proc_list, model=self.model, setting=self.setting)
        self.recommendation = rec
        self.chosen_algorithm = str(rec["algorithm"])
        self._delegate = get_policy(self.chosen_algorithm)
        self._delegate.on_start(proc_list)
        from schedsim.overhead import get_startup_cost_us

        return get_startup_cost_us(self.model, len(proc_list))

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the process ID to execute for the current simulation tick.
        2. How does the specific function work:
           Asserts that on_start() has been executed to configure the delegate policy, then
           delegates directly to self._delegate.select(view).
        3. Concepts used in the function:
           Delegation pattern, precondition validation, polymorphic dispatch.
        4. Places where this function is imported and used:
           Invoked on each simulation tick by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Observable simulation View.
           After: Returns selected PID string determined by the delegate policy.
        """
        if self._delegate is None:
            raise RuntimeError(
                f"{self.name}: select() invoked before on_start() configured the delegate policy."
            )
        return self._delegate.select(view)

    def decision_cost_us(self, view: View) -> float:
        """
        1. The use of function:
           Returns the per-decision inference latency in microseconds for non-forced decisions.
        2. How does the specific function work:
           Returns 0.0 µs as Category 1 models are charged once at start, never per decision.
        3. Concepts used in the function:
           Zero per-decision overhead guarantee, overhead model contract compliance.
        4. Places where this function is imported and used:
           Invoked by Engine when tick_us is configured for overhead accounting.
        5. Final output of the function, before and after:
           Before: Observable simulation View.
           After: Returns 0.0 µs float latency.
        """
        return 0.0

    def reset(self) -> None:
        """
        1. The use of function:
           Resets policy and internal delegate state between simulation runs.
        2. How does the specific function work:
           Calls reset() on delegate if active, clears chosen algorithm, recommendation,
           and delegate references.
        3. Concepts used in the function:
           State clearing, lifecycle management, test isolation.
        4. Places where this function is imported and used:
           Invoked between simulation runs or when resetting stateful simulation environments.
        5. Final output of the function, before and after:
           Before: Configured delegate and historical recommendation state.
           After: Cleared state ready for next on_start() invocation.
        """
        if self._delegate is not None:
            self._delegate.reset()
        self._delegate = None
        self.chosen_algorithm = None
        self.recommendation = None


class DTPolicy(SelectorPolicy):
    """
    1. The use of class:
       Convenience specialization of SelectorPolicy pre-configured for Decision Tree selection.
    2. What the class contains:
       - __init__: Initializes SelectorPolicy with model='dt' and default name 'DT'.
    3. Technologies used and why:
       Subclassing to enable simple, parameter-free registration and factory instantiation.
    4. Key design and overall design of the class:
       Provides clean alias mapping for registry lookup 'DT' under the Category 1 architecture.
    """

    def __init__(
        self,
        setting: str = "S1",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        1. The use of function:
           Initializes a DTPolicy instance.
        2. How does the specific function work:
           Forwards model='dt', setting, and name (default 'DT') to SelectorPolicy constructor.
        3. Concepts used in the function:
           Constructor chaining, specialized subclassing.
        4. Places where this function is imported and used:
           Instantiated via get_policy('DT') or direct import.
        5. Final output of the function, before and after:
           Before: Uninitialized DTPolicy.
           After: DTPolicy instance configured with model='dt'.
        """
        super().__init__(model="dt", setting=setting, name=name or "DT")


class RFPolicy(SelectorPolicy):
    """
    1. The use of class:
       Convenience specialization of SelectorPolicy pre-configured for Random Forest selection.
    2. What the class contains:
       - __init__: Initializes SelectorPolicy with model='rf' and default name 'RF'.
    3. Technologies used and why:
       Subclassing to enable simple, parameter-free registration and factory instantiation.
    4. Key design and overall design of the class:
       Provides clean alias mapping for registry lookup 'RF' under the Category 1 architecture.
    """

    def __init__(
        self,
        setting: str = "S1",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        1. The use of function:
           Initializes an RFPolicy instance.
        2. How does the specific function work:
           Forwards model='rf', setting, and name (default 'RF') to SelectorPolicy constructor.
        3. Concepts used in the function:
           Constructor chaining, specialized subclassing.
        4. Places where this function is imported and used:
           Instantiated via get_policy('RF') or direct import.
        5. Final output of the function, before and after:
           Before: Uninitialized RFPolicy.
           After: RFPolicy instance configured with model='rf'.
        """
        super().__init__(model="rf", setting=setting, name=name or "RF")


__all__ = [
    "DTPolicy",
    "RFPolicy",
    "SelectorPolicy",
]
