"""
1. What this file does:
   Defines unit and property test suites for dataset split partition helpers in
   src/schedsim/workloads/splits.py.
2. Overall design:
   Validates partition boundaries, seed mapping accuracy, scenario balancing, partition purity,
   capacity limits, deterministic reproduction, and error raising on invalid inputs.
3. Elements in the file:
   - TestSplitSeedMapping: Tests seed to split resolution across boundaries and gaps.
   - TestGenerateSplitWorkloads: Tests batch generation, scenario balance, and determinism.
   - TestSplitErrorHandling: Tests boundary exceedance and type/value exception handling.
4. Final output of the file:
   Pytest assertion results validating complete conformance to Section 3.4 seed isolation rules.
"""

import pytest

from schedsim.workloads.scenarios import ALL_SCENARIOS
from schedsim.workloads.splits import (
    ALL_SPLITS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    generate_split_workloads,
    get_split_for_seed,
)


class TestSplitSeedMapping:
    """
    1. The use of class:
       Unit tests validating get_split_for_seed boundary mappings and error raising.
    2. What the class contain:
       Test methods checking boundaries of train, val, and test ranges, plus gaps and invalid types.
    3. Technologies used and why:
       pytest assertions and pytest.raises for exception verification.
    4. Key design and overall design of the class:
       Ensures zero seed ambiguity across the three partitions defined in Section 3.4.
    """

    def test_train_seed_boundaries(self) -> None:
        """
        1. The use of function:
           Tests that get_split_for_seed correctly resolves train partition boundaries.
        2. How does the specific function work:
           Asserts seeds at lower, middle, and upper bounds of TRAIN_SEED_RANGE map to 'train'.
        3. Concepts used in the function:
           Range boundary testing, partition membership.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Seed integer.
           After: Returns 'train'.
        """
        assert get_split_for_seed(0) == SPLIT_TRAIN
        assert get_split_for_seed(50000) == SPLIT_TRAIN
        assert get_split_for_seed(99999) == SPLIT_TRAIN

    def test_val_seed_boundaries(self) -> None:
        """
        1. The use of function:
           Tests that get_split_for_seed correctly resolves validation partition boundaries.
        2. How does the specific function work:
           Asserts seeds at lower, middle, and upper bounds of VAL_SEED_RANGE map to 'val'.
        3. Concepts used in the function:
           Range boundary testing, partition membership.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Seed integer.
           After: Returns 'val'.
        """
        assert get_split_for_seed(100000) == SPLIT_VAL
        assert get_split_for_seed(100250) == SPLIT_VAL
        assert get_split_for_seed(100499) == SPLIT_VAL

    def test_test_seed_boundaries(self) -> None:
        """
        1. The use of function:
           Tests that get_split_for_seed correctly resolves test partition boundaries.
        2. How does the specific function work:
           Asserts seeds at lower, middle, and upper bounds of TEST_SEED_RANGE map to 'test'.
        3. Concepts used in the function:
           Range boundary testing, partition membership.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Seed integer.
           After: Returns 'test'.
        """
        assert get_split_for_seed(200000) == SPLIT_TEST
        assert get_split_for_seed(200500) == SPLIT_TEST
        assert get_split_for_seed(200999) == SPLIT_TEST

    def test_gaps_and_out_of_bounds_seeds_raise_value_error(self) -> None:
        """
        1. The use of function:
           Verifies that seeds falling into unallocated gaps raise ValueError.
        2. How does the specific function work:
           Passes negative numbers, gap between val and test (100500..199999), and above test range.
        3. Concepts used in the function:
           Partition purity, gap rejection.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Unmapped seed.
           After: Raises ValueError with informative message.
        """
        unmapped_seeds = [-1, -100, 100500, 150000, 199999, 201000, 999999]
        for s in unmapped_seeds:
            with pytest.raises(ValueError, match="does not belong to any defined split"):
                get_split_for_seed(s)

    def test_invalid_seed_types_raise_type_error(self) -> None:
        """
        1. The use of function:
           Verifies that non-integer seed arguments raise TypeError.
        2. How does the specific function work:
           Passes strings, floats, None, and booleans to get_split_for_seed.
        3. Concepts used in the function:
           Runtime type checking, boolean subclass guard.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Non-int input.
           After: Raises TypeError.
        """
        for bad in ["42", 12.34, None, True, False]:
            with pytest.raises(TypeError, match="Seed must be an int"):
                get_split_for_seed(bad)  # type: ignore[arg-type]


class TestGenerateSplitWorkloads:
    """
    1. The use of class:
       Unit tests validating generate_split_workloads behavior.
    2. What the class contain:
       Tests for scenario balancing, split purity, deterministic reproduction, and process counts.
    3. Technologies used and why:
       pytest for test orchestration and collection inspection.
    4. Key design and overall design of the class:
       Ensures balanced synthetic workload generation across splits.
    """

    def test_generate_split_workloads_scenario_balance(self) -> None:
        """
        1. The use of function:
           Verifies that generated workloads cycle evenly through all 9 canonical scenarios.
        2. How does the specific function work:
           Generates 18 workloads (exactly 2 cycles of 9 scenarios) and checks counts per scenario.
        3. Concepts used in the function:
           Class balance, round-robin scenario cycling.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Unverified batch.
           After: Asserts each scenario appears exactly 2 times.
        """
        workloads = generate_split_workloads(split="val", n_workloads=18)
        assert len(workloads) == 18
        scenario_counts: dict[str, int] = {}
        for w in workloads:
            scenario_counts[w.scenario] = scenario_counts.get(w.scenario, 0) + 1

        assert len(scenario_counts) == len(ALL_SCENARIOS)
        for sc in ALL_SCENARIOS:
            assert scenario_counts[sc] == 2

    def test_generate_split_workloads_split_purity(self) -> None:
        """
        1. The use of function:
           Verifies that all generated workloads strictly possess seeds within target split.
        2. How does the specific function work:
           Generates workloads from train, val, and test splits and asserts get_split_for_seed.
        3. Concepts used in the function:
           Seed purity, partition isolation.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Workloads generated from target split.
           After: Asserts every workload's seed maps back to the specified split.
        """
        for sp in ALL_SPLITS:
            w_list = generate_split_workloads(split=sp, n_workloads=9)
            for w in w_list:
                assert get_split_for_seed(w.seed) == sp
                assert w.workload_id.startswith(f"{sp}_")

    def test_generate_split_workloads_determinism(self) -> None:
        """
        1. The use of function:
           Verifies deterministic reproduction for identical parameters.
        2. How does the specific function work:
           Calls generate_split_workloads twice with identical args and asserts equality.
        3. Concepts used in the function:
           Reproducibility, value object equality.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Two separate invocations.
           After: Asserts w1 == w2 across all elements.
        """
        w1 = generate_split_workloads(split="train", n_workloads=10, seed_offset=50)
        w2 = generate_split_workloads(split="train", n_workloads=10, seed_offset=50)
        assert w1 == w2

    def test_generate_split_workloads_default_n_processes(self) -> None:
        """
        1. The use of function:
           Verifies that default_n_processes pins process count across all generated workloads.
        2. How does the specific function work:
           Calls generate_split_workloads with default_n_processes=5 and asserts
           len(w.processes) == 5.
        3. Concepts used in the function:
           Process count parameterization.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Parameter default_n_processes=5.
           After: All workloads have exactly 5 processes.
        """
        workloads = generate_split_workloads(
            split="val", n_workloads=5, default_n_processes=5
        )
        for w in workloads:
            assert len(w.processes) == 5


class TestSplitErrorHandling:
    """
    1. The use of class:
       Tests edge case and error boundary conditions for split generation.
    2. What the class contain:
       Tests for capacity exceedance, invalid types, invalid split names, and negative offsets.
    3. Technologies used and why:
       pytest.raises for defensive boundary validation.
    4. Key design and overall design of the class:
       Guarantees robust error signaling on out-of-spec calls.
    """

    def test_unknown_split_raises_value_error(self) -> None:
        """
        1. The use of function:
           Tests that uncatalogued split names raise ValueError.
        2. How does the specific function work:
           Passes invalid string to generate_split_workloads.
        3. Concepts used in the function:
           Input validation.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Invalid split name.
           After: Raises ValueError.
        """
        with pytest.raises(ValueError, match="Unknown split"):
            generate_split_workloads(split="invalid_split", n_workloads=5)

    def test_capacity_exceeded_raises_value_error(self) -> None:
        """
        1. The use of function:
           Tests that requesting more workloads than split capacity raises ValueError.
        2. How does the specific function work:
           Requests 501 workloads from validation split (capacity is 500: 100000..100499).
        3. Concepts used in the function:
           Capacity enforcement, seed bound check.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: 501 workloads requested from val split.
           After: Raises ValueError with capacity details.
        """
        with pytest.raises(ValueError, match="exceeds split 'val' capacity"):
            generate_split_workloads(split="val", n_workloads=501)

    def test_invalid_parameter_types_raise_type_error(self) -> None:
        """
        1. The use of function:
           Tests type checks on split, n_workloads, seed_offset, and default_n_processes.
        2. How does the specific function work:
           Passes invalid types (int for split, float/bool for counts/offsets).
        3. Concepts used in the function:
           Runtime type checking.
        4. Places where this function is imported and used:
           Executed by pytest runner.
        5. Final output of the function, before and after:
           Before: Invalid types passed.
           After: Raises TypeError.
        """
        with pytest.raises(TypeError, match="Split must be a str"):
            generate_split_workloads(split=123, n_workloads=5)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="n_workloads must be an int"):
            generate_split_workloads(split="val", n_workloads=True)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="seed_offset must be an int"):
            generate_split_workloads(
                split="val", n_workloads=5, seed_offset="0"
            )  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="default_n_processes must be an int"):
            generate_split_workloads(
                split="val", n_workloads=5, default_n_processes=3.5
            )  # type: ignore[arg-type]
