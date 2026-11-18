"""
1. What this file does:
   Tests workload feature extraction for the Decision Tree policy recommender, asserting exact
   mathematical correctness of all 12 feature calculations, edge cases, vector conversions, and
   defensive input validation.
2. Overall design:
   Verifies `extract_features` and `extract_features_vector` from `schedsim.ml.features` against
   hand-calculated reference fixtures, single-process workloads, simultaneous arrivals, staggered
   arrivals, and invalid input types.
3. Elements in the file:
   - test_feature_names_constant: Validates canonical feature names list.
   - test_extract_features_known_values: Validates calculations against hand-derived values.
   - test_extract_features_single_process: Validates single process edge cases.
   - test_extract_features_simultaneous_scenario: Validates simultaneous arrival metrics.
   - test_extract_features_staggered_scenario: Validates staggered arrival deltas.
   - test_extract_features_vector_format: Validates numpy vector shape, dtype, and ordering.
   - test_extract_features_validation_errors: Validates exception raising on invalid inputs.
4. Final output of the file:
   Pytest test suite validating feature extraction integrity and numerical fidelity.
"""

import math

import numpy as np
import pytest

from schedsim.domain.process import Process
from schedsim.ml.features import FEATURE_NAMES, extract_features, extract_features_vector


def test_feature_names_constant() -> None:
    """
    1. The use of function:
       Validates the canonical 12 feature names constant tuple.
    2. How does the specific function work:
       Asserts length is 12 and checks that all required features are present in the tuple.
    3. Concepts used in the function:
       Schema validation, constant integrity checking.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified FEATURE_NAMES constant.
       After: Asserts exact length 12 and presence of all required feature strings.
    """
    assert len(FEATURE_NAMES) == 12
    expected = {
        "n",
        "mean_burst",
        "std_burst",
        "cv_burst",
        "max_burst",
        "min_burst",
        "max_over_median_burst",
        "mean_interarrival",
        "std_interarrival",
        "arrival_span",
        "load_ratio",
        "frac_simultaneous",
    }
    assert set(FEATURE_NAMES) == expected


def test_extract_features_known_values() -> None:
    """
    1. The use of function:
       Tests feature extraction on a 3-process workload with hand-calculated values.
    2. How does the specific function work:
       Constructs P1(0, 4), P2(0, 8), P3(2, 6). Evaluates moments, arrival deltas, load ratio,
       and simultaneous fractions against analytical values.
    3. Concepts used in the function:
       Mathematical verification, moments testing, simultaneous arrival counting.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested feature dictionary output.
       After: Asserts exact or floating-point close matches for all 12 features.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=4),
        Process(pid="P2", arrival=0, burst=8),
        Process(pid="P3", arrival=2, burst=6),
    ]
    feats = extract_features(procs)

    assert feats["n"] == 3.0
    assert feats["mean_burst"] == 6.0
    expected_std = math.sqrt(((4 - 6) ** 2 + (8 - 6) ** 2 + (6 - 6) ** 2) / 3.0)
    assert pytest.approx(feats["std_burst"]) == expected_std
    assert pytest.approx(feats["cv_burst"]) == expected_std / 6.0
    assert feats["max_burst"] == 8.0
    assert feats["min_burst"] == 4.0
    assert pytest.approx(feats["max_over_median_burst"]) == 8.0 / 6.0

    # Arrivals: 0, 0, 2 -> deltas: 0, 2
    assert feats["mean_interarrival"] == 1.0
    assert pytest.approx(feats["std_interarrival"]) == 1.0
    assert feats["arrival_span"] == 2.0
    # Total burst = 18; load_ratio = 18 / (2 + 1) = 6.0
    assert feats["load_ratio"] == 6.0
    # Simultaneous: 2 processes at tick 0, 1 at tick 2 -> 2/3
    assert pytest.approx(feats["frac_simultaneous"]) == 2.0 / 3.0


def test_extract_features_single_process() -> None:
    """
    1. The use of function:
       Tests feature extraction boundary behavior on a single-process workload.
    2. How does the specific function work:
       Constructs a workload with 1 process and asserts all delta/spread statistics default
       gracefully to 0 without division by zero.
    3. Concepts used in the function:
       Edge-case handling, zero-variance boundary validation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested single process input.
       After: Asserts n=1, std=0, cv=0, deltas=0, span=0, frac_simultaneous=0.
    """
    procs = [Process(pid="P1", arrival=5, burst=10)]
    feats = extract_features(procs)

    assert feats["n"] == 1.0
    assert feats["mean_burst"] == 10.0
    assert feats["std_burst"] == 0.0
    assert feats["cv_burst"] == 0.0
    assert feats["max_burst"] == 10.0
    assert feats["min_burst"] == 10.0
    assert feats["max_over_median_burst"] == 1.0
    assert feats["mean_interarrival"] == 0.0
    assert feats["std_interarrival"] == 0.0
    assert feats["arrival_span"] == 0.0
    assert feats["load_ratio"] == 10.0
    assert feats["frac_simultaneous"] == 0.0


def test_extract_features_simultaneous_scenario() -> None:
    """
    1. The use of function:
       Tests feature extraction on completely simultaneous arrivals.
    2. How does the specific function work:
       Constructs 4 processes all arriving at tick 0. Asserts frac_simultaneous is exactly 1.0
       and arrival span is 0.0.
    3. Concepts used in the function:
       Simultaneous arrival detection, saturation fraction.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested simultaneous arrivals.
       After: Asserts frac_simultaneous == 1.0 and arrival_span == 0.0.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=2),
        Process(pid="P2", arrival=0, burst=4),
        Process(pid="P3", arrival=0, burst=6),
        Process(pid="P4", arrival=0, burst=8),
    ]
    feats = extract_features(procs)
    assert feats["frac_simultaneous"] == 1.0
    assert feats["arrival_span"] == 0.0
    assert feats["mean_interarrival"] == 0.0
    assert feats["std_interarrival"] == 0.0


def test_extract_features_staggered_scenario() -> None:
    """
    1. The use of function:
       Tests feature extraction on distinct staggered arrivals.
    2. How does the specific function work:
       Constructs processes arriving at [0, 5, 10, 15]. Asserts frac_simultaneous is 0.0 and
       mean interarrival is 5.0 with std 0.0.
    3. Concepts used in the function:
       Staggered arrival metrics, delta variance.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested staggered arrivals.
       After: Asserts frac_simultaneous == 0.0 and mean_interarrival == 5.0.
    """
    procs = [
        Process(pid=f"P{i + 1}", arrival=i * 5, burst=4)
        for i in range(4)
    ]
    feats = extract_features(procs)
    assert feats["frac_simultaneous"] == 0.0
    assert feats["arrival_span"] == 15.0
    assert feats["mean_interarrival"] == 5.0
    assert feats["std_interarrival"] == 0.0


def test_extract_features_vector_format() -> None:
    """
    1. The use of function:
       Tests `extract_features_vector` output shape, data type, and feature alignment.
    2. How does the specific function work:
       Invokes `extract_features_vector()`, validates returned array shape is (12,), dtype is
       float64, and values match `extract_features()` dictionary values in canonical order.
    3. Concepts used in the function:
       Array projection, vector integrity, scikit-learn compliance.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested extract_features_vector.
       After: Asserts 1D float64 array exactly matches FEATURE_NAMES order.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=3),
        Process(pid="P2", arrival=2, burst=7),
    ]
    vec = extract_features_vector(procs)
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (12,)
    assert vec.dtype == np.float64

    dict_feats = extract_features(procs)
    for idx, name in enumerate(FEATURE_NAMES):
        assert vec[idx] == dict_feats[name]


def test_extract_features_validation_errors() -> None:
    """
    1. The use of function:
       Tests input validation and error handling in `extract_features`.
    2. How does the specific function work:
       Tests non-sequence types, empty lists, and invalid Process objects, asserting that
       TypeError or ValueError is raised.
    3. Concepts used in the function:
       Defensive programming, input validation testing.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid inputs to extract_features.
       After: Asserts expected exceptions raised on invalid parameters.
    """
    with pytest.raises(TypeError, match="processes must be a Sequence"):
        extract_features("invalid_string")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot be empty"):
        extract_features([])

    with pytest.raises(TypeError, match="Expected Process instance"):
        extract_features([1, 2, 3])  # type: ignore[list-item]
