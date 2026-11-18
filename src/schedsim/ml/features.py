"""
1. What this file does:
   Implements domain-informed workload feature extraction for the Decision Tree policy recommender,
   computing structural, statistical, and contention features directly and solely from raw
   process specifications without running simulations.
2. Overall design:
   Accepts a sequence of Process domain entities, validates domain invariants, and extracts exactly
   12 predictive features characterizing process counts, burst duration moments (mean, std, CV,
   extrema, max-over-median skew), arrival dispersion (mean/std inter-arrival, arrival span),
   load density (load ratio), and contention intensity (fraction of simultaneous arrivals).
3. Elements in the file:
   - FEATURE_NAMES: Canonical ordered tuple of the 12 extracted feature identifiers.
   - extract_features: Computes and returns feature dictionary mapping feature names to floats.
   - extract_features_vector: Returns an ordered 1D numpy array suitable for scikit-learn models.
4. Final output of the file:
   Exports FEATURE_NAMES, extract_features, and extract_features_vector for ML dataset building,
   model training, and real-time policy recommendation inference.
"""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from schedsim.domain.process import Process, validate_processes

FEATURE_NAMES: tuple[str, ...] = (
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
)


def extract_features(processes: Sequence[Process]) -> dict[str, float]:
    """
    1. The use of function:
       Extracts 12 descriptive workload features from an arrival/burst process collection.
    2. How does the specific function work:
       Validates input collection using `validate_processes`, extracts burst and arrival sequences,
       computes burst distribution statistics (mean, std, cv, min, max, max/median), arrival
       dispersion metrics (interarrival deltas, span), system load ratio, and the fraction of
       processes sharing an arrival timestamp with another process.
    3. Concepts used in the function:
       Statistical feature engineering, moments calculation, contention density estimation.
    4. Places where this function is imported and used:
       Imported by dataset_builder.py, evaluate_tree.py, recommend() API, and unit test suites.
    5. Final output of the function, before and after:
       Before: Sequence of validated Process objects.
       After: Dictionary containing all 12 feature values keyed by canonical names in FEATURE_NAMES.
    """
    if not isinstance(processes, Sequence) or isinstance(processes, (str, bytes)):
        raise TypeError(
            f"processes must be a Sequence[Process], got {type(processes).__name__}."
        )
    proc_list = list(processes)
    validate_processes(proc_list)

    n = len(proc_list)
    bursts = [float(p.burst) for p in proc_list]
    arrivals = [int(p.arrival) for p in proc_list]

    mean_burst = float(np.mean(bursts))
    std_burst = float(np.std(bursts))
    cv_burst = float(std_burst / mean_burst) if mean_burst > 0.0 else 0.0
    max_burst = float(np.max(bursts))
    min_burst = float(np.min(bursts))
    median_burst = float(np.median(bursts))
    max_over_median = float(max_burst / median_burst) if median_burst > 0.0 else 1.0

    sorted_arrivals = sorted(arrivals)
    if n > 1:
        deltas = [
            float(sorted_arrivals[i] - sorted_arrivals[i - 1])
            for i in range(1, n)
        ]
        mean_interarrival = float(np.mean(deltas))
        std_interarrival = float(np.std(deltas)) if len(deltas) > 1 else 0.0
    else:
        mean_interarrival = 0.0
        std_interarrival = 0.0

    arrival_span = float(sorted_arrivals[-1] - sorted_arrivals[0])
    total_burst = float(sum(bursts))
    load_ratio = float(total_burst / (arrival_span + 1.0))

    if n > 1:
        arr_counts = Counter(sorted_arrivals)
        simultaneous_count = sum(c for c in arr_counts.values() if c >= 2)
        frac_simultaneous = float(simultaneous_count / n)
    else:
        frac_simultaneous = 0.0

    features: dict[str, float] = {
        "n": float(n),
        "mean_burst": mean_burst,
        "std_burst": std_burst,
        "cv_burst": cv_burst,
        "max_burst": max_burst,
        "min_burst": min_burst,
        "max_over_median_burst": max_over_median,
        "mean_interarrival": mean_interarrival,
        "std_interarrival": std_interarrival,
        "arrival_span": arrival_span,
        "load_ratio": load_ratio,
        "frac_simultaneous": frac_simultaneous,
    }
    return features


def extract_features_vector(processes: Sequence[Process]) -> np.ndarray:
    """
    1. The use of function:
       Extracts workload features as a 1D numpy array aligned with FEATURE_NAMES.
    2. How does the specific function work:
       Invokes `extract_features()` to compute the feature dictionary, and projects values into
       a 1D float64 numpy array preserving the exact column ordering defined in FEATURE_NAMES.
    3. Concepts used in the function:
       Vectorization, scikit-learn compatible input preparation, canonical column mapping.
    4. Places where this function is imported and used:
       Imported by Decision Tree inference pipeline, dataset builder, and unit test suites.
    5. Final output of the function, before and after:
       Before: Sequence of Process domain entities.
       After: 1D numpy array of shape (12,) with dtype float64.
    """
    feats = extract_features(processes)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float64)


__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "extract_features_vector",
]
