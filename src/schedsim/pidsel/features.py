"""
1. What this file does:
   Defines the 13 observable per-candidate and system context features for process-level scheduling
   (PID selection), along with vectorized rank normalization, feature extraction from simulation
   views, and neural batch padding utilities.
2. Overall design:
   Extracts candidate-level features entirely from the observable Markovian View state without any
   future lookahead, PID strings, or slot indices. Features fall into three groups: Process group
   (rem, burst, rem_burst_ratio, waited, is_last), Relative group (rank_rem, rank_waited,
   rem_min_ratio, rem_mean_ratio), and Context group (n_ready, sum_rem, cs_cost, pending_arrivals).
   Vectorized using NumPy to ensure permutation equivariance and high computational efficiency.
3. Elements in the file:
   - FEATURE_NAMES: Canonical tuple of the 13 feature string identifiers.
   - compute_normalized_rank: Pure NumPy symmetric fractional rank normalization in [0, 1].
   - extract_candidate_features: Extracts (n_candidates, 13) feature matrix and candidate PIDs.
   - pad_and_stack_features: Pads variable candidate sets to fixed shape (B, max_c, 13) with mask.
4. Final output of the file:
   Exports FEATURE_NAMES, compute_normalized_rank, extract_candidate_features, and
   pad_and_stack_features for dataset construction, GBDT ranking, and neural candidate scoring.
"""

from collections.abc import Sequence

import numpy as np

from schedsim.domain.engine import View

FEATURE_NAMES: tuple[str, ...] = (
    "rem",
    "burst",
    "rem_burst_ratio",
    "waited",
    "is_last",
    "rank_rem",
    "rank_waited",
    "rem_min_ratio",
    "rem_mean_ratio",
    "n_ready",
    "sum_rem",
    "cs_cost",
    "pending_arrivals",
)


def compute_normalized_rank(values: np.ndarray | Sequence[float | int]) -> np.ndarray:
    """
    1. The use of function:
       Calculates symmetric fractional ranks for a numeric sequence, normalized to [0, 1].
    2. How does the specific function work:
       Converts input to a 1D float64 NumPy array. If length N <= 1, returns zeros of length N.
       Otherwise computes pairwise comparisons: count of values strictly less and count of values
       equal to each element. Applies symmetric rank formula:
       rank = (sum(v < x) + 0.5 * (sum(v == x) - 1)) / (N - 1).
       Tied elements receive identical symmetric midpoint ranks (e.g., all tied receives 0.5).
    3. Concepts used in the function:
       Fractional ranking, pairwise broadcasting, tie preservation, symmetric [0, 1] normalization.
    4. Places where this function is imported and used:
       Imported by extract_candidate_features, dataset verification suites, and model trainers.
    5. Final output of the function, before and after:
       Before: 1D array of arbitrary real values.
       After: 1D float32 array of normalized ranks in [0, 1].
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"values must be 1-dimensional, got shape {arr.shape}.")

    n = len(arr)
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    # Pairwise comparison broadcasting: shape (n, n)
    less = (arr[:, None] > arr[None, :]).sum(axis=1)
    equal = (arr[:, None] == arr[None, :]).sum(axis=1)
    ranks = (less + 0.5 * (equal - 1.0)) / (n - 1.0)
    return ranks.astype(np.float32)


def extract_candidate_features(
    view: View,
    cs_cost: int = 0,
    pending_arrivals: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    1. The use of function:
       Extracts the 13 observable per-candidate and system context features from a simulation View.
    2. How does the specific function work:
       Validates view, cs_cost >= 0, and resolved pending_arrivals >= 0. If view has no ready
       processes, returns an empty (0, 13) float32 matrix and empty PID list. Otherwise extracts
       candidate remaining, burst, waited, and is_last indicators. Computes relative group
       (rank_rem, rank_waited, rem_min_ratio, rem_mean_ratio) and context group (n_ready, sum_rem,
       cs_cost, min(pending_arrivals, 10)). Stacks into (n_candidates, 13) float32 array.
    3. Concepts used in the function:
       Markovian state feature extraction, permutation-equivariant relative ratios, bounded context.
    4. Places where this function is imported and used:
       Imported by build_pid_dataset, GBDT ranker, neural candidate scorer, and test suites.
    5. Final output of the function, before and after:
       Before: Simulation View snapshot, integer cs_cost, and optional pending arrivals count.
       After: Tuple of ((n_candidates, 13) float32 array, list of candidate PID strings).
    """
    if not isinstance(view, View):
        raise TypeError(f"view must be a View instance, got {type(view).__name__}.")

    if isinstance(cs_cost, bool) or not isinstance(cs_cost, int) or cs_cost < 0:
        raise ValueError(f"cs_cost must be a non-negative int, got {cs_cost}.")

    if pending_arrivals is None:
        pending = getattr(view, "pending_arrivals", 0)
    else:
        pending = pending_arrivals

    if isinstance(pending, bool) or not isinstance(pending, int) or pending < 0:
        raise ValueError(f"pending_arrivals must be a non-negative int, got {pending}.")

    if not view.ready:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), []

    candidate_pids = [p.pid for p in view.ready]
    n = len(candidate_pids)

    rem = np.array([p.remaining for p in view.ready], dtype=np.float64)
    burst = np.array([p.burst for p in view.ready], dtype=np.float64)
    waited = np.array([p.waited for p in view.ready], dtype=np.float64)

    is_last = np.array(
        [
            1.0 if (view.last_pid is not None and p.pid == view.last_pid) else 0.0
            for p in view.ready
        ],
        dtype=np.float32,
    )

    # Process group
    f_rem = np.log1p(rem).astype(np.float32)
    f_burst = np.log1p(burst).astype(np.float32)
    f_rem_burst_ratio = (rem / burst).astype(np.float32)
    f_waited = np.log1p(waited).astype(np.float32)
    f_is_last = is_last

    # Relative group
    f_rank_rem = compute_normalized_rank(rem)
    f_rank_waited = compute_normalized_rank(waited)
    min_rem = np.min(rem)
    mean_rem = np.mean(rem)
    f_rem_min_ratio = (rem / min_rem).astype(np.float32)
    f_rem_mean_ratio = (rem / mean_rem).astype(np.float32)

    # Context group
    f_n_ready = np.full(n, np.log1p(n), dtype=np.float32)
    f_sum_rem = np.full(n, np.log1p(np.sum(rem)), dtype=np.float32)
    f_cs_cost = np.full(n, float(cs_cost), dtype=np.float32)
    f_pending = np.full(n, float(min(pending, 10)), dtype=np.float32)

    features = np.column_stack(
        [
            f_rem,
            f_burst,
            f_rem_burst_ratio,
            f_waited,
            f_is_last,
            f_rank_rem,
            f_rank_waited,
            f_rem_min_ratio,
            f_rem_mean_ratio,
            f_n_ready,
            f_sum_rem,
            f_cs_cost,
            f_pending,
        ]
    ).astype(np.float32)

    return features, candidate_pids


def pad_and_stack_features(
    batch_features: Sequence[np.ndarray],
    max_candidates: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1. The use of function:
       Pads variable-length candidate feature matrices into uniform tensors with boolean masks.
    2. How does the specific function work:
       Validates input batch sequence. Computes or validates max_candidates dimension. Allocates
       a zeros array of shape (B, max_candidates, 13) and boolean mask of shape (B, max_candidates).
       Copies each candidate feature matrix up to its candidate count, setting corresponding mask
       entries to True.
    3. Concepts used in the function:
       Batch tensor padding, attention masking, neural network input standardization.
    4. Places where this function is imported and used:
       Imported by dataset builders, neural candidate scorer training, and inference pipelines.
    5. Final output of the function, before and after:
       Before: Sequence of 2D candidate feature matrices of varying row counts.
       After: Tuple of (padded_features [B, max_c, 13] float32, mask [B, max_c] bool).
    """
    if not isinstance(batch_features, Sequence):
        raise TypeError(
            f"batch_features must be a Sequence, got {type(batch_features).__name__}."
        )

    b = len(batch_features)
    d = len(FEATURE_NAMES)

    if b == 0:
        max_c = max_candidates if max_candidates is not None else 0
        if max_c < 0:
            raise ValueError(f"max_candidates must be non-negative, got {max_c}.")
        return np.zeros((0, max_c, d), dtype=np.float32), np.zeros((0, max_c), dtype=bool)

    sample_lens = [len(f) for f in batch_features]
    max_observed = max(sample_lens)

    if max_candidates is None:
        target_max_c = max_observed
    else:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
            raise TypeError(
                f"max_candidates must be int or None, got {type(max_candidates).__name__}."
            )
        if max_candidates < max_observed:
            raise ValueError(
                f"max_candidates ({max_candidates}) is smaller than maximum observed candidate "
                f"count ({max_observed})."
            )
        target_max_c = max_candidates

    padded = np.zeros((b, target_max_c, d), dtype=np.float32)
    mask = np.zeros((b, target_max_c), dtype=bool)

    for i, feats in enumerate(batch_features):
        if not isinstance(feats, np.ndarray):
            raise TypeError(f"batch element {i} must be a np.ndarray, got {type(feats).__name__}.")
        k = len(feats)
        if k > 0:
            if feats.shape[1] != d:
                raise ValueError(
                    f"Feature dimension mismatch at index {i}: expected {d}, got {feats.shape[1]}."
                )
            padded[i, :k, :] = feats
            mask[i, :k] = True

    return padded, mask
