"""
1. What this file does:
   Defines dataset split constants, seed partition boundary validators, and balanced workload
   generators partitioned across Train, Validation, and Test seed allocations.
2. Overall design:
   Enforces Section 3.4 seed isolation rules:
   - Train split: seeds 0 to 99,999 (used for Decision Tree training and RL training episodes)
   - Validation split: seeds 100,000 to 100,499 (used for hyperparameter tuning & evaluation)
   - Test split: seeds 200,000 to 200,999 (held-out test benchmark, never touched before Stage 7)
   Provides round-robin scenario balancing across all 9 canonical workload scenarios.
3. Elements in the file:
   - SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST: Standard string constants for dataset partitions.
   - ALL_SPLITS: Immutable tuple containing all valid split identifiers.
   - get_split_for_seed: Maps a random seed integer to its designated dataset partition.
   - generate_split_workloads: Synthesizes balanced collections of Workload instances.
4. Final output of the file:
   Exports split constants, seed validation logic, and multi-scenario workload batch generators.
"""

from typing import Optional, Tuple

import config
from schedsim.workloads.generator import generate_workload
from schedsim.workloads.io import Workload
from schedsim.workloads.scenarios import ALL_SCENARIOS

# Canonical split identifiers
SPLIT_TRAIN: str = "train"
SPLIT_VAL: str = "val"
SPLIT_TEST: str = "test"
ALL_SPLITS: Tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)

# Partition boundary aliases from global configuration
TRAIN_SEED_RANGE: Tuple[int, int] = config.TRAIN_SEED_RANGE
VAL_SEED_RANGE: Tuple[int, int] = config.VAL_SEED_RANGE
TEST_SEED_RANGE: Tuple[int, int] = config.TEST_SEED_RANGE


def get_split_for_seed(seed: int) -> str:
    """
    1. The use of function:
       Identifies which dataset split (train, val, or test) a given random seed belongs to.
    2. How does the specific function work:
       Validates seed is an integer (rejecting booleans), checks membership within TRAIN_SEED_RANGE,
       VAL_SEED_RANGE, and TEST_SEED_RANGE, and returns the canonical split string.
    3. Concepts used in the function:
       Boundary validation, seed partition isolation, exception raising for out-of-bounds seeds.
    4. Places where this function is imported and used:
       Imported by workloads package, ML/RL training scripts, evaluation runners, and tests.
    5. Final output of the function, before and after:
       Before: Raw integer seed.
       After: Canonical split string ("train", "val", or "test"), or raises ValueError / TypeError.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"Seed must be an int, got {type(seed).__name__}.")

    if TRAIN_SEED_RANGE[0] <= seed <= TRAIN_SEED_RANGE[1]:
        return SPLIT_TRAIN
    if VAL_SEED_RANGE[0] <= seed <= VAL_SEED_RANGE[1]:
        return SPLIT_VAL
    if TEST_SEED_RANGE[0] <= seed <= TEST_SEED_RANGE[1]:
        return SPLIT_TEST

    raise ValueError(
        f"Seed {seed} does not belong to any defined split: "
        f"Train={TRAIN_SEED_RANGE}, Val={VAL_SEED_RANGE}, Test={TEST_SEED_RANGE}."
    )


def generate_split_workloads(
    split: str,
    n_workloads: int,
    seed_offset: int = 0,
    default_n_processes: Optional[int] = None,
) -> list[Workload]:
    """
    1. The use of function:
       Synthesizes a balanced batch of Workload instances belonging to a target dataset split.
    2. How does the specific function work:
       Validates split name against ALL_SPLITS and validates requested workload count against
       available partition capacity. Cycles round-robin across all 9 canonical scenarios to maintain
       equal class proportions. Allocates sequential seeds starting from the split base plus
       seed_offset. Samples process count n deterministically, calls generate_workload, and
       assembles validated immutable Workload instances with descriptive IDs.
    3. Concepts used in the function:
       Class-balanced generation, round-robin scenario cycling, deterministic seed allocation,
       workload batch generation.
    4. Places where this function is imported and used:
       Imported by CLI build-workloads command, ML dataset builder, RL training pipeline, and tests.
    5. Final output of the function, before and after:
       Before: Split name, batch count, optional seed offset, optional fixed process count.
       After: List of validated immutable Workload instances, or raises ValueError / TypeError.
    """
    if not isinstance(split, str):
        raise TypeError(f"Split must be a str, got {type(split).__name__}.")

    canonical_split = split.strip().lower()
    if canonical_split not in ALL_SPLITS:
        raise ValueError(
            f"Unknown split '{split}'. Available splits: {', '.join(ALL_SPLITS)}."
        )

    if isinstance(n_workloads, bool) or not isinstance(n_workloads, int):
        raise TypeError(
            f"n_workloads must be an int, got {type(n_workloads).__name__}."
        )
    if n_workloads < 1:
        raise ValueError(f"n_workloads must be >= 1, got {n_workloads}.")

    if isinstance(seed_offset, bool) or not isinstance(seed_offset, int):
        raise TypeError(
            f"seed_offset must be an int, got {type(seed_offset).__name__}."
        )
    if seed_offset < 0:
        raise ValueError(f"seed_offset must be >= 0, got {seed_offset}.")

    if default_n_processes is not None:
        if isinstance(default_n_processes, bool) or not isinstance(
            default_n_processes, int
        ):
            raise TypeError(
                f"default_n_processes must be an int, got {type(default_n_processes).__name__}."
            )
        if default_n_processes < 1 or default_n_processes > config.MAX_PROCESSES:
            raise ValueError(
                f"default_n_processes must be between 1 and {config.MAX_PROCESSES}, "
                f"got {default_n_processes}."
            )

    # Determine seed bounds
    if canonical_split == SPLIT_TRAIN:
        base_seed, max_seed = TRAIN_SEED_RANGE
    elif canonical_split == SPLIT_VAL:
        base_seed, max_seed = VAL_SEED_RANGE
    else:
        base_seed, max_seed = TEST_SEED_RANGE

    start_seed = base_seed + seed_offset
    end_seed = start_seed + n_workloads - 1
    if end_seed > max_seed:
        raise ValueError(
            f"Requested {n_workloads} workloads with offset {seed_offset} exceeds split "
            f"'{canonical_split}' capacity (start={start_seed}, end={end_seed}, max={max_seed})."
        )

    workloads: list[Workload] = []
    num_scenarios = len(ALL_SCENARIOS)

    for i in range(n_workloads):
        current_seed = start_seed + i
        scenario = ALL_SCENARIOS[i % num_scenarios]

        if default_n_processes is not None:
            n_proc = default_n_processes
        else:
            # Deterministically vary process count in [3, MAX_PROCESSES]
            spread = max(config.MAX_PROCESSES - 3 + 1, 1)
            n_proc = 3 + (current_seed % spread)

        processes = generate_workload(scenario=scenario, n=n_proc, seed=current_seed)
        workload_id = f"{canonical_split}_{scenario}_{current_seed:06d}"
        workload = Workload(
            workload_id=workload_id,
            scenario=scenario,
            seed=current_seed,
            processes=tuple(processes),
        )
        workloads.append(workload)

    return workloads


__all__ = [
    "ALL_SPLITS",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "TEST_SEED_RANGE",
    "TRAIN_SEED_RANGE",
    "VAL_SEED_RANGE",
    "generate_split_workloads",
    "get_split_for_seed",
]
