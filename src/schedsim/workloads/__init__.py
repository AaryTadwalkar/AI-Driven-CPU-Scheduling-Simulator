"""
1. What this file does:
   Initializes the workloads package, exporting canonical scenario definitions, synthetic workload
   generation routines, workload serialization/deserialization helpers, dataset partition
   utilities, and predefined preset workloads.
2. Overall design:
   Aggregates the synthetic data layer for the CPU scheduling simulator. Re-exports scenario
   constants, generator functions, CSV persistence utilities, split partition managers, and
   hand-verified preset workloads.
3. Elements in the file:
   - ALL_SCENARIOS, get_scenario: Workload scenario constants and validator.
   - generate_workload: Main deterministic process generator.
   - Workload, save_workloads_to_csv, load_workloads_from_csv: CSV workload I/O.
   - SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, ALL_SPLITS: Dataset partition identifiers.
   - get_split_for_seed, generate_split_workloads: Seed partition utilities.
   - get_preset, list_presets: Standard hand-crafted evaluation workloads.
4. Final output of the file:
   Public API interface for schedsim.workloads package.
"""

from schedsim.workloads.generator import generate_workload
from schedsim.workloads.io import (
    Workload,
    load_workloads_from_csv,
    save_workloads_to_csv,
)
from schedsim.workloads.presets import get_preset, list_presets
from schedsim.workloads.scenarios import (
    ALL_SCENARIOS,
    BIMODAL,
    CONVOY,
    EQUAL_BURSTS,
    HIGH_LOAD,
    LONG_JOBS,
    MIXED,
    SCENARIO_BIMODAL,
    SCENARIO_CONVOY,
    SCENARIO_EQUAL_BURSTS,
    SCENARIO_HIGH_LOAD,
    SCENARIO_LONG_JOBS,
    SCENARIO_MIXED,
    SCENARIO_SHORT_JOBS,
    SCENARIO_SIMULTANEOUS,
    SCENARIO_STAGGERED,
    SHORT_JOBS,
    SIMULTANEOUS,
    STAGGERED,
    get_scenario,
)
from schedsim.workloads.splits import (
    ALL_SPLITS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TEST_SEED_RANGE,
    TRAIN_SEED_RANGE,
    VAL_SEED_RANGE,
    generate_split_workloads,
    get_split_for_seed,
)

__all__ = [
    "ALL_SCENARIOS",
    "ALL_SPLITS",
    "BIMODAL",
    "CONVOY",
    "EQUAL_BURSTS",
    "HIGH_LOAD",
    "LONG_JOBS",
    "MIXED",
    "SCENARIO_BIMODAL",
    "SCENARIO_CONVOY",
    "SCENARIO_EQUAL_BURSTS",
    "SCENARIO_HIGH_LOAD",
    "SCENARIO_LONG_JOBS",
    "SCENARIO_MIXED",
    "SCENARIO_SHORT_JOBS",
    "SCENARIO_SIMULTANEOUS",
    "SCENARIO_STAGGERED",
    "SHORT_JOBS",
    "SIMULTANEOUS",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VAL",
    "STAGGERED",
    "TEST_SEED_RANGE",
    "TRAIN_SEED_RANGE",
    "VAL_SEED_RANGE",
    "Workload",
    "generate_split_workloads",
    "generate_workload",
    "get_preset",
    "get_scenario",
    "get_split_for_seed",
    "list_presets",
    "load_workloads_from_csv",
    "save_workloads_to_csv",
]
