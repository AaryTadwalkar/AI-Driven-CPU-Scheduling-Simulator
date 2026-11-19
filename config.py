"""
1. What this file does:
   Defines global configuration constants, hyperparameters, path references, benchmark settings,
   seed partitions, and PID rollout oracle configurations used across the CPU scheduling simulator,
   ML models, RL agent, and evaluation suite.
2. Overall design:
   Centralizes all mutable simulator tunables, policy defaults, dataset split boundaries,
   hardware overhead models, file system directories, and PID selector hyperparameters into
   standardized constants to ensure single-source-of-truth behavior across all stages.
3. Elements in the file:
   - Setting: Frozen dataclass representing a benchmark setting (name, cs_cost).
   - SETTINGS: Dictionary mapping setting names ('S0', 'S1') to Setting instances.
   - TICK_US_SWEEP: List of tick length sweep values in microseconds.
   - DEFAULT_TICK_US: Baseline simulation tick length in microseconds (1000 us = 1 ms).
   - FORCED_DECISION_FREE: Boolean flag specifying whether single-candidate decisions are free.
   - RF_PARAM_GRID: Hyperparameter grid dictionary for Random Forest model optimization.
   - PID selector & oracle constants: ORACLE_BASE_POLICIES, PID_BC_TRAIN_WORKLOADS, DAGGER_ITERS,
     DAGGER_WORKLOADS_PER_ITER, PID_MAX_STATES, GBDT_PARAMS, NN_HIDDEN, NN_SEEDS.
   - Core scheduling constants: TIME_QUANTUM, CONTEXT_SWITCH_COST, MAX_PROCESSES.
   - Machine Learning labeling parameters: LABEL_CS_COST, LABEL_WEIGHTS, LABEL_TIE_TOLERANCE.
   - Reinforcement Learning hyperparameters: ALPHA, GAMMA, EPS_START, EPS_MIN, EPS_DECAY,
     RL_EPISODES, RL_EVAL_EVERY, RL_TRAIN_SEEDS.
   - Seed partition definitions: TRAIN_SEED_RANGE, VAL_SEED_RANGE, TEST_SEED_RANGE.
   - Directory paths: BASE_DIR, DATA_DIR, MODELS_DIR, RESULTS_DIR, DOCS_DIR.
4. Final output of the file:
   Exported configuration variables, settings, and paths accessible by all project modules.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Base repository directories
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
MODELS_DIR: Path = BASE_DIR / "models"
RESULTS_DIR: Path = BASE_DIR / "results"
DOCS_DIR: Path = BASE_DIR / "docs"

# Core scheduling engine constants
TIME_QUANTUM: int = 2
CONTEXT_SWITCH_COST: int = 0
LABEL_CS_COST: int = 1
MAX_PROCESSES: int = 12


@dataclass(frozen=True)
class Setting:
    """
    1. The use of class:
       Defines a benchmark simulation configuration setting, pairing a setting identifier
       with a specific context switch cost under the waiting-time objective.
    2. What the class contains:
       - name: Identifier string for the setting (e.g., 'S0', 'S1').
       - cs_cost: Integer context switch tick cost penalty for process preemptions.
    3. Technologies used and why:
       - Frozen dataclass for immutability, thread safety, and value-based equality/hash.
       - Python 3.11 static typing for clean integration across simulation runs.
    4. Key design and overall design of the class:
       Standardizes benchmark evaluation settings (S0: cs_cost=0, S1: cs_cost=1) under
       the waiting-time objective.
    """

    name: str
    cs_cost: int


# Benchmark settings (cs_cost only; objective is always total waiting time)
SETTINGS: dict[str, Setting] = {
    "S0": Setting(name="S0", cs_cost=0),
    "S1": Setting(name="S1", cs_cost=1),
}

# AI overhead model constants (Section 3 & 4)
TICK_US_SWEEP: list[int] = [1, 10, 100, 1000, 10000]
DEFAULT_TICK_US: int = 1000
FORCED_DECISION_FREE: bool = True

# Random Forest (Category 1) hyperparameter grid (Section 3)
RF_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [50, 100],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5],
}

# PID data / models & rollout oracle (Section 3 & Stages 13-16)
ORACLE_BASE_POLICIES: tuple[str, ...] = ("SRTF", "SJF")
PID_BC_TRAIN_WORKLOADS: int = 3000
DAGGER_ITERS: int = 2
DAGGER_WORKLOADS_PER_ITER: int = 1000
PID_MAX_STATES: int = 100_000
GBDT_PARAMS: dict[str, Any] = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}
NN_HIDDEN: int = 32
NN_SEEDS: list[int] = [0, 1, 2]

# Decision Tree labeling criteria
LABEL_WEIGHTS: Dict[str, float] = {
    "wait": 0.5,
    "resp": 0.3,
    "tat": 0.2,
}
LABEL_TIE_TOLERANCE: float = 0.02

# Reinforcement Learning (Q-learning) defaults
ALPHA: float = 0.1
GAMMA: float = 1.0
EPS_START: float = 1.0
EPS_MIN: float = 0.05
EPS_DECAY: float = 0.9995
RL_EPISODES: int = 20000
RL_EVAL_EVERY: int = 500
RL_TRAIN_SEEDS: List[int] = [0, 1, 2, 3, 4]

# Reinforcement Learning State Space Discretization Buckets (Section 5 Stage 5)
STATE_READY_COUNT_EDGES: Tuple[int, ...] = (1, 2, 3)
STATE_SHORTEST_REMAINING_EDGES: Tuple[int, ...] = (1, 2, 4, 8)
STATE_LONGEST_REMAINING_EDGES: Tuple[int, ...] = (2, 4, 8, 16)
STATE_MAX_WAITED_EDGES: Tuple[int, ...] = (0, 2, 5)

# Seed ranges for reproducible splits (Section 3.4)
TRAIN_SEED_RANGE: Tuple[int, int] = (0, 99999)
VAL_SEED_RANGE: Tuple[int, int] = (100000, 100499)
TEST_SEED_RANGE: Tuple[int, int] = (200000, 200999)

