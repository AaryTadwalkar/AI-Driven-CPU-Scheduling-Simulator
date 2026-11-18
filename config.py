"""
1. What this file does:
   Defines global configuration constants, hyperparameters, path references, and seed partitions
   used across the CPU scheduling simulator, ML models, RL agent, and evaluation suite.
2. Overall design:
   Centralizes all mutable simulator tunables, policy defaults, dataset split boundaries, and
   file system directories into standardized constants to ensure single-source-of-truth behavior
   across all stages of development.
3. Elements in the file:
   - Core scheduling constants: TIME_QUANTUM, CONTEXT_SWITCH_COST, MAX_PROCESSES.
   - Machine Learning labeling parameters: LABEL_CS_COST, LABEL_WEIGHTS, LABEL_TIE_TOLERANCE.
   - Reinforcement Learning hyperparameters: ALPHA, GAMMA, EPS_START, EPS_MIN, EPS_DECAY,
     RL_EPISODES, RL_EVAL_EVERY, RL_TRAIN_SEEDS.
   - Seed partition definitions: TRAIN_SEED_RANGE, VAL_SEED_RANGE, TEST_SEED_RANGE.
   - Directory paths: BASE_DIR, DATA_DIR, MODELS_DIR, RESULTS_DIR, DOCS_DIR.
4. Final output of the file:
   Exported configuration variables and paths accessible by all modules in the project.
"""

from pathlib import Path
from typing import Dict, List, Tuple

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
