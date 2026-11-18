"""
1. What this file does:
   Constructs, serializes, and deserializes the supervised machine learning dataset for Decision
   Tree policy recommendation, mapping raw workload structures into feature vectors, composite
   labels, optimality margins, and split partitions.
2. Overall design:
   Processes collections of Workload entities, extracts 12 workload domain features via
   `extract_features()`, computes multi-metric composite labels via `compute_workload_label()`,
   determines dataset split partition mappings via `get_split_for_seed()`, and formats the records
   into a structured pandas DataFrame persisted to `data/scheduling_dataset.csv`.
3. Elements in the file:
   - DATASET_COLUMNS: Canonical ordered list of CSV header columns for the ML dataset.
   - build_dataset_row: Transforms a single Workload instance into an evaluated feature/label dict.
   - build_dataset: Assembles a pandas DataFrame across a sequence of Workload entities.
   - save_dataset_to_csv: Persists the structured ML DataFrame to disk with directory creation.
   - load_dataset_from_csv: Loads and validates the ML dataset from a CSV path.
4. Final output of the file:
   Exports dataset building and I/O utilities for CLI commands, model training, and evaluations.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import config
from schedsim.ml.features import FEATURE_NAMES, extract_features
from schedsim.ml.labeling import compute_workload_label
from schedsim.workloads.io import Workload
from schedsim.workloads.splits import get_split_for_seed

DATASET_COLUMNS: tuple[str, ...] = (
    "workload_id",
    "scenario",
    "seed",
    "split",
    *FEATURE_NAMES,
    "label",
    "margin",
    "is_ambiguous",
    "score_fcfs",
    "score_sjf",
    "score_rr",
)


def build_dataset_row(
    workload: Workload,
    split: Optional[str] = None,
    cs_cost: int = config.LABEL_CS_COST,
    quantum: int = config.TIME_QUANTUM,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """
    1. The use of function:
       Transforms a single Workload into a feature-extracted, labeled row dictionary.
    2. How does the specific function work:
       Validates the Workload instance, resolves its dataset split (explicitly passed or via
       `get_split_for_seed`), extracts 12 workload features, runs composite policy labeling,
       and merges metadata, features, label, margin, ambiguity, and individual policy scores.
    3. Concepts used in the function:
       Feature extraction, simulation labeling, record formatting, partition resolution.
    4. Places where this function is imported and used:
       Imported by `build_dataset` in dataset_builder.py and tested in test_ml_dataset_builder.py.
    5. Final output of the function, before and after:
       Before: Raw Workload instance and simulation parameters.
       After: Dictionary with keys strictly matching DATASET_COLUMNS.
    """
    if not isinstance(workload, Workload):
        raise TypeError(f"workload must be a Workload instance, got {type(workload).__name__}.")

    effective_split = split if split is not None else get_split_for_seed(workload.seed)

    feats = extract_features(workload.processes)
    label_res = compute_workload_label(
        processes=workload.processes,
        cs_cost=cs_cost,
        quantum=quantum,
        weights=weights,
    )

    row: dict[str, Any] = {
        "workload_id": workload.workload_id,
        "scenario": workload.scenario,
        "seed": int(workload.seed),
        "split": effective_split,
    }
    for feat_name in FEATURE_NAMES:
        row[feat_name] = feats[feat_name]

    row["label"] = label_res.label
    row["margin"] = float(label_res.margin)
    row["is_ambiguous"] = bool(label_res.is_ambiguous)
    row["score_fcfs"] = float(label_res.scores.get("FCFS", 0.0))
    row["score_sjf"] = float(label_res.scores.get("SJF", 0.0))
    row["score_rr"] = float(label_res.scores.get("RR", 0.0))

    return row


def build_dataset(
    workloads: Sequence[Workload],
    split: Optional[str] = None,
    cs_cost: int = config.LABEL_CS_COST,
    quantum: int = config.TIME_QUANTUM,
    weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """
    1. The use of function:
       Constructs a complete labeled pandas DataFrame from a collection of Workload instances.
    2. How does the specific function work:
       Validates input collection, iterates across workloads invoking `build_dataset_row()`,
       and constructs a pandas DataFrame with canonical columns defined by DATASET_COLUMNS.
    3. Concepts used in the function:
       Dataset synthesis, tabular transformation, pandas DataFrame aggregation.
    4. Places where this function is imported and used:
       Imported by CLI build-dataset command, training workflows, and test suites.
    5. Final output of the function, before and after:
       Before: Sequence of Workload objects.
       After: Evaluated pandas DataFrame ready for supervised model training and evaluation.
    """
    if not isinstance(workloads, Sequence) or isinstance(workloads, (str, bytes)):
        raise TypeError(f"workloads must be a Sequence[Workload], got {type(workloads).__name__}.")

    rows = [
        build_dataset_row(
            workload=w,
            split=split,
            cs_cost=cs_cost,
            quantum=quantum,
            weights=weights,
        )
        for w in workloads
    ]
    df = pd.DataFrame(rows, columns=list(DATASET_COLUMNS))
    return df


def save_dataset_to_csv(
    df: pd.DataFrame,
    path: Path | str = config.DATA_DIR / "scheduling_dataset.csv",
) -> Path:
    """
    1. The use of function:
       Persists an evaluated dataset DataFrame to disk as a CSV file.
    2. How does the specific function work:
       Ensures destination parent directory exists, validates that DataFrame contains required
       columns, and writes CSV using UTF-8 encoding without the pandas index column.
    3. Concepts used in the function:
       CSV persistence, directory auto-creation, data integrity enforcement.
    4. Places where this function is imported and used:
       Imported by CLI dataset commands, batch generation scripts, and test suites.
    5. Final output of the function, before and after:
       Before: Evaluated pandas DataFrame and destination file path.
       After: File written to disk, returns Path object to persisted CSV file.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_path, index=False)
    return target_path


def load_dataset_from_csv(
    path: Path | str = config.DATA_DIR / "scheduling_dataset.csv",
) -> pd.DataFrame:
    """
    1. The use of function:
       Loads and validates a previously persisted scheduling dataset CSV into a pandas DataFrame.
    2. How does the specific function work:
       Checks file existence, reads CSV via `pd.read_csv`, verifies presence of required
       columns from DATASET_COLUMNS, and returns the verified DataFrame.
    3. Concepts used in the function:
       CSV loading, schema verification, defensive validation.
    4. Places where this function is imported and used:
       Imported by `train_tree.py`, `evaluate_tree.py`, CLI commands, and test suites.
    5. Final output of the function, before and after:
       Before: File path to existing dataset CSV.
       After: Validated pandas DataFrame loaded from disk.
    """
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {target_path}")

    df = pd.read_csv(target_path)
    missing = [c for c in DATASET_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Corrupt dataset schema. Missing expected columns: {missing}")

    return df


__all__ = [
    "DATASET_COLUMNS",
    "build_dataset",
    "build_dataset_row",
    "load_dataset_from_csv",
    "save_dataset_to_csv",
]
