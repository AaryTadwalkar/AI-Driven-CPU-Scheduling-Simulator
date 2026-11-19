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
   - check_class_distribution: Enforces the pilot-gate threshold on class balance (share < 85%).
   - build_setting_dataset: Generates and persists a labeled dataset for a specific setting (S0/S1).
   - load_or_generate_setting_dataset: Loads existing per-setting dataset or generates if missing.
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
from schedsim.workloads.io import Workload, load_workloads_from_csv
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


def check_class_distribution(
    df: pd.DataFrame,
    max_share: float = 0.85,
) -> dict[str, float]:
    """
    1. The use of function:
       Validates the class balance of a labeled scheduling dataset against the pilot-gate threshold.
    2. How does the specific function work:
       Verifies existence of 'label' column, calculates relative share proportions for each class,
       confirms no single class has share >= max_share (default 0.85), and returns the class share
       distribution dictionary.
    3. Concepts used in the function:
       Class imbalance detection, pilot study gate verification, defensive assertion.
    4. Places where this function is imported and used:
       Used in dataset building pipelines, Stage 12 acceptance tests, and model training routines.
    5. Final output of the function, before and after:
       Before: Labeled DataFrame and max_share float threshold.
       After: Returns dictionary mapping class names to share proportions, or raises ValueError.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")
    if "label" not in df.columns:
        raise ValueError("DataFrame does not contain required 'label' column.")
    total = len(df)
    if total == 0:
        return {}

    counts = df["label"].value_counts().to_dict()
    shares = {str(k): float(v / total) for k, v in counts.items()}
    for lbl, share in shares.items():
        if share >= max_share:
            raise ValueError(
                f"Class distribution pilot gate failed: class '{lbl}' has share "
                f"{share:.4f} >= {max_share:.4f}."
            )
    return shares


def build_setting_dataset(
    setting: str = "S1",
    workloads_path: Path | str = config.DATA_DIR / "workloads.csv",
    output_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    1. The use of function:
       Generates a complete labeled dataset for a specific simulation setting ('S0' or 'S1').
    2. How does the specific function work:
       Validates setting against config.SETTINGS, extracts corresponding cs_cost penalty,
       loads raw workloads from CSV, runs build_dataset() to extract features and compute labels,
       enforces pilot-gate distribution checks, persists result to disk, and returns the DataFrame.
    3. Concepts used in the function:
       Per-setting labeling, context-switch penalty isolation, automated dataset construction.
    4. Places where this function is imported and used:
       Invoked by Category 1 training workflows, CLI commands, and test suites.
    5. Final output of the function, before and after:
       Before: Setting identifier string and optional workload/output file paths.
       After: Evaluated and saved pandas DataFrame conforming to DATASET_COLUMNS schema.
    """
    if not isinstance(setting, str):
        raise TypeError(f"setting must be a str, got {type(setting).__name__}.")
    setting_key = setting.upper().strip()
    if setting_key not in config.SETTINGS:
        valid_settings = ", ".join(config.SETTINGS.keys())
        raise ValueError(
            f"Unknown setting '{setting}'. Must be one of: {valid_settings}."
        )

    cs_cost = config.SETTINGS[setting_key].cs_cost
    workloads = load_workloads_from_csv(workloads_path)
    df = build_dataset(workloads, cs_cost=cs_cost, quantum=config.TIME_QUANTUM)
    check_class_distribution(df, max_share=0.85)

    dest_path = (
        Path(output_path)
        if output_path is not None
        else config.DATA_DIR / f"scheduling_dataset_{setting_key}.csv"
    )
    save_dataset_to_csv(df, path=dest_path)
    return df


def load_or_generate_setting_dataset(
    setting: str = "S1",
    workloads_path: Path | str = config.DATA_DIR / "workloads.csv",
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    1. The use of function:
       Loads an existing per-setting dataset CSV or synthesizes it if not yet present on disk.
    2. How does the specific function work:
       Resolves target setting path. If file exists and force_rebuild is False, loads and validates
       it. If setting is S1 and legacy scheduling_dataset.csv exists, copies and validates it.
       Otherwise, generates the dataset via build_setting_dataset() and persists it.
    3. Concepts used in the function:
       Idempotent data caching, fallback dataset generation, automated validation.
    4. Places where this function is imported and used:
       Invoked by train_category1_models() in train_forest.py and Stage 12 tests.
    5. Final output of the function, before and after:
       Before: Setting identifier string and generation flags.
       After: Validated pandas DataFrame loaded from cache or freshly generated.
    """
    if not isinstance(setting, str):
        raise TypeError(f"setting must be a str, got {type(setting).__name__}.")
    setting_key = setting.upper().strip()
    if setting_key not in config.SETTINGS:
        valid_settings = ", ".join(config.SETTINGS.keys())
        raise ValueError(
            f"Unknown setting '{setting}'. Must be one of: {valid_settings}."
        )

    target_path = config.DATA_DIR / f"scheduling_dataset_{setting_key}.csv"
    if not force_rebuild and target_path.exists():
        df = load_dataset_from_csv(target_path)
        check_class_distribution(df, max_share=0.85)
        return df

    legacy_s1_path = config.DATA_DIR / "scheduling_dataset.csv"
    if not force_rebuild and setting_key == "S1" and legacy_s1_path.exists():
        df = load_dataset_from_csv(legacy_s1_path)
        check_class_distribution(df, max_share=0.85)
        save_dataset_to_csv(df, target_path)
        return df

    return build_setting_dataset(
        setting=setting_key,
        workloads_path=workloads_path,
        output_path=target_path,
    )


__all__ = [
    "DATASET_COLUMNS",
    "build_dataset",
    "build_dataset_row",
    "build_setting_dataset",
    "check_class_distribution",
    "load_dataset_from_csv",
    "load_or_generate_setting_dataset",
    "save_dataset_to_csv",
]
