"""
1. What this file does:
   Implements grid-search hyperparameter optimization, training, and baseline comparison for the
   Decision Tree policy recommender, persisting the optimized model and comprehensive metadata.
2. Overall design:
   Loads labeled workload records from `data/scheduling_dataset.csv`, partitions features into
   Train and Validation splits, runs 5-fold cross-validated grid search over tree depths (3..8),
   leaf sample minimums (1..20), and class weighting schemes (None, balanced). Evaluates baseline
   models (Majority Class Dummy, Random Forest, K-Nearest Neighbors) for comparative rigor,
   persisting the chosen model to `models/decision_tree.joblib` and metadata to
   `models/decision_tree_meta.json`.
3. Elements in the file:
   - train_decision_tree: Fits DecisionTreeClassifier with cross-validation and benchmarks.
   - evaluate_model_metrics: Calculates accuracy and macro-F1 for a fitted classifier.
4. Final output of the file:
   Exports train_decision_tree for CLI `train-tree` execution, automated pipelines, and testing.
"""

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import config
from schedsim.ml.dataset_builder import load_dataset_from_csv
from schedsim.ml.features import FEATURE_NAMES


def evaluate_model_metrics(
    model: Any,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """
    1. The use of function:
       Computes standardized classification metrics (accuracy, macro-F1) for an estimator.
    2. How does the specific function work:
       Generates predictions using `model.predict()`, computes accuracy and macro-averaged
       F1 score against ground-truth labels, and returns a formatted dictionary.
    3. Concepts used in the function:
       Statistical classification evaluation, macro-averaged F1 score, model comparison.
    4. Places where this function is imported and used:
       Called internally by `train_decision_tree` to benchmark tree and baseline candidates.
    5. Final output of the function, before and after:
       Before: Fitted estimator and validation feature/target arrays.
       After: Dictionary with "accuracy" and "macro_f1" float scores.
    """
    preds = model.predict(x_val)
    acc = float(accuracy_score(y_val, preds))
    macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }


def train_decision_tree(
    dataset: Optional[pd.DataFrame | Path | str] = None,
    val_dataset: Optional[pd.DataFrame | Path | str] = None,
    output_model_path: Path | str = config.MODELS_DIR / "decision_tree.joblib",
    output_meta_path: Path | str = config.MODELS_DIR / "decision_tree_meta.json",
    seed: int = 42,
) -> dict[str, Any]:
    """
    1. The use of function:
       Orchestrates hyperparameter grid search, baseline comparisons, and model persistence.
    2. How does the specific function work:
       Loads dataset (defaulting to data/scheduling_dataset.csv), extracts train and validation
       splits, fits DecisionTreeClassifier via GridSearchCV over max_depth, min_samples_leaf,
       and class_weight. Fits Dummy, Random Forest, and KNN baselines. Persists chosen model
       via joblib and writes metadata JSON containing hyperparameters and comparison scores.
    3. Concepts used in the function:
       Hyperparameter tuning, cross-validation, baseline benchmarking, model serialization.
    4. Places where this function is imported and used:
       Dispatched by `cli.py train-tree`, run during automated builds, and tested in test suites.
    5. Final output of the function, before and after:
       Before: Raw dataset and optional file destinations.
       After: Serialized model file, metadata JSON, and summary dictionary returned to caller.
    """
    if dataset is None:
        dataset = config.DATA_DIR / "scheduling_dataset.csv"

    if isinstance(dataset, (str, Path)):
        df = load_dataset_from_csv(dataset)
    elif isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
    else:
        raise TypeError(f"dataset must be DataFrame, Path, or str; got {type(dataset).__name__}.")

    # Resolve train split
    if "split" in df.columns:
        df_train = df[df["split"] == "train"]
    else:
        df_train = df

    # Resolve validation split
    if val_dataset is not None:
        if isinstance(val_dataset, (str, Path)):
            df_val = load_dataset_from_csv(val_dataset)
        elif isinstance(val_dataset, pd.DataFrame):
            df_val = val_dataset.copy()
        else:
            raise TypeError("val_dataset must be DataFrame, Path, or str.")
    elif "split" in df.columns and (df["split"] == "val").any():
        df_val = df[df["split"] == "val"]
    else:
        # Fallback to train-test split if no explicit validation split exists
        val_size = max(1, int(len(df_train) * 0.2))
        df_val = df_train.sample(n=val_size, random_state=seed)
        df_train = df_train.drop(df_val.index)

    feature_cols = list(FEATURE_NAMES)
    x_train = df_train[feature_cols].values
    y_train = df_train["label"].values.astype(str)
    x_val = df_val[feature_cols].values
    y_val = df_val["label"].values.astype(str)

    # 1. Grid search for DecisionTreeClassifier
    param_grid = {
        "max_depth": [3, 4, 5, 6, 8],
        "min_samples_leaf": [1, 5, 20],
        "class_weight": [None, "balanced"],
    }
    base_tree = DecisionTreeClassifier(random_state=seed)
    cv_folds = min(5, len(y_train) // max(len(np.unique(y_train)), 1))
    cv_folds = max(cv_folds, 2)

    grid_search = GridSearchCV(
        estimator=base_tree,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv_folds,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    best_tree: DecisionTreeClassifier = grid_search.best_estimator_
    tree_val_metrics = evaluate_model_metrics(best_tree, x_val, y_val)

    # 2. Train and evaluate comparison baselines
    # Majority class baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    dummy_metrics = evaluate_model_metrics(dummy, x_val, y_val)

    # Random Forest baseline
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(x_train, y_train)
    rf_metrics = evaluate_model_metrics(rf, x_val, y_val)

    # K-Nearest Neighbors baseline (scaled)
    knn_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=min(5, len(x_train)))),
    ])
    knn_pipe.fit(x_train, y_train)
    knn_metrics = evaluate_model_metrics(knn_pipe, x_val, y_val)

    # 3. Model persistence
    target_model_path = Path(output_model_path)
    target_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_tree, target_model_path)

    # 4. Metadata persistence
    target_meta_path = Path(output_meta_path)
    target_meta_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "model_type": "DecisionTreeClassifier",
        "feature_names": feature_cols,
        "classes": [str(c) for c in best_tree.classes_],
        "chosen_params": {
            "max_depth": int(best_tree.max_depth) if best_tree.max_depth else None,
            "min_samples_leaf": int(best_tree.min_samples_leaf),
            "class_weight": (
                str(best_tree.class_weight) if best_tree.class_weight is not None else None
            ),
            "criterion": str(best_tree.criterion),
            "random_state": seed,
        },
        "training_samples": len(x_train),
        "validation_samples": len(x_val),
        "label_config": {
            "cs_cost": config.LABEL_CS_COST,
            "quantum": config.TIME_QUANTUM,
            "weights": config.LABEL_WEIGHTS,
            "tie_tolerance": config.LABEL_TIE_TOLERANCE,
        },
        "metrics_validation": {
            "decision_tree": tree_val_metrics,
            "majority_class_dummy": dummy_metrics,
            "random_forest": rf_metrics,
            "k_nearest_neighbors": knn_metrics,
        },
        "beats_majority_baseline": bool(
            tree_val_metrics["macro_f1"] > dummy_metrics["macro_f1"]
        ),
    }

    with open(target_meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


__all__ = [
    "evaluate_model_metrics",
    "train_decision_tree",
]
