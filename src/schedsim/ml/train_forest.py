"""
1. What this file does:
   Implements Category 1 model training (Decision Tree and Random Forest) per benchmark setting
   ('S0' and 'S1'), performing cross-validated hyperparameter optimization, validation regret
   evaluation, majority-class baseline comparison, and artifact persistence.
2. Overall design:
   Loads per-setting labeled datasets (scheduling_dataset_S0.csv and scheduling_dataset_S1.csv),
   partitions features across Train and Validation splits, tunes DecisionTreeClassifier and
   RandomForestClassifier using GridSearchCV over standardized parameter grids, evaluates accuracy,
   macro-F1, and decision regret against majority-class baselines, and serializes model joblibs and
   JSON metadata into the models directory.
3. Elements in the file:
   - train_category1_for_setting: Trains and evaluates DT and RF models for a single setting.
   - train_category1_models: Orchestrates Category 1 training across all benchmark settings.
4. Final output of the file:
   Exports train_category1_models and train_category1_for_setting for CLI, pipelines, and tests.
"""

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import config
from schedsim.ml.dataset_builder import load_or_generate_setting_dataset
from schedsim.ml.evaluate_tree import compute_regret
from schedsim.ml.features import FEATURE_NAMES

DT_PARAM_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 4, 5, 6, 8],
    "min_samples_leaf": [1, 2, 5, 10],
}


def train_category1_for_setting(
    setting: str,
    dataset: Optional[pd.DataFrame] = None,
    models_dir: Path | str = config.MODELS_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """
    1. The use of function:
       Trains, tunes, and evaluates Decision Tree and Random Forest models for a specific setting.
    2. How does the specific function work:
       Loads setting dataset if not supplied, partitions into train and validation splits, fits
       Decision Tree and Random Forest via GridSearchCV with 5-fold cross-validation, computes
       accuracy, macro-F1, and validation regret against a majority-class dummy baseline, saves
       model joblibs and metadata JSONs, and synchronizes S1 Decision Tree to baseline paths.
    3. Concepts used in the function:
       Hyperparameter tuning, cross-validation, regret evaluation, model serialization.
    4. Places where this function is imported and used:
       Called by train_category1_models() and Stage 12 unit/acceptance tests.
    5. Final output of the function, before and after:
       Before: Raw dataset or setting string identifier.
       After: Serialized joblib and JSON artifacts saved to disk; returns result metrics dictionary.
    """
    setting_key = setting.upper().strip()
    if setting_key not in config.SETTINGS:
        raise ValueError(
            f"Unknown setting '{setting}'. Must be one of: {list(config.SETTINGS.keys())}."
        )

    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset is not None:
        df = dataset.copy()
    else:
        df = load_or_generate_setting_dataset(setting=setting_key)

    if "split" in df.columns:
        df_train = df[df["split"] == "train"]
        df_val = df[df["split"] == "val"]
    else:
        val_size = max(1, int(len(df) * 0.2))
        df_val = df.sample(n=val_size, random_state=seed)
        df_train = df.drop(df_val.index)

    feature_cols = list(FEATURE_NAMES)
    x_train = df_train[feature_cols].values
    y_train = df_train["label"].values.astype(str)
    x_val = df_val[feature_cols].values
    y_val = df_val["label"].values.astype(str)

    cv_folds = min(5, len(y_train) // max(len(set(y_train)), 1))
    cv_folds = max(cv_folds, 2)

    # 1. Majority class dummy baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    dummy_preds = dummy.predict(x_val)
    dummy_metrics = {
        "accuracy": float(accuracy_score(y_val, dummy_preds)),
        "macro_f1": float(f1_score(y_val, dummy_preds, average="macro", zero_division=0)),
        "mean_regret": compute_regret(df_val, dummy_preds),
    }

    # 2. Train Decision Tree
    dt_base = DecisionTreeClassifier(random_state=seed)
    dt_grid = GridSearchCV(
        estimator=dt_base,
        param_grid=DT_PARAM_GRID,
        scoring="f1_macro",
        cv=cv_folds,
        n_jobs=-1,
    )
    dt_grid.fit(x_train, y_train)
    best_dt: DecisionTreeClassifier = dt_grid.best_estimator_

    dt_preds = best_dt.predict(x_val)
    dt_metrics = {
        "accuracy": float(accuracy_score(y_val, dt_preds)),
        "macro_f1": float(f1_score(y_val, dt_preds, average="macro", zero_division=0)),
        "mean_regret": compute_regret(df_val, dt_preds),
        "best_params": {
            "max_depth": int(best_dt.max_depth) if best_dt.max_depth else None,
            "min_samples_leaf": int(best_dt.min_samples_leaf),
        },
    }

    # 3. Train Random Forest
    rf_base = RandomForestClassifier(random_state=seed, n_jobs=-1)
    rf_grid = GridSearchCV(
        estimator=rf_base,
        param_grid=config.RF_PARAM_GRID,
        scoring="f1_macro",
        cv=cv_folds,
        n_jobs=-1,
    )
    rf_grid.fit(x_train, y_train)
    best_rf: RandomForestClassifier = rf_grid.best_estimator_

    rf_preds = best_rf.predict(x_val)
    rf_metrics = {
        "accuracy": float(accuracy_score(y_val, rf_preds)),
        "macro_f1": float(f1_score(y_val, rf_preds, average="macro", zero_division=0)),
        "mean_regret": compute_regret(df_val, rf_preds),
        "best_params": {
            "n_estimators": int(best_rf.n_estimators),
            "max_depth": int(best_rf.max_depth) if best_rf.max_depth else None,
            "min_samples_leaf": int(best_rf.min_samples_leaf),
        },
    }

    # 4. Save artifacts
    dt_model_path = out_dir / f"decision_tree_{setting_key}.joblib"
    dt_meta_path = out_dir / f"decision_tree_{setting_key}_meta.json"
    rf_model_path = out_dir / f"random_forest_{setting_key}.joblib"
    rf_meta_path = out_dir / f"random_forest_{setting_key}_meta.json"

    joblib.dump(best_dt, dt_model_path)
    joblib.dump(best_rf, rf_model_path)

    dt_meta = {
        "model_type": "DecisionTreeClassifier",
        "setting": setting_key,
        "feature_names": feature_cols,
        "classes": [str(c) for c in best_dt.classes_],
        "chosen_params": dt_metrics["best_params"],
        "training_samples": len(x_train),
        "validation_samples": len(x_val),
        "metrics_validation": {
            "decision_tree": dt_metrics,
            "majority_class_dummy": dummy_metrics,
            **dt_metrics,
        },
        "majority_class_baseline": dummy_metrics,
        "beats_majority_baseline": bool(
            dt_metrics["macro_f1"] > dummy_metrics["macro_f1"]
        ),
    }
    rf_meta = {
        "model_type": "RandomForestClassifier",
        "setting": setting_key,
        "feature_names": feature_cols,
        "classes": [str(c) for c in best_rf.classes_],
        "chosen_params": rf_metrics["best_params"],
        "training_samples": len(x_train),
        "validation_samples": len(x_val),
        "metrics_validation": {
            "random_forest": rf_metrics,
            "majority_class_dummy": dummy_metrics,
            **rf_metrics,
        },
        "majority_class_baseline": dummy_metrics,
        "beats_majority_baseline": bool(
            rf_metrics["macro_f1"] > dummy_metrics["macro_f1"]
        ),
        "rf_vs_dt_regret_delta": float(rf_metrics["mean_regret"] - dt_metrics["mean_regret"]),
        "rf_beats_dt_regret": bool(rf_metrics["mean_regret"] <= dt_metrics["mean_regret"]),
    }

    with open(dt_meta_path, "w", encoding="utf-8") as f:
        json.dump(dt_meta, f, indent=2)

    with open(rf_meta_path, "w", encoding="utf-8") as f:
        json.dump(rf_meta, f, indent=2)

    # Maintain backward compatibility for Part A tests by copying S1 artifacts
    if setting_key == "S1":
        legacy_dt_path = out_dir / "decision_tree.joblib"
        legacy_meta_path = out_dir / "decision_tree_meta.json"
        shutil.copyfile(dt_model_path, legacy_dt_path)
        shutil.copyfile(dt_meta_path, legacy_meta_path)

    return {
        "setting": setting_key,
        "decision_tree": dt_metrics,
        "random_forest": rf_metrics,
        "majority_baseline": dummy_metrics,
        "comparison": {
            "dt_beats_baseline": bool(dt_metrics["macro_f1"] > dummy_metrics["macro_f1"]),
            "rf_beats_baseline": bool(rf_metrics["macro_f1"] > dummy_metrics["macro_f1"]),
            "rf_vs_dt_regret_delta": float(
                rf_metrics["mean_regret"] - dt_metrics["mean_regret"]
            ),
            "rf_beats_dt_regret": bool(rf_metrics["mean_regret"] <= dt_metrics["mean_regret"]),
        },
    }


def train_category1_models(
    settings: Sequence[str] = ("S0", "S1"),
    models_dir: Path | str = config.MODELS_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """
    1. The use of function:
       Executes complete Category 1 model training pipeline across all designated settings.
    2. How does the specific function work:
       Iterates across provided settings, invokes train_category1_for_setting() for each setting,
       aggregates validation metrics, and returns summary comparison dictionary.
    3. Concepts used in the function:
       Multi-setting training orchestration, model benchmarking, reproducibility.
    4. Places where this function is imported and used:
       Dispatched by training commands and Stage 12 acceptance tests.
    5. Final output of the function, before and after:
       Before: Sequence of setting identifier strings.
       After: Trained artifacts for all settings saved to disk; returns composite report dictionary.
    """
    results: dict[str, Any] = {}
    for stg in settings:
        res = train_category1_for_setting(
            setting=stg,
            models_dir=models_dir,
            seed=seed,
        )
        results[stg] = res
    return results


__all__ = [
    "DT_PARAM_GRID",
    "train_category1_for_setting",
    "train_category1_models",
]
