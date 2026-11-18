"""
1. What this file does:
   Implements validation evaluation, regret calculation, tree visualization, and the single-point
   policy recommendation API for the Decision Tree scheduler.
2. Overall design:
   Loads the trained Decision Tree model and validation records. Evaluates accuracy, macro-F1,
   confusion matrices, feature importances, and decision regret (relative composite score penalty).
   Generates a visual tree diagram capped at depth 4 via matplotlib, writes metrics to JSON, and
   provides the public `recommend()` function allowing external callers to obtain policy
   recommendations from raw Process sequences.
3. Elements in the file:
   - compute_regret: Calculates mean relative excess composite score for predicted policies.
   - evaluate_decision_tree: Evaluates tree model on validation split and saves metrics/plots.
   - recommend: Public inference API returning predicted policy, probabilities, and features.
4. Final output of the file:
   Exports evaluate_decision_tree, compute_regret, and recommend for evaluation and CLI usage.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree

import config
from schedsim.domain.process import Process, validate_processes
from schedsim.ml.dataset_builder import load_dataset_from_csv
from schedsim.ml.features import FEATURE_NAMES, extract_features

_MODEL_CACHE: dict[str, Any] = {}


def compute_regret(
    df: pd.DataFrame,
    predictions: Sequence[str],
    eps: float = 1e-6,
) -> float:
    """
    1. The use of function:
       Computes the mean relative composite-score regret of predicted policies.
    2. How does the specific function work:
       For each workload row, retrieves the composite score of the predicted policy and compares
       it against the optimal policy score (the minimum score among candidate baselines). Computes
       the relative excess penalty `(predicted_score - best_score) / (best_score + eps)` and returns
       the mean regret across all evaluated samples.
    3. Concepts used in the function:
       Decision regret, relative optimality loss, practical utility assessment.
    4. Places where this function is imported and used:
       Called by `evaluate_decision_tree` and tested in test suites.
    5. Final output of the function, before and after:
       Before: DataFrame with score columns and sequence of string policy predictions.
       After: Non-negative float representing the mean relative regret.
    """
    if len(df) == 0:
        return 0.0

    score_cols = {
        "FCFS": "score_fcfs",
        "SJF": "score_sjf",
        "RR": "score_rr",
    }

    regrets: list[float] = []
    for (_, row), pred in zip(df.iterrows(), predictions):
        best_score = min(float(row["score_fcfs"]), float(row["score_sjf"]), float(row["score_rr"]))
        pred_col = score_cols.get(str(pred), "score_fcfs")
        pred_score = float(row[pred_col]) if pred_col in row else best_score
        regret = max(0.0, (pred_score - best_score) / (best_score + eps))
        regrets.append(regret)

    return float(np.mean(regrets)) if regrets else 0.0


def evaluate_decision_tree(
    model: Optional[DecisionTreeClassifier] = None,
    model_path: Path | str = config.MODELS_DIR / "decision_tree.joblib",
    dataset: Optional[pd.DataFrame | Path | str] = None,
    output_metrics_path: Path | str = config.RESULTS_DIR / "tree_metrics_val.json",
    output_plot_path: Optional[Path | str] = config.RESULTS_DIR / "figures" / "decision_tree.png",
) -> dict[str, Any]:
    """
    1. The use of function:
       Evaluates a trained Decision Tree on the validation dataset split.
    2. How does the specific function work:
       Loads model from path if not provided directly. Resolves validation dataset split, computes
       accuracy, macro-F1, confusion matrix, feature importances, unambiguous-subset accuracy,
       and relative regret. Optionally renders and saves a tree diagram capped at depth 4 via
       matplotlib, and persists metrics JSON.
    3. Concepts used in the function:
       Validation evaluation, confusion matrix, regret analysis, tree visualization.
    4. Places where this function is imported and used:
       Imported by CLI evaluate commands, report generation, and Stage 4 test suites.
    5. Final output of the function, before and after:
       Before: Trained model and validation dataset.
       After: Saved metrics JSON, optional figure PNG, and evaluation metrics dictionary.
    """
    if model is None:
        target_model_path = Path(model_path)
        if not target_model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {target_model_path}")
        model = joblib.load(target_model_path)

    if dataset is None:
        dataset = config.DATA_DIR / "scheduling_dataset.csv"

    if isinstance(dataset, (str, Path)):
        df = load_dataset_from_csv(dataset)
    elif isinstance(dataset, pd.DataFrame):
        df = dataset.copy()
    else:
        raise TypeError(f"dataset must be DataFrame, Path, or str; got {type(dataset).__name__}.")

    if "split" in df.columns and (df["split"] == "val").any():
        df_val = df[df["split"] == "val"].copy()
    else:
        df_val = df.copy()

    feature_cols = list(FEATURE_NAMES)
    x_val = df_val[feature_cols].values
    y_val = df_val["label"].values.astype(str)

    preds = model.predict(x_val)
    acc = float(accuracy_score(y_val, preds))
    macro_f1 = float(f1_score(y_val, preds, average="macro", zero_division=0))

    classes = [str(c) for c in model.classes_]
    cm = confusion_matrix(y_val, preds, labels=classes).tolist()

    importances = {
        feat: float(imp)
        for feat, imp in zip(feature_cols, model.feature_importances_)
    }

    # Regret metric
    mean_regret = compute_regret(df_val, preds)

    # Accuracy excluding ambiguous rows
    if "is_ambiguous" in df_val.columns:
        unambig_mask = ~df_val["is_ambiguous"].astype(bool)
        if unambig_mask.sum() > 0:
            unambig_acc = float(
                accuracy_score(y_val[unambig_mask], preds[unambig_mask])
            )
        else:
            unambig_acc = acc
    else:
        unambig_acc = acc

    metrics: dict[str, Any] = {
        "validation_samples": len(df_val),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "accuracy_unambiguous": unambig_acc,
        "mean_regret": mean_regret,
        "classes": classes,
        "confusion_matrix": cm,
        "feature_importances": importances,
    }

    # Save metrics JSON
    target_metrics_path = Path(output_metrics_path)
    target_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save readable tree visualization
    if output_plot_path is not None:
        plot_path = Path(output_plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
        plot_tree(
            model,
            max_depth=4,
            feature_names=feature_cols,
            class_names=classes,
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax,
        )
        ax.set_title("Decision Tree Policy Recommender (Capped Depth = 4)", fontsize=12)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)

    return metrics


def recommend(
    processes: Sequence[Process],
    model_path: Path | str = config.MODELS_DIR / "decision_tree.joblib",
) -> dict[str, Any]:
    """
    1. The use of function:
       Single-point public inference API recommending an optimal scheduling policy for a workload.
    2. How does the specific function work:
       Validates raw processes, extracts the 12 workload features, loads the fitted Decision Tree
       model (cached in memory for high-performance inference), predicts the optimal policy,
       extracts class probabilities, and returns a structured prediction package.
    3. Concepts used in the function:
       Model inference, in-memory caching, probabilistic classification, input validation.
    4. Places where this function is imported and used:
       Imported by CLI recommend command, FastAPI recommender routes, and Stage 4 acceptance tests.
    5. Final output of the function, before and after:
       Before: Sequence of raw Process domain entities.
       After: Dictionary with 'algorithm' string, 'probabilities' dict, and 'features' dict.
    """
    if not isinstance(processes, Sequence) or isinstance(processes, (str, bytes)):
        raise TypeError(
            f"processes must be a Sequence[Process], got {type(processes).__name__}."
        )
    proc_list = list(processes)
    validate_processes(proc_list)

    str_path = str(model_path)
    if str_path not in _MODEL_CACHE:
        target_path = Path(model_path)
        if not target_path.exists():
            raise FileNotFoundError(
                f"Trained Decision Tree model not found at {target_path}. "
                "Run `python cli.py train-tree` first."
            )
        _MODEL_CACHE[str_path] = joblib.load(target_path)

    model: DecisionTreeClassifier = _MODEL_CACHE[str_path]

    feats = extract_features(proc_list)
    feature_vector = np.array([[feats[col] for col in FEATURE_NAMES]], dtype=np.float64)

    predicted_class = str(model.predict(feature_vector)[0])
    probabilities_raw = model.predict_proba(feature_vector)[0]
    probabilities = {
        str(cls_name): float(prob)
        for cls_name, prob in zip(model.classes_, probabilities_raw)
    }

    return {
        "algorithm": predicted_class,
        "probabilities": probabilities,
        "features": feats,
    }


__all__ = [
    "compute_regret",
    "evaluate_decision_tree",
    "recommend",
]
