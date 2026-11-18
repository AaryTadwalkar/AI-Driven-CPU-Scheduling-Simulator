"""
1. What this file does:
   Initializes the machine learning package for Decision Tree policy recommendation, re-exporting
   feature extraction, composite labeling, dataset construction, training, evaluation, and
   inference APIs.
2. Overall design:
   Aggregates core ML submodules into a clean public package interface:
   - features: 12 workload-derived statistical and structural metrics.
   - labeling: Multi-objective composite scoring, margin estimation, and pilot study runner.
   - dataset_builder: Supervised CSV dataset synthesis and I/O.
   - train_tree: Grid search cross-validation and baseline comparison benchmarking.
   - evaluate_tree: Validation evaluation, regret assessment, tree plotting, and `recommend()` API.
3. Elements in the file:
   - Feature extraction helpers: FEATURE_NAMES, extract_features, extract_features_vector.
   - Labeling tools: CANDIDATE_POLICIES, LabelResult, compute_workload_label, run_pilot_study.
   - Dataset helpers: DATASET_COLUMNS, build_dataset, build_dataset_row, load/save dataset.
   - Training & evaluation: train_decision_tree, evaluate_decision_tree, compute_regret, recommend.
4. Final output of the file:
   Clean top-level import namespace for schedsim.ml.
"""

from schedsim.ml.dataset_builder import (
    DATASET_COLUMNS,
    build_dataset,
    build_dataset_row,
    load_dataset_from_csv,
    save_dataset_to_csv,
)
from schedsim.ml.evaluate_tree import (
    compute_regret,
    evaluate_decision_tree,
    recommend,
)
from schedsim.ml.features import (
    FEATURE_NAMES,
    extract_features,
    extract_features_vector,
)
from schedsim.ml.labeling import (
    CANDIDATE_POLICIES,
    LabelResult,
    compute_workload_label,
    run_pilot_study,
)
from schedsim.ml.train_tree import (
    evaluate_model_metrics,
    train_decision_tree,
)

__all__ = [
    "CANDIDATE_POLICIES",
    "DATASET_COLUMNS",
    "FEATURE_NAMES",
    "LabelResult",
    "build_dataset",
    "build_dataset_row",
    "compute_regret",
    "compute_workload_label",
    "evaluate_decision_tree",
    "evaluate_model_metrics",
    "extract_features",
    "extract_features_vector",
    "load_dataset_from_csv",
    "recommend",
    "run_pilot_study",
    "save_dataset_to_csv",
    "train_decision_tree",
]
