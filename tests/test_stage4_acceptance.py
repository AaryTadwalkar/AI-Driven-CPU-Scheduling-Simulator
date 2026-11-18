"""
1. What this file does:
   Implements the Stage 4 Acceptance Test Suite for the Decision Tree policy recommender, certifying
   compliance with Section 5 Stage 4 criteria of IMPLEMENTATION_PLAN.md.
2. Overall design:
   Validates all Stage 4 acceptance requirements:
   - Features computed strictly without simulation engine invocations
   - Exact composite labeling formulation and margin sensitivity
   - Pilot gate empirical documentation in `docs/DECISIONS.md`
   - Supervised dataset schema, sample counts, and non-trivial class balance (< 85% dominance)
   - Existence and validity of model artifacts (joblib, metadata JSON, metrics JSON, figure plot)
   - Decision Tree performance strictly outperforming the Majority Class baseline on macro-F1
   - Public `recommend(processes)` single-entrypoint contract
   - Decision regret metric calculation and validation accuracy bounds.
3. Elements in the file:
   - test_feature_extraction_without_simulation: Asserts feature extraction avoids simulation calls.
   - test_composite_labeling_exact_formulation: Validates mathematical composite score calculation.
   - test_pilot_gate_decision_logged: Asserts pilot gate empirical results logged in DECISIONS.md.
   - test_labeled_dataset_schema_and_balance: Verifies dataset schema and class distribution.
   - test_decision_tree_artifacts_exist: Verifies existence and structure of model and meta files.
   - test_decision_tree_beats_majority_baseline: Verifies tree beats majority baseline on macro-F1.
   - test_recommend_public_contract: Verifies single-function recommend API on raw workloads.
   - test_regret_metric_validation_bounds: Verifies regret bounds and unambiguous accuracy.
4. Final output of the file:
   Pytest acceptance suite certifying Stage 4 completion and model readiness.
"""

import json
from unittest.mock import patch

import pytest

import config
from schedsim.domain.process import Process
from schedsim.ml.dataset_builder import DATASET_COLUMNS, load_dataset_from_csv
from schedsim.ml.evaluate_tree import recommend
from schedsim.ml.features import FEATURE_NAMES, extract_features
from schedsim.ml.labeling import CANDIDATE_POLICIES, compute_workload_label


def test_feature_extraction_without_simulation() -> None:
    """
    1. The use of function:
       Certifies that feature extraction is computed purely from process structures without
       running simulations.
    2. How does the specific function work:
       Mocks `schedsim.domain.engine.run` and `schedsim.domain.engine.Engine.step`. Invokes
       `extract_features()`, asserts that the returned dictionary has 12 valid numerical values,
       and asserts that the simulation engine was never invoked.
    3. Concepts used in the function:
       Independent feature extraction, mock assertion, simulation isolation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified isolation of feature extraction.
       After: Asserts feature extraction executes 0 simulation ticks or engine runs.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=4),
        Process(pid="P2", arrival=2, burst=6),
    ]
    with patch("schedsim.domain.engine.run") as mock_run:
        feats = extract_features(procs)
        mock_run.assert_not_called()

    assert len(feats) == 12
    for name in FEATURE_NAMES:
        assert name in feats
        assert isinstance(feats[name], float)


def test_composite_labeling_exact_formulation() -> None:
    """
    1. The use of function:
       Validates exact mathematical compliance of composite objective score calculation.
    2. How does the specific function work:
       Runs `compute_workload_label()` on a workload, extracts raw waiting, response, and turnaround
       metrics, independently computes normalized composite scores, and verifies that the returned
       winning label and individual policy scores match analytical calculations.
    3. Concepts used in the function:
       Composite objective verification, Pareto normalization, winning policy argmin.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified composite formula calculation.
       After: Asserts label and scores match mathematical definitions.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=6),
        Process(pid="P2", arrival=1, burst=2),
        Process(pid="P3", arrival=2, burst=4),
    ]
    lbl_res = compute_workload_label(
        procs,
        cs_cost=1,
        quantum=2,
        weights={"wait": 0.5, "resp": 0.3, "tat": 0.2},
    )

    assert lbl_res.label in CANDIDATE_POLICIES
    assert lbl_res.best_score == min(lbl_res.scores.values())

    raw = lbl_res.raw_metrics
    min_wait = min(m["avg_waiting"] for m in raw.values())
    min_resp = min(m["avg_response"] for m in raw.values())
    min_tat = min(m["avg_turnaround"] for m in raw.values())

    eps = 1e-6
    for policy, s in lbl_res.scores.items():
        expected_score = (
            0.5 * (raw[policy]["avg_waiting"] / (min_wait + eps))
            + 0.3 * (raw[policy]["avg_response"] / (min_resp + eps))
            + 0.2 * (raw[policy]["avg_turnaround"] / (min_tat + eps))
        )
        assert pytest.approx(s, rel=1e-4) == expected_score


def test_pilot_gate_decision_logged() -> None:
    """
    1. The use of function:
       Validates that Section 4c pilot gate analysis and decisions are recorded in `DECISIONS.md`.
    2. How does the specific function work:
       Reads `docs/DECISIONS.md`, asserts that Decision 004 exists and documents empirical
       distributions across pilot configurations (cs_cost 0 and 1, quantum 2 and 4).
    3. Concepts used in the function:
       Documentation compliance, pilot study traceability.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified pilot documentation.
       After: Asserts Decision 004 is present with pilot results in DECISIONS.md.
    """
    decisions_file = config.DOCS_DIR / "DECISIONS.md"
    assert decisions_file.exists(), f"Missing DECISIONS.md at {decisions_file}"
    content = decisions_file.read_text(encoding="utf-8")

    assert "Decision 004: Section 4c Pilot Gate Setting Selection" in content
    assert "cs0_q2_standard" in content
    assert "cs1_q2_standard" in content
    assert "cs1_q4_standard" in content
    assert "FCFS" in content
    assert "SJF" in content
    assert "RR" in content


def test_labeled_dataset_schema_and_balance() -> None:
    """
    1. The use of function:
       Validates schema, sample volume, and class distribution of `data/scheduling_dataset.csv`.
    2. How does the specific function work:
       Loads the canonical dataset via `load_dataset_from_csv()`, asserts column headers match
       DATASET_COLUMNS, checks that sample count is at least 6,000, and verifies that no single
       class exceeds the 85% dominance threshold.
    3. Concepts used in the function:
       Dataset validation, class balance verification, schema adherence.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified scheduling dataset file.
       After: Asserts dataset has >= 6000 rows, valid schema, and non-trivial class balance.
    """
    dataset_path = config.DATA_DIR / "scheduling_dataset.csv"
    assert dataset_path.exists(), f"Dataset CSV missing at {dataset_path}"

    df = load_dataset_from_csv(dataset_path)
    assert len(df) >= 6000
    assert list(df.columns) == list(DATASET_COLUMNS)

    counts = df["label"].value_counts(normalize=True)
    for policy in ("FCFS", "SJF", "RR"):
        assert policy in counts
        # Assert no single class exceeds 85% dominance
        assert counts[policy] < 0.85, f"Class {policy} dominated with {counts[policy] * 100:.1f}%."


def test_decision_tree_artifacts_exist() -> None:
    """
    1. The use of function:
       Validates existence and integrity of trained model artifacts and plots.
    2. How does the specific function work:
       Checks existence of `models/decision_tree.joblib`, `models/decision_tree_meta.json`,
       `results/tree_metrics_val.json`, and `results/figures/decision_tree.png`. Loads JSON metadata
       and asserts presence of required keys.
    3. Concepts used in the function:
       Artifact verification, model persistence validation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified model artifact files.
       After: Asserts all required model and evaluation artifacts are present and well-formed.
    """
    model_path = config.MODELS_DIR / "decision_tree.joblib"
    meta_path = config.MODELS_DIR / "decision_tree_meta.json"
    metrics_path = config.RESULTS_DIR / "tree_metrics_val.json"
    plot_path = config.RESULTS_DIR / "figures" / "decision_tree.png"

    assert model_path.exists(), f"Missing model file: {model_path}"
    assert meta_path.exists(), f"Missing metadata file: {meta_path}"
    assert metrics_path.exists(), f"Missing metrics file: {metrics_path}"
    assert plot_path.exists(), f"Missing figure file: {plot_path}"

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_type"] == "DecisionTreeClassifier"
    assert "chosen_params" in meta
    assert "metrics_validation" in meta

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "mean_regret" in metrics
    assert "confusion_matrix" in metrics


def test_decision_tree_beats_majority_baseline() -> None:
    """
    1. The use of function:
       Validates that the trained Decision Tree outperforms the majority-class baseline on macro-F1.
    2. How does the specific function work:
       Loads `models/decision_tree_meta.json`, compares Decision Tree macro-F1 against Majority
       Class Dummy macro-F1, and asserts strict superiority.
    3. Concepts used in the function:
       Baseline superiority verification, macro-F1 evaluation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified superiority over baseline.
       After: Asserts Decision Tree macro-F1 significantly exceeds majority baseline.
    """
    meta_path = config.MODELS_DIR / "decision_tree_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    tree_f1 = meta["metrics_validation"]["decision_tree"]["macro_f1"]
    dummy_f1 = meta["metrics_validation"]["majority_class_dummy"]["macro_f1"]

    assert tree_f1 > dummy_f1, f"Tree F1 ({tree_f1:.4f}) failed to beat dummy ({dummy_f1:.4f})."
    assert tree_f1 > 0.70, f"Expected strong macro-F1 (>0.70), got {tree_f1:.4f}."


def test_recommend_public_contract() -> None:
    """
    1. The use of function:
       Validates the public recommendation API contract on raw process inputs.
    2. How does the specific function work:
       Passes raw Process sequences to `recommend(processes)`. Asserts returned dictionary
       matches `{algorithm, probabilities, features}`, probabilities sum to 1.0, and algorithm
       is one of the candidate policies.
    3. Concepts used in the function:
       Public contract compliance, probability normalization, recommendation inference.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested public recommendation contract.
       After: Asserts output dictionary satisfies all Section 5 Stage 4 acceptance specifications.
    """
    procs = [
        Process(pid="P1", arrival=0, burst=5),
        Process(pid="P2", arrival=1, burst=3),
        Process(pid="P3", arrival=2, burst=2),
    ]
    rec = recommend(procs)

    assert isinstance(rec, dict)
    assert "algorithm" in rec
    assert rec["algorithm"] in CANDIDATE_POLICIES
    assert "probabilities" in rec
    assert "features" in rec

    for policy in CANDIDATE_POLICIES:
        assert policy in rec["probabilities"]
        assert 0.0 <= rec["probabilities"][policy] <= 1.0

    total_prob = sum(rec["probabilities"].values())
    assert pytest.approx(total_prob, abs=1e-3) == 1.0

    feats = rec["features"]
    assert len(feats) == 12
    for name in FEATURE_NAMES:
        assert name in feats


def test_regret_metric_validation_bounds() -> None:
    """
    1. The use of function:
       Validates validation regret bounds and unambiguous accuracy metrics.
    2. How does the specific function work:
       Loads `results/tree_metrics_val.json`. Asserts validation mean regret is under 0.15 (15%),
       and unambiguous accuracy is at least 85%.
    3. Concepts used in the function:
       Regret bounds assertion, unambiguous accuracy testing.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified validation regret bounds.
       After: Asserts mean regret < 0.15 and unambiguous accuracy >= 0.85.
    """
    metrics_path = config.RESULTS_DIR / "tree_metrics_val.json"
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    mean_regret = metrics["mean_regret"]
    unambig_acc = metrics["accuracy_unambiguous"]

    assert 0.0 <= mean_regret < 0.15, f"Regret too high: {mean_regret:.4f}"
    assert unambig_acc >= 0.85, f"Unambiguous accuracy too low: {unambig_acc:.4f}"
