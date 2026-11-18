"""
1. What this file does:
   Tests Decision Tree training, baseline benchmarking, evaluation metrics (accuracy, macro-F1,
   regret, confusion matrix), tree plotting, and the public `recommend()` inference API.
2. Overall design:
   Uses temporary paths and synthetic datasets to validate `train_decision_tree`,
   `evaluate_decision_tree`, `compute_regret`, and `recommend`. Verifies model serialization,
   metadata formatting, regret mathematics, probability distribution outputs, and error handling.
3. Elements in the file:
   - test_compute_regret_math: Verifies analytical correctness of relative regret calculations.
   - test_train_decision_tree_workflow: Tests end-to-end training and baseline comparisons.
   - test_evaluate_decision_tree_workflow: Tests validation evaluation, metrics JSON, and plotting.
   - test_recommend_inference_api: Tests public `recommend()` function on raw process inputs.
   - test_recommend_missing_model_error: Verifies FileNotFoundError when model path is missing.
   - test_recommend_invalid_input_error: Verifies TypeError on invalid process input.
4. Final output of the file:
   Pytest test suite certifying Decision Tree training, evaluation, and recommendation pipelines.
"""

from pathlib import Path

import pandas as pd
import pytest

from schedsim.domain.process import Process
from schedsim.ml.dataset_builder import DATASET_COLUMNS, build_dataset
from schedsim.ml.evaluate_tree import compute_regret, evaluate_decision_tree, recommend
from schedsim.ml.train_tree import train_decision_tree
from schedsim.workloads.generator import generate_workload
from schedsim.workloads.io import Workload
from schedsim.workloads.scenarios import ALL_SCENARIOS


@pytest.fixture
def mini_workload_dataset(tmp_path: Path) -> pd.DataFrame:
    """
    Creates a small, balanced multi-scenario dataset with Train and Val splits for ML testing.
    """
    workloads: list[Workload] = []
    # Generate 18 train workloads and 9 val workloads across all scenarios
    for i in range(18):
        scenario = ALL_SCENARIOS[i % len(ALL_SCENARIOS)]
        procs = generate_workload(scenario=scenario, n=4, seed=i)
        workloads.append(
            Workload(
                workload_id=f"train_{scenario}_{i}",
                scenario=scenario,
                seed=i,
                processes=tuple(procs),
            )
        )

    for i in range(9):
        scenario = ALL_SCENARIOS[i % len(ALL_SCENARIOS)]
        seed = 100000 + i
        procs = generate_workload(scenario=scenario, n=4, seed=seed)
        workloads.append(
            Workload(
                workload_id=f"val_{scenario}_{seed}",
                scenario=scenario,
                seed=seed,
                processes=tuple(procs),
            )
        )

    df = build_dataset(workloads)
    return df


def test_compute_regret_math() -> None:
    """
    1. The use of function:
       Tests the mathematical calculation of decision regret on synthetic scores.
    2. How does the specific function work:
       Constructs synthetic DataFrame with score_fcfs, score_sjf, score_rr. Tests that choosing
       optimal policy yields 0.0 regret, and sub-optimal policy yields expected relative excess.
    3. Concepts used in the function:
       Decision regret, relative loss calculation, boundary conditions.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested compute_regret formula.
       After: Asserts exact match with analytical regret values.
    """
    data = [
        {"score_fcfs": 1.0, "score_sjf": 2.0, "score_rr": 3.0},
        {"score_fcfs": 2.0, "score_sjf": 1.0, "score_rr": 4.0},
    ]
    df = pd.DataFrame(data)

    # Perfect predictions: FCFS for row 0 (best=1.0), SJF for row 1 (best=1.0)
    assert compute_regret(df, ["FCFS", "SJF"]) == pytest.approx(0.0)

    # Sub-optimal predictions: SJF for row 0 (2.0 vs 1.0 -> excess 1.0), FCFS for row 1 (2.0 vs 1.0)
    # Mean relative excess: ( (2.0 - 1.0)/1.0 + (2.0 - 1.0)/1.0 ) / 2 = 1.0
    assert compute_regret(df, ["SJF", "FCFS"]) == pytest.approx(1.0, rel=1e-3)

    # Empty DataFrame edge case
    assert compute_regret(pd.DataFrame(columns=list(DATASET_COLUMNS)), []) == 0.0


def test_train_decision_tree_workflow(tmp_path: Path, mini_workload_dataset: pd.DataFrame) -> None:
    """
    1. The use of function:
       Tests `train_decision_tree` executing grid search and persisting model/metadata.
    2. How does the specific function work:
       Passes synthetic DataFrame to `train_decision_tree()`. Asserts model joblib and metadata JSON
       are written, verifies metadata structure and presence of baseline comparisons.
    3. Concepts used in the function:
       Model training workflow, baseline comparison, metadata validation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untrained model on synthetic dataset.
       After: Asserts files created and metadata reports validation metrics for all models.
    """
    model_path = tmp_path / "models" / "decision_tree.joblib"
    meta_path = tmp_path / "models" / "decision_tree_meta.json"

    meta = train_decision_tree(
        dataset=mini_workload_dataset,
        output_model_path=model_path,
        output_meta_path=meta_path,
        seed=123,
    )

    assert model_path.exists()
    assert meta_path.exists()

    assert meta["model_type"] == "DecisionTreeClassifier"
    assert "chosen_params" in meta
    assert "metrics_validation" in meta
    metrics = meta["metrics_validation"]
    assert "decision_tree" in metrics
    assert "majority_class_dummy" in metrics
    assert "random_forest" in metrics
    assert "k_nearest_neighbors" in metrics
    assert "accuracy" in metrics["decision_tree"]
    assert "macro_f1" in metrics["decision_tree"]


def test_evaluate_decision_tree_workflow(
    tmp_path: Path, mini_workload_dataset: pd.DataFrame
) -> None:
    """
    1. The use of function:
       Tests `evaluate_decision_tree` validation reporting, regret calculation, and visualization.
    2. How does the specific function work:
       Trains a tree on synthetic dataset, then runs `evaluate_decision_tree()`. Verifies metrics
       JSON and plot PNG are generated and metrics fields are populated.
    3. Concepts used in the function:
       Validation evaluation, decision tree visualization, plot saving.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Un-evaluated model on validation split.
       After: Asserts metrics JSON written and plot image created.
    """
    model_path = tmp_path / "models" / "decision_tree.joblib"
    meta_path = tmp_path / "models" / "decision_tree_meta.json"
    train_decision_tree(
        dataset=mini_workload_dataset,
        output_model_path=model_path,
        output_meta_path=meta_path,
        seed=123,
    )

    metrics_path = tmp_path / "results" / "tree_metrics_val.json"
    plot_path = tmp_path / "results" / "figures" / "decision_tree.png"

    metrics = evaluate_decision_tree(
        model_path=model_path,
        dataset=mini_workload_dataset,
        output_metrics_path=metrics_path,
        output_plot_path=plot_path,
    )

    assert metrics_path.exists()
    assert plot_path.exists()
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "mean_regret" in metrics
    assert "confusion_matrix" in metrics
    assert "feature_importances" in metrics
    assert metrics["mean_regret"] >= 0.0


def test_recommend_inference_api(tmp_path: Path, mini_workload_dataset: pd.DataFrame) -> None:
    """
    1. The use of function:
       Tests the public `recommend()` function on a raw Process sequence.
    2. How does the specific function work:
       Trains model, invokes `recommend(processes, model_path=model_path)`, validates returned
       dictionary format (`algorithm`, `probabilities`, `features`), and verifies probabilities
       sum to 1.0.
    3. Concepts used in the function:
       Inference API verification, probability normalization, recommendation contract.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested recommend function.
       After: Asserts valid recommendation returned conforming to acceptance criteria.
    """
    model_path = tmp_path / "models" / "decision_tree.joblib"
    meta_path = tmp_path / "models" / "decision_tree_meta.json"
    train_decision_tree(
        dataset=mini_workload_dataset,
        output_model_path=model_path,
        output_meta_path=meta_path,
        seed=123,
    )

    procs = [
        Process(pid="P1", arrival=0, burst=4),
        Process(pid="P2", arrival=1, burst=2),
        Process(pid="P3", arrival=2, burst=6),
    ]
    rec = recommend(procs, model_path=model_path)

    assert isinstance(rec, dict)
    assert "algorithm" in rec
    assert rec["algorithm"] in ("FCFS", "SJF", "RR")
    assert "probabilities" in rec
    assert "features" in rec

    probs = rec["probabilities"]
    assert len(probs) >= 1
    assert pytest.approx(sum(probs.values()), abs=1e-3) == 1.0
    assert len(rec["features"]) == 12


def test_recommend_missing_model_error(tmp_path: Path) -> None:
    """
    1. The use of function:
       Tests error handling in `recommend` when the specified model path does not exist.
    2. How does the specific function work:
       Passes a non-existent path to `recommend` and asserts FileNotFoundError.
    3. Concepts used in the function:
       Defensive path validation, missing model guidance.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested missing model handling in recommend.
       After: Asserts FileNotFoundError raised with guidance to run train-tree.
    """
    non_existent = tmp_path / "missing_model.joblib"
    procs = [Process(pid="P1", arrival=0, burst=4)]
    with pytest.raises(FileNotFoundError, match="Trained Decision Tree model not found"):
        recommend(procs, model_path=non_existent)


def test_recommend_invalid_input_error() -> None:
    """
    1. The use of function:
       Tests input validation in `recommend` on non-sequence inputs.
    2. How does the specific function work:
       Passes invalid string input and asserts TypeError.
    3. Concepts used in the function:
       Defensive input validation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid inputs to recommend.
       After: Asserts TypeError raised.
    """
    with pytest.raises(TypeError, match="processes must be a Sequence"):
        recommend("invalid_procs")  # type: ignore[arg-type]
