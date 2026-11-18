"""
1. What this file does:
   Tests CLI entry points for machine learning subcommands: `build-dataset`, `train-tree`, and
   `recommend`.
2. Overall design:
   Validates CLI subcommands using temporary file fixtures and synthetic datasets. Asserts return
   codes, stdout tabular reporting, model artifact generation, probability distribution outputs,
   and error diagnostics.
3. Elements in the file:
   - test_cli_build_dataset_from_workloads: Tests building labeled dataset from workloads via CLI.
   - test_cli_train_tree_and_evaluate: Tests training tree and baseline benchmarks via CLI.
   - test_cli_recommend_preset: Tests policy recommendation inference on preset workloads via CLI.
   - test_cli_recommend_error_missing_model: Tests error reporting when model path is invalid.
   - test_cli_build_dataset_error_handling: Tests error reporting when invalid input is provided.
4. Final output of the file:
   Pytest test suite validating CLI machine learning and inference capabilities.
"""

from pathlib import Path

import pytest

from cli import main
from schedsim.ml.dataset_builder import build_dataset, load_dataset_from_csv, save_dataset_to_csv
from schedsim.workloads.generator import generate_workload
from schedsim.workloads.io import Workload
from schedsim.workloads.scenarios import ALL_SCENARIOS


@pytest.fixture
def synthetic_dataset_file(tmp_path: Path) -> Path:
    """Creates a small labeled dataset CSV for CLI training tests."""
    workloads: list[Workload] = []
    for i in range(18):
        scen = ALL_SCENARIOS[i % len(ALL_SCENARIOS)]
        procs = generate_workload(scenario=scen, n=4, seed=i)
        workloads.append(
            Workload(
                workload_id=f"train_{scen}_{i}",
                scenario=scen,
                seed=i,
                processes=tuple(procs),
            )
        )
    for i in range(9):
        scen = ALL_SCENARIOS[i % len(ALL_SCENARIOS)]
        seed = 100000 + i
        procs = generate_workload(scenario=scen, n=4, seed=seed)
        workloads.append(
            Workload(
                workload_id=f"val_{scen}_{seed}",
                scenario=scen,
                seed=seed,
                processes=tuple(procs),
            )
        )
    df = build_dataset(workloads)
    csv_path = tmp_path / "scheduling_dataset.csv"
    save_dataset_to_csv(df, path=csv_path)
    return csv_path


def test_cli_build_dataset_from_workloads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    1. The use of function:
       Tests the `build-dataset` CLI command synthesizing a labeled dataset.
    2. How does the specific function work:
       Invokes `main(['build-dataset', '--n', '9', '--output', ...])`. Asserts exit code 0,
       verifies stdout report tables, and checks resulting CSV via `load_dataset_from_csv`.
    3. Concepts used in the function:
       CLI dataset building, stdout inspection, CSV schema assertion.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested build-dataset CLI invocation.
       After: Asserts successful execution, stdout report, and valid CSV file.
    """
    out_csv = tmp_path / "cli_dataset.csv"
    exit_code = main(["build-dataset", "--n", "9", "--output", str(out_csv)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Supervised Scheduling Dataset Report" in captured.out
    assert "Class Label Distribution:" in captured.out
    assert "Ambiguous Samples:" in captured.out
    assert str(out_csv) in captured.out

    assert out_csv.exists()
    df = load_dataset_from_csv(out_csv)
    assert len(df) >= 9


def test_cli_train_tree_and_evaluate(
    tmp_path: Path,
    synthetic_dataset_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    1. The use of function:
       Tests the `train-tree` CLI command fitting a decision tree and printing benchmarks.
    2. How does the specific function work:
       Invokes `main(['train-tree', '--dataset', ..., '--output-model', ..., ...])`.
       Asserts exit code 0, verifies stdout contains comparison table, and verifies output files.
    3. Concepts used in the function:
       CLI model training, stdout benchmark verification, artifact creation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested train-tree CLI command.
       After: Asserts exit code 0, model and metadata generated, and comparison benchmarks printed.
    """
    model_path = tmp_path / "models" / "decision_tree.joblib"
    meta_path = tmp_path / "models" / "decision_tree_meta.json"
    metrics_path = tmp_path / "results" / "tree_metrics_val.json"

    exit_code = main(
        [
            "train-tree",
            "--dataset",
            str(synthetic_dataset_file),
            "--output-model",
            str(model_path),
            "--output-meta",
            str(meta_path),
            "--output-metrics",
            str(metrics_path),
        ]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Decision Tree Training & Benchmark Report" in captured.out
    assert "Validation Model Comparison:" in captured.out
    assert "Majority Class Dummy" in captured.out
    assert "Random Forest" in captured.out
    assert "K-Nearest Neighbors" in captured.out
    assert "Validation Mean Regret:" in captured.out

    assert model_path.exists()
    assert meta_path.exists()
    assert metrics_path.exists()


def test_cli_recommend_preset(
    tmp_path: Path,
    synthetic_dataset_file: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    1. The use of function:
       Tests the `recommend` CLI command producing an inference report for a preset workload.
    2. How does the specific function work:
       Trains a tree, invokes `main(['recommend', '--preset', '1', '--model', ...])`.
       Asserts exit code 0 and verifies recommended policy, class probabilities, and feature
       values are printed.
    3. Concepts used in the function:
       CLI inference, preset workload recommendation, probability reporting.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested recommend CLI command.
       After: Asserts exit code 0 and complete recommendation report printed to stdout.
    """
    model_path = tmp_path / "models" / "decision_tree.joblib"
    meta_path = tmp_path / "models" / "decision_tree_meta.json"
    metrics_path = tmp_path / "results" / "tree_metrics_val.json"
    main(
        [
            "train-tree",
            "--dataset",
            str(synthetic_dataset_file),
            "--output-model",
            str(model_path),
            "--output-meta",
            str(meta_path),
            "--output-metrics",
            str(metrics_path),
        ]
    )

    capsys.readouterr()  # clear training stdout
    exit_code = main(["recommend", "--preset", "1", "--model", str(model_path)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Policy Recommendation Report: Preset=1" in captured.out
    assert "Recommended Policy:" in captured.out
    assert "Class Probabilities:" in captured.out
    assert "Extracted Workload Features:" in captured.out


def test_cli_recommend_error_missing_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    1. The use of function:
       Tests error handling when `recommend` is called with a non-existent model path.
    2. How does the specific function work:
       Invokes `main(['recommend', '--model', 'missing.joblib'])`, asserts exit code 1, and
       checks stderr error message.
    3. Concepts used in the function:
       CLI error handling, missing file diagnostics.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested recommend error handling.
       After: Asserts exit code 1 and error printed to stderr.
    """
    missing = tmp_path / "non_existent.joblib"
    exit_code = main(["recommend", "--preset", "1", "--model", str(missing)])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Recommendation error:" in captured.err


def test_cli_build_dataset_error_handling(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    1. The use of function:
       Tests error handling in `build-dataset` when invalid options are provided.
    2. How does the specific function work:
       Passes invalid non-existent path to `--input`, asserts exit code 0 or 1 depending on
       synthesis fallback, or invalid argument error.
    3. Concepts used in the function:
       CLI resilience testing.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested build-dataset error handling.
       After: Asserts graceful exit without uncaught exceptions.
    """
    bad_output = tmp_path / "bad_dir\0" / "output.csv"
    exit_code = main(["build-dataset", "--n", "2", "--output", str(bad_output)])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Dataset building error:" in captured.err
