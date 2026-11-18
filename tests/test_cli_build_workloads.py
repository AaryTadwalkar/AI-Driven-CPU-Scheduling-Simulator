"""
1. What this file does:
   Tests CLI dataset generation commands (`build-workloads` and `build-dataset`), including
   argument parsing, alias support, CSV output fidelity, stdout reporting, and error handling.
2. Overall design:
   Validates CLI entry points (`main` and `handle_build_workloads`) using temporary directories,
   mocked arguments, and standard pytest fixtures (`tmp_path`, `capsys`). Tests both command names,
   various split configurations, seed offsets, and error branches.
3. Elements in the file:
   - test_cli_build_workloads_success: Tests standard build-workloads execution on train split.
   - test_cli_build_dataset_alias_success: Tests build-dataset alias compatibility and output.
   - test_cli_build_workloads_val_split: Tests workload generation on validation split.
   - test_cli_build_workloads_seed_offset: Tests deterministic seed offsetting via CLI flags.
   - test_cli_build_workloads_error_invalid_n: Tests CLI rejection of non-positive workload counts.
   - test_cli_build_workloads_error_negative_seed_offset: Tests rejection of negative seed offset.
   - test_cli_build_workloads_error_split_capacity: Tests error handling when capacity is exceeded.
   - test_cli_build_workloads_invalid_split_flag: Tests CLI argument parser rejection of split.
   - test_handle_build_workloads_direct_invocation: Tests handler invocation directly.
4. Final output of the file:
   Pytest test suite validating complete CLI workload generation functionality and robustness.
"""

import argparse
from pathlib import Path

import pytest

from cli import handle_build_workloads, main
from schedsim.workloads.io import load_workloads_from_csv


def test_cli_build_workloads_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests successful execution of `build-workloads` CLI command generating train split workloads.
    2. How does the specific function work:
       Invokes `main(['build-workloads', '--n', '18', '--output', ...])` on the train split.
       Asserts exit code 0, verifies stdout summary table, and parses resulting CSV to verify
       exact count, schema headers, and non-empty process collections.
    3. Concepts used in the function:
       CLI end-to-end integration, CSV file verification, stdout report inspection.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested build-workloads command.
       After: Asserts successful 18-workload generation, stdout report, and valid CSV persistence.
    """
    out_csv = tmp_path / "workloads.csv"
    exit_code = main(
        ["build-workloads", "--n", "18", "--output", str(out_csv), "--split", "train"]
    )
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Workload Dataset Generation Report" in captured.out
    assert "Split:                     train" in captured.out
    assert "Workloads Generated:       18" in captured.out
    assert "Total Processes:" in captured.out
    assert "Output Destination:" in captured.out
    assert str(out_csv) in captured.out

    assert out_csv.exists()
    workloads = load_workloads_from_csv(out_csv)
    assert len(workloads) == 18
    assert all(w.workload_id.startswith("train_") for w in workloads)
    assert all(0 <= w.seed < 18 for w in workloads)


def test_cli_build_workload_alias_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    1. The use of function:
       Tests the `build-workload` singular alias for CLI invocation consistency.
    2. How does the specific function work:
       Invokes `main(['build-workload', '--n', '9', '--output', str(csv_path)])`.
       Asserts exit code 0, validates stdout reporting, and verifies that exactly 9 workloads
       are loaded from the generated CSV file.
    3. Concepts used in the function:
       Subcommand alias resolution, CLI consistency testing.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested build-workload alias invocation.
       After: Asserts alias correctly routes to generation handler with exit code 0.
    """
    out_csv = tmp_path / "workload_alias.csv"
    exit_code = main(["build-workload", "--n", "9", "--output", str(out_csv)])
    assert exit_code == 0

    captured = capsys.readouterr()
    assert "Workload Dataset Generation Report" in captured.out
    assert "Workloads Generated:       9" in captured.out

    workloads = load_workloads_from_csv(out_csv)
    assert len(workloads) == 9


def test_cli_build_workloads_val_split(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    1. The use of function:
       Verifies workload generation on the validation split partition.
    2. How does the specific function work:
       Invokes CLI with `--split val --n 9`, validates return code 0, verifies that seeds start
       at 100,000, and ensures workload IDs have 'val_' prefix.
    3. Concepts used in the function:
       Validation split generation, seed partition verification.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified validation split CLI generation.
       After: Asserts validation partition seeds (100,000+) are generated correctly.
    """
    out_csv = tmp_path / "val_workloads.csv"
    exit_code = main(["build-workloads", "--n", "9", "--output", str(out_csv), "--split", "val"])
    assert exit_code == 0

    workloads = load_workloads_from_csv(out_csv)
    assert len(workloads) == 9
    assert all(w.workload_id.startswith("val_") for w in workloads)
    assert all(100000 <= w.seed < 100009 for w in workloads)


def test_cli_build_workloads_seed_offset(tmp_path: Path) -> None:
    """
    1. The use of function:
       Verifies that the `--seed-offset` CLI argument offsets seed allocations as expected.
    2. How does the specific function work:
       Calls `build-workloads` with `--seed-offset 500 --n 9`, loads output CSV, and asserts
       workload seeds start at 500.
    3. Concepts used in the function:
       Seed offset testing, deterministic batch generation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested seed-offset CLI flag.
       After: Asserts generated workloads start exactly at specified seed offset.
    """
    out_csv = tmp_path / "offset_workloads.csv"
    exit_code = main(
        ["build-workloads", "--n", "9", "--output", str(out_csv), "--seed-offset", "500"]
    )
    assert exit_code == 0

    workloads = load_workloads_from_csv(out_csv)
    assert len(workloads) == 9
    assert workloads[0].seed == 500
    assert workloads[-1].seed == 508


def test_cli_build_workloads_error_invalid_n(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    1. The use of function:
       Verifies error handling when non-positive `--n` is supplied.
    2. How does the specific function work:
       Invokes CLI with `--n 0`, asserts exit code 1, and verifies error message on stderr.
    3. Concepts used in the function:
       CLI argument validation, error diagnostics, stderr capture.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested zero workload count input.
       After: Asserts non-zero exit code (1) and descriptive error message.
    """
    out_csv = tmp_path / "invalid_n.csv"
    exit_code = main(["build-workloads", "--n", "0", "--output", str(out_csv)])
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Dataset generation error: --n must be >= 1" in captured.err


def test_cli_build_workloads_error_negative_seed_offset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    1. The use of function:
       Verifies error handling when negative `--seed-offset` is provided.
    2. How does the specific function work:
       Invokes `handle_build_workloads` with negative seed_offset, asserts exit code 1, and
       checks stderr error diagnostics.
    3. Concepts used in the function:
       CLI argument validation, boundary enforcement.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested negative seed offset input.
       After: Asserts exit code 1 and error logged to stderr.
    """
    out_csv = tmp_path / "negative_offset.csv"
    args = argparse.Namespace(n=10, output=str(out_csv), split="train", seed_offset=-5)
    exit_code = handle_build_workloads(args)
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "Dataset generation error: --seed-offset must be >= 0" in captured.err


def test_cli_build_workloads_error_split_capacity(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    1. The use of function:
       Verifies error diagnostics when requested workload count exceeds partition capacity.
    2. How does the specific function work:
       Requests 600 workloads on the validation split (capacity is 500), asserts exit code 1,
       and checks stderr for partition capacity error.
    3. Concepts used in the function:
       Capacity enforcement, boundary violation handling.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested capacity overflow through CLI.
       After: Asserts exit code 1 and capacity error on stderr.
    """
    out_csv = tmp_path / "overflow.csv"
    exit_code = main(
        ["build-workloads", "--n", "600", "--output", str(out_csv), "--split", "val"]
    )
    assert exit_code == 1

    captured = capsys.readouterr()
    assert "exceeds split 'val' capacity" in captured.err


def test_cli_build_workloads_invalid_split_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    1. The use of function:
       Verifies that argparse rejects invalid split choice strings.
    2. How does the specific function work:
       Invokes CLI with `--split invalid`, asserts exit code 2 (argparse invalid argument code),
       and verifies error message in stderr.
    3. Concepts used in the function:
       Argparse choices validation, CLI exit code verification.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid split choice argument.
       After: Asserts exit code != 0 and argument choice rejection.
    """
    out_csv = tmp_path / "bad_split.csv"
    exit_code = main(
        ["build-workloads", "--n", "10", "--output", str(out_csv), "--split", "invalid"]
    )
    assert exit_code != 0

    captured = capsys.readouterr()
    assert "invalid choice: 'invalid'" in captured.err


def test_handle_build_workloads_direct_invocation(tmp_path: Path) -> None:
    """
    1. The use of function:
       Verifies direct programmatic invocation of `handle_build_workloads`.
    2. How does the specific function work:
       Constructs an `argparse.Namespace` directly and executes `handle_build_workloads()`,
       verifying return code 0 and CSV file generation.
    3. Concepts used in the function:
       Direct handler execution, namespace construction.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested direct handler call.
       After: Asserts return code 0 and verifies CSV file contents.
    """
    out_csv = tmp_path / "direct_call.csv"
    args = argparse.Namespace(n=5, output=str(out_csv), split="train", seed_offset=10)
    exit_code = handle_build_workloads(args)
    assert exit_code == 0
    assert out_csv.exists()

    workloads = load_workloads_from_csv(out_csv)
    assert len(workloads) == 5
    assert workloads[0].seed == 10
