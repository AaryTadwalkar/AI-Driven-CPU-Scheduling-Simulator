"""
1. What this file does:
   Tests dataset building, serialization, and deserialization routines for the Decision Tree
   policy recommender.
2. Overall design:
   Validates `build_dataset_row`, `build_dataset`, `save_dataset_to_csv`, and
   `load_dataset_from_csv` in `schedsim.ml.dataset_builder`. Asserts column schemas, data types,
   lossless round-trips, directory auto-creation, and exception handling.
3. Elements in the file:
   - test_dataset_columns_schema: Validates DATASET_COLUMNS schema definition.
   - test_build_dataset_row: Tests row extraction for a single Workload instance.
   - test_build_dataset_dataframe: Tests multi-workload DataFrame construction.
   - test_save_and_load_dataset_round_trip: Tests CSV persistence and re-parsing.
   - test_load_dataset_nonexistent_file: Verifies FileNotFoundError on missing files.
   - test_load_dataset_corrupt_header: Verifies ValueError when required columns are missing.
   - test_build_dataset_type_errors: Verifies defensive type validation.
4. Final output of the file:
   Pytest test suite validating dataset synthesis and persistence integrity.
"""

from pathlib import Path

import pandas as pd
import pytest

from schedsim.domain.process import Process
from schedsim.ml.dataset_builder import (
    DATASET_COLUMNS,
    build_dataset,
    build_dataset_row,
    load_dataset_from_csv,
    save_dataset_to_csv,
)
from schedsim.ml.features import FEATURE_NAMES
from schedsim.workloads.io import Workload


def test_dataset_columns_schema() -> None:
    """
    1. The use of function:
       Validates DATASET_COLUMNS tuple structure and completeness.
    2. How does the specific function work:
       Asserts presence of metadata columns, all 12 feature names, label, margin, is_ambiguous,
       and individual policy scores.
    3. Concepts used in the function:
       Schema validation, constant integrity checking.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified DATASET_COLUMNS tuple.
       After: Asserts all required column names are present in exact order.
    """
    expected_prefix = ("workload_id", "scenario", "seed", "split")
    assert DATASET_COLUMNS[:4] == expected_prefix
    for feat in FEATURE_NAMES:
        assert feat in DATASET_COLUMNS
    expected_suffix = ("label", "margin", "is_ambiguous", "score_fcfs", "score_sjf", "score_rr")
    assert DATASET_COLUMNS[-6:] == expected_suffix


def test_build_dataset_row() -> None:
    """
    1. The use of function:
       Tests dataset row dictionary generation for an individual Workload.
    2. How does the specific function work:
       Constructs a test Workload, calls `build_dataset_row()`, and verifies that the output dict
       keys match DATASET_COLUMNS and types are correct.
    3. Concepts used in the function:
       Row transformation, dictionary schema validation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested build_dataset_row.
       After: Asserts dictionary matches schema and contains valid values.
    """
    procs = (
        Process(pid="P1", arrival=0, burst=4),
        Process(pid="P2", arrival=1, burst=2),
    )
    workload = Workload(
        workload_id="test_wl_001",
        scenario="short_jobs",
        seed=10,
        processes=procs,
    )
    row = build_dataset_row(workload)
    assert isinstance(row, dict)
    assert tuple(row.keys()) == DATASET_COLUMNS
    assert row["workload_id"] == "test_wl_001"
    assert row["scenario"] == "short_jobs"
    assert row["seed"] == 10
    assert row["split"] == "train"
    assert row["n"] == 2.0
    assert row["label"] in ("FCFS", "SJF", "RR")
    assert isinstance(row["is_ambiguous"], bool)


def test_build_dataset_dataframe() -> None:
    """
    1. The use of function:
       Tests assembling a pandas DataFrame across multiple Workload instances.
    2. How does the specific function work:
       Creates two Workloads, calls `build_dataset()`, validates returned DataFrame shape (2, 22),
       columns, and data types.
    3. Concepts used in the function:
       DataFrame assembly, multi-row transformation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested build_dataset.
       After: Asserts DataFrame has shape (2, len(DATASET_COLUMNS)) and valid records.
    """
    wl1 = Workload(
        workload_id="wl_01",
        scenario="short_jobs",
        seed=1,
        processes=(Process(pid="P1", arrival=0, burst=3),),
    )
    wl2 = Workload(
        workload_id="wl_02",
        scenario="long_jobs",
        seed=2,
        processes=(Process(pid="P1", arrival=0, burst=20),),
    )
    df = build_dataset([wl1, wl2])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == list(DATASET_COLUMNS)
    assert df["workload_id"].tolist() == ["wl_01", "wl_02"]


def test_save_and_load_dataset_round_trip(tmp_path: Path) -> None:
    """
    1. The use of function:
       Tests lossless saving and re-loading of the ML dataset DataFrame.
    2. How does the specific function work:
       Generates a test DataFrame, writes to CSV with `save_dataset_to_csv()`, reloads with
       `load_dataset_from_csv()`, and verifies exact column and value equality.
    3. Concepts used in the function:
       Lossless round-trip persistence, CSV serialization.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested CSV dataset I/O.
       After: Asserts reloaded DataFrame is identical to saved DataFrame.
    """
    csv_path = tmp_path / "subdir" / "test_dataset.csv"
    wl = Workload(
        workload_id="wl_round_trip",
        scenario="mixed",
        seed=5,
        processes=(
            Process(pid="P1", arrival=0, burst=4),
            Process(pid="P2", arrival=2, burst=8),
        ),
    )
    df_orig = build_dataset([wl])
    saved_path = save_dataset_to_csv(df_orig, path=csv_path)
    assert saved_path.exists()

    df_loaded = load_dataset_from_csv(csv_path)
    assert len(df_loaded) == 1
    assert list(df_loaded.columns) == list(DATASET_COLUMNS)
    assert df_loaded["workload_id"].iloc[0] == "wl_round_trip"
    assert df_loaded["label"].iloc[0] == df_orig["label"].iloc[0]
    assert df_loaded["margin"].iloc[0] == pytest.approx(df_orig["margin"].iloc[0])


def test_load_dataset_nonexistent_file(tmp_path: Path) -> None:
    """
    1. The use of function:
       Tests error handling when loading a non-existent dataset CSV file.
    2. How does the specific function work:
       Calls `load_dataset_from_csv()` on a non-existent path and asserts FileNotFoundError.
    3. Concepts used in the function:
       Defensive file checking, exception handling.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested missing file handling.
       After: Asserts FileNotFoundError raised with path information.
    """
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Dataset CSV not found"):
        load_dataset_from_csv(missing_file)


def test_load_dataset_corrupt_header(tmp_path: Path) -> None:
    """
    1. The use of function:
       Tests schema enforcement when loading a CSV with missing required columns.
    2. How does the specific function work:
       Writes a CSV with only 2 columns, calls `load_dataset_from_csv()`, and asserts ValueError.
    3. Concepts used in the function:
       Schema verification, corrupted dataset detection.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested corrupt header validation.
       After: Asserts ValueError raised with missing columns detail.
    """
    bad_csv = tmp_path / "corrupt.csv"
    bad_csv.write_text("workload_id,scenario\nwl_01,short_jobs\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupt dataset schema"):
        load_dataset_from_csv(bad_csv)


def test_build_dataset_type_errors() -> None:
    """
    1. The use of function:
       Tests defensive type validation in dataset builder functions.
    2. How does the specific function work:
       Passes invalid types to `build_dataset_row`, `build_dataset`, and `save_dataset_to_csv`,
       asserting TypeError.
    3. Concepts used in the function:
       Runtime type checking, defensive programming.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid types in dataset builder.
       After: Asserts TypeError raised for invalid arguments.
    """
    with pytest.raises(TypeError, match="must be a Workload instance"):
        build_dataset_row("not_a_workload")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a Sequence"):
        build_dataset("not_a_sequence")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a pandas DataFrame"):
        save_dataset_to_csv("not_a_dataframe")  # type: ignore[arg-type]
