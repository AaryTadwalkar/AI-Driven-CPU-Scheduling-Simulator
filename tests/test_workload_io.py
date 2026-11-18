"""
1. What this file does:
   Defines unit tests for workload CSV persistence, deserialization, and the Workload aggregate
   dataclass implemented in schedsim.workloads.io.
2. Overall design:
   Verifies lossless round-trip serialization across all 9 canonical workload scenarios, CSV
   schema validation, directory auto-creation, order preservation, domain boundary enforcement,
   and robust error handling for missing or corrupted files.
3. Elements in the file:
   - TestWorkloadDataclass: Tests Workload post-init validation, canonicalization, and immutability.
   - TestSaveWorkloads: Tests CSV writing, header emission, directory creation, and type checks.
   - TestLoadWorkloadsRoundTrip: Tests lossless serialization round-trips and ordering.
   - TestIOErrorHandling: Tests exceptions for missing files, corrupt headers, and malformed rows.
   - TestPackageExports: Tests module and package-level symbols and __all__ exports.
4. Final output of the file:
   Provides comprehensive test coverage for Stage 3 Task 2 executed during pytest runs.
"""

import csv
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from schedsim.domain.process import Process
from schedsim.workloads import (
    ALL_SCENARIOS,
    Workload,
    generate_workload,
    load_workloads_from_csv,
    save_workloads_to_csv,
)
from schedsim.workloads.io import WORKLOAD_CSV_HEADER


class TestWorkloadDataclass:
    """
    1. The use of class:
       Validates construction, type enforcement, invariant checking, and immutability of Workload.
    2. What the class contains:
       Unit tests covering valid creation with tuple/list, scenario normalization, negative seed
       rejection, invalid process sequence handling, duplicate PID rejection, and hashability.
    3. Technologies used and why:
       pytest assertions and dataclass reflection to ensure immutable domain object semantics.
    4. Key design and overall design of the class:
       Unit test fixture asserting that Workload acts as a robust aggregate root for workloads.
    """

    def test_valid_workload_instantiation_with_tuple(self) -> None:
        """
        1. The use of function:
           Verifies valid Workload instantiation when processes are supplied as a tuple.
        2. How does the specific function work:
           Constructs a Workload with tuple of Process instances and asserts all attributes match.
        3. Concepts used in the function:
           Dataclass instantiation, attribute verification, tuple preservation.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Raw processes tuple and metadata.
           After: Verified Workload instance with exact matching fields.
        """
        procs = (
            Process(pid="P1", arrival=0, burst=5, priority=1),
            Process(pid="P2", arrival=2, burst=3, priority=0),
        )
        wl = Workload(
            workload_id="wl_001",
            scenario="short_jobs",
            seed=42,
            processes=procs,
        )
        assert wl.workload_id == "wl_001"
        assert wl.scenario == "short_jobs"
        assert wl.seed == 42
        assert wl.processes == procs
        assert isinstance(wl.processes, tuple)

    def test_valid_workload_instantiation_with_list(self) -> None:
        """
        1. The use of function:
           Verifies that supplying a list of Process objects automatically coerces to a tuple.
        2. How does the specific function work:
           Instantiates a Workload with a list of Process instances and asserts processes is tuple.
        3. Concepts used in the function:
           Type coercion in post_init, defensive immutability.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Mutable list of Process instances passed to constructor.
           After: Confirmed that wl.processes is coerced to an immutable tuple.
        """
        proc_list = [
            Process(pid="P1", arrival=0, burst=4),
            Process(pid="P2", arrival=1, burst=6),
        ]
        wl = Workload(
            workload_id="wl_list",
            scenario="convoy",
            seed=10,
            processes=proc_list,  # type: ignore[arg-type]
        )
        assert isinstance(wl.processes, tuple)
        assert wl.processes == tuple(proc_list)

    def test_scenario_canonicalization(self) -> None:
        """
        1. The use of function:
           Verifies that scenario names are canonicalized to lowercase canonical identifiers.
        2. How does the specific function work:
           Supplies mixed-case scenario strings and asserts canonical lowercase representation.
        3. Concepts used in the function:
           Scenario normalization via get_scenario inside Workload post-init.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Non-normalized scenario strings like 'SHORT_JOBS' or 'Convoy'.
           After: Canonical scenario string stored on the Workload instance.
        """
        p = (Process(pid="P1", arrival=0, burst=2),)
        wl1 = Workload(workload_id="w1", scenario="SHORT_JOBS", seed=1, processes=p)
        assert wl1.scenario == "short_jobs"

        wl2 = Workload(workload_id="w2", scenario="  Convoy  ", seed=2, processes=p)
        assert wl2.scenario == "convoy"

    def test_invalid_workload_id_raises(self) -> None:
        """
        1. The use of function:
           Verifies rejection of invalid workload_id types and empty/whitespace strings.
        2. How does the specific function work:
           Attempts instantiation with non-string, empty, and whitespace-only workload_ids.
        3. Concepts used in the function:
           Defensive type and boundary validation.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Malformed workload_id values.
           After: Raises TypeError or ValueError as appropriate.
        """
        p = (Process(pid="P1", arrival=0, burst=2),)
        with pytest.raises(TypeError, match="workload_id must be a str"):
            Workload(
                workload_id=123,  # type: ignore[arg-type]
                scenario="short_jobs",
                seed=1,
                processes=p,
            )

        with pytest.raises(TypeError, match="workload_id must be a str"):
            Workload(
                workload_id=True,  # type: ignore[arg-type]
                scenario="short_jobs",
                seed=1,
                processes=p,
            )

        with pytest.raises(ValueError, match="workload_id cannot be empty"):
            Workload(workload_id="", scenario="short_jobs", seed=1, processes=p)

        with pytest.raises(ValueError, match="workload_id cannot be empty"):
            Workload(workload_id="   ", scenario="short_jobs", seed=1, processes=p)

    def test_invalid_scenario_raises(self) -> None:
        """
        1. The use of function:
           Verifies rejection of unrecognized scenario identifiers.
        2. How does the specific function work:
           Instantiates Workload with unknown scenario name and asserts ValueError is raised.
        3. Concepts used in the function:
           Scenario catalog validation.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Unrecognized scenario name.
           After: Raises ValueError indicating valid scenario options.
        """
        p = (Process(pid="P1", arrival=0, burst=2),)
        with pytest.raises(ValueError, match="Unknown scenario"):
            Workload(workload_id="w1", scenario="nonexistent", seed=1, processes=p)

    def test_invalid_seed_raises(self) -> None:
        """
        1. The use of function:
           Verifies rejection of non-integer, boolean, or negative seed values.
        2. How does the specific function work:
           Tests negative ints, floats, booleans, and strings for seed attribute.
        3. Concepts used in the function:
           Boundary checking, type checking.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Invalid seed values.
           After: Raises TypeError or ValueError.
        """
        p = (Process(pid="P1", arrival=0, burst=2),)
        with pytest.raises(ValueError, match="seed must be non-negative"):
            Workload(workload_id="w1", scenario="short_jobs", seed=-1, processes=p)

        with pytest.raises(TypeError, match="seed must be an int"):
            Workload(
                workload_id="w1",
                scenario="short_jobs",
                seed=True,  # type: ignore[arg-type]
                processes=p,
            )

        with pytest.raises(TypeError, match="seed must be an int"):
            Workload(
                workload_id="w1",
                scenario="short_jobs",
                seed="42",  # type: ignore[arg-type]
                processes=p,
            )

    def test_invalid_processes_raises(self) -> None:
        """
        1. The use of function:
           Verifies rejection of empty collections, non-Process elements, and duplicate PIDs.
        2. How does the specific function work:
           Tests empty sequence, non-sequence types, non-Process items, and duplicates.
        3. Concepts used in the function:
           validate_processes domain validation invocation.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Malformed process collections.
           After: Raises TypeError or ValueError.
        """
        with pytest.raises(ValueError, match="Process collection cannot be empty"):
            Workload(workload_id="w1", scenario="short_jobs", seed=1, processes=())

        with pytest.raises(TypeError, match="processes must be a tuple/list of Process"):
            Workload(
                workload_id="w1",
                scenario="short_jobs",
                seed=1,
                processes="not_a_seq",  # type: ignore[arg-type]
            )

        with pytest.raises(TypeError, match="Expected Process instance"):
            Workload(
                workload_id="w1",
                scenario="short_jobs",
                seed=1,
                processes=(
                    Process(pid="P1", arrival=0, burst=2),
                    "bad_elem",  # type: ignore[arg-type]
                ),
            )

        with pytest.raises(ValueError, match="Duplicate process ID detected"):
            Workload(
                workload_id="w1",
                scenario="short_jobs",
                seed=1,
                processes=(
                    Process(pid="P1", arrival=0, burst=2),
                    Process(pid="P1", arrival=1, burst=3),
                ),
            )

    def test_immutability_and_hashability(self) -> None:
        """
        1. The use of function:
           Verifies that Workload instances are frozen (immutable) and can be hashed and set-keyed.
        2. How does the specific function work:
           Attempts attribute reassignment, computes hashes, and uses instances in set and dict.
        3. Concepts used in the function:
           Frozen dataclass semantics, __hash__, __eq__.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Two identically configured Workload instances.
           After: Verified FrozenInstanceError on mutation and successful hashing/equality.
        """
        p = (Process(pid="P1", arrival=0, burst=2),)
        wl1 = Workload(workload_id="w1", scenario="short_jobs", seed=1, processes=p)
        wl2 = Workload(workload_id="w1", scenario="short_jobs", seed=1, processes=p)

        with pytest.raises(FrozenInstanceError):
            wl1.seed = 99  # type: ignore[misc]

        assert wl1 == wl2
        assert hash(wl1) == hash(wl2)
        workload_set = {wl1, wl2}
        assert len(workload_set) == 1
        workload_dict = {wl1: "metadata"}
        assert workload_dict[wl2] == "metadata"


class TestSaveWorkloads:
    """
    1. The use of class:
       Tests the save_workloads_to_csv function for disk writing, formatting, and validation.
    2. What the class contains:
       Unit tests verifying header writing for empty lists, parent directory creation, CSV row
       formatting, and rejection of invalid argument types or duplicate workload IDs.
    3. Technologies used and why:
       pytest tmp_path fixtures and standard library csv reader for filesystem verification.
    4. Key design and overall design of the class:
       Unit test fixture asserting that CSV serialization conforms strictly to Section 5 Stage 3.
    """

    def test_save_empty_workloads_writes_header_only(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that saving an empty sequence of workloads produces a valid CSV with header.
        2. How does the specific function work:
           Calls save_workloads_to_csv with an empty list, reads the file, and checks content.
        3. Concepts used in the function:
           CSV header emission, boundary condition handling.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Empty workloads sequence and destination path.
           After: File created with exactly the 7-column header line and no data rows.
        """
        csv_file = tmp_path / "empty.csv"
        save_workloads_to_csv([], csv_file)

        assert csv_file.exists()
        lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert lines[0] == ",".join(WORKLOAD_CSV_HEADER)

    def test_save_auto_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that save_workloads_to_csv creates intermediate parent directories if missing.
        2. How does the specific function work:
           Provides a deeply nested non-existent path and asserts directory creation.
        3. Concepts used in the function:
           Atomic directory tree creation (mkdir parents=True).
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Deeply nested non-existent directory path.
           After: All parent directories created and CSV written successfully.
        """
        deep_file = tmp_path / "nested" / "dir" / "level" / "workloads.csv"
        assert not deep_file.parent.exists()

        p = (Process(pid="P1", arrival=0, burst=4),)
        wl = Workload(workload_id="w1", scenario="bimodal", seed=5, processes=p)
        save_workloads_to_csv([wl], deep_file)

        assert deep_file.exists()
        assert deep_file.is_file()

    def test_save_csv_structure_and_row_count(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies the exact line count, header, and column format of saved CSV workloads.
        2. How does the specific function work:
           Saves two workloads with 2 and 3 processes respectively, asserting 6 lines in CSV.
        3. Concepts used in the function:
           CSV formatting and row expansion per process.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Sequence of 2 workloads with 5 total processes.
           After: CSV file with 1 header line and 5 data rows matching exact schema.
        """
        csv_file = tmp_path / "dataset.csv"
        wl1 = Workload(
            workload_id="wl_A",
            scenario="short_jobs",
            seed=101,
            processes=(
                Process(pid="P1", arrival=0, burst=3, priority=1),
                Process(pid="P2", arrival=2, burst=5, priority=0),
            ),
        )
        wl2 = Workload(
            workload_id="wl_B",
            scenario="long_jobs",
            seed=102,
            processes=(
                Process(pid="P1", arrival=1, burst=15, priority=2),
                Process(pid="P2", arrival=3, burst=20, priority=0),
                Process(pid="P3", arrival=5, burst=25, priority=1),
            ),
        )

        save_workloads_to_csv([wl1, wl2], str(csv_file))

        with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        assert len(reader) == 6
        assert reader[0] == list(WORKLOAD_CSV_HEADER)
        assert reader[1] == ["wl_A", "short_jobs", "101", "P1", "0", "3", "1"]
        assert reader[2] == ["wl_A", "short_jobs", "101", "P2", "2", "5", "0"]
        assert reader[3] == ["wl_B", "long_jobs", "102", "P1", "1", "15", "2"]
        assert reader[4] == ["wl_B", "long_jobs", "102", "P2", "3", "20", "0"]
        assert reader[5] == ["wl_B", "long_jobs", "102", "P3", "5", "25", "1"]

    def test_save_type_and_uniqueness_validation(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies input validation on workloads argument, item types, and workload ID uniqueness.
        2. How does the specific function work:
           Passes invalid argument types and duplicate workload IDs to save_workloads_to_csv.
        3. Concepts used in the function:
           Defensive parameter checks and uniqueness enforcement.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Malformed arguments or duplicate workload IDs.
           After: Raises TypeError or ValueError.
        """
        csv_file = tmp_path / "test.csv"
        with pytest.raises(TypeError, match="workloads must be a Sequence"):
            save_workloads_to_csv("not_a_seq", csv_file)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Element at index 0 is not a Workload"):
            save_workloads_to_csv(["not_a_workload"], csv_file)  # type: ignore[list-item]

        with pytest.raises(TypeError, match="path must be a str or Path"):
            save_workloads_to_csv([], 12345)  # type: ignore[arg-type]

        p = (Process(pid="P1", arrival=0, burst=2),)
        wl_dup1 = Workload(workload_id="wl_dup", scenario="mixed", seed=1, processes=p)
        wl_dup2 = Workload(workload_id="wl_dup", scenario="mixed", seed=2, processes=p)
        with pytest.raises(ValueError, match="Duplicate workload_id 'wl_dup'"):
            save_workloads_to_csv([wl_dup1, wl_dup2], csv_file)


class TestLoadWorkloadsRoundTrip:
    """
    1. The use of class:
       Tests the load_workloads_from_csv function for lossless deserialization and reconstruction.
    2. What the class contains:
       Unit tests covering round-trip fidelity across all 9 canonical scenarios, order preservation,
       empty list loading, and whitespace tolerance.
    3. Technologies used and why:
       pytest parameterization, generator integration, and object equality assertions.
    4. Key design and overall design of the class:
       Unit test fixture verifying that save -> load preserves bit-for-bit data integrity.
    """

    def test_lossless_round_trip_diverse_scenarios(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Tests lossless round-trip serialization and deserialization across all 9 scenarios.
        2. How does the specific function work:
           Generates synthetic workloads for every scenario in ALL_SCENARIOS, saves to CSV,
           reloads from CSV, and asserts exact structural and value equality.
        3. Concepts used in the function:
           End-to-end round trip, scenario coverage, value object equality.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Batch of 9 diverse synthetic Workload objects.
           After: Reconstructed list of Workload objects exactly equal to the original.
        """
        csv_file = tmp_path / "round_trip.csv"
        original_workloads: list[Workload] = []

        for idx, sc in enumerate(ALL_SCENARIOS):
            procs = generate_workload(scenario=sc, n=4, seed=2000 + idx)
            wl = Workload(
                workload_id=f"wl_{sc}_{idx:02d}",
                scenario=sc,
                seed=2000 + idx,
                processes=tuple(procs),
            )
            original_workloads.append(wl)

        save_workloads_to_csv(original_workloads, csv_file)
        loaded_workloads = load_workloads_from_csv(csv_file)

        assert len(loaded_workloads) == len(original_workloads)
        assert loaded_workloads == original_workloads

        for orig, loaded in zip(original_workloads, loaded_workloads, strict=True):
            assert loaded.workload_id == orig.workload_id
            assert loaded.scenario == orig.scenario
            assert loaded.seed == orig.seed
            assert isinstance(loaded.processes, tuple)
            assert loaded.processes == orig.processes

    def test_round_trip_empty_list(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that saving and loading an empty list of workloads succeeds losslessly.
        2. How does the specific function work:
           Saves an empty list, loads the resulting CSV, and asserts an empty list is returned.
        3. Concepts used in the function:
           Empty collection round-trip handling.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Empty workloads list.
           After: Empty workloads list returned from load_workloads_from_csv.
        """
        csv_file = tmp_path / "empty_round_trip.csv"
        save_workloads_to_csv([], csv_file)
        loaded = load_workloads_from_csv(csv_file)
        assert loaded == []

    def test_preserves_workload_ordering(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that workloads appear in the exact encounter order specified in the CSV.
        2. How does the specific function work:
           Saves workloads in order [wl_3, wl_1, wl_2] and asserts that loaded order matches.
        3. Concepts used in the function:
           Insertion-order preservation in CSV parsing.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Specifically ordered list of workloads.
           After: Loaded list preserving identical sequence order.
        """
        csv_file = tmp_path / "ordered.csv"
        p = (Process(pid="P1", arrival=0, burst=5),)
        w3 = Workload(workload_id="wl_003", scenario="staggered", seed=30, processes=p)
        w1 = Workload(workload_id="wl_001", scenario="short_jobs", seed=10, processes=p)
        w2 = Workload(workload_id="wl_002", scenario="convoy", seed=20, processes=p)

        save_workloads_to_csv([w3, w1, w2], csv_file)
        loaded = load_workloads_from_csv(csv_file)

        assert [wl.workload_id for wl in loaded] == ["wl_003", "wl_001", "wl_002"]

    def test_whitespace_tolerance_in_csv_fields(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that CSV files with surrounding whitespace in header and data parse cleanly.
        2. How does the specific function work:
           Manually writes a CSV file with padding whitespace around all values and loads it.
        3. Concepts used in the function:
           Defensive whitespace stripping on CSV tokens.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: CSV text with padded whitespace.
           After: Cleanly parsed Workload object with stripped strings and parsed ints.
        """
        csv_file = tmp_path / "whitespace.csv"
        content = (
            " workload_id , scenario , seed , pid , arrival , burst , priority \n"
            "  wl_space   , short_jobs ,  42  ,  P1  ,    0    ,   5   ,    1     \n"
            "  wl_space   , short_jobs ,  42  ,  P2  ,    2    ,   3   ,    0     \n"
        )
        csv_file.write_text(content, encoding="utf-8")

        loaded = load_workloads_from_csv(csv_file)
        assert len(loaded) == 1
        wl = loaded[0]
        assert wl.workload_id == "wl_space"
        assert wl.scenario == "short_jobs"
        assert wl.seed == 42
        assert len(wl.processes) == 2
        assert wl.processes[0] == Process(pid="P1", arrival=0, burst=5, priority=1)
        assert wl.processes[1] == Process(pid="P2", arrival=2, burst=3, priority=0)


class TestIOErrorHandling:
    """
    1. The use of class:
       Tests robust error detection and diagnostic exceptions in load_workloads_from_csv.
    2. What the class contains:
       Unit tests covering non-existent files, directory paths, 0-byte files, header corruption,
       missing/excess columns, bad integers, domain invariant breaches, and metadata conflict.
    3. Technologies used and why:
       pytest.raises assertions verifying specific exception classes and diagnostic messages.
    4. Key design and overall design of the class:
       Unit test fixture guaranteeing system resilience against corrupt or invalid CSV data.
    """

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that FileNotFoundError is raised when attempting to load a non-existent file.
        2. How does the specific function work:
           Calls load_workloads_from_csv on a path that does not exist.
        3. Concepts used in the function:
           Filesystem existence checking and standard exception raising.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Non-existent file path.
           After: Raises FileNotFoundError with clear message.
        """
        non_existent = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError, match="Workload CSV file not found"):
            load_workloads_from_csv(non_existent)

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that ValueError is raised when path points to a directory rather than a file.
        2. How does the specific function work:
           Calls load_workloads_from_csv targeting an existing directory.
        3. Concepts used in the function:
           Path type validation.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Directory path.
           After: Raises ValueError indicating path is not a regular file.
        """
        with pytest.raises(ValueError, match="Path is not a regular file"):
            load_workloads_from_csv(tmp_path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that an empty (0 bytes) CSV file raises ValueError for missing header.
        2. How does the specific function work:
           Creates an empty file and calls load_workloads_from_csv.
        3. Concepts used in the function:
           Empty file detection and schema validation.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: 0-byte CSV file.
           After: Raises ValueError indicating missing header.
        """
        empty_file = tmp_path / "zero_bytes.csv"
        empty_file.touch()
        with pytest.raises(ValueError, match="is empty; missing required header"):
            load_workloads_from_csv(empty_file)

    def test_missing_header_columns_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that a CSV missing one or more required header columns raises ValueError.
        2. How does the specific function work:
           Writes a header with only 6 columns (omits priority) and calls load_workloads_from_csv.
        3. Concepts used in the function:
           Exact header matching and schema enforcement.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Incomplete CSV header line.
           After: Raises ValueError detailing expected vs received header.
        """
        csv_file = tmp_path / "missing_col.csv"
        csv_file.write_text("workload_id,scenario,seed,pid,arrival,burst\n", encoding="utf-8")
        with pytest.raises(ValueError, match="CSV header mismatch"):
            load_workloads_from_csv(csv_file)

    def test_corrupted_header_names_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that unexpected header names trigger ValueError.
        2. How does the specific function work:
           Writes an arbitrary header row and attempts to load the CSV.
        3. Concepts used in the function:
           Schema header verification.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Incorrect header columns.
           After: Raises ValueError indicating schema mismatch.
        """
        csv_file = tmp_path / "corrupt_header.csv"
        csv_file.write_text("id,scenario,seed,pid,arr,len,prio\n", encoding="utf-8")
        with pytest.raises(ValueError, match="CSV header mismatch"):
            load_workloads_from_csv(csv_file)

    def test_malformed_row_column_count_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that data rows with incorrect column counts raise ValueError.
        2. How does the specific function work:
           Writes a valid header followed by a row containing only 4 columns.
        3. Concepts used in the function:
           Row length validation and line-number diagnostic reporting.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Data row with 4 columns instead of 7.
           After: Raises ValueError indicating line number and expected column count.
        """
        csv_file = tmp_path / "short_row.csv"
        content = (
            f"{','.join(WORKLOAD_CSV_HEADER)}\n"
            "wl_1,short_jobs,42,P1\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed row at line 2"):
            load_workloads_from_csv(csv_file)

    def test_malformed_row_unparseable_integer_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that non-integer values in numeric columns raise ValueError.
        2. How does the specific function work:
           Tests unparseable burst, arrival, seed, and priority values in data rows.
        3. Concepts used in the function:
           Type parsing and diagnostic wrapping.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Data row with string in integer column.
           After: Raises ValueError indicating malformed integer field.
        """
        csv_file = tmp_path / "bad_int.csv"
        content = (
            f"{','.join(WORKLOAD_CSV_HEADER)}\n"
            "wl_1,short_jobs,42,P1,0,not_a_number,0\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed integer field at line 2"):
            load_workloads_from_csv(csv_file)

    def test_empty_string_fields_raise(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that empty string fields in workload_id, scenario, or pid raise ValueError.
        2. How does the specific function work:
           Writes rows with empty workload_id, scenario, or pid strings and attempts loading.
        3. Concepts used in the function:
           Mandatory field presence checking.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Row with empty mandatory string field.
           After: Raises ValueError.
        """
        csv_file = tmp_path / "empty_id.csv"
        content = (
            f"{','.join(WORKLOAD_CSV_HEADER)}\n"
            ",short_jobs,42,P1,0,5,0\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Empty workload_id at line 2"):
            load_workloads_from_csv(csv_file)

    def test_invalid_process_domain_invariants_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that process boundary violations (e.g. burst <= 0) raise ValueError.
        2. How does the specific function work:
           Writes a row with burst=0 and verifies that Process domain invariant triggers ValueError.
        3. Concepts used in the function:
           Domain invariant enforcement during CSV row parsing.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: CSV row with burst=0.
           After: Raises ValueError detailing invalid process attributes.
        """
        csv_file = tmp_path / "zero_burst.csv"
        content = (
            f"{','.join(WORKLOAD_CSV_HEADER)}\n"
            "wl_1,short_jobs,42,P1,0,0,0\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid process attributes at line 2"):
            load_workloads_from_csv(csv_file)

    def test_invalid_scenario_in_csv_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that unrecognized scenario names in CSV data rows raise ValueError.
        2. How does the specific function work:
           Writes a row with an unknown scenario name and calls load_workloads_from_csv.
        3. Concepts used in the function:
           Scenario catalog validation during CSV parsing.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: CSV row with invalid scenario 'quantum_jobs'.
           After: Raises ValueError indicating unknown scenario.
        """
        csv_file = tmp_path / "bad_scenario.csv"
        content = (
            f"{','.join(WORKLOAD_CSV_HEADER)}\n"
            "wl_1,quantum_jobs,42,P1,0,5,0\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid scenario 'quantum_jobs' at line 2"):
            load_workloads_from_csv(csv_file)

    def test_duplicate_pid_in_workload_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that duplicate PIDs within a single workload raise ValueError upon loading.
        2. How does the specific function work:
           Writes two rows with pid='P1' for the same workload and asserts validation failure.
        3. Concepts used in the function:
           validate_processes invariant check on reconstructed Workload.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: CSV containing repeated PID in the same workload.
           After: Raises ValueError indicating duplicate process ID.
        """
        csv_file = tmp_path / "dup_pid.csv"
        content = (
            f"{','.join(WORKLOAD_CSV_HEADER)}\n"
            "wl_1,short_jobs,42,P1,0,5,0\n"
            "wl_1,short_jobs,42,P1,2,3,0\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate process ID detected"):
            load_workloads_from_csv(csv_file)

    def test_inconsistent_metadata_for_workload_id_raises(self, tmp_path: Path) -> None:
        """
        1. The use of function:
           Verifies that rows sharing a workload_id but having conflicting scenario or seed raise.
        2. How does the specific function work:
           Writes two rows for 'wl_1' with differing scenario and seed values.
        3. Concepts used in the function:
           Workload identity consistency check.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: CSV with inconsistent metadata for a single workload_id.
           After: Raises ValueError detailing inconsistent metadata.
        """
        csv_file = tmp_path / "inconsistent_meta.csv"
        content = (
            f"{','.join(WORKLOAD_CSV_HEADER)}\n"
            "wl_1,short_jobs,42,P1,0,5,0\n"
            "wl_1,long_jobs,99,P2,2,15,0\n"
        )
        csv_file.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Inconsistent metadata for workload_id 'wl_1'"):
            load_workloads_from_csv(csv_file)

    def test_invalid_path_type_raises(self) -> None:
        """
        1. The use of function:
           Verifies that load_workloads_from_csv raises TypeError when path is not str or Path.
        2. How does the specific function work:
           Calls load_workloads_from_csv with an integer argument.
        3. Concepts used in the function:
           Type checking on path arguments.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: Integer passed as path.
           After: Raises TypeError.
        """
        with pytest.raises(TypeError, match="path must be a str or Path"):
            load_workloads_from_csv(9999)  # type: ignore[arg-type]


class TestPackageExports:
    """
    1. The use of class:
       Verifies module-level and package-level exports for the workload I/O subsystem.
    2. What the class contains:
       Unit tests asserting that Workload, save_workloads_to_csv, and load_workloads_from_csv
       are exposed at the schedsim.workloads package root and listed in __all__.
    3. Technologies used and why:
       Python introspection and package __all__ membership verification.
    4. Key design and overall design of the class:
       API surface regression test for Stage 3 package interfaces.
    """

    def test_workload_io_package_exports(self) -> None:
        """
        1. The use of function:
           Verifies that Workload, save_workloads_to_csv, and load_workloads_from_csv are exported.
        2. How does the specific function work:
           Inspects schedsim.workloads.__all__ and verifies attribute presence.
        3. Concepts used in the function:
           Package namespace exports and API contract validation.
        4. Places where this function is imported and used:
           Invoked by pytest during workload IO test execution.
        5. Final output of the function, before and after:
           Before: schedsim.workloads module attributes.
           After: Verified that all required I/O symbols are present and callable.
        """
        import schedsim.workloads as sw

        assert "Workload" in sw.__all__
        assert "save_workloads_to_csv" in sw.__all__
        assert "load_workloads_from_csv" in sw.__all__

        assert hasattr(sw, "Workload")
        assert hasattr(sw, "save_workloads_to_csv")
        assert hasattr(sw, "load_workloads_from_csv")

        assert sw.Workload is Workload
        assert sw.save_workloads_to_csv is save_workloads_to_csv
        assert sw.load_workloads_from_csv is load_workloads_from_csv
