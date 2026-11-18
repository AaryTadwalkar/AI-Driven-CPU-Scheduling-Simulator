"""
1. What this file does:
   Implements CSV workload serialization and deserialization, including the immutable Workload
   dataclass and helper functions to persist and reload synthetic CPU scheduling datasets.
2. Overall design:
   Provides a robust persistence boundary matching the Section 5 Stage 3 schema specification:
   (workload_id, scenario, seed, pid, arrival, burst, priority). Groups serialized rows into
   validated immutable Workload objects while enforcing domain invariants, directory creation,
   error handling, and workload order preservation.
3. Elements in the file:
   - WORKLOAD_CSV_HEADER: Standardized 7-column header tuple for workload CSV persistence.
   - Workload: Frozen dataclass encapsulating workload_id, scenario, seed, and process tuple.
   - save_workloads_to_csv: Serializes a sequence of Workload instances to CSV.
   - load_workloads_from_csv: Deserializes and validates a CSV file into a list of Workload objects.
4. Final output of the file:
   Exports Workload, save_workloads_to_csv, and load_workloads_from_csv for data generation,
   CLI commands, ML/RL training pipelines, and evaluation suites.
"""

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from schedsim.domain.process import Process, validate_processes
from schedsim.workloads.scenarios import get_scenario

WORKLOAD_CSV_HEADER: tuple[str, ...] = (
    "workload_id",
    "scenario",
    "seed",
    "pid",
    "arrival",
    "burst",
    "priority",
)


@dataclass(frozen=True)
class Workload:
    """
    1. The use of class:
       Encapsulates an immutable synthetic CPU workload specification with associated metadata
       and validated process domain entities.
    2. What the class contains:
       - workload_id: Unique string identifier for the workload.
       - scenario: Canonical scenario string indicating the arrival/burst generation profile.
       - seed: Non-negative integer random seed used for deterministic generation.
       - processes: Immutable tuple of validated Process domain entities.
       - __post_init__: Post-initialization routine enforcing type, scenario, and domain validity.
    3. Technologies used and why:
       - Python dataclass with frozen=True to guarantee immutability, thread-safety, value equality,
         and hashability across simulation runs and dataset caches.
       - Defensive typing and domain validation hooks to ensure data integrity.
    4. Key design and overall design of the class:
       Acts as an aggregate root value object combining workload metadata with its constituent
       process entities, serving as the canonical input unit for simulation and ML evaluation.
    """

    workload_id: str
    scenario: str
    seed: int
    processes: tuple[Process, ...]

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates the integrity, types, and boundaries of workload fields upon instantiation.
        2. How does the specific function work:
           Verifies that workload_id is a non-empty string, canonicalizes the scenario identifier
           via get_scenario, ensures seed is a non-negative integer, coerces processes to a tuple,
           and validates collection invariants using validate_processes.
        3. Concepts used in the function:
           Dataclass post-initialization hooks, defensive argument validation, scenario
           normalization, and collection integrity checks.
        4. Places where this function is imported and used:
           Invoked automatically by the dataclass constructor whenever a Workload is instantiated.
        5. Final output of the function, before and after:
           Before: Unverified raw field values passed to Workload constructor.
           After: Verified, canonicalized Workload state or raises TypeError/ValueError.
        """
        if isinstance(self.workload_id, bool) or not isinstance(self.workload_id, str):
            raise TypeError(
                f"workload_id must be a str, got {type(self.workload_id).__name__}."
            )
        if not self.workload_id.strip():
            raise ValueError("workload_id cannot be empty or whitespace only.")

        canonical_scenario = get_scenario(self.scenario)
        object.__setattr__(self, "scenario", canonical_scenario)

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(
                f"seed must be an int, got {type(self.seed).__name__}."
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}.")

        if self.processes is None:
            raise ValueError("processes collection cannot be None or empty.")
        if not isinstance(self.processes, (tuple, list)):
            raise TypeError(
                f"processes must be a tuple/list of Process, got {type(self.processes).__name__}."
            )

        proc_tuple = tuple(self.processes)
        object.__setattr__(self, "processes", proc_tuple)
        validate_processes(self.processes)


def save_workloads_to_csv(workloads: Sequence[Workload], path: Path | str) -> None:
    """
    1. The use of function:
       Persists a collection of Workload objects to disk in CSV format matching the project schema.
    2. How does the specific function work:
       Validates input types, ensures missing parent directories are created, writes the canonical
       header row, and emits one line per process containing workload metadata and process metrics.
    3. Concepts used in the function:
       CSV serialization, file I/O, path handling, atomic directory creation, input validation.
    4. Places where this function is imported and used:
       Imported by CLI dataset builders, workload generation scripts, and test suites.
    5. Final output of the function, before and after:
       Before: Sequence of Workload objects in memory.
       After: CSV file written to the specified path with header and rows, or raises error.
    """
    if isinstance(workloads, (str, bytes)) or not isinstance(workloads, Sequence):
        raise TypeError(
            f"workloads must be a Sequence of Workload, got {type(workloads).__name__}."
        )
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}.")

    seen_ids: set[str] = set()
    for idx, wl in enumerate(workloads):
        if not isinstance(wl, Workload):
            raise TypeError(
                f"Element at index {idx} is not a Workload instance, got {type(wl).__name__}."
            )
        if wl.workload_id in seen_ids:
            raise ValueError(
                f"Duplicate workload_id '{wl.workload_id}' detected in workloads sequence."
            )
        seen_ids.add(wl.workload_id)

    target_path = Path(path)
    if target_path.parent and str(target_path.parent) != "":
        target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(list(WORKLOAD_CSV_HEADER))
        for wl in workloads:
            for proc in wl.processes:
                writer.writerow([
                    wl.workload_id,
                    wl.scenario,
                    wl.seed,
                    proc.pid,
                    proc.arrival,
                    proc.burst,
                    proc.priority,
                ])


def load_workloads_from_csv(path: Path | str) -> list[Workload]:
    """
    1. The use of function:
       Reads and deserializes a CSV file into a reconstructed list of immutable Workload objects.
    2. How does the specific function work:
       Validates file existence, verifies the exact header specification, groups rows by workload
       identity while preserving order, validates process fields and domain invariants, and returns
       a list of immutable Workload instances.
    3. Concepts used in the function:
       CSV parsing, structured row validation, grouping with encounter order preservation, domain
       invariant checking, defensive error handling.
    4. Places where this function is imported and used:
       Imported by ML dataset loaders, CLI benchmark routines, RL trainers, and test suites.
    5. Final output of the function, before and after:
       Before: Persisted CSV file on disk.
       After: Reconstructed list of Workload instances, or raises FileNotFoundError / ValueError.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError(f"path must be a str or Path, got {type(path).__name__}.")

    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Workload CSV file not found: '{target_path}'.")
    if not target_path.is_file():
        raise ValueError(f"Path is not a regular file: '{target_path}'.")

    with open(target_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(
                f"Workload CSV file '{target_path}' is empty; missing required header."
            )

        cleaned_header = tuple(col.strip() for col in header)
        if cleaned_header != WORKLOAD_CSV_HEADER:
            raise ValueError(
                f"CSV header mismatch in '{target_path}'. "
                f"Expected {list(WORKLOAD_CSV_HEADER)}, got {list(cleaned_header)}."
            )

        grouped_processes: dict[tuple[str, str, int], list[Process]] = {}
        seen_workload_meta: dict[str, tuple[str, int]] = {}

        for line_no, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(WORKLOAD_CSV_HEADER):
                raise ValueError(
                    f"Malformed row at line {line_no} in '{target_path}': "
                    f"expected {len(WORKLOAD_CSV_HEADER)} columns, got {len(row)}."
                )

            wl_id, sc_raw, seed_raw, pid, arr_raw, burst_raw, prio_raw = (
                field.strip() for field in row
            )

            if not wl_id:
                raise ValueError(
                    f"Empty workload_id at line {line_no} in '{target_path}'."
                )
            if not sc_raw:
                raise ValueError(
                    f"Empty scenario at line {line_no} in '{target_path}'."
                )
            if not pid:
                raise ValueError(
                    f"Empty pid at line {line_no} in '{target_path}'."
                )

            try:
                seed_val = int(seed_raw)
                arr_val = int(arr_raw)
                burst_val = int(burst_raw)
                prio_val = int(prio_raw)
            except ValueError as err:
                raise ValueError(
                    f"Malformed integer field at line {line_no} in '{target_path}': {err}."
                ) from err

            try:
                canonical_sc = get_scenario(sc_raw)
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Invalid scenario '{sc_raw}' at line {line_no} in '{target_path}': {err}."
                ) from err

            if wl_id in seen_workload_meta and seen_workload_meta[wl_id] != (
                canonical_sc,
                seed_val,
            ):
                raise ValueError(
                    f"Inconsistent metadata for workload_id '{wl_id}' at line {line_no} in "
                    f"'{target_path}': previously saw {seen_workload_meta[wl_id]}, "
                    f"now got ({canonical_sc}, {seed_val})."
                )
            seen_workload_meta[wl_id] = (canonical_sc, seed_val)

            try:
                proc = Process(
                    pid=pid,
                    arrival=arr_val,
                    burst=burst_val,
                    priority=prio_val,
                )
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"Invalid process attributes at line {line_no} in '{target_path}': {err}."
                ) from err

            key = (wl_id, canonical_sc, seed_val)
            if key not in grouped_processes:
                grouped_processes[key] = []
            grouped_processes[key].append(proc)

    result: list[Workload] = []
    for (wl_id, sc, seed_val), procs in grouped_processes.items():
        try:
            wl = Workload(
                workload_id=wl_id,
                scenario=sc,
                seed=seed_val,
                processes=tuple(procs),
            )
        except (ValueError, TypeError) as err:
            raise ValueError(
                f"Failed to instantiate Workload '{wl_id}' from '{target_path}': {err}."
            ) from err
        result.append(wl)

    return result
