"""
1. What this file does:
   Implements the synthetic CPU workload generator producing reproducible, bounded Process
   sequences across canonical scheduling scenarios.
2. Overall design:
   Generates lists of Process objects parameterized by scenario name, job count, and random seed.
   Uses NumPy default_rng for isolated pseudo-random number generation, maps scenario
   specifications to arrival and burst distributions, enforces collection-level domain invariants,
   and guarantees reproducible ordering sorted by (arrival, pid).
3. Elements in the file:
   - generate_workload: Main generator function synthesizing validated Process collections.
4. Final output of the file:
   Exports generate_workload for workload persistence, dataset generation, RL training,
   CLI commands, and testing.
"""

import numpy as np

import config
from schedsim.domain.process import Process, validate_processes
from schedsim.workloads.scenarios import (
    SCENARIO_BIMODAL,
    SCENARIO_CONVOY,
    SCENARIO_EQUAL_BURSTS,
    SCENARIO_HIGH_LOAD,
    SCENARIO_LONG_JOBS,
    SCENARIO_MIXED,
    SCENARIO_SHORT_JOBS,
    SCENARIO_SIMULTANEOUS,
    SCENARIO_STAGGERED,
    get_scenario,
)


def generate_workload(scenario: str, n: int, seed: int) -> list[Process]:
    """
    1. The use of function:
       Generates a deterministically reproducible list of Process objects for a given scenario.
    2. How does the specific function work:
       Validates process count n against [1, config.MAX_PROCESSES] and normalizes scenario using
       get_scenario. Initializes NumPy pseudo-random generator with the provided seed. Synthesizes
       arrival times and burst durations tailored to the scenario rules (e.g., convoy effect with
       P1 burst >= 3 * median(others), simultaneous arrivals at tick 0, equal bursts, staggered
       arrivals with 3..6 tick spacing, bimodal mixtures, and high load). Sorts processes by
       (arrival, pid) and validates collection invariants with validate_processes.
    3. Concepts used in the function:
       Synthetic workload generation, pseudo-random seed determinism, CPU burst distributions,
       arrival time modeling, domain invariant validation, stable sorting.
    4. Places where this function is imported and used:
       Imported by workloads package root, dataset generators (io.py), CLI build-workloads
       command, Decision Tree feature extractors, RL training loops, and test suites.
    5. Final output of the function, before and after:
       Before: Raw parameters (scenario name string, process count n, non-negative integer seed).
       After: Validated list of Process instances sorted by (arrival, pid), or raises
       ValueError / TypeError if parameters violate domain constraints.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"Process count n must be an int, got {type(n).__name__}.")
    if n < 1 or n > config.MAX_PROCESSES:
        raise ValueError(
            f"Process count n must be between 1 and {config.MAX_PROCESSES}, got {n}."
        )

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"Seed must be an int, got {type(seed).__name__}.")
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}.")

    canonical = get_scenario(scenario)
    rng = np.random.default_rng(seed)

    arrivals: list[int]
    bursts: list[int]

    if canonical == SCENARIO_SHORT_JOBS:
        bursts = [int(rng.integers(1, 5)) for _ in range(n)]
        arrivals = [0]
        arr = 0
        for _ in range(1, n):
            arr += int(rng.integers(0, 4))
            arrivals.append(arr)

    elif canonical == SCENARIO_LONG_JOBS:
        bursts = [int(rng.integers(15, 31)) for _ in range(n)]
        arrivals = [0]
        arr = 0
        for _ in range(1, n):
            arr += int(rng.integers(0, 5))
            arrivals.append(arr)

    elif canonical == SCENARIO_MIXED:
        bursts = [
            int(rng.integers(1, 5)) if rng.random() < 0.5 else int(rng.integers(8, 16))
            for _ in range(n)
        ]
        if n >= 2:
            if not any(1 <= b <= 4 for b in bursts):
                bursts[int(rng.integers(0, n))] = int(rng.integers(1, 5))
            if not any(8 <= b <= 15 for b in bursts):
                bursts[int(rng.integers(0, n))] = int(rng.integers(8, 16))
        arrivals = [0]
        arr = 0
        for _ in range(1, n):
            arr += int(rng.integers(0, 4))
            arrivals.append(arr)

    elif canonical == SCENARIO_BIMODAL:
        bursts = [
            int(rng.integers(1, 4)) if rng.random() < 0.8 else int(rng.integers(20, 36))
            for _ in range(n)
        ]
        if n >= 2:
            if not any(1 <= b <= 3 for b in bursts):
                bursts[int(rng.integers(0, n))] = int(rng.integers(1, 4))
            if not any(20 <= b <= 35 for b in bursts):
                bursts[int(rng.integers(0, n))] = int(rng.integers(20, 36))
        arrivals = [0]
        arr = 0
        for _ in range(1, n):
            arr += int(rng.integers(0, 4))
            arrivals.append(arr)

    elif canonical == SCENARIO_SIMULTANEOUS:
        bursts = [int(rng.integers(1, 11)) for _ in range(n)]
        arrivals = [0] * n

    elif canonical == SCENARIO_STAGGERED:
        bursts = [int(rng.integers(2, 9)) for _ in range(n)]
        arrivals = [0]
        arr = 0
        for _ in range(1, n):
            arr += int(rng.integers(3, 7))
            arrivals.append(arr)

    elif canonical == SCENARIO_CONVOY:
        p1_burst = int(rng.integers(20, 36))
        other_bursts = [int(rng.integers(1, 4)) for _ in range(n - 1)]
        if other_bursts:
            median_other = float(np.median(other_bursts))
            if p1_burst < 3 * median_other:
                p1_burst = max(p1_burst, int(np.ceil(3 * median_other)))
        bursts = [p1_burst] + other_bursts
        arrivals = list(range(n))

    elif canonical == SCENARIO_EQUAL_BURSTS:
        b = int(rng.integers(2, 11))
        bursts = [b] * n
        arrivals = [0]
        arr = 0
        for _ in range(1, n):
            arr += int(rng.integers(0, 4))
            arrivals.append(arr)

    elif canonical == SCENARIO_HIGH_LOAD:
        bursts = [int(rng.integers(4, 13)) for _ in range(n)]
        arrivals = [0]
        arr = 0
        for _ in range(1, n):
            arr += int(rng.integers(0, 2))
            arrivals.append(arr)

    else:
        raise ValueError(f"Unhandled scenario '{canonical}'.")

    processes = [
        Process(pid=f"P{i + 1}", arrival=arrivals[i], burst=bursts[i], priority=0)
        for i in range(n)
    ]
    processes.sort(key=lambda p: (p.arrival, p.pid))
    validate_processes(processes)
    return processes


__all__ = [
    "generate_workload",
]
