"""
1. What this file does:
   Implements the comprehensive Stage 3 Acceptance Test Suite and performance benchmark for the
   Workload Generator and Data Layer of the CPU Scheduling Simulator.
2. Overall design:
   Validates all Section 5 Stage 3 acceptance criteria:
   - Seed determinism (identical seed yields identical workloads)
   - Seed divergence (different seeds produce distinct workloads)
   - Domain invariants (process count bounds, positive bursts, non-negative arrivals, unique PIDs)
   - Specific invariant properties for all 9 canonical workload scenarios
   - Lossless serialization and deserialization round-trip via CSV
   - Dataset split partition isolation (Train, Val, Test purity without overlap)
   - Execution performance benchmark: 6,000 workloads generated and persisted in < 60 seconds.
3. Elements in the file:
   - test_workload_seed_determinism_acceptance: Verifies deterministic repeatability across seeds.
   - test_workload_seed_divergence_acceptance: Verifies distinct outputs for divergent seeds.
   - test_workload_bounds_respected_acceptance: Verifies domain invariants across random workloads.
   - test_all_scenario_properties_acceptance: Verifies invariants for all 9 canonical scenarios.
   - test_csv_round_trip_lossless_acceptance: Verifies lossless CSV serialization and re-parsing.
   - test_split_seed_isolation_acceptance: Verifies complete separation between dataset splits.
   - test_build_workloads_speed_benchmark: Benchmarks 6,000 workload generation under 60 seconds.
4. Final output of the file:
   Pytest acceptance suite certifying Stage 3 compliance with implementation plan specifications.
"""

import time
from pathlib import Path

import numpy as np

import config
from cli import main
from schedsim.workloads.generator import generate_workload
from schedsim.workloads.io import load_workloads_from_csv, save_workloads_to_csv
from schedsim.workloads.scenarios import (
    ALL_SCENARIOS,
    SCENARIO_BIMODAL,
    SCENARIO_CONVOY,
    SCENARIO_EQUAL_BURSTS,
    SCENARIO_HIGH_LOAD,
    SCENARIO_LONG_JOBS,
    SCENARIO_MIXED,
    SCENARIO_SHORT_JOBS,
    SCENARIO_SIMULTANEOUS,
    SCENARIO_STAGGERED,
)
from schedsim.workloads.splits import (
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    TEST_SEED_RANGE,
    TRAIN_SEED_RANGE,
    VAL_SEED_RANGE,
    generate_split_workloads,
    get_split_for_seed,
)


def test_workload_seed_determinism_acceptance() -> None:
    """
    1. The use of function:
       Validates strict seed determinism for workload generation across all 9 canonical scenarios.
    2. How does the specific function work:
       Iterates over all canonical scenarios and multiple seeds, generating workloads twice with
       identical parameters. Asserts that process counts, PIDs, arrivals, bursts, and priorities
       match exactly between runs.
    3. Concepts used in the function:
       Deterministic pseudo-randomness, reproducibility validation, invariant assertion.
    4. Places where this function is imported and used:
       Executed by pytest as part of Stage 3 acceptance verification.
    5. Final output of the function, before and after:
       Before: Unverified repeatability of workload generator across seeds.
       After: Asserts identical process outputs for identical generation inputs.
    """
    seeds = [0, 42, 1234, 99999]
    for scenario in ALL_SCENARIOS:
        for seed in seeds:
            w1 = generate_workload(scenario=scenario, n=6, seed=seed)
            w2 = generate_workload(scenario=scenario, n=6, seed=seed)

            assert len(w1) == len(w2)
            for p1, p2 in zip(w1, w2):
                assert p1.pid == p2.pid
                assert p1.arrival == p2.arrival
                assert p1.burst == p2.burst
                assert p1.priority == p2.priority


def test_workload_seed_divergence_acceptance() -> None:
    """
    1. The use of function:
       Validates that distinct random seeds produce divergent process attributes.
    2. How does the specific function work:
       Generates workloads for the same scenario and count using two distinct seeds (10 vs 20),
       and verifies that at least one arrival or burst attribute differs between the two sets.
    3. Concepts used in the function:
       Seed divergence, non-trivial randomness, pseudo-random generator state difference.
    4. Places where this function is imported and used:
       Executed by pytest as part of Stage 3 acceptance verification.
    5. Final output of the function, before and after:
       Before: Unverified diversity of pseudo-random generation across different seeds.
       After: Asserts distinct seed runs produce divergent workload parameters.
    """
    for scenario in ALL_SCENARIOS:
        w1 = generate_workload(scenario=scenario, n=8, seed=101)
        w2 = generate_workload(scenario=scenario, n=8, seed=202)

        # Equal bursts scenario has identical bursts within a workload, but different across seeds
        attrs1 = [(p.arrival, p.burst) for p in w1]
        attrs2 = [(p.arrival, p.burst) for p in w2]
        assert (
            attrs1 != attrs2
        ), f"Scenario {scenario} produced identical workloads on seeds 101/202."


def test_workload_bounds_respected_acceptance() -> None:
    """
    1. The use of function:
       Validates core domain bounds and invariants across a wide randomized sample of workloads.
    2. How does the specific function work:
       Generates 200 workloads with varying process counts (1 to MAX_PROCESSES) and seeds.
       Asserts that $1 \\le n \\le MAX_PROCESSES$, arrival $\\ge 0$, burst $\\ge 1$,
       PIDs are unique strings, and arrivals are monotonically sorted.
    3. Concepts used in the function:
       Domain invariant checking, boundary fuzzing, structural validation.
    4. Places where this function is imported and used:
       Executed by pytest as part of Stage 3 acceptance verification.
    5. Final output of the function, before and after:
       Before: Unverified domain integrity across diverse synthetic parameters.
       After: Asserts all generated workloads strictly respect domain invariants.
    """
    for i in range(200):
        scenario = ALL_SCENARIOS[i % len(ALL_SCENARIOS)]
        n = 1 + (i % config.MAX_PROCESSES)
        seed = 1000 + i
        procs = generate_workload(scenario=scenario, n=n, seed=seed)

        assert 1 <= len(procs) <= config.MAX_PROCESSES
        pids = [p.pid for p in procs]
        assert len(pids) == len(set(pids)), f"Duplicate PIDs found in {pids}."

        for idx, p in enumerate(procs):
            assert p.arrival >= 0
            assert p.burst >= 1
            assert p.priority == 0
            if idx > 0:
                assert p.arrival >= procs[idx - 1].arrival


def test_all_scenario_properties_acceptance() -> None:
    """
    1. The use of function:
       Validates specific characteristic properties for each of the 9 canonical workload scenarios.
    2. How does the specific function work:
       Samples multiple seeds for each scenario and verifies:
       - convoy: P1 burst >= 3 * median(other bursts)
       - simultaneous: all arrivals == 0
       - equal_bursts: all bursts identical
       - staggered: arrivals strictly monotonic with delta >= 3
       - bimodal: contains short (<= 3) and long (>= 20) jobs
       - mixed: contains short (<= 4) and medium/long (>= 8) jobs
       - high_load: tightly spaced arrivals (delta <= 1)
       - short_jobs: all bursts in 1..4
       - long_jobs: all bursts in 15..30
    3. Concepts used in the function:
       Scenario invariant testing, statistical property assertions, domain verification.
    4. Places where this function is imported and used:
       Executed by pytest as part of Stage 3 acceptance verification.
    5. Final output of the function, before and after:
       Before: Unverified compliance of scenario generator with Section 3.4 specifications.
       After: Asserts all 9 scenario generators satisfy their canonical domain definitions.
    """
    sample_seeds = [10, 55, 99, 1001, 7777]

    for seed in sample_seeds:
        # 1. Convoy scenario
        convoy = generate_workload(scenario=SCENARIO_CONVOY, n=6, seed=seed)
        p1_burst = convoy[0].burst
        others_median = float(np.median([p.burst for p in convoy[1:]]))
        assert p1_burst >= 3.0 * others_median

        # 2. Simultaneous scenario
        simul = generate_workload(scenario=SCENARIO_SIMULTANEOUS, n=6, seed=seed)
        assert all(p.arrival == 0 for p in simul)

        # 3. Equal bursts scenario
        eq = generate_workload(scenario=SCENARIO_EQUAL_BURSTS, n=6, seed=seed)
        first_burst = eq[0].burst
        assert all(p.burst == first_burst for p in eq)

        # 4. Staggered scenario
        stag = generate_workload(scenario=SCENARIO_STAGGERED, n=6, seed=seed)
        for i in range(1, len(stag)):
            assert stag[i].arrival - stag[i - 1].arrival >= 3

        # 5. Bimodal scenario
        bimodal = generate_workload(scenario=SCENARIO_BIMODAL, n=6, seed=seed)
        assert any(p.burst <= 3 for p in bimodal)
        assert any(p.burst >= 20 for p in bimodal)

        # 6. Mixed scenario
        mixed = generate_workload(scenario=SCENARIO_MIXED, n=6, seed=seed)
        assert any(p.burst <= 4 for p in mixed)
        assert any(p.burst >= 8 for p in mixed)

        # 7. High load scenario
        high_load = generate_workload(scenario=SCENARIO_HIGH_LOAD, n=6, seed=seed)
        for i in range(1, len(high_load)):
            assert high_load[i].arrival - high_load[i - 1].arrival <= 1

        # 8. Short jobs scenario
        short_jobs = generate_workload(scenario=SCENARIO_SHORT_JOBS, n=6, seed=seed)
        assert all(1 <= p.burst <= 4 for p in short_jobs)

        # 9. Long jobs scenario
        long_jobs = generate_workload(scenario=SCENARIO_LONG_JOBS, n=6, seed=seed)
        assert all(15 <= p.burst <= 30 for p in long_jobs)


def test_csv_round_trip_lossless_acceptance(tmp_path: Path) -> None:
    """
    1. The use of function:
       Validates completely lossless persistence and restoration of workloads via CSV.
    2. How does the specific function work:
       Generates 27 multi-scenario workloads across train and validation splits, saves them to
       a CSV file using `save_workloads_to_csv()`, reloads them using `load_workloads_from_csv()`,
       and verifies that every workload ID, scenario, seed, process count, PID, arrival, burst,
       and priority match the original in-memory structures exactly.
    3. Concepts used in the function:
       Lossless data serialization, round-trip fidelity testing, schema compliance.
    4. Places where this function is imported and used:
       Executed by pytest as part of Stage 3 acceptance verification.
    5. Final output of the function, before and after:
       Before: Unverified end-to-end serialization fidelity across diverse workloads.
       After: Asserts exact structural equivalence between original and reloaded workloads.
    """
    csv_file = tmp_path / "acceptance_round_trip.csv"
    original = generate_split_workloads(split=SPLIT_TRAIN, n_workloads=18)
    original.extend(generate_split_workloads(split=SPLIT_VAL, n_workloads=9))

    save_workloads_to_csv(original, path=csv_file)
    reloaded = load_workloads_from_csv(csv_file)

    assert len(original) == len(reloaded)
    for orig, rel in zip(original, reloaded):
        assert orig.workload_id == rel.workload_id
        assert orig.scenario == rel.scenario
        assert orig.seed == rel.seed
        assert len(orig.processes) == len(rel.processes)
        for p_orig, p_rel in zip(orig.processes, rel.processes):
            assert p_orig.pid == p_rel.pid
            assert p_orig.arrival == p_rel.arrival
            assert p_orig.burst == p_rel.burst
            assert p_orig.priority == p_rel.priority


def test_split_seed_isolation_acceptance() -> None:
    """
    1. The use of function:
       Validates strict seed partition isolation between Train, Validation, and Test splits.
    2. How does the specific function work:
       Verifies mathematical non-overlap between defined ranges TRAIN_SEED_RANGE, VAL_SEED_RANGE,
       and TEST_SEED_RANGE. Tests `get_split_for_seed` boundary conditions and asserts out-of-range
       seeds raise ValueError.
    3. Concepts used in the function:
       Dataset contamination prevention, seed partition isolation, boundary value analysis.
    4. Places where this function is imported and used:
       Executed by pytest as part of Stage 3 acceptance verification.
    5. Final output of the function, before and after:
       Before: Unverified partition boundaries and leakage risk.
       After: Asserts zero seed leakage between Train, Validation, and Test splits.
    """
    train_min, train_max = TRAIN_SEED_RANGE
    val_min, val_max = VAL_SEED_RANGE
    test_min, test_max = TEST_SEED_RANGE

    # Verify ranges are non-overlapping
    assert train_max < val_min, "Train and Validation seed ranges overlap!"
    assert val_max < test_min, "Validation and Test seed ranges overlap!"

    # Verify boundary points map correctly
    assert get_split_for_seed(train_min) == SPLIT_TRAIN
    assert get_split_for_seed(train_max) == SPLIT_TRAIN
    assert get_split_for_seed(val_min) == SPLIT_VAL
    assert get_split_for_seed(val_max) == SPLIT_VAL
    assert get_split_for_seed(test_min) == SPLIT_TEST
    assert get_split_for_seed(test_max) == SPLIT_TEST

    # Verify gaps raise ValueError
    gap1 = train_max + 1
    if gap1 < val_min:
        try:
            get_split_for_seed(gap1)
            assert False, f"Expected ValueError for unallocated seed {gap1}."
        except ValueError:
            pass

    gap2 = val_max + 1
    if gap2 < test_min:
        try:
            get_split_for_seed(gap2)
            assert False, f"Expected ValueError for unallocated seed {gap2}."
        except ValueError:
            pass


def test_build_workloads_speed_benchmark(tmp_path: Path) -> None:
    """
    1. The use of function:
       Benchmarks execution speed for generating and persisting 6,000 synthetic workloads.
    2. How does the specific function work:
       Invokes `main(['build-workloads', '--n', '6000', '--output', ...])`, measures total elapsed
       execution duration, asserts return code 0, verifies file row count, and enforces that
       the operation completes well under the 60-second acceptance ceiling.
    3. Concepts used in the function:
       Performance benchmarking, throughput evaluation, latency bounds assertion.
    4. Places where this function is imported and used:
       Executed by pytest as part of Stage 3 acceptance verification.
    5. Final output of the function, before and after:
       Before: Unbenchmarked dataset generation pipeline.
       After: Asserts 6,000 workloads generate and write to CSV in under 60 seconds.
    """
    out_csv = tmp_path / "benchmark_workloads.csv"
    t0 = time.perf_counter()
    exit_code = main(
        ["build-workloads", "--n", "6000", "--output", str(out_csv), "--split", "train"]
    )
    elapsed = time.perf_counter() - t0

    assert exit_code == 0
    assert out_csv.exists()
    assert elapsed < 60.0, f"Generation exceeded 60s limit: {elapsed:.2f}s."

    # Validate output has 6,000 distinct workload entries
    workloads = load_workloads_from_csv(out_csv)
    assert len(workloads) == 6000
