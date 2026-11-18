"""
1. What this file does:
   Defines unit tests for the synthetic workload generator and scenario catalog subsystems
   (scenarios.py and generator.py).
2. Overall design:
   Validates seed determinism, seed divergence, input parameter boundary enforcement, scenario
   lookup normalization, domain invariant compliance, and mathematical distribution properties
   across all 9 canonical scheduling scenarios.
3. Elements in the file:
   - TestSeedDeterminism: Tests exact repeatability and seed-based divergence.
   - TestParameterBoundaries: Tests parameter bounds for n, seed, and type validation.
   - TestScenarioValidationAndLookup: Tests scenario normalization, casing, and error handling.
   - TestScenarioPropertyInvariants: Tests domain and statistical properties for all 9 scenarios.
   - TestDomainComplianceAndExports: Tests domain invariants, PID naming, and package exports.
4. Final output of the file:
   Provides comprehensive unit test suite executed by pytest during test discovery and CI runs.
"""

import numpy as np
import pytest

import config
from schedsim.domain.process import Process, validate_processes
from schedsim.workloads import (
    ALL_SCENARIOS,
    generate_workload,
    get_scenario,
)
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
)


class TestSeedDeterminism:
    """
    1. The use of class:
       Tests pseudo-random generator determinism and divergence across synthetic workloads.
    2. What the class contains:
       Unit tests verifying that identical seeds produce identical Process sequences and different
       seeds produce divergent workloads.
    3. Technologies used and why:
       pytest assertions and NumPy BitGenerator reproducibility to ensure reproducible benchmarks.
    4. Key design and overall design of the class:
       Compares process attributes (pid, arrival, burst, priority) across identical and differing
       seeds across multiple scenarios and job counts.
    """

    def test_identical_seed_produces_identical_workload(self) -> None:
        """
        1. The use of function:
           Verifies that identical (scenario, n, seed) inputs yield identical Process sequences.
        2. How does the specific function work:
           Invokes generate_workload twice with identical parameters for each scenario and verifies
           that every process attribute matches exactly in sequence.
        3. Concepts used in the function:
           Deterministic pseudo-random number generation, sequence equivalence.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified generator repeatability across identical seeds.
           After: Verified bitwise identical Process sequences across all scenarios.
        """
        for scenario in ALL_SCENARIOS:
            for n in (1, 5, config.MAX_PROCESSES):
                for seed in (0, 42, 99999):
                    workload_a = generate_workload(scenario, n, seed)
                    workload_b = generate_workload(scenario, n, seed)
                    assert len(workload_a) == len(workload_b) == n
                    for p_a, p_b in zip(workload_a, workload_b, strict=True):
                        assert p_a.pid == p_b.pid
                        assert p_a.arrival == p_b.arrival
                        assert p_a.burst == p_b.burst
                        assert p_a.priority == p_b.priority

    def test_different_seeds_produce_divergent_workloads(self) -> None:
        """
        1. The use of function:
           Verifies that differing seeds produce distinct Process sequences for random scenarios.
        2. How does the specific function work:
           Generates workloads with different seeds for scenarios with stochastic components and
           asserts that at least one burst or arrival attribute differs.
        3. Concepts used in the function:
           Seed divergence, stochastic sampling verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified seed sensitivity for stochastic scenarios.
           After: Confirmed divergence across distinct seeds.
        """
        for scenario in (
            SCENARIO_SHORT_JOBS,
            SCENARIO_LONG_JOBS,
            SCENARIO_MIXED,
            SCENARIO_BIMODAL,
            SCENARIO_STAGGERED,
            SCENARIO_CONVOY,
            SCENARIO_HIGH_LOAD,
        ):
            workload_a = generate_workload(scenario, n=8, seed=123)
            workload_b = generate_workload(scenario, n=8, seed=456)
            a_bursts = [p.burst for p in workload_a]
            b_bursts = [p.burst for p in workload_b]
            a_arrivals = [p.arrival for p in workload_a]
            b_arrivals = [p.arrival for p in workload_b]
            assert a_bursts != b_bursts or a_arrivals != b_arrivals


class TestParameterBoundaries:
    """
    1. The use of class:
       Validates input parameter boundaries and defensive type checking in generate_workload.
    2. What the class contains:
       Unit tests covering valid boundary values (n=1, n=MAX_PROCESSES, seed=0) and invalid values
       raising ValueError or TypeError.
    3. Technologies used and why:
       pytest.raises to verify boundary conditions and robust error handling.
    4. Key design and overall design of the class:
       Groups boundary tests by parameter: process count n, random seed, and typing invariants.
    """

    def test_boundary_process_counts_valid(self) -> None:
        """
        1. The use of function:
           Tests that extreme valid process counts (n=1 and n=MAX_PROCESSES) succeed.
        2. How does the specific function work:
           Calls generate_workload with n=1 and n=config.MAX_PROCESSES across canonical scenarios,
           verifying that workloads of exact length are returned.
        3. Concepts used in the function:
           Boundary value analysis, capacity limits.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unchecked edge process counts.
           After: Verified generation at minimum and maximum process bounds.
        """
        for n in (1, config.MAX_PROCESSES):
            workload = generate_workload(SCENARIO_SHORT_JOBS, n, seed=1)
            assert len(workload) == n
            assert all(isinstance(p, Process) for p in workload)

    def test_invalid_process_counts_raise_value_error(self) -> None:
        """
        1. The use of function:
           Verifies that process counts outside [1, MAX_PROCESSES] raise ValueError.
        2. How does the specific function work:
           Tests n=0, n=-1, n=MAX_PROCESSES+1, and n=100, ensuring ValueError is raised with
           descriptive diagnostics.
        3. Concepts used in the function:
           Defensive parameter validation, boundary rejection.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Potential out-of-bound process counts.
           After: Verified ValueError raised for all out-of-bound process counts.
        """
        for invalid_n in (0, -1, config.MAX_PROCESSES + 1, 100):
            with pytest.raises(ValueError, match="Process count n must be between 1 and"):
                generate_workload(SCENARIO_SHORT_JOBS, invalid_n, seed=1)

    def test_invalid_process_count_types_raise_type_error(self) -> None:
        """
        1. The use of function:
           Verifies that non-integer process count types raise TypeError.
        2. How does the specific function work:
           Passes str, float, None, and bool values for n and verifies TypeError is raised.
        3. Concepts used in the function:
           Strict type checking, boolean-as-int rejection.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unchecked non-integer process count types.
           After: Verified TypeError raised for invalid process count types.
        """
        for invalid_n in ("5", 3.14, None, True, False):
            with pytest.raises(TypeError, match="Process count n must be an int"):
                generate_workload(
                    SCENARIO_SHORT_JOBS, invalid_n, seed=1  # type: ignore[arg-type]
                )

    def test_seed_zero_is_valid(self) -> None:
        """
        1. The use of function:
           Tests that seed=0 is accepted as a valid non-negative seed.
        2. How does the specific function work:
           Calls generate_workload with seed=0 and checks that a valid workload is returned.
        3. Concepts used in the function:
           Boundary value analysis for random seeds.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unchecked boundary seed value 0.
           After: Confirmed seed=0 generates valid workloads successfully.
        """
        workload = generate_workload(SCENARIO_EQUAL_BURSTS, n=4, seed=0)
        assert len(workload) == 4

    def test_negative_seed_raises_value_error(self) -> None:
        """
        1. The use of function:
           Verifies that negative seed values raise ValueError.
        2. How does the specific function work:
           Passes seed=-1 and seed=-100 and verifies ValueError is raised.
        3. Concepts used in the function:
           Defensive parameter validation, domain constraints.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Potential negative seed values.
           After: Verified ValueError raised for negative seeds.
        """
        for invalid_seed in (-1, -100):
            with pytest.raises(ValueError, match="Seed must be non-negative"):
                generate_workload(SCENARIO_SHORT_JOBS, n=4, seed=invalid_seed)

    def test_invalid_seed_types_raise_type_error(self) -> None:
        """
        1. The use of function:
           Verifies that non-integer seed types raise TypeError.
        2. How does the specific function work:
           Passes str, float, None, and bool values for seed and verifies TypeError is raised.
        3. Concepts used in the function:
           Strict type checking, boolean-as-int rejection.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unchecked non-integer seed types.
           After: Verified TypeError raised for invalid seed types.
        """
        for invalid_seed in ("42", 2.718, None, True, False):
            with pytest.raises(TypeError, match="Seed must be an int"):
                generate_workload(
                    SCENARIO_SHORT_JOBS, n=4, seed=invalid_seed  # type: ignore[arg-type]
                )


class TestScenarioValidationAndLookup:
    """
    1. The use of class:
       Tests scenario normalization, case-insensitivity, and error reporting in get_scenario.
    2. What the class contains:
       Unit tests validating canonical scenario constants, whitespace trimming, casing variations,
       and error raising for unknown or non-string scenario names.
    3. Technologies used and why:
       pytest assertions and exception inspection to verify string validation contracts.
    4. Key design and overall design of the class:
       Validates lookup helper get_scenario both directly and via generate_workload.
    """

    def test_all_canonical_scenarios_valid(self) -> None:
        """
        1. The use of function:
           Verifies that all 9 canonical scenario identifiers pass get_scenario validation.
        2. How does the specific function work:
           Iterates over ALL_SCENARIOS and asserts that get_scenario returns the identical string.
        3. Concepts used in the function:
           Identity mapping, canonical enumeration completeness.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified canonical scenarios in get_scenario.
           After: Confirmed each scenario in ALL_SCENARIOS resolves identically.
        """
        for scenario in ALL_SCENARIOS:
            assert get_scenario(scenario) == scenario

    def test_scenario_case_insensitivity_and_whitespace(self) -> None:
        """
        1. The use of function:
           Verifies case-insensitive lookup and whitespace trimming for scenario names.
        2. How does the specific function work:
           Tests uppercase, mixed case, and padded scenario strings in get_scenario and
           generate_workload.
        3. Concepts used in the function:
           String normalization, robust input handling.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified scenario string normalization.
           After: Confirmed normalization succeeds and returns canonical lowercase strings.
        """
        test_cases = [
            ("  SHORT_JOBS  ", SCENARIO_SHORT_JOBS),
            ("Long_Jobs", SCENARIO_LONG_JOBS),
            ("  mixed\t", SCENARIO_MIXED),
            ("BIMODAL", SCENARIO_BIMODAL),
            (" Simultaneous ", SCENARIO_SIMULTANEOUS),
            ("STAGGERED", SCENARIO_STAGGERED),
            ("Convoy", SCENARIO_CONVOY),
            ("equal_bursts", SCENARIO_EQUAL_BURSTS),
            ("  HIGH_LOAD  ", SCENARIO_HIGH_LOAD),
        ]
        for raw, expected in test_cases:
            assert get_scenario(raw) == expected
            workload = generate_workload(raw, n=3, seed=1)
            assert len(workload) == 3

    def test_unknown_scenario_raises_value_error(self) -> None:
        """
        1. The use of function:
           Verifies that unknown scenario names raise ValueError with valid scenario listings.
        2. How does the specific function work:
           Passes unrecognized strings to get_scenario and generate_workload, asserting ValueError
           contains the offending name.
        3. Concepts used in the function:
           Defensive parameter validation, actionable error messages.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Potential unhandled scenario strings.
           After: Verified ValueError raised for unknown scenario strings.
        """
        for unknown in ("unknown_scenario", "batch", "fcfs_best", "", "   "):
            with pytest.raises(ValueError, match="Unknown scenario"):
                get_scenario(unknown)
            with pytest.raises(ValueError, match="Unknown scenario"):
                generate_workload(unknown, n=3, seed=1)

    def test_non_string_scenario_raises_type_error(self) -> None:
        """
        1. The use of function:
           Verifies that non-string scenario names raise TypeError.
        2. How does the specific function work:
           Passes integer, None, list, and boolean types to get_scenario and generate_workload.
        3. Concepts used in the function:
           Strict type checking, type safety.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unchecked non-string scenario argument types.
           After: Verified TypeError raised for invalid scenario types.
        """
        for invalid_type in (123, None, ["short_jobs"], True, False):
            with pytest.raises(TypeError, match="Scenario name must be a str"):
                get_scenario(invalid_type)  # type: ignore[arg-type]
            with pytest.raises(TypeError, match="Scenario name must be a str"):
                generate_workload(invalid_type, n=3, seed=1)  # type: ignore[arg-type]

    def test_all_scenarios_tuple_is_immutable(self) -> None:
        """
        1. The use of function:
           Verifies that ALL_SCENARIOS is an immutable tuple containing exactly 9 canonical names.
        2. How does the specific function work:
           Checks tuple instance type, length 9, and unique scenario string elements.
        3. Concepts used in the function:
           Immutability, catalog integrity.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified catalog container structure.
           After: Confirmed ALL_SCENARIOS is an immutable tuple with 9 unique scenario identifiers.
        """
        assert isinstance(ALL_SCENARIOS, tuple)
        assert len(ALL_SCENARIOS) == 9
        assert len(set(ALL_SCENARIOS)) == 9


class TestScenarioPropertyInvariants:
    """
    1. The use of class:
       Validates mathematical and domain invariants for each of the 9 canonical scenarios.
    2. What the class contains:
       Unit tests validating burst bounds, arrival intervals, convoy median ratios, bimodal/mixed
       bimodality, simultaneous arrival synchrony, and high-load saturation properties.
    3. Technologies used and why:
       pytest, NumPy statistical functions (median) to assert invariant preservation across seeds.
    4. Key design and overall design of the class:
       Tests each scenario across multiple random seeds and process counts to guarantee invariants.
    """

    def test_short_jobs_property_invariants(self) -> None:
        """
        1. The use of function:
           Verifies short_jobs scenario enforces burst bounds in [1, 4] and non-decreasing arrivals.
        2. How does the specific function work:
           Generates short_jobs workloads across seeds and asserts all bursts are in [1, 4],
           arrivals are non-negative, and arrivals are sorted non-decreasingly starting at 0.
        3. Concepts used in the function:
           Range assertions, temporal ordering verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified short_jobs distribution properties.
           After: Verified all processes satisfy burst and arrival invariants.
        """
        for seed in range(10):
            workload = generate_workload(SCENARIO_SHORT_JOBS, n=10, seed=seed)
            assert all(1 <= p.burst <= 4 for p in workload)
            arrivals = [p.arrival for p in workload]
            assert arrivals[0] == 0
            assert all(arrivals[i] <= arrivals[i + 1] for i in range(len(arrivals) - 1))

    def test_long_jobs_property_invariants(self) -> None:
        """
        1. The use of function:
           Verifies long_jobs scenario enforces bursts in [15, 30] and non-decreasing arrivals.
        2. How does the specific function work:
           Generates long_jobs workloads across seeds and asserts all bursts are in [15, 30]
           and arrivals start at 0 and are non-decreasing.
        3. Concepts used in the function:
           Range assertions, temporal ordering verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified long_jobs distribution properties.
           After: Verified all processes satisfy long burst invariants.
        """
        for seed in range(10):
            workload = generate_workload(SCENARIO_LONG_JOBS, n=8, seed=seed)
            assert all(15 <= p.burst <= 30 for p in workload)
            arrivals = [p.arrival for p in workload]
            assert arrivals[0] == 0
            assert all(arrivals[i] <= arrivals[i + 1] for i in range(len(arrivals) - 1))

    def test_mixed_property_invariants(self) -> None:
        """
        1. The use of function:
           Verifies mixed scenario contains both short ([1, 4]) and medium/long ([8, 15]) jobs.
        2. How does the specific function work:
           Generates mixed workloads for n >= 2 across seeds and asserts that both short and long
           jobs are present, and all bursts belong to [1, 4] union [8, 15].
        3. Concepts used in the function:
           Bimodal mix guarantees, category coverage.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified mixed workload composition.
           After: Verified presence of both short and medium/long bursts for n >= 2.
        """
        for seed in range(10):
            workload = generate_workload(SCENARIO_MIXED, n=6, seed=seed)
            bursts = [p.burst for p in workload]
            assert all((1 <= b <= 4) or (8 <= b <= 15) for b in bursts)
            assert any(1 <= b <= 4 for b in bursts)
            assert any(8 <= b <= 15 for b in bursts)

    def test_bimodal_property_invariants(self) -> None:
        """
        1. The use of function:
           Verifies bimodal scenario contains short ([1, 3]) and long ([20, 35]) jobs for n >= 2.
        2. How does the specific function work:
           Generates bimodal workloads across seeds and asserts presence of short and long jobs,
           with all bursts in [1, 3] union [20, 35].
        3. Concepts used in the function:
           Bimodal distribution properties, mixture verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified bimodal composition guarantees.
           After: Verified presence of both modes across all generated bimodal workloads.
        """
        for seed in range(10):
            workload = generate_workload(SCENARIO_BIMODAL, n=6, seed=seed)
            bursts = [p.burst for p in workload]
            assert all((1 <= b <= 3) or (20 <= b <= 35) for b in bursts)
            assert any(1 <= b <= 3 for b in bursts)
            assert any(20 <= b <= 35 for b in bursts)

    def test_simultaneous_property_invariants(self) -> None:
        """
        1. The use of function:
           Verifies simultaneous scenario sets all process arrival times strictly to tick 0.
        2. How does the specific function work:
           Generates simultaneous workloads across seeds and asserts every process has arrival == 0
           and burst in [1, 10].
        3. Concepts used in the function:
           Arrival synchrony, tie-breaking stress testing.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified arrival synchrony.
           After: Confirmed arrival == 0 for all processes across all seeds.
        """
        for seed in range(10):
            workload = generate_workload(SCENARIO_SIMULTANEOUS, n=7, seed=seed)
            assert all(p.arrival == 0 for p in workload)
            assert all(1 <= p.burst <= 10 for p in workload)

    def test_staggered_property_invariants(self) -> None:
        """
        1. The use of function:
           Verifies staggered scenario spaces successive arrivals by positive spacing (3..6 ticks).
        2. How does the specific function work:
           Generates staggered workloads across seeds and asserts arrivals are strictly increasing
           with interarrival in [3, 6], and bursts in [2, 8].
        3. Concepts used in the function:
           Strictly increasing sequences, positive interarrival intervals.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified staggered arrival spacing.
           After: Verified 3 <= interarrival <= 6 for all successive arrivals.
        """
        for seed in range(10):
            workload = generate_workload(SCENARIO_STAGGERED, n=6, seed=seed)
            assert all(2 <= p.burst <= 8 for p in workload)
            arrivals = [p.arrival for p in workload]
            assert arrivals[0] == 0
            for i in range(1, len(arrivals)):
                interarrival = arrivals[i] - arrivals[i - 1]
                assert 3 <= interarrival <= 6

    def test_convoy_property_invariants(self) -> None:
        """
        1. The use of function:
           Verifies convoy scenario enforces P1 arriving at 0 with burst >= 3 * median(others).
        2. How does the specific function work:
           Generates convoy workloads for n >= 2 across seeds and asserts P1 arrives at 0 with
           burst >= 20, arrivals at 1, 2, ..., and P1.burst >= 3 * median(others).
        3. Concepts used in the function:
           Convoy effect modeling, statistical median invariant verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified convoy effect invariant.
           After: Verified P1 arrival, subsequent arrivals, and 3x median burst dominance.
        """
        for seed in range(15):
            workload = generate_workload(SCENARIO_CONVOY, n=6, seed=seed)
            assert workload[0].pid == "P1"
            assert workload[0].arrival == 0
            assert workload[0].burst >= 20
            other_bursts = [p.burst for p in workload[1:]]
            assert all(1 <= b <= 3 for b in other_bursts)
            median_other = float(np.median(other_bursts))
            assert workload[0].burst >= 3.0 * median_other
            for idx, proc in enumerate(workload):
                assert proc.arrival == idx

    def test_equal_bursts_property_invariants(self) -> None:
        """
        1. The use of function:
           Verifies equal_bursts scenario assigns identical bursts to all processes in a workload.
        2. How does the specific function work:
           Generates equal_bursts workloads across seeds and asserts all processes share the exact
           same burst B in [2, 10].
        3. Concepts used in the function:
           Homogeneous burst duration, degenerate tie evaluation.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified burst homogeneity.
           After: Verified all processes in equal_bursts share an identical burst B in [2, 10].
        """
        for seed in range(10):
            workload = generate_workload(SCENARIO_EQUAL_BURSTS, n=6, seed=seed)
            bursts = [p.burst for p in workload]
            assert len(set(bursts)) == 1
            assert 2 <= bursts[0] <= 10

    def test_high_load_property_invariants(self) -> None:
        """
        1. The use of function:
           Verifies high_load scenario creates queue saturation where arrival rate exceeds service.
        2. How does the specific function work:
           Generates high_load workloads across seeds and asserts bursts in [4, 12], interarrivals
           in [0, 1], and total burst exceeds arrival span significantly.
        3. Concepts used in the function:
           Queue saturation modeling, load ratio invariants.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified high load saturation characteristics.
           After: Verified bursts in [4, 12], interarrival in [0, 1], and heavy queue pressure.
        """
        for seed in range(10):
            workload = generate_workload(SCENARIO_HIGH_LOAD, n=8, seed=seed)
            assert all(4 <= p.burst <= 12 for p in workload)
            arrivals = [p.arrival for p in workload]
            for i in range(1, len(arrivals)):
                interarrival = arrivals[i] - arrivals[i - 1]
                assert 0 <= interarrival <= 1
            total_burst = sum(p.burst for p in workload)
            arrival_span = arrivals[-1] - arrivals[0]
            assert total_burst > arrival_span


class TestDomainComplianceAndExports:
    """
    1. The use of class:
       Verifies domain compliance, sequential PID naming, sorting invariants, and package exports.
    2. What the class contains:
       Unit tests validating that generated workloads pass validate_processes, maintain PID
       patterns P1..Pn, are sorted by (arrival, pid), and are properly exported at the package root.
    3. Technologies used and why:
       Domain validation helpers and module namespace introspection.
    4. Key design and overall design of the class:
       Ensures generator output conforms to core engine requirements and package export contracts.
    """

    def test_pid_naming_and_sorting(self) -> None:
        """
        1. The use of function:
           Verifies that generated processes have sequential PIDs and are sorted by (arrival, pid).
        2. How does the specific function work:
           Generates workloads across scenarios and asserts that process list is strictly sorted
           by key (arrival, pid) and all PIDs follow P1, P2, ..., Pn.
        3. Concepts used in the function:
           Identifier naming conventions, stable tie-break ordering.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unchecked process ordering and PID naming.
           After: Confirmed sequential PIDs and deterministic sorting by (arrival, pid).
        """
        for scenario in ALL_SCENARIOS:
            workload = generate_workload(scenario, n=6, seed=77)
            expected_pids = {f"P{i + 1}" for i in range(6)}
            assert {p.pid for p in workload} == expected_pids
            for i in range(len(workload) - 1):
                p_curr = workload[i]
                p_next = workload[i + 1]
                assert (p_curr.arrival, p_curr.pid) <= (p_next.arrival, p_next.pid)

    def test_workload_passes_domain_validation(self) -> None:
        """
        1. The use of function:
           Verifies that generated workloads pass domain validate_processes without error.
        2. How does the specific function work:
           Generates workloads for all scenarios and passes them directly to validate_processes.
        3. Concepts used in the function:
           Domain integrity validation, contractual interface conformance.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified domain contract validation on generated workloads.
           After: Confirmed validate_processes accepts all generated workloads.
        """
        for scenario in ALL_SCENARIOS:
            workload = generate_workload(scenario, n=5, seed=42)
            validate_processes(workload)

    def test_package_level_exports(self) -> None:
        """
        1. The use of function:
           Verifies ALL_SCENARIOS, generate_workload, and get_scenario are exported at package root.
        2. How does the specific function work:
           Imports symbols from schedsim.workloads and asserts identity with submodule definitions.
        3. Concepts used in the function:
           Package namespace re-exports, public API surface verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified package-level symbol exposure.
           After: Confirmed all generator and scenario symbols are exposed at package root.
        """
        import schedsim.workloads as pkg
        from schedsim.workloads.generator import generate_workload as direct_gen
        from schedsim.workloads.scenarios import ALL_SCENARIOS as direct_scenarios
        from schedsim.workloads.scenarios import get_scenario as direct_get

        assert pkg.generate_workload is direct_gen
        assert pkg.get_scenario is direct_get
        assert pkg.ALL_SCENARIOS is direct_scenarios
