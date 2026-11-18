"""
1. What this file does:
   Tests the preset workloads module (presets.py) and workloads package exports.
2. Overall design:
   Contains test suites validating preset retrieval, case-insensitivity, alias resolution,
   workload specification correctness, domain validation invariant compliance, instance
   isolation / immutability, error diagnostics (KeyError and TypeError), and end-to-end
   execution of presets through the simulation Engine.
3. Elements in the file:
   - TestPresetRetrieval: Tests retrieving presets by canonical names, aliases, and formatting.
   - TestPresetDomainValidation: Verifies all presets pass domain collection invariants.
   - TestPresetIndependence: Verifies distinct instance generation and immutability.
   - TestPresetErrorHandling: Tests KeyError diagnostics and TypeError boundary conditions.
   - TestPresetSimulationExecution: End-to-end execution of presets with scheduling policies.
4. Final output of the file:
   Confirms full test coverage and reliability of workload preset components for pytest.
"""

import pytest

from schedsim.domain.engine import Engine
from schedsim.domain.process import Process, validate_processes
from schedsim.policies.fcfs import FCFSPolicy
from schedsim.policies.sjf import SJFPolicy
from schedsim.workloads import get_preset as pkg_get_preset
from schedsim.workloads import list_presets as pkg_list_presets
from schedsim.workloads.presets import get_preset, list_presets


class TestPresetRetrieval:
    """
    1. The use of class:
       Tests retrieval of preset workloads via canonical names, aliases, and string variations.
    2. What the class contains:
       Unit tests covering all standard presets, alias equivalence, casing, and list introspection.
    3. Technologies used and why:
       pytest test runner and assertions to verify data structures and dictionary mappings.
    4. Key design and overall design of the class:
       Organized by functional concern, checking both individual preset structures and collective
       introspection capabilities.
    """

    def test_list_presets_contains_all_required_presets(self) -> None:
        """
        1. The use of function:
           Verifies that list_presets returns a dictionary containing all expected preset keys.
        2. How does the specific function work:
           Calls list_presets() and checks for the presence of canonical names and aliases with
           non-empty description strings.
        3. Concepts used in the function:
           Dictionary introspection, string validation, collection membership.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified list_presets dictionary.
           After: Verified complete mapping with valid non-empty description strings.
        """
        presets = list_presets()
        required_keys = [
            "section_4_6",
            "1",
            "convoy",
            "equal_bursts",
            "simultaneous",
            "long_idle_gap",
            "staggered",
            "high_contention",
            "mixed_priorities",
        ]
        for key in required_keys:
            assert key in presets, f"Expected '{key}' in list_presets()"
            assert isinstance(presets[key], str)
            assert len(presets[key].strip()) > 0

    def test_list_presets_defensive_copy(self) -> None:
        """
        1. The use of function:
           Verifies that modifying the dictionary returned by list_presets does not mutate internal
           preset registry state.
        2. How does the specific function work:
           Retrieves presets, alters the returned dictionary, and checks that a fresh call to
           list_presets() retains its original contents.
        3. Concepts used in the function:
           Defensive copying, immutability, state isolation.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified dictionary return behavior.
           After: Confirmed defensive copy behavior preventing external state corruption.
        """
        p1 = list_presets()
        p1["mutated_key"] = "test"
        p2 = list_presets()
        assert "mutated_key" not in p2

    def test_package_reexports_match_module(self) -> None:
        """
        1. The use of function:
           Verifies that schedsim.workloads re-exports get_preset and list_presets identically.
        2. How does the specific function work:
           Asserts identity between package-level and module-level function objects.
        3. Concepts used in the function:
           Package namespace re-export verification, identity checks.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified package namespace symbols.
           After: Confirmed identity of exported factory and listing functions.
        """
        assert pkg_get_preset is get_preset
        assert pkg_list_presets is list_presets

    def test_section_4_6_specifications_and_alias(self) -> None:
        """
        1. The use of function:
           Verifies the Section 4.6 hand-verified reference fixture and its '1' alias.
        2. How does the specific function work:
           Retrieves preset 'section_4_6' and '1', verifies their exact process attributes, and
           confirms equality between both retrieval paths.
        3. Concepts used in the function:
           Canonical reference fixture checking, alias equivalence, attribute inspection.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified Section 4.6 process parameters.
           After: Confirmed exact arrival and burst matches (P1: 0/5, P2: 1/3, P3: 2/2).
        """
        procs_canonical = get_preset("section_4_6")
        procs_alias = get_preset("1")
        assert procs_canonical == procs_alias

        assert len(procs_canonical) == 3
        p1, p2, p3 = procs_canonical
        assert p1 == Process("P1", arrival=0, burst=5, priority=0)
        assert p2 == Process("P2", arrival=1, burst=3, priority=0)
        assert p3 == Process("P3", arrival=2, burst=2, priority=0)

    def test_convoy_preset_specifications(self) -> None:
        """
        1. The use of function:
           Verifies the convoy preset contains one long CPU-bound process followed by short jobs.
        2. How does the specific function work:
           Retrieves 'convoy', checks process count (5), verifies P1 burst is 20, and verifies
           subsequent processes have small bursts (1-2) arriving sequentially.
        3. Concepts used in the function:
           Convoy effect pattern verification, process attribute inspection.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified convoy preset structure.
           After: Confirmed P1 has burst 20 and P2-P5 have bursts <= 2 arriving at ticks 1..4.
        """
        procs = get_preset("convoy")
        assert len(procs) == 5
        assert procs[0] == Process("P1", arrival=0, burst=20, priority=0)
        assert procs[1] == Process("P2", arrival=1, burst=2, priority=0)
        assert procs[2] == Process("P3", arrival=2, burst=2, priority=0)
        assert procs[3] == Process("P4", arrival=3, burst=1, priority=0)
        assert procs[4] == Process("P5", arrival=4, burst=2, priority=0)

    def test_equal_bursts_preset_specifications(self) -> None:
        """
        1. The use of function:
           Verifies the equal_bursts preset features identical burst lengths.
        2. How does the specific function work:
           Retrieves 'equal_bursts', checks count (4), verifies all bursts equal 4, and arrivals
           are 0, 1, 2, 3.
        3. Concepts used in the function:
           Attribute uniformity verification, sequence matching.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified equal_bursts configuration.
           After: Confirmed identical burst lengths across all sequential arrivals.
        """
        procs = get_preset("equal_bursts")
        assert len(procs) == 4
        assert [p.burst for p in procs] == [4, 4, 4, 4]
        assert [p.arrival for p in procs] == [0, 1, 2, 3]

    def test_simultaneous_preset_specifications(self) -> None:
        """
        1. The use of function:
           Verifies the simultaneous preset features concurrent arrivals at tick 0.
        2. How does the specific function work:
           Retrieves 'simultaneous', asserts all arrival times equal 0, and bursts are [8, 4, 2, 6].
        3. Concepts used in the function:
           Concurrent arrival invariant checking, burst sequence validation.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified simultaneous arrival configuration.
           After: Confirmed all arrivals are 0 with varying CPU burst lengths.
        """
        procs = get_preset("simultaneous")
        assert len(procs) == 4
        assert all(p.arrival == 0 for p in procs)
        assert [p.burst for p in procs] == [8, 4, 2, 6]

    def test_long_idle_gap_preset_specifications(self) -> None:
        """
        1. The use of function:
           Verifies the long_idle_gap preset separates process clusters by a wide idle margin.
        2. How does the specific function work:
           Retrieves 'long_idle_gap', checks that early jobs arrive at ticks 0, 1 and finish by
           tick 5, while subsequent jobs arrive at ticks 15, 16.
        3. Concepts used in the function:
           Workload clustering verification, gap duration calculation.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified idle gap structure.
           After: Confirmed 10-tick gap between batch 1 completion and batch 2 arrival.
        """
        procs = get_preset("long_idle_gap")
        assert len(procs) == 4
        assert procs[0] == Process("P1", arrival=0, burst=3)
        assert procs[1] == Process("P2", arrival=1, burst=2)
        assert procs[2] == Process("P3", arrival=15, burst=4)
        assert procs[3] == Process("P4", arrival=16, burst=2)

    def test_staggered_preset_specifications(self) -> None:
        """
        1. The use of function:
           Verifies the staggered preset arrivals occur at periodic intervals.
        2. How does the specific function work:
           Retrieves 'staggered', verifies arrivals [0, 3, 6, 9] and identical bursts [4, 4, 4, 4].
        3. Concepts used in the function:
           Periodic arrival spacing verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified staggered arrival intervals.
           After: Confirmed periodic 3-tick offsets and uniform burst lengths.
        """
        procs = get_preset("staggered")
        assert len(procs) == 4
        assert [p.arrival for p in procs] == [0, 3, 6, 9]
        assert [p.burst for p in procs] == [4, 4, 4, 4]

    def test_high_contention_preset_specifications(self) -> None:
        """
        1. The use of function:
           Verifies the high_contention preset includes 6 processes arriving across ticks 0..3.
        2. How does the specific function work:
           Retrieves 'high_contention', asserts process count is 6 and all arrival times are <= 3.
        3. Concepts used in the function:
           High-load queue contention verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified high contention parameters.
           After: Confirmed 6 processes arriving within the initial 4 simulation ticks.
        """
        procs = get_preset("high_contention")
        assert len(procs) == 6
        assert all(0 <= p.arrival <= 3 for p in procs)

    def test_mixed_priorities_preset_specifications(self) -> None:
        """
        1. The use of function:
           Verifies the mixed_priorities preset defines non-zero process priorities.
        2. How does the specific function work:
           Retrieves 'mixed_priorities', confirms non-default priority values (e.g. 1, 2, 3, 4).
        3. Concepts used in the function:
           Priority scheduling input verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified priority settings.
           After: Confirmed presence of varying priority levels across processes.
        """
        procs = get_preset("mixed_priorities")
        assert len(procs) == 5
        priorities = [p.priority for p in procs]
        assert set(priorities) == {0, 1, 2, 3, 4}

    @pytest.mark.parametrize(
        "query,expected_canonical",
        [
            ("CONVOY", "convoy"),
            ("Convoy", "convoy"),
            ("  convoy  ", "convoy"),
            ("equal-bursts", "equal_bursts"),
            ("EQUAL BURSTS", "equal_bursts"),
            ("equal_bursts", "equal_bursts"),
            ("simultaneous", "simultaneous"),
            ("SIMULTANEOUS", "simultaneous"),
            ("long-idle-gap", "long_idle_gap"),
            ("long idle gap", "long_idle_gap"),
            ("staggered", "staggered"),
            ("STAGGERED", "staggered"),
            ("high-contention", "high_contention"),
            ("high contention", "high_contention"),
            ("HIGH_CONTENTION", "high_contention"),
            ("section-4-6", "section_4_6"),
            ("section 4.6", "section_4_6"),
            ("section_4_6", "section_4_6"),
            ("SECTION_4_6", "section_4_6"),
            ("1", "section_4_6"),
            ("  1  ", "section_4_6"),
            ("mixed-priorities", "mixed_priorities"),
            ("MIXED PRIORITIES", "mixed_priorities"),
        ],
    )
    def test_case_and_alias_normalization(
        self, query: str, expected_canonical: str
    ) -> None:
        """
        1. The use of function:
           Tests case-insensitivity, spacing, hyphens, and alias resolution in get_preset.
        2. How does the specific function work:
           Queries get_preset with variations of casing, hyphens, and spaces, and verifies that the
           returned process list matches the canonical preset.
        3. Concepts used in the function:
           String normalization, parameterized testing, equivalence partitioning.
        4. Places where this function is imported and used:
           Executed by pytest parameterized runner.
        5. Final output of the function, before and after:
           Before: Unverified string normalization query.
           After: Confirmed identical output to the canonical preset retrieval.
        """
        result = get_preset(query)
        expected = get_preset(expected_canonical)
        assert result == expected


class TestPresetDomainValidation:
    """
    1. The use of class:
       Validates that all presets conform to core domain invariants via validate_processes.
    2. What the class contains:
       Iterative tests passing all presets through validate_processes.
    3. Technologies used and why:
       Integration with schedsim.domain.process.validate_processes to guarantee input sanity.
    4. Key design and overall design of the class:
       Ensures that every defined workload satisfies PID uniqueness, non-empty collections,
       and non-negative parameters before engines or policies consume them.
    """

    def test_all_presets_pass_validate_processes(self) -> None:
        """
        1. The use of function:
           Iterates through all registered presets and executes validate_processes on each.
        2. How does the specific function work:
           Calls list_presets() to get all keys, loads each with get_preset(key), and invokes
           validate_processes(procs), verifying no ValueError or TypeError is raised.
        3. Concepts used in the function:
           Domain invariant enforcement, batch validation.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified batch of preset process collections.
           After: Confirmed 100% compliance with domain collection constraints.
        """
        all_presets = list_presets()
        for preset_name in all_presets:
            procs = get_preset(preset_name)
            assert len(procs) > 0
            # validate_processes raises an exception if invariants are violated
            validate_processes(procs)

    def test_pids_are_unique_and_non_empty(self) -> None:
        """
        1. The use of function:
           Verifies that every preset possesses strictly unique, non-empty process IDs.
        2. How does the specific function work:
           Inspects every process in every preset to verify string validity and PID uniqueness.
        3. Concepts used in the function:
           Set uniqueness, PID boundary validation.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unverified PID uniqueness across presets.
           After: Confirmed unique non-empty identifiers for all processes in all presets.
        """
        for preset_name in list_presets():
            procs = get_preset(preset_name)
            pids = [p.pid for p in procs]
            assert len(pids) == len(set(pids)), f"Duplicate PIDs in preset '{preset_name}'"
            for pid in pids:
                assert isinstance(pid, str) and len(pid.strip()) > 0


class TestPresetIndependence:
    """
    1. The use of class:
       Tests instance isolation and defensive reproduction of Process lists from get_preset.
    2. What the class contains:
       Tests checking object identity (is not) between successive calls and mutation resilience.
    3. Technologies used and why:
       Python object identity checks (is, id) and collection mutation assertions.
    4. Key design and overall design of the class:
       Guarantees that simulation executions cannot corrupt shared workload definitions.
    """

    def test_get_preset_returns_fresh_instances(self) -> None:
        """
        1. The use of function:
           Verifies get_preset produces distinct list objects and distinct Process instances.
        2. How does the specific function work:
           Calls get_preset('convoy') twice, asserts the list instances are distinct, and asserts
           each contained Process object is distinct in memory.
        3. Concepts used in the function:
           Object identity vs value equality, instance independence, immutability.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Two workload batches from identical preset query.
           After: Verified equality (==) but distinct identities (is not).
        """
        call1 = get_preset("convoy")
        call2 = get_preset("convoy")

        assert call1 == call2
        assert call1 is not call2

        for p1, p2 in zip(call1, call2):
            assert p1 == p2
            assert p1 is not p2

    def test_mutation_of_returned_list_does_not_corrupt_preset(self) -> None:
        """
        1. The use of function:
           Verifies that modifying a returned process list does not affect subsequent calls.
        2. How does the specific function work:
           Retrieves 'section_4_6', mutates the list via append and pop, then retrieves
           'section_4_6' again and verifies its original 3 processes remain intact.
        3. Concepts used in the function:
           State corruption prevention, defensive instance creation.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Mutated process list.
           After: Verified uncorrupted subsequent call returning clean baseline processes.
        """
        workload = get_preset("section_4_6")
        original_len = len(workload)
        workload.append(Process("MUTATED", 0, 10))
        workload.pop(0)

        fresh = get_preset("section_4_6")
        assert len(fresh) == original_len
        assert [p.pid for p in fresh] == ["P1", "P2", "P3"]


class TestPresetErrorHandling:
    """
    1. The use of class:
       Tests error handling and diagnostics for invalid preset requests.
    2. What the class contains:
       Tests for unknown preset names, empty strings, and non-string argument types.
    3. Technologies used and why:
       pytest.raises to verify KeyError and TypeError exceptions and error message formats.
    4. Key design and overall design of the class:
       Ensures descriptive error messages guide users and developers to available presets.
    """

    def test_unknown_preset_raises_key_error_with_available_list(self) -> None:
        """
        1. The use of function:
           Verifies that querying an unrecognized preset raises a KeyError listing valid presets.
        2. How does the specific function work:
           Invokes get_preset with an unknown string, catches KeyError, and verifies the exception
           message includes 'Unknown preset workload' and mentions available options.
        3. Concepts used in the function:
           Exception assertion, diagnostic error message formatting.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unrecognized preset query.
           After: Raised KeyError containing available preset names in its message.
        """
        with pytest.raises(KeyError) as exc_info:
            get_preset("non_existent_preset_xyz")

        err_msg = str(exc_info.value)
        assert "Unknown preset workload 'non_existent_preset_xyz'" in err_msg
        assert "Available presets:" in err_msg
        assert "convoy" in err_msg
        assert "section_4_6" in err_msg

    def test_empty_and_whitespace_preset_raises_key_error(self) -> None:
        """
        1. The use of function:
           Verifies that empty or whitespace strings raise KeyError.
        2. How does the specific function work:
           Invokes get_preset('') and get_preset('   '), asserting KeyError is raised with
           available presets.
        3. Concepts used in the function:
           Boundary condition testing, string whitespace validation.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Empty or whitespace query string.
           After: Raised KeyError with diagnostic choices.
        """
        with pytest.raises(KeyError, match="Unknown preset workload"):
            get_preset("")

        with pytest.raises(KeyError, match="Unknown preset workload"):
            get_preset("   ")

    @pytest.mark.parametrize("invalid_arg", [None, 123, 4.5, True, ["convoy"], {"name": "convoy"}])
    def test_invalid_type_raises_type_error(self, invalid_arg: object) -> None:
        """
        1. The use of function:
           Verifies that passing non-string values raises TypeError.
        2. How does the specific function work:
           Iterates over invalid types (int, float, bool, list, dict, None) and asserts TypeError.
        3. Concepts used in the function:
           Defensive type checking, parameterized exception verification.
        4. Places where this function is imported and used:
           Executed by pytest parameterized runner.
        5. Final output of the function, before and after:
           Before: Non-string parameter passed to get_preset.
           After: Raised TypeError with informative type name message.
        """
        with pytest.raises(TypeError, match="Preset name_or_id must be a str"):
            get_preset(invalid_arg)  # type: ignore[arg-type]


class TestPresetSimulationExecution:
    """
    1. The use of class:
       Tests end-to-end simulation of preset workloads through Engine and classical policies.
    2. What the class contains:
       Verification of Section 4.6 exact hand-calculated metrics and convoy effect execution.
    3. Technologies used and why:
       Integration between workloads, Engine, policies, and Metrics to verify end-to-end integrity.
    4. Key design and overall design of the class:
       Confirms presets are immediately operational and produce expected theoretical outcomes.
    """

    def test_section_4_6_fcfs_exact_metrics(self) -> None:
        """
        1. The use of function:
           Executes Section 4.6 reference fixture with FCFS and verifies exact theoretical metrics.
        2. How does the specific function work:
           Loads 'section_4_6', initializes Engine with FCFSPolicy (cs_cost=0), executes to
           completion, and verifies exact waiting times, turnaround times, and makespan.
        3. Concepts used in the function:
           End-to-end integration, theoretical verification, exact metric assertion.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unsimulated Section 4.6 preset processes.
           After: Verified completion at tick 10 with P1 wait=0, P2 wait=4, P3 wait=6.
        """
        procs = get_preset("section_4_6")
        engine = Engine(processes=procs, cs_cost=0)
        policy = FCFSPolicy()

        view = engine.reset()
        while not engine.done:
            pid = policy.select(view)
            step_result = engine.step(pid)
            view = step_result.view

        result = engine.result()
        assert result.simulation_metrics.makespan == 10
        assert result.timeline.busy_ticks == 10
        assert result.timeline.idle_ticks == 0

        p1_m = next(m for m in result.process_metrics if m.pid == "P1")
        p2_m = next(m for m in result.process_metrics if m.pid == "P2")
        p3_m = next(m for m in result.process_metrics if m.pid == "P3")

        # P1: arrival 0, burst 5 -> completes at 5, wait = 0, turnaround = 5
        assert p1_m.completion == 5
        assert p1_m.waiting == 0
        assert p1_m.turnaround == 5

        # P2: arrival 1, burst 3 -> completes at 8, wait = 4, turnaround = 7
        assert p2_m.completion == 8
        assert p2_m.waiting == 4
        assert p2_m.turnaround == 7

        # P3: arrival 2, burst 2 -> completes at 10, wait = 6, turnaround = 8
        assert p3_m.completion == 10
        assert p3_m.waiting == 6
        assert p3_m.turnaround == 8

        # Summary averages
        assert result.simulation_metrics.avg_waiting == 10 / 3
        assert result.simulation_metrics.avg_turnaround == 20 / 3

    def test_convoy_preset_runs_cleanly_under_fcfs_and_sjf(self) -> None:
        """
        1. The use of function:
           Runs the convoy preset under FCFS and SJF, verifying clean execution and convoy impact.
        2. How does the specific function work:
           Simulates 'convoy' under both FCFS and non-preemptive SJF. Verifies both finish with
           identical makespan (27 ticks) and that all processes complete.
        3. Concepts used in the function:
           Comparative policy execution, convoy effect verification.
        4. Places where this function is imported and used:
           Executed by pytest during test discovery.
        5. Final output of the function, before and after:
           Before: Unsimulated convoy workload.
           After: Confirmed clean completion under multiple policies with identical makespan.
        """
        for policy_cls in (FCFSPolicy, SJFPolicy):
            procs = get_preset("convoy")
            engine = Engine(processes=procs, cs_cost=0)
            policy = policy_cls()

            view = engine.reset()
            while not engine.done:
                pid = policy.select(view)
                step_result = engine.step(pid)
                view = step_result.view

            result = engine.result()
            assert result.simulation_metrics.makespan == 27
            assert len(result.process_metrics) == 5
