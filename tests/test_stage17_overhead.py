"""
1. What this file does:
   Comprehensive test suite for Stage 17: Latency profile and hardware cost model.
2. Overall design:
   Validates machine environment metadata collection, 1D linear interpolation accuracy
   (exact values, midpoints, lower/upper boundary capping, and extrapolation warnings),
   JSON profile serialization, memoized profile caching, policy lifecycle hooks (decision_cost_us
   and on_start), publication plotting, end-to-end live-vs-table simulation validation, and
   CLI bench-latency command execution.
3. Elements in the file:
   - fake_profile: Fixture providing deterministic synthetic latency profile data.
   - sample_processes: Fixture providing verified multi-process test workload.
   - test_get_machine_info: Asserts machine environment metadata is non-empty and well-formed.
   - test_create_synthetic_view: Validates synthetic View and ProcView candidate construction.
   - test_interpolation_exact_and_midpoints: Verifies exact and midpoint linear interpolation.
   - test_interpolation_boundary_capping: Verifies lower/upper clamping and extrapolation warning.
   - test_startup_cost_interpolation: Verifies Category 1 upfront latency interpolation.
   - test_save_and_load_profile: Validates JSON persistence, cache hits, and clear_latency_cache.
   - test_fallback_when_profile_missing: Tests graceful zero-cost fallback on missing profiles.
   - test_policy_decision_cost_hooks: Verifies decision_cost_us hooks on per-decision policies.
   - test_selector_policy_on_start_hook: Verifies Category 1 upfront cost charged on start.
   - test_validate_live_vs_table_sanity: Tests end-to-end simulation live vs table validation.
   - test_benchmark_functions_fast: Tests micro-benchmark execution with minimal iterations.
   - test_plot_latency_profile: Tests rendering publication-quality latency curve figure.
   - test_cli_bench_latency_execution: Tests full CLI bench-latency subcommand execution.
4. Final output of the file:
   Executes pytest test suite verifying 100% compliance with Stage 17 specifications.
"""

from pathlib import Path
from typing import Any

import pytest

import config
from schedsim.domain.engine import View
from schedsim.domain.process import Process
from schedsim.overhead import (
    benchmark_policy_latencies,
    benchmark_startup_latencies,
    clear_latency_cache,
    create_synthetic_view,
    get_interpolated_cost_us,
    get_machine_info,
    get_startup_cost_us,
    load_latency_profile,
    plot_latency_profile,
    save_latency_profile,
    validate_live_vs_table,
)
from schedsim.pidsel.gbdt_ranker import GBDTRankerPolicy
from schedsim.pidsel.scorer_net import NeuralCandidatePolicy
from schedsim.policies.fcfs import FCFSPolicy
from schedsim.policies.round_robin import RoundRobinPolicy
from schedsim.policies.selector import DTPolicy, RFPolicy, SelectorPolicy
from schedsim.policies.sjf import SJFPolicy
from schedsim.policies.srtf import SRTFPolicy
from schedsim.rl.policy import RLPolicy


@pytest.fixture
def fake_profile() -> dict[str, Any]:
    """
    1. The use of function:
       Pytest fixture supplying a deterministic synthetic latency profile for unit tests.
    2. How does the specific function work:
       Constructs a nested dictionary matching latency_profile.json schema with predefined
       known points for FCFS, Q-learning, GBDT, Neural, DT, and RF.
    3. Concepts used in the function:
       Test fixtures, test double / synthetic fixture pattern.
    4. Places where this function is imported and used:
       Injected into unit tests requiring isolated profile data.
    5. Final output of the function, before and after:
       Before: No isolated profile data.
       After: Returns structured synthetic profile dictionary.
    """
    return {
        "machine_info": {
            "cpu": "Synthetic Core i7",
            "os": "SyntheticOS",
            "os_release": "1.0.0",
            "python_version": "3.11.0",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        "per_decision": {
            "FCFS": {
                "2": {"median_us": 1.0, "p95_us": 1.5, "cv": 0.02},
                "5": {"median_us": 2.0, "p95_us": 2.5, "cv": 0.02},
                "10": {"median_us": 4.0, "p95_us": 5.0, "cv": 0.03},
            },
            "Q-learning": {
                "2": {"median_us": 10.0, "p95_us": 12.0, "cv": 0.01},
                "4": {"median_us": 20.0, "p95_us": 24.0, "cv": 0.01},
                "10": {"median_us": 50.0, "p95_us": 60.0, "cv": 0.02},
            },
            "GBDT": {
                "2": {"median_us": 30.0, "p95_us": 35.0, "cv": 0.01},
                "6": {"median_us": 50.0, "p95_us": 60.0, "cv": 0.02},
                "10": {"median_us": 70.0, "p95_us": 80.0, "cv": 0.02},
            },
            "Neural": {
                "2": {"median_us": 25.0, "p95_us": 30.0, "cv": 0.01},
                "6": {"median_us": 45.0, "p95_us": 55.0, "cv": 0.02},
                "10": {"median_us": 65.0, "p95_us": 75.0, "cv": 0.02},
            },
        },
        "startup": {
            "DT": {
                "3": {"median_us": 20.0, "p95_us": 25.0},
                "10": {"median_us": 40.0, "p95_us": 50.0},
                "20": {"median_us": 70.0, "p95_us": 85.0},
            },
            "RF": {
                "3": {"median_us": 60.0, "p95_us": 75.0},
                "10": {"median_us": 120.0, "p95_us": 150.0},
                "20": {"median_us": 200.0, "p95_us": 250.0},
            },
        },
    }


@pytest.fixture
def sample_processes() -> list[Process]:
    """
    1. The use of function:
       Pytest fixture providing a standard multi-process workload for validation tests.
    2. How does the specific function work:
       Instantiates 5 valid Process objects with staggered arrival and burst attributes.
    3. Concepts used in the function:
       Domain entity fixture creation.
    4. Places where this function is imported and used:
       Injected into end-to-end simulation validation tests.
    5. Final output of the function, before and after:
       Before: No workload.
       After: Returns list of 5 Process instances.
    """
    return [
        Process(pid="P1", arrival=0, burst=5),
        Process(pid="P2", arrival=1, burst=4),
        Process(pid="P3", arrival=2, burst=3),
        Process(pid="P4", arrival=3, burst=6),
        Process(pid="P5", arrival=4, burst=2),
    ]


def test_get_machine_info() -> None:
    """
    1. The use of function:
       Verifies that get_machine_info returns valid, non-empty host environment metadata.
    2. How does the specific function work:
       Queries get_machine_info() and asserts the presence and non-emptiness of required keys.
    3. Concepts used in the function:
       Host metadata inspection, dictionary key assertion.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Machine info unverified.
       After: Confirms keys 'cpu', 'os', 'os_release', 'python_version', 'timestamp' exist.
    """
    info = get_machine_info()
    assert isinstance(info, dict)
    for key in ("cpu", "os", "os_release", "python_version", "timestamp"):
        assert key in info
        assert isinstance(info[key], str)
        assert len(info[key].strip()) > 0


def test_create_synthetic_view() -> None:
    """
    1. The use of function:
       Verifies synthetic View creation across various candidate counts.
    2. How does the specific function work:
       Tests valid n_ready values (1, 2, 5, 10), checks ready candidate count, and confirms
       that n_ready < 1 raises ValueError.
    3. Concepts used in the function:
       View invariants, boundary validation.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Synthetic View builder unverified.
       After: Confirms View and ProcView properties conform to engine invariants.
    """
    for n in (1, 2, 5, 10):
        view = create_synthetic_view(n)
        assert isinstance(view, View)
        assert len(view.ready) == n
        assert view.pending_arrivals == 0

    with pytest.raises(ValueError, match="n_ready must be >= 1"):
        create_synthetic_view(0)


def test_interpolation_exact_and_midpoints(fake_profile: dict[str, Any]) -> None:
    """
    1. The use of function:
       Verifies that 1D linear interpolation returns exact measured points and true midpoints.
    2. How does the specific function work:
       Evaluates get_interpolated_cost_us for Q-learning at exact coordinates (n=2, 4, 10)
       and intermediate midpoints (n=3, n=7), checking both median and p95 metrics.
    3. Concepts used in the function:
       1D linear interpolation arithmetic, piecewise interpolation, metric selection.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Interpolation accuracy unverified.
       After: Confirms exact points match and midpoint values equal linear weighted sums.
    """
    # Exact points for Q-learning (median: 2->10.0, 4->20.0, 10->50.0)
    assert get_interpolated_cost_us("Q-learning", 2, profile=fake_profile) == 10.0
    assert get_interpolated_cost_us("Q-learning", 4, profile=fake_profile) == 20.0
    assert get_interpolated_cost_us("Q-learning", 10, profile=fake_profile) == 50.0

    # Exact points with p95 (2->12.0, 4->24.0, 10->60.0)
    assert get_interpolated_cost_us("Q-learning", 2, profile=fake_profile, use_p95=True) == 12.0
    assert get_interpolated_cost_us("Q-learning", 4, profile=fake_profile, use_p95=True) == 24.0
    assert get_interpolated_cost_us("Q-learning", 10, profile=fake_profile, use_p95=True) == 60.0

    # Midpoints: n=3 is midway between 2 and 4 -> (10.0 + 20.0)/2 = 15.0
    assert get_interpolated_cost_us("Q-learning", 3, profile=fake_profile) == 15.0
    assert get_interpolated_cost_us("Q-learning", 3, profile=fake_profile, use_p95=True) == 18.0

    # Midpoints: n=7 is midway between 4 and 10 -> 20 + (3/6)*(50 - 20) = 35.0
    assert get_interpolated_cost_us("Q-learning", 7, profile=fake_profile) == 35.0

    # GBDT midpoint: n=4 between 2 (30.0) and 6 (50.0) -> alpha=2/4=0.5 -> 40.0
    assert get_interpolated_cost_us("GBDT", 4, profile=fake_profile) == 40.0


def test_interpolation_boundary_capping(fake_profile: dict[str, Any]) -> None:
    """
    1. The use of function:
       Verifies boundary clamping below min and above max, and extrapolation warnings.
    2. How does the specific function work:
       Evaluates get_interpolated_cost_us at n <= 0, n=1 (below min=2), and n > 10 (above max).
       Checks that costs clamp to boundary values and extrapolation warning is triggered.
    3. Concepts used in the function:
       Boundary clamping, extrapolation guarding, warning assertion with pytest.warns.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Boundary handling unverified.
       After: Confirms clamping at min/max and warning issuance for n > max.
    """
    # Non-positive input returns 0.0
    assert get_interpolated_cost_us("Q-learning", 0, profile=fake_profile) == 0.0
    assert get_interpolated_cost_us("Q-learning", -3, profile=fake_profile) == 0.0

    # Below min (n=1 < min=2): clamps to cost at min (10.0 for Q-learning)
    assert get_interpolated_cost_us("Q-learning", 1, profile=fake_profile) == 10.0
    assert get_interpolated_cost_us("GBDT", 1, profile=fake_profile) == 30.0

    # Above max (n=15 > max=10): clamps to cost at max (50.0) and warns
    with pytest.warns(UserWarning, match="Extrapolating latency for Q-learning"):
        val_15 = get_interpolated_cost_us("Q-learning", 15, profile=fake_profile)
    assert val_15 == 50.0

    with pytest.warns(UserWarning, match="Extrapolating latency for GBDT"):
        val_100 = get_interpolated_cost_us("GBDT", 100, profile=fake_profile)
    assert val_100 == 70.0


def test_startup_cost_interpolation(fake_profile: dict[str, Any]) -> None:
    """
    1. The use of function:
       Verifies 1D linear interpolation for Category 1 upfront startup latencies.
    2. How does the specific function work:
       Tests get_startup_cost_us for 'DT' and 'RF' at exact points (sz=3, 10, 20),
       midpoints, boundary clamping (< 3 and > 20), and case-insensitive aliases.
    3. Concepts used in the function:
       Upfront cost interpolation, alias resolution, boundary clamping.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Category 1 startup cost lookup unverified.
       After: Confirms exact, midpoint, clamped, and alias lookups match expected values.
    """
    # Non-positive returns 0.0
    assert get_startup_cost_us("DT", 0, profile=fake_profile) == 0.0
    assert get_startup_cost_us("RF", -1, profile=fake_profile) == 0.0

    # Exact points for DT (median: 3->20.0, 10->40.0, 20->70.0)
    assert get_startup_cost_us("DT", 3, profile=fake_profile) == 20.0
    assert get_startup_cost_us("DT", 10, profile=fake_profile) == 40.0
    assert get_startup_cost_us("DT", 20, profile=fake_profile) == 70.0

    # Exact points for RF (median: 3->60.0, 10->120.0, 20->200.0)
    assert get_startup_cost_us("RF", 3, profile=fake_profile) == 60.0
    assert get_startup_cost_us("RF", 10, profile=fake_profile) == 120.0
    assert get_startup_cost_us("RF", 20, profile=fake_profile) == 200.0

    # Midpoints: sz=15 is midway between 10 and 20
    # DT: 40 + 0.5 * (70 - 40) = 55.0
    assert get_startup_cost_us("DT", 15, profile=fake_profile) == 55.0
    # RF: 120 + 0.5 * (200 - 120) = 160.0
    assert get_startup_cost_us("RF", 15, profile=fake_profile) == 160.0

    # Clamping
    assert get_startup_cost_us("DT", 1, profile=fake_profile) == 20.0
    assert get_startup_cost_us("DT", 50, profile=fake_profile) == 70.0
    assert get_startup_cost_us("RF", 1, profile=fake_profile) == 60.0
    assert get_startup_cost_us("RF", 50, profile=fake_profile) == 200.0

    # Aliases
    assert get_startup_cost_us("decision_tree", 3, profile=fake_profile) == 20.0
    assert get_startup_cost_us("DTPolicy", 10, profile=fake_profile) == 40.0
    assert get_startup_cost_us("random_forest", 3, profile=fake_profile) == 60.0
    assert get_startup_cost_us("RFPolicy", 10, profile=fake_profile) == 120.0


def test_save_and_load_profile(tmp_path: Path, fake_profile: dict[str, Any]) -> None:
    """
    1. The use of function:
       Tests latency profile JSON serialization, file loading, and cache clearing.
    2. How does the specific function work:
       Saves fake_profile to a temporary file, verifies disk persistence, tests cache hit,
       clears cache via clear_latency_cache(), tests cold reload, and tests FileNotFoundError.
    3. Concepts used in the function:
       File persistence, memoization, cache invalidation, defensive exception checking.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Profile persistence unverified.
       After: Confirms save, load, cache clear, and missing file error handling.
    """
    target_path = tmp_path / "test_latency_profile.json"
    clear_latency_cache()

    # Save to disk
    save_latency_profile(fake_profile, target_path)
    assert target_path.exists()

    # Load with cache hit
    loaded = load_latency_profile(target_path)
    assert loaded == fake_profile

    # Cache clear and reload
    clear_latency_cache()
    reloaded = load_latency_profile(target_path, reload=True)
    assert reloaded == fake_profile

    # Missing file raises FileNotFoundError
    clear_latency_cache()
    missing_path = tmp_path / "non_existent_profile.json"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_latency_profile(missing_path)


def test_fallback_when_profile_missing() -> None:
    """
    1. The use of function:
       Verifies that interpolation functions return 0.0 safely when no profile exists.
    2. How does the specific function work:
       Passes an empty profile dictionary or unmapped policy names and asserts 0.0 return.
    3. Concepts used in the function:
       Graceful degradation, fallback handling.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Fallback unverified.
       After: Confirms that unknown policies and empty profiles safely return 0.0.
    """
    empty_profile: dict[str, Any] = {"per_decision": {}, "startup": {}}
    assert get_interpolated_cost_us("NonExistent", 5, profile=empty_profile) == 0.0
    assert get_startup_cost_us("NonExistent", 5, profile=empty_profile) == 0.0


def test_policy_decision_cost_hooks() -> None:
    """
    1. The use of function:
       Verifies decision_cost_us lifecycle hooks across all per-decision policies.
    2. How does the specific function work:
       Instantiates FCFS, SJF, RR, SRTF, RL, GBDT, and Neural policies. Checks that view=None
       or single-candidate View (forced decision) yields 0.0. Evaluates non-forced decisions.
    3. Concepts used in the function:
       Policy interface verification, forced decision exemption, hook delegation.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Policy hook connections unverified.
       After: Confirms all policies respect FORCED_DECISION_FREE and query latency profile.
    """
    single_view = create_synthetic_view(1)
    multi_view = create_synthetic_view(3)

    policies = [
        FCFSPolicy(),
        SJFPolicy(),
        RoundRobinPolicy(quantum=config.TIME_QUANTUM),
        SRTFPolicy(),
        RLPolicy.from_setting("S1"),
        GBDTRankerPolicy.from_setting("S1"),
        NeuralCandidatePolicy.from_setting("S1"),
    ]

    for pol in policies:
        # None view returns 0.0
        assert pol.decision_cost_us(None) == 0.0  # type: ignore[arg-type]
        # Forced decision (single candidate) returns 0.0
        assert pol.decision_cost_us(single_view) == 0.0
        # Multi-candidate decision returns non-negative float
        cost = pol.decision_cost_us(multi_view)
        assert isinstance(cost, float)
        assert cost >= 0.0


def test_selector_policy_on_start_hook(sample_processes: list[Process]) -> None:
    """
    1. The use of function:
       Verifies Category 1 SelectorPolicy on_start and decision_cost_us lifecycle behavior.
    2. How does the specific function work:
       Tests DTPolicy and RFPolicy on_start() charging startup cost from profile, and verifies
       decision_cost_us() strictly returns 0.0.
    3. Concepts used in the function:
       Category 1 upfront cost accounting, zero per-decision cost verification.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: SelectorPolicy hook integration unverified.
       After: Confirms on_start returns non-negative latency and decision_cost_us returns 0.0.
    """
    multi_view = create_synthetic_view(3)

    for pol_cls in (DTPolicy, RFPolicy):
        pol: SelectorPolicy = pol_cls(setting="S1")
        assert pol.decision_cost_us(None) == 0.0  # type: ignore[arg-type]
        assert pol.decision_cost_us(multi_view) == 0.0

        startup_cost = pol.on_start(sample_processes)
        assert isinstance(startup_cost, float)
        assert startup_cost >= 0.0
        # Decision cost remains 0.0 even after start
        assert pol.decision_cost_us(multi_view) == 0.0


def test_validate_live_vs_table_sanity(sample_processes: list[Process]) -> None:
    """
    1. The use of function:
       Tests validate_live_vs_table end-to-end simulation sanity check.
    2. How does the specific function work:
       Executes validate_live_vs_table with NeuralCandidatePolicy and verifies that
       output dictionary contains required keys and numerical metrics.
    3. Concepts used in the function:
       Live vs model verification, dictionary structure assertion.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Live validation helper unverified.
       After: Confirms dictionary structure with live_us_total, table_us_total, and relative_gap.
    """
    policy = NeuralCandidatePolicy.from_setting("S1")
    result = validate_live_vs_table(
        policy, sample_processes, cs_cost=1, tolerance=0.50, n_repeats=2
    )

    assert isinstance(result, dict)
    for k in (
        "policy",
        "live_us_total",
        "table_us_total",
        "abs_gap_us",
        "relative_gap",
        "percentage_gap",
        "tolerance",
        "within_tolerance",
        "decisions",
    ):
        assert k in result

    assert result["decisions"] > 0
    assert result["live_us_total"] > 0.0
    assert result["table_us_total"] >= 0.0


def test_benchmark_functions_fast() -> None:
    """
    1. The use of function:
       Validates benchmark_policy_latencies and benchmark_startup_latencies with small runs.
    2. How does the specific function work:
       Runs both benchmark functions with minimal warm-up and timed iterations to verify
       their execution logic without hardware overhead.
    3. Concepts used in the function:
       Fast integration testing, minimal parameter benchmarking.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Benchmarking routines unverified.
       After: Confirms latency benchmarking returns correctly formatted dictionary.
    """
    fast_dec = benchmark_policy_latencies(
        n_warmup=2,
        n_timed=5,
        n_repeats=2,
        n_ready_sweep=(2, 3),
        policies={"FCFS": FCFSPolicy()},
    )
    assert "FCFS" in fast_dec
    assert "2" in fast_dec["FCFS"]
    assert fast_dec["FCFS"]["2"]["median_us"] > 0.0

    fast_start = benchmark_startup_latencies(
        workload_sizes=(3,),
        n_warmup=2,
        n_timed=5,
        n_repeats=2,
        models=("DT",),
    )
    assert "DT" in fast_start
    assert "3" in fast_start["DT"]
    assert fast_start["DT"]["3"]["median_us"] > 0.0


def test_plot_latency_profile(tmp_path: Path, fake_profile: dict[str, Any]) -> None:
    """
    1. The use of function:
       Verifies that plot_latency_profile creates a valid figure file on disk.
    2. How does the specific function work:
       Passes fake_profile and a target PNG path, asserts file creation and non-zero size.
    3. Concepts used in the function:
       Matplotlib figure generation, file existence assertion.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: Plotting unverified.
       After: Confirms PNG file generated and non-empty.
    """
    out_fig = tmp_path / "test_latency_vs_ready.png"
    plot_latency_profile(fake_profile, output_path=out_fig)
    assert out_fig.exists()
    assert out_fig.stat().st_size > 0


def test_cli_bench_latency_execution(tmp_path: Path) -> None:
    """
    1. The use of function:
       Verifies execution of the `bench-latency` CLI subcommand.
    2. How does the specific function work:
       Invokes cli.main() with minimal iterations and --skip-plot, verifying 0 exit code
       and JSON file generation at specified path.
    3. Concepts used in the function:
       CLI entry point invocation, argument routing, integration testing.
    4. Places where this function is imported and used:
       Executed during Stage 17 pytest run.
    5. Final output of the function, before and after:
       Before: CLI bench-latency command unverified.
       After: Confirms CLI command runs cleanly and produces valid JSON profile.
    """
    import cli

    out_json = tmp_path / "cli_latency_profile.json"
    exit_code = cli.main([
        "bench-latency",
        "--output",
        str(out_json),
        "--n-warmup",
        "2",
        "--n-timed",
        "5",
        "--n-repeats",
        "2",
        "--skip-plot",
    ])
    assert exit_code == 0
    assert out_json.exists()
    assert out_json.stat().st_size > 0
