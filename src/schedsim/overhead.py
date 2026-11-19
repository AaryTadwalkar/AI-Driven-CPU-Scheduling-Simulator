"""
1. What this file does:
   Manages hardware overhead profiling, micro-benchmarking, 1D linear latency interpolation,
   profile caching, latency-vs-ready plotting, and live-timing simulation validation.
2. Overall design:
   Establishes single-threaded execution controls for reproducible CPU timing. Micro-benchmarks
   per-decision scheduling policies (FCFS, SJF, RR, SRTF, Q-learning, GBDT ranker, Neural scorer)
   across ready queue sizes, and Category 1 models (Decision Tree, Random Forest) across workload
   sizes. Exports structured latency profiles to JSON, provides fast 1D linear interpolation with
   boundary clamping, generates publication-quality visualization figures, and validates
   table-based cost modeling against live execution timing.
3. Elements in the file:
   - _CACHED_PROFILE: In-memory cache holding loaded latency profile data.
   - _LiveTimingProxy: Policy proxy recording live nanoseconds elapsed during simulation.
   - get_machine_info: Retrieves platform, CPU, OS, and Python runtime metadata.
   - create_synthetic_view: Generates synthetic View with valid ProcView candidates.
   - benchmark_policy_latencies: Times per-decision policies across ready queue sizes.
   - benchmark_startup_latencies: Times Category 1 feature extraction and prediction.
   - benchmark_all: Executes full micro-benchmark suite combining per-decision and startup models.
   - save_latency_profile: Persists structured latency profile to a JSON file.
   - load_latency_profile: Reads latency profile from JSON and caches in memory.
   - clear_latency_cache: Clears in-memory profile cache for testing and reloading.
   - get_interpolated_cost_us: 1D linear interpolation of decision cost based on n_ready.
   - get_startup_cost_us: 1D linear interpolation of startup cost based on workload length.
   - plot_latency_profile: Generates publication-quality figure of latency vs n_ready.
   - validate_live_vs_table: End-to-end simulation sanity check comparing live vs table totals.
4. Final output of the file:
   Exports profiling, interpolation, persistence, plotting, and validation APIs for Stage 17.
"""

import gc
import json
import logging
import os
import platform
import sys
import time
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Enforce single-threaded execution for deterministic micro-benchmarks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import torch

    torch.set_num_threads(1)
except ImportError:
    pass

import config
from schedsim.domain.engine import ProcView, View, run
from schedsim.domain.process import Process
from schedsim.policies.base import Policy

logger = logging.getLogger(__name__)

_CACHED_PROFILE: Optional[dict[str, Any]] = None


class _LiveTimingProxy(Policy):
    """
    1. The use of class:
       Wraps a target Policy to measure live nanoseconds elapsed during on_start() and select().
    2. What the class contains:
       - _target: Underlying Policy instance being evaluated.
       - live_startup_us: Measured live upfront duration in microseconds.
       - live_decision_us_total: Measured cumulative live decision duration in microseconds.
       - non_forced_decisions: Integer count of non-forced decisions observed.
       - __init__: Initializes proxy wrapping target policy.
       - on_start: Measures live latency of target on_start() and returns target startup cost.
       - decision_cost_us: Delegates directly to target.decision_cost_us(view).
       - select: Measures live latency of target.select(view) on non-forced decisions.
       - reset: Resets timing accumulators and delegates to target.reset().
    3. Technologies used and why:
       - Policy inheritance and delegation pattern for seamless Engine compatibility.
       - time.perf_counter_ns() for high-precision monotonic timing.
    4. Key design and overall design of the class:
       Transparently intercepts simulation lifecycle hooks to measure live execution latency
       without mutating target policy behavior or simulation mechanics.
    """

    def __init__(self, target_policy: Policy) -> None:
        """
        1. The use of function:
           Initializes a _LiveTimingProxy wrapping an underlying Policy.
        2. How does the specific function work:
           Configures proxy policy name and stores target reference and timing accumulators.
        3. Concepts used in the function:
           Constructor delegation, state initialization.
        4. Places where this function is imported and used:
           Used internally by validate_live_vs_table().
        5. Final output of the function, before and after:
           Before: Uninitialized _LiveTimingProxy.
           After: Initialized proxy ready to wrap simulation execution.
        """
        super().__init__(name=f"LiveProxy_{target_policy.name}")
        self._target: Policy = target_policy
        self.live_startup_us: float = 0.0
        self.live_decision_us_total: float = 0.0
        self.non_forced_decisions: int = 0

    def on_start(self, processes: Sequence[Process]) -> float:
        """
        1. The use of function:
           Measures live latency of target on_start() while returning target startup cost.
        2. How does the specific function work:
           Times target.on_start(processes) with time.perf_counter_ns(), converts duration to
           microseconds, records in live_startup_us, and returns the target's reported cost.
        3. Concepts used in the function:
           High-resolution benchmarking, transparent delegation.
        4. Places where this function is imported and used:
           Invoked by Engine runner at workload start.
        5. Final output of the function, before and after:
           Before: Unrecorded live startup time.
           After: live_startup_us recorded; returns table-based startup cost float.
        """
        t0 = time.perf_counter_ns()
        cost = self._target.on_start(processes)
        t1 = time.perf_counter_ns()
        self.live_startup_us = (t1 - t0) / 1000.0
        return float(cost)

    def decision_cost_us(self, view: View) -> float:
        """
        1. The use of function:
           Returns the target policy's table-based decision cost in microseconds.
        2. How does the specific function work:
           Delegates directly to self._target.decision_cost_us(view).
        3. Concepts used in the function:
           Delegation pattern.
        4. Places where this function is imported and used:
           Invoked by Engine runner on non-forced scheduling steps.
        5. Final output of the function, before and after:
           Before: Observable simulation View.
           After: Returns float decision cost from target policy.
        """
        return float(self._target.decision_cost_us(view))

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Executes target.select(view) and times non-forced decisions.
        2. How does the specific function work:
           Determines whether decision is non-forced (len(view.ready) > 1). If non-forced,
           measures elapsed nanoseconds, converts to microseconds, accumulates into
           live_decision_us_total, and increments non_forced_decisions. Returns chosen PID.
        3. Concepts used in the function:
           High-resolution timing, conditional latency tracking.
        4. Places where this function is imported and used:
           Invoked on every simulation step by Engine.step().
        5. Final output of the function, before and after:
           Before: Untimed state.
           After: Elapsed time added to live_decision_us_total; returns selected PID string.
        """
        is_non_forced = len(view.ready) > 1
        if is_non_forced:
            t0 = time.perf_counter_ns()
            pid = self._target.select(view)
            t1 = time.perf_counter_ns()
            self.live_decision_us_total += (t1 - t0) / 1000.0
            self.non_forced_decisions += 1
            return pid
        return self._target.select(view)

    def reset(self) -> None:
        """
        1. The use of function:
           Resets proxy timing accumulators and target policy internal state.
        2. How does the specific function work:
           Sets live_startup_us to 0.0, live_decision_us_total to 0.0, non_forced_decisions to 0,
           and calls self._target.reset().
        3. Concepts used in the function:
           Lifecycle management, accumulator reset.
        4. Places where this function is imported and used:
           Called before running new simulation episodes.
        5. Final output of the function, before and after:
           Before: Historical timing accumulators.
           After: Zeroed accumulators and reset target policy state.
        """
        self.live_startup_us = 0.0
        self.live_decision_us_total = 0.0
        self.non_forced_decisions = 0
        self._target.reset()


def get_machine_info() -> dict[str, str]:
    """
    1. The use of function:
       Gathers host hardware, operating system, and Python environment metadata.
    2. How does the specific function work:
       Queries platform and sys modules for processor architecture, OS system name, OS release,
       Python version, and UTC timestamp, returning a structured metadata dictionary.
    3. Concepts used in the function:
       System introspection, hardware reproducibility logging.
    4. Places where this function is imported and used:
       Called when constructing latency profile JSON files and CLI tabular summaries.
    5. Final output of the function, before and after:
       Before: Host environment not documented.
       After: Returns dictionary with 'cpu', 'os', 'os_release', 'python_version', 'timestamp'.
    """
    proc = platform.processor() or platform.machine() or "Unknown CPU"
    return {
        "cpu": str(proc).strip(),
        "os": str(platform.system()).strip(),
        "os_release": str(platform.release()).strip(),
        "python_version": str(sys.version).split()[0],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def create_synthetic_view(n_ready: int) -> View:
    """
    1. The use of function:
       Constructs an observable View instance with n_ready valid ProcView candidates.
    2. How does the specific function work:
       Generates n_ready ProcView instances with distinct PIDs ('P1', 'P2', ...), staggered
       bursts, remaining times, and waiting times. Bundles them into an immutable View tuple.
    3. Concepts used in the function:
       Domain view creation, synthetic benchmarking fixture.
    4. Places where this function is imported and used:
       Used by benchmark_policy_latencies to evaluate policies at arbitrary candidate counts.
    5. Final output of the function, before and after:
       Before: Integer n_ready count.
       After: Valid View instance containing exactly n_ready ProcView objects.
    """
    if n_ready < 1:
        raise ValueError(f"n_ready must be >= 1, got {n_ready}.")
    ready_procs = tuple(
        ProcView(
            pid=f"P{i + 1}",
            arrival=0,
            burst=10 + (i % 5),
            remaining=max(1, 10 - (i % 7)),
            waited=i * 2,
            last_run_end=None,
        )
        for i in range(n_ready)
    )
    return View(
        time=10,
        ready=ready_procs,
        last_pid=ready_procs[0].pid,
        newly_arrived=(),
        pending_arrivals=0,
    )


def benchmark_policy_latencies(
    n_warmup: int = 200,
    n_timed: int = 2000,
    n_repeats: int = 3,
    n_ready_sweep: tuple[int, ...] = (2, 3, 5, 8, 10),
    policies: Optional[dict[str, Policy]] = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    1. The use of function:
       Micro-benchmarks per-decision scheduling policies across ready queue sizes.
    2. How does the specific function work:
       For each policy and ready queue size, constructs a synthetic View. Performs warm-up calls,
       disables garbage collection, and times individual select() calls across repeats using
       time.perf_counter_ns(). Computes median of medians, p95, standard deviation, and CV.
       Logs a warning if the coefficient of variation exceeds 10%.
    3. Concepts used in the function:
       Micro-benchmarking, GC isolation, median-of-medians statistic, tail latency (p95).
    4. Places where this function is imported and used:
       Called by benchmark_all(), CLI bench-latency, and performance testing suites.
    5. Final output of the function, before and after:
       Before: Unmeasured policy dispatch latencies.
       After: Returns structured dictionary mapping policy names to ready sizes and metrics.
    """
    if policies is None:
        from schedsim.pidsel.gbdt_ranker import GBDTRankerPolicy
        from schedsim.pidsel.scorer_net import NeuralCandidatePolicy
        from schedsim.policies.fcfs import FCFSPolicy
        from schedsim.policies.round_robin import RoundRobinPolicy
        from schedsim.policies.sjf import SJFPolicy
        from schedsim.policies.srtf import SRTFPolicy
        from schedsim.rl.policy import RLPolicy

        policies = {
            "FCFS": FCFSPolicy(),
            "SJF": SJFPolicy(),
            "RR": RoundRobinPolicy(quantum=config.TIME_QUANTUM),
            "SRTF": SRTFPolicy(),
            "Q-learning": RLPolicy.from_setting("S1"),
            "GBDT": GBDTRankerPolicy.from_setting("S1"),
            "Neural": NeuralCandidatePolicy.from_setting("S1"),
        }

    results: dict[str, dict[str, dict[str, float]]] = {}

    for pol_name, policy in policies.items():
        results[pol_name] = {}
        for n_ready in n_ready_sweep:
            view = create_synthetic_view(n_ready)
            policy.reset()

            # Warm-up phase
            for _ in range(n_warmup):
                policy.select(view)

            # Timed repeats with GC disabled
            repeats_medians: list[float] = []
            repeats_p95: list[float] = []

            for _ in range(n_repeats):
                policy.reset()
                timings_ns: list[int] = []
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    for _ in range(n_timed):
                        t0 = time.perf_counter_ns()
                        policy.select(view)
                        t1 = time.perf_counter_ns()
                        timings_ns.append(t1 - t0)
                finally:
                    if gc_was_enabled:
                        gc.enable()

                arr_us = np.asarray(timings_ns, dtype=np.float64) / 1000.0
                repeats_medians.append(float(np.median(arr_us)))
                repeats_p95.append(float(np.percentile(arr_us, 95.0)))

            median_us = float(np.median(repeats_medians))
            p95_us = float(np.median(repeats_p95))
            mean_med = float(np.mean(repeats_medians))
            std_med = float(np.std(repeats_medians))
            cv = std_med / max(mean_med, 1e-9)

            if cv > 0.10:
                logger.warning(
                    f"Latency variation for {pol_name} (n={n_ready}): "
                    f"CV={cv:.1%} exceeds 10% threshold across repeats."
                )

            results[pol_name][str(n_ready)] = {
                "median_us": round(median_us, 4),
                "p95_us": round(p95_us, 4),
                "cv": round(cv, 4),
            }

    return results


def benchmark_startup_latencies(
    workload_sizes: tuple[int, ...] = (3, 5, 8, 10, 15, 20),
    n_warmup: int = 50,
    n_timed: int = 500,
    n_repeats: int = 3,
    models: tuple[str, ...] = ("DT", "RF"),
) -> dict[str, dict[str, dict[str, float]]]:
    """
    1. The use of function:
       Micro-benchmarks Category 1 upfront feature extraction and prediction latencies.
    2. How does the specific function work:
       For each model ('DT', 'RF') and workload size, constructs a list of Process domain entities.
       Executes warm-up calls to recommend(), then times the full startup path (feature extraction
       plus tree inference) across repeats with GC disabled. Computes median and p95 in µs.
    3. Concepts used in the function:
       Category 1 startup benchmarking, feature extraction overhead, ensemble inference time.
    4. Places where this function is imported and used:
       Called by benchmark_all() and CLI bench-latency.
    5. Final output of the function, before and after:
       Before: Unmeasured Category 1 startup overheads.
       After: Returns structured dictionary mapping model names to workload sizes and latencies.
    """
    from schedsim.ml.evaluate_tree import recommend

    results: dict[str, dict[str, dict[str, float]]] = {}

    for model_name in models:
        norm_key = "dt" if model_name.upper().startswith("D") else "rf"
        results[model_name] = {}

        for size in workload_sizes:
            procs = [
                Process(pid=f"P{i + 1}", arrival=i * 2, burst=3 + (i % 7))
                for i in range(size)
            ]

            # Warm-up phase
            for _ in range(n_warmup):
                recommend(procs, model=norm_key, setting="S1")

            # Timed repeats with GC disabled
            repeats_medians: list[float] = []
            repeats_p95: list[float] = []

            for _ in range(n_repeats):
                timings_ns: list[int] = []
                gc_was_enabled = gc.isenabled()
                gc.disable()
                try:
                    for _ in range(n_timed):
                        t0 = time.perf_counter_ns()
                        recommend(procs, model=norm_key, setting="S1")
                        t1 = time.perf_counter_ns()
                        timings_ns.append(t1 - t0)
                finally:
                    if gc_was_enabled:
                        gc.enable()

                arr_us = np.asarray(timings_ns, dtype=np.float64) / 1000.0
                repeats_medians.append(float(np.median(arr_us)))
                repeats_p95.append(float(np.percentile(arr_us, 95.0)))

            median_us = float(np.median(repeats_medians))
            p95_us = float(np.median(repeats_p95))

            results[model_name][str(size)] = {
                "median_us": round(median_us, 4),
                "p95_us": round(p95_us, 4),
            }

    return results


def benchmark_all(
    n_warmup_decision: int = 200,
    n_timed_decision: int = 2000,
    n_warmup_startup: int = 50,
    n_timed_startup: int = 500,
    n_repeats: int = 3,
    n_ready_sweep: tuple[int, ...] = (2, 3, 5, 8, 10),
    workload_sizes: tuple[int, ...] = (3, 5, 8, 10, 15, 20),
) -> dict[str, Any]:
    """
    1. The use of function:
       Runs the complete micro-benchmark suite covering both per-decision and startup models.
    2. How does the specific function work:
       Captures machine info, benchmarks per-decision policies via benchmark_policy_latencies(),
       benchmarks startup models via benchmark_startup_latencies(), and packages into profile dict.
    3. Concepts used in the function:
       Unified benchmarking pipeline, composition of performance profiles.
    4. Places where this function is imported and used:
       Called by CLI bench-latency and full evaluation runners.
    5. Final output of the function, before and after:
       Before: Unprofiled simulator environment.
       After: Complete latency profile dictionary with 'machine_info', 'per_decision', 'startup'.
    """
    info = get_machine_info()
    per_dec = benchmark_policy_latencies(
        n_warmup=n_warmup_decision,
        n_timed=n_timed_decision,
        n_repeats=n_repeats,
        n_ready_sweep=n_ready_sweep,
    )
    startup = benchmark_startup_latencies(
        workload_sizes=workload_sizes,
        n_warmup=n_warmup_startup,
        n_timed=n_timed_startup,
        n_repeats=n_repeats,
    )
    return {
        "machine_info": info,
        "per_decision": per_dec,
        "startup": startup,
    }


def save_latency_profile(
    profile: dict[str, Any],
    path: Path | str = config.MODELS_DIR / "latency_profile.json",
) -> None:
    """
    1. The use of function:
       Saves the structured latency profile dictionary to a formatted JSON file.
    2. How does the specific function work:
       Creates parent directories if necessary, writes JSON with indent=2, and updates
       the module-level _CACHED_PROFILE in-memory cache.
    3. Concepts used in the function:
       JSON persistence, atomic cache update.
    4. Places where this function is imported and used:
       Called by benchmark runners, CLI bench-latency, and test fixtures.
    5. Final output of the function, before and after:
       Before: Profile in memory only.
       After: Profile written to disk and updated in _CACHED_PROFILE.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    global _CACHED_PROFILE
    _CACHED_PROFILE = profile


def load_latency_profile(
    path: Path | str = config.MODELS_DIR / "latency_profile.json",
    reload: bool = False,
) -> dict[str, Any]:
    """
    1. The use of function:
       Loads latency profile data from disk with memoized in-memory caching.
    2. How does the specific function work:
       If reload is False and _CACHED_PROFILE is set, returns cached profile immediately.
       Otherwise reads JSON from path, validates structure, updates cache, and returns it.
    3. Concepts used in the function:
       Memoization, lazy loading, file persistence.
    4. Places where this function is imported and used:
       Called by get_interpolated_cost_us, get_startup_cost_us, and CLI commands.
    5. Final output of the function, before and after:
       Before: Cached state or unread file.
       After: Returns loaded profile dictionary.
    """
    global _CACHED_PROFILE
    if not reload and _CACHED_PROFILE is not None:
        return _CACHED_PROFILE

    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Latency profile not found at {target}.")

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)

    _CACHED_PROFILE = data
    return data


def clear_latency_cache() -> None:
    """
    1. The use of function:
       Clears the module-level in-memory latency profile cache.
    2. How does the specific function work:
       Sets _CACHED_PROFILE to None.
    3. Concepts used in the function:
       Cache invalidation, test isolation.
    4. Places where this function is imported and used:
       Called by unit test fixtures to test cold loads and custom profile files.
    5. Final output of the function, before and after:
       Before: _CACHED_PROFILE holds dictionary or None.
       After: _CACHED_PROFILE is strictly None.
    """
    global _CACHED_PROFILE
    _CACHED_PROFILE = None


def _normalize_decision_policy_name(name: str) -> str:
    """
    1. The use of function:
       Normalizes diverse policy name representations to canonical profile keys.
    2. How does the specific function work:
       Strips whitespace and maps case-insensitive substrings/aliases to 'FCFS', 'SJF',
       'RR', 'SRTF', 'Q-learning', 'GBDT', or 'Neural'.
    3. Concepts used in the function:
       String normalization, canonical alias mapping.
    4. Places where this function is imported and used:
       Used internally by get_interpolated_cost_us.
    5. Final output of the function, before and after:
       Before: Raw policy name string (e.g. 'rlpolicy_cs1', 'PID_NN').
       After: Canonical profile key string (e.g. 'Q-learning', 'Neural').
    """
    key = name.upper().strip()
    if key in ("FCFS", "FCFSPOLICY"):
        return "FCFS"
    if key in ("SJF", "SJFPOLICY"):
        return "SJF"
    if key in ("RR", "ROUNDROBIN", "ROUNDROBINPOLICY") or key.startswith("RR("):
        return "RR"
    if key in ("SRTF", "SRTFPOLICY"):
        return "SRTF"
    if "Q-LEARNING" in key or "QLEARNING" in key or "RL" in key:
        return "Q-learning"
    if "GBDT" in key:
        return "GBDT"
    if "NEURAL" in key or "PID_NN" in key or "SCORER" in key:
        return "Neural"
    return name


def _normalize_startup_policy_name(name: str) -> str:
    """
    1. The use of function:
       Normalizes Category 1 model names to canonical profile keys ('DT' or 'RF').
    2. How does the specific function work:
       Strips whitespace and maps variants of decision tree to 'DT' and random forest to 'RF'.
    3. Concepts used in the function:
       String normalization, canonical alias mapping.
    4. Places where this function is imported and used:
       Used internally by get_startup_cost_us.
    5. Final output of the function, before and after:
       Before: Raw model name string (e.g. 'dt', 'RandomForest', 'Selector_RF_S1').
       After: Canonical profile key string ('DT' or 'RF').
    """
    key = name.upper().strip()
    if "DT" in key or "TREE" in key:
        return "DT"
    if "RF" in key or "FOREST" in key:
        return "RF"
    return name


def get_interpolated_cost_us(
    policy_name: str,
    n_ready: int,
    profile: Optional[dict[str, Any]] = None,
    use_p95: bool = False,
) -> float:
    """
    1. The use of function:
       Performs 1D linear interpolation of per-decision policy latency based on ready queue size.
    2. How does the specific function work:
       Returns 0.0 if n_ready <= 0. Resolves profile from cache/disk if omitted. Normalizes policy
       name and extracts measured (x, y) coordinates. If n_ready <= min_x, returns cost at min_x.
       If n_ready >= max_x, returns cost at max_x and flags extrapolation. Otherwise computes 1D
       linear interpolation between the enclosing bracket points.
    3. Concepts used in the function:
       1D linear interpolation, boundary clamping, safe fallback handling.
    4. Places where this function is imported and used:
       Called by decision_cost_us() across all per-decision policy implementations.
    5. Final output of the function, before and after:
       Before: Policy name string and ready queue size int.
       After: Returns interpolated non-negative float latency in microseconds.
    """
    if n_ready <= 0:
        return 0.0

    if profile is None:
        try:
            profile = load_latency_profile()
        except Exception:
            return 0.0

    per_dec = profile.get("per_decision", {})
    norm_name = _normalize_decision_policy_name(policy_name)

    points_dict = per_dec.get(norm_name) or per_dec.get(policy_name)
    if not points_dict:
        return 0.0

    metric_key = "p95_us" if use_p95 else "median_us"
    xs = sorted(int(k) for k in points_dict.keys())
    if not xs:
        return 0.0
    ys = [float(points_dict[str(x)][metric_key]) for x in xs]

    # Lower boundary clamping
    if n_ready <= xs[0]:
        return float(ys[0])

    # Upper boundary clamping with extrapolation warning
    if n_ready >= xs[-1]:
        if n_ready > xs[-1]:
            warnings.warn(
                f"Extrapolating latency for {norm_name}: n_ready={n_ready} > max={xs[-1]}",
                UserWarning,
                stacklevel=2,
            )
        return float(ys[-1])

    # 1D Linear interpolation
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= n_ready <= x1:
            if x0 == x1:
                return float(ys[i])
            alpha = (n_ready - x0) / (x1 - x0)
            return float(ys[i] + alpha * (ys[i + 1] - ys[i]))

    return float(ys[-1])


def get_startup_cost_us(
    policy_name: str,
    n_processes: int,
    profile: Optional[dict[str, Any]] = None,
    use_p95: bool = False,
) -> float:
    """
    1. The use of function:
       Performs 1D linear interpolation of Category 1 startup latency based on workload size.
    2. How does the specific function work:
       Returns 0.0 if n_processes <= 0. Resolves profile from cache/disk if omitted. Normalizes
       model name ('DT' or 'RF') and extracts measured coordinates. Clamps at boundaries and
       interpolates linearly between adjacent workload size brackets.
    3. Concepts used in the function:
       1D linear interpolation, boundary clamping, amortized upfront model cost lookup.
    4. Places where this function is imported and used:
       Called by SelectorPolicy.on_start() to charge upfront feature+inference latency.
    5. Final output of the function, before and after:
       Before: Model name string and workload process count int.
       After: Returns interpolated non-negative float startup latency in microseconds.
    """
    if n_processes <= 0:
        return 0.0

    if profile is None:
        try:
            profile = load_latency_profile()
        except Exception:
            return 0.0

    startup_dec = profile.get("startup", {})
    norm_name = _normalize_startup_policy_name(policy_name)

    points_dict = startup_dec.get(norm_name) or startup_dec.get(policy_name)
    if not points_dict:
        return 0.0

    metric_key = "p95_us" if use_p95 else "median_us"
    xs = sorted(int(k) for k in points_dict.keys())
    if not xs:
        return 0.0
    ys = [float(points_dict[str(x)][metric_key]) for x in xs]

    if n_processes <= xs[0]:
        return float(ys[0])
    if n_processes >= xs[-1]:
        return float(ys[-1])

    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        if x0 <= n_processes <= x1:
            if x0 == x1:
                return float(ys[i])
            alpha = (n_processes - x0) / (x1 - x0)
            return float(ys[i] + alpha * (ys[i + 1] - ys[i]))

    return float(ys[-1])


def plot_latency_profile(
    profile: dict[str, Any],
    output_path: Path | str = config.RESULTS_DIR / "figures" / "latency_vs_ready.png",
) -> None:
    """
    1. The use of function:
       Renders publication-quality latency-vs-n_ready comparison curves for all policies.
    2. How does the specific function work:
       Uses Matplotlib (Agg backend) to plot median latency in microseconds on a log scale
       against ready queue sizes (2, 3, 5, 8, 10). Adds distinct markers, grid, styling,
       and saves the figure at 300 DPI to output_path.
    3. Concepts used in the function:
       Data visualization, logarithmic scaling, publication-grade figure generation.
    4. Places where this function is imported and used:
       Called by CLI bench-latency and evaluation scripts.
    5. Final output of the function, before and after:
       Before: Raw latency dictionary.
       After: Figure saved at output_path with clean styling and legends.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    per_dec = profile.get("per_decision", {})
    if not per_dec:
        logger.warning("No per_decision entries in profile to plot.")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    styles = {
        "FCFS": ("#2b5c8f", "o-"),
        "SJF": ("#4682b4", "s-"),
        "RR": ("#20b2aa", "^-"),
        "SRTF": ("#3cb371", "d-"),
        "Q-learning": ("#e69138", "v--"),
        "GBDT": ("#cc0000", "p-"),
        "Neural": ("#674ea7", "*-"),
    }

    for name, points in per_dec.items():
        xs = sorted(int(k) for k in points.keys())
        ys = [points[str(x)]["median_us"] for x in xs]
        color, marker = styles.get(name, ("#333333", "o-"))
        ax.plot(
            xs,
            ys,
            marker,
            color=color,
            linewidth=2.0,
            markersize=7,
            label=name,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Ready Queue Size ($n_{\\text{ready}}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Decision Latency ($\\mu$s, log scale)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Scheduling Policy Decision Latency vs. Ready Queue Size",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(title="Policy", framealpha=0.9, loc="upper left")

    plt.tight_layout()
    fig.savefig(target, dpi=300)
    plt.close(fig)
    logger.info(f"Saved latency profile plot to {target}")


def validate_live_vs_table(
    policy: Policy,
    processes: Sequence[Process],
    cs_cost: int = 1,
    tolerance: float = 0.15,
    n_repeats: int = 3,
) -> dict[str, Any]:
    """
    1. The use of function:
       Runs an end-to-end simulation sanity check comparing live timing vs table-based costs.
    2. How does the specific function work:
       Wraps the policy in a _LiveTimingProxy. Runs warmup episodes, then executes n_repeats
       measured simulation runs through run() with tick_us=1000. Collects median live AI latency
       and compares against the deterministic table-based inference_us_total. Computes absolute
       and relative percentage gap, checks tolerance, and logs findings.
    3. Concepts used in the function:
       Live vs model validation, simulation instrumentation, regression verification.
    4. Places where this function is imported and used:
       Called by CLI bench-latency and test_stage17_overhead.py.
    5. Final output of the function, before and after:
       Before: Model accuracy vs live hardware unverified.
       After: Returns validation dict reporting live_us_total, table_us_total, and relative_gap.
    """
    proxy = _LiveTimingProxy(policy)
    proc_list = list(processes)

    # Warmup runs
    for _ in range(2):
        proxy.reset()
        warm_procs = [Process(p.pid, p.arrival, p.burst, p.priority) for p in proc_list]
        run(proxy, warm_procs, cs_cost=cs_cost, tick_us=1000)

    # Measured runs
    live_totals: list[float] = []
    table_us_total: float = 0.0

    for _ in range(n_repeats):
        proxy.reset()
        fresh_procs = [Process(p.pid, p.arrival, p.burst, p.priority) for p in proc_list]
        res = run(proxy, fresh_procs, cs_cost=cs_cost, tick_us=1000)
        table_us_total = float(res.simulation_metrics.inference_us_total)
        live_run_total = proxy.live_startup_us + proxy.live_decision_us_total
        live_totals.append(live_run_total)

    median_live_us = float(np.median(live_totals))
    abs_gap_us = abs(median_live_us - table_us_total)
    rel_gap = abs_gap_us / max(table_us_total, 1e-6)
    within_tol = rel_gap <= tolerance

    logger.info(
        f"Live vs Table validation for {policy.name}: "
        f"live={median_live_us:.2f} µs, table={table_us_total:.2f} µs, "
        f"gap={rel_gap:.2%} (tolerance={tolerance:.0%})"
    )

    return {
        "policy": policy.name,
        "live_us_total": round(median_live_us, 4),
        "table_us_total": round(table_us_total, 4),
        "abs_gap_us": round(abs_gap_us, 4),
        "relative_gap": round(rel_gap, 6),
        "percentage_gap": round(rel_gap * 100.0, 2),
        "tolerance": tolerance,
        "within_tolerance": within_tol,
        "decisions": proxy.non_forced_decisions,
    }


__all__ = [
    "benchmark_all",
    "benchmark_policy_latencies",
    "benchmark_startup_latencies",
    "clear_latency_cache",
    "create_synthetic_view",
    "get_interpolated_cost_us",
    "get_machine_info",
    "get_startup_cost_us",
    "load_latency_profile",
    "plot_latency_profile",
    "save_latency_profile",
    "validate_live_vs_table",
]
