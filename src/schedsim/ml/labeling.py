"""
1. What this file does:
   Implements multi-metric composite labeling for CPU scheduling workloads and provides the
   mandatory Stage 4c pilot study framework comparing candidate policies (FCFS, SJF, RR).
2. Overall design:
   Runs candidate policies across a single unified unit-step engine with configurable context switch
   penalties and time quanta. Computes composite scores normalizing waiting time, response time, and
   turnaround time against best-in-workload values. Determines the winning policy (label), relative
   optimality margin, ambiguity flags, and pilot study class distributions across diverse settings.
3. Elements in the file:
   - CANDIDATE_POLICIES: Standard tuple of algorithms evaluated for labeling ("FCFS", "SJF", "RR").
   - LabelResult: Frozen dataclass encapsulating label, scores, margin, ambiguity, and metrics.
   - compute_workload_label: Evaluates candidate policies and computes composite objective score.
   - run_pilot_study: Evaluates pilot workloads across configurations to guide parameter choice.
4. Final output of the file:
   Exports CANDIDATE_POLICIES, LabelResult, compute_workload_label, and run_pilot_study for dataset
   construction, pilot analysis, and supervised recommender training.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import config
from schedsim.domain.engine import run
from schedsim.domain.process import Process, validate_processes
from schedsim.policies.registry import get_policy
from schedsim.workloads.io import Workload

CANDIDATE_POLICIES: tuple[str, ...] = ("FCFS", "SJF", "RR")


@dataclass(frozen=True)
class LabelResult:
    """
    1. The use of class:
       Encapsulates the outcome of multi-metric composite labeling for a single workload.
    2. What the class contains:
       - label: Winning policy name ("FCFS", "SJF", or "RR").
       - best_score: Minimum composite score achieved by the winning policy.
       - margin: Relative gap between the best and second-best composite score.
       - is_ambiguous: Boolean indicating whether margin is below tie tolerance threshold.
       - scores: Dictionary mapping each candidate policy to its composite score.
       - raw_metrics: Nested dictionary of average waiting, response, and turnaround metrics.
    3. Technologies used and why:
       Python frozen dataclass for immutable state persistence and type safety across ML stages.
    4. Key design and overall design of the class:
       Read-only data transfer object ensuring labeling reproducibility and downstream transparency.
    """

    label: str
    best_score: float
    margin: float
    is_ambiguous: bool
    scores: dict[str, float]
    raw_metrics: dict[str, dict[str, float]]


def compute_workload_label(
    processes: Sequence[Process],
    cs_cost: int = config.LABEL_CS_COST,
    quantum: int = config.TIME_QUANTUM,
    weights: Optional[dict[str, float]] = None,
    tie_tolerance: float = config.LABEL_TIE_TOLERANCE,
    eps: float = 1e-6,
) -> LabelResult:
    """
    1. The use of function:
       Computes the optimal policy label for a workload using multi-metric composite evaluation.
    2. How does the specific function work:
       Simulates candidate policies (FCFS, SJF, RR) on the workload using the unit-step engine.
       Measures average waiting, response, and turnaround times. Determines minimum baselines
       across algorithms, computes composite scores weighted by (wait, resp, tat), identifies
       the minimum score policy with deterministic tie-breaking, and calculates relative margin.
    3. Concepts used in the function:
       Multi-criteria decision making, normalization against Pareto baselines, margin thresholding.
    4. Places where this function is imported and used:
       Imported by dataset_builder.py, pilot study runner, test suites, and evaluation pipelines.
    5. Final output of the function, before and after:
       Before: Raw process sequence and execution hyperparameters.
       After: Validated immutable LabelResult with label, composite scores, margin, and ambiguity.
    """
    if not isinstance(processes, Sequence) or isinstance(processes, (str, bytes)):
        raise TypeError(
            f"processes must be a Sequence[Process], got {type(processes).__name__}."
        )
    proc_list = list(processes)
    validate_processes(proc_list)

    if weights is None:
        weights = config.LABEL_WEIGHTS

    w_wait = float(weights.get("wait", 0.5))
    w_resp = float(weights.get("resp", 0.3))
    w_tat = float(weights.get("tat", 0.2))

    raw_metrics: dict[str, dict[str, float]] = {}

    for policy_name in CANDIDATE_POLICIES:
        if policy_name == "RR":
            policy = get_policy("RR", quantum=quantum)
        else:
            policy = get_policy(policy_name)

        sim_res = run(policy, proc_list, cs_cost=cs_cost)
        sm = sim_res.simulation_metrics
        raw_metrics[policy_name] = {
            "avg_waiting": float(sm.avg_waiting),
            "avg_response": float(sm.avg_response),
            "avg_turnaround": float(sm.avg_turnaround),
        }

    best_wait = min(m["avg_waiting"] for m in raw_metrics.values())
    best_resp = min(m["avg_response"] for m in raw_metrics.values())
    best_tat = min(m["avg_turnaround"] for m in raw_metrics.values())

    scores: dict[str, float] = {}
    for policy_name in CANDIDATE_POLICIES:
        m = raw_metrics[policy_name]
        score = (
            w_wait * (m["avg_waiting"] / (best_wait + eps))
            + w_resp * (m["avg_response"] / (best_resp + eps))
            + w_tat * (m["avg_turnaround"] / (best_tat + eps))
        )
        scores[policy_name] = float(score)

    # Sort deterministically: ascending score, then alphabetical policy name
    sorted_scores = sorted(scores.items(), key=lambda item: (item[1], item[0]))
    best_algo, best_score = sorted_scores[0]
    _, second_score = sorted_scores[1]

    margin = float((second_score - best_score) / (best_score + eps))
    is_ambiguous = bool(margin < tie_tolerance)

    return LabelResult(
        label=best_algo,
        best_score=best_score,
        margin=margin,
        is_ambiguous=is_ambiguous,
        scores=scores,
        raw_metrics=raw_metrics,
    )


def run_pilot_study(
    workloads: Sequence[Any],
    configurations: Optional[list[dict[str, Any]]] = None,
) -> dict[str, dict[str, Any]]:
    """
    1. The use of function:
       Executes the mandatory Section 4c pilot gate study across candidate labeling configurations.
    2. How does the specific function work:
       Evaluates a representative sample of workloads (typically 600 pilot workloads) under
       principled variations of context switch costs, Round Robin quanta, and metric weights.
       Aggregates label counts, percentage distributions, and ambiguity frequencies per setting.
    3. Concepts used in the function:
       Hyperparameter sensitivity analysis, class imbalance diagnostics, pilot validation.
    4. Places where this function is imported and used:
       Invoked during Stage 4 pilot verification and documented in `docs/DECISIONS.md`.
    5. Final output of the function, before and after:
       Before: Sequence of workloads and candidate parameter settings.
       After: Structured report dictionary containing label distributions and ambiguity stats.
    """
    if configurations is None:
        configurations = [
            {
                "name": "cs0_q2_standard",
                "cs_cost": 0,
                "quantum": 2,
                "weights": {"wait": 0.5, "resp": 0.3, "tat": 0.2},
            },
            {
                "name": "cs1_q2_standard",
                "cs_cost": 1,
                "quantum": 2,
                "weights": {"wait": 0.5, "resp": 0.3, "tat": 0.2},
            },
            {
                "name": "cs1_q4_standard",
                "cs_cost": 1,
                "quantum": 4,
                "weights": {"wait": 0.5, "resp": 0.3, "tat": 0.2},
            },
            {
                "name": "cs1_q2_balanced_weights",
                "cs_cost": 1,
                "quantum": 2,
                "weights": {"wait": 0.333, "resp": 0.333, "tat": 0.334},
            },
        ]

    # Extract process lists if Workload instances are passed
    extracted_workloads: list[list[Process]] = []
    for item in workloads:
        if isinstance(item, Workload):
            extracted_workloads.append(list(item.processes))
        elif isinstance(item, Sequence):
            extracted_workloads.append(list(item))
        else:
            raise TypeError(f"Unsupported workload element type: {type(item).__name__}.")

    total_workloads = len(extracted_workloads)
    results: dict[str, dict[str, Any]] = {}

    for cfg in configurations:
        cfg_name = cfg["name"]
        cs_cost = cfg.get("cs_cost", config.LABEL_CS_COST)
        quantum = cfg.get("quantum", config.TIME_QUANTUM)
        weights = cfg.get("weights", config.LABEL_WEIGHTS)

        labels: list[str] = []
        ambiguous_count = 0

        for procs in extracted_workloads:
            lbl_res = compute_workload_label(
                processes=procs,
                cs_cost=cs_cost,
                quantum=quantum,
                weights=weights,
            )
            labels.append(lbl_res.label)
            if lbl_res.is_ambiguous:
                ambiguous_count += 1

        counts = Counter(labels)
        dist = {
            policy: {
                "count": counts.get(policy, 0),
                "percentage": (
                    float(counts.get(policy, 0) / total_workloads * 100.0)
                    if total_workloads > 0
                    else 0.0
                ),
            }
            for policy in CANDIDATE_POLICIES
        }

        results[cfg_name] = {
            "cs_cost": cs_cost,
            "quantum": quantum,
            "weights": weights,
            "total_workloads": total_workloads,
            "distribution": dist,
            "ambiguous_count": ambiguous_count,
            "ambiguous_percentage": (
                float(ambiguous_count / total_workloads * 100.0)
                if total_workloads > 0
                else 0.0
            ),
        }

    return results


__all__ = [
    "CANDIDATE_POLICIES",
    "LabelResult",
    "compute_workload_label",
    "run_pilot_study",
]
