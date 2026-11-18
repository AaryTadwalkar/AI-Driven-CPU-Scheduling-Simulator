"""
1. What this file does:
   Executes rigorous, multi-policy benchmark experiments across held-out dataset splits,
   evaluating classical, machine learning, and reinforcement learning scheduling algorithms.
2. Overall design:
   Loads held-out workloads (defaulting to the 1,000-workload Test split), runs FCFS, SJF, RR(2),
   SRTF, tabular Q-learning (cost-specific checkpoints), and Decision Tree recommendation under
   both zero and unit context-switch penalties. Collects detailed per-workload performance metrics,
   fairness indices, composite objective scores, and starvation indicators, outputting a complete
   evaluation dataset to results/evaluation_results.csv.
3. Elements in the file:
   - EVAL_POLICIES: Tuple of evaluated policy identifiers.
   - run_workload_all_policies: Runs all candidate policies on a single workload at a given cs_cost.
   - run_evaluation_experiments: Orchestrates full multi-workload, multi-cost benchmark experiments.
4. Final output of the file:
   Exports run_evaluation_experiments and run_workload_all_policies, writing experimental results
   to CSV for statistical aggregation and visualization.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

import config
from schedsim.domain.engine import SimResult, run
from schedsim.domain.process import Process
from schedsim.ml.evaluate_tree import recommend
from schedsim.policies.base import Policy
from schedsim.policies.registry import get_policy
from schedsim.rl.agent import QLearningAgent
from schedsim.rl.policy import RLPolicy
from schedsim.workloads.io import Workload
from schedsim.workloads.splits import SPLIT_TEST, generate_split_workloads

EVAL_POLICIES: tuple[str, ...] = ("FCFS", "SJF", "RR", "SRTF", "RL", "DT")


def _instantiate_policy(
    policy_name: str,
    processes: Sequence[Process],
    rl_agent: QLearningAgent | None = None,
    dt_model_path: Path | str | None = None,
) -> Policy:
    """
    1. The use of function:
       Helper to instantiate a concrete Policy instance given policy identifier and context.
    2. How does the specific function work:
       Matches policy_name: for classical policies (FCFS, SJF, SRTF), queries the registry;
       for RR, passes configured quantum; for RL, wraps the provided QLearningAgent in RLPolicy;
       for DT, queries recommend() to predict the optimal algorithm, then instantiates it.
    3. Concepts used in the function:
       Polymorphic policy factory pattern, RL agent adapter binding, supervised recommendation.
    4. Places where this function is imported and used:
       Internal helper for run_workload_all_policies.
    5. Final output of the function, before and after:
       Before: String identifier, process list, optional agent and model path.
       After: Instantiated Policy object ready to execute in the unit-step Engine.
    """
    canonical = policy_name.strip().upper()
    if canonical == "FCFS":
        return get_policy("fcfs")
    if canonical == "SJF":
        return get_policy("sjf")
    if canonical == "RR":
        return get_policy("rr", quantum=config.TIME_QUANTUM)
    if canonical == "SRTF":
        return get_policy("srtf")
    if canonical == "RL":
        if rl_agent is None:
            raise ValueError("rl_agent must be provided when instantiating RL policy.")
        return RLPolicy(agent=rl_agent, processes=processes, name="RLPolicy")
    if canonical == "DT":
        m_path = dt_model_path or (config.MODELS_DIR / "decision_tree.joblib")
        rec = recommend(processes=processes, model_path=m_path)
        chosen = rec["algorithm"].strip().lower()
        if chosen == "rr":
            return get_policy("rr", quantum=config.TIME_QUANTUM)
        return get_policy(chosen)
    raise ValueError(f"Unknown evaluation policy identifier: {policy_name}")


def run_workload_all_policies(
    workload: Workload,
    cs_cost: int,
    rl_agent: QLearningAgent | None = None,
    dt_model_path: Path | str | None = None,
    policies: Sequence[str] = EVAL_POLICIES,
) -> list[dict[str, Any]]:
    """
    1. The use of function:
       Executes all designated scheduling policies on a single workload at a specified cs_cost.
    2. How does the specific function work:
       Iterates over the policy names, instantiates each policy via _instantiate_policy, executes
       the unit-step simulation via run(), extracts process and system metrics, calculates the
       composite score using standard weights, flags starvation condition, and packages results.
    3. Concepts used in the function:
       Deterministic simulation execution, metric aggregation, starvation detection, composite cost.
    4. Places where this function is imported and used:
       Imported by evaluation runner, test suites, and batch processing utilities.
    5. Final output of the function, before and after:
       Before: Workload definition, context switch cost, and policy references.
       After: List of dictionaries, each recording full metrics for one policy on the workload.
    """
    records: list[dict[str, Any]] = []
    w_wait = float(config.LABEL_WEIGHTS.get("wait", 0.5))
    w_resp = float(config.LABEL_WEIGHTS.get("resp", 0.3))
    w_tat = float(config.LABEL_WEIGHTS.get("tat", 0.2))

    for pol_name in policies:
        policy = _instantiate_policy(
            policy_name=pol_name,
            processes=workload.processes,
            rl_agent=rl_agent,
            dt_model_path=dt_model_path,
        )
        sim_res: SimResult = run(policy, workload.processes, cs_cost=cs_cost)
        sm = sim_res.simulation_metrics

        comp_score = (
            w_wait * sm.avg_waiting + w_resp * sm.avg_response + w_tat * sm.avg_turnaround
        )
        is_starved = sm.max_waiting > 3.0 * sm.avg_waiting if sm.avg_waiting > 0 else False

        rec = {
            "workload_id": workload.workload_id,
            "scenario": workload.scenario,
            "seed": workload.seed,
            "n_processes": len(workload.processes),
            "cs_cost": cs_cost,
            "policy": pol_name,
            "avg_waiting": float(sm.avg_waiting),
            "avg_turnaround": float(sm.avg_turnaround),
            "avg_response": float(sm.avg_response),
            "max_waiting": int(sm.max_waiting),
            "makespan": int(sm.makespan),
            "cpu_utilization": float(sm.cpu_utilization),
            "overhead_ticks": int(sm.overhead_ticks),
            "throughput": float(sm.throughput),
            "context_switches": int(sm.context_switches),
            "jain_fairness": float(sm.jain_fairness),
            "composite_score": float(round(comp_score, 4)),
            "starvation": bool(is_starved),
        }
        records.append(rec)

    return records


def _run_chunk(
    workloads: list[Workload],
    cs_cost: int,
    agent_path: str,
    dt_path: str,
    policies: tuple[str, ...],
) -> list[dict[str, Any]]:
    """
    1. The use of function:
       Worker execution function evaluating a chunk of workloads in a child process.
    2. How does the specific function work:
       Loads the Q-learning agent from agent_path, iterates through the workload batch,
       and collects records by invoking run_workload_all_policies.
    3. Concepts used in the function:
       Multiprocessing worker isolation, checkpoint deserialization.
    4. Places where this function is imported and used:
       Internal helper invoked by run_evaluation_experiments via ProcessPoolExecutor.
    5. Final output of the function, before and after:
       Before: Workload sub-list and model paths.
       After: Aggregated list of metric records for all workloads in the chunk.
    """
    agent = QLearningAgent.load(agent_path) if "RL" in policies else None
    results: list[dict[str, Any]] = []
    for wl in workloads:
        chunk_res = run_workload_all_policies(
            workload=wl,
            cs_cost=cs_cost,
            rl_agent=agent,
            dt_model_path=dt_path,
            policies=policies,
        )
        results.extend(chunk_res)
    return results


def run_evaluation_experiments(
    split: str = SPLIT_TEST,
    n_workloads: int = 1000,
    cs_costs: Sequence[int] = (0, 1),
    policies: Sequence[str] = EVAL_POLICIES,
    output_path: Path | str | None = None,
    q_table_cs0_path: Path | str | None = None,
    q_table_cs1_path: Path | str | None = None,
    dt_model_path: Path | str | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    1. The use of function:
       Executes complete multi-policy evaluation across a split and returns a pandas DataFrame.
    2. How does the specific function work:
       Generates split workloads (defaulting to test split), resolves model file paths, divides
       workloads into parallel chunks across available CPU workers, collects execution rows for
       all context switch costs, constructs a DataFrame, optionally persists to CSV, and returns.
    3. Concepts used in the function:
       Parallel evaluation, held-out test partitioning, experimental reproducibility, CSV I/O.
    4. Places where this function is imported and used:
       Imported by cli.py (evaluate command), acceptance tests, and notebook analysis pipelines.
    5. Final output of the function, before and after:
       Before: Benchmark specifications (split, workload count, cs_costs).
       After: Complete evaluation DataFrame and saved results/evaluation_results.csv file.
    """
    out_file = (
        Path(output_path)
        if output_path is not None
        else (config.RESULTS_DIR / "evaluation_results.csv")
    )
    p_cs0 = (
        Path(q_table_cs0_path)
        if q_table_cs0_path is not None
        else (config.MODELS_DIR / "q_table_cs0.pkl")
    )
    p_cs1 = (
        Path(q_table_cs1_path)
        if q_table_cs1_path is not None
        else (config.MODELS_DIR / "q_table_cs1.pkl")
    )
    p_dt = (
        Path(dt_model_path)
        if dt_model_path is not None
        else (config.MODELS_DIR / "decision_tree.joblib")
    )

    workloads = generate_split_workloads(split=split, n_workloads=n_workloads)
    policy_tuple = tuple(policies)

    workers = n_workers if n_workers is not None else min(os.cpu_count() or 1, 8)
    all_records: list[dict[str, Any]] = []

    for cost in cs_costs:
        agent_file = str(p_cs0 if cost == 0 else p_cs1)
        if workers > 1 and len(workloads) >= 20:
            chunk_size = max(1, len(workloads) // workers)
            chunks = [
                workloads[i : i + chunk_size] for i in range(0, len(workloads), chunk_size)
            ]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _run_chunk,
                        chunk,
                        cost,
                        agent_file,
                        str(p_dt),
                        policy_tuple,
                    )
                    for chunk in chunks
                ]
                for fut in futures:
                    all_records.extend(fut.result())
        else:
            agent = QLearningAgent.load(agent_file) if "RL" in policy_tuple else None
            for wl in workloads:
                recs = run_workload_all_policies(
                    workload=wl,
                    cs_cost=cost,
                    rl_agent=agent,
                    dt_model_path=p_dt,
                    policies=policy_tuple,
                )
                all_records.extend(recs)

    df = pd.DataFrame(all_records)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    return df


__all__ = [
    "EVAL_POLICIES",
    "run_evaluation_experiments",
    "run_workload_all_policies",
]
