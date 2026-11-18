"""
1. What this file does:
   Implements domain-level service logic for the FastAPI backend, bridging HTTP schemas to
   the unit-step simulation Engine, classical policies, Decision Tree, and Q-learning agents.
2. Overall design:
   Contains business logic for simulation execution, step playback tracing, RL decision logging,
   multi-policy comparative runs, and supervised recommendation. Enforces model availability
   checks, raising ModelMissingException (translated to HTTP 503) when required artifacts
   have not yet been trained.
3. Elements in the file:
   - ModelMissingException: Exception indicating an AI model file is absent on disk.
   - processes_from_schemas: Converts Pydantic ProcessSchema items to domain Process objects.
   - run_simulation_service: Drives unit-step simulation with ready queue and decision logging.
   - run_comparison_service: Benchmarks multiple policies on a workload.
   - run_recommend_service: Extracts features and queries Decision Tree recommendation.
   - generate_workload_service: Synthesizes synthetic workloads using deterministic RNG.
4. Final output of the file:
   Exports service functions and exceptions for FastAPI route handlers.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import config
from schedsim.api.schemas import (
    CompareResponse,
    DecisionTraceEntry,
    GenerateWorkloadResponse,
    ProcessMetricsSchema,
    ProcessSchema,
    ReadyQueueLogEntry,
    RecommendResponse,
    SimulateResponse,
    SimulationMetricsSchema,
    TimelineSegmentSchema,
)
from schedsim.domain.engine import Engine, SimResult
from schedsim.domain.process import Process
from schedsim.ml.evaluate_tree import recommend
from schedsim.policies.base import Policy
from schedsim.policies.registry import get_policy
from schedsim.rl.actions import ACTION_NAMES, action_to_pid, get_valid_actions
from schedsim.rl.agent import QLearningAgent
from schedsim.rl.state import encode_state
from schedsim.workloads.generator import generate_workload
from schedsim.workloads.scenarios import ALL_SCENARIOS


class ModelMissingException(Exception):
    """
    1. The use of class:
       Exception raised when an endpoint requires an AI model that is not yet trained or loaded.
    2. What the class contains:
       detail: Explanatory error message indicating the required CLI command to train the model.
    3. Technologies used and why:
       Standard Python Exception subclass caught by FastAPI exception handlers.
    4. Key design and overall design of the class:
       Enables HTTP 503 Service Unavailable responses with prescriptive instructions.
    """

    def __init__(self, detail: str) -> None:
        """
        1. The use of function:
           Initializes ModelMissingException with an informative error message.
        2. How does the specific function work:
           Stores the detail string and passes it to super().__init__().
        3. Concepts used in the function:
           Exception initialization.
        4. Places where this function is imported and used:
           Raised in service.py and caught in app.py.
        5. Final output of the function, before and after:
           Before: Exception message string.
           After: Instantiated ModelMissingException.
        """
        super().__init__(detail)
        self.detail = detail


def processes_from_schemas(schemas: Sequence[ProcessSchema]) -> list[Process]:
    """
    1. The use of function:
       Converts validated Pydantic ProcessSchema models into domain Process instances.
    2. How does the specific function work:
       Iterates over the input schemas, instantiating Process domain dataclass objects.
    3. Concepts used in the function:
       Data model mapping, type transformation, domain boundary protection.
    4. Places where this function is imported and used:
       Imported by route handlers and service functions.
    5. Final output of the function, before and after:
       Before: Sequence of Pydantic ProcessSchema objects.
       After: List of domain Process instances.
    """
    return [
        Process(
            pid=p.pid,
            arrival=p.arrival,
            burst=p.burst,
            priority=p.priority,
        )
        for p in schemas
    ]


def run_simulation_service(
    processes: list[Process],
    algorithm: str,
    quantum: int = 2,
    cs_cost: int = 0,
    rl_agent: QLearningAgent | None = None,
) -> SimulateResponse:
    """
    1. The use of function:
       Executes simulation for a single policy, recording step playback and decision traces.
    2. How does the specific function work:
       Initializes Engine. For classical policies, queries policy registry. For QLEARNING/RL,
       verifies rl_agent presence (or raises ModelMissingException), selecting actions greedily.
       At each simulation tick, logs ready queue state; for RL, also logs state, valid actions,
       Q-values, and step reward. Packages final SimResult into SimulateResponse.
    3. Concepts used in the function:
       Discrete unit-step simulation, playback logging, RL decision explainability.
    4. Places where this function is imported and used:
       Imported by simulate and compare routes.
    5. Final output of the function, before and after:
       Before: Process list, algorithm name, quantum, context switch cost, and optional RL agent.
       After: Complete SimulateResponse model with metrics, timeline, and traces.
    """
    algo_key = algorithm.strip().upper()
    is_rl = algo_key in ("QLEARNING", "RL", "Q-LEARNING", "Q_LEARNING")

    policy: Policy | None = None
    if is_rl:
        if rl_agent is None:
            raise ModelMissingException("Train the model first: python cli.py train-rl")
    elif algo_key == "RR":
        policy = get_policy("rr", quantum=quantum)
    else:
        policy = get_policy(algo_key.lower())

    engine = Engine(processes=processes, cs_cost=cs_cost)
    view = engine.reset()

    if policy is not None and hasattr(policy, "reset"):
        policy.reset()

    ready_queue_log: list[ReadyQueueLogEntry] = []
    decision_trace: list[DecisionTraceEntry] | None = [] if is_rl else None

    while not engine.done:
        ready_pids = [p.pid for p in view.ready]
        current_tick = view.time

        if is_rl:
            assert rl_agent is not None
            st = encode_state(
                view,
                more_arrivals_pending=engine.has_pending_arrivals,
            )
            val_acts = get_valid_actions(view)
            q_vals = {ACTION_NAMES[a]: float(rl_agent.get_q(st, a)) for a in val_acts}
            act = rl_agent.best_action(st, val_acts, record_unseen=False)
            pid = action_to_pid(view, act)
        else:
            assert policy is not None
            pid = policy.select(view)

        step_info = engine.step(pid)

        ready_queue_log.append(
            ReadyQueueLogEntry(
                time=current_tick,
                running_pid=pid,
                ready_pids=ready_pids,
            )
        )

        if is_rl and decision_trace is not None:
            assert rl_agent is not None
            decision_trace.append(
                DecisionTraceEntry(
                    time=current_tick,
                    state=list(st),
                    valid_actions=[ACTION_NAMES[a] for a in val_acts],
                    q_values=q_vals,
                    chosen_action=ACTION_NAMES[act],
                    chosen_pid=pid,
                    reward=-float(step_info.penalty),
                )
            )

        view = step_info.view

    sim_res: SimResult = engine.result()
    sm = sim_res.simulation_metrics

    p_metrics = [
        ProcessMetricsSchema(
            pid=pm.pid,
            arrival=pm.arrival,
            burst=pm.burst,
            completion=pm.completion,
            turnaround=pm.turnaround,
            waiting=pm.waiting,
            response=pm.response,
        )
        for pm in sim_res.process_metrics
    ]

    s_metrics = SimulationMetricsSchema(
        avg_waiting=float(sm.avg_waiting),
        avg_turnaround=float(sm.avg_turnaround),
        avg_response=float(sm.avg_response),
        max_waiting=int(sm.max_waiting),
        makespan=int(sm.makespan),
        cpu_utilization=float(sm.cpu_utilization),
        overhead_ticks=int(sm.overhead_ticks),
        throughput=float(sm.throughput),
        context_switches=int(sm.context_switches),
        jain_fairness=float(sm.jain_fairness),
    )

    t_segments = [
        TimelineSegmentSchema(
            label=seg.label,
            start=seg.start,
            end=seg.end,
            duration=seg.duration,
        )
        for seg in sim_res.timeline.to_segments()
    ]

    return SimulateResponse(
        algorithm=algo_key,
        quantum=quantum,
        cs_cost=cs_cost,
        process_metrics=p_metrics,
        simulation_metrics=s_metrics,
        timeline=t_segments,
        ready_queue_log=ready_queue_log,
        decision_trace=decision_trace,
    )


def run_comparison_service(
    processes: list[Process],
    algorithms: list[str] | None = None,
    quantum: int = 2,
    cs_cost: int = 0,
    rl_agent: QLearningAgent | None = None,
) -> CompareResponse:
    """
    1. The use of function:
       Runs all designated scheduling algorithms on a single workload for side-by-side comparison.
    2. How does the specific function work:
       Resolves algorithm list (defaulting to FCFS, SJF, RR, SRTF, and RL if rl_agent available).
       Simulates each policy via run_simulation_service, aggregates results into a dictionary,
       and builds a comparative summary table highlighting key metrics across policies.
    3. Concepts used in the function:
       Batch algorithm comparison, metric aggregation, comparative table generation.
    4. Places where this function is imported and used:
       Imported by the /api/compare route handler.
    5. Final output of the function, before and after:
       Before: Processes and optional algorithm list.
       After: CompareResponse with per-algorithm simulations and summary list.
    """
    if algorithms is None:
        target_algos = ["FCFS", "SJF", "RR", "SRTF"]
        if rl_agent is not None:
            target_algos.append("RL")
    else:
        target_algos = algorithms

    results: dict[str, SimulateResponse] = {}
    summary_rows: list[dict[str, Any]] = []

    for algo in target_algos:
        sim_resp = run_simulation_service(
            processes=processes,
            algorithm=algo,
            quantum=quantum,
            cs_cost=cs_cost,
            rl_agent=rl_agent,
        )
        results[algo] = sim_resp
        sm = sim_resp.simulation_metrics
        summary_rows.append({
            "algorithm": algo,
            "avg_waiting": sm.avg_waiting,
            "avg_response": sm.avg_response,
            "avg_turnaround": sm.avg_turnaround,
            "max_waiting": sm.max_waiting,
            "context_switches": sm.context_switches,
            "makespan": sm.makespan,
            "cpu_utilization": sm.cpu_utilization,
            "jain_fairness": sm.jain_fairness,
        })

    return CompareResponse(results=results, comparison_summary=summary_rows)


def run_recommend_service(
    processes: list[Process],
    dt_model_path: Path | str | None = None,
) -> RecommendResponse:
    """
    1. The use of function:
       Queries the Decision Tree classifier to recommend an optimal scheduling policy.
    2. How does the specific function work:
       Resolves model file path. Verifies file existence; if absent, raises ModelMissingException.
       Calls recommend(processes) and formats prediction, class probabilities, and features.
    3. Concepts used in the function:
       Supervised inference, model artifact verification, error propagation.
    4. Places where this function is imported and used:
       Imported by the /api/recommend route handler.
    5. Final output of the function, before and after:
       Before: Domain Process list and optional model path.
       After: RecommendResponse containing algorithm, probabilities, and extracted features.
    """
    m_path = (
        Path(dt_model_path)
        if dt_model_path is not None
        else (config.MODELS_DIR / "decision_tree.joblib")
    )
    if not m_path.exists():
        raise ModelMissingException("Train the model first: python cli.py train-tree")

    rec = recommend(processes=processes, model_path=m_path)
    return RecommendResponse(
        algorithm=rec["algorithm"],
        probabilities=rec["probabilities"],
        features=rec["features"],
    )


def generate_workload_service(
    scenario: str,
    n: int = 5,
    seed: int = 42,
) -> GenerateWorkloadResponse:
    """
    1. The use of function:
       Generates a reproducible synthetic workload using domain generator functions.
    2. How does the specific function work:
       Validates scenario name against ALL_SCENARIOS. Calls generate_workload(scenario, n, seed),
       maps resulting Process objects to ProcessSchema, and returns GenerateWorkloadResponse.
    3. Concepts used in the function:
       Synthetic workload synthesis, seed reproducibility, schema conversion.
    4. Places where this function is imported and used:
       Imported by /api/workloads/generate route handler.
    5. Final output of the function, before and after:
       Before: Scenario name string, process count n, and integer seed.
       After: GenerateWorkloadResponse with unique workload ID and process schemas.
    """
    canonical_scenario = scenario.strip().lower()
    if canonical_scenario not in ALL_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Available scenarios: {', '.join(ALL_SCENARIOS)}."
        )

    procs = generate_workload(scenario=canonical_scenario, n=n, seed=seed)
    p_schemas = [
        ProcessSchema(
            pid=p.pid,
            arrival=p.arrival,
            burst=p.burst,
            priority=p.priority,
        )
        for p in procs
    ]

    wl_id = f"api_{canonical_scenario}_{seed:06d}"
    return GenerateWorkloadResponse(
        workload_id=wl_id,
        scenario=canonical_scenario,
        seed=seed,
        processes=p_schemas,
    )


__all__ = [
    "ModelMissingException",
    "generate_workload_service",
    "processes_from_schemas",
    "run_comparison_service",
    "run_recommend_service",
    "run_simulation_service",
]
