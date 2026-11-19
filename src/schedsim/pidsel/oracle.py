"""
1. What this file does:
   Implements the 1-step lookahead rollout oracle for process-level scheduling (PID selection).
   Provides state evaluation against base rollout policies (SRTF, SJF), candidate cost and regret
   vector computation, full workload oracle simulation, RolloutOraclePolicy wrapper, and parallel
   dataset labelling.
2. Overall design:
   For any observable non-forced simulation state, the oracle clones the engine, branches each
   valid candidate ready process by one unit step, rolls out to completion under each policy in
   ORACLE_BASE_POLICIES, and computes candidate cost as immediate penalty + min finished rollout
   penalty. Ties are broken strictly by (cost, remaining, arrival, pid).
3. Elements in the file:
   - OracleDecision: Frozen dataclass encapsulating best PID, best cost, cost map, and regrets.
   - OracleStepRecord: Frozen dataclass recording simulation time, ready PIDs, last PID, decision.
   - RolloutOraclePolicy: Policy class executing oracle lookahead with a synchronized shadow engine.
   - evaluate_state: Evaluates all ready candidates from current engine state via rollout cloning.
   - run_oracle: Simulates full workload under oracle guidance, returning result and step records.
   - _worker_run_oracle: Multiprocessing worker function executing run_oracle for a workload task.
   - label_workloads_parallel: Parallel batch oracle runner preserving input sequence ordering.
4. Final output of the file:
   Exports OracleDecision, OracleStepRecord, RolloutOraclePolicy, evaluate_state, run_oracle,
   and label_workloads_parallel for PID dataset generation, training, and benchmarking.
"""

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from config import ORACLE_BASE_POLICIES
from schedsim.domain.engine import Engine, SimResult, View
from schedsim.domain.process import Process, validate_processes
from schedsim.policies.base import Policy
from schedsim.policies.registry import get_policy


@dataclass(frozen=True)
class OracleDecision:
    """
    1. The use of class:
       Encapsulates the decision outcome, candidate costs, and regret values computed by the
       rollout oracle for a specific simulation decision step.
    2. What the class contains:
       - best_pid: Process identifier chosen as the optimal dispatch decision.
       - best_cost: Non-negative integer representing the minimum finished cost achieved.
       - costs: Dictionary mapping each candidate PID to its finished waiting time cost.
       - regrets: Dictionary mapping each candidate PID to its suboptimality regret (cost - best).
       - __post_init__: Integrity validator enforcing field types, bounds, and consistency.
    3. Technologies used and why:
       - Frozen dataclass for immutability, thread safety, and structural equality.
       - Python 3.11 strict static typing.
    4. Key design and overall design of the class:
       Represents the fundamental supervision signal for process-level ML/DL models (GBDT ranker
       and neural candidate scorer), guaranteeing non-negative regret and zero regret for best_pid.
    """

    best_pid: str
    best_cost: int
    costs: dict[str, int]
    regrets: dict[str, int]

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates data types, algebraic invariants, and structural integrity of OracleDecision.
        2. How does the specific function work:
           Verifies best_pid is non-empty string, best_cost is non-negative int, costs and regrets
           are non-empty dictionaries mapping candidate PIDs to non-negative integers, best_pid
           is present in both dictionaries, regrets[best_pid] == 0, and regrets[pid] == costs[pid]
           - best_cost for all candidates.
        3. Concepts used in the function:
           Dataclass post-init validation, invariant assertion, algebraic regret identity.
        4. Places where this function is imported and used:
           Invoked automatically during OracleDecision instantiation in evaluate_state.
        5. Final output of the function, before and after:
           Before: Unverified raw attribute values.
           After: Verified immutable OracleDecision instance or raises TypeError / ValueError.
        """
        if not isinstance(self.best_pid, str) or not self.best_pid.strip():
            raise ValueError("best_pid must be a non-empty string.")

        if (
            isinstance(self.best_cost, bool)
            or not isinstance(self.best_cost, int)
            or self.best_cost < 0
        ):
            raise ValueError(f"best_cost must be a non-negative int, got {self.best_cost}.")

        if not isinstance(self.costs, dict) or not self.costs:
            raise ValueError("costs must be a non-empty dictionary mapping PIDs to costs.")

        if not isinstance(self.regrets, dict) or not self.regrets:
            raise ValueError("regrets must be a non-empty dictionary mapping PIDs to regrets.")

        for pid, cost in self.costs.items():
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError(f"costs PID key must be a non-empty string, got {pid!r}.")
            if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
                raise ValueError(f"Candidate cost for PID {pid} must be a non-negative int.")

        for pid, regret in self.regrets.items():
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError(f"regrets PID key must be a non-empty string, got {pid!r}.")
            if isinstance(regret, bool) or not isinstance(regret, int) or regret < 0:
                raise ValueError(f"Candidate regret for PID {pid} must be a non-negative int.")

        if self.best_pid not in self.costs:
            raise ValueError(f"best_pid '{self.best_pid}' is missing from costs dictionary.")

        if self.best_pid not in self.regrets:
            raise ValueError(f"best_pid '{self.best_pid}' is missing from regrets dictionary.")

        if self.costs[self.best_pid] != self.best_cost:
            raise ValueError(
                f"costs[best_pid] ({self.costs[self.best_pid]}) must equal "
                f"best_cost ({self.best_cost})."
            )

        if self.regrets[self.best_pid] != 0:
            raise ValueError(
                f"regret for best_pid must be exactly 0, got {self.regrets[self.best_pid]}."
            )

        for pid, cost in self.costs.items():
            expected_regret = cost - self.best_cost
            if self.regrets.get(pid) != expected_regret:
                raise ValueError(
                    f"Regret mismatch for PID {pid}: expected {expected_regret}, "
                    f"got {self.regrets.get(pid)}."
                )


@dataclass(frozen=True)
class OracleStepRecord:
    """
    1. The use of class:
       Records the simulation context and oracle decision for a single non-forced decision point.
    2. What the class contains:
       - time: Integer simulation tick at which the decision occurred.
       - ready_pids: Tuple of ready process PIDs available for dispatch at this step.
       - last_pid: Identifier of the process running in the previous tick, or None.
       - decision: OracleDecision instance containing costs, regrets, and winning PID.
       - __post_init__: Integrity validator enforcing non-negativity and consistency.
    3. Technologies used and why:
       - Frozen dataclass for immutability and memory-efficient serialization.
       - Strict tuple typing for ready PIDs.
    4. Key design and overall design of the class:
       Provides a self-contained decision trajectory record used to assemble Stage 14 behavioral
       cloning feature datasets (features.py and dataset.py).
    """

    time: int
    ready_pids: tuple[str, ...]
    last_pid: str | None
    decision: OracleDecision

    def __post_init__(self) -> None:
        """
        1. The use of function:
           Validates field types, boundaries, and consistency between ready PIDs and decision costs.
        2. How does the specific function work:
           Checks time is non-negative int, ready_pids is non-empty tuple of valid strings,
           last_pid is None or non-empty str, decision is OracleDecision, and the set of candidate
           PIDs in decision matches ready_pids exactly.
        3. Concepts used in the function:
           Post-init validation, set consistency, domain invariant verification.
        4. Places where this function is imported and used:
           Invoked automatically during OracleStepRecord instantiation in run_oracle.
        5. Final output of the function, before and after:
           Before: Unverified raw record attributes.
           After: Verified immutable OracleStepRecord instance or raises TypeError / ValueError.
        """
        if isinstance(self.time, bool) or not isinstance(self.time, int) or self.time < 0:
            raise ValueError(f"time must be a non-negative int, got {self.time}.")

        if not isinstance(self.ready_pids, tuple) or not self.ready_pids:
            raise ValueError("ready_pids must be a non-empty tuple of PID strings.")

        for pid in self.ready_pids:
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError(f"ready_pids contains invalid PID: {pid!r}.")

        if self.last_pid is not None and (
            not isinstance(self.last_pid, str) or not self.last_pid.strip()
        ):
            raise ValueError(f"last_pid must be None or non-empty str, got {self.last_pid!r}.")

        if not isinstance(self.decision, OracleDecision):
            raise TypeError(
                f"decision must be an OracleDecision, got {type(self.decision).__name__}."
            )

        if set(self.ready_pids) != set(self.decision.costs.keys()):
            raise ValueError(
                f"ready_pids {self.ready_pids} does not match decision costs keys "
                f"{set(self.decision.costs.keys())}."
            )


def evaluate_state(
    engine: Engine,
    base_policies: Sequence[str] = ORACLE_BASE_POLICIES,
) -> OracleDecision:
    """
    1. The use of function:
       Evaluates all ready process candidates at the current simulation state via 1-step rollout
       cloning and base policy completion, returning optimal choice, costs, and regret vectors.
    2. How does the specific function work:
       Extracts ready processes from engine.current_view. For each candidate PID, creates an
       isolated clone of the engine, steps the candidate, and checks if completed. If not completed,
       clones again for each policy in base_policies, simulates to completion, and computes
       candidate cost = immediate_penalty + min(rollout_penalties). Breaks ties using the strict
       order (cost, remaining, arrival, pid), computes regrets, and returns OracleDecision.
    3. Concepts used in the function:
       Rollout algorithms, discrete-event simulation cloning, 1-step lookahead, tie-breaking rules,
       suboptimality regret calculation.
    4. Places where this function is imported and used:
       Imported by run_oracle, RolloutOraclePolicy, dataset builders, and testing suites.
    5. Final output of the function, before and after:
       Before: Engine at an active simulation state with at least one ready process.
       After: Returns an OracleDecision with best_pid, best_cost, costs dict, and regrets dict.
    """
    if not isinstance(engine, Engine):
        raise TypeError(f"engine must be an Engine instance, got {type(engine).__name__}.")
    if engine.done:
        raise RuntimeError("Cannot evaluate state on a completed simulation.")

    view = engine.current_view
    if view is None or not view.ready:
        raise ValueError("Cannot evaluate state: ready queue is empty.")

    if not base_policies:
        raise ValueError("base_policies cannot be empty.")

    candidates = [p.pid for p in view.ready]
    remaining = {p.pid: p.remaining for p in view.ready}
    arrival = {p.pid: p.arrival for p in view.ready}

    policy_instances = [get_policy(name) for name in base_policies]
    costs: dict[str, int] = {}

    for cand in candidates:
        cloned = engine.clone()
        info = cloned.step(cand)
        if cloned.done:
            costs[cand] = info.penalty
        else:
            finished_costs: list[int] = []
            for policy in policy_instances:
                rollout = cloned.clone()
                policy.reset()
                rollout_penalty = 0
                while not rollout.done:
                    r_view = rollout.current_view
                    assert r_view is not None
                    r_pid = policy.select(r_view)
                    r_info = rollout.step(r_pid)
                    rollout_penalty += r_info.penalty
                finished_costs.append(info.penalty + rollout_penalty)
            costs[cand] = min(finished_costs)

    best_pid = min(
        candidates,
        key=lambda pid: (costs[pid], remaining[pid], arrival[pid], pid),
    )
    best_cost = costs[best_pid]
    regrets = {pid: costs[pid] - best_cost for pid in candidates}

    return OracleDecision(
        best_pid=best_pid,
        best_cost=best_cost,
        costs=costs,
        regrets=regrets,
    )


def run_oracle(
    processes: Sequence[Process],
    cs_cost: int = 0,
    base_policies: Sequence[str] = ORACLE_BASE_POLICIES,
) -> tuple[SimResult, list[OracleStepRecord]]:
    """
    1. The use of function:
       Executes a full simulation workload driven by the 1-step lookahead rollout oracle,
       recording decision trajectory records at every non-forced decision point.
    2. How does the specific function work:
       Validates input processes and cs_cost. Initializes an Engine instance and advances step by
       step: if ready queue has > 1 processes, calls evaluate_state, appends an OracleStepRecord,
       and steps the best PID. If ready queue has exactly 1 process (forced decision), steps the
       single process directly without invoking rollout search. Returns SimResult and step records.
    3. Concepts used in the function:
       Oracle simulation, forced decision skipping, trajectory logging, discrete simulation.
    4. Places where this function is imported and used:
       Imported by label_workloads_parallel, Stage 14 dataset generation, and acceptance tests.
    5. Final output of the function, before and after:
       Before: Workload process definitions, context switch cost, and base policies.
       After: Tuple containing the completed SimResult and list of non-forced OracleStepRecords.
    """
    validate_processes(processes)
    if isinstance(cs_cost, bool) or not isinstance(cs_cost, int) or cs_cost < 0:
        raise ValueError(f"cs_cost must be a non-negative int, got {cs_cost}.")

    engine = Engine(processes=processes, cs_cost=cs_cost)
    engine.reset()
    step_records: list[OracleStepRecord] = []

    while not engine.done:
        view = engine.current_view
        assert view is not None
        if not view.ready:
            break

        if len(view.ready) > 1:
            decision = evaluate_state(engine, base_policies=base_policies)
            record = OracleStepRecord(
                time=view.time,
                ready_pids=tuple(p.pid for p in view.ready),
                last_pid=view.last_pid,
                decision=decision,
            )
            step_records.append(record)
            engine.step(decision.best_pid)
        else:
            forced_pid = view.ready[0].pid
            engine.step(forced_pid)

    return engine.result(), step_records


class RolloutOraclePolicy(Policy):
    """
    1. The use of class:
       Provides a Policy-compliant adapter wrapping the rollout oracle for standard simulation
       dispatch loops and benchmark evaluations via run().
    2. What the class contains:
       - cs_cost: Configured integer context switch tick penalty.
       - base_policies: Sequence of policy names used for rollout trajectory completion.
       - _processes: Cached workload processes tuple bound via bind_workload.
       - _shadow_engine: Synchronized internal Engine replica providing cloneable simulation state.
       - bind_workload: Binds and resets shadow engine for a new workload.
       - reset: Re-initializes shadow engine from cached workload processes.
       - select: Selects next PID via rollout search on shadow engine and advances shadow engine.
    3. Technologies used and why:
       - Subclasses Policy ABC for seamless interoperability with Engine, run(), and CLI.
       - Synchronized shadow engine pattern to supply rich simulation state to evaluate_state.
    4. Key design and overall design of the class:
       Enables the rollout oracle to be scheduled directly inside standard engine simulation loops
       while maintaining identical decision outcomes to run_oracle.
    """

    def __init__(
        self,
        cs_cost: int = 0,
        base_policies: Sequence[str] = ORACLE_BASE_POLICIES,
        name: str = "RolloutOracle",
    ) -> None:
        """
        1. The use of function:
           Initializes a RolloutOraclePolicy instance with context switch cost and base policies.
        2. How does the specific function work:
           Invokes Policy superclass constructor with name. Validates and stores cs_cost and
           base_policies tuple. Initializes shadow engine and cached processes to None.
        3. Concepts used in the function:
           Constructor initialization, parameter validation, defensive sequence conversion.
        4. Places where this function is imported and used:
           Instantiated in simulation pipelines, benchmark comparisons, and integration tests.
        5. Final output of the function, before and after:
           Before: Uninitialized RolloutOraclePolicy.
           After: Configured policy instance ready for bind_workload and select calls.
        """
        super().__init__(name=name)
        if isinstance(cs_cost, bool) or not isinstance(cs_cost, int) or cs_cost < 0:
            raise ValueError(f"cs_cost must be a non-negative int, got {cs_cost}.")
        if not base_policies:
            raise ValueError("base_policies cannot be empty.")

        self.cs_cost: int = cs_cost
        self.base_policies: tuple[str, ...] = tuple(base_policies)
        self._processes: tuple[Process, ...] | None = None
        self._shadow_engine: Engine | None = None

    def bind_workload(self, processes: Sequence[Process]) -> None:
        """
        1. The use of function:
           Binds a new workload process sequence to the policy and initializes the shadow engine.
        2. How does the specific function work:
           Validates processes using validate_processes. Stores a defensive copy of processes tuple.
           Creates a new Engine instance with self.cs_cost and resets it to establish initial state.
        3. Concepts used in the function:
           Workload binding lifecycle hook, shadow engine synchronization, defensive copying.
        4. Places where this function is imported and used:
           Invoked by run() runner function prior to beginning simulation execution loops.
        5. Final output of the function, before and after:
           Before: Unbound policy or previous workload state.
           After: Policy bound to new workload with initialized, reset shadow engine.
        """
        validate_processes(processes)
        self._processes = tuple(processes)
        self._shadow_engine = Engine(processes=self._processes, cs_cost=self.cs_cost)
        self._shadow_engine.reset()

    def reset(self) -> None:
        """
        1. The use of function:
           Resets the policy's internal shadow engine state between simulation episodes.
        2. How does the specific function work:
           If processes were previously bound, re-instantiates and resets the shadow engine.
           Otherwise clears shadow engine reference to None.
        3. Concepts used in the function:
           State reset lifecycle hook, reproducible simulation setup.
        4. Places where this function is imported and used:
           Invoked by runner harnesses or user pipelines between simulation episodes.
        5. Final output of the function, before and after:
           Before: Policy with progressed shadow engine state.
           After: Shadow engine reset to initial tick 0 or None.
        """
        if self._processes is not None:
            self._shadow_engine = Engine(processes=self._processes, cs_cost=self.cs_cost)
            self._shadow_engine.reset()
        else:
            self._shadow_engine = None

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the next process PID to execute using 1-step rollout oracle search.
        2. How does the specific function work:
           Validates ready queue is non-empty. For forced decisions (len <= 1), returns the single
           ready PID and steps shadow engine to keep it in sync. For non-forced decisions, evaluates
           shadow engine state using evaluate_state, selects decision.best_pid, advances shadow
           engine by one step with best_pid, and returns best_pid.
        3. Concepts used in the function:
           Shadow engine advancement, forced decision fast path, rollout policy selection.
        4. Places where this function is imported and used:
           Invoked at each simulation tick by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Simulation View snapshot at current tick.
           After: Returns selected PID string and advances internal shadow engine state.
        """
        if not view.ready:
            raise ValueError("Cannot select process: ready queue is empty.")

        if len(view.ready) <= 1:
            forced_pid = view.ready[0].pid
            if self._shadow_engine is not None and not self._shadow_engine.done:
                self._shadow_engine.step(forced_pid)
            return forced_pid

        if self._shadow_engine is None:
            raise RuntimeError(
                "RolloutOraclePolicy must be bound via bind_workload() before select()."
            )

        decision = evaluate_state(self._shadow_engine, base_policies=self.base_policies)
        best_pid = decision.best_pid
        self._shadow_engine.step(best_pid)
        return best_pid


def _worker_run_oracle(
    task: tuple[Sequence[Process], int, tuple[str, ...]],
) -> tuple[SimResult, list[OracleStepRecord]]:
    """
    1. The use of function:
       Multiprocessing worker task helper executing run_oracle for a single workload task tuple.
    2. How does the specific function work:
       Unpacks task tuple containing (processes, cs_cost, base_policies) and forwards directly
       to run_oracle, returning the SimResult and step records.
    3. Concepts used in the function:
       Multiprocessing worker function, pickle-safe top-level function, task unpacking.
    4. Places where this function is imported and used:
       Target function invoked by ProcessPoolExecutor inside label_workloads_parallel.
    5. Final output of the function, before and after:
       Before: Pickled task argument tuple.
       After: Returns tuple of (SimResult, list[OracleStepRecord]).
    """
    procs, cs_cost, base_policies = task
    return run_oracle(processes=procs, cs_cost=cs_cost, base_policies=base_policies)


def label_workloads_parallel(
    workloads: Sequence[Any],
    cs_cost: int = 0,
    base_policies: Sequence[str] = ORACLE_BASE_POLICIES,
    n_workers: int | None = None,
) -> list[tuple[SimResult, list[OracleStepRecord]]]:
    """
    1. The use of function:
       Simulates and labels a collection of workloads under the rollout oracle in parallel,
       guaranteeing deterministic input-order preservation.
    2. How does the specific function work:
       Extracts process sequences from workloads (supporting Workload instances or raw process
       sequences). Validates each process collection. If n_workers is 1 or workloads has 1 item,
       executes sequentially in current thread. Otherwise launches a ProcessPoolExecutor across
       n_workers (defaulting to CPU count capped at 8) using executor.map to strictly preserve
       the input sequence ordering.
    3. Concepts used in the function:
       Parallel computation, deterministic sequence ordering, ProcessPoolExecutor, batch labelling.
    4. Places where this function is imported and used:
       Imported by Stage 14 dataset generation scripts, benchmark evaluators, and test suites.
    5. Final output of the function, before and after:
       Before: Collection of workloads, context switch cost, base policies, optional worker count.
       After: List of (SimResult, list[OracleStepRecord]) tuples matching input sequence order.
    """
    if not workloads:
        return []

    procs_list: list[tuple[Process, ...]] = []
    for item in workloads:
        if hasattr(item, "processes"):
            procs = tuple(item.processes)
        elif isinstance(item, Sequence):
            procs = tuple(item)
        else:
            raise TypeError(f"Unsupported workload element type: {type(item).__name__}.")
        validate_processes(procs)
        procs_list.append(procs)

    if isinstance(cs_cost, bool) or not isinstance(cs_cost, int) or cs_cost < 0:
        raise ValueError(f"cs_cost must be a non-negative int, got {cs_cost}.")

    if not base_policies:
        raise ValueError("base_policies cannot be empty.")

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 8)
    elif isinstance(n_workers, bool) or not isinstance(n_workers, int) or n_workers < 1:
        raise ValueError(f"n_workers must be a positive int or None, got {n_workers}.")

    if len(procs_list) == 1 or n_workers == 1:
        return [
            run_oracle(procs, cs_cost=cs_cost, base_policies=base_policies)
            for procs in procs_list
        ]

    tasks = [(procs, cs_cost, tuple(base_policies)) for procs in procs_list]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_worker_run_oracle, tasks))

    return results
