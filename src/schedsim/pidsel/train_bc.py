"""
1. What this file does:
   Implements behavioral cloning and DAgger (Dataset Aggregation) training for Stage 16: Model 5 -
   Neural Candidate Scorer. Defines the temperature-scaled soft cross-entropy loss function,
   parallelized learner rollout collection, iterative DAgger retraining loop, closed-loop
   multi-seed validation benchmarking against Rollout Oracle and SRTF, and persistence of pure-NumPy
   weights, metadata, and validation CSV summaries.
2. Overall design:
   Supervises the neural candidate scorer using soft regret distributions p*(a) ∝ exp(-regret_a / T)
   over valid candidates. Runs DAgger (DAGGER_ITERS = 2): iteration 0 trains on initial oracle
   trajectories; subsequent iterations roll out the current neural model on fresh train workloads,
   mixing oracle actions with probability beta = 0.5^i, query evaluate_state at every visited
   non-forced state, append new transitions, and retrain warm-start. Evaluates across multiple seeds
   (NN_SEEDS = [0, 1, 2]) on 200 fixed validation workloads, saving the best model and compiling
   mean +/- std results in results/pid_m2_val.csv.
3. Elements in the file:
   - soft_cross_entropy_loss: Temperature-scaled cross-entropy loss over masked candidate regrets.
   - _worker_dagger_rollout: Multiprocessing worker rolling out mixed policy and recording states.
   - _collect_dagger_states: Orchestrates parallel or sequential DAgger state collection.
   - evaluate_neural_on_validation: Closed-loop benchmark on validation workloads vs Oracle & SRTF.
   - train_neural_scorer: Top-level multi-seed DAgger training and evaluation pipeline.
4. Final output of the file:
   Exports soft_cross_entropy_loss, evaluate_neural_on_validation, and train_neural_scorer.
"""

import json
import os
import random
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from config import (
    DAGGER_ITERS,
    DAGGER_WORKLOADS_PER_ITER,
    DATA_DIR,
    MODELS_DIR,
    NN_HIDDEN,
    NN_SEEDS,
    PID_BC_TRAIN_WORKLOADS,
    RESULTS_DIR,
)
from schedsim.domain.engine import Engine, run
from schedsim.domain.process import Process
from schedsim.pidsel.features import (
    FEATURE_NAMES,
    extract_candidate_features,
    pad_and_stack_features,
)
from schedsim.pidsel.oracle import (
    RolloutOraclePolicy,
    evaluate_state,
    label_workloads_parallel,
)
from schedsim.pidsel.scorer_net import (
    NeuralCandidatePolicy,
    Scorer,
    export_numpy_weights,
    numpy_score,
)
from schedsim.policies.srtf import SRTFPolicy
from schedsim.workloads.io import Workload, load_workloads_from_csv
from schedsim.workloads.splits import generate_split_workloads


def soft_cross_entropy_loss(
    logits: torch.Tensor,
    regrets: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    1. The use of function:
       Computes soft cross-entropy / KL divergence loss between model logits and oracle regret
       target distributions over valid ready process candidates.
    2. How does the specific function work:
       Validates tensor inputs and temperature > 0. Computes target soft probability distribution
       p*(a) = softmax(-regret / T) over valid candidates where mask is True. Computes model log
       probabilities log q(a) = log_softmax(logits) over valid candidates. Multiplies p*(a) and
       log q(a) masked safely to avoid 0 * -inf = NaN, sums over candidates, and averages across
       valid decision instances.
    3. Concepts used in the function:
       Soft behavioral cloning, temperature-scaled Boltzmann distribution, masked log-softmax,
       numerical stability against zero-probability infinite logits.
    4. Places where this function is imported and used:
       Exported in schedsim.pidsel, used in train_neural_scorer, and tested in Stage 16 test suite.
    5. Final output of the function, before and after:
       Before: Logits [B, N], regrets [B, N], mask [B, N] bool, temperature float.
       After: Scalar PyTorch loss tensor with gradient tracking.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}.")
    if not isinstance(regrets, torch.Tensor):
        raise TypeError(f"regrets must be a torch.Tensor, got {type(regrets).__name__}.")
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask).__name__}.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    # Target soft distribution: p*(a) ∝ exp(-regret_a / T)
    target_logits = (-regrets / float(temperature)).masked_fill(~mask, float("-inf"))
    target_p = torch.softmax(target_logits, dim=-1)

    # Model log distribution: log q(a)
    model_logits = logits.masked_fill(~mask, float("-inf"))
    model_log_p = torch.log_softmax(model_logits, dim=-1)

    # Safe elementwise product: 0.0 * -inf -> 0.0
    elem_loss = torch.where(mask, target_p * model_log_p, torch.zeros_like(model_log_p))
    row_loss = -elem_loss.sum(dim=-1)

    valid_rows = mask.any(dim=-1)
    if not valid_rows.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return row_loss[valid_rows].mean()


def _worker_dagger_rollout(
    task: tuple[tuple[Process, ...], int, dict[str, np.ndarray], float, int],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, int, str]]:
    """
    1. The use of function:
       Multiprocessing worker rolling out a mixed learner/oracle policy on a single workload.
    2. How does the specific function work:
       Unpacks processes, cs_cost, weights, mixing probability beta, and rollout seed. Sets RNG.
       Initializes simulation engine. At forced decisions (len <= 1), steps candidate directly.
       At non-forced decisions (len >= 2), queries evaluate_state for oracle costs and regrets,
       extracts candidate features, computes learner scores via numpy_score, selects action with
       probability beta from oracle or 1-beta from learner, records transition, and steps engine.
    3. Concepts used in the function:
       DAgger mixed rollout policy, on-policy state exploration, oracle supervision labelling.
    4. Places where this function is imported and used:
       Executed by worker processes mapped by ProcessPoolExecutor in _collect_dagger_states.
    5. Final output of the function, before and after:
       Before: Task tuple (processes, cs_cost, weights, beta, seed).
       After: List of recorded state tuples (features, costs, regrets, best_idx, best_pid).
    """
    processes, cs_cost, weights, beta, seed = task
    rng = random.Random(seed)
    engine = Engine(processes=processes, cs_cost=cs_cost)
    engine.reset()
    collected: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, str]] = []

    while not engine.done:
        view = engine.current_view
        if view is None or not view.ready:
            break

        if len(view.ready) <= 1:
            step_pid = view.ready[0].pid
        else:
            decision = evaluate_state(engine, base_policies=("SRTF", "SJF"))
            x_cands, candidate_pids = extract_candidate_features(
                view, cs_cost=cs_cost, pending_arrivals=view.pending_arrivals
            )
            scores = numpy_score(
                weights, x_cands[np.newaxis, :, :], np.ones((1, len(candidate_pids)), dtype=bool)
            )[0]

            proc_map = {p.pid: p for p in view.ready}
            best_pair = min(
                zip(candidate_pids, scores, strict=True),
                key=lambda pair: (
                    -float(pair[1]),
                    proc_map[pair[0]].remaining,
                    proc_map[pair[0]].arrival,
                    pair[0],
                ),
            )
            learner_pid = best_pair[0]

            if rng.random() < beta:
                step_pid = decision.best_pid
            else:
                step_pid = learner_pid

            costs_vec = np.array(
                [float(decision.costs[pid]) for pid in candidate_pids], dtype=np.float32
            )
            regrets_vec = np.array(
                [float(decision.regrets[pid]) for pid in candidate_pids], dtype=np.float32
            )
            best_idx = candidate_pids.index(decision.best_pid)

            collected.append((x_cands, costs_vec, regrets_vec, best_idx, decision.best_pid))

        engine.step(step_pid)

    return collected


def _collect_dagger_states(
    workloads: Sequence[Workload],
    cs_cost: int,
    weights: dict[str, np.ndarray],
    beta: float,
    seed_base: int = 0,
    n_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1. The use of function:
       Gathers new training states from workloads under the DAgger mixed policy.
    2. How does the specific function work:
       Constructs tasks for all workloads with distinct seeds. Dispatches to ProcessPoolExecutor
       when worker count > 1, or runs sequentially as a reliable fallback. Aggregates recorded
       states, formats them into padded tensors with pad_and_stack_features, and returns
       padded features, mask, and padded regrets.
    3. Concepts used in the function:
       Parallel state trajectory harvesting, feature tensor padding, DAgger buffer aggregation.
    4. Places where this function is imported and used:
       Called during iterations 1..DAGGER_ITERS inside train_neural_scorer.
    5. Final output of the function, before and after:
       Before: Workload sequence, cs_cost, weights dict, beta float, seed base.
       After: Tuple of (new_features [M, max_c, 13], new_mask [M, max_c], new_regrets [M, max_c]).
    """
    tasks = [
        (w.processes, cs_cost, weights, beta, seed_base + idx) for idx, w in enumerate(workloads)
    ]

    all_raw: list[tuple[np.ndarray, np.ndarray, np.ndarray, int, str]] = []

    resolved_workers = n_workers if n_workers is not None else min(os.cpu_count() or 1, 8)
    if resolved_workers > 1 and len(tasks) > 4:
        try:
            with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
                for res in executor.map(_worker_dagger_rollout, tasks):
                    all_raw.extend(res)
        except Exception:
            all_raw.clear()
            for t in tasks:
                all_raw.extend(_worker_dagger_rollout(t))
    else:
        for t in tasks:
            all_raw.extend(_worker_dagger_rollout(t))

    if not all_raw:
        return (
            np.zeros((0, 0, len(FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0, 0), dtype=bool),
            np.zeros((0, 0), dtype=np.float32),
        )

    feat_list = [item[0] for item in all_raw]
    padded_feats, mask = pad_and_stack_features(feat_list)

    m = len(all_raw)
    max_c = padded_feats.shape[1]
    padded_regrets = np.zeros((m, max_c), dtype=np.float32)

    for i, item in enumerate(all_raw):
        k = len(item[2])
        padded_regrets[i, :k] = item[2]

    return padded_feats, mask, padded_regrets


def evaluate_neural_on_validation(
    policy: NeuralCandidatePolicy,
    val_workloads: Sequence[Workload],
    cs_cost: int = 0,
    cached_oracle_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    1. The use of function:
       Evaluates a NeuralCandidatePolicy on a fixed set of validation workloads against Rollout
       Oracle and SRTF baseline policies.
    2. How does the specific function work:
       Iterates over validation workloads. Runs policy, SRTF, and RolloutOracle (or uses precomputed
       oracle results) to compute closed-loop average waiting times. Tracks decision-level top-1
       agreement and mean regret. Computes closed-loop gap vs Oracle and closed-loop gap vs SRTF.
    3. Concepts used in the function:
       Closed-loop policy evaluation, relative optimality gap, top-1 agreement percentage.
    4. Places where this function is imported and used:
       Exported in schedsim.pidsel, called during train_neural_scorer validation check.
    5. Final output of the function, before and after:
       Before: NeuralCandidatePolicy instance, validation workloads, context switch cost.
       After: Metrics dictionary containing agreement, regret, waiting times, and gaps.
    """
    srtf_policy = SRTFPolicy()
    oracle_policy = RolloutOraclePolicy(cs_cost=cs_cost)

    j_policy_list: list[float] = []
    j_srtf_list: list[float] = []
    j_oracle_list: list[float] = []

    agreed_decisions = 0
    total_decisions = 0
    total_regret = 0.0

    for idx, w in enumerate(val_workloads):
        sim_p = run(policy, w.processes, cs_cost=cs_cost)
        j_policy_list.append(sim_p.simulation_metrics.avg_waiting)

        sim_s = run(srtf_policy, w.processes, cs_cost=cs_cost)
        j_srtf_list.append(sim_s.simulation_metrics.avg_waiting)

        if cached_oracle_results is not None and "j_oracle_list" in cached_oracle_results:
            j_oracle_list.append(cached_oracle_results["j_oracle_list"][idx])
        elif cs_cost == 0:
            # At S0 (cs_cost=0), SRTF is optimal in hindsight; oracle waiting matches SRTF exactly.
            j_oracle_list.append(sim_s.simulation_metrics.avg_waiting)
        else:
            sim_o = run(oracle_policy, w.processes, cs_cost=cs_cost)
            j_oracle_list.append(sim_o.simulation_metrics.avg_waiting)

        # Decision-level agreement along oracle trajectory
        eng = Engine(processes=w.processes, cs_cost=cs_cost)
        eng.reset()
        v = eng.current_view
        rec_idx = 0
        w_records = (
            cached_oracle_results["val_oracle_records"][idx]
            if cached_oracle_results is not None and "val_oracle_records" in cached_oracle_results
            else None
        )
        while not eng.done and v is not None:
            if not v.ready:
                break
            if len(v.ready) <= 1:
                p_step = v.ready[0].pid
            else:
                total_decisions += 1
                if cs_cost == 0:
                    proc_map = {p.pid: p for p in v.ready}
                    min_rem = min(p.remaining for p in v.ready)
                    optimal_pids = {p.pid for p in v.ready if p.remaining == min_rem}
                    chosen_pid = policy.select(v)
                    if chosen_pid in optimal_pids:
                        agreed_decisions += 1
                    total_regret += float(proc_map[chosen_pid].remaining - min_rem)
                    p_step = chosen_pid
                elif w_records is not None and rec_idx < len(w_records):
                    decision = w_records[rec_idx]
                    rec_idx += 1
                    best_c = decision.best_cost
                    optimal_pids = {
                        pid for pid, c in decision.costs.items() if c <= best_c * 1.01 + 1e-6
                    }
                    chosen_pid = policy.select(v)
                    if chosen_pid in optimal_pids:
                        agreed_decisions += 1
                    total_regret += float(decision.regrets.get(chosen_pid, 0.0))
                    p_step = decision.best_pid
                else:
                    decision = evaluate_state(eng, base_policies=("SRTF", "SJF"))
                    best_c = decision.best_cost
                    optimal_pids = {
                        pid for pid, c in decision.costs.items() if c <= best_c * 1.01 + 1e-6
                    }
                    chosen_pid = policy.select(v)
                    if chosen_pid in optimal_pids:
                        agreed_decisions += 1
                    total_regret += float(decision.regrets.get(chosen_pid, 0.0))
                    p_step = decision.best_pid
            v = eng.step(p_step).view

    top1_agreement = float(agreed_decisions / total_decisions) if total_decisions > 0 else 1.0
    mean_regret = float(total_regret / total_decisions) if total_decisions > 0 else 0.0
    mean_j_p = float(np.mean(j_policy_list))
    mean_j_o = float(np.mean(j_oracle_list))
    mean_j_s = float(np.mean(j_srtf_list))

    gap_o = float((mean_j_p - mean_j_o) / mean_j_o) if mean_j_o > 0 else 0.0
    gap_s = float((mean_j_p - mean_j_s) / mean_j_s) if mean_j_s > 0 else 0.0

    return {
        "top1_agreement": top1_agreement,
        "mean_regret_per_decision": mean_regret,
        "total_non_forced_decisions": total_decisions,
        "agreed_decisions": agreed_decisions,
        "mean_j_policy": mean_j_p,
        "mean_j_oracle": mean_j_o,
        "mean_j_srtf": mean_j_s,
        "closed_loop_gap_oracle": gap_o,
        "closed_loop_gap_srtf": gap_s,
        "j_oracle_list": j_oracle_list,
    }


def train_neural_scorer(
    setting: str = "S0",
    data_dir: Path | str = DATA_DIR,
    models_dir: Path | str = MODELS_DIR,
    results_dir: Path | str = RESULTS_DIR,
    dagger_iters: int = DAGGER_ITERS,
    dagger_workloads_per_iter: int = DAGGER_WORKLOADS_PER_ITER,
    nn_seeds: list[int] | None = None,
    temperature: float = 1.0,
    batch_size: int = 512,
    lr: float = 1e-3,
    epochs_per_iter: int = 15,
    max_train_states: int | None = 30000,
    n_val_workloads: int = 200,
    n_workers: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    1. The use of function:
       Trains the neural candidate scorer using behavioral cloning and DAgger across seeds,
       evaluating against Rollout Oracle and SRTF, and exporting best weights and validation CSV.
    2. How does the specific function work:
       Validates setting and loads initial dataset from pid_bc_<S>.npz. For each seed in nn_seeds:
       initializes Scorer(13, 32), trains on initial buffer for epochs_per_iter, then for each
       DAgger iteration collects learner rollout states with mixing prob 0.5^i, appends to buffer,
       and retrains. Evaluates each seed on fixed validation workloads. Selects best checkpoint
       across seeds, saves models/pid_scorer_<S>_np.npz and meta.json, and writes pid_m2_val.csv.
    3. Concepts used in the function:
       DAgger algorithm, warm-start retraining, multi-seed variance estimation, artifact export.
    4. Places where this function is imported and used:
       Exported in schedsim.pidsel, invoked by CLI train-pid-nn subcommand and test suites.
    5. Final output of the function, before and after:
       Before: Raw dataset and hyperparameter specifications.
       After: Trained models saved, results/pid_m2_val.csv updated, returns summary dictionary.
    """
    setting_key = setting.upper().strip()
    if setting_key not in config.SETTINGS:
        valid = ", ".join(config.SETTINGS.keys())
        raise ValueError(f"Unknown setting '{setting}'. Must be one of: {valid}.")
    cs_cost = config.SETTINGS[setting_key].cs_cost

    data_dir_path = Path(data_dir)
    models_dir_path = Path(models_dir)
    results_dir_path = Path(results_dir)
    models_dir_path.mkdir(parents=True, exist_ok=True)
    results_dir_path.mkdir(parents=True, exist_ok=True)

    npz_path = data_dir_path / f"pid_bc_{setting_key}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Behavioral cloning dataset not found at {npz_path}.")

    loaded_npz = np.load(npz_path)
    init_features = loaded_npz["features"]
    init_mask = loaded_npz["mask"]
    init_regrets = loaded_npz["regrets"]

    if max_train_states is not None and max_train_states > 0:
        n_slice = min(max_train_states, len(init_features))
        init_features = init_features[:n_slice]
        init_mask = init_mask[:n_slice]
        init_regrets = init_regrets[:n_slice]

    # Load validation workloads
    csv_path = data_dir_path / "workloads.csv"
    val_workloads: list[Workload] = []
    all_train_workloads: list[Workload] = []
    if csv_path.exists():
        try:
            all_ws = load_workloads_from_csv(csv_path)
            val_workloads = [w for w in all_ws if w.workload_id.startswith("val")][:n_val_workloads]
            all_train_workloads = [w for w in all_ws if w.workload_id.startswith("train")]
        except (ValueError, OSError):
            val_workloads = []
            all_train_workloads = []

    if len(val_workloads) < n_val_workloads:
        needed = n_val_workloads - len(val_workloads)
        val_workloads.extend(
            generate_split_workloads("val", n_workloads=needed, seed_offset=len(val_workloads))
        )
    val_workloads = val_workloads[:n_val_workloads]

    # Pre-cache oracle validation results for speed across multiple seeds
    cached_oracle: dict[str, Any] | None = None
    if cs_cost > 0:
        oracle_runs = label_workloads_parallel(
            val_workloads,
            cs_cost=cs_cost,
            n_workers=n_workers,
        )
        cached_oracle = {
            "j_oracle_list": [res.simulation_metrics.avg_waiting for res, _ in oracle_runs],
            "val_oracle_records": [[rec.decision for rec in steps] for _, steps in oracle_runs],
        }

    seeds = list(nn_seeds) if nn_seeds is not None else list(NN_SEEDS)
    per_seed_results: list[dict[str, Any]] = []

    best_seed_metrics: dict[str, Any] | None = None
    best_weights: dict[str, np.ndarray] | None = None
    best_seed = seeds[0]

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = Scorer(d=len(FEATURE_NAMES), h=NN_HIDDEN)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        cur_features = init_features.copy()
        cur_mask = init_mask.copy()
        cur_regrets = init_regrets.copy()

        # DAgger loop: iteration 0 on initial buffer, 1..dagger_iters on aggregated buffer
        for dagger_iter in range(dagger_iters + 1):
            t_feats = torch.from_numpy(cur_features).to(device)
            t_mask = torch.from_numpy(cur_mask).to(device)
            t_regs = torch.from_numpy(cur_regrets).to(device)

            dataset = TensorDataset(t_feats, t_mask, t_regs)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            model.train()
            for _ in range(epochs_per_iter):
                for bx, bm, br in loader:
                    optimizer.zero_grad()
                    logits = model(bx, bm)
                    loss = soft_cross_entropy_loss(logits, br, bm, temperature=temperature)
                    loss.backward()
                    optimizer.step()

            # If more DAgger iterations remain, collect states with current model
            if dagger_iter < dagger_iters and dagger_workloads_per_iter > 0:
                model.eval()
                interim_weights = export_numpy_weights(model)
                beta = float(0.5 ** (dagger_iter + 1))

                start_idx = PID_BC_TRAIN_WORKLOADS + dagger_iter * dagger_workloads_per_iter
                end_idx = start_idx + dagger_workloads_per_iter
                iter_workloads = all_train_workloads[start_idx:end_idx]

                if len(iter_workloads) < dagger_workloads_per_iter:
                    needed_w = dagger_workloads_per_iter - len(iter_workloads)
                    iter_workloads.extend(
                        generate_split_workloads(
                            "train", n_workloads=needed_w, seed_offset=start_idx
                        )
                    )

                new_f, new_m, new_r = _collect_dagger_states(
                    iter_workloads,
                    cs_cost=cs_cost,
                    weights=interim_weights,
                    beta=beta,
                    seed_base=start_idx + seed * 10000,
                    n_workers=n_workers,
                )

                if len(new_f) > 0:
                    max_c = max(cur_features.shape[1], new_f.shape[1])
                    if cur_features.shape[1] < max_c:
                        pad_len = max_c - cur_features.shape[1]
                        cur_features = np.pad(cur_features, ((0, 0), (0, pad_len), (0, 0)))
                        cur_mask = np.pad(cur_mask, ((0, 0), (0, pad_len)))
                        cur_regrets = np.pad(cur_regrets, ((0, 0), (0, pad_len)))
                    if new_f.shape[1] < max_c:
                        pad_len = max_c - new_f.shape[1]
                        new_f = np.pad(new_f, ((0, 0), (0, pad_len), (0, 0)))
                        new_m = np.pad(new_m, ((0, 0), (0, pad_len)))
                        new_r = np.pad(new_r, ((0, 0), (0, pad_len)))

                    cur_features = np.concatenate([cur_features, new_f], axis=0)
                    cur_mask = np.concatenate([cur_mask, new_m], axis=0)
                    cur_regrets = np.concatenate([cur_regrets, new_r], axis=0)

        # Evaluation on validation set
        model.eval()
        trained_weights = export_numpy_weights(model)
        eval_policy = NeuralCandidatePolicy(
            weights=trained_weights,
            cs_cost=cs_cost,
            name=f"PID_NN_{setting_key}_seed{seed}",
        )
        val_metrics = evaluate_neural_on_validation(
            eval_policy,
            val_workloads,
            cs_cost=cs_cost,
            cached_oracle_results=cached_oracle,
        )

        res_entry = {
            "setting": setting_key,
            "seed": seed,
            "top1_agreement": val_metrics["top1_agreement"],
            "mean_waiting_time": val_metrics["mean_j_policy"],
            "gap_vs_oracle": val_metrics["closed_loop_gap_oracle"],
            "gap_vs_srtf": val_metrics["closed_loop_gap_srtf"],
            "mean_j_oracle": val_metrics["mean_j_oracle"],
            "mean_j_srtf": val_metrics["mean_j_srtf"],
            "mean_regret": val_metrics["mean_regret_per_decision"],
        }
        per_seed_results.append(res_entry)

        # Track best seed: lowest gap vs oracle, breaking ties with highest top-1 agreement
        if (
            best_seed_metrics is None
            or res_entry["gap_vs_oracle"] < best_seed_metrics["gap_vs_oracle"]
            or (
                res_entry["gap_vs_oracle"] == best_seed_metrics["gap_vs_oracle"]
                and res_entry["top1_agreement"] > best_seed_metrics["top1_agreement"]
            )
        ):
            best_seed_metrics = res_entry
            best_weights = trained_weights
            best_seed = seed

    assert best_weights is not None
    assert best_seed_metrics is not None

    # Save best checkpoint
    npz_out = models_dir_path / f"pid_scorer_{setting_key}_np.npz"
    np.savez_compressed(npz_out, **best_weights)

    # Compute multi-seed statistics
    agreements = [r["top1_agreement"] for r in per_seed_results]
    waitings = [r["mean_waiting_time"] for r in per_seed_results]
    gaps_o = [r["gap_vs_oracle"] for r in per_seed_results]
    gaps_s = [r["gap_vs_srtf"] for r in per_seed_results]

    summary_stats = {
        "top1_agreement_mean": float(np.mean(agreements)),
        "top1_agreement_std": float(np.std(agreements)),
        "waiting_mean": float(np.mean(waitings)),
        "waiting_std": float(np.std(waitings)),
        "gap_oracle_mean": float(np.mean(gaps_o)),
        "gap_oracle_std": float(np.std(gaps_o)),
        "gap_srtf_mean": float(np.mean(gaps_s)),
        "gap_srtf_std": float(np.std(gaps_s)),
    }

    meta_dict = {
        "setting": setting_key,
        "cs_cost": cs_cost,
        "hidden_dim": NN_HIDDEN,
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "best_seed": best_seed,
        "dagger_iters": dagger_iters,
        "epochs_per_iter": epochs_per_iter,
        "best_seed_metrics": best_seed_metrics,
        "seeds_summary": summary_stats,
        "all_seeds": per_seed_results,
    }

    meta_out = models_dir_path / f"pid_scorer_{setting_key}_meta.json"
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    # Update results/pid_m2_val.csv preserving any other existing settings
    csv_val_path = results_dir_path / "pid_m2_val.csv"
    existing_rows: list[str] = []
    header = "setting,seed,top1_agreement,mean_waiting_time,gap_vs_oracle,gap_vs_srtf\n"

    if csv_val_path.exists():
        try:
            with open(csv_val_path, encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    for line in lines[1:]:
                        if not line.startswith(f"{setting_key},"):
                            existing_rows.append(line)
        except OSError:
            existing_rows = []

    new_rows: list[str] = []
    for r in per_seed_results:
        new_rows.append(
            f"{setting_key},{r['seed']},{r['top1_agreement']:.6f},"
            f"{r['mean_waiting_time']:.4f},{r['gap_vs_oracle']:.6f},{r['gap_vs_srtf']:.6f}\n"
        )
    new_rows.append(
        f"{setting_key},mean,{summary_stats['top1_agreement_mean']:.6f},"
        f"{summary_stats['waiting_mean']:.4f},{summary_stats['gap_oracle_mean']:.6f},"
        f"{summary_stats['gap_srtf_mean']:.6f}\n"
    )
    new_rows.append(
        f"{setting_key},std,{summary_stats['top1_agreement_std']:.6f},"
        f"{summary_stats['waiting_std']:.4f},{summary_stats['gap_oracle_std']:.6f},"
        f"{summary_stats['gap_srtf_std']:.6f}\n"
    )

    with open(csv_val_path, "w", encoding="utf-8") as f:
        f.write(header)
        for r in existing_rows:
            f.write(r)
        for r in new_rows:
            f.write(r)

    return meta_dict


__all__ = [
    "evaluate_neural_on_validation",
    "soft_cross_entropy_loss",
    "train_neural_scorer",
]
