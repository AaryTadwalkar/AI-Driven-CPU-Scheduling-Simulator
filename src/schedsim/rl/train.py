"""
1. What this file does:
   Implements reinforcement learning training pipelines, validation evaluation loops, hyperparameter
   grid search, and final multi-seed model generation for tabular Q-learning scheduling.
2. Overall design:
   - evaluate_on_validation: Runs greedy evaluation over validation workloads, logging mean waiting
     time, turnaround time, unseen state rate, and Q-table size.
   - train_q_learning: Executes episodic Q-learning training with epsilon decay, periodic validation
     evaluation checkpoints, and training trajectory logging to CSV.
   - run_hparam_search: Performs systematic grid search across learning rate (alpha), discount
     factor (gamma), and exploration decay across seeds on the validation set, logging to
     results/rl_hparam_search.csv with multi-core parallelization.
   - train_final_agents: Trains final Q-learning models across RL_TRAIN_SEEDS for cs_cost=0 and
     cs_cost=1, selecting the best seed and saving model checkpoints and JSON summaries.
3. Elements in the file:
   - evaluate_on_validation: Function assessing greedy agent performance on validation workloads.
   - train_q_learning: Core episodic training loop with periodic validation checkpointing.
   - run_hparam_search: Grid search evaluator optimizing alpha, gamma, and epsilon decay.
   - train_final_agents: Orchestrator training and saving canonical Q-table artifacts.
4. Final output of the file:
   Produces models/q_table.pkl, models/q_table_cs0.pkl, models/q_table_cs1.pkl,
   models/q_table_summary.json, results/rl_hparam_search.csv, and data/rl_training_logs.csv.
"""

import concurrent.futures
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from schedsim.domain.process import Process
from schedsim.rl.agent import QLearningAgent
from schedsim.rl.env import Environment
from schedsim.workloads.generator import generate_workload
from schedsim.workloads.io import Workload
from schedsim.workloads.scenarios import ALL_SCENARIOS
from schedsim.workloads.splits import generate_split_workloads


def evaluate_on_validation(
    agent: QLearningAgent,
    val_workloads: Sequence[Workload | Sequence[Process]],
    cs_cost: int = 1,
) -> dict[str, float]:
    """
    1. The use of function:
       Assesses greedy agent policy performance across a benchmark batch of validation workloads.
    2. How does the specific function work:
       Resets agent's unseen-state evaluation tracking counters. Initializes an Environment with
       the specified context switch cost. For each validation workload, resets the environment
       and steps greedily using agent.best_action(record_unseen=True) until termination.
       Aggregates average waiting time, average turnaround time, unseen state rate, Q-table size,
       and total unique visited states into a summary dictionary.
    3. Concepts used in the function:
       Validation evaluation, greedy exploitation, diagnostic metrics, unseen-state rate tracking.
    4. Places where this function is imported and used:
       Called periodically during training checkpoints, hyperparameter search, and acceptance tests.
    5. Final output of the function, before and after:
       Before: Trained QLearningAgent and list of validation workloads.
       After: Returns dictionary of float metrics: avg_waiting, avg_turnaround, unseen_state_rate,
       q_table_size, unique_states.
    """
    agent.reset_eval_tracking()
    env = Environment(cs_cost=cs_cost)

    total_waiting: float = 0.0
    total_turnaround: float = 0.0
    n_workloads = len(val_workloads)

    if n_workloads == 0:
        return {
            "avg_waiting": 0.0,
            "avg_turnaround": 0.0,
            "unseen_state_rate": 0.0,
            "q_table_size": agent.q_table_size,
            "unique_states": agent.unique_states_visited,
        }

    for item in val_workloads:
        procs = item.processes if isinstance(item, Workload) else item
        env.reset(procs)

        while not env.done:
            valid = env.valid_actions()
            assert env.state is not None
            action = agent.best_action(env.state, valid, record_unseen=True)
            env.step(action)

        res = env.result()
        total_waiting += res.simulation_metrics.avg_waiting
        total_turnaround += res.simulation_metrics.avg_turnaround

    return {
        "avg_waiting": round(total_waiting / n_workloads, 4),
        "avg_turnaround": round(total_turnaround / n_workloads, 4),
        "unseen_state_rate": round(agent.unseen_state_rate, 4),
        "q_table_size": agent.q_table_size,
        "unique_states": agent.unique_states_visited,
    }


def train_q_learning(
    episodes: int = config.RL_EPISODES,
    seed: int = 0,
    cs_cost: int = 1,
    alpha: float = config.ALPHA,
    gamma: float = config.GAMMA,
    eps_start: float = config.EPS_START,
    eps_min: float = config.EPS_MIN,
    eps_decay: float = config.EPS_DECAY,
    eval_every: int = config.RL_EVAL_EVERY,
    val_workloads: Sequence[Workload | Sequence[Process]] | None = None,
    save_logs: bool = False,
    log_path: Path | str | None = None,
    eval_init: bool = True,
) -> tuple[QLearningAgent, list[dict[str, Any]]]:
    """
    1. The use of function:
       Executes episodic Q-learning training with exploration decay and validation evaluation.
    2. How does the specific function work:
       Instantiates QLearningAgent and Environment. Uses a seeded Generator to uniformly sample
       workload scenarios from ALL_SCENARIOS, process counts in [3, 10], and seeds in
       TRAIN_SEED_RANGE. Steps the environment with epsilon-greedy action selection and executes
       temporal difference updates. Decays epsilon per episode. Evaluates greedy performance on
       the validation set every eval_every episodes. Optionally appends logs to CSV.
    3. Concepts used in the function:
       Q-learning training loop, epsilon decay, off-policy temporal difference updates,
       validation evaluation checkpoints, seed isolation.
    4. Places where this function is imported and used:
       Called by run_hparam_search, train_final_agents, CLI train-rl, and acceptance test suites.
    5. Final output of the function, before and after:
       Before: Raw training hyperparameters and optional validation workloads.
       After: Returns (trained QLearningAgent, list of evaluation checkpoint metric dictionaries).
    """
    rng = np.random.default_rng(seed)
    scenarios = list(ALL_SCENARIOS)

    if val_workloads is None:
        val_workloads = generate_split_workloads("val", n_workloads=500)

    meta = {
        "episodes": episodes,
        "seed": seed,
        "cs_cost": cs_cost,
        "alpha": alpha,
        "gamma": gamma,
        "eps_start": eps_start,
        "eps_min": eps_min,
        "eps_decay": eps_decay,
        "eval_every": eval_every,
    }

    agent = QLearningAgent(
        alpha=alpha,
        gamma=gamma,
        eps_start=eps_start,
        eps_min=eps_min,
        eps_decay=eps_decay,
        metadata=meta,
    )
    env = Environment(cs_cost=cs_cost)
    logs: list[dict[str, Any]] = []

    # Initial pre-training baseline evaluation if requested
    if eval_init:
        init_metrics = evaluate_on_validation(agent, val_workloads, cs_cost=cs_cost)
        logs.append(
            {
                "episode": 0,
                "seed": seed,
                "cs_cost": cs_cost,
                "epsilon": agent.epsilon,
                "alpha": alpha,
                "gamma": gamma,
                "eps_decay": eps_decay,
                **init_metrics,
            }
        )

    for ep in range(1, episodes + 1):
        scenario = scenarios[ep % len(scenarios)]
        n = int(rng.integers(3, 11))
        workload_seed = int(
            rng.integers(config.TRAIN_SEED_RANGE[0], config.TRAIN_SEED_RANGE[1] + 1)
        )
        workload = generate_workload(scenario=scenario, n=n, seed=workload_seed)

        env.reset(workload)

        while not env.done:
            valid = env.valid_actions()
            assert env.state is not None
            current_state = env.state
            action = agent.select_action(current_state, valid, rng, training=True)

            next_state, reward, done, _ = env.step(action)
            next_valid = env.valid_actions()

            agent.update(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                next_valid_actions=next_valid,
                done=done,
            )

        agent.decay_epsilon()

        if ep % eval_every == 0 or ep == episodes:
            eval_metrics = evaluate_on_validation(agent, val_workloads, cs_cost=cs_cost)
            log_entry = {
                "episode": ep,
                "seed": seed,
                "cs_cost": cs_cost,
                "epsilon": round(agent.epsilon, 4),
                "alpha": alpha,
                "gamma": gamma,
                "eps_decay": eps_decay,
                **eval_metrics,
            }
            logs.append(log_entry)

    if save_logs and log_path:
        out_p = Path(log_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(logs).to_csv(out_p, index=False)

    return agent, logs


def _eval_hparam_config(
    args: tuple[float, float, float, Sequence[int], int, int, Sequence[Any]],
) -> dict[str, Any]:
    """
    1. The use of function:
       Worker task evaluating a single hyperparameter configuration across multiple seeds.
    2. How does the specific function work:
       Unpacks alpha, gamma, decay, seeds, search_episodes, cs_cost, and val_workloads.
       Trains an agent for each seed without initial evaluation overhead, retrieves the final
       validation metrics, and calculates the mean and standard deviation across seeds.
    3. Concepts used in the function:
       Parallel worker dispatch, multi-seed aggregation, validation score computation.
    4. Places where this function is imported and used:
       Executed by ProcessPoolExecutor inside run_hparam_search.
    5. Final output of the function, before and after:
       Before: Parameter tuple.
       After: Returns summary dictionary with mean and standard deviation metrics.
    """
    alpha, gamma, decay, seeds, search_episodes, cs_cost, val_workloads = args
    seed_waiting: list[float] = []
    seed_unseen: list[float] = []
    seed_q_size: list[int] = []

    for s in seeds:
        agent, logs = train_q_learning(
            episodes=search_episodes,
            seed=s,
            cs_cost=cs_cost,
            alpha=alpha,
            gamma=gamma,
            eps_decay=decay,
            eval_every=search_episodes,
            val_workloads=val_workloads,
            save_logs=False,
            eval_init=False,
        )
        metrics = logs[-1]
        seed_waiting.append(metrics["avg_waiting"])
        seed_unseen.append(metrics["unseen_state_rate"])
        seed_q_size.append(metrics["q_table_size"])

    return {
        "alpha": alpha,
        "gamma": gamma,
        "eps_decay": decay,
        "episodes": search_episodes,
        "n_seeds": len(seeds),
        "mean_val_waiting": round(float(np.mean(seed_waiting)), 4),
        "std_val_waiting": round(float(np.std(seed_waiting)), 4),
        "mean_unseen_rate": round(float(np.mean(seed_unseen)), 4),
        "mean_q_table_size": round(float(np.mean(seed_q_size)), 1),
    }


def run_hparam_search(
    seeds: Sequence[int] = (0, 1, 2),
    search_episodes: int = 3000,
    cs_cost: int = 1,
    alphas: Sequence[float] = (0.05, 0.1, 0.2),
    gammas: Sequence[float] = (0.95, 0.99, 1.0),
    decays: Sequence[float] = (0.999, 0.9995),
    output_path: Path | str | None = None,
    val_workloads: Sequence[Workload | Sequence[Process]] | None = None,
    max_workers: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    1. The use of function:
       Performs hyperparameter grid search on validation workloads across alpha, gamma, and decay.
    2. How does the specific function work:
       Constructs tasks for all combinations of alphas x gammas x decays. Uses ProcessPoolExecutor
       to run configurations in parallel across CPU cores. For each config, averages metrics across
       the provided seeds. Sorts by mean validation waiting time, writes results to CSV, and returns
       best hyperparameter configuration.
    3. Concepts used in the function:
       Grid search, multi-process parallelism, validation hyperparameter tuning.
    4. Places where this function is imported and used:
       Called during RL pipeline setup, CLI train-rl --hparam-search, and tests.
    5. Final output of the function, before and after:
       Before: Parameter ranges to evaluate.
       After: Returns (best_hyperparameter_dict, results_dataframe) and writes CSV.
    """
    if val_workloads is None:
        val_workloads = generate_split_workloads("val", n_workloads=150)

    tasks = [
        (alpha, gamma, decay, seeds, search_episodes, cs_cost, val_workloads)
        for alpha in alphas
        for gamma in gammas
        for decay in decays
    ]

    workers = max_workers or min(len(tasks), max((os.cpu_count() or 4) - 2, 1), 12)
    records: list[dict[str, Any]] = []

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_eval_hparam_config, tasks))
    except Exception:
        # Fallback to sequential execution if ProcessPoolExecutor is unavailable
        records = [_eval_hparam_config(t) for t in tasks]

    df = pd.DataFrame(records).sort_values("mean_val_waiting")
    best_row = df.iloc[0]
    best_config = {
        "alpha": float(best_row["alpha"]),
        "gamma": float(best_row["gamma"]),
        "eps_decay": float(best_row["eps_decay"]),
        "mean_val_waiting": float(best_row["mean_val_waiting"]),
    }

    target_csv = Path(output_path) if output_path else config.RESULTS_DIR / "rl_hparam_search.csv"
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target_csv, index=False)

    return best_config, df


def _train_single_seed_worker(
    args: tuple[int, int, int, float, float, float, int, Sequence[Any]],
) -> tuple[int, int, QLearningAgent, list[dict[str, Any]]]:
    """
    1. The use of function:
       Worker task executing full training for one (cs_cost, seed) combination.
    2. How does the specific function work:
       Unpacks training parameters, invokes train_q_learning, and returns the result tuple.
    3. Concepts used in the function:
       Multi-process training worker, parallel seed dispatch.
    4. Places where this function is imported and used:
       Executed by ProcessPoolExecutor inside train_final_agents.
    5. Final output of the function, before and after:
       Before: Training parameter tuple.
       After: Returns (cs_cost, seed, trained_agent, logs).
    """
    cs_cost, s, episodes, alpha, gamma, eps_decay, eval_every, val_workloads = args
    agent, logs = train_q_learning(
        episodes=episodes,
        seed=s,
        cs_cost=cs_cost,
        alpha=alpha,
        gamma=gamma,
        eps_decay=eps_decay,
        eval_every=eval_every,
        val_workloads=val_workloads,
        save_logs=False,
        eval_init=True,
    )
    return cs_cost, s, agent, logs


def train_final_agents(
    episodes: int = config.RL_EPISODES,
    seeds: Sequence[int] = config.RL_TRAIN_SEEDS,
    alpha: float = config.ALPHA,
    gamma: float = config.GAMMA,
    eps_decay: float = config.EPS_DECAY,
    eval_every: int = config.RL_EVAL_EVERY,
    models_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    data_dir: Path | str | None = None,
    max_workers: int | None = None,
) -> dict[str, Any]:
    """
    1. The use of function:
       Trains canonical final Q-learning models across seeds for cs_cost=0 and cs_cost=1.
    2. How does the specific function work:
       Generates 500 validation workloads. Dispatches training across seeds and costs in parallel
       using ProcessPoolExecutor. For each cost (0 and 1), selects the best-performing seed on
       validation waiting time. Saves model checkpoints (q_table_cs0.pkl, q_table_cs1.pkl,
       q_table.pkl) and JSON inspection summaries, and writes rl_training_logs.csv.
    3. Concepts used in the function:
       Multi-seed parallel training, model checkpoint selection, JSON summary export.
    4. Places where this function is imported and used:
       Called by CLI train-rl, pipeline orchestrator, and final acceptance verification.
    5. Final output of the function, before and after:
       Before: Unsaved final model artifacts.
       After: Persists .pkl models, .json summaries, and returns metadata dictionary.
    """
    target_models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
    target_results_dir = Path(results_dir) if results_dir else config.RESULTS_DIR
    target_data_dir = Path(data_dir) if data_dir else config.DATA_DIR

    target_models_dir.mkdir(parents=True, exist_ok=True)
    target_results_dir.mkdir(parents=True, exist_ok=True)
    target_data_dir.mkdir(parents=True, exist_ok=True)

    val_workloads = generate_split_workloads("val", n_workloads=500)

    tasks = [
        (cs_cost, s, episodes, alpha, gamma, eps_decay, eval_every, val_workloads)
        for cs_cost in [0, 1]
        for s in seeds
    ]

    workers = max_workers or min(len(tasks), max((os.cpu_count() or 4) - 2, 1), 10)
    results: list[tuple[int, int, QLearningAgent, list[dict[str, Any]]]] = []

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_train_single_seed_worker, tasks))
    except Exception:
        results = [_train_single_seed_worker(t) for t in tasks]

    all_logs: list[dict[str, Any]] = []
    trained_artifacts: dict[str, Any] = {}

    for cs_cost in [0, 1]:
        cost_runs = [r for r in results if r[0] == cs_cost]
        best_agent: QLearningAgent | None = None
        best_val_waiting: float = float("inf")
        best_seed: int = seeds[0]

        for _, s, agent, logs in cost_runs:
            all_logs.extend(logs)
            final_metric = logs[-1]["avg_waiting"]
            if final_metric < best_val_waiting:
                best_val_waiting = final_metric
                best_agent = agent
                best_seed = s

        assert best_agent is not None

        cs_tag = f"cs{cs_cost}"
        model_file = target_models_dir / f"q_table_{cs_tag}.pkl"
        summary_file = target_models_dir / f"q_table_summary_{cs_tag}.json"

        meta = {
            "cs_cost": cs_cost,
            "best_seed": best_seed,
            "val_avg_waiting": best_val_waiting,
            "episodes": episodes,
            "alpha": alpha,
            "gamma": gamma,
            "eps_decay": eps_decay,
        }
        best_agent.save(model_file, metadata=meta)

        summary_data = best_agent.get_summary(top_n=500)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        trained_artifacts[cs_tag] = {
            "model_path": str(model_file),
            "summary_path": str(summary_file),
            "best_seed": best_seed,
            "val_avg_waiting": best_val_waiting,
        }

        if cs_cost == 1:
            default_model = target_models_dir / "q_table.pkl"
            default_summary = target_models_dir / "q_table_summary.json"
            best_agent.save(default_model, metadata=meta)
            with open(default_summary, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)
            trained_artifacts["default"] = {
                "model_path": str(default_model),
                "summary_path": str(default_summary),
            }

    log_csv = target_data_dir / "rl_training_logs.csv"
    pd.DataFrame(all_logs).to_csv(log_csv, index=False)
    trained_artifacts["training_logs"] = str(log_csv)

    return trained_artifacts


__all__ = [
    "evaluate_on_validation",
    "run_hparam_search",
    "train_final_agents",
    "train_q_learning",
]
