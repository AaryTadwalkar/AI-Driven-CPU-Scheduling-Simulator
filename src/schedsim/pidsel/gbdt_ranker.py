"""
1. What this file does:
   Implements Stage 15: Model 4 - Gradient-Boosted Candidate Ranker for process-level scheduling.
   Extracts tree ensembles from scikit-learn GradientBoostingClassifier into flat pure-NumPy
   arrays, performs branchless vectorized candidate scoring, prepares regret-weighted behavioral
   cloning training datasets, trains setting-specific rankers, provides a Policy-compliant
   GBDTRankerPolicy adapter, and evaluates closed-loop validation performance against Oracle and
   SRTF.
2. Overall design:
   At every non-forced decision point (len(ready) >= 2), the ranker scores candidate processes
   using a flat tree traversal without Python control loops per sample. Scored candidates are
   ranked with canonical tie-breaking (highest score -> smaller remaining burst -> earlier arrival
   -> pid). Pure NumPy inference eliminates scikit-learn runtime overhead during simulation.
3. Elements in the file:
   - export_tree_ensemble: Extracts flat tree dictionaries, learning rate, and initial logit.
   - score_candidates: Pure-NumPy vectorized candidate scoring over feature matrices.
   - prepare_training_data: Flattens NPZ dataset, computes optimal binary labels and weights.
   - train_gbdt: Fits GradientBoostingClassifier on behavioral cloning data and exports artifacts.
   - evaluate_gbdt_on_validation: Benchmarks ranker on validation workloads against Oracle and SRTF.
   - GBDTRankerPolicy: Policy subclass integrating the GBDT ranker into Engine.run().
4. Final output of the file:
   Exports GBDTRankerPolicy, export_tree_ensemble, score_candidates, prepare_training_data,
   train_gbdt, and evaluate_gbdt_on_validation for simulation, benchmarking, and CLI dispatch.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

import config
from config import DATA_DIR, GBDT_PARAMS, MODELS_DIR, RESULTS_DIR
from schedsim.domain.engine import Engine, View, run
from schedsim.domain.process import Process, validate_processes
from schedsim.pidsel.features import FEATURE_NAMES, extract_candidate_features
from schedsim.pidsel.oracle import RolloutOraclePolicy, evaluate_state
from schedsim.policies.base import Policy
from schedsim.policies.srtf import SRTFPolicy
from schedsim.workloads.io import Workload, load_workloads_from_csv
from schedsim.workloads.splits import generate_split_workloads


def export_tree_ensemble(
    model: GradientBoostingClassifier,
) -> tuple[list[dict[str, np.ndarray]], float, float]:
    """
    1. The use of function:
       Extracts flat, portable tree array dictionaries from a scikit-learn
       GradientBoostingClassifier instance for fast pure-NumPy candidate scoring.
    2. How does the specific function work:
       Iterates over the first column of estimators (binary classification). For each tree,
       extracts children_left ('l'), children_right ('r'), feature ('f'), threshold ('th'),
       and leaf value ('v'). Extracts learning_rate and initial bias value init_val from
       model._raw_predict_init.
    3. Concepts used in the function:
       Decision tree serialization, ensemble parameter extraction, pure-NumPy inference export.
    4. Places where this function is imported and used:
       Exported in schedsim.pidsel, used by train_gbdt, GBDTRankerPolicy, and Stage 15 tests.
    5. Final output of the function, before and after:
       Before: Fitted scikit-learn GradientBoostingClassifier instance.
       After: Tuple of (list of tree dictionaries, learning rate float, initial logit float).
    """
    if not isinstance(model, GradientBoostingClassifier):
        raise TypeError(
            f"model must be a GradientBoostingClassifier, got {type(model).__name__}."
        )

    trees: list[dict[str, np.ndarray]] = []
    for tree_estimator in model.estimators_[:, 0]:
        t = tree_estimator.tree_
        tree_dict = {
            "l": t.children_left.copy(),
            "r": t.children_right.copy(),
            "f": t.feature.copy(),
            "th": t.threshold.copy(),
            "v": t.value[:, 0, 0].copy(),
        }
        trees.append(tree_dict)

    lr = float(model.learning_rate)

    if hasattr(model, "_raw_predict_init"):
        init_val = float(
            model._raw_predict_init(np.zeros((1, getattr(model, "n_features_in_", 13))))[0, 0]
        )
    elif hasattr(model, "init_") and hasattr(model.init_, "class_prior_"):
        p = float(model.init_.class_prior_[1])
        init_val = float(np.log(p / (1.0 - p)))
    elif hasattr(model, "init_") and hasattr(model.init_, "predict"):
        init_val = float(
            model.init_.predict(np.zeros((1, getattr(model, "n_features_in_", 13))))[0]
        )
    else:
        init_val = 0.0

    return trees, lr, init_val


def score_candidates(
    trees: list[dict[str, np.ndarray]],
    lr: float,
    init_val: float,
    x: np.ndarray,
) -> np.ndarray:
    """
    1. The use of function:
       Computes fast vectorized prediction scores for candidate feature rows using pure NumPy.
    2. How does the specific function work:
       Validates input matrix x is 2D. Allocates output array initialized to init_val.
       For each tree in the ensemble, traverses the decision tree nodes simultaneously across
       all candidate rows using vectorized array indexing until all rows reach terminal leaf nodes.
       Accumulates scaled leaf values (lr * v) into the output array.
    3. Concepts used in the function:
       Vectorized tree traversal, NumPy array masking, branchless evaluation, ensemble scoring.
    4. Places where this function is imported and used:
       Exported in schedsim.pidsel, used by GBDTRankerPolicy.select and validation benchmarks.
    5. Final output of the function, before and after:
       Before: Candidate features matrix of shape (n_candidates, d).
       After: 1D float64 score array of shape (n_candidates,).
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"x must be a 2D array of shape (n_candidates, d), got shape {arr.shape}.")

    n_candidates = len(arr)
    if n_candidates == 0:
        return np.zeros(0, dtype=np.float64)

    out = np.full(n_candidates, init_val, dtype=np.float64)
    idx = np.arange(n_candidates)

    for t in trees:
        node = np.zeros(n_candidates, dtype=np.intp)
        while True:
            internal = t["l"][node] != -1
            if not internal.any():
                break
            feat = np.where(internal, t["f"][node], 0)
            go_left = arr[idx, feat] <= t["th"][node]
            node = np.where(internal, np.where(go_left, t["l"][node], t["r"][node]), node)
        out += lr * t["v"][node]

    return out


def prepare_training_data(
    npz_path: str | Path,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1. The use of function:
       Loads and prepares training samples, binary optimal labels, and regret-based weights
       from a process-level behavioral cloning NPZ dataset.
    2. How does the specific function work:
       Loads features, mask, costs, and regrets from npz_path. Flattens valid candidate rows
       using boolean mask. Computes minimum oracle cost per state across valid candidates.
       Marks candidates with cost <= best_cost * 1.01 + 1e-6 as optimal (label 1, else 0).
       Calculates sample weights as 1.0 + log1p(max(0.0, regret)). Truncates to max_samples
       if specified.
    3. Concepts used in the function:
       Dataset preparation, binary label conversion with tolerance, regret-proportional weighting.
    4. Places where this function is imported and used:
       Exported in schedsim.pidsel, called by train_gbdt and Stage 15 test suites.
    5. Final output of the function, before and after:
       Before: Compressed NPZ file containing padded states and oracle evaluations.
       After: Tuple of (X features [N, 13], y binary labels [N], sample_weights [N]).
    """
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}.")

    with np.load(path) as data:
        features = data["features"]
        mask = data["mask"]
        costs = data["costs"]
        regrets = data["regrets"]

    masked_costs = np.where(mask, costs, np.inf)
    min_costs = np.min(masked_costs, axis=1, keepdims=True)
    is_best_mat = (costs <= min_costs * 1.01 + 1e-6) & mask

    x_all = features[mask]
    y_all = is_best_mat[mask].astype(np.int32)
    valid_regrets = np.maximum(0.0, regrets[mask])
    weights_all = (1.0 + np.log1p(valid_regrets)).astype(np.float64)

    if max_samples is not None and max_samples < len(x_all):
        return x_all[:max_samples], y_all[:max_samples], weights_all[:max_samples]

    return x_all, y_all, weights_all


def train_gbdt(
    setting: str = "S0",
    data_dir: Path | str = DATA_DIR,
    models_dir: Path | str = MODELS_DIR,
    params: dict[str, Any] | None = None,
    max_states: int | None = None,
    random_state: int = 42,
) -> tuple[GradientBoostingClassifier, dict[str, Any]]:
    """
    1. The use of function:
       Trains a GradientBoostingClassifier candidate ranker for a specific setting and saves
       portable exported artifacts.
    2. How does the specific function work:
       Resolves setting (S0 or S1). Loads training dataset from NPZ file. Optionally caps to
       max_states. Prepares binary optimal labels and regret weights. Fits classifier using
       provided or default hyperparameters. Extracts flat tree arrays, saves model artifact
       to joblib and metadata summary to JSON.
    3. Concepts used in the function:
       Supervised behavioral cloning, gradient boosting, tree extraction, model artifact saving.
    4. Places where this function is imported and used:
       Exported in schedsim.pidsel, called by train_pid_gbdt CLI and Stage 15 tests.
    5. Final output of the function, before and after:
       Before: Raw dataset and configuration parameters.
       After: Tuple of (fitted GradientBoostingClassifier, exported model dictionary).
    """
    setting_key = setting.upper().strip()
    if setting_key not in config.SETTINGS:
        valid = ", ".join(config.SETTINGS.keys())
        raise ValueError(f"Unknown setting '{setting}'. Must be one of: {valid}.")
    cs_cost = config.SETTINGS[setting_key].cs_cost

    data_dir_path = Path(data_dir)
    models_dir_path = Path(models_dir)
    models_dir_path.mkdir(parents=True, exist_ok=True)

    npz_path = data_dir_path / f"pid_bc_{setting_key}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Dataset for {setting_key} not found at {npz_path}. Run build-pid-data first."
        )

    if max_states is not None:
        with np.load(npz_path) as data:
            f_slice = data["features"][:max_states]
            m_slice = data["mask"][:max_states]
            c_slice = data["costs"][:max_states]
            r_slice = data["regrets"][:max_states]
        masked_c = np.where(m_slice, c_slice, np.inf)
        min_c = np.min(masked_c, axis=1, keepdims=True)
        is_best_mat = (c_slice <= min_c * 1.01 + 1e-6) & m_slice
        x_train = f_slice[m_slice]
        y_train = is_best_mat[m_slice].astype(np.int32)
        valid_r = np.maximum(0.0, r_slice[m_slice])
        w_train = (1.0 + np.log1p(valid_r)).astype(np.float64)
    else:
        x_train, y_train, w_train = prepare_training_data(npz_path)

    gbdt_params = dict(GBDT_PARAMS)
    if params is not None:
        gbdt_params.update(params)
    gbdt_params["random_state"] = random_state

    model = GradientBoostingClassifier(**gbdt_params)
    model.fit(x_train, y_train, sample_weight=w_train)

    exported_trees, lr, init_val = export_tree_ensemble(model)

    model_artifact: dict[str, Any] = {
        "trees": exported_trees,
        "lr": lr,
        "init_val": init_val,
        "sklearn_model": model,
        "feature_names": list(FEATURE_NAMES),
        "setting": setting_key,
        "cs_cost": cs_cost,
        "params": gbdt_params,
    }

    joblib_path = models_dir_path / f"pid_gbdt_{setting_key}.joblib"
    joblib.dump(model_artifact, joblib_path)

    subsample_n = min(len(x_train), 10000)
    train_acc = float(model.score(x_train[:subsample_n], y_train[:subsample_n]))

    meta_dict = {
        "model_type": "GradientBoostingClassifier",
        "setting": setting_key,
        "cs_cost": cs_cost,
        "n_train_samples": int(len(x_train)),
        "train_sample_accuracy_subsample": train_acc,
        "feature_names": list(FEATURE_NAMES),
        "params": gbdt_params,
        "n_trees": len(exported_trees),
        "learning_rate": lr,
        "init_val": init_val,
    }
    meta_path = models_dir_path / f"pid_gbdt_{setting_key}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)

    return model, model_artifact


def evaluate_gbdt_on_validation(
    model_data: dict[str, Any] | str | Path,
    setting: str = "S0",
    n_workloads: int = 500,
    data_dir: Path | str = DATA_DIR,
    results_dir: Path | str = RESULTS_DIR,
) -> dict[str, Any]:
    """
    1. The use of function:
       Evaluates a trained GBDT candidate ranker on validation workloads, reporting top-1 agreement
       with Oracle, mean regret per decision, and closed-loop waiting time gaps vs Oracle and SRTF.
    2. How does the specific function work:
       Loads model artifacts if path provided. Resolves cs_cost from setting. Loads validation
       workloads from workloads.csv or generates them. Evaluates top-1 agreement with Oracle on
       non-forced decision states. Runs closed-loop simulations under GBDT ranker, SRTF, and
       Oracle policies. Computes relative gaps and persists results to results/pid_gbdt_val.json.
    3. Concepts used in the function:
       Validation evaluation, closed-loop simulation, Oracle agreement, comparative metrics.
    4. Places where this function is imported and used:
       Exported in schedsim.pidsel, called by train_pid_gbdt CLI and Stage 15 test suites.
    5. Final output of the function, before and after:
       Before: Model artifact, setting name, and evaluation configuration.
       After: Dictionary of validation metrics, persisted into results/pid_gbdt_val.json.
    """
    if isinstance(model_data, (str, Path)):
        artifact_path = Path(model_data)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}.")
        data: dict[str, Any] = joblib.load(artifact_path)
    else:
        data = model_data

    setting_key = setting.upper().strip()
    if setting_key not in config.SETTINGS:
        valid = ", ".join(config.SETTINGS.keys())
        raise ValueError(f"Unknown setting '{setting}'. Must be one of: {valid}.")
    cs_cost = config.SETTINGS[setting_key].cs_cost

    trees = data["trees"]
    lr = float(data["lr"])
    init_val = float(data["init_val"])

    csv_path = Path(data_dir) / "workloads.csv"
    val_workloads: list[Workload] = []
    if csv_path.exists():
        try:
            loaded = load_workloads_from_csv(csv_path)
            val_workloads = [w for w in loaded if w.workload_id.startswith("val")][:n_workloads]
        except (ValueError, OSError):
            val_workloads = []

    if len(val_workloads) < n_workloads:
        needed = n_workloads - len(val_workloads)
        val_workloads.extend(
            generate_split_workloads("val", n_workloads=needed, seed_offset=len(val_workloads))
        )
    val_workloads = val_workloads[:n_workloads]

    gbdt_policy = GBDTRankerPolicy(
        trees=trees, lr=lr, init_val=init_val, cs_cost=cs_cost, name=f"GBDT_{setting_key}"
    )
    srtf_policy = SRTFPolicy()
    oracle_policy = RolloutOraclePolicy(cs_cost=cs_cost)

    agreed_decisions = 0
    total_decisions = 0
    total_regret = 0.0

    j_policy_list: list[float] = []
    j_oracle_list: list[float] = []
    j_srtf_list: list[float] = []

    for w in val_workloads:
        res_p = run(gbdt_policy, w.processes, cs_cost=cs_cost)
        j_policy_list.append(res_p.simulation_metrics.avg_waiting)

        res_s = run(srtf_policy, w.processes, cs_cost=cs_cost)
        j_srtf_list.append(res_s.simulation_metrics.avg_waiting)

        if cs_cost == 0:
            j_oracle_list.append(res_s.simulation_metrics.avg_waiting)
        else:
            res_o = run(oracle_policy, w.processes, cs_cost=cs_cost)
            j_oracle_list.append(res_o.simulation_metrics.avg_waiting)

        # Evaluate decision-level top-1 agreement and regret on non-forced states
        eng = Engine(w.processes, cs_cost=cs_cost)
        v = eng.reset()
        while not eng.done:
            if len(v.ready) <= 1:
                p_step = v.ready[0].pid
            else:
                total_decisions += 1
                if cs_cost == 0:
                    min_rem = min(p.remaining for p in v.ready)
                    optimal_pids = {p.pid for p in v.ready if p.remaining == min_rem}
                    chosen_pid = gbdt_policy.select(v)
                    if chosen_pid in optimal_pids:
                        agreed_decisions += 1
                    proc_map = {p.pid: p for p in v.ready}
                    total_regret += float(proc_map[chosen_pid].remaining - min_rem)
                    p_step = chosen_pid
                else:
                    decision = evaluate_state(eng, base_policies=("SRTF", "SJF"))
                    best_c = decision.best_cost
                    optimal_pids = {
                        pid for pid, c in decision.costs.items() if c <= best_c * 1.01 + 1e-6
                    }
                    chosen_pid = gbdt_policy.select(v)
                    if chosen_pid in optimal_pids:
                        agreed_decisions += 1
                    total_regret += float(decision.regrets.get(chosen_pid, 0.0))
                    p_step = decision.best_pid
            step_info = eng.step(p_step)
            v = step_info.view

    top1_agreement = float(agreed_decisions / total_decisions) if total_decisions > 0 else 1.0
    mean_regret = float(total_regret / total_decisions) if total_decisions > 0 else 0.0
    mean_j_p = float(np.mean(j_policy_list))
    mean_j_o = float(np.mean(j_oracle_list))
    mean_j_s = float(np.mean(j_srtf_list))

    gap_o = float((mean_j_p - mean_j_o) / mean_j_o) if mean_j_o > 0 else 0.0
    gap_s = float((mean_j_p - mean_j_s) / mean_j_s) if mean_j_s > 0 else 0.0

    metrics: dict[str, Any] = {
        "setting": setting_key,
        "cs_cost": cs_cost,
        "n_workloads": len(val_workloads),
        "top1_agreement": top1_agreement,
        "mean_regret_per_decision": mean_regret,
        "total_non_forced_decisions": total_decisions,
        "agreed_decisions": agreed_decisions,
        "mean_j_policy": mean_j_p,
        "mean_j_oracle": mean_j_o,
        "mean_j_srtf": mean_j_s,
        "closed_loop_gap_oracle": gap_o,
        "closed_loop_gap_srtf": gap_s,
    }

    results_dir_path = Path(results_dir)
    results_dir_path.mkdir(parents=True, exist_ok=True)
    val_json_path = results_dir_path / "pid_gbdt_val.json"

    existing_data: dict[str, Any] = {}
    if val_json_path.exists():
        try:
            with open(val_json_path, encoding="utf-8") as f:
                loaded_json = json.load(f)
                if isinstance(loaded_json, dict):
                    existing_data = loaded_json
        except (json.JSONDecodeError, OSError):
            existing_data = {}

    existing_data[setting_key] = metrics
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2)

    return metrics


class GBDTRankerPolicy(Policy):
    """
    1. The use of class:
       Implements a Policy-compliant scheduling adapter utilizing a trained GBDT tree ensemble to
       rank ready processes at each non-forced scheduling decision point.
    2. What the class contains:
       - cs_cost: Integer context switch penalty.
       - trees: Flat tree array dictionaries.
       - lr: Ensemble learning rate.
       - init_val: Initial ensemble prediction bias logit.
       - select: Evaluates candidate features, scores candidates, breaks ties, returns PID.
       - from_cs_cost: Factory constructor loading model matching cs_cost.
       - from_setting: Factory constructor loading model matching benchmark setting.
       - on_start: AI startup latency accounting hook (returns 0.0 µs).
       - decision_cost_us: Per-decision AI latency accounting hook (returns 0.0 µs).
       - reset: Resets policy state between runs.
    3. Technologies used and why:
       - Policy ABC inheritance for seamless plug-in into Engine.run() and CLI simulation.
       - Pure-NumPy vectorized scoring eliminating scikit-learn runtime latency.
    4. Key design and overall design of the class:
       Provides fast process-level candidate ranking with deterministic tie-breaking.
    """

    name: str = "GBDTRankerPolicy"

    def __init__(
        self,
        trees: Sequence[dict[str, np.ndarray]] | None = None,
        lr: float = 0.1,
        init_val: float = 0.0,
        cs_cost: int = 0,
        model: GradientBoostingClassifier | None = None,
        setting: str | None = None,
        models_dir: Path | str = MODELS_DIR,
        name: str = "GBDTRankerPolicy",
    ) -> None:
        """
        1. The use of function:
           Initializes a GBDTRankerPolicy instance from trees, a model, or loaded setting artifacts.
        2. How does the specific function work:
           Validates inputs. If trees are passed, stores them. Else if model is passed, extracts
           flat tree arrays via export_tree_ensemble. Else loads trained model artifact for setting.
        3. Concepts used in the function:
           Constructor overloading, defensive argument validation, lazy artifact resolution.
        4. Places where this function is imported and used:
           Directly instantiated by simulations, evaluation runners, and registry factory.
        5. Final output of the function, before and after:
           Before: Uninitialized policy reference.
           After: Fully configured GBDTRankerPolicy ready for simulation execution.
        """
        super().__init__(name=name)
        if isinstance(cs_cost, bool) or not isinstance(cs_cost, int) or cs_cost < 0:
            raise ValueError(f"cs_cost must be a non-negative int, got {cs_cost}.")

        if trees is not None:
            self.trees: list[dict[str, np.ndarray]] = list(trees)
            self.lr: float = float(lr)
            self.init_val: float = float(init_val)
            self.cs_cost: int = int(cs_cost)
        elif model is not None:
            exp_trees, exp_lr, exp_init = export_tree_ensemble(model)
            self.trees = exp_trees
            self.lr = exp_lr
            self.init_val = exp_init
            self.cs_cost = int(cs_cost)
        else:
            resolved_setting = setting or f"S{cs_cost}"
            instance = self.from_setting(resolved_setting, models_dir=models_dir, name=name)
            self.trees = instance.trees
            self.lr = instance.lr
            self.init_val = instance.init_val
            self.cs_cost = instance.cs_cost

        self._processes: tuple[Process, ...] | None = None

    @classmethod
    def from_cs_cost(
        cls,
        cs_cost: int = 0,
        models_dir: Path | str = MODELS_DIR,
        name: str | None = None,
    ) -> "GBDTRankerPolicy":
        """
        1. The use of function:
           Factory method instantiating GBDTRankerPolicy with the trained model for cs_cost.
        2. How does the specific function work:
           Resolves model file path matching cs_cost (pid_gbdt_S{cs_cost}.joblib or .pkl),
           loads artifact dictionary via joblib, and instantiates GBDTRankerPolicy.
        3. Concepts used in the function:
           Factory pattern, artifact persistence loading, setting encapsulation.
        4. Places where this function is imported and used:
           Used by simulation runners, CLI handlers, and unit tests.
        5. Final output of the function, before and after:
           Before: cs_cost integer and optional directory.
           After: Configured GBDTRankerPolicy instance.
        """
        dir_path = Path(models_dir)
        target_path = dir_path / f"pid_gbdt_S{cs_cost}.joblib"
        if not target_path.exists():
            fallback_path = dir_path / f"pid_gbdt_S{cs_cost}.pkl"
            if fallback_path.exists():
                target_path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Trained GBDT model for cs_cost={cs_cost} not found at {target_path}."
                )

        data = joblib.load(target_path)
        policy_name = name or f"GBDT_cs{cs_cost}"
        return cls(
            trees=data["trees"],
            lr=float(data["lr"]),
            init_val=float(data["init_val"]),
            cs_cost=cs_cost,
            name=policy_name,
        )

    @classmethod
    def from_setting(
        cls,
        setting: str = "S0",
        models_dir: Path | str = MODELS_DIR,
        name: str | None = None,
    ) -> "GBDTRankerPolicy":
        """
        1. The use of function:
           Factory method instantiating GBDTRankerPolicy with the trained model for a setting.
        2. How does the specific function work:
           Validates setting name against config.SETTINGS, resolves corresponding cs_cost,
           and delegates to cls.from_cs_cost().
        3. Concepts used in the function:
           Setting resolution, classmethod factory pattern.
        4. Places where this function is imported and used:
           Used by multi-setting benchmarks, CLI dispatch, and acceptance tests.
        5. Final output of the function, before and after:
           Before: Setting name string.
           After: Configured GBDTRankerPolicy instance.
        """
        setting_key = setting.upper().strip()
        if setting_key not in config.SETTINGS:
            valid = ", ".join(config.SETTINGS.keys())
            raise ValueError(f"Unknown setting '{setting}'. Must be one of: {valid}.")
        cs_cost = config.SETTINGS[setting_key].cs_cost
        policy_name = name or f"GBDT_{setting_key}"
        return cls.from_cs_cost(
            cs_cost=cs_cost,
            models_dir=models_dir,
            name=policy_name,
        )

    def bind_workload(self, processes: Sequence[Process]) -> None:
        """
        1. The use of function:
           Binds a workload process sequence to track processes across the simulation episode.
        2. How does the specific function work:
           Validates processes using validate_processes and stores defensive immutable tuple.
        3. Concepts used in the function:
           Workload binding lifecycle hook, defensive copying.
        4. Places where this function is imported and used:
           Invoked by engine runner run() before simulation execution loops.
        5. Final output of the function, before and after:
           Before: Unbound policy or previous workload state.
           After: Policy storing bound workload process tuple.
        """
        validate_processes(processes)
        self._processes = tuple(processes)

    def reset(self) -> None:
        """
        1. The use of function:
           Resets policy internal state between simulation runs.
        2. How does the specific function work:
           Stateless ranker; keeps configuration intact and resets cached workload processes.
        3. Concepts used in the function:
           Lifecycle hook, test isolation.
        4. Places where this function is imported and used:
           Called by Engine.run() between simulation episodes.
        5. Final output of the function, before and after:
           Before: Active policy state.
           After: Reset policy state.
        """
        self._processes = None

    def on_start(self, processes: Sequence[Process]) -> float:
        """
        1. The use of function:
           Lifecycle hook returning upfront model loading latency in microseconds.
        2. How does the specific function work:
           Binds workload processes and returns 0.0 µs baseline latency placeholder.
        3. Concepts used in the function:
           Lifecycle accounting, latency profiling hook.
        4. Places where this function is imported and used:
           Called by Engine.run() prior to simulation start.
        5. Final output of the function, before and after:
           Before: Workload process sequence.
           After: Returns float latency in microseconds (0.0).
        """
        self.bind_workload(processes)
        return 0.0

    def decision_cost_us(self, view: View) -> float:
        """
        1. The use of function:
           Returns the estimated decision latency in microseconds from the latency profile.
        2. How does the specific function work:
           Returns 0.0 if view is None or ready count is <= 1 (forced decision). Otherwise
           queries get_interpolated_cost_us("GBDT", len(view.ready)).
        3. Concepts used in the function:
           Per-decision overhead accounting, latency profile lookup, interpolation.
        4. Places where this function is imported and used:
           Called by Engine.step() after each non-forced scheduling decision.
        5. Final output of the function, before and after:
           Before: Current simulation state View.
           After: Returns float latency in microseconds.
        """
        if view is None or (config.FORCED_DECISION_FREE and len(view.ready) <= 1):
            return 0.0
        from schedsim.overhead import get_interpolated_cost_us

        return get_interpolated_cost_us("GBDT", len(view.ready))

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the next process PID to execute using pure-NumPy GBDT candidate ranking.
        2. How does the specific function work:
           Validates ready queue is non-empty. For forced decisions (len <= 1), returns the single
           ready PID immediately. For non-forced decisions, extracts candidate feature matrix X,
           evaluates scores using score_candidates, breaks ties canonically (highest score ->
           smaller remaining burst -> earlier arrival -> pid string), and returns chosen PID.
        3. Concepts used in the function:
           Fast candidate scoring, deterministic multi-tier tie-breaking, forced-decision bypass.
        4. Places where this function is imported and used:
           Invoked at each discrete simulation step by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Simulation View snapshot.
           After: Returns selected PID string to execute on CPU.
        """
        if not view.ready:
            raise ValueError("Cannot select process: ready queue is empty.")

        if len(view.ready) <= 1:
            return view.ready[0].pid

        x_cands, candidate_pids = extract_candidate_features(
            view, cs_cost=self.cs_cost, pending_arrivals=view.pending_arrivals
        )
        scores = score_candidates(self.trees, self.lr, self.init_val, x_cands)

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
        return best_pair[0]


__all__ = [
    "GBDTRankerPolicy",
    "evaluate_gbdt_on_validation",
    "export_tree_ensemble",
    "prepare_training_data",
    "score_candidates",
    "train_gbdt",
]
