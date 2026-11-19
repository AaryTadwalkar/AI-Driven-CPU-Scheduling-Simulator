"""
1. What this file does:
   Defines comprehensive unit and acceptance tests for Stage 12: Category 1 (Random Forest and
   Per-Setting Retraining) of the CPU Scheduling Simulator.
2. Overall design:
   Validates all Stage 12 deliverables in accordance with REVISED_IMPLEMENTATION_PLAN.md:
   - RF_PARAM_GRID specification in config.py.
   - Dataset generation and pilot-gate class distribution compliance (< 85% share) for S0 and S1.
   - Trained model artifact existence, valid schema, and ability to beat majority-class baseline.
   - Validation regret computation and comparative assessment between Random Forest and
     Decision Tree.
   - Public recommender inference API across models ('dt', 'rf') and settings ('S0', 'S1').
   - SelectorPolicy lifecycle hooks (on_start upfront charge, 0.0 decision cost, reset, execution).
   - RLPolicy per-setting constructors and explicit decision_cost_us hook.
3. Elements in the file:
   - test_stage12_config_rf_param_grid: Verifies RF parameter grid in config.
   - test_stage12_dataset_per_setting_class_distribution: Checks S0/S1 distributions & pilot gate.
   - test_stage12_model_artifacts_exist_and_load: Verifies saved joblib and JSON artifacts.
   - test_stage12_models_beat_majority_baseline: Asserts DT and RF outperform dummy baseline.
   - test_stage12_validation_regret_and_rf_comparison: Compares DT and RF on validation regret.
   - test_stage12_recommender_api: Validates recommend() API across models and settings.
   - test_stage12_selector_policy_execution: Tests SelectorPolicy and registry aliases in run().
   - test_stage12_rl_policy_per_setting: Tests RLPolicy per-setting factory and lifecycle hooks.
   - test_stage12_train_category1_pipeline_execution: Tests isolated training on small slice.
4. Final output of the file:
   Pytest test suite executing verification of Category 1 scheduling components.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

import config
from schedsim.domain.engine import View, run
from schedsim.domain.process import Process
from schedsim.ml.dataset_builder import (
    check_class_distribution,
    load_dataset_from_csv,
    load_or_generate_setting_dataset,
)
from schedsim.ml.evaluate_tree import recommend
from schedsim.ml.train_forest import train_category1_for_setting
from schedsim.policies.base import Policy
from schedsim.policies.registry import get_policy
from schedsim.policies.selector import DTPolicy, RFPolicy, SelectorPolicy
from schedsim.rl.policy import RLPolicy


def test_stage12_config_rf_param_grid() -> None:
    """
    1. The use of function:
       Tests the existence and correct schema of RF_PARAM_GRID in config.py.
    2. How does the specific function work:
       Checks that RF_PARAM_GRID is a dictionary containing n_estimators, max_depth, and
       min_samples_leaf with expected parameter search values.
    3. Concepts used in the function:
       Configuration validation, parameter grid specification.
    4. Places where this function is imported and used:
       Executed by pytest during Stage 12 verification.
    5. Final output of the function, before and after:
       Before: Unverified configuration dictionary.
       After: Confirms RF_PARAM_GRID matches Section 3 specification.
    """
    assert hasattr(config, "RF_PARAM_GRID")
    grid = config.RF_PARAM_GRID
    assert isinstance(grid, dict)
    assert "n_estimators" in grid
    assert "max_depth" in grid
    assert "min_samples_leaf" in grid
    assert 50 in grid["n_estimators"]
    assert 100 in grid["n_estimators"]


def test_stage12_dataset_per_setting_class_distribution() -> None:
    """
    1. The use of function:
       Validates S0 and S1 datasets and verifies the Stage 4c pilot gate rule (< 85% share).
    2. How does the specific function work:
       Loads S0 and S1 datasets, validates split row counts (6000 train, 500 val), runs
       check_class_distribution to assert no class has >= 85% share, and tests that
       check_class_distribution raises ValueError on an artificially skewed dataset.
    3. Concepts used in the function:
       Class imbalance detection, pilot gate rule, dataset schema validation.
    4. Places where this function is imported and used:
       Executed by pytest during Stage 12 verification.
    5. Final output of the function, before and after:
       Before: S0 and S1 class balance unverified.
       After: Confirms both datasets pass the pilot gate.
    """
    for setting in ("S0", "S1"):
        df = load_or_generate_setting_dataset(setting)
        assert len(df) == 6500
        assert (df["split"] == "train").sum() == 6000
        assert (df["split"] == "val").sum() == 500

        shares = check_class_distribution(df, max_share=0.85)
        for cls_name, share in shares.items():
            assert share < 0.85, f"Class {cls_name} exceeded 85% in {setting}!"
            assert share > 0.0

    # Test that check_class_distribution raises ValueError on skewed dataset
    skewed_df = pd.DataFrame({"label": ["RR"] * 90 + ["SJF"] * 10})
    with pytest.raises(ValueError, match="pilot gate failed"):
        check_class_distribution(skewed_df, max_share=0.85)


def test_stage12_model_artifacts_exist_and_load() -> None:
    """
    1. The use of function:
       Verifies that all required model and metadata artifacts exist in models/ and load cleanly.
    2. How does the specific function work:
       Checks existence and loads joblib models and JSON metadata for decision_tree_S0,
       decision_tree_S1, random_forest_S0, random_forest_S1, and legacy decision_tree.
    3. Concepts used in the function:
       Model persistence verification, schema integrity, backward compatibility.
    4. Places where this function is imported and used:
       Executed by pytest during Stage 12 verification.
    5. Final output of the function, before and after:
       Before: Model artifact integrity unverified.
       After: Confirms all artifacts are intact and loadable.
    """
    expected_prefixes = [
        "decision_tree_S0",
        "random_forest_S0",
        "decision_tree_S1",
        "random_forest_S1",
        "decision_tree",
    ]
    for prefix in expected_prefixes:
        model_path = config.MODELS_DIR / f"{prefix}.joblib"
        meta_path = config.MODELS_DIR / f"{prefix}_meta.json"
        assert model_path.exists(), f"Missing artifact: {model_path}"
        assert meta_path.exists(), f"Missing metadata: {meta_path}"

        model = joblib.load(model_path)
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        assert "classes" in meta
        assert "training_samples" in meta


def test_stage12_models_beat_majority_baseline() -> None:
    """
    1. The use of function:
       Tests that DT and RF models for both S0 and S1 beat the majority-class baseline macro-F1.
    2. How does the specific function work:
       Loads metadata JSON for each trained model, asserts beats_majority_baseline is True,
       and checks that model validation macro-F1 exceeds dummy baseline macro-F1.
    3. Concepts used in the function:
       Comparative baseline evaluation, macro-F1 score assessment.
    4. Places where this function is imported and used:
       Executed by pytest during Stage 12 verification.
    5. Final output of the function, before and after:
       Before: Model superiority over majority baseline unverified.
       After: Confirms all models outperform baseline on macro-F1.
    """
    for setting in ("S0", "S1"):
        for m_type in ("decision_tree", "random_forest"):
            meta_path = config.MODELS_DIR / f"{m_type}_{setting}_meta.json"
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            assert meta["beats_majority_baseline"] is True
            val_f1 = meta["metrics_validation"]["macro_f1"]
            dummy_f1 = meta["majority_class_baseline"]["macro_f1"]
            assert val_f1 > dummy_f1, f"{m_type}_{setting} failed to beat dummy F1!"


def test_stage12_validation_regret_and_rf_comparison() -> None:
    """
    1. The use of function:
       Validates validation regret computation and checks RF vs DT comparison metrics.
    2. How does the specific function work:
       Loads metadata JSON for both S0 and S1, checks that mean_regret is non-negative,
       validates that rf_vs_dt_regret_delta is computed, and confirms RF achieves equal or
       lower validation regret than DT.
    3. Concepts used in the function:
       Decision regret metric, model ensemble comparison.
    4. Places where this function is imported and used:
       Executed by pytest during Stage 12 verification.
    5. Final output of the function, before and after:
       Before: Validation regret comparison unverified.
       After: Confirms regret calculation and RF comparison are valid.
    """
    for setting in ("S0", "S1"):
        dt_meta_path = config.MODELS_DIR / f"decision_tree_{setting}_meta.json"
        rf_meta_path = config.MODELS_DIR / f"random_forest_{setting}_meta.json"

        with open(dt_meta_path, encoding="utf-8") as f:
            dt_meta = json.load(f)
        with open(rf_meta_path, encoding="utf-8") as f:
            rf_meta = json.load(f)

        dt_regret = dt_meta["metrics_validation"]["mean_regret"]
        rf_regret = rf_meta["metrics_validation"]["mean_regret"]

        assert dt_regret >= 0.0
        assert rf_regret >= 0.0

        assert "rf_vs_dt_regret_delta" in rf_meta
        delta = rf_meta["rf_vs_dt_regret_delta"]
        assert pytest.approx(rf_regret - dt_regret, abs=1e-6) == delta
        assert rf_meta["rf_beats_dt_regret"] is True


def test_stage12_recommender_api() -> None:
    """
    1. The use of function:
       Tests the public recommend() inference API across models and settings.
    2. How does the specific function work:
       Passes sample processes to recommend() with model='dt'/'rf' and setting='S0'/'S1'.
       Validates returned dictionary keys ('algorithm', 'probabilities', 'features'),
       verifies probability normalization, and asserts invalid arguments raise errors.
    3. Concepts used in the function:
       Inference API verification, probability distribution assertion, defensive error checking.
    4. Places where this function is imported and used:
       Executed by pytest during Stage 12 verification.
    5. Final output of the function, before and after:
       Before: recommend() multi-model multi-setting support unverified.
       After: Confirms recommend() functions correctly across all configurations.
    """
    procs = [
        Process("P1", 0, 10, 1),
        Process("P2", 1, 2, 2),
        Process("P3", 2, 4, 1),
    ]

    # Test backward-compatible default call
    rec_default = recommend(procs)
    assert isinstance(rec_default, dict)
    assert rec_default["algorithm"] in ("FCFS", "SJF", "RR")
    assert pytest.approx(sum(rec_default["probabilities"].values()), abs=1e-5) == 1.0
    assert len(rec_default["features"]) == 12

    # Test explicit model and setting combinations
    for m in ("dt", "rf", "decision_tree", "random_forest"):
        for s in ("S0", "S1"):
            rec = recommend(procs, model=m, setting=s)
            assert rec["algorithm"] in ("FCFS", "SJF", "RR")
            assert pytest.approx(sum(rec["probabilities"].values()), abs=1e-5) == 1.0

    # Error handling tests
    with pytest.raises(ValueError, match="Unknown model type"):
        recommend(procs, model="invalid_model")

    with pytest.raises(ValueError, match="Unknown setting"):
        recommend(procs, setting="S99")

    with pytest.raises(TypeError, match="processes must be a Sequence"):
        recommend("not_processes")  # type: ignore[arg-type]


def test_stage12_selector_policy_execution() -> None:
    """
    1. The use of function:
       Tests SelectorPolicy lifecycle, execution, and registry integration.
    2. How does the specific function work:
       Tests SelectorPolicy instantiation, verifies on_start() runs recommendation and
       instantiates delegate, verifies select() forwards to delegate, confirms decision_cost_us()
       returns 0.0, checks reset() clears state, and verifies get_policy('DT') and get_policy('RF').
    3. Concepts used in the function:
       Policy delegation, lifecycle hooks, simulation harness integration.
    4. Places where this function is imported and used:
       Executed by pytest during Stage 12 verification.
    5. Final output of the function, before and after:
       Before: SelectorPolicy execution unverified.
       After: Confirms SelectorPolicy and aliases run seamlessly in simulations.
    """
    procs = [
        Process("P1", 0, 8, 1),
        Process("P2", 1, 3, 2),
    ]

    # 1. Direct instantiation
    policy_dt = SelectorPolicy(model="dt", setting="S1")
    assert policy_dt.model == "dt"
    assert policy_dt.setting == "S1"
    assert policy_dt.decision_cost_us(None) == 0.0  # type: ignore[arg-type]

    # Precondition: select before on_start raises RuntimeError
    dummy_view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(RuntimeError, match="select\\(\\) invoked before on_start"):
        policy_dt.select(dummy_view)

    # 2. Run simulation with DT policy
    result_dt = run(policy_dt, procs, cs_cost=1)
    assert result_dt.simulation_metrics.makespan > 0
    assert policy_dt.chosen_algorithm in ("FCFS", "SJF", "RR")
    assert policy_dt.recommendation is not None

    # Reset clears state
    policy_dt.reset()
    assert policy_dt.chosen_algorithm is None
    assert policy_dt._delegate is None

    # 3. Direct instantiation of subclasses
    p_dt_sub = DTPolicy(setting="S0")
    assert isinstance(p_dt_sub, SelectorPolicy)
    assert p_dt_sub.model == "dt"
    assert p_dt_sub.setting == "S0"

    p_rf_sub = RFPolicy(setting="S1")
    assert isinstance(p_rf_sub, SelectorPolicy)
    assert p_rf_sub.model == "rf"
    assert p_rf_sub.setting == "S1"

    # 4. Registry aliases DT and RF
    reg_dt = get_policy("DT")
    assert isinstance(reg_dt, SelectorPolicy)
    assert reg_dt.model == "dt"

    reg_rf = get_policy("RF")
    assert isinstance(reg_rf, SelectorPolicy)
    assert reg_rf.model == "rf"

    result_rf = run(reg_rf, procs, cs_cost=1)
    assert result_rf.simulation_metrics.makespan > 0
    assert reg_rf.chosen_algorithm in ("FCFS", "SJF", "RR")


def test_stage12_rl_policy_per_setting() -> None:
    """
    1. The use of function:
       Tests RLPolicy per-setting factory methods and lifecycle hooks.
    2. How does the specific function work:
       Tests RLPolicy.from_setting('S0') and from_setting('S1'), verifies on_start() returns
       0.0 µs and binds workload, verifies decision_cost_us() returns 0.0 µs, and executes
       simulations through run() at both cs_cost=0 and cs_cost=1.
    3. Concepts used in the function:
       Factory pattern, lifecycle hooks, tabular RL policy dispatch.
    4. Places where this function is imported and used:
       Executed by pytest during Stage 12 verification.
    5. Final output of the function, before and after:
       Before: RLPolicy per-setting integration unverified.
       After: Confirms RLPolicy operates correctly across benchmark settings.
    """
    procs = [
        Process("P1", 0, 6, 1),
        Process("P2", 2, 3, 2),
    ]

    # Test from_setting factory
    policy_s0 = RLPolicy.from_setting("S0")
    policy_s1 = RLPolicy.from_setting("S1")
    assert isinstance(policy_s0, Policy)
    assert isinstance(policy_s1, Policy)

    # Test from_cs_cost factory
    policy_cs0 = RLPolicy.from_cs_cost(0)
    policy_cs1 = RLPolicy.from_cs_cost(1)
    assert isinstance(policy_cs0, Policy)
    assert isinstance(policy_cs1, Policy)

    # Lifecycle hooks
    assert policy_s0.decision_cost_us(None) == 0.0  # type: ignore[arg-type]
    start_cost = policy_s0.on_start(procs)
    assert start_cost == 0.0
    assert policy_s0._processes is not None

    # End-to-end runs
    res_s0 = run(policy_s0, procs, cs_cost=0)
    assert res_s0.simulation_metrics.makespan > 0

    res_s1 = run(policy_s1, procs, cs_cost=1)
    assert res_s1.simulation_metrics.makespan > 0

    # Error handling
    with pytest.raises(ValueError, match="Unknown setting"):
        RLPolicy.from_setting("INVALID")


def test_stage12_train_category1_pipeline_execution(tmp_path: Path) -> None:
    """
    1. The use of function:
       Tests train_category1_for_setting on an isolated data slice in a temporary directory.
    2. How does the specific function work:
       Creates a small valid labeled slice of the S1 dataset in tmp_path, fits DT and RF models,
       verifies returned summary dictionary structure, and confirms artifact file generation.
    3. Concepts used in the function:
       Integration testing, isolated environment verification, temporary fixture usage.
    4. Places where this function is imported and used:
       Executed by pytest during Stage 12 verification.
    5. Final output of the function, before and after:
       Before: Isolated training execution unverified.
       After: Confirms training pipeline completes cleanly on custom datasets.
    """
    df_s1 = load_dataset_from_csv(config.DATA_DIR / "scheduling_dataset_S1.csv")
    train_slice = df_s1[df_s1["split"] == "train"].iloc[:100]
    val_slice = df_s1[df_s1["split"] == "val"].iloc[:30]
    test_df = pd.concat([train_slice, val_slice], ignore_index=True)

    result = train_category1_for_setting(
        setting="S1",
        dataset=test_df,
        models_dir=tmp_path,
        seed=42,
    )

    assert result["setting"] == "S1"
    assert "decision_tree" in result
    assert "random_forest" in result
    assert "majority_baseline" in result
    assert "comparison" in result

    assert (tmp_path / "decision_tree_S1.joblib").exists()
    assert (tmp_path / "decision_tree_S1_meta.json").exists()
    assert (tmp_path / "random_forest_S1.joblib").exists()
    assert (tmp_path / "random_forest_S1_meta.json").exists()
    assert (tmp_path / "decision_tree.joblib").exists()
