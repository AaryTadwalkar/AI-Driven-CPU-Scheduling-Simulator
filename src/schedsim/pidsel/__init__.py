"""
1. What this file does:
   Defines the package root for the process-level scheduling module (schedsim.pidsel),
   exporting the rollout oracle, decision structures, candidate feature extractors, dataset
   builders, dataset report generators, the Stage 15 GBDT candidate ranker, and the Stage 16
   neural candidate scorer.
2. Overall design:
   Packages process-level PID scheduling components, including Stage 13 rollout oracle,
   Stage 14 candidate feature extractors and dataset generators, Stage 15 GBDT ranker, and
   Stage 16 neural candidate scorer. Exposes high-level interfaces for lookahead search,
   trajectory labelling, feature extraction, behavioral cloning dataset persistence, pure-NumPy
   GBDT ranking, and pure-NumPy neural candidate scoring.
3. Elements in the file:
   - OracleDecision: Dataclass recording oracle dispatch decision, candidate costs, and regrets.
   - OracleStepRecord: Dataclass capturing simulation step context and oracle decision.
   - RolloutOraclePolicy: Policy wrapper executing oracle search with synchronized shadow engine.
   - evaluate_state: Evaluates ready process candidates via 1-step lookahead rollouts.
   - label_workloads_parallel: Batch parallel oracle execution preserving input sequence ordering.
   - run_oracle: Simulates full workload under oracle guidance with trajectory recording.
   - FEATURE_NAMES: Canonical tuple of the 13 feature string identifiers.
   - compute_normalized_rank: Symmetric fractional rank normalization in [0, 1].
   - extract_candidate_features: Vectorized observable candidate feature extraction from View.
   - pad_and_stack_features: Tensor padding utility with boolean validity masks.
   - build_pid_dataset: Collects non-forced oracle decision states into behavioral cloning dataset.
   - generate_and_save_pid_datasets: Generates and persists S0 and S1 datasets to NPZ.
   - generate_dataset_report: Compiles comparative Markdown report contrasting S0 and S1 datasets.
   - GBDTRankerPolicy: Policy subclass implementing the GBDT candidate ranking policy.
   - export_tree_ensemble: Extracts flat tree dictionaries and parameters from scikit-learn GBDT.
   - score_candidates: Branchless pure-NumPy vectorized scoring over candidate feature matrices.
   - train_gbdt: Trains GBDT candidate ranker and exports model artifacts.
   - evaluate_gbdt_on_validation: Benchmarks GBDT ranker on validation workloads vs Oracle and SRTF.
   - Scorer: PyTorch neural candidate scorer module (~3.6k parameters).
   - export_numpy_weights: Extracts neural network weights to flat NumPy float32 dictionary.
   - numpy_score: Vectorized pure-NumPy neural candidate scoring.
   - NeuralCandidatePolicy: Policy subclass implementing pure-NumPy neural candidate scoring.
   - soft_cross_entropy_loss: Temperature-scaled cross-entropy loss over masked candidate regrets.
   - evaluate_neural_on_validation: Benchmarks neural policy on validation workloads.
   - train_neural_scorer: Multi-seed DAgger training and evaluation pipeline for neural scorer.
4. Final output of the file:
   Exported symbols accessible via `from schedsim.pidsel import ...`.
"""

from schedsim.pidsel.dataset import (
    build_pid_dataset,
    generate_and_save_pid_datasets,
    generate_dataset_report,
)
from schedsim.pidsel.features import (
    FEATURE_NAMES,
    compute_normalized_rank,
    extract_candidate_features,
    pad_and_stack_features,
)
from schedsim.pidsel.gbdt_ranker import (
    GBDTRankerPolicy,
    evaluate_gbdt_on_validation,
    export_tree_ensemble,
    prepare_training_data,
    score_candidates,
    train_gbdt,
)
from schedsim.pidsel.oracle import (
    OracleDecision,
    OracleStepRecord,
    RolloutOraclePolicy,
    evaluate_state,
    label_workloads_parallel,
    run_oracle,
)
from schedsim.pidsel.scorer_net import (
    NeuralCandidatePolicy,
    Scorer,
    export_numpy_weights,
    numpy_score,
)
from schedsim.pidsel.train_bc import (
    evaluate_neural_on_validation,
    soft_cross_entropy_loss,
    train_neural_scorer,
)

__all__ = [
    "FEATURE_NAMES",
    "GBDTRankerPolicy",
    "NeuralCandidatePolicy",
    "OracleDecision",
    "OracleStepRecord",
    "RolloutOraclePolicy",
    "Scorer",
    "build_pid_dataset",
    "compute_normalized_rank",
    "evaluate_gbdt_on_validation",
    "evaluate_neural_on_validation",
    "evaluate_state",
    "export_numpy_weights",
    "export_tree_ensemble",
    "extract_candidate_features",
    "generate_and_save_pid_datasets",
    "generate_dataset_report",
    "label_workloads_parallel",
    "numpy_score",
    "pad_and_stack_features",
    "prepare_training_data",
    "run_oracle",
    "score_candidates",
    "soft_cross_entropy_loss",
    "train_gbdt",
    "train_neural_scorer",
]
