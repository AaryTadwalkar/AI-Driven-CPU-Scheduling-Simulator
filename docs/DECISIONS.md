# Architectural Decisions Log

This document records key technical decisions, rationale, deviations from the baseline plan, and alternatives evaluated/rejected throughout the implementation of the AI-Driven CPU Scheduling Simulator.

---

## Decision 001: Unified Unit-Step Simulation Engine
- **Status:** Accepted
- **Context:** Classical algorithms (FCFS, SJF, RR) and RL agent require strict metric parity and identical context switch modeling.
- **Decision:** Implement one common unit-step engine (`Engine.step()`) operating in discrete integer ticks. All policies (classical, ML-chosen, or RL) submit actions through this unified engine.
- **Rationale:** Prevents metric distortion and unfair comparisons between preemptive and non-preemptive algorithms or between classical and RL policies.
- **Alternatives Rejected:** Implementing separate closed-form simulators for each baseline.

## Decision 002: Permutation-Invariant State Space for RL
- **Status:** Accepted
- **Context:** Per-process state representations explode combinatorially and fail to generalize when process IDs or counts vary.
- **Decision:** Use aggregate queue features (e.g., ready queue length, burst variance, remaining time statistics) with rule-based action dispatch.
- **Rationale:** Keeps tabular Q-learning state space compact (1k–2k states), enabling sample-efficient convergence within 20,000 episodes.
- **Alternatives Rejected:** Full process slot representation (rejected due to state explosion and memorization).

## Decision 003: Composite Objective for Decision Tree Labeling
- **Status:** Accepted
- **Context:** Using average waiting time alone leads SJF to dominate almost all non-preemptive workloads, causing class collapse.
- **Decision:** Label workloads using a weighted composite score (50% waiting time, 30% response time, 20% turnaround time) with context switch cost included.
- **Rationale:** Reflects realistic multi-objective OS scheduling and produces balanced, non-trivial classification targets.
- **Alternatives Rejected:** Single-metric waiting time labeling.

## Decision 004: Section 4c Pilot Gate Setting Selection
- **Status:** Accepted
- **Context:** Stage 4c mandates evaluating 600 pilot workloads across candidate labeling configurations before building the full training dataset.
- **Empirical Results (600 Workloads):**
  - `cs0_q2_standard` (`cs_cost=0, q=2, weights=(0.5, 0.3, 0.2)`):
    * FCFS: 59 (9.8%), SJF: 182 (30.3%), RR: 359 (59.8%). Ambiguous: 12.3%.
  - `cs1_q2_standard` (`cs_cost=1, q=2, weights=(0.5, 0.3, 0.2)`):
    * FCFS: 123 (20.5%), SJF: 305 (50.8%), RR: 172 (28.7%). Ambiguous: 21.3%.
  - `cs1_q4_standard` (`cs_cost=1, q=4, weights=(0.5, 0.3, 0.2)`):
    * FCFS: 122 (20.3%), SJF: 333 (55.5%), RR: 145 (24.2%). Ambiguous: 23.5%.
  - `cs1_q2_balanced_weights` (`cs_cost=1, q=2, weights=(0.333, 0.333, 0.334)`):
    * FCFS: 123 (20.5%), SJF: 300 (50.0%), RR: 177 (29.5%). Ambiguous: 21.3%.
- **Decision:** Adopt `cs1_q2_standard` (`cs_cost=1, quantum=2, weights={"wait": 0.5, "resp": 0.3, "tat": 0.2}`) for full dataset construction.
- **Rationale:**
  1. Setting `cs_cost=1` realistically models CPU preemption context switch overheads, preventing artificial Round Robin inflation.
  2. The resulting class distribution is well balanced across all three candidate policies: FCFS (20.5%), SJF (50.8%), and RR (28.7%).
  3. No class approaches the 85% dominance threshold, creating an informative, multi-class learning problem.
  4. Preserves the canonical assignment baseline parameters defined in `config.py`.
- **Alternatives Rejected:**
  - `cs0_q2_standard`: Rejected because 0 context switch penalty artificially favors RR preemption.
  - `cs1_q4_standard`: Larger quantum (q=4) slightly degrades RR responsiveness without changing the balance advantage.

## Decision 005: Strict 5-Model Roster Scope & Deprecations
- **Status:** Accepted
- **Context:** `REVISED_IMPLEMENTATION_PLAN.md` supersedes the complex model scope of `PID_SELECTOR_ADDENDUM_PLAN.md` (Stages 11–21) to prioritize simplicity and answer: *Which scheduler does best once AI execution overhead is charged?*
- **Decision:** Restrict trained models strictly to 5:
  1. Algorithm-level: Decision Tree (baseline, charged once)
  2. Algorithm-level: Random Forest (stronger, charged once)
  3. Algorithm-level (rule-level): Tabular Q-learning (charged per non-forced decision)
  4. Process-level (PID): GBDT Candidate Ranker (charged per non-forced decision)
  5. Process-level (PID): Neural Candidate Scorer (Deep Learning, charged per non-forced decision)
- **Rationale:** Focuses research directly on model types (classical ML vs deep learning) and decision scopes (algorithm-level vs process-level) without superfluous training pipelines.
- **Alternatives Rejected / Dropped:** PPO fine-tuned scorer (unstable, requires critic/KL tuning with low expected delta over supervised scorer), k-NN model, majority-class model as a model (kept as a reference scalar), exact DP solver, SLOWDOWN objective, HoldPolicy, T-large/T-holdout datasets, and Wilcoxon/Holm tests.

## Decision 006: Category 1 Model Strategy (DT Baseline vs RF Strong)
- **Status:** Accepted
- **Context:** Category 1 models decide the scheduling algorithm once per workload at startup.
- **Decision:** Use Decision Tree as the interpretable baseline and Random Forest as the strong model. Evaluate both on validation regret.
- **Rationale:** Because inference is charged only once per workload, the small latency difference between DT and RF (sub-millisecond) is negligible compared to total workload duration; therefore, accuracy and lower regret dominate.
- **Alternatives Rejected:** MLP or Gradient Boosting for Category 1 (tabular data with 12 structural features is already optimal for tree ensembles).

## Decision 007: Q-Learning Category & Charging Convention
- **Status:** Accepted
- **Context:** Q-learning was trained in Stage 6 to select among 4 meta-rules.
- **Decision:** Retain Q-learning in the algorithm/rule-level group, but charge inference per non-forced decision tick.
- **Rationale:** The agent selects a dispatch rule tick-by-tick rather than scoring individual process IDs. Charging per decision accurately reflects its dynamic dispatch cost.

## Decision 008: Settings S0 and S1 with Waiting Time Objective
- **Status:** Accepted
- **Context:** Need standard evaluation benchmarks for preemption penalties.
- **Decision:** Restrict all subsequent experiments strictly to S0 (`cs_cost = 0`) and S1 (`cs_cost = 1`) under the total waiting time objective.
- **Rationale:** S0 provides an empirical sanity check (SRTF optimality bound). S1 introduces non-zero preemption penalties where AI schedulers can learn switching economy. Adding slowdown objectives would double computation without altering qualitative conclusions.

## Decision 009: Deep Learning Implementation Boundaries
- **Status:** Accepted
- **Context:** Model 5 requires a deep learning neural scorer with permutation-equivariant candidate scoring.
- **Decision:** Permit PyTorch for offline training with DAgger; export learned weights to NumPy arrays (`.npz`) for zero-dependency, pure-NumPy runtime inference.
- **Rationale:** Documented, controlled relaxation of the zero-DL rule for training only; ensures uniform, lightweight NumPy-based inference during benchmarking.

## Decision 010: Rollout Oracle with Two Base Policies
- **Status:** Accepted
- **Context:** PID models require expert supervision labels for candidate ranking.
- **Decision:** Build a rollout oracle based on `ORACLE_BASE_POLICIES = ("SRTF", "SJF")` with immediate penalty and best rollout completion.
- **Rationale:** Validated by S0 matching SRTF and S1 bounded by $\le \min(\text{SRTF}, \text{SJF})$, eliminating the need for an intractable exponential DP solver.

## Decision 011: Strict Test Split Preservation
- **Status:** Accepted
- **Context:** Evaluation integrity across both Part A (Stages 0–9) and Part B (Stages 10–20).
- **Decision:** Keep test split seeds 200,000–200,999 untouched until Stage 18 final comparison. Register hypotheses in `docs/HYPOTHESES.md` prior to executing the test benchmark.
- **Rationale:** Prevents data snooping and model overfitting on the test partition.

## Decision 012: Category 1 Per-Setting Retraining & Backward Compatibility
- **Status:** Accepted
- **Context:** Stage 12 requires per-setting datasets (`scheduling_dataset_S0.csv`, `scheduling_dataset_S1.csv`) and separate models for Decision Tree and Random Forest for S0 and S1.
- **Decision:**
  1. Generate independent datasets for S0 (`cs_cost=0`) and S1 (`cs_cost=1`), validating class distributions (< 85% dominance). In S0: RR 60.3%, SJF 31.2%, FCFS 8.5%. In S1: SJF 52.8%, RR 27.7%, FCFS 19.6%.
  2. Perform 5-fold cross-validated grid search for DT and RF independently per setting.
  3. Mirror `decision_tree_S1.joblib` and metadata to `decision_tree.joblib` to maintain 100% backward compatibility with Part A tests.
  4. Implement `SelectorPolicy` with upfront `on_start()` inference charging and zero per-decision latency.
  5. Register `DT` and `RF` aliases in the central policy registry without modifying the baseline `list_policies()` output, preserving all Part A classical registry tests.


