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

