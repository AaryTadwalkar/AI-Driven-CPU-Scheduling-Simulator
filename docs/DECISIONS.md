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
