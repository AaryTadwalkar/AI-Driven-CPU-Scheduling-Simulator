# Project Brief: AI-Driven CPU Scheduling Simulator

## Overview
- **Course:** Operating System (AI3002), AY 2026–27, Topic 1.
- **Goal:** Deliver a reproducible, fully-tested Python CPU scheduling simulator featuring classical algorithms (FCFS, SJF, SRTF, Round Robin), a supervised Decision Tree policy recommender, a tabular Reinforcement Learning (Q-learning) scheduler, an academic evaluation suite, and a modern React dashboard.

## Core Scope & Deliverables
1. **Unified Simulation Engine:** A discrete unit-step simulator (`Engine.step()`) operating with exact context switch modeling, idle jumps, and reproducible process lifecycle tracking.
2. **Classical Policies:**
   - First-Come First-Served (FCFS)
   - Shortest Job First (SJF, non-preemptive)
   - Shortest Remaining Time First (SRTF, reference baseline)
   - Round Robin (RR, configurable quantum, default 2)
3. **Machine Learning Component:**
   - Supervised Decision Tree recommender predicting the optimal policy for a given workload profile based on structural workload features (burst CV, load ratio, arrival dispersion).
   - Balanced multi-objective labeling criteria (waiting time, response time, turnaround time, context switch overhead).
4. **Reinforcement Learning Component:**
   - Tabular Q-learning agent with a compact, permutation-invariant aggregate queue state space (1k–2k states).
   - Rule-based action dispatch (pick shortest remaining, longest remaining, earliest arrival, continue current).
   - Exact equivalence between episode return and negative sum of waiting times.
5. **Evaluation Suite:**
   - Rigid seed partitions (Train: 0–99,999; Validation: 100,000–100,499; Test: 200,000–200,999).
   - Regret analysis, learning curves with confidence intervals, Jain's fairness index, Gantt chart generation.
6. **Web Dashboard:**
   - FastAPI REST API backend.
   - Vite + React + TypeScript + Tailwind CSS + shadcn/ui + Recharts frontend.
