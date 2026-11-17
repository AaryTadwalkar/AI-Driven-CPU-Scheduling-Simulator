# Product Context: AI-Driven CPU Scheduling Simulator

## Why This Project Exists
Operating systems must schedule CPU time among competing processes under conflicting objectives: minimizing waiting time, minimizing response time, preventing starvation, and minimizing context switch overhead. Classical heuristics (FCFS, SJF, RR) excel in specific workload regimes but degrade in others (e.g., FCFS under convoy effects, SJF under starvation risk, RR under excessive context switching).

This project bridges classical OS scheduling theory with modern machine learning:
1. Predicting which heuristic best fits a batch workload profile before execution via a Decision Tree.
2. Dynamically making tick-by-tick dispatch decisions under unknown future arrivals via Reinforcement Learning.

## Problems Solved
- **Convoy Effect & Inefficiency:** Demonstrates when and how SJF/SRTF outperform FCFS.
- **Fairness vs. Throughput Trade-offs:** Quantifies starvation and response latency using Jain's fairness index and maximum waiting times.
- **Context Switch Realism:** Accurately models non-zero switch penalties, penalizing hyperactive preemption.
- **Educational & Visual Understanding:** Provides interactive timeline/Gantt visualizations and comparative metric tables for academic demonstration.

## User Experience Goals
- **CLI User:** Fast, intuitive CLI commands (`simulate`, `compare`, `build-dataset`, `train-tree`, `train-rl`, `evaluate`) with formatted terminal tables and ASCII Gantt charts.
- **Web Dashboard User:** Clean, responsive web UI with workload editors, animated Gantt execution charts, policy comparison radar/bar charts, and real-time model inference.
