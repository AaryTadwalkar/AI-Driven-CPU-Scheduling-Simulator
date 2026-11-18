# Project Progress Log

Course: Operating System (AI3002), AY 2026–27, Topic 1
Project: AI-Driven CPU Scheduling Simulator

## Stage Tracking

- [x] **Stage 0 — Project scaffold**: Directory structure, `pyproject.toml`, `config.py`, CLI skeleton, virtual environment, git initialized.
- [x] **Stage 1 — Domain core**: Process, unit-step engine, metrics, timeline.
- [ ] **Stage 2 — Classical policies**: FCFS, SJF, SRTF, RR(q), policy registry, preset workloads.
- [ ] **Stage 3 — Workload generator and data layer**: Scenarios, generator with seeded RNG, dataset splits.
- [ ] **Stage 4 — Decision Tree**: Supervised features, composite labeling, pilot study, training and evaluation.
- [ ] **Stage 5 — RL Environment & Formulation**: State space, action space, unit-step gym-like env, reward verification.
- [ ] **Stage 6 — Tabular Q-learning Agent**: Q-table, epsilon decay, validation curves, convergence tracking.
- [ ] **Stage 7 — Evaluation Suite & Report Artifacts**: Held-out test split, statistical metrics, regret analysis, charts.
- [ ] **Stage 8 — FastAPI Backend**: API endpoints for simulation, ML inference, and RL execution.
- [ ] **Stage 9 — React Dashboard**: Interactive Gantt visualization, metric comparisons, workload editor.

---

## Log Entries

### Stage 0 — Scaffold Setup (2026-11-18)
- Initialized local git repository.
- Created virtual environment `.venv` using Python 3.11.9.
- Defined project structure per Section 3.2: `src/schedsim/`, `tests/`, `data/`, `models/`, `results/`, `docs/`, `memory-bank/`.
- Configured `pyproject.toml` with dependencies, pytest, and ruff settings.
- Established `config.py` with single-source-of-truth constants and seed partitions.
- Created `cli.py` with subcommand argument parser (`simulate`, `compare`, `build-dataset`, `train-tree`, `train-rl`, `evaluate`).
- Memory bank initialized to preserve multi-session architectural context.

### Stage 1 — Domain Core (2026-11-18)
- Implemented `Process` dataclass with defensive validation in `src/schedsim/domain/process.py`.
- Implemented `TimelineSegment` and `Timeline` with interval merging and partition verification in `src/schedsim/domain/timeline.py`.
- Implemented `ProcessMetrics`, `SimulationMetrics`, and calculation utilities in `src/schedsim/domain/metrics.py`.
- Implemented unit-step `Engine`, `ProcView`, `View`, `StepInfo`, `SimResult`, `ScriptedPolicy`, and `run()` in `src/schedsim/domain/engine.py`.
- Implemented comprehensive domain acceptance suite in `tests/test_domain.py`.
- Verification: 92/92 tests passing in pytest (100% statement coverage across `schedsim`), 0 ruff errors, all lines <= 100 characters.

