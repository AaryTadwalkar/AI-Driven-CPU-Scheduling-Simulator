# Project Progress Log

Course: Operating System (AI3002), AY 2026–27, Topic 1
Project: AI-Driven CPU Scheduling Simulator

## Stage Tracking

- [x] **Stage 0 — Project scaffold**: Directory structure, `pyproject.toml`, `config.py`, CLI skeleton, virtual environment, git initialized.
- [x] **Stage 1 — Domain core**: Process, unit-step engine, metrics, timeline.
- [x] **Stage 2 — Classical policies**: FCFS, SJF, SRTF, RR(q), policy registry, preset workloads.
- [x] **Stage 3 — Workload generator and data layer**: Scenarios, generator with seeded RNG, dataset splits.
- [x] **Stage 4 — Decision Tree**: Supervised features, composite labeling, pilot study, training and evaluation.
- [x] **Stage 5 — RL Environment & Formulation**: State space, action space, unit-step gym-like env, reward verification.
- [x] **Stage 6 — Tabular Q-learning Agent**: Q-table, epsilon decay, validation curves, convergence tracking.
- [x] **Stage 7 — Evaluation Suite & Report Artifacts**: Held-out test split, statistical metrics, regret analysis, charts.
- [x] **Stage 8 — FastAPI Backend**: API endpoints for simulation, ML inference, and RL execution.
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

### Stage 2 — Classical Policies (2026-11-18)
- Implemented `Policy` base contract, `FCFSPolicy`, `SJFPolicy`, `SRTFPolicy`, and `RoundRobinPolicy(quantum=2)` in `src/schedsim/policies/`.
- Implemented policy registry with dynamic lookup, aliases, and factory in `src/schedsim/policies/registry.py`.
- Implemented canonical presets in `src/schedsim/workloads/presets.py` (convoy, Section 4.6, simultaneous, equal bursts, long idle gap, staggered, high contention, mixed priorities).
- Implemented CLI `simulate` subcommand with tabular formatting and ASCII Gantt chart in `cli.py`.
- Implemented Stage 2 acceptance test suite in `tests/test_classical_policies.py` covering Section 4.6 hand-verified fixtures, RR tie-breaking convention, 200 seeded random workloads verifying SRTF optimality and conservation, manual CS counts at `cs_cost = 1`, and all named presets.
- Verification: 177/177 passing tests in pytest (100% statement coverage across `src/schedsim`, 96% in `cli.py`), 0 ruff errors, all lines <= 100 characters.

### Stage 3 — Workload Generator and Data Layer (2026-11-18)
- Implemented 9 canonical workload scenarios in `src/schedsim/workloads/scenarios.py`.
- Implemented reproducible generator in `src/schedsim/workloads/generator.py` using seeded `numpy.random.default_rng`.
- Implemented CSV persistence layer in `src/schedsim/workloads/io.py` (`Workload`, `save_workloads_to_csv`, `load_workloads_from_csv`).
- Implemented seed partition splits and validators in `src/schedsim/workloads/splits.py` (Train 0–99,999, Val 100,000–100,499, Test 200,000–200,999).
- Implemented CLI `build-workloads` subcommand in `cli.py`.
- Synthesized 6,000 workload training dataset at `data/workloads.csv` (45,000 processes, 0.28s, >21,000 workloads/s).
- Implemented Stage 3 acceptance tests in `tests/test_stage3_acceptance.py`.
- Verification: 260/260 passing tests in pytest, 0 ruff errors across entire codebase.

### Stage 4 — Decision Tree Policy Recommender (2026-11-18)
- Implemented 12 workload-derived domain features in `src/schedsim/ml/features.py` computed strictly without running simulation (`extract_features`, `extract_features_vector`).
- Implemented multi-metric composite labeling in `src/schedsim/ml/labeling.py` evaluating FCFS, SJF, RR at `cs_cost=1`, normalized by Pareto baselines with margin calculation and ambiguity flagging.
- Completed mandatory Section 4c pilot gate study across 600 workloads under 4 configurations, confirming non-trivial multi-class distribution (FCFS 20.5%, SJF 50.8%, RR 28.7%, no class >= 85%) and documenting Decision 004 in `docs/DECISIONS.md`.
- Implemented supervised dataset builder in `src/schedsim/ml/dataset_builder.py` and synthesized `data/scheduling_dataset.csv` with 6,500 samples (6,000 train + 500 val).
- Implemented hyperparameter grid search and baseline comparisons in `src/schedsim/ml/train_tree.py`, fitting Decision Tree (best: max_depth=6, min_samples_leaf=5), Dummy Majority Class, Random Forest, and Scaled KNN.
- Implemented validation evaluation, regret calculation, and matplotlib tree plot generation in `src/schedsim/ml/evaluate_tree.py`, generating `models/decision_tree.joblib`, `models/decision_tree_meta.json`, `results/tree_metrics_val.json`, and `results/figures/decision_tree.png`.
- Implemented single-point public inference API `recommend(processes) -> {algorithm, probabilities, features}`.
- Integrated CLI commands `build-dataset`, `train-tree`, and `recommend` into `cli.py`.
- Decision Tree validation performance: 89.0% accuracy, 0.8762 macro-F1 (crushing Majority Class baseline of 0.2269 macro-F1), 91.3% unambiguous accuracy, 0.0480 validation mean regret.
- Verification: 300/300 passing tests in pytest, 0 ruff errors across entire codebase, all lines <= 100 characters.

### Stage 6 — Tabular Q-learning Agent and Training (2026-11-18)
- Implemented `QLearningAgent` in `src/schedsim/rl/agent.py` featuring sparse dictionary Q-table, epsilon-greedy action selection over valid action masks, Bellman TD update ($Q \leftarrow Q + \alpha [r + \gamma \max_{a'} Q(s',a') - Q]$), canonical lowest-index tie-breaking, neutral `EARLIEST_ARRIVAL` fallback on unseen states with tracking, geometric epsilon decay, joblib checkpointing, and top-500 state JSON summary exports.
- Implemented `RLPolicy` adapter in `src/schedsim/rl/policy.py` inheriting from `Policy` ABC, allowing trained Q-learning agents to plug seamlessly into the unified `Engine.run()` harness with automated workload binding.
- Implemented validation evaluation, episodic training loop, multi-core hyperparameter grid search, and multi-seed orchestrator in `src/schedsim/rl/train.py`:
  - Completed validation grid search across 18 configurations ($\alpha \in \{0.05, 0.1, 0.2\}$, $\gamma \in \{0.95, 0.99, 1.0\}$, decay $\in \{0.999, 0.9995\}$) over 3 seeds, logging to `results/rl_hparam_search.csv`. Best parameters found: $\alpha = 0.2$, $\gamma = 0.95$, decay $= 0.9995$ (mean validation waiting: 21.10 ticks).
  - Executed canonical final training across all 5 seeds (`[0, 1, 2, 3, 4]`) for 20,000 episodes for both `cs_cost=0` and `cs_cost=1`, logging trajectories to `data/rl_training_logs.csv`.
  - Saved best models and JSON summaries: `models/q_table_cs0.pkl`, `models/q_table_cs1.pkl`, `models/q_table.pkl`, `models/q_table_summary_cs0.json`, `models/q_table_summary_cs1.json`, `models/q_table_summary.json`.
- Implemented diagnostic evaluation and figure visualizers in `src/schedsim/rl/evaluate.py`:
  - Generated validation learning curves with $\pm 1$ std error bands at `results/figures/rl_learning_curve.png`.
  - Generated optimal action shares across top visited states at `results/figures/rl_action_distribution.png`.
- Integrated `train-rl` CLI subcommand in `cli.py` with support for single-seed training, multi-seed training (`--multi-seed`), and hyperparameter search (`--hparam-search`).
- Implemented unit and acceptance test suites (`test_rl_agent.py`, `test_rl_policy.py`, `test_cli_rl.py`, `test_stage6_acceptance.py`):
  - Verified learning curve convergence across all seeds.
  - Verified unseen-state rate on 500 validation workloads is strictly $< 5\%$ (empirically $0.07\%$).
  - Verified $cs\_cost=0$ agent approaches preemptive SRTF baseline within theoretical bounds.
  - Verified RLPolicy compatibility with `run()`.
- Verification: 352/352 passing tests across full repository, 0 ruff errors across entire codebase, all lines $\le 100$ characters. Zero git commits or pushes executed.

### Stage 7 — Evaluation Suite & Report Artifacts (2026-11-18)
- Implemented `src/schedsim/evaluation/run_experiments.py`:
  - Multi-policy evaluation harness evaluating FCFS, SJF, RR(q=2), SRTF, tabular Q-learning (RL), and Decision Tree recommendation across held-out splits at `cs_cost=0` and `cs_cost=1`.
  - Multiprocessing chunking supporting parallel execution of 12,000 simulations in ~30 seconds.
  - Per-workload CSV export to `results/evaluation_results.csv` with 18 performance, fairness, and starvation metrics.
- Implemented `src/schedsim/evaluation/stats.py`:
  - Non-parametric percentile bootstrap confidence interval estimator (1,000 resamples, 95% CI).
  - Aggregate summary tables computed across all policies and cost models, exported to `results/summary_table.csv` and publication-ready Markdown table `results/summary_table.md`.
  - Pairwise win-rate tournament matrices and scenario breakdown exports (`results/scenario_breakdown.csv`).
  - Held-out test split evaluation for Decision Tree: test accuracy 90.20%, macro-F1 0.8878, and test regret 0.0487 (vs Always-SJF 0.2655, Always-RR 0.4110, Always-FCFS 0.5402), saved to `results/decision_tree_test_metrics.json`.
- Implemented `src/schedsim/evaluation/plots.py`:
  - `results/figures/comparison_bars.png`: 4-panel comparison bar charts with 95% bootstrap error bars for waiting time and response time.
  - `results/figures/gantt_fixtures.png`: Stacked Gantt charts on reference fixture across policies at `cs_cost=0` and `cs_cost=1`.
  - `results/figures/dt_confusion_matrix.png`: Normalized annotated heatmap of test confusion matrix.
  - `results/figures/dt_feature_importance.png`: Ranked horizontal bar chart of Gini feature importances.
  - `results/figures/fairness_starvation.png`: Dual-panel comparison of Jain fairness and starvation rates.
  - `results/figures/scenario_breakdown.png`: Heatmap profiling policy performance across all 9 scenarios.
- Integrated `evaluate` CLI subcommand in `cli.py` (`python cli.py evaluate`).
- Authored comprehensive report notes mapping all figures and tables to report sections in `docs/REPORT_NOTES.md`.
- Implemented Stage 7 acceptance test suite in `tests/test_stage7_acceptance.py` (7 tests).
- Verification: 359/359 passing tests across full repository (`pytest -q` clean), `ruff check .` clean (0 errors), all lines $\le 100$ characters. Zero git commits or pushes executed.

### Stage 8 — FastAPI Backend Service (2026-11-18)
- Implemented Pydantic v2 validation models in `src/schedsim/api/schemas.py` enforcing strict bounds: $n \le 50$, burst $\le 100$, arrival $\le 500$, unique PIDs, and consistent error schema `{error, detail}`.
- Implemented domain-level service logic in `src/schedsim/api/service.py`:
  - `run_simulation_service`: Runs unit-step simulation with per-tick ready queue playback log (`ready_queue_log`) and step-by-step Q-learning explainability trace (`decision_trace` recording time, state, valid actions, Q-values, chosen action, chosen PID, and reward).
  - Exact metric and timeline parity verified between `/api/simulate` and `Engine.run()`.
  - `run_comparison_service`: Multi-policy concurrent evaluation generating comparison summary rows.
  - `run_recommend_service`: Real-time Decision Tree inference returning winning policy, class probabilities, and 12 features.
  - `generate_workload_service`: Reproducible synthetic workload synthesis.
- Implemented FastAPI application and ASGI entrypoint in `src/schedsim/api/app.py` and `src/schedsim/api/main.py`:
  - Lifespan context manager pre-loading Q-learning agent instances at startup for zero-latency in-memory inference.
  - CORS middleware configured for `http://localhost:5173`.
  - Global exception handlers for `ModelMissingException` (HTTP 503 with prescriptive CLI commands), `ValueError` (HTTP 400), and `RequestValidationError` (HTTP 422).
  - All 10 required `/api` endpoints fully functional and registered in OpenAPI schema at `/docs` and `/openapi.json`:
    - `GET /api/health`: Health status, version, and model presence flags.
    - `GET /api/presets`: Preset workloads with descriptions and schedules.
    - `POST /api/workloads/generate`: Synthetic workload synthesis.
    - `POST /api/simulate`: Execution with ready queue log and RL decision trace.
    - `POST /api/compare`: Multi-algorithm benchmarking.
    - `POST /api/recommend`: Supervised policy recommendation.
    - `GET /api/ml/model-info`: Decision Tree architecture, feature importances, and test metrics.
    - `GET /api/rl/training-log`: Downsampled episodic training trajectories for dashboard learning curves.
    - `GET /api/rl/policy-summary`: Q-table summary with top states and action shares.
    - `GET /api/results/summary`: Summary tables, scenario breakdowns, and report notes.
  - Verified `uvicorn schedsim.api.main:app` ASGI target compatibility.
- Implemented comprehensive unit and acceptance test suites (`tests/test_api.py`, `tests/test_stage8_acceptance.py`, 17 tests).
- Verification: 376/376 passing tests across 32 test files in pytest, 0 ruff errors across entire codebase, all lines $\le 100$ characters. Zero git commits or pushes executed.

### Stage 9 — React Dashboard (In Progress: Sub-stages F1, F2, F3, F4, F5 Complete)
- Sub-stage F1: Scaffolded Vite 5 + React 18 + TypeScript strict dashboard with Tailwind CSS v3 semantic tokens, light/dark themes, AppShell, TopBar health dot, auto-generated OpenAPI types, and client/hooks layers.
- Sub-stage F2: Built WorkloadEditor, PresetPicker, GeneratorForm, ProcessTable, Zod schemas, Zustand store with persistence, and CSV import/export.
- Sub-stage F3: Built SimulatorPage with discrete unit-step Gantt playback, PlaybackControls, MetricCards, ReadyQueueView, ProcessMetricsTable, AlgorithmPicker, and DecisionTrace.
- Sub-stage F4: Built ComparePage with multi-policy benchmarking:
  - `ComparisonTable`: lists all algorithms with best-per-metric highlighting (lower-is-better for latency/switches, higher-is-better for utilization/fairness), relative baseline deltas (+/- differences), and CSV export.
  - `ComparisonBars`: Recharts grouped bar charts with "By Metric" and "By Algorithm" toggling, brand palette colors, and custom tabular hover tooltips.
  - `GanttStack`: Stack of horizontal timelines scaled to uniform max makespan with algorithm badges, proportional PID colors, `hatch-idle` and `hatch-cs` fills, shared bottom time ruler, hover inspection, and PNG export via `html-to-image`.
  - Common building blocks: `EmptyState`, `ErrorState` with HTTP 503 missing-model recovery, and `Skeletons` wireframe loaders.
- Sub-stage F5 (2026-11-18): Built AI Lab & Results Pages:
  - `ConfusionMatrix`: 3x3 heatmap grid visualizing Decision Tree test confusion matrix (FCFS, RR, SJF) with sample counts, row-normalized percentage color shading, accuracy and macro-F1 badges, and an accessible HTML table toggle fallback.
  - `FeatureImportance`: Recharts horizontal bar chart visualizing ranked Gini feature importances across all 12 structural features with clean human-readable labels (`formatFeatureLabel`), percentage values, and interactive tooltips.
  - `LearningCurve`: Recharts `ComposedChart` displaying Tabular Q-Learning training trajectories across 5 seeds over 20,000 episodes, featuring mean waiting curves, shaded +/- 1 std deviation bands, context switch cost toggle (0 vs 1), and horizontal reference lines for SRTF, SJF, and FCFS heuristic baselines.
  - `PolicyTable`: Searchable, paginated policy inspection table displaying top-visited states (decoded 7-tuple aggregate state), visit counts and share percentages, optimal action badges, and Q-values for all 4 meta-rules with optimal action highlighting.
  - `LiveRecommender`: Interactive testbed for the Decision Tree recommender allowing users to load processes from active simulator workload or edit processes inline, executing `/api/recommend` to display predicted policy badges, confidence score, horizontal class probability distribution bars (SJF, RR, FCFS), and all 12 extracted structural features.
  - `SummaryTable`: Canonical evaluation benchmark summary table from `/api/results/summary` displaying policies (FCFS, SJF, SRTF, RR, RL, DT) with point estimates and 95% bootstrap confidence intervals [lower, upper], green optimality badges for best-in-class metrics, cost toggle (0 vs 1), and "Download Summary CSV" button.
  - `ScenarioBreakdown`: Interactive scenario explorer covering all 9 synthetic regimes (`short_jobs`, `long_jobs`, `mixed`, `bimodal`, `simultaneous`, `staggered`, `convoy`, `equal_bursts`, `high_load`) with scenario characteristics, algorithm performance rankings, comparative Recharts bar chart, cost toggle, and "Download Scenario CSV" button.
  - `LimitationsPanel`: Honest technical limitations accordion articulating discrete unit tick granularity, tabular state abstraction boundaries, offline feature extraction assumptions, and single-core virtual CPU semantics with mitigations conforming to Section 8 of the report.
  - Extended `src/lib/csv.ts` with `exportSummaryTableToCsv`, `downloadSummaryTableCsv`, `exportScenarioBreakdownToCsv`, and `downloadScenarioBreakdownCsv`.
  - Rebuilt `src/pages/AILabPage.tsx` with dual tabbed navigation ("Decision Tree Recommender" & "Q-Learning Tabular Agent"), architecture overview cards, test KPI cards, graceful HTTP 503 missing-model detection with exact CLI guidance (`python cli.py train-tree` / `python cli.py train-rl`), and simulator navigation callbacks.
  - Rebuilt `src/pages/ResultsPage.tsx` with Executive Summary cards highlighting 3 major findings (5.4x Regret Reduction, 56% Starvation Reduction, 46% Switching Economy), embedding SummaryTable, ScenarioBreakdown, LimitationsPanel, and CSV download triggers.
  - Added comprehensive test suites in `src/test/ai-lab.test.tsx` (9 tests) and `src/test/results.test.tsx` (8 tests). All 9 test files (96 tests) pass cleanly.
  - Production build clean (`npm run build` succeeds in 6.76s with 0 errors).






