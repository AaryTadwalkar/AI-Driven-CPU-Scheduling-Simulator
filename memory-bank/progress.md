# Project Progress & Status

## What Works
- **Repository Setup:** Git repository initialized locally.
- **Environment:** Python 3.11 virtual environment (`.venv`) created and verified.
- **Project Structure:** Fully defined directory scaffold according to Section 3.2:
  - `src/schedsim/` (`domain/`, `policies/`, `workloads/`, `ml/`, `rl/`, `evaluation/`, `api/routes/`)
  - `tests/`, `data/`, `models/`, `results/`, `docs/`, `memory-bank/`
- **Configuration & Dependencies:** `pyproject.toml` and `config.py` configured with single-source-of-truth constants, seed ranges, and labeling weights.
- **Tooling:** Pytest and Ruff linting configured in `pyproject.toml`.
- **CLI Skeleton:** `cli.py` subcommands configured (`simulate`, `compare`, `build-dataset`, `train-tree`, `train-rl`, `evaluate`).
- **Memory Bank:** Initialized and kept synchronized adhering to the multi-agent persistence protocol.
- **Domain Core (Stage 1 Complete):**
  - Task 1: `Process` dataclass with type/value validation and `validate_processes` (`src/schedsim/domain/process.py`).
  - Task 2: `TimelineSegment` and `Timeline` with segment merging, partition validation, and counters (`src/schedsim/domain/timeline.py`).
  - Task 3: `ProcessMetrics`, `SimulationMetrics`, `calculate_jain_fairness`, `calculate_process_metrics`, and `calculate_simulation_metrics` (`src/schedsim/domain/metrics.py`).
  - Task 4: `ProcView`, `View`, `StepInfo`, `SimResult`, `ScriptedPolicy`, `Engine`, and `run()` helper (`src/schedsim/domain/engine.py`).
  - Task 5: Domain acceptance test suite (`tests/test_domain.py`) validating idle gaps, simultaneous arrivals, hand-verified CS fixtures, penalty invariants, work conservation, timeline partitions, and engine independence.
  - 100% test coverage across all domain components (92 passing tests).

- **Classical Policies & Workloads (Stage 2 Complete):**
  - Task 1: `Policy` ABC, `FCFSPolicy`, `SJFPolicy`, `SRTFPolicy`, `RoundRobinPolicy`.
  - Task 2: Policy registry and factory `get_policy`, `list_policies`, `register_policy` (`src/schedsim/policies/registry.py`).
  - Task 3: Preset workloads module `get_preset`, `list_presets` (`src/schedsim/workloads/presets.py`) and package exports.
  - Task 4: CLI simulation subcommand with ASCII Gantt chart renderer (`cli.py`, `tests/test_cli_simulate.py`).
  - Task 5: Acceptance test suite (`tests/test_classical_policies.py`) verifying Section 4.6 benchmarks, RR tie-breaking convention, 200 random workloads for SRTF optimality, manual CS counts at `cs_cost = 1`, and named presets.
  - 177 passing tests across the repo (100% statement coverage in `src/schedsim`); `ruff check .` clean.

- **Workload Generator & Data Layer (Stage 3 Complete):**
  - Task 1: Implemented 9 canonical workload scenarios (`src/schedsim/workloads/scenarios.py`), reproducible generator `generate_workload` (`src/schedsim/workloads/generator.py`), and comprehensive unit test suite (`tests/test_workload_generator.py`, 25 tests) with domain invariants, bounds checking, and deterministic sorting.
  - Task 2: Implemented workload CSV persistence (`src/schedsim/workloads/io.py`), immutable `Workload` aggregate dataclass, `save_workloads_to_csv`, `load_workloads_from_csv`, and unit test suite (`tests/test_workload_io.py`, 30 tests) covering lossless round-trip across all 9 scenarios, schema enforcement, auto parent directory creation, and error matrix.
  - Task 3: Implemented seed splits and partition validators in `src/schedsim/workloads/splits.py` and `tests/test_splits.py` (12 tests) enforcing Section 3.4 Train (0–99,999), Val (100,000–100,499), and Test (200,000–200,999) ranges with round-robin class balancing.
  - Task 4: Implemented CLI `build-workloads` and `build-dataset` subcommands in `cli.py` and unit tests in `tests/test_cli_build_workloads.py` (9 tests) with formatted stdout metrics and timing.
  - Task 5: Implemented comprehensive Stage 3 acceptance test suite and performance benchmark in `tests/test_stage3_acceptance.py` (7 tests) verifying seed determinism, divergence, bounds, 9 scenario invariants, lossless CSV round-trip, partition purity, and benchmark (< 60s).
  - Generated full 6,000 workload training dataset at `data/workloads.csv` (45,000 processes, 0.28s, >21,000 workloads/s).
  - 260 passing tests across the repository and 0 ruff errors.

- **Decision Tree Policy Recommender (Stage 4 Complete):**
  - Task 1: Implemented 12 workload-derived domain features in `src/schedsim/ml/features.py` without simulation calls (`extract_features`, `extract_features_vector`), tested in `tests/test_ml_features.py` (7 tests).
  - Task 2: Multi-metric composite objective labeling in `src/schedsim/ml/labeling.py` evaluating FCFS, SJF, RR at `cs_cost=1`, margin calculations, ambiguity detection, tested in `tests/test_ml_labeling.py` (7 tests).
  - Task 3: Completed Section 4c pilot gate study on 600 workloads, selecting principled `cs1_q2_standard` configuration (FCFS 20.5%, SJF 50.8%, RR 28.7%, no class >= 85%) and documenting Decision 004 in `docs/DECISIONS.md`.
  - Task 4: Implemented supervised dataset builder in `src/schedsim/ml/dataset_builder.py` (`data/scheduling_dataset.csv`, 6,500 samples), tested in `tests/test_ml_dataset_builder.py` (7 tests).
  - Task 5: Implemented grid-search training and baseline comparisons in `src/schedsim/ml/train_tree.py` (max_depth=6, min_samples_leaf=5; comparison with Dummy, Random Forest, KNN).
  - Task 6: Implemented validation evaluation, regret calculation, and matplotlib tree plot generation in `src/schedsim/ml/evaluate_tree.py` (`models/decision_tree.joblib`, `models/decision_tree_meta.json`, `results/tree_metrics_val.json`, `results/figures/decision_tree.png`), tested in `tests/test_ml_tree.py` (6 tests).
  - Task 7: Implemented single-point public inference API `recommend(processes) -> {algorithm, probabilities, features}`.
  - Task 8: Integrated CLI commands `build-dataset`, `train-tree`, and `recommend` in `cli.py`, tested in `tests/test_cli_ml.py` (5 tests).
  - Task 9: Stage 4 acceptance test suite in `tests/test_stage4_acceptance.py` (8 tests) verifying feature isolation, composite math, pilot gate logging, dataset balance, model artifacts, baseline outperformance (0.8762 vs 0.2269 macro-F1), single-entrypoint contract, and regret bounds (0.0480).
  - 300 passing tests across the repository and 0 ruff errors.

- **Reinforcement Learning Environment & Formulation (Stage 5 Complete):**
  - Implemented 4 discrete rule-based actions (`SHORTEST_REMAINING`, `LONGEST_REMAINING`, `EARLIEST_ARRIVAL`, `CONTINUE_CURRENT`) in `src/schedsim/rl/actions.py` with canonical Section 4.4 tie-breaking and action masking (`get_valid_actions`, `action_to_pid`), tested in `tests/test_rl_actions.py` (7 tests).
  - Implemented compact permutation-invariant aggregate state space in `src/schedsim/rl/state.py` (7-tuple: ready count, shortest remaining, longest remaining, max waited, last is ready, more arrivals pending, last is shortest; at most 3,200 states) with bucket edges in `config.py`, tested in `tests/test_rl_state.py` (6 tests).
  - Implemented standardized gym-like `Environment` class wrapping `Engine` in `src/schedsim/rl/env.py` (`reset`, `valid_actions`, `step`, `reward = -StepInfo.penalty`, `result`), tested in `tests/test_rl_env.py` (9 tests).
  - Implemented comprehensive Stage 5 acceptance test suite in `tests/test_stage5_acceptance.py` (7 tests):
    - Exact reward identity $\sum \text{reward} == -\sum \text{waiting}$ on 600 stochastic rollouts (300 at `cs_cost=0`, 300 at `cs_cost=1`).
    - Exact reproduction of `SRTFPolicy` with `SHORTEST_REMAINING` at `cs_cost=0`.
    - Exact reproduction of `FCFSPolicy` with `EARLIEST_ARRIVAL` at `cs_cost=0` and `cs_cost=1`.
    - Exact reproduction of non-preemptive FCFS with `CONTINUE_CURRENT` (earliest-arrival fallback).
    - Strict action masking: `CONTINUE_CURRENT` never offered when `last_pid` is completed or idle.
    - State hashability and 100% permutation invariance under PID renaming/reordering.
    - Large-scale stability: 1,000 random-action rollouts completed without errors.
  - 329 passing tests across the repository and 0 ruff errors.

- **Tabular Q-learning Agent and Training (Stage 6 Complete):**
  - Implemented `QLearningAgent` in `src/schedsim/rl/agent.py` featuring sparse dict Q-table, epsilon-greedy action selection over valid action masks, Bellman TD update, canonical lowest-index tie-breaking, neutral `EARLIEST_ARRIVAL` fallback on unseen states with tracking, geometric epsilon decay, joblib checkpointing, and top-500 state JSON summary exports, tested in `tests/test_rl_agent.py` (10 tests).
  - Implemented `RLPolicy` adapter in `src/schedsim/rl/policy.py` inheriting from `Policy` ABC, allowing trained Q-learning agents to plug seamlessly into the unified `Engine.run()` harness with automated workload binding, tested in `tests/test_rl_policy.py` (4 tests).
  - Implemented validation evaluation, episodic training loop, multi-core hyperparameter grid search, and multi-seed orchestrator in `src/schedsim/rl/train.py`:
    - Completed validation grid search across 18 configurations over 3 seeds in `results/rl_hparam_search.csv` (best: $\alpha = 0.2$, $\gamma = 0.95$, decay $= 0.9995$).
    - Executed canonical final training across all 5 seeds (`[0, 1, 2, 3, 4]`) for 20,000 episodes for both `cs_cost=0` and `cs_cost=1`, logging trajectories to `data/rl_training_logs.csv`.
    - Saved best models and JSON summaries: `models/q_table_cs0.pkl`, `models/q_table_cs1.pkl`, `models/q_table.pkl`, `models/q_table_summary_cs0.json`, `models/q_table_summary_cs1.json`, `models/q_table_summary.json`.
  - Implemented diagnostic evaluation and figure visualizers in `src/schedsim/rl/evaluate.py`:
    - Generated validation learning curves with $\pm 1$ std error bands at `results/figures/rl_learning_curve.png`.
    - Generated optimal action shares across top visited states at `results/figures/rl_action_distribution.png`.
  - Integrated `train-rl` CLI subcommand in `cli.py` (`tests/test_cli_rl.py`, 2 tests).
  - Implemented Stage 6 acceptance test suite in `tests/test_stage6_acceptance.py` (7 tests) verifying learning convergence, unseen state rate strictly $< 5\%$ ($0.07\%$), cs0 convergence to SRTF, and figure files.
  - 352 passing tests across the repository and 0 ruff errors.

- **Evaluation Suite & Report Artifacts (Stage 7 Complete):**
  - Implemented multi-policy evaluation harness in `src/schedsim/evaluation/run_experiments.py` evaluating FCFS, SJF, RR(q=2), SRTF, Q-learning, and Decision Tree on held-out test split (1,000 workloads, 12,000 runs) at `cs_cost=0` and `cs_cost=1`, writing `results/evaluation_results.csv`.
  - Implemented statistical bootstrap estimator (1,000 resamples, 95% CI), aggregate summary tables (`results/summary_table.csv`, `results/summary_table.md`), scenario breakdown (`results/scenario_breakdown.csv`), and Decision Tree test evaluation in `src/schedsim/evaluation/stats.py`:
    - Test accuracy: 90.20%, macro-F1: 0.8878.
    - Decision Tree regret: 0.0487 (vs Always-SJF: 0.2655, Always-RR: 0.4110, Always-FCFS: 0.5402).
    - Saved to `results/decision_tree_test_metrics.json`.
  - Implemented publication-quality figure generators in `src/schedsim/evaluation/plots.py`:
    - `results/figures/comparison_bars.png`: 4-panel waiting and response comparison with 95% bootstrap error bars.
    - `results/figures/gantt_fixtures.png`: Gantt chart for Section 4.6 fixture across policies at `cs_cost=0` and `cs_cost=1`.
    - `results/figures/dt_confusion_matrix.png`: Normalized heatmap of test confusion matrix.
    - `results/figures/dt_feature_importance.png`: Ranked horizontal bar chart of Gini feature importances.
    - `results/figures/fairness_starvation.png`: Dual-panel comparison of Jain fairness and starvation rates.
    - `results/figures/scenario_breakdown.png`: Heatmap profiling policy performance across all 9 scenarios.
  - Authored comprehensive report notes mapping all figures and tables to report sections in `docs/REPORT_NOTES.md`.
  - Integrated `evaluate` CLI subcommand in `cli.py` (`python cli.py evaluate`).
  - Implemented Stage 7 acceptance test suite in `tests/test_stage7_acceptance.py` (7 tests).
  - 359 passing tests across the repository and 0 ruff errors.

- **FastAPI Backend Service (Stage 8 Complete):**
  - Implemented Pydantic v2 validation models in `src/schedsim/api/schemas.py` bounding $n \le 50$, burst $\le 100$, arrival $\le 500$, unique PIDs, and uniform error schema `{error, detail}`.
  - Implemented simulation, comparison, recommendation, and synthetic generation services in `src/schedsim/api/service.py`:
    - Full step playback logging (`ready_queue_log`) for UI scrubber.
    - Full Q-learning explainability trace (`decision_trace`) recording time, state, valid actions, Q-values, chosen action, and reward.
    - Verified exact metric parity between `/api/simulate` and `Engine.run()`.
  - Implemented FastAPI application, lifespan startup model caching, CORS middleware, and global exception handlers in `src/schedsim/api/app.py` and `src/schedsim/api/main.py`:
    - All 10 required `/api` endpoints operational and verified in OpenAPI schema.
    - Prescriptive HTTP 503 fallback when model files are missing.
    - `uvicorn schedsim.api.main:app` verified.
  - Implemented comprehensive unit and acceptance test suites (`tests/test_api.py`, `tests/test_stage8_acceptance.py`, 17 tests).
  - 376 passing tests across the repository and 0 ruff errors.

- **React Dashboard - Sub-stage F1: Scaffold & Design System (Stage 9 In Progress):**
  - Initialized `frontend/` with Vite 5, React 18, TypeScript (strict), Tailwind CSS v3, and Lucide icons.
  - Configured `tailwind.config.ts` and `src/index.css` with semantic CSS tokens for light and dark themes (`--background`, `--surface`, `--primary`, `--text`, etc.) plus diagonal hatch patterns for `IDLE` and `CS`.
  - Generated strongly typed backend schemas in `src/api/types.generated.ts` from `results/openapi.json`.
  - Implemented typed API client in `src/api/client.ts` with custom `ApiError` handling.
  - Implemented TanStack Query hooks in `src/api/hooks.ts` covering all 10 endpoints.
  - Implemented `src/lib/cn.ts`, `src/lib/colors.ts` (Okabe-Ito 12-color palette + algorithm colors), and `src/lib/format.ts`.
  - Implemented responsive layout shell in `src/components/layout/`: `AppShell.tsx`, `Sidebar.tsx`, `TopBar.tsx`, `ThemeToggle.tsx`.
  - Setup routing in `src/routes.tsx` with scaffolded placeholder pages: `SimulatorPage.tsx`, `ComparePage.tsx`, `AILabPage.tsx`, `ResultsPage.tsx`.
- **React Dashboard - Sub-stage F2: Workload Editor (Stage 9 In Progress):**
  - Implemented Zod validation schemas (`src/lib/validation.ts`) for processes and workloads (bounds: arrival 0-500, burst 1-100, unique PIDs).
  - Implemented CSV utilities (`src/lib/csv.ts`) for RFC-compliant exporting, fault-tolerant parsing with header detection and error logging, and client-side download triggering.
  - Implemented Zustand store (`src/store/workloadStore.ts`) with localStorage persistence under key `schedsim_workload_v1`, defaulting to Section 4.6 canonical fixture (P1 0,5; P2 1,3; P3 2,2).
  - Implemented `PresetPicker.tsx` consuming `/api/presets` with category and process count previews.
  - Implemented `GeneratorForm.tsx` covering all 9 scenarios, process slider (1 to 50), random seed dice generator, and run reproducibility confirmation toast.
  - Implemented `ProcessTable.tsx` with inline inputs, real-time validation error alerts, PID color badges, CSV import modal (file upload or raw text paste), export button, and reset fixture button.
  - Implemented composite `WorkloadEditor.tsx` unifying Table, Presets, and Generator tabs.
  - Integrated `WorkloadEditor` and statistics summary into `src/pages/SimulatorPage.tsx`.
  - Added unit and integration tests in `src/test/workload.test.ts` and `src/test/WorkloadEditor.test.tsx` (44 passing tests total).
  - Production build clean (`npm run build` succeeds with 0 errors).

- **React Dashboard - Sub-stage F3: Simulator Page & Gantt Playback (Stage 9 In Progress):**
  - Implemented Zustand playback store (`src/store/playbackStore.ts`) managing play/pause, step forward/backward, reset, tick scrubber, speed multipliers (1x, 2x, 4x), and self-contained timer intervals.
  - Implemented `src/components/common/StatCard.tsx` with tabular monospace numbers, metric definitions, and baseline delta formatting.
  - Implemented `src/components/sim/MetricCards.tsx` visualizing all 6 simulation KPIs (Waiting, Turnaround, Response, CPU Utilization, Jain's Fairness, Context Switches).
  - Implemented `src/components/sim/GanttChart.tsx` with proportional timeline segments, hatched IDLE/CS styling, synchronized playhead indicator, time ruler axis, hover tooltips, and PNG export via `html-to-image`.
  - Implemented `src/components/sim/PlaybackControls.tsx` with play/pause, stepping, scrubbing, speed toggles, and global keyboard shortcuts (Space, ArrowLeft, ArrowRight).
  - Implemented `src/components/sim/ProcessMetricsTable.tsx` detailing per-process completion, TAT, waiting, and response with PID color dots and workload mean footer.
  - Implemented `src/components/sim/ReadyQueueView.tsx` rendering active CPU Core and waiting ready queue ordering at any discrete playhead tick.
  - Implemented `src/components/sim/AlgorithmPicker.tsx` covering FCFS, SJF, SRTF, RR (with quantum controls), Q-Learning, context switch cost toggles, simulation trigger, and Decision Tree recommendation badge with auto-selection.
  - Implemented `src/components/ai/DecisionTrace.tsx` logging discrete RL state vectors, valid actions, Q-values, chosen actions, and rewards, with interactive playhead synchronization on row click.
  - Integrated full simulator suite into `src/pages/SimulatorPage.tsx` with initial fixture auto-run, debounced ML recommendations, and friendly 503 model-missing alert.
  - Added unit and integration tests in `src/test/playback.test.ts` (9 tests) and `src/test/simulator.test.tsx` (11 tests). 64 total frontend tests pass.
  - Production build clean (`npm run build` succeeds with 0 errors).

- **React Dashboard - Sub-stage F4: Compare Page (Stage 9 In Progress):**
  - Implemented `src/components/compare/ComparisonTable.tsx` benchmarking all algorithms (FCFS, SJF, SRTF, RR, Q-Learning) on identical workloads, evaluating best-per-metric optimality (emerald badges for lower-is-better and higher-is-better metrics), interactive baseline differential inspection (+/- relative deltas), and CSV export button.
  - Implemented `src/components/compare/ComparisonBars.tsx` grouped interactive bar charts using Recharts with toggle between "By Metric" and "By Algorithm", consistent brand palette colors (`getAlgorithmColor`), and custom engineering tabular hover tooltips.
  - Implemented `src/components/compare/GanttStack.tsx` rendering stacked horizontal Gantt charts aligned to a common maximum makespan time axis, with sticky algorithm lanes, proportional PID-colored execution blocks, hatched `IDLE` and `CS` blocks, shared bottom time ruler, hover tooltips, and PNG export via `html-to-image`.
  - Implemented `src/components/common/EmptyState.tsx`, `ErrorState.tsx`, and `Skeletons.tsx` providing standard empty states, skeleton wireframe loaders (cards, tables, charts, gantt stack), and diagnostic error banners featuring actionable CLI guidance for HTTP 503 missing RL model states.
  - Extended `src/lib/csv.ts` with `exportComparisonToCsv` and `downloadComparisonCsv`.
  - Extended `src/lib/colors.ts` and `src/lib/format.ts` with RL / Q-Learning aliases.
  - Rebuilt `src/pages/ComparePage.tsx` with active workload overview header, configurable benchmark parameters (context switch cost 0/1 toggle, RR quantum input, algorithm selection checkboxes, Run All button), auto-run on mount, and full state management (empty, loading, error, success).
  - Added unit and integration tests in `src/test/compare.test.tsx` (15 tests). All 7 test files (79 tests) pass cleanly.
  - Production build clean (`npm run build` succeeds with 0 errors).

- **React Dashboard - Sub-stage F5: AI Lab & Results Pages (Stage 9 In Progress):**
  - Implemented `src/components/ai/ConfusionMatrix.tsx`: 3x3 heatmap grid visualizing Decision Tree test confusion matrix (FCFS, RR, SJF) with sample counts, row-normalized percentage color shading, accuracy and macro-F1 badges, and an accessible HTML table toggle fallback.
  - Implemented `src/components/ai/FeatureImportance.tsx`: Recharts horizontal bar chart visualizing ranked Gini feature importances across all 12 structural features with clean human-readable labels (`formatFeatureLabel`), percentage values, and interactive tooltips.
  - Implemented `src/components/ai/LearningCurve.tsx`: Recharts `ComposedChart` displaying Tabular Q-Learning training trajectories across 5 seeds over 20,000 episodes, featuring mean waiting curves, shaded +/- 1 std deviation bands, context switch cost toggle (0 vs 1), and horizontal reference lines for SRTF, SJF, and FCFS heuristic baselines.
  - Implemented `src/components/ai/PolicyTable.tsx`: Searchable, paginated policy inspection table displaying top-visited states (decoded 7-tuple aggregate state), visit counts and share percentages, optimal action badges, and Q-values for all 4 meta-rules with optimal action highlighting.
  - Implemented `src/components/ai/LiveRecommender.tsx`: Interactive testbed for the Decision Tree recommender allowing users to load processes from active simulator workload or edit processes inline, executing `/api/recommend` to display predicted policy badges, confidence score, horizontal class probability distribution bars (SJF, RR, FCFS), and all 12 extracted structural features.
  - Implemented `src/components/results/SummaryTable.tsx`: Canonical evaluation benchmark summary table from `/api/results/summary` displaying policies (FCFS, SJF, SRTF, RR, RL, DT) with point estimates and 95% bootstrap confidence intervals [lower, upper], green optimality badges for best-in-class metrics, cost toggle (0 vs 1), and "Download Summary CSV" button.
  - Implemented `src/components/results/ScenarioBreakdown.tsx`: Interactive scenario explorer covering all 9 synthetic regimes (`short_jobs`, `long_jobs`, `mixed`, `bimodal`, `simultaneous`, `staggered`, `convoy`, `equal_bursts`, `high_load`) with scenario characteristics, algorithm performance rankings, comparative Recharts bar chart, cost toggle, and "Download Scenario CSV" button.
  - Implemented `src/components/results/LimitationsPanel.tsx`: Honest technical limitations accordion articulating discrete unit tick granularity, tabular state abstraction boundaries, offline feature extraction assumptions, and single-core virtual CPU semantics with mitigations conforming to Section 8 of the report.
  - Extended `src/lib/csv.ts` with `exportSummaryTableToCsv`, `downloadSummaryTableCsv`, `exportScenarioBreakdownToCsv`, and `downloadScenarioBreakdownCsv`.
  - Rebuilt `src/pages/AILabPage.tsx` with dual tabbed navigation ("Decision Tree Recommender" & "Q-Learning Tabular Agent"), architecture overview cards, test KPI cards, graceful HTTP 503 missing-model detection with exact CLI guidance (`python cli.py train-tree` / `python cli.py train-rl`), and simulator navigation callbacks.
  - Rebuilt `src/pages/ResultsPage.tsx` with Executive Summary cards highlighting 3 major findings (5.4x Regret Reduction, 56% Starvation Reduction, 46% Switching Economy), embedding SummaryTable, ScenarioBreakdown, LimitationsPanel, and CSV download triggers.
  - Added unit and integration tests in `src/test/ai-lab.test.tsx` (9 tests) and `src/test/results.test.tsx` (8 tests). All 9 test files (96 tests) pass cleanly.
  - Production build clean (`npm run build` succeeds in 6.76s with 0 errors).

- **React Dashboard - Sub-stage F6: Polish, Verification, Screenshots & Documentation (Stage 9 Complete):**
  - Executed responsive design and accessibility audit across 1440px desktop, 1024px small desktop, 768px tablet, and 390px mobile viewports across all 4 pages (`/`, `/compare`, `/ai-lab`, `/results`). Verified keyboard shortcuts (`Space` for play/pause, `ArrowLeft`/`ArrowRight` for step forward/back) and theme toggle.
  - Set up Playwright (`@playwright/test` and Chromium) in `frontend/`: authored `playwright.config.ts`, `e2e/smoke.spec.ts` (preset loading, simulation, Gantt, metrics, route traversal), and `e2e/responsive-a11y.spec.ts` (all 20 E2E tests passing).
  - Authored standalone automated screenshot capture script `frontend/scripts/capture_screenshots.ts` (`npm run screenshots`) capturing all 14 mandatory report checklist screenshots at 1440×900 into `screenshots/`, verified as non-empty PNG files (119 KB to 182 KB).
  - Added `"Q_LEARNING"` and `"Q-LEARNING"` alias handling in `src/schedsim/api/service.py` to seamlessly accept all policy casing formats from the frontend workbench.
  - Created comprehensive `frontend/README.md` documenting architecture, setup instructions, backend synchronization, component breakdown, screenshot mapping, and Definition of Done verification.
  - Verified `npm test` runs 96 Vitest unit tests cleanly (2.35s), `npm run test:e2e` passes 20 Playwright tests (16.2s), and `npm run build` succeeds with strict TypeScript typechecking and minification (6.42s).

- **Revised Implementation Plan — Stage 11 Complete:**
  - Implemented AI stall timeline label (`AI`), fractional microsecond-to-tick debt accumulator, decision counts, and start overhead charging.
  - Implemented `Engine.clone()` and `restore()`.
  - Added benchmark `Setting(name, cs_cost)` in `config.py` (S0: cs_cost=0, S1: cs_cost=1).
  - All 11 tests in `tests/test_stage11_engine.py` passing.

- **Revised Implementation Plan — Stage 12 Complete (Category 1: Random Forest & Per-Setting Retraining):**
  - Generated and validated `data/scheduling_dataset_S0.csv` and `data/scheduling_dataset_S1.csv` passing pilot gate (< 85% share per class).
  - Implemented `src/schedsim/ml/train_forest.py` and trained Decision Tree and Random Forest for both S0 and S1 with 5-fold CV.
  - Evaluated accuracy, macro-F1, and regret; verified that both DT and RF beat majority-class dummy baselines, and RF matches/beats DT on validation regret.
  - Saved model artifacts (`decision_tree_S0.joblib`, `random_forest_S0.joblib`, `decision_tree_S1.joblib`, `random_forest_S1.joblib`, and synced `decision_tree.joblib` / `_meta.json`).
  - Updated `recommend(processes, model, setting)` inference API in `src/schedsim/ml/evaluate_tree.py`.
  - Implemented `SelectorPolicy(Policy)` in `src/schedsim/policies/selector.py` with `on_start()` upfront charging, `0.0` per-decision cost, `reset()`, and `DTPolicy`/`RFPolicy` subclasses.
  - Registered `DT` and `RF` aliases in `src/schedsim/policies/registry.py`.
  - Updated `RLPolicy` in `src/schedsim/rl/policy.py` with `from_setting()`, `from_cs_cost()`, `on_start()` auto-binding, and explicit `decision_cost_us()`.
  - Added 9 unit/acceptance tests in `tests/test_stage12_category1.py`.
  - Full suite passing: 396 passed, 0 ruff errors.

- **Revised Implementation Plan — Stage 13 Complete (Rollout Oracle - labels only):**
  - Added PID selector & oracle constants to `config.py` (`ORACLE_BASE_POLICIES`, `PID_BC_TRAIN_WORKLOADS`, `DAGGER_ITERS`, `DAGGER_WORKLOADS_PER_ITER`, `PID_MAX_STATES`, `GBDT_PARAMS`, `NN_HIDDEN`, `NN_SEEDS`).
  - Created package `schedsim.pidsel` (`__init__.py`, `oracle.py`).
  - Implemented `OracleDecision` and `OracleStepRecord` frozen dataclasses with strict invariant checking.
  - Implemented `evaluate_state(engine, base_policies)` using `Engine.clone()`, 1-step lookahead, and base rollout simulation with 4-tier tie-breaking: `(cost, remaining, arrival, pid)`.
  - Implemented `run_oracle(processes, cs_cost, base_policies)` with forced decision skipping and non-forced trajectory recording.
  - Implemented `RolloutOraclePolicy(Policy)` supporting shadow engine synchronization inside `run()`.
  - Implemented `label_workloads_parallel` with multiprocessing `ProcessPoolExecutor` and deterministic sequence ordering.
  - Added 10 tests in `tests/test_stage13_oracle.py`:
    - S0 acceptance: Oracle waiting matches SRTF exactly across 300 random workloads (0 violations).
    - S1 acceptance: Oracle waiting bounded by `min(SRTF, SJF)` across 300 random workloads (0 violations, 16 strict improvements, 284 ties).
    - Determinism verified on repeated runs.
    - Tie-breaking rules verified across cost, remaining burst, arrival, and PID.
    - Regret properties verified (best candidate regret 0, suboptimal > 0, algebraic identity).
    - Full suite passing: 406 passed, 0 ruff errors.

- **Revised Implementation Plan — Stage 14 Complete (Per-Candidate Features and Training Data):**
  - Implemented 13 per-candidate features in `src/schedsim/pidsel/features.py`: Process group (`rem`, `burst`, `rem_burst_ratio`, `waited`, `is_last`), Relative group (`rank_rem`, `rank_waited`, `rem_min_ratio`, `rem_mean_ratio`), and Context group (`n_ready`, `sum_rem`, `cs_cost`, `pending_arrivals`).
  - Implemented pure NumPy symmetric fractional rank `compute_normalized_rank`.
  - Implemented `extract_candidate_features` and `pad_and_stack_features` producing permutation-equivariant float32 tensors with no PIDs or lookahead.
  - Implemented `build_pid_dataset` in `src/schedsim/pidsel/dataset.py` running the rollout oracle, filtering forced decisions (`len(ready) <= 1`), and capturing features, masks, costs, regrets, and best candidate index.
  - Generated and saved `data/pid_bc_S0.npz` and `data/pid_bc_S1.npz` (100,000 states each, 2,467 and 2,420 workloads).
  - Implemented `generate_dataset_report` and produced `results/pid_dataset_report.md` documenting:
    - Min ready candidates = 2 across both datasets (zero single-candidate states).
    - S0 oracle behavior: 100.00% SRTF alignment, 0.0% continuation-only.
    - S1 oracle behavior: 99.01% SRTF alignment, 0.99% continuation-only (987 states where preemption switch stall was avoided).
  - Added `pending_arrivals` to `View` and `Engine._build_view` with 100% backward compatibility.
  - Integrated `build-pid-data` CLI subcommand into `cli.py`.
  - Added 12 comprehensive unit and integration tests in `tests/test_stage14_pidsel_data.py`.
  - Full suite passing: 418 passed, 0 ruff errors, 0 lines > 100 chars.

- **Revised Implementation Plan — Stage 15 Complete (Model 4: Gradient-Boosted Candidate Ranker):**
  - Implemented `src/schedsim/pidsel/gbdt_ranker.py`:
    - `export_tree_ensemble`: Flat array extraction (`l`, `r`, `f`, `th`, `v`), `lr`, and `init_val` from scikit-learn `GradientBoostingClassifier`.
    - `score_candidates`: Vectorized branchless candidate scoring using pure NumPy.
    - `prepare_training_data`: Slices NPZ dataset into candidate rows with regret-weighted sample weights and binary optimal labels (cost <= best_cost * 1.01).
    - `train_gbdt`: Trains `GradientBoostingClassifier` with `GBDT_PARAMS` on S0/S1 datasets, exports `models/pid_gbdt_<S>.joblib` and `models/pid_gbdt_<S>_meta.json`.
    - `evaluate_gbdt_on_validation`: Evaluates 500 validation workloads, producing `results/pid_gbdt_val.json`:
      - S0: Top-1 agreement 99.39%, Mean J_policy 14.727, Mean J_oracle 14.656, Gap vs Oracle/SRTF = +0.486% (<= 0.02 threshold met).
      - S1: Top-1 agreement 99.81%, Mean J_policy 18.008, Mean J_oracle 17.983, Mean J_srtf 18.148, Gap vs Oracle = +0.139%, Gap vs SRTF = -0.771%.
    - `GBDTRankerPolicy(Policy)`: Fast inference with 4-tier tie-breaking (highest score -> smaller remaining -> earlier arrival -> pid string).
  - Centralized policy registry integration in `src/schedsim/policies/registry.py`: `PID_GBDT` and `GBDT` aliases registered.
  - Package exports in `src/schedsim/pidsel/__init__.py`.
  - Added CLI subcommand `train-pid-gbdt` to `cli.py` supporting `--setting`, `--data-dir`, `--models-dir`, `--val-eval`.
  - Created 11 comprehensive unit and acceptance tests in `tests/test_stage15_gbdt.py`.
  - Full suite passing: 429 passed, 0 ruff errors, 0 lines > 100 chars.

- **Revised Implementation Plan — Stage 16 Complete (Model 5: Neural Candidate Scorer):**
  - Implemented `src/schedsim/pidsel/scorer_net.py`:
    - `Scorer(nn.Module)`: Permutation-equivariant candidate-scoring neural network architecture (~3.6k parameters). Applies candidate-wise MLP phi (13 -> 32 -> 32), mean-pools across valid candidates for context aggregation, and maps concatenated features [z, ctx] through scoring head (64 -> 32 -> 1). Masked candidates filled with -inf.
    - `export_numpy_weights`: Extracts all layer weights and biases into pure-NumPy float32 arrays.
    - `numpy_score`: Vectorized pure-NumPy forward pass matching PyTorch within 1.7e-8.
    - `NeuralCandidatePolicy(Policy)`: Fast inference adapter executing without PyTorch dependencies. Canonical 4-tier tie breaking: score desc -> remaining asc -> arrival asc -> pid asc.
  - Implemented `src/schedsim/pidsel/train_bc.py`:
    - `soft_cross_entropy_loss`: Soft target distribution p*(a) ∝ exp(-regret_a / T) over valid candidates with numerically stable masked log-softmax.
    - `train_neural_scorer`: Iterative DAgger training loop (DAGGER_ITERS = 2) across NN_SEEDS = [0, 1, 2]. Iteration 0 trains on oracle trajectories; subsequent iterations roll out current model on fresh train workloads mixing oracle actions with probability beta = 0.5^i, query evaluate_state, append transitions, and retrain warm-start.
    - Multi-seed validation evaluation on 200 fixed validation workloads:
      - S0: Top-1 agreement 100.0%, Mean J_policy 14.4624, Gap vs Oracle/SRTF = 0.000% (<= 0.02 threshold met).
      - S1: Top-1 agreement 99.90%, Mean J_policy 17.8104 +/- 0.0002, Gap vs Oracle = +0.098%, Gap vs SRTF = -0.824%.
    - Persisted `models/pid_scorer_S0_np.npz`, `pid_scorer_S0_meta.json`, `models/pid_scorer_S1_np.npz`, `pid_scorer_S1_meta.json`, and `results/pid_m2_val.csv`.
  - Registered `PID_NN` and `PID_NEURAL` aliases in `src/schedsim/policies/registry.py`.
  - Exported Stage 16 components in `src/schedsim/pidsel/__init__.py`.
  - Added CLI subcommand `train-pid-nn` to `cli.py` supporting all Stage 16 flags.
  - Created 16 comprehensive unit and acceptance tests in `tests/test_stage16_neural.py`.
  - Full suite passing: 445 passed, 0 ruff errors, 0 lines > 100 chars.

- **Revised Implementation Plan — Stage 17 Complete (Latency profile and cost model):**
  - Implemented `src/schedsim/overhead.py`:
    - Enforced single-threaded execution controls (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(1)`).
    - `benchmark_policy_latencies`: Measures median of medians and p95 across 3 repeats with GC disabled over synthetic `View` states for FCFS, SJF, RR, SRTF, Q-learning, GBDT, and Neural across $n_{\text{ready}} \in \{2, 3, 5, 8, 10\}$.
    - `benchmark_startup_latencies`: Measures upfront feature extraction + tree prediction latency across workload sizes (3, 5, 8, 10, 15, 20) for Category 1 Decision Tree and Random Forest.
    - `save_latency_profile` & `load_latency_profile`: JSON persistence with in-memory caching (`_CACHED_PROFILE`).
    - `get_interpolated_cost_us` & `get_startup_cost_us`: 1D linear interpolation with boundary clamping and extrapolation warnings.
    - `plot_latency_profile`: Generates publication-quality figure (`results/figures/latency_vs_ready.png`).
    - `validate_live_vs_table`: Simulates workload with transparent `_LiveTimingProxy`, comparing live vs table totals.
  - Policy Life-Cycle Hook Connections:
    - Connected `decision_cost_us(view)` across `FCFSPolicy`, `SJFPolicy`, `SRTFPolicy`, `RoundRobinPolicy`, `RLPolicy`, `GBDTRankerPolicy`, and `NeuralCandidatePolicy`.
    - Exempted forced decisions (`len(view.ready) <= 1`) and `view is None` (returns 0.0 µs).
    - Connected `on_start(processes)` in `SelectorPolicy` (`DTPolicy`, `RFPolicy`) returning upfront latency.
  - CLI Integration:
    - Added `bench-latency` subcommand to `cli.py` with tabular reporting and plot generation.
    - Generated `models/latency_profile.json` and `results/figures/latency_vs_ready.png`.
  - Tests:
    - Created `tests/test_stage17_overhead.py` with 13 comprehensive unit, property, and CLI tests.
    - Full suite passing: 458 passed (100%), `ruff check .` clean (0 errors), 0 lines > 100 chars.

- **Revised Implementation Plan — Stage 18 Complete (Final Comparison — The Deliverable):**
  - Implemented `src/schedsim/evaluation/final_comparison.py`:
    - Evaluated run matrix across S0 ($c_s = 0$) and S1 ($c_s = 1$) on 1,000 test split workloads (seeds 200,000–200,999).
    - Evaluated 11 policies: Classical (`FCFS`, `SJF`, `RR` q=2, `SRTF`), AI Category 1 (`DT`, `RF`), AI Category 2 (`Q-learning`), AI Category 3 (`PID_GBDT`, `PID_NN`), and References (`Best-Classical-Hindsight`, `Oracle`).
    - Verified S0 optimality sanity check (no policy beats SRTF before overhead at $c_s = 0$).
    - Computed 15-column comparison matrices with 95% bootstrap CIs (1,000 resamples).
    - Exported deliverables: `results/final_comparison_S0.csv`, `results/final_comparison_S0.md`, `results/final_comparison_S1.csv`, `results/final_comparison_S1.md`.
    - Computed 1D linear break-even tick frequencies across sweep $[1, 10, 100, 1000, 10000]\,\mu\text{s}$.
    - Generated 5 publication-quality figures at 300 DPI in `results/figures/`:
      1. `effective_cost_vs_tick.png`
      2. `gap_to_oracle_bars.png`
      3. `overhead_breakdown.png`
      4. `final_scenario_heatmap.png`
      5. `gantt_with_ai_stalls.png`
    - Evaluated pre-registered hypotheses H1–H4 in `docs/HYPOTHESES_RESULTS.md` with formal verdicts:
      * H1: CONFIRMED (SRTF optimal at S0; learned models degrade after overhead).
      * H2: CONFIRMED (Oracle beats SRTF at S1: 17.76 vs 17.90; Neural 17.77 and GBDT 17.78 capture gap).
      * H3: CONFIRMED (Break-even tick thresholds calculated: Q-learning 1 µs, Neural 100 µs, GBDT 10,000 µs).
      * H4: REFUTED (Random Forest 100-tree ensemble latency 31,193 µs imposes 38.06% makespan overhead, refuting < 0.1%).
  - CLI Integration:
    - Added `compare-all` subcommand to `cli.py` (`python cli.py compare-all`).
  - Tests:
    - Created `tests/test_stage18_comparison.py` with 11 acceptance tests (100% pass in 22.8s).
    - Full test suite passing: 469 passed (100% in 363.2s), 0 failures.
    - Zero ruff errors (`ruff check .` clean), 0 lines > 100 chars.

## What's Left to Build
- **Stage 19:** Minimal API and dashboard extension (`/simulate`, `/compare`, AI stall visualization).
- **Stage 20:** Submission pack (`README.md`, `docs/REPORT_NOTES.md`, `make smoke`).

## Current Status
- Stages 0 to 9 complete.
- Stages 11, 12, 13, 14, 15, 16, 17, and 18 complete and verified (469 pytest tests passing, 0 ruff errors, 0 lines > 100 chars).
- Next up: Stage 19 (Minimal API and dashboard extension).
