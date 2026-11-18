# CPU Scheduling Simulator & AI Dispatcher — Frontend Dashboard

An interactive, production-grade observability dashboard and research testbed for operating system CPU scheduling algorithms and AI-driven meta-rule dispatchers. Built with React 18, TypeScript (strict mode), Vite, Tailwind CSS, Zustand, TanStack Query, and Recharts.

---

## 1. Architecture Overview

The frontend dashboard serves as a graphical workbench for discrete-event CPU scheduling simulations, multi-policy empirical benchmarking, supervised Decision Tree workload classification, and Tabular Q-Learning explainability.

```
┌────────────────────────────────────────────────────────────────────────┐
│                          React 18 + Vite Shell                         │
├────────────────────────────────────────────────────────────────────────┤
│  TopBar (API Health Dot, Active View Title, Light/Dark Theme Switcher) │
├──────────────┬─────────────────────────────────────────────────────────┤
│   Sidebar    │  Page Views (React Router v6):                          │
│   (Desktop,  │  ├── SimulatorPage (WorkloadEditor, Gantt, Playback)    │
│    Tablet,   │  ├── ComparePage (Multi-Policy Benchmark, Bars, Stacks) │
│    Mobile    │  ├── AILabPage (Decision Tree, RL Trajectory & Policy)  │
│    Docking)  │  └── ResultsPage (Canonical 1,000-Test Benchmark & CIs) │
├──────────────┴─────────────────────────────────────────────────────────┤
│  State Management & Observability:                                     │
│  ├── Zustand workloadStore (Processes, Presets, Synthetic Generator)   │
│  ├── Zustand playbackStore (Discrete Tick Scrubbing, Transport)        │
│  └── TanStack React Query v5 (FastAPI Endpoint Mutations & Caching)    │
└────────────────────────────────────────────────────────────────────────┘
```

### Core Technologies
- **Vite 5**: Sub-second Hot Module Replacement (HMR) and optimized rollup production bundling.
- **React 18**: Concurrent-mode functional components with hooks and clean separation of concerns.
- **TypeScript (Strict Mode)**: 100% type safety without `any` bypasses; auto-generated API schema interfaces from FastAPI OpenAPI specifications (`src/api/types.generated.ts`).
- **Tailwind CSS**: Calm observability design system utilizing semantic CSS variables, custom typography (`font-mono` for all metrics), and seamless dark mode tokens.
- **Zustand**: Dual stores separating workload persistence (`localStorage` synchronization) from high-frequency playback transport state.
- **TanStack Query v5**: Server state management with optimistic caching, automated background refetching, and graceful HTTP 503 missing-model recovery banners.
- **Recharts**: Scalable SVG charting (`ResponsiveContainer`, `BarChart`, `ComposedChart`) for latency comparisons and episodic training convergence bands.
- **Playwright**: End-to-end regression test suite and automated 1440×900 report screenshot capture.

---

## 2. Setup Instructions

### Prerequisites
- Node.js >= 18.0.0
- npm >= 9.0.0
- Python virtual environment with backend dependencies installed (Python >= 3.10)

### Installation
From the repository root or `frontend/` directory:

```bash
# Navigate to the frontend directory
cd frontend

# Install all runtime and development dependencies
npm install
```

### Environment Configuration
Create a `.env` file in `frontend/` (defaults to `http://localhost:8000` if omitted):

```env
VITE_API_URL=http://localhost:8000
```

### Development Server
```bash
npm run dev
# Starts local development server at http://localhost:5173
```

### Production Build & Typecheck
```bash
npm run build
# Compiles TypeScript strictly (tsc) and generates minified dist/ bundles via Vite
```

### Testing Suites
```bash
# Run Vitest unit & integration tests (96 tests across 9 test suites)
npm test

# Run Playwright End-to-End smoke and responsive tests (20 tests)
npm run test:e2e

# Run automated 14-screenshot report capture script (saves to ../screenshots/)
npm run screenshots
```

### Regenerate TypeScript API Types
When backend FastAPI Pydantic schemas change:
```bash
npm run gen:types
```

---

## 3. Running Alongside the Backend

The frontend communicates with the FastAPI service asynchronously via JSON REST endpoints.

1. **Start the FastAPI Backend Service** (from repository root):
   ```bash
   # Using virtual environment Python
   python -m uvicorn schedsim.api.main:app --port 8000
   ```
2. **Start the Frontend Dashboard** (in another terminal):
   ```bash
   cd frontend
   npm run dev
   ```
3. Open `http://localhost:5173` in your browser.

### TopBar Health Status Dot Semantics
- **Green (Connected)**: Backend API is reachable and all trained AI checkpoints (`decision_tree.joblib`, `q_table.pkl`) are loaded.
- **Amber (Models Missing)**: Backend API is reachable, but one or more optional AI checkpoints are not yet trained. Friendly in-app callouts display exact CLI training commands (`python cli.py train-tree` or `python cli.py train-rl`).
- **Red (Backend Offline)**: Backend is unreachable at `http://localhost:8000`. A top notification banner warns the user and suggests remediation commands.

---

## 4. Page & Component Architecture Breakdown

### 1. Simulator Page (`/`)
Interactive workbench for single-policy execution and timeline scrubbing.
- `WorkloadEditor`: Segmented interface toggling between:
  - **Manual Table**: Inline editing of PID, arrival tick, burst duration, priority with instant Zod validation.
  - **Preset Picker**: Dropdown catalog of canonical workloads (`section_4_6`, `convoy`, `staggered`, `simultaneous`, `bimodal`, `equal_bursts`).
  - **Synthetic Generator**: Parameterized generator supporting 9 statistical scenarios and seed reproducibility.
- `AlgorithmPicker`: Configures target policy (FCFS, SJF, SRTF, RR, Q-Learning), quantum (1-50), and context switch penalty (0 vs 1 tick). Features real-time Decision Tree recommendation badge.
- `MetricCards`: Responsive 6-card KPI grid displaying Avg Waiting, Avg Turnaround, Avg Response, CPU Utilization, Jain's Fairness Index, and Context Switches with baseline delta indicators.
- `GanttChart`: Scalable CPU timeline with process allocations, hatched IDLE and CS intervals, synchronized vertical playhead, hover tooltips, and PNG export.
- `PlaybackControls`: Scrubbing slider, transport buttons (Play/Pause, Step Forward/Back, Reset, Jump to End), speed multipliers (1x, 2x, 4x), and keyboard listeners (`Space`, `ArrowLeft`, `ArrowRight`).
- `ReadyQueueView`: Real-time inspection of CPU running process and ready queue contents at playhead tick `t`.
- `ProcessMetricsTable`: Per-process completion, turnaround, waiting, and response times.
- `DecisionTrace`: Explainable AI table for Tabular Q-Learning displaying 7-tuple states, valid actions, Q-values, and chosen action; clicking any row jumps the playhead to that tick.

### 2. Compare Page (`/compare`)
Multi-algorithm side-by-side benchmarking on identical workloads.
- **Benchmark Controls**: Algorithm selection checkboxes (FCFS, SJF, SRTF, RR, Q-Learning), RR quantum input, and context-switch penalty toggle (`0 ticks` pure vs `1 tick` penalty).
- `ComparisonTable`: Canonical comparison table with bold green badges indicating the winning algorithm per metric and baseline delta comparisons.
- `ComparisonBars`: Recharts grouped bar charts with dual viewing modes ("Group by Metric" and "Group by Algorithm") and exact tabular tooltips.
- `GanttStack`: Vertically stacked, horizontally synchronized Gantt timelines displaying timeline alignment and context-switch overhead across all candidate policies.

### 3. AI Lab Page (`/ai-lab`)
Dual AI research and explainability laboratory.
- **Tab 1: Decision Tree Workload Recommender**:
  - Model architecture and hyperparameter card (`max_depth=5`, `criterion=entropy`).
  - KPI test statistics: Accuracy (90.2%) and Macro-F1 (0.888).
  - `ConfusionMatrix`: 3x3 heatmap grid visualizing test set classifications across FCFS, RR, and SJF with row percentages and accessible table fallback.
  - `FeatureImportance`: Horizontal bar chart ranking all 12 structural workload features (burst CV, load factor, arrival CV, etc.) by Gini importance.
  - `LiveRecommender`: Interactive testbed allowing users to test custom or active simulator processes, executing `/api/recommend` to inspect predicted class badges, confidence scores, probability distribution bars, and extracted features.
- **Tab 2: Tabular Q-Learning Dispatcher**:
  - MDP architecture card: 7-tuple aggregate state vector, 4 meta-rules (`SRTF`, `LRTF`, `FIFO`, `HOLD`).
  - `LearningCurve`: Recharts ComposedChart visualizing training trajectories across 5 random seeds over 20,000 episodes with mean waiting curve, shaded ±1 std deviation band, context switch cost toggle (0 vs 1), and horizontal heuristic baseline reference lines (SRTF, SJF, FCFS).
  - `PolicyTable`: Searchable, paginated policy inspection table displaying top-visited aggregate states, visit counts, optimal action badges, and estimated Q-values.

### 4. Results Page (`/results`)
Canonical evaluation benchmark on 1,000 held-out synthetic workloads (12,000 simulated runs).
- **Executive Summary Cards**: Highlights 3 major empirical findings:
  - *5.4x Regret Reduction* via Decision Tree meta-policy recommendation.
  - *56% Starvation Reduction* achieved by Tabular Q-Learning over greedy SJF.
  - *46% Switching Economy* under non-zero context switch costs.
- `SummaryTable`: Canonical benchmark table displaying point estimates with 95% bootstrap confidence intervals (`[lower, upper]`) for all metrics and policies under `cs_cost=0` and `cs_cost=1`, with CSV download capability.
- `ScenarioBreakdown`: Interactive scenario explorer analyzing performance across all 9 synthetic regimes (`short_jobs`, `long_jobs`, `mixed`, `bimodal`, `simultaneous`, `staggered`, `convoy`, `equal_bursts`, `high_load`).
- `LimitationsPanel`: Transparent engineering limitations disclosure detailing discrete unit tick granularity, tabular state abstraction boundaries, offline feature extraction assumptions, and single-core virtual CPU semantics.

---

## 5. Report Screenshot Gallery Mapping

All 14 checklist screenshots specified in Section 9 of `FRONTEND_GUIDE.md` are automated via Playwright and stored in `screenshots/` at 1440×900:

| # | Filename | Description | View |
|---|----------|-------------|------|
| 1 | `01_input_workload_editor.png` | Workload process table with `convoy` preset loaded and validated. | Simulator (`/`) |
| 2 | `02_generator_seeded.png` | Synthetic generator form showing scenario distribution, seed 42, and confirmation alert. | Simulator (`/`) |
| 3 | `03_gantt_fcfs.png` | FCFS execution timeline with metric cards prominently displayed. | Simulator (`/`) |
| 4 | `04_gantt_sjf.png` | SJF execution timeline with metric cards prominently displayed. | Simulator (`/`) |
| 5 | `05_gantt_rr.png` | Round Robin execution timeline (quantum=2) with metric cards visible. | Simulator (`/`) |
| 6 | `06_gantt_qlearning_trace.png` | Q-Learning RL Gantt timeline plus step-by-step Decision Trace explainability table. | Simulator (`/`) |
| 7 | `07_compare_all.png` | Multi-algorithm comparison table, grouped bar charts, and stacked Gantt timelines (`cs_cost=0`). | Compare (`/compare`) |
| 8 | `08_compare_cs1.png` | Multi-algorithm comparison with 1-tick context switch penalty demonstrating overhead. | Compare (`/compare`) |
| 9 | `09_dt_recommendation.png` | Decision Tree live recommendation card with predicted policy and probability bars. | AI Lab (`/ai-lab`) |
| 10 | `10_dt_confusion_importance.png` | 3x3 test confusion matrix heatmap and Gini feature importances bar chart. | AI Lab (`/ai-lab`) |
| 11 | `11_rl_learning_curve.png` | Q-Learning 20,000 episode training trajectory with ±1 std deviation band and heuristic baselines. | AI Lab (`/ai-lab`) |
| 12 | `12_rl_policy_table.png` | Searchable Q-table policy inspection table with state vectors and optimal actions. | AI Lab (`/ai-lab`) |
| 13 | `13_results_summary.png` | Canonical benchmark evaluation summary table with 95% bootstrap confidence intervals. | Results (`/results`) |
| 14 | `14_dark_mode_simulator.png` | Polished dark mode capture of the Simulator workbench. | Simulator (`/`) |

---

## 6. Definition of Done (Frontend) Verification Checklist

- [x] **Four Pages Operational**: Simulator, Compare, AI Lab, and Results run against the live FastAPI backend with zero client-side mock fixtures in production code.
- [x] **Theme & Responsiveness**: Light and dark themes polished; responsive layout verified via automated tests at 1440px desktop, 1024px small desktop, 768px tablet, and 390px mobile viewports.
- [x] **Comprehensive UI States**: Every data view implements loading skeletons, descriptive error states, empty placeholders, and graceful HTTP 503 missing-model callouts.
- [x] **Gantt Chart & Playback**: Timeline features PID labels, time ruler, hover tooltips, hatched `IDLE` and `CS` styling, scrubbing slider, and full keyboard shortcut transport (`Space`, `ArrowLeft`, `ArrowRight`).
- [x] **Strict Scheduling Truth**: No client-side scheduling logic; all displayed metrics, timelines, and decision traces strictly mirror FastAPI simulation responses.
- [x] **Clean Compilation & Testing**: `npm run build` succeeds cleanly with TypeScript strict compilation; `npm test` passes 96 Vitest unit and integration tests; `npm run test:e2e` passes 20 Playwright end-to-end tests.
- [x] **Automated Screenshot Checklist**: All 14 report checklist screenshots captured at 1440×900 and verified as valid non-empty PNG files in `screenshots/`.
- [x] **Documentation**: Complete architectural reference, setup instructions, backend integration guide, and component taxonomy documented in `frontend/README.md`.
