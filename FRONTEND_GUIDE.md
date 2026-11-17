# Frontend Guide — CPU Scheduling AI Dashboard

**Read this only when you reach Stage 9 of `IMPLEMENTATION_PLAN.md`.** The backend API (Stage 8) must already work. This dashboard is a thin, polished view over that API. It contains **no scheduling logic**; every number it shows comes from the backend.

Working rules are the same as the main plan: one sub-stage at a time, log deviations with rationale in `docs/DECISIONS.md`, prefer small targeted edits over rewrites, never fabricate data in the UI.

---

## 1. Purpose and scope

The dashboard is the interactive face of the simulator. A user can build or generate a workload, run any algorithm, watch the schedule as a Gantt chart, compare algorithms on identical input, see what the Decision Tree recommends, and inspect what the Q-learning agent learned.

**In scope:** four pages (Simulator, Compare, AI Lab, Results), light and dark themes, responsive layout, loading/error/empty states, step-through playback, exportable chart images.
**Out of scope:** authentication, saving user accounts, editing models in the browser, real-time WebSockets, mobile-first native feel (it must be usable on a tablet and readable on a phone, but desktop is the primary target).

---

## 2. Tech stack (fixed)

| Concern | Choice | Reason |
|---|---|---|
| Build | **Vite + React 18 + TypeScript (strict)** | Fast, minimal config |
| Styling | **Tailwind CSS v3** + CSS variables for theme tokens | Quick, consistent, easy dark mode |
| Components | **shadcn/ui** (Radix primitives) + **lucide-react** icons | Accessible, professional defaults, code lives in your repo |
| Server state | **TanStack Query** | Caching, loading and error states |
| UI state | **Zustand** (workload editor, playback) | Small and simple |
| Routing | **React Router** | Four routes |
| Charts | **Recharts** (bars, lines, scatter) | Easy responsive charts |
| Gantt | **Custom component** (div/SVG), not a library | Needs CS/IDLE styling and playback |
| Forms | **react-hook-form + zod** | Validation matching API limits |
| Animation | **framer-motion**, sparingly (page transitions, playback cursor) | Polish without noise |
| Testing | **Vitest + React Testing Library**; **Playwright** for one smoke test and screenshots | Enough, not excessive |
| API types | **openapi-typescript** generated from FastAPI's `/openapi.json` | Types cannot drift from the backend |

Do not add other UI libraries (no MUI, no Ant Design, no Bootstrap).

---

## 3. Folder structure

```
frontend/
├── index.html
├── package.json  vite.config.ts  tsconfig.json  tailwind.config.ts
├── .env.example                      # VITE_API_URL=http://localhost:8000
├── src/
│   ├── main.tsx  App.tsx  routes.tsx
│   ├── api/        client.ts  hooks.ts  types.generated.ts
│   ├── store/      workloadStore.ts  playbackStore.ts  settingsStore.ts
│   ├── lib/        colors.ts  format.ts  gantt.ts  csv.ts  cn.ts
│   ├── components/
│   │   ├── layout/    AppShell.tsx  Sidebar.tsx  TopBar.tsx  ThemeToggle.tsx
│   │   ├── workload/  WorkloadEditor.tsx  PresetPicker.tsx  GeneratorForm.tsx  ProcessTable.tsx
│   │   ├── sim/       AlgorithmPicker.tsx  GanttChart.tsx  PlaybackControls.tsx  ReadyQueueView.tsx
│   │   │              MetricCards.tsx  ProcessMetricsTable.tsx
│   │   ├── compare/   ComparisonBars.tsx  ComparisonTable.tsx  GanttStack.tsx
│   │   ├── ai/        RecommendationCard.tsx  ProbabilityBars.tsx  ConfusionMatrix.tsx
│   │   │              FeatureImportance.tsx  LearningCurve.tsx  PolicyTable.tsx  DecisionTrace.tsx
│   │   └── common/    StatCard.tsx  EmptyState.tsx  ErrorState.tsx  Skeletons.tsx  ExportButton.tsx
│   ├── pages/      SimulatorPage.tsx  ComparePage.tsx  AILabPage.tsx  ResultsPage.tsx
│   └── test/
└── e2e/            smoke.spec.ts  screenshots.spec.ts
```

---

## 4. Design system

**Feel:** a calm, data-dense, engineering tool. Think observability dashboard, not marketing site. Plenty of whitespace, restrained color, strong typography, numbers that line up.

### 4.1 Tokens

Define as CSS variables on `:root` and `.dark`, mapped into Tailwind.

| Token | Light | Dark |
|---|---|---|
| `--background` | `#F8FAFC` | `#0B1020` |
| `--surface` (cards) | `#FFFFFF` | `#121A2F` |
| `--surface-muted` | `#F1F5F9` | `#182240` |
| `--border` | `#E2E8F0` | `#25314F` |
| `--text` | `#0F172A` | `#E6EAF5` |
| `--text-muted` | `#64748B` | `#93A0BF` |
| `--primary` | `#4F46E5` | `#818CF8` |
| `--success` / `--warning` / `--danger` | `#16A34A` / `#D97706` / `#DC2626` | lighter equivalents |

- **Typography:** Inter for UI, JetBrains Mono for numbers, pids and code (`font-variant-numeric: tabular-nums` on all metrics). Scale: 12 / 14 / 16 / 20 / 28.
- **Radius** 10px cards, 8px controls. **Shadow** very subtle (one level). **Spacing** 4-px grid.
- **Process colors:** a fixed, colorblind-safe categorical palette of 12 colors in `lib/colors.ts` (e.g. Okabe-Ito extended). The same pid keeps the same color on every page. Special segments: `IDLE` = diagonal-hatch light gray, `CS` (context switch) = diagonal-hatch amber.
- **Algorithm colors:** FCFS blue, SJF green, RR orange, SRTF purple, Q-Learning pink/magenta; used in comparison bars and tags.
- Support `prefers-color-scheme` on first load, then persist user choice in `localStorage`.

### 4.2 Responsive behavior

| Breakpoint | Layout |
|---|---|
| ≥1280px | Fixed sidebar (240px) + content, two-column page grids |
| 768–1279px | Collapsible icon sidebar, single column with side-by-side where it fits |
| <768px | Bottom or drawer navigation, cards stack, tables scroll horizontally inside their container, Gantt scrolls horizontally with sticky pid labels |

### 4.3 Accessibility

Keyboard operable controls, visible focus rings, ARIA labels on icon buttons, `role="img"` with a text summary for charts, never rely on color alone (pid labels printed inside Gantt blocks), contrast at least WCAG AA, respect `prefers-reduced-motion`.

---

## 5. Pages and wireframes

### 5.1 App shell

Sidebar (logo, nav: Simulator, Compare, AI Lab, Results), top bar (page title, API status dot from `/health`, theme toggle). The API status dot is green if reachable, amber if models missing, red if unreachable. A red state shows a helpful banner: "Backend not reachable at {VITE_API_URL}".

### 5.2 Simulator page (`/`)

```
┌ Workload ───────────────────────────┐ ┌ Run settings ───────────────────┐
│ [Preset ▾] [Generate…] [Add process]│ │ Algorithm: FCFS|SJF|RR|SRTF|AI  │
│ PID  Arrival  Burst  Priority   ✕   │ │ Quantum [2]  Context switch [0] │
│ P1     0       5       2             │ │ [ Run simulation ]              │
│ …                                    │ └─────────────────────────────────┘
└──────────────────────────────────────┘
┌ Metric cards: Avg wait | Avg turnaround | Avg response | CPU util | Ctx switches | Fairness ┐
┌ Gantt chart ────────────────────────────────────────────────────────────────────────────┐
│ time axis  0 ─ 1 ─ 2 ─ …            [⏮ ◀ ▶ ⏭]  speed 1× 2× 4×     ○ playhead           │
│ ▇P1▇▇▇▇▇▇ ▇P2▇▇▇ ▇P3▇▇                                                                   │
└──────────────────────────────────────────────────────────────────────────────────────────┘
┌ Ready queue at time t ─────────────┐ ┌ Per-process table (arrival, burst, start, finish, wait, TAT, resp) ┐
```

Behaviors:
- Workload editor validates with zod (burst ≥ 1, arrival ≥ 0, unique pids, ≤ 50 processes, API limits). Inline errors. Table supports add/remove/edit rows, CSV import and export.
- **Generator form:** scenario dropdown, number of processes slider, seed input with a "random seed" button. Shows the seed used so any run is reproducible.
- **Playback:** the Gantt reveals progressively up to the playhead. Step buttons move one tick. The ready-queue view and metric preview update with the playhead (derived from the per-tick log returned by `/simulate`). A "Show full result" toggle jumps to the end.
- **Q-learning selected:** a **Decision Trace** panel appears below the Gantt. Each row is a decision (time, state summary, chosen action name, Q-values for valid actions, reward). Clicking a row moves the playhead to that time.
- A **Decision Tree recommendation** badge ("Recommended: SJF, 74%") sits near the algorithm picker, from `/recommend`, with a "Use this" button.

### 5.3 Compare page (`/compare`)

Uses the same workload store. "Run all" calls `/compare`. Shows a comparison table (best value per metric highlighted), grouped bar charts for avg waiting / turnaround / response, and a **stacked Gantt** (one row per algorithm, shared time axis). A toggle runs at context-switch cost 0 and 1 side by side. Always highlight which algorithm wins on each metric; never present a single "best" without naming the metric.

### 5.4 AI Lab page (`/ai-lab`), two tabs

**Decision Tree tab:** model card (training data size, class distribution chart, chosen hyperparameters), accuracy / macro-F1 / regret stat cards, confusion matrix heatmap, feature-importance horizontal bars, and the interactive "Try it" panel: edit a workload, press Recommend, see probabilities as bars and the extracted features.
**Q-Learning tab:** hyperparameter summary, **learning curve** (validation avg waiting vs episode, mean line plus std band, with horizontal reference lines for FCFS / SJF / RR / SRTF), Q-table size and unseen-state rate, and a **Policy table** (top states by visits: bucketed state description, best action, Q-values, visit count) with filters.

### 5.5 Results page (`/results`)

Renders `/results/summary`: the final test-set table with mean ± CI, per-scenario breakdown selector, fairness metrics, and a "Limitations" panel (static text from `docs/REPORT_NOTES.md` surfaced via the API or copied as markdown). Include "Download CSV" and "Download chart PNG" buttons.

### 5.6 States every data view must have

- **Loading:** skeleton shapes matching the final layout (no spinners alone).
- **Error:** message plus Retry button; show the API's `error` and `detail`.
- **Empty:** for example, "No workload yet. Pick a preset or generate one."
- **Model missing (503):** friendly card explaining which command to run.

---

## 6. Key components (implementation notes)

### GanttChart

- Props: `segments: {label, start, end}[]`, `playhead?: number`, `pidColor(pid)`, `height`, `showLabels`.
- Render as a CSS grid or SVG with width = `(end − start) · unitWidth`; `unitWidth` auto-fits the container and has a minimum so long schedules scroll horizontally inside their wrapper with a **sticky label column** (algorithm name or lane).
- Time axis with ticks at a sensible step (1, 2, 5, 10 units based on width).
- Tooltip on hover: pid, start–end, duration, process's arrival and remaining at that moment if available.
- Contiguous segments with the same label are already merged by the backend; do not re-compute scheduling client-side.
- `IDLE` and `CS` use hatched fills and a legend.
- Playback: segments past the playhead render dimmed or hidden; a vertical playhead line follows `playhead`.
- Export to PNG (via `html-to-image` or SVG serialization) from the `ExportButton`.

### PlaybackControls and store

`playbackStore`: `{tick, maxTick, playing, speed}`. A `requestAnimationFrame` or interval loop advances `tick` while `playing`. Pause at `maxTick`. Keyboard: Space = play/pause, ←/→ = step.

### MetricCards

`StatCard` with label, value (mono, tabular), optional delta versus a chosen baseline (green when better, red when worse, correctly handling "lower is better" metrics), and a tiny info tooltip with the metric's definition (turnaround, waiting, response, utilization, Jain fairness).

### ConfusionMatrix

Simple CSS grid heatmap with counts and row-normalized color intensity, class labels on both axes, and an accessible text table fallback.

### LearningCurve

Recharts `ComposedChart`: area for ±std, line for mean, `ReferenceLine`s for baselines with labels. X axis = training episode.

---

## 7. API usage rules

- One `api/client.ts` wrapper around `fetch` using `VITE_API_URL`; throws a typed `ApiError {status, error, detail}`.
- One TanStack Query hook per endpoint in `api/hooks.ts`; mutations for `simulate`, `compare`, `recommend`, `generate`.
- Cache static endpoints (`/presets`, `/ml/model-info`, `/rl/*`, `/results/summary`) with `staleTime` of several minutes.
- Types come from `types.generated.ts` (`npm run gen:types` runs `openapi-typescript $VITE_API_URL/openapi.json`). Do not hand-write response types.
- Debounce `/recommend` (500 ms) when the user edits the workload.
- Never recompute metrics on the client. The only client-side derivation allowed is slicing the per-tick log by `playhead`.

---

## 8. Sub-stages

### F1 — Scaffold and design system
Vite + TS + Tailwind + shadcn/ui, theme tokens, light/dark toggle, AppShell with sidebar, API client with `/health` status dot, generated types, routes with placeholder pages.
**Done when:** `npm run dev` shows the shell in both themes, status dot reflects backend state, `npm run build` and `npm run lint` pass.

### F2 — Workload editor
Process table (edit/add/remove, CSV import/export), preset picker, generator form with seed, zod validation, Zustand store, persisted in `localStorage`.
**Done when:** invalid input shows inline errors, presets load, generated workloads are reproducible by seed, unit tests cover validation.

### F3 — Simulator page and Gantt
AlgorithmPicker, Run button, MetricCards, GanttChart, ProcessMetricsTable, ReadyQueueView, PlaybackControls.
**Done when:** the preset from the plan's fixture (P1/P2/P3) renders the same segments and metrics as the backend tests for all four algorithms; playback works with mouse and keyboard; loading/error/empty states exist.

### F4 — Compare page
Run-all, comparison table with highlights, bar charts, stacked Gantt, cost 0/1 toggle.
**Done when:** the table values equal `/compare` output exactly (test with mocked API), best-per-metric highlighting is correct for lower-is-better metrics.

### F5 — AI Lab and Results
Recommendation card, probability bars, confusion matrix, feature importance, learning curve, policy table, Decision Trace, results table with CI, downloads. Handle the 503 "model missing" path.
**Done when:** each chart renders from the real artifacts, empty and 503 states verified, Decision Trace click moves the playhead.

### F6 — Polish, tests, screenshots
Responsive pass at 1440, 1024, 768 and 390 px widths; accessibility pass (keyboard, focus, labels, contrast); Vitest tests for stores, validation, format helpers, Gantt layout math; one Playwright smoke test (load → pick preset → run → see Gantt and metrics); a Playwright script that captures the screenshot checklist below at 1440×900 in light and dark mode.
**Done when:** the Definition of Done below is satisfied.

---

## 9. Screenshot checklist (for the assignment report)

Save to `screenshots/` with these names, 1440×900, light mode unless stated:

1. `01_input_workload_editor.png`: workload table with a preset loaded.
2. `02_generator_seeded.png`: generator form showing scenario and seed.
3. `03_gantt_fcfs.png`, `04_gantt_sjf.png`, `05_gantt_rr.png`: each with the metric cards visible.
4. `06_gantt_qlearning_trace.png`: Gantt plus Decision Trace.
5. `07_compare_all.png`: comparison table, bars and stacked Gantt.
6. `08_compare_cs1.png`: comparison with context-switch cost 1.
7. `09_dt_recommendation.png`: recommendation card with probabilities.
8. `10_dt_confusion_importance.png`: confusion matrix and feature importances.
9. `11_rl_learning_curve.png`: learning curve with baseline lines.
10. `12_rl_policy_table.png`: policy table.
11. `13_results_summary.png`: final test results with CIs.
12. `14_dark_mode_simulator.png`: one dark mode capture.

---

## 10. Definition of done (frontend)

- [ ] Four pages work against the real backend; no mocked data in production code.
- [ ] Light and dark themes polished; layout verified at 1440, 1024, 768 and 390 px.
- [ ] Every data view has loading, error and empty states; the missing-model (503) state is friendly.
- [ ] Gantt chart shows pid labels, time axis, hover tooltips, `IDLE`/`CS` styling, playback with keyboard controls.
- [ ] No client-side scheduling logic; displayed numbers equal API responses.
- [ ] `npm run lint`, `npm run build`, `npm test` pass; Playwright smoke test passes.
- [ ] Screenshot checklist captured automatically.
- [ ] `frontend/README.md` documents setup (`npm install`, `.env`, `npm run dev`, `npm run gen:types`) and how to run the backend alongside.
