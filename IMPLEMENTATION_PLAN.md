# AI-Driven CPU Scheduling Simulator — Implementation Plan v2

**Course:** Operating System (AI3002), AY 2026–27, Topic 1
**Deliverable:** A reproducible, tested Python simulator with FCFS / SJF / Round Robin, a Decision Tree recommender, a tabular Q-learning scheduler (bonus), an evaluation suite, and a React dashboard.
**Companion file:** `FRONTEND_GUIDE.md` (read it only when you reach Stage 9).

---

## 0. How the coding agent must work

1. Work **one stage at a time**, in order. Do not start Stage N+1 until every acceptance criterion of Stage N passes.
2. At the start of each stage, re-read Sections 3 and 4 (conventions and semantics). They are the source of truth.
3. At the end of each stage: run `pytest -q`, run the stage's smoke command, then append a short entry to `docs/PROGRESS.md` (what was done, test status).
4. Any time you deviate from this plan, or make a judgment call, add an entry to `docs/DECISIONS.md` with the **rationale** (what, why, alternatives rejected). Prefer small, targeted edits to existing files over rewrites.
5. **Never invent results.** Every number in tables, figures or docs must come from a script run. Save raw outputs to `results/`.
6. Never touch test-split data (Section 3.4) before Stage 7.
7. If an acceptance criterion cannot be met, stop and write the blocker to `docs/DECISIONS.md` instead of weakening the test.

---

## 1. Review of the v1 plan: what stays, what changes

The v1 plan has a good skeleton (virtual CPU, three baselines, Decision Tree, Q-learning, evaluation discipline, seeds, held-out test set). These parts stay. The following issues would have hurt the project, so they change:

| # | Problem in v1 | Why it matters | Change in v2 |
|---|---|---|---|
| 1 | RL state = full per-process tuple (4 slots × status × remaining × waiting + time bucket + previous process) | Millions of states; 10k episodes will not cover them, so the agent memorizes instead of generalizing. Slot identity (P1..P4) is meaningless to scheduling. | Compact, **permutation-invariant** state of aggregate features (about 1–2k states). |
| 2 | Action = "pick process P_i" | Same slot-identity problem; max 4 processes. | Action = **pick by rule** (shortest remaining, longest remaining, earliest arrival, continue current). Works for any process count. |
| 3 | Decision Tree labelled by average waiting time alone | Non-preemptive SJF wins almost every workload, so the classifier is trivial (predict "SJF"). | **Composite objective** (waiting + response + turnaround), context-switch cost, scenario-targeted workloads, a **pilot gate** on class balance, and "regret" as a metric. |
| 4 | RL is preemptive per-unit; baselines are non-preemptive | Unfair comparison. The preemptive optimum for waiting time is SRTF, which the plan never includes. | Add **SRTF** as a reference baseline. Every algorithm runs through the **same engine**. |
| 5 | Gamma = 0.9 | The objective (total waiting time) is undiscounted and episodic. Discounting biases the agent toward short-term reward. | Gamma = 1.0 baseline, 0.99 and 0.95 tried on validation. |
| 6 | Reward not verified against the metric | Easy to ship a reward that does not equal the claimed objective. | Unit test: **episode return == −(sum of waiting times)** exactly. |
| 7 | Learning curve from noisy training reward | Epsilon-greedy training reward is not the policy's quality. | Learning curve = **greedy evaluation on a fixed validation set** every N episodes, multiple training seeds, mean ± std. |
| 8 | Context switches optional / vague | Without switch cost, the RL problem has an obvious answer (always SRTF) and RR looks artificially good. | One precise context-switch definition (Section 4.3), used by every algorithm, evaluated at cost 0 **and** cost 1. |
| 9 | Fairness only mentioned | The guide's professor will ask about starvation. | Add max waiting time and Jain's fairness index to metrics. |
| 10 | Streamlit | You asked for a professional React dashboard. | FastAPI + React (Stage 8–9), built **after** the core is done. |

---

## 2. Locked decisions (answers to your open questions)

### 2.1 Do we need a dataset?

| Component | Dataset needed? | Where the data comes from |
|---|---|---|
| Classical algorithms | No | Hand-written fixed test workloads (known correct answers). |
| **Decision Tree** | **Yes (supervised, labelled)** | Generated: thousands of synthetic workloads, each run through all three algorithms, labelled with the best one. This is the "workload data collection" the assignment names. |
| **Q-learning** | **No labelled dataset** | It learns from interaction. It needs an **environment plus a workload generator**, and it saves **episode logs** for plotting. |
| Evaluation | Yes (held-out workloads) | Separate seed ranges (Section 3.4). |
| Optional realism check | Optional | If time permits, a small set of workloads derived from a public trace format (e.g. Standard Workload Format). Treated only as an extra test set, never for training. |

Synthetic data is acceptable and expected here, but the report must say clearly that results describe the simulator's workload distributions, not a real OS.

### 2.2 Is a dashboard needed?

The assignment does not require one. It does require **screenshots of implementation, input, output and results**, and the topic is literally a "simulator". A small, polished dashboard makes the project easy to demonstrate and score. Decision: **build it, but only after Stage 7 passes**. The core must be fully usable from a CLI first, so a dashboard problem can never block submission. Stack: **FastAPI backend + React (Vite, TypeScript, Tailwind, shadcn/ui, Recharts)**. Details are in `FRONTEND_GUIDE.md`.

### 2.3 Other locked decisions

- Single-core virtual CPU, integer time units, integer bursts.
- One unified **unit-step engine** runs every algorithm (baselines and RL). This guarantees identical semantics and metrics.
- Algorithms: FCFS, SJF (non-preemptive), Round Robin (quantum configurable, default 2), plus **SRTF** as reference only. The Decision Tree chooses among FCFS / SJF / RR (the three the assignment names).
- Priority is stored on processes for completeness but no algorithm uses it. Do not implement priority scheduling.

---

## 3. Global conventions

### 3.1 Stack

- Python 3.11+, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `pydantic` v2, `fastapi`, `uvicorn`, `pytest`, `pytest-cov`, `ruff`.
- No deep-learning frameworks. No RL libraries.

### 3.2 Repository layout

```
cpu-scheduling-ai/
├── README.md
├── pyproject.toml            # or requirements.txt
├── config.py                 # all constants (Section 3.3)
├── src/schedsim/
│   ├── domain/    process.py  engine.py  metrics.py  timeline.py
│   ├── policies/  base.py  fcfs.py  sjf.py  srtf.py  round_robin.py  registry.py
│   ├── workloads/ generator.py  scenarios.py  presets.py  io.py
│   ├── ml/        features.py  labeling.py  dataset_builder.py  train_tree.py  evaluate_tree.py
│   ├── rl/        actions.py  state.py  env.py  agent.py  train.py  evaluate.py
│   ├── evaluation/ run_experiments.py  stats.py  plots.py
│   └── api/       main.py  schemas.py  services.py  routes/
├── cli.py                    # simulate / compare / build-dataset / train-tree / train-rl / evaluate
├── tests/
├── data/                     # workloads.csv, scheduling_dataset.csv (generated)
├── models/                   # decision_tree.joblib, q_table.pkl, q_table_summary.json
├── results/                  # CSVs, JSON, figures/
├── docs/                     # PROGRESS.md, DECISIONS.md, REPORT_NOTES.md
└── frontend/                 # Stage 9, see FRONTEND_GUIDE.md
```

### 3.3 `config.py` constants

```python
TIME_QUANTUM = 2
CONTEXT_SWITCH_COST = 0          # core default; experiments also run with 1
LABEL_CS_COST = 1                # used for Decision Tree labelling (see Stage 4, may be revised by the pilot)
LABEL_WEIGHTS = {"wait": 0.5, "resp": 0.3, "tat": 0.2}
LABEL_TIE_TOLERANCE = 0.02
MAX_PROCESSES = 12               # generator upper bound
# RL defaults (tuned on validation in Stage 6)
ALPHA = 0.1; GAMMA = 1.0; EPS_START = 1.0; EPS_MIN = 0.05; EPS_DECAY = 0.9995
RL_EPISODES = 20000; RL_EVAL_EVERY = 500; RL_TRAIN_SEEDS = [0, 1, 2, 3, 4]
```

### 3.4 Seed ranges and splits (never mix)

| Split | Workload seeds | Used for |
|---|---|---|
| Train | 0 – 99,999 | DT training data, RL training episodes |
| Validation | 100,000 – 100,499 | Hyperparameter selection, RL learning curve |
| Test | 200,000 – 200,999 | Final evaluation only (Stage 7) |
| Fixed presets | named, hand-written | Correctness tests, screenshots |

Every workload is `(scenario, n_processes, seed)` and must regenerate identically. For the Decision Tree, split **by workload** (and keep scenario proportions equal across splits).

### 3.5 Quality rules

Type hints everywhere, dataclasses for domain objects, no global mutable state, no randomness without an explicit seeded `numpy.random.Generator`, docstrings on public functions, `ruff` clean.

---

## 4. Semantics specification (the source of truth)

### 4.1 Time model

- Time is an integer tick counter. Tick `t` means the interval `[t, t+1)`.
- A process with `arrival <= t` and `remaining > 0` is **ready** at tick `t`.
- Idle: if nothing is ready, the engine jumps to the next arrival and records one `IDLE` segment. The policy is not consulted while idle.

### 4.2 Engine contract

```python
@dataclass(frozen=True)
class ProcView:      pid; arrival; burst; remaining; waited; last_run_end   # waited = ticks spent ready and not running

@dataclass(frozen=True)
class View:          time; ready: tuple[ProcView, ...]; last_pid: str|None; newly_arrived: tuple[str, ...]

class Engine:
    def __init__(self, processes, cs_cost: int = 0): ...
    def reset(self) -> View: ...
    def step(self, pid: str) -> StepInfo: ...   # runs ONE execution tick of `pid` (after any context-switch overhead)
    @property
    def done(self) -> bool: ...
    def result(self) -> SimResult: ...          # per-process metrics, aggregate metrics, timeline

def run(policy, processes, cs_cost=0) -> SimResult   # loops: view -> policy.select(view) -> step
```

`StepInfo.penalty` = number of ready processes that did **not** execute during the ticks consumed by this step (this is exactly the RL reward magnitude).
`newly_arrived` lists processes admitted since the previous `View` was produced, ordered by `(arrival, pid)`.

### 4.3 Context switch (single definition, used everywhere)

- A **dispatch** happens when `step(pid)` is called. If `last_pid is not None` and `pid != last_pid` and `cs_cost > 0`, the engine first spends `cs_cost` overhead ticks (timeline label `CS`, no process progresses, arrivals are still admitted, the decision is not revisited), then one execution tick.
- The first dispatch of a run is free. Idle gaps do not reset `last_pid`.
- Switching away from a finished process counts as a switch. Continuing the same pid never does.
- `context_switches` = number of dispatches where `pid != last_pid` and `last_pid is not None` (counted even when `cs_cost = 0`).
- Overhead ticks count as waiting time for every ready process, including the one about to run.

### 4.4 Policies (all unit-step, all deterministic)

Tie-break order everywhere: smaller key first, then earlier arrival, then lexicographically smaller pid.

| Policy | `select(view)` rule |
|---|---|
| FCFS | ready process with minimum `(arrival, pid)`. This never preempts, because the running process already has the minimum. |
| SJF (non-preemptive) | if `last_pid` is ready (still running), keep it; otherwise minimum `(burst, arrival, pid)`. |
| SRTF (reference) | minimum `(remaining, running-first, arrival, pid)`. Running process wins ties (avoids pointless switches). |
| Round Robin(q) | FIFO deque. On each call: append `newly_arrived` in order; if the current process used its full quantum and is still ready, append it **after** those arrivals; keep running the current process until quantum or completion; else pop left. Quantum counts execution ticks only (not CS overhead). |

### 4.5 Metrics

Per process: `completion`, `turnaround = completion − arrival`, `waiting = turnaround − burst`, `response = first_start − arrival`. With `cs_cost > 0`, waiting includes overhead time.

Aggregate: `avg_waiting`, `avg_turnaround`, `avg_response`, `max_waiting`, `makespan`, `cpu_utilization = busy_execution_ticks / makespan`, `overhead_ticks`, `throughput = n / makespan`, `context_switches`, `jain_fairness` computed over per-process slowdown `turnaround/burst`: `(Σx)² / (n·Σx²)`.

### 4.6 Hand-verified fixture (must be a test; `cs_cost = 0`)

Workload: P1 (arr 0, burst 5), P2 (arr 1, burst 3), P3 (arr 2, burst 2).

| Policy | Completions (P1, P2, P3) | Waiting (P1, P2, P3) | Avg wait | Avg turnaround | Avg response |
|---|---|---|---|---|---|
| FCFS | 5, 8, 10 | 0, 4, 6 | 3.333 | 6.667 | 3.333 |
| SJF | 5, 10, 7 | 0, 6, 3 | 3.000 | 6.333 | 3.000 |
| SRTF | 10, 4, 6 | 5, 0, 2 | 2.333 | 5.667 | 0.667 |
| RR q=2 | 10, 9, 6 | 5, 5, 2 | 4.000 | 7.333 | 1.000 |

Check these numbers yourself before using them in tests. If you find a mistake, log it in `DECISIONS.md` and fix the table, but never alter the engine to match a wrong table.

---

## 5. Stages

### Stage 0 — Project scaffold

**Goal:** empty but runnable repo.
**Tasks:** create layout (3.2), `pyproject.toml`, `config.py`, `docs/` stubs, `pytest` + `ruff` config, `README.md` skeleton, `.gitignore` (ignore `data/*.csv`, `models/*`, but keep `.gitkeep`).
**Acceptance:** `pytest -q` runs (zero tests OK), `python cli.py --help` prints commands.

### Stage 1 — Domain core: process, engine, metrics, timeline

**Tasks:**
1. `Process` dataclass with validation (`burst >= 1`, `arrival >= 0`, unique pids).
2. Engine per 4.1–4.3 (unit-step, idle jump, CS overhead, arrivals).
3. `timeline.py`: per-tick log merged into segments `{label, start, end}` where label is a pid, `CS` or `IDLE`.
4. `metrics.py` per 4.5.
5. A trivial scripted policy for testing (feeds a fixed pid sequence).

**Tests:** idle gap at start; two processes arriving at the same tick; CS overhead with `cs_cost = 2` (hand-calculate one example); `penalty` sums equal `Σ waiting`; segments partition `[0, makespan)` with no gaps or overlaps; conservation: busy ticks == Σ bursts.
**Acceptance:** all tests pass; engine has no policy-specific logic.

### Stage 2 — Classical policies

**Tasks:** implement FCFS, SJF, SRTF, RR (4.4) and a `registry.py` (`get_policy("RR", quantum=2)`). Add `cli.py simulate --policy RR --preset 1` that prints a table and an ASCII Gantt.
**Tests:**
- the 4.6 fixture for all four policies;
- RR tie convention (arrival at the exact tick a quantum expires goes ahead of the preempted process);
- property tests on 200 seeded random workloads: SRTF avg waiting ≤ every other policy's (at `cs_cost = 0`); every policy's completion set is complete; `Σ burst == busy ticks`;
- with `cs_cost = 1`, context-switch counts match a manual count on one fixture.
**Acceptance:** all pass; **5–10 named presets** in `workloads/presets.py` (include the v1 examples, a convoy case, equal bursts, simultaneous arrivals, a long idle gap).

### Stage 3 — Workload generator and data layer

**Tasks:**
1. `generator.py`: `generate_workload(scenario, n, seed) -> list[Process]`, using `numpy.random.default_rng(seed)`.
2. Scenarios (`scenarios.py`), each a parameter set for burst/arrival distributions:
   - `short_jobs`, `long_jobs`, `mixed`, `bimodal` (many short, few very long),
   - `simultaneous` (all arrive near 0), `staggered` (spread arrivals),
   - `convoy` (one long job first, many short jobs right after),
   - `equal_bursts`, `high_load` (arrivals denser than service capacity).
3. `io.py`: save and load workloads to `data/workloads.csv` (`workload_id, scenario, seed, pid, arrival, burst, priority`).
4. Split helpers implementing 3.4.

**Tests:** same seed gives the same workload; different seeds differ; bounds respected (`1 <= n <= MAX_PROCESSES`, `burst >= 1`); every scenario produces its intended property (e.g. convoy's first job is at least 3× the median of the others); CSV round-trip is lossless.
**Acceptance:** `cli.py build-workloads --n 6000` writes the file in under a minute.

### Stage 4 — Decision Tree (supervised component)

**4a. Features** (`features.py`), computed **only from the workload, never from simulation results**: `n`, `mean_burst`, `std_burst`, `cv_burst`, `max_burst`, `min_burst`, `max_over_median_burst`, `mean_interarrival`, `std_interarrival`, `arrival_span`, `load_ratio = Σburst / (arrival_span + 1)`, `frac_simultaneous` (fraction of processes sharing an arrival tick with another).

**4b. Labelling** (`labeling.py`): run FCFS, SJF, RR(q) with `LABEL_CS_COST`. For each algorithm compute the **composite score**:
`score = w_wait·(avg_wait / best_wait) + w_resp·(avg_resp / best_resp) + w_tat·(avg_tat / best_tat)`
where each `best_*` is the minimum across the three algorithms (add a small epsilon to avoid division by zero). Lower is better. Label = argmin. Also store `margin` (relative gap between best and second best) and the three raw scores. If margin < `LABEL_TIE_TOLERANCE`, the row is flagged `ambiguous`.

**4c. Pilot gate (mandatory before the full build):** label 600 pilot workloads under a few principled settings (cs cost 0 or 1, quantum 2 or 4, weights as in config). Print the class distribution per setting. Pick the setting by reasoning (realistic switch cost, balanced response/waiting emphasis), not by chasing balance blindly. Record the choice and the distribution in `DECISIONS.md`. If one class still holds more than 85% of labels, say so in the report, rely on class weights and macro-F1, and do not hide it.

**4d. Dataset build:** `cli.py build-dataset` writes `data/scheduling_dataset.csv` with features, label, margin, ambiguous flag, scenario, seed, split.

**4e. Training** (`train_tree.py`): `DecisionTreeClassifier` with grid search (`max_depth` ∈ {3,4,5,6,8}, `min_samples_leaf` ∈ {1,5,20}, `class_weight` ∈ {None, "balanced"}) using cross-validation on the train split only; choose on validation. Baselines for comparison: majority class, `RandomForestClassifier`, `KNeighborsClassifier` (scaled features). Save `models/decision_tree.joblib` plus a `models/decision_tree_meta.json` (feature order, class names, chosen params, label config).

**4f. Evaluation** (`evaluate_tree.py`, validation split now, test split only in Stage 7): accuracy, macro-F1, confusion matrix, feature importances, and **regret** = mean relative composite-score excess of the predicted algorithm over the true best (this is the practical metric: a wrong prediction that costs 1% is not a real failure). Also report accuracy excluding ambiguous rows. Save a readable tree plot (depth capped at 4 for the figure) and JSON metrics to `results/`.

**Acceptance:** model loads and predicts from a raw workload through a single function `recommend(processes) -> {algorithm, probabilities, features}`; metrics JSON exists; the tree beats the majority-class baseline on macro-F1, or the report states honestly that it did not.

### Stage 5 — RL environment

**Tasks (`rl/actions.py`, `rl/state.py`, `rl/env.py`):**

**Actions (4):** `0 SHORTEST_REMAINING`, `1 LONGEST_REMAINING`, `2 EARLIEST_ARRIVAL`, `3 CONTINUE_CURRENT`. Each maps a `View` to a pid with the tie-break order from 4.4. `CONTINUE_CURRENT` is valid only if `last_pid` is ready; otherwise it is masked. If two actions map to the same pid, keep both (the agent sees them as equal-value choices).

**State (tuple of small ints, permutation-invariant):**

| Feature | Buckets |
|---|---|
| ready count | 1, 2, 3, 4+ → 4 |
| shortest remaining among ready | 1, 2, 3–4, 5–8, 9+ → 5 |
| longest remaining among ready | 1–2, 3–4, 5–8, 9–16, 17+ → 5 |
| max waited among ready | 0, 1–2, 3–5, 6+ → 4 |
| last process still ready? | 0/1 |
| more arrivals pending? | 0/1 |
| is last process the shortest-remaining? | 0/1 (only meaningful if it is ready) |

That is at most about 3,200 raw combinations (far fewer reachable). Bucket edges live in `config.py`. Do **not** put absolute time or pids in the state.

**Environment:** wraps the Engine. `reset(workload) -> state`; `valid_actions() -> list[int]`; `step(action) -> (next_state, reward, done, info)` with `reward = -StepInfo.penalty`. Valid actions for the **next** state are always obtained from the environment after the step (v1 pseudocode warning).

**Tests:**
- **Reward identity:** for 300 random workloads and random action sequences, `Σ reward == −Σ waiting_time` from `metrics`. Run at `cs_cost` 0 and 1.
- A scripted agent always choosing `SHORTEST_REMAINING` reproduces the SRTF policy's metrics exactly at `cs_cost = 0`.
- Always `EARLIEST_ARRIVAL` reproduces FCFS; always `CONTINUE_CURRENT` (fallback to earliest arrival) reproduces non-preemptive behaviour.
- Masking: `CONTINUE_CURRENT` is never offered when the last process finished.
- State is hashable, identical for permuted pids.

**Acceptance:** all pass; a random-action rollout completes on 1,000 workloads without errors.

### Stage 6 — Q-learning agent and training

**Tasks:**
1. `agent.py`: sparse dict Q-table `{(state, action): value}`, epsilon-greedy over **valid** actions, standard update `Q ← Q + α[r + γ·max_{a'∈valid(s')} Q(s',a') − Q]`, terminal target = `r`. Greedy ties broken by lowest action index. For a never-seen state at evaluation time, fall back to `EARLIEST_ARRIVAL` (a neutral choice, not the known-best one) and **count** how often this happens.
2. `train.py`: each episode draws a workload from the **train** seed range with a scenario sampled uniformly and `n` in 3–10; epsilon decays per episode; every `RL_EVAL_EVERY` episodes run the greedy policy on the 500 **validation** workloads and log `avg_waiting`, `avg_turnaround`, `unseen_state_rate`, `q_table_size`. Write `data/rl_training_logs.csv`.
3. Hyperparameter search on validation only: α ∈ {0.05, 0.1, 0.2}, γ ∈ {0.95, 0.99, 1.0}, ε decay ∈ {0.999, 0.9995}. Use at least 3 training seeds per configuration and compare means. Log the table in `results/rl_hparam_search.csv`.
4. Final training with the chosen config on `RL_TRAIN_SEEDS` (5 seeds). Save the best-by-validation Q-table as `models/q_table.pkl` with metadata (config, seed, git hash if available) and a human-readable `models/q_table_summary.json` (for each state: visit count, best action, Q-values; cap at the top 500 states by visits).
5. `RLPolicy` wrapper so the trained agent also plugs into `engine.run` as a normal policy (greedy, no learning).

**Expectations (write these in the report as the hypothesis, then confirm or refute with data):**
- At `cs_cost = 0` the optimal policy is SRTF, so the agent should converge toward choosing `SHORTEST_REMAINING` almost always and approach SRTF's waiting time. This is a sanity check, not the interesting result.
- At `cs_cost = 1` the agent has a real trade-off (switching costs time), so it may learn to continue the current process in some states and beat both SRTF and RR on total cost. Train and evaluate a **separate Q-table per `cs_cost`** and report both.

**Acceptance:** learning curve (validation avg waiting vs episode) improves or stabilizes across seeds; unseen-state rate on validation < 5% (otherwise train longer or coarsen the state); `RLPolicy` runs through the same `run()` as the baselines.

### Stage 7 — Evaluation, statistics and figures

**Tasks (`evaluation/run_experiments.py`, `stats.py`, `plots.py`):**
1. Run FCFS, SJF, RR(2), SRTF, Q-learning on the **test** seeds (1,000 workloads, all scenarios), at `cs_cost` 0 and 1. Save per-workload rows to `results/evaluation_results.csv` and aggregates to `results/summary_table.csv` and `.md`.
2. Report each metric as mean ± 95% bootstrap CI (1,000 resamples). Add per-scenario breakdown and win-rate vs each baseline.
3. Decision Tree on the test split: accuracy, macro-F1, confusion matrix, regret. Plus an end-to-end check: for each test workload, run the algorithm the tree recommends and compare with the best fixed algorithm and with always-SJF.
4. Fairness: `max_waiting` and Jain index per policy; flag starvation (max waiting more than 3× average waiting, count of workloads).
5. Figures in `results/figures/` (PNG, 200 dpi, consistent style): Gantt charts for the fixture under all policies, RL learning curve (mean ± std over seeds), policy action share per state bucket, comparison bars with CIs, DT confusion matrix, feature importances, tree plot, class distribution.
6. Write `docs/REPORT_NOTES.md` mapping every figure and table to the report section list (Section 6 below), plus an honest "Limitations" list.

**Acceptance:** `cli.py evaluate` regenerates all tables and figures from scratch deterministically; no figure or number is hand-typed; if the RL agent does not beat a baseline, the results still ship with an honest explanation.

### Stage 8 — Backend API (FastAPI)

**Goal:** a thin layer over the library. No scheduling logic in the API.
**Endpoints (all JSON, prefix `/api`):**

| Method + path | Purpose |
|---|---|
| `GET /health` | status plus which model files are loaded |
| `GET /presets` | list preset workloads |
| `POST /workloads/generate` | `{scenario, n, seed}` → processes |
| `POST /simulate` | `{processes, algorithm, quantum, cs_cost}` → per-process metrics, aggregates, segments, per-tick ready-queue log. For `QLEARNING`, also a per-decision trace: state, valid actions, Q-values, chosen action, reward |
| `POST /compare` | same workload through all algorithms (and Q-learning if a model exists) |
| `POST /recommend` | Decision Tree recommendation: algorithm, class probabilities, features |
| `GET /ml/model-info` | metrics JSON, feature importances, confusion matrix, class distribution |
| `GET /rl/training-log` | learning-curve data |
| `GET /rl/policy-summary` | contents of `q_table_summary.json` |
| `GET /results/summary` | the final summary tables |

**Requirements:** Pydantic request/response schemas in `schemas.py` (limits: n ≤ 50 processes, burst ≤ 100, arrival ≤ 500); CORS for `http://localhost:5173`; models loaded once at startup; if a model file is missing, AI endpoints return HTTP 503 with a clear message (`"Train the model first: python cli.py train-rl"`) and the rest of the API still works; consistent error format `{error, detail}`; OpenAPI served at `/docs`.
**Tests:** FastAPI `TestClient` for each endpoint, a 503 test with models absent, a payload-limit test, and one test confirming `/simulate` output equals `run()` output directly.
**Acceptance:** `uvicorn schedsim.api.main:app` serves everything; OpenAPI schema is valid.

### Stage 9 — React dashboard

Follow `FRONTEND_GUIDE.md` completely (stack, design system, pages, components, state, testing, screenshot checklist). Build it in its own sub-stages F1–F6 there.
**Acceptance:** the guide's definition of done.

### Stage 10 — Submission pack

**Tasks:**
1. `README.md`: install, run commands, reproduction steps (`cli.py build-workloads → build-dataset → train-tree → train-rl → evaluate`), seeds, and folder map.
2. `docs/REPORT_NOTES.md` final: section-by-section material for the nine required report sections.
3. A `screenshots/` folder with the captures listed in the guide's checklist (input data, algorithm outputs, Gantt charts, DT results, RL curve, comparison).
4. A one-command smoke test: `make smoke` (or a script) that runs a tiny version of the whole pipeline in under two minutes.
5. Clean-clone check: fresh virtualenv, install, run smoke test, run the test suite.

**Acceptance:** a fresh clone reproduces the figures and tables without manual edits.

---

## 6. Report mapping (assignment's required structure)

| Report section | Where the material comes from |
|---|---|
| 1. Title and introduction | `README.md`, CPU scheduling background |
| 2. Problem statement and objectives | Assignment objective, Sections 1–2 of this plan |
| 3. Existing system and proposed methodology | FCFS/SJF/RR and their limitations, then the DT + Q-learning proposal |
| 4. Working principle and algorithm | Section 4 (semantics), Stage 4 labelling, Stage 6 Q-learning update and reward |
| 5. Flowchart / system architecture | Pipeline: workload generator → engine → features/labels → DT; engine ↔ RL env ↔ agent; API ↔ dashboard |
| 6. Implementation and sample input data | Stages 1–6, preset workloads, dataset head, screenshots |
| 7. Output results and performance analysis | Stage 7 tables, CIs, figures, per-scenario analysis, fairness |
| 8. Conclusion and future scope | Honest findings, limits, future work (more processes, DQN, priorities, multi-core, real traces) |
| 9. References | Silberschatz, Galvin & Gagne, *Operating System Concepts*; Tanenbaum & Bos, *Modern Operating Systems*; Sutton & Barto, *Reinforcement Learning: An Introduction*; Watkins & Dayan (1992), "Q-learning", *Machine Learning* 8; Pedregosa et al. (2011), scikit-learn, JMLR 12; Jain, Chiu & Hawe (1984), fairness index (DEC-TR-301). Verify details against the originals before citing. |

---

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Decision Tree labels are 90%+ one class | Pilot gate (4c), regret metric, macro-F1, honest reporting |
| Q-learning does not beat SRTF at `cs_cost = 0` | Expected, since SRTF is optimal there. The claim to test is "matches SRTF at 0, helps at cost > 0" |
| State coverage too thin | Track unseen-state rate; coarsen buckets; more episodes |
| Engine/policy subtle bugs | Shared engine, hand-verified fixture, property tests, reward identity test |
| Dashboard consumes the schedule | It starts only after Stage 7, and the CLI + matplotlib figures are a complete fallback for screenshots |
| Results look "too good" | Held-out seeds, multiple training seeds, bootstrap CIs, label/feature leakage test (features never use simulation output) |

---

## 8. Definition of done

- [ ] Stages 0–7 and 10 complete; Stages 8–9 complete or consciously dropped with a note in `DECISIONS.md`.
- [ ] `pytest -q` is green; `ruff` clean.
- [ ] All results regenerate from `cli.py evaluate`.
- [ ] The assignment's explicit items are covered: FCFS/SJF/RR, workload data collection, classification model (Decision Tree), RL bonus (Q-learning), screenshots of input, output and results, and a report with references.
