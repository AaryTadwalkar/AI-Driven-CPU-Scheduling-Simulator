# Revised Implementation Plan v3 — AI-Driven CPU Scheduling Simulator (Simplified)

**Supersedes:** the model/evaluation scope of `PID_SELECTOR_ADDENDUM_PLAN.md` (Stages 11–21) and Stage 10 of `IMPLEMENTATION_PLAN.md`.
**Keeps unchanged:** everything in `IMPLEMENTATION_PLAN.md` Stages 0–9 (engine, classical policies, generator, Decision Tree pipeline, Q-learning, evaluation, API, dashboard) and its Sections 3–4 (conventions and semantics).
**Priority:** simplicity. The one question this project answers: *which scheduler does best, once the cost of running the AI model is included?*

Working rules from the original plan still apply: one stage at a time, tests green before moving on, every deviation logged in `docs/DECISIONS.md` with rationale, surgical edits over rewrites, no invented numbers, test split untouched until the final comparison (Stage 18).

---

## 1. Final model roster (5 trained models, nothing else is trained)

| # | Category | Model | Type | Decides | Inference charged |
|---|---|---|---|---|---|
| 1 | Algorithm-level | **Decision Tree** (existing, Stage 4) | Classical ML, baseline | FCFS / SJF / RR once per workload | Once, at start |
| 2 | Algorithm-level | **Random Forest** (small) | Classical ML, stronger | FCFS / SJF / RR once per workload | Once, at start |
| 3 | Algorithm-level (rule-level) | **Q-learning** (existing, Stage 6) | Reinforcement learning (tabular) | One of 4 rules at every decision | Every non-forced decision |
| 4 | Process-level (PID) | **Gradient-Boosted Candidate Ranker (GBDT)** | Classical ML | Which ready process runs next | Every non-forced decision |
| 5 | Process-level (PID) | **Neural Candidate Scorer** | **Deep learning** | Which ready process runs next | Every non-forced decision |

**Dropped from the old plans (code may stay in the repo, but it is not featured, trained further or reported):** k-NN, majority-class model as a *model* (it is kept only as a one-line reference number), Random Forest as "just a baseline" (it is now a real Category 1 model), PPO fine-tuned scorer (M3), exact DP solver, SLOWDOWN objective (S2/S3), decision-granularity `HoldPolicy`, T-large and T-holdout test sets, all ablations, Wilcoxon/Holm tests.

**Non-trained components kept (each minimal):** classical schedulers (FCFS, SJF, RR, SRTF), rollout oracle (labels only), latency profiler, overhead cost model.

### 1.1 Why these choices (record in `DECISIONS.md`)

| Decision | Rationale | Rejected alternative |
|---|---|---|
| DT = baseline, RF = strong model | The tree is interpretable and nearly free to run; the forest is usually more accurate. Because Category 1 is charged **once per workload**, the forest's extra inference time (about milliseconds) is negligible against a whole schedule, so accuracy wins. Both already exist from Stage 4. | Gradient boosting or an MLP for Category 1: tabular data with about 12 features and a few thousand rows is where tree ensembles already do well; a neural net adds complexity with no expected gain. |
| Q-learning stays in the algorithm group | Fixed by the assignment owner. Note: it picks a *rule* (shortest-remaining, longest-remaining, earliest-arrival, continue-current) **at every decision**, not one algorithm per workload, so it is charged per decision, like the PID models. | Moving it to the PID group: it does not select a specific process by learned scoring. |
| PID group = GBDT ranker + Neural scorer | Same candidate-scoring formulation for both, so the comparison isolates the model type (small boosted trees vs small neural net). The neural scorer is the deep learning model. | PPO fine-tuning: needs a critic, KL penalty and tuning, is unstable, and is least likely to help; it would add a whole stage for a model that likely ties the neural scorer. |
| Deep learning only in the PID group | Per-decision, per-candidate scoring is the natural neural-net fit (variable-size sets, shared weights). | DL for Category 1: tabular + small data, trees win. |
| Two settings only | S0: `cs_cost = 0` (sanity: nobody can beat SRTF). S1: `cs_cost = 1` (SRTF no longer optimal, so AI has room). Both use the **waiting-time objective**. Matches the two Q-tables already trained in Part A. | Four settings with a slowdown objective: doubles training and reporting without changing the lesson. |

### 1.2 Plain-language meaning of the three PID models in the old addendum

| Old name | What it does | Verdict |
|---|---|---|
| Gradient-boosted candidate ranker | For each waiting process, builds a row of numbers (remaining time, how long it has waited, and so on). A small set of boosted decision trees gives each row a "run this now" score. The highest score runs. Trained to copy an expert. | **Keep (model 4)** |
| Neural candidate scorer | Same idea, but a small neural network produces the score, sharing the same weights across all candidates. Trained to copy the expert, then corrected with DAgger so it handles situations it creates itself. | **Keep (model 5, deep learning)** |
| PPO fine-tuned neural scorer | Takes the neural scorer and keeps training it by trial and error (RL) to try to beat the expert. | **Drop** |

---

## 2. Cross-check: what must already exist (Stages 0–9 audit)

I could only review the plans, not your code. Run this audit before anything new. Every item must pass; if one fails, fix it first (log in `DECISIONS.md`).

| # | Check | Command | Pass criterion |
|---|---|---|---|
| A1 | Four classical policies registered | `python -c "from schedsim.policies.registry import get_policy as g; [g(n) for n in ('FCFS','SJF','SRTF','RR')]"` | No exception |
| A2 | Hand-verified fixture (plan Section 4.6) | `pytest -q -k fixture` | FCFS 3.333, SJF 3.000, SRTF 2.333, RR(q=2) 4.000 average waiting |
| A3 | Property test: SRTF optimal at `cs_cost = 0` | `pytest -q -k srtf` | Pass on 200 random workloads |
| A4 | Reward identity | `pytest -q -k reward` | `Σ reward == −Σ waiting` at `cs_cost` 0 and 1 |
| A5 | Presets run through the CLI | `python cli.py simulate --policy RR --preset 1` | Table + Gantt printed; repeat for the other three policies |
| A6 | Test-split results exist for all classical policies and Q-learning, at both `cs_cost` values | `head results/summary_table.csv` | Rows for FCFS, SJF, RR, SRTF, QLEARNING at cs 0 and cs 1 |
| A7 | DT and RF artifacts | `ls models/` | `decision_tree.joblib` exists; RF metrics exist in `results/` (RF was a Stage 4e comparison model) |
| A8 | Q-tables per `cs_cost` | `ls models/` | One Q-table (or a clearly named pair) for cs 0 and cs 1 |
| A9 | Whole suite | `pytest -q && ruff check .` | Green |

**Status to assume if all pass:** classical baselines are done and need no new work. Stage 10 (submission pack) of the old plan is **not** done and moves to Stage 20 here.

---

## 3. New and changed global constants (`config.py`)

```python
# Settings (cs_cost only; objective is always total waiting time)
SETTINGS = {"S0": 0, "S1": 1}

# AI overhead model
TICK_US_SWEEP = [1, 10, 100, 1000, 10000]   # microseconds of real time per simulation tick
DEFAULT_TICK_US = 1000                       # headline tables
FORCED_DECISION_FREE = True                  # one ready process => no inference, no cost

# Random Forest (Category 1)
RF_PARAM_GRID = {"n_estimators": [50, 100], "max_depth": [4, 6, 8], "min_samples_leaf": [1, 5]}

# PID data / models
PID_BC_TRAIN_WORKLOADS = 3000               # per setting, train seed range only
DAGGER_ITERS = 2
DAGGER_WORKLOADS_PER_ITER = 1000
PID_MAX_STATES = 100_000                    # cap per setting
ORACLE_BASE_POLICIES = ("SRTF", "SJF")      # rollout bases, both already implemented
GBDT_PARAMS = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}  # tuned on validation closed-loop gap
NN_HIDDEN = 32
NN_SEEDS = [0, 1, 2]
```

Seed ranges are unchanged (Section 3.4 of the main plan): train 0–99,999, validation 100,000–100,499, test 200,000–200,999.

---

## 4. The overhead model (identical for all policies)

1. **Measure** each policy's decision latency in a micro-benchmark (Stage 17) as a function of ready-set size. Store in `models/latency_profile.json`.
2. **Convert** microseconds to ticks with an assumed `tick_us`. Never report one number: report the sweep `TICK_US_SWEEP`.
3. **Charge** as stall ticks labelled `AI`, using the same mechanism as context-switch overhead (every ready process waits; no one progresses). A fractional debt accumulator keeps totals exact.
4. **Who pays when:**

| Policy | Charge |
|---|---|
| FCFS, SJF, RR, SRTF | Their measured per-decision latency, every non-forced decision (a few microseconds) |
| Decision Tree, Random Forest | One charge at the first dispatch: feature extraction + prediction, measured from raw processes |
| Q-learning, GBDT ranker, Neural scorer | Per non-forced decision, measured by ready-set size |

5. **Forced decisions are free** for every policy (one ready process means nothing to decide).
6. **Headline metrics:** effective average turnaround and effective makespan (both include `AI` ticks), plus raw inference µs and overhead as a percentage of makespan.
7. **Limitations to state in the report:** Python timing is not kernel timing (use relative costs and the sweep, not absolute numbers); burst times are assumed known; Category 1 uses whole-workload statistics (clairvoyant), as in Part A; single core.

Engine hooks (small, backwards compatible):

```python
class Policy:
    def on_start(self, processes) -> float: return 0.0        # microseconds, charged once
    def decision_cost_us(self, view) -> float: return 0.0     # microseconds, per non-forced decision

class Engine:
    def __init__(self, processes, cs_cost=0, tick_us=None): ...   # tick_us=None => zero charge (Part A unchanged)
    def clone(self) -> "Engine": ...                               # copy mutable state only
```

---

## 5. Stages

### Stage 10 — Audit (Section 2)
**Acceptance:** A1–A9 all pass. Log the result in `docs/PROGRESS.md`.

### Stage 11 — Engine: AI stalls, decision counts, clone
**Tasks (edit `domain/engine.py`, `policies/base.py`, `domain/metrics.py`, `domain/timeline.py` only):**
1. Add `on_start` and `decision_cost_us` as default no-ops (Section 4). Add `tick_us` and the debt accumulator. `AI` ticks go after the decision and before any context-switch overhead.
2. New timeline label `AI`. New metrics: `ai_overhead_ticks`, `decisions`, `inference_us_total`, `effective_makespan`, `effective_turnaround`.
3. `Engine.clone()` and `restore()`.
4. Add `Setting(name, cs_cost)` for S0/S1 in `config.py`. No objective generalization is needed (waiting only).

**Tests / pass criteria:**

| Test | Pass |
|---|---|
| Entire Part A suite with default `tick_us=None` | Unchanged and green |
| Hand case: P1 (arr 0, burst 2), P2 (arr 0, burst 1), FCFS-like policy, 1 tick per non-forced decision, `cs_cost=0` | Decisions at t=0 and t=2, 2 `AI` ticks, completions P1=4, P2=5, waiting 2 and 4, average 3.0, `effective_makespan`=5. **Verify by hand before coding the test.** |
| 0.25 tick per decision over 8 decisions | Exactly 2 `AI` ticks |
| Forced decisions | No cost, not counted |
| `clone()` then divergent steps | Original unaffected; `restore()` reproduces state |
| Timeline | Partitions `[0, effective_makespan)` with labels `pid | CS | AI | IDLE` |

### Stage 12 — Category 1: Random Forest and per-setting retraining
**Tasks:**
1. Re-run the Stage 4 labelling with `LABEL_CS_COST` set to the setting (0 for S0, 1 for S1) so each setting has its own label set. Keep the Stage 4c pilot-gate rule: print class distribution and log it.
2. Train Decision Tree (existing grid) and Random Forest (`RF_PARAM_GRID`, cross-validation on train, choose on validation) **per setting**. Save `models/decision_tree_<S>.joblib`, `models/random_forest_<S>.joblib`, matching `_meta.json`.
3. `recommend(processes, model="dt"|"rf")` returns algorithm, probabilities, features.
4. `SelectorPolicy(model)`: `on_start` returns measured feature+predict latency (0 until Stage 17), then delegates to the chosen classical policy.
5. Q-learning: reuse the Part A Q-table for each `cs_cost`; wrap as `RLPolicy` with `decision_cost_us` (0 until Stage 17).

**Metrics (validation):** accuracy, macro-F1, regret (Stage 4f definition). Report the majority-class number as a one-line reference, no model.
**Acceptance:** each model beats the majority-class macro-F1 or the report says it did not; DT and RF both produce a full `regret` number; RF is compared with DT on **validation regret, not only accuracy**.

### Stage 13 — Rollout oracle (labels only)
**Tasks (`pidsel/oracle.py`):** for a state, for every valid candidate: clone the engine, apply the candidate, then finish the run with each policy in `ORACLE_BASE_POLICIES`; cost = immediate penalty + best finished cost. Return the argmin (ties: smaller remaining, then arrival, then pid) and the full cost vector (regret per candidate = cost − best). Parallelize labelling with deterministic ordering.

**Pass criteria:**

| Test | Pass |
|---|---|
| S0, 300 random workloads | Oracle total waiting == SRTF total waiting |
| S1, 300 random workloads | Oracle total ≤ `min(SRTF, SJF)` per workload; any violation is logged, not hidden |
| Determinism | Same workload gives identical output across runs and workers |

Dropped on purpose: the exact DP solver (the two checks above already validate the oracle against the known optimum at S0).

### Stage 14 — Per-candidate features and training data
**Features (`pidsel/features.py`), computed only from observable state, vectorized in NumPy:**

| Group | Features for ready process `i` at time `t` |
|---|---|
| Process | `log1p(remaining_i)`, `log1p(burst_i)`, `remaining_i/burst_i`, `log1p(waited_i)`, `is_last_i` |
| Relative | rank of remaining (0–1), rank of waited (0–1), `remaining_i/min_remaining`, `remaining_i/mean_remaining` |
| Context | `log1p(n_ready)`, `log1p(sum_remaining)`, `cs_cost`, pending arrivals (capped) |

Never include the pid string, slot index or anything from future simulation.

**Dataset (`pidsel/dataset.py`):** per setting, follow the oracle on `PID_BC_TRAIN_WORKLOADS` train workloads; at every non-forced state store the padded feature matrix, validity mask, oracle cost vector and metadata. Save `data/pid_bc_<S>.npz`, capped at `PID_MAX_STATES`.

**Pass criteria:** no state with a single candidate; no NaN/inf; shuffling the ready order only permutes the feature rows (test); same seeds rebuild an identical file; a one-page `results/pid_dataset_report.md` (state counts, how often the oracle picks shortest-remaining / current / earliest-arrival).

### Stage 15 — Model 4: Gradient-boosted candidate ranker
**Tasks (`pidsel/gbdt_ranker.py`):**
1. Each candidate becomes one row: candidate features + context. Label = 1 if its oracle cost is within 1% of the best, else 0. Sample weight = regret-based (bigger weight on costly mistakes).
2. Train `sklearn.ensemble.GradientBoostingClassifier(**GBDT_PARAMS)`. Tune depth, rounds and learning rate on **validation closed-loop gap** (below), not on log-loss.
3. Choose the candidate with the highest raw score (ordering only; no sigmoid needed).
4. **Export to flat arrays** for fast, fair inference. Each fitted tree in `model.estimators_[k, 0].tree_` exposes `children_left`, `children_right`, `feature`, `threshold`, `value`:

```python
def export(model):
    return [dict(l=t.tree_.children_left, r=t.tree_.children_right,
                 f=t.tree_.feature, th=t.tree_.threshold,
                 v=t.tree_.value[:, 0, 0]) for t in model.estimators_[:, 0]], model.learning_rate

def score(trees, lr, x):                    # x: [n_candidates, d] -> [n_candidates]
    out = np.zeros(len(x))
    for t in trees:
        node = np.zeros(len(x), dtype=int)
        while True:
            internal = t["l"][node] != -1
            if not internal.any(): break
            go_left = x[np.arange(len(x)), np.where(internal, t["f"][node], 0)] <= t["th"][node]
            node = np.where(internal, np.where(go_left, t["l"][node], t["r"][node]), node)
        out += lr * t["v"][node]
    return out
```

**Pass criteria:** exported scores match scikit-learn `decision_function` within 1e-6 on 10,000 rows; choices are permutation-equivariant; seeded training is deterministic.
**Validation metrics (500 validation workloads, per setting):** top-1 agreement with oracle (ties count), mean regret per decision, **closed-loop gap** `(J_policy − J_oracle)/J_oracle`, gap vs SRTF. **At S0 the gap vs SRTF must be near zero**; otherwise fix features before continuing.

### Stage 16 — Model 5: Neural candidate scorer (the deep learning model)
**Architecture (`pidsel/scorer_net.py`, PyTorch, CPU), about 4–5k parameters:**

```python
class Scorer(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2*h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x, mask):                       # x: [B, N, d], mask: [B, N] bool
        z = self.phi(x)                               # same weights for every candidate
        m = mask.unsqueeze(-1)
        ctx = (z * m).sum(1, keepdim=True) / m.sum(1, keepdim=True)   # mean-pool context
        s = self.head(torch.cat([z, ctx.expand_as(z)], -1)).squeeze(-1)
        return s.masked_fill(~mask, float("-inf"))    # masked candidates never chosen
```

**Training (`pidsel/train_bc.py`):**
1. Soft-label loss: target `p*(a) ∝ exp(−regret_a / T)` over valid candidates (tune `T` on validation); cross-entropy against the model's masked softmax.
2. **DAgger, `DAGGER_ITERS = 2`:** iteration 0 trains on oracle trajectories; each later iteration rolls out the current model on fresh train workloads (mixing in the oracle action with probability `0.5^i`), labels every visited non-forced state with the oracle, appends, retrains (warm start).
3. Adam, lr 1e-3, batch 512, early stopping on **validation closed-loop gap** (fixed set of 200 validation workloads), seeds `NN_SEEDS`. Keep the best checkpoint by validation gap.
4. Export weights to `models/pid_scorer_<S>_np.npz` and write a pure-NumPy forward pass; the benchmark and final evaluation use only the NumPy version.

**Pass criteria:** scores are permutation-equivariant; masked candidates never selected; torch vs NumPy outputs differ by < 1e-5; one epoch reduces loss on a fixed batch; DAgger dataset grows each iteration and contains learner-visited states; at S0 the closed-loop gap vs SRTF is within noise; at S1 the gap to the oracle is reported as it is. Results to `results/pid_m2_val.csv` (mean ± std over 3 seeds).
**Note for `DECISIONS.md`:** PyTorch is a deliberate, documented relaxation of Part A's "no deep-learning frameworks" rule, used for training only; inference is NumPy.

### Stage 17 — Latency profile and cost model (one small module, `overhead.py`)
**Tasks:**
1. Benchmark every policy (FCFS, SJF, RR, SRTF, Q-learning, GBDT, Neural) at `n_ready ∈ {2, 3, 5, 8, 10}` with `time.perf_counter_ns`: 200 warm-up calls, 2,000 timed calls, 3 repeats, median of medians, GC disabled, single thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`). Record p95 and machine info; warn if the coefficient of variation exceeds 10%.
2. Time DT and RF as **feature extraction + predict from raw processes** at several workload sizes (charged once via `on_start`).
3. Write `models/latency_profile.json` (`{policy: {n_ready: {median_us, p95_us}}}`) and a latency-vs-`n_ready` figure.
4. `decision_cost_us(view)` in every policy reads the profile with linear interpolation (extrapolation flagged and capped). Unit-test with a fake profile.
5. Sanity check: one end-to-end run in live-timing mode must land within 15% of the table-based total for the same run; log the gap.

**Acceptance:** profile file exists, figure saved, live-vs-table check logged.

### Stage 18 — Final comparison (the deliverable)
**Test set:** existing test seeds 200,000–200,999 only (n = 3–10, all scenarios). Part A already evaluated classical + Q-learning + DT here; the new models see this set for the first time. **Do not tune anything on it.**

**Hypotheses (write `docs/HYPOTHESES.md` and commit it before running):**
- **H1:** At S0 no policy beats SRTF; GBDT, Neural and Oracle are within noise of SRTF before overhead, and are worse than SRTF after overhead.
- **H2:** At S1 the oracle beats SRTF; the Neural scorer captures part of that gap.
- **H3:** Per-decision AI models (Q-learning, GBDT, Neural) lose to the best classical policy once `tick_us` is small enough; there is a break-even `tick_us`.
- **H4:** Category 1 (DT, RF) overhead is negligible (charged once) and its total cost is close to, but never better than, best-of-classical in hindsight.

**Run matrix:** S0, S1 × 4 classical + 5 AI + 2 references × `TICK_US_SWEEP`. Headline tables use `DEFAULT_TICK_US`.
References (not deployable): **Best-of-classical in hindsight** (per workload, best of FCFS/SJF/RR) and the **Rollout Oracle**.

**The comparison matrix (one table per setting; generated, never typed):**

| Column | Meaning |
|---|---|
| avg waiting, avg turnaround, avg response | Raw scheduling quality |
| **effective avg turnaround, effective makespan** | **Including AI stall ticks (the headline)** |
| max waiting, Jain fairness | Starvation and fairness check |
| context switches, decisions | Behaviour |
| inference µs total, overhead % of makespan | Cost of the model itself |
| gain over best classical | Per workload, then averaged |
| gap to oracle | `(J − J_oracle)/J_oracle` |
| win / tie / loss vs best classical | Tie if within 0.5% |

**Statistics (keep Part A's tooling):** mean ± 95% bootstrap CI (1,000 resamples), paired on identical workloads; per-scenario breakdown; learned models report mean ± std across training seeds. **Break-even:** for each per-decision AI model, the smallest `tick_us` in the sweep at which its effective cost is not worse than the best classical (linear interpolation between sweep points; no bootstrap on the break-even).

**Figures (scripted):** effective cost vs `tick_us` for every policy (one panel per setting); gap-to-oracle bars; overhead breakdown (execution, context switch, AI, idle); per-scenario heatmap; one Gantt per category on the same workload with `AI` stalls visible; latency vs `n_ready`.

**Acceptance:** `cli.py compare-all` regenerates all tables and figures from scratch deterministically; S0 sanity holds (if any learned model beats SRTF at S0 before overhead, find the bug first); `docs/HYPOTHESES_RESULTS.md` marks each hypothesis confirmed, refuted or inconclusive with the supporting figure. A refuted hypothesis is a finding, not a failure.

### Stage 19 — Minimal API and dashboard extension (optional, consciously droppable)
Reuse existing endpoints and pages; no new pages.
- `/simulate` and `/compare`: accept the new algorithm names `DT`, `RF`, `QLEARNING`, `PID_GBDT`, `PID_NN` and an optional `tick_us`; segments may now contain `AI`. Return 503 with a clear message when a model file is missing (existing pattern).
- `GET /results/final`: the generated Stage 18 headline tables.
- Frontend: add the new policies to the Compare page, an `AI` hatched segment style with legend in the Gantt, and one table (headline comparison matrix) on the Results page. Optional: a tick-length slider on the Compare page recomputing effective cost from stored overhead figures.
**Tests:** one API test per changed endpoint; `/compare` equals the library harness output.
**Screenshots added:** `15_comparison_matrix.png`, `16_cost_vs_tick.png`, `17_gantt_with_ai_stalls.png`.
If time is short, drop this stage and log it in `DECISIONS.md`; the CLI and figures are a complete deliverable.

### Stage 20 — Submission pack (the old Stage 10, moved to the end)
1. `README.md`: install; reproduction order `build-workloads → build-dataset → train-tree → train-forest → train-rl → build-pid-data → train-pid-gbdt → train-pid-nn → bench-latency → compare-all`; seeds; folder map; measured run times.
2. `docs/REPORT_NOTES.md`: material for the nine report sections, including the model roster (Section 1), overhead model (Section 4), hypotheses outcomes and the limitations list.
3. `screenshots/`: the original checklist plus the Stage 19 captures (or CLI/figure equivalents if Stage 19 was dropped).
4. `make smoke`: tiny end-to-end run in under two minutes.
5. Fresh-clone check: new virtualenv, install, `make smoke`, `pytest -q`.
**Acceptance:** a fresh clone reproduces all tables and figures without manual edits.

---

## 6. `docs/DECISIONS.md` entries to add now

1. Model roster reduced to 5 (Section 1) and why PPO, k-NN, majority model and the exact DP solver were dropped.
2. DT = baseline, RF = strong model for Category 1; accuracy is chosen over inference time because Category 1 is charged once.
3. Q-learning kept in the algorithm group as a rule-per-decision selector and charged per decision.
4. Only settings S0 and S1, waiting objective only.
5. PyTorch allowed for training the neural scorer only; inference in NumPy.
6. Oracle uses two base policies (SRTF, SJF), validated by the S0 and S1 checks instead of an exact solver.
7. Test seeds 200,000–200,999 were used in Part A; new models use them once, with hypotheses pre-registered.

---

## 7. Risks (reduced)

| Risk | Mitigation |
|---|---|
| Oracle labelling is slow in Python | Skip forced states, cap states, parallelize, fewer base policies (already 2) |
| Neural scorer matches SRTF at S0 and adds nothing at S1 | Acceptable finding; the overhead analysis still answers the assignment's question |
| Latency numbers vary between runs | Single thread, GC off, median of medians, table-based costs in simulation |
| Per-decision models look slow in Python | State the limitation; rely on relative cost and the tick sweep |
| Time runs out | Stages 12–18 are the core; Stage 19 is optional; every earlier stage leaves a working CLI |

---

## 8. Definition of done

- [ ] Stage 10 audit passes (classical FCFS, SJF, RR, SRTF verified).
- [ ] Stages 11–18 and 20 complete; Stage 19 complete or dropped with a note.
- [ ] Exactly five trained models: DT, RF, Q-learning, GBDT ranker, Neural scorer.
- [ ] One deep learning model (Neural scorer) present and compared.
- [ ] Comparison matrix includes AI inference time as stall ticks, with the `tick_us` sweep and break-even.
- [ ] `pytest -q` green, `ruff` clean, every reported number traceable to a generated file.
