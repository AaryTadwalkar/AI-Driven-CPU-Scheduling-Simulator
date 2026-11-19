# Pre-Registered Hypotheses Evaluation Results

**Document Generation Timestamp:** Automated final comparison execution.  
**Target Split:** Test partition seeds `200,000–200,999` ($N = 1,000$ workloads).  
**Baseline Hardware Tick:** $\text{DEFAULT\_TICK\_US} = 1000\,\mu\text{s}$.  

---

## Hypothesis 1 (H1: S0 Optimality Bound & Overhead Degradation): **CONFIRMED**

- **Pre-Registered Claim:** At Setting S0 ($c_s = 0$), no policy beats SRTF in raw waiting.
  Learned models perform within statistical noise of SRTF before overhead, and degrade after hardware overhead.
- **Empirical Evidence:**
  * SRTF raw average waiting: `12.80` ticks.
  * Rollout Oracle raw average waiting: `12.80` ticks (0.00% gap vs SRTF).
  * Neural Candidate Scorer raw waiting: `12.80` ticks (gap vs SRTF: `+0.00%`).
  * GBDT Ranker raw average waiting: `12.80` ticks (gap vs SRTF: `+0.00%`).
  * At $\text{tick}_{\mu\text{s}} = 1000\,\mu\text{s}$, effective turnaround for SRTF is `22.94` ticks,
    compared to `27.21` ticks for Neural and `113.16` ticks for GBDT.
- **Supporting Figures:** `results/figures/effective_cost_vs_tick.png` (Panel S0) and `results/figures/gap_to_oracle_bars.png`.

---

## Hypothesis 2 (H2: S1 Preemption Economy & Oracle Gap Capture): **CONFIRMED**

- **Pre-Registered Claim:** At Setting S1 ($c_s = 1$), excessive preemption imposes penalties.
  Rollout Oracle strictly outperforms SRTF by avoiding suboptimal preemptions, and learned models capture part of that gap.
- **Empirical Evidence:**
  * SRTF raw average waiting: `15.67` ticks.
  * Rollout Oracle raw average waiting: `15.42` ticks (improvement: `1.60%` over SRTF).
  * Neural Candidate Scorer raw waiting: `15.42` ticks (improvement: `+1.60%` over SRTF).
  * GBDT Ranker raw waiting: `15.47` ticks (improvement: `+1.28%` over SRTF).
- **Supporting Figures:** `results/figures/gap_to_oracle_bars.png` (Panel S1) and `results/figures/final_scenario_heatmap.png`.

---

## Hypothesis 3 (H3: Per-Decision Break-Even Tick Granularity): **CONFIRMED**

- **Pre-Registered Claim:** AI models lose to classical heuristics when $\text{{tick}}_{{\mu\text{{s}}}}$ is small,
  and algorithmic gains dominate when $\text{{tick}}_{{\mu\text{{s}}}}$ is large, with identifiable break-even points.
- **Empirical Evidence:**
- **Q-learning** at S1 does not break even within $[1, 10000]\,\mu\text{s}$ (cost remains higher than SJF).
- **PID_GBDT** at S1 does not break even within $[1, 10000]\,\mu\text{s}$ (cost remains higher than SJF).
- **PID_NN** at S1 breaks even at $\text{tick}^* = 7208.6\,\mu\text{s}$ against SJF.
- **Supporting Figures:** `results/figures/effective_cost_vs_tick.png` (marked with red intersection points).

---

## Hypothesis 4 (H4: Category 1 Amortization & Bounded Suboptimality): **REFUTED**

- **Pre-Registered Claim:** Category 1 models (DT, RF) have negligible overhead (< 0.1% makespan),
  remain invariant across tick durations, and closely track Best-Classical-Hindsight without beating it.
- **Empirical Evidence:**
  * Decision Tree overhead percentage of makespan: `0.00%` (< 0.1%).
  * Random Forest overhead percentage of makespan: `62.08%` (< 0.1%).
  * Best-Classical-Hindsight raw waiting: `16.88` ticks vs Random Forest: `32.70` ticks (gap: `+93.67%`).
- **Supporting Figures:** `results/figures/overhead_breakdown.png` and `results/figures/effective_cost_vs_tick.png`.
