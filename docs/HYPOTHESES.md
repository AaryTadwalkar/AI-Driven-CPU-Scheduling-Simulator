# Pre-Registered Hypotheses for Stage 18 Final Comparison

**Document Creation Timestamp:** 2026-11-19T02:59:00Z  
**Evaluation Target Split:** Test partition seeds `200,000–200,999` ($N = 1,000$ workloads, sizes $n \in [3, 10]$, all 9 scenarios).  
**Benchmark Settings:** Setting S0 ($c_s = 0$) and Setting S1 ($c_s = 1$).  
**Microsecond Tick Sweep:** $\text{tick}_{\mu\text{s}} \in [1, 10, 100, 1000, 10000]$ (Baseline: $\text{DEFAULT\_TICK\_US} = 1000$).

---

## Pre-Registered Hypotheses

### Hypothesis 1 (H1: S0 Optimality Bound & Overhead Degradation)
- **Statement:** At Setting S0 ($c_s = 0$), no policy beats SRTF in raw waiting time ($J$). The learned process-level models (GBDT Candidate Ranker and Neural Candidate Scorer) and the Rollout Oracle will perform within statistical noise of SRTF before accounting for AI inference overhead:
  $$\frac{J_{\text{raw}} - J_{\text{SRTF}}}{J_{\text{SRTF}}} \approx 0.0$$
  Once hardware execution overhead is charged ($\text{tick}_{\mu\text{s}} \le 1000\,\mu\text{s}$), all per-decision AI schedulers (Q-learning, GBDT, Neural) will produce strictly higher effective waiting time and effective makespan than SRTF.
- **Verification Metric:** Paired percentage gap against SRTF before overhead, and effective turnaround / makespan after overhead at default $\text{tick}_{\mu\text{s}} = 1000\,\mu\text{s}$.

### Hypothesis 2 (H2: S1 Preemption Economy & Oracle Gap Capture)
- **Statement:** At Setting S1 ($c_s = 1$), excessive preemption imposes severe context switch stalls ($c_s = 1$ tick). The Rollout Oracle will strictly outperform SRTF on average waiting time by avoiding suboptimal preemptions:
  $$J_{\text{Oracle}} < J_{\text{SRTF}}$$
  The Neural Candidate Scorer (and GBDT Ranker) will capture a measurable portion of this gap over SRTF before AI stall penalties are applied.
- **Verification Metric:** Percentage waiting time reduction of Oracle vs SRTF, and of Neural/GBDT vs SRTF at S1.

### Hypothesis 3 (H3: Per-Decision Break-Even Tick Granularity)
- **Statement:** For per-decision AI models (Tabular Q-learning, GBDT Ranker, Neural Candidate Scorer), their competitiveness against the best classical heuristic is governed by CPU clock / tick scale ($\text{tick}_{\mu\text{s}}$).
  - When $\text{tick}_{\mu\text{s}}$ is large ($\ge 10,000\,\mu\text{s}$), AI overhead accounts for a negligible fraction of tick duration ($< 1\%$), allowing algorithmic gains to dominate.
  - When $\text{tick}_{\mu\text{s}}$ is small ($\le 10\,\mu\text{s}$), accumulated AI stall ticks degrade effective waiting time, causing AI models to lose to classical heuristics.
  - There exists a finite, identifiable **break-even tick length** $\text{tick}^*$ for each per-decision AI model where its effective scheduling performance equals the best classical policy.
- **Verification Metric:** Linear interpolation of effective turnaround time vs $\text{tick}_{\mu\text{s}}$ curves across the sweep $[1, 10, 100, 1000, 10000]$.

### Hypothesis 4 (H4: Category 1 Amortization & Bounded Suboptimality)
- **Statement:** Category 1 models (Decision Tree and Random Forest) execute inference exactly once per workload at startup ($t=0$). Their overhead is completely amortized across the entire workload lifecycle, contributing $< 0.1\%$ to total makespan. Consequently:
  - Their effective waiting time and makespan will remain nearly invariant across the entire $\text{tick}_{\mu\text{s}}$ sweep.
  - Their performance will closely track the theoretical "Best-of-Classical in Hindsight" oracle, but will never strictly exceed it since Category 1 models can only select one classical policy per workload.
- **Verification Metric:** Startup overhead as percentage of makespan, stability of effective waiting time across $\text{tick}_{\mu\text{s}} \in [1, 10000]$, and paired gap vs Best-of-Classical in Hindsight.
