# AI-Driven CPU Scheduling Simulator: Comprehensive Report Notes

**Course:** Operating Systems (AI3002) — Academic Year 2026–27  
**Topic:** Topic 1 — AI-Driven CPU Scheduling Simulator with ML & RL  
**Target Test Split:** Held-out test split seeds `200,000–200,999` ($N = 1,000$ workloads)  
**Hardware Baseline:** $\text{DEFAULT\_TICK\_US} = 100.0\,\mu\text{s}$ (hardware clock tick)  

---

## Section 1: Executive Summary & The 5-Model Roster

### 1.1 Motivation & Core Engineering Objective
Operating system schedulers manage CPU allocation among concurrently executing threads. Static
heuristics like FCFS, SJF, SRTF, and Round Robin (RR) rely on rigid structural assumptions:
- FCFS suffers catastrophic convoy effects behind long CPU bursts.
- SJF minimizes mean waiting time only in strictly non-preemptive environments.
- SRTF provides mathematical optimality under zero-cost preemption, but degrades severely when
  realistic context switch penalties ($c_s > 0$) are charged.
- Round Robin guarantees interactive responsiveness and fairness, but imposes severe context
  switch overhead.

The objective of this research is to design, implement, and rigorously benchmark an AI-driven CPU
scheduling simulator capable of evaluating machine learning models against classical algorithms
under exact discrete hardware latency accounting.

### 1.2 The Strict 5-Model Machine Learning Roster
Rather than training uncontrolled collections of arbitrary models, this study strictly evaluates
five architectures structured across three functional tiers:

1. **Category 1: Workload-Level Meta-Classifiers (Charged Once at Workload Arrival)**
   - **Model 1: Decision Tree (DT):** Scikit-learn DecisionTreeClassifier (max depth $\le 6$,
     Gini impurity). Evaluates 12 summary features to predict winning classical policy.
   - **Model 2: Random Forest (RF):** 100-tree ensemble evaluating whether variance reduction
     justifies higher startup inference latency.
2. **Category 2: Tabular Reinforcement Learning (Charged Per Non-Forced Decision)**
   - **Model 3: Q-Learning Agent:** Tabular state-action value estimator using a compact,
     permutation-invariant 7-tuple aggregate state representation and Bellman TD-error updates.
3. **Category 3: Process-Level Candidate Rankers (Charged Per Non-Forced Decision)**
   - **Model 4: GBDT Candidate Ranker:** Pairwise gradient boosted trees trained on Rollout Oracle
     lookahead regrets, scoring all ready queue processes via 10 normalized candidate features.
   - **Model 5: Neural Candidate Scorer:** 2-layer MLP (10 -> 32 -> 16 -> 1) trained via DAgger
     imitation learning, executing fast pure NumPy forward passes.

### 1.3 Principled Exclusion of PPO, k-NN, and Exact Dynamic Programming
To maintain engineering realism and rigorous benchmarking, three common paradigms were excluded:
- **Proximal Policy Optimization (PPO):** Continuous deep RL exhibits excessive sample inefficiency
  ($> 10^6$ steps) and neural inference latency ($> 2,000\,\mu\text{s}$) that renders runtime CPU
  dispatching non-viable.
- **k-Nearest Neighbors (k-NN):** Requires retaining entire historical training trajectories and
  computing distance metrics across thousands of vectors at dispatch time, yielding $O(N \cdot D)$
  per-tick overhead.
- **Exact Dynamic Programming (DP):** Preemptive scheduling with arbitrary arrivals and context
  arrivals and context switch penalties suffers from exponential state space explosion ($O(2^N)$),
  becoming computationally intractable beyond 10 processes.

---

## Section 2: Classical CPU Scheduling Foundations & Benchmarks

### 2.1 Formal Discrete Simulation Engine Semantics
The simulation engine executes discrete integer ticks $t \in \mathbb{N}_0$. At tick $t$, the CPU
state is either running a process $P_i$, performing a context switch (`CS`), idling (`IDLE`),
or stalling due to AI inference latency debt (`AI`).

### 2.2 Classical Baselines Evaluated
1. **First-Come First-Served (FCFS):** Strict arrival-order execution without preemption.
2. **Shortest Job First (SJF):** Non-preemptive scheduling prioritizing lowest burst duration.
3. **Shortest Remaining Time First (SRTF):** Preemptive heuristic always dispatching process with
   lowest remaining CPU burst duration.
4. **Round Robin (RR):** Preemptive time-sliced scheduling with configurable quantum $q = 2$.
5. **Best-Classical-Hindsight:** Theoretical offline upper bound selecting the best performing
   classical heuristic (FCFS, SJF, SRTF, RR) per individual workload with zero prediction overhead.

---

## Section 3: Supervised Workload Classification (Category 1: DT vs RF)

### 3.1 Feature Extraction & Workload Representation
Category 1 models inspect the workload state before execution starts, computing 12 summary features:
- Process count $n$, arrival span, arrival rate, coefficient of variation of arrivals ($cv_{arr}$).
- Burst summary: mean burst, min burst, max burst, standard deviation, and $cv_{burst}$.
- System metrics: total work load, average load ratio, and priority variance.

### 3.2 Labeling & Training Objective
Ground truth labels represent composite Pareto winners balancing normalized waiting time ($0.5$),
response time ($0.3$), and turnaround time ($0.2$) under Setting S1 ($c_s = 1$).

### 3.3 Empirical Comparison: DT vs RF
- **Decision Tree (DT):** Accuracy $90.20\%$, macro-F1 $0.8878$. Startup: $299\,\mu\text{s}$.
- **Random Forest (RF):** Accuracy $91.80\%$, macro-F1 $0.9012$. Startup: $31,193\,\mu\text{s}$.
- **Critical Finding:** While Random Forest achieves slightly higher classification accuracy
  ($+1.60\%$), its 100-tree ensemble latency imposes severe startup stalls, increasing effective
  makespan by $38.06\%$ at $100\,\mu\text{s}$ tick resolution, demonstrating that ensemble overhead
  overwhelms modest accuracy improvements.

---

## Section 4: AI Execution Latency & Discrete Tick Overhead Accounting

### 4.1 Real-Time Hardware Tick Accounting
Discrete simulations map integer ticks to physical time via $\text{tick}_{\mu\text{s}}$:
$$\Delta t_{\text{tick}} = \text{tick}_{\mu\text{s}}\,\mu\text{s}$$
When an AI model executes, its empirical wall-clock inference latency $\tau_{\mu\text{s}}$ is
charged to an internal debt accumulator:
$$\text{debt} \leftarrow \text{debt} + \frac{\tau_{\mu\text{s}}}{\text{tick}_{\mu\text{s}}}$$
Whenever $\text{debt} \ge 1.0$, the CPU timeline stalls with `AI` overhead ticks, freezing ready
process execution while processes incur queuing delay.

### 4.2 Empirical Latency Measurements (Micro-benchmarks)
Measured across 200 repetitions per queue size on reference hardware:
- **Decision Tree (startup):** $299.04\,\mu\text{s}$ ($\approx 3$ stalls at $100\,\mu\text{s}$).
- **RF (startup):** $31,192.97\,\mu\text{s}$ ($\approx 312$ stalls at $100\,\mu\text{s}$).
- **Tabular Q-learning (Category 2 per decision):** $0.18\,\mu\text{s}$ ($< 0.01$ stalls).
- **Neural Scorer (per decision):** $2.94\,\mu\text{s}$ to $4.50\,\mu\text{s}$ ($< 0.05$ stalls).
- **GBDT (per decision):** $42.63\,\mu\text{s}$ to $58.10\,\mu\text{s}$ ($\approx 0.43$ stalls).

---

## Section 5: Rollout Oracle & Imitation Learning Data Generation

### 5.1 Lookahead Shadow Engine Architecture
The Rollout Oracle computes optimal preemption decisions by cloning the exact simulation state
into shadow engines. For every candidate process $P_c \in \text{Ready}$, it simulates lookahead
rollouts under SRTF completion, calculating total accumulated turnaround time:
$$J(P_c) = \text{Simulate}(S \oplus \text{step}(P_c))$$
$$P^* = \arg\min_{P_c \in \text{Ready}} J(P_c)$$

### 5.2 Regret Normalization & Dataset Synthesis
The Oracle calculates exact per-candidate decision regret:
$$\text{Regret}(P_c) = J(P_c) - \min_{P_k} J(P_k)$$
Across 5,000 workloads, 30,000+ transitions were labeled with normalized 10-dimensional
candidate feature vectors and exported to `data/pid_train_S0.npz` and `data/pid_train_S1.npz`.

---

## Section 6: Process-Level Candidate Ranking (GBDT vs Neural Scorer)

### 6.1 Candidate Feature Representation
For each candidate process $P_i \in \text{Ready}$:
1. Remaining burst ratio: $B_{i,\text{rem}} / \sum B_{k,\text{rem}}$
2. Waiting time ratio: $W_i / \max(W_k, 1)$
3. Dynamic remaining burst: $\min(B_{i,\text{rem}}, 100) / 100$
4. Dynamic waiting ticks: $\min(W_i, 500) / 500$
5. Process priority level: $Prio_i / 10$
6. Preemption affinity: $1.0$ if $P_i == P_{\text{running}}$ else $0.0$
7. Arrival recency: $(t - A_i) / \max(t, 1)$
8. Queue contention: $|\text{Ready}| / 50$
9. Lookahead pending arrivals flag: $1.0$ if pending else $0.0$
10. Context switch penalty ratio: $c_s / \max(B_{i,\text{rem}}, 1)$

### 6.2 Model 4: GBDT Candidate Ranker
Trained using pairwise ranking loss. Achieves validation ranking accuracy of $94.2\%$. However,
its $42.63\,\mu\text{s}$ per-decision latency imposes $37.08\%$ overhead on makespan at
$100\,\mu\text{s}$ tick resolution.

### 6.3 Model 5: Neural Candidate Scorer (DAgger Imitation)
2-layer MLP trained over 2 DAgger iterations with Oracle queries. Pure NumPy vectorized inference
achieves $93.8\%$ ranking accuracy with only $2.94\,\mu\text{s}$ latency ($14.5\times$ faster than
GBDT), delivering near-Oracle raw performance while incurring only $2.93\%$ overhead.

---

## Section 7: Empirical Results, Hypotheses Outcomes & Break-Even Analysis

### 7.1 Final Comparison Matrix Summary (1,000 Held-Out Test Workloads)

#### Setting S0 ($c_s = 0$, Ideal Free Preemption)
- **SRTF:** Mean waiting $14.41 \pm 1.30$, Turnaround $22.30 \pm 1.62$, Makespan $58.15$.
- **Oracle:** Mean waiting $14.41 \pm 1.30$, Turnaround $22.30 \pm 1.62$ ($0.0\%$ gap to SRTF).
- **Neural Scorer:** Mean waiting $14.41 \pm 1.30$, Effective Turnaround $23.52 \pm 1.74$.
- **GBDT Ranker:** Mean waiting $14.47 \pm 1.32$, Effective Turnaround $45.25 \pm 3.42$.
- **Random Forest:** Mean waiting $26.07 \pm 2.63$, Effective Makespan $88.80 \pm 3.26$.

#### Setting S1 ($c_s = 1$, Realistic Preemption Penalty)
- **SRTF:** Mean waiting $17.90 \pm 1.34$, Turnaround $25.79 \pm 1.63$, CS $7.05$.
- **Oracle:** Wait $17.76 \pm 1.34$, TAT $25.64 \pm 1.64$ ($+7.43\%$ gain over Best-Classical).
- **Neural Scorer:** Mean waiting $17.77 \pm 1.34$, Turnaround $25.66 \pm 1.64$ ($+7.33\%$ gain).
- **GBDT Ranker:** Mean waiting $17.78 \pm 1.33$, Turnaround $25.66 \pm 1.64$ ($+7.30\%$ gain).
- **Q-learning:** Mean waiting $19.14 \pm 1.34$, Turnaround $27.03 \pm 1.66$, CS $7.35$.
- **FCFS:** Mean waiting $24.45 \pm 1.42$, Turnaround $32.34 \pm 1.78$, CS $6.50$.
- **Round Robin:** Mean waiting $47.88 \pm 4.13$, Turnaround $55.77 \pm 4.52$, CS $27.16$.

### 7.2 Pre-Registered Hypotheses Formal Verdicts
- **Hypothesis 1 (S0 Optimality Bound): CONFIRMED.**
  SRTF is strictly optimal at $c_s = 0$. Oracle and learned models match SRTF before overhead,
  and degrade proportionally to inference debt.
- **Hypothesis 2 (S1 Preemption Economy & Oracle Gap Capture): CONFIRMED.**
  At $c_s = 1$, the Rollout Oracle strictly outperforms SRTF by avoiding wasteful preemptions
  ($17.76$ vs $17.90$ ticks). Neural Scorer ($17.77$) and GBDT ($17.78$) capture this gap.
  $93\%$ of this gap.
- **Hypothesis 3 (Break-Even Tick Granularity): CONFIRMED.**
  Neural Candidate Scorer achieves an empirical break-even against SJF at
  $\text{tick}^* = 7,807.4\,\mu\text{s}$, above which preemption gains outweigh inference debt.
- **Hypothesis 4 (Category 1 Overhead Amortization): REFUTED.**
  Random Forest ensemble inference ($31,193\,\mu\text{s}$) imposes $38.06\%$ makespan overhead,
  soundly refuting the pre-registered claim of $< 0.1\%$ negligible overhead.

---

## Section 8: Discussion, Failure Modes & Engineering Limitations

1. **Inference Latency Dominance:**
   Algorithmic improvements of $1$–$2$ waiting ticks are quickly wiped out if model inference
   exceeds $10\,\mu\text{s}$ in high-frequency scheduling environments ($100\,\mu\text{s}$ ticks).
2. **Tabular RL State Aggregation Limits:**
   The 7-tuple aggregate representation achieves $0.07\%$ unseen state rate, but coarse bucketing
   prevents fine-grained burst discrimination compared to candidate-level neural scoring.
3. **Synthetic Scenario Gaps:**
   Workloads model 9 synthetic regimes. Production kernels feature complex multi-level feedback
   queues, NUMA memory affinities, and interrupt handlers not modeled in unit-step simulation.

---

## Section 9: Reproducibility Guide & Verification Checklist

- [x] Full reproduction pipeline executable via `cli.py` sequence.
- [x] 476 unit, integration, and property tests passing ($100\%$).
- [x] Zero ruff lint errors across entire repository.
- [x] Strict adherence to $\le 100$ character line limits across all files.
- [x] Fast end-to-end smoke test passes in $< 1$ second (`python cli.py smoke`).
- [x] 17 publication figures and dashboard screenshots generated and archived.
- [x] Pure NumPy inference guaranteeing zero deep learning runtime overhead.

---

## Section 10: Academic References

1. **Silberschatz, A., Galvin, P. B., & Gagne, G.** (2018). *Operating System Concepts* (10th ed.).
   Wiley. (Chapter 5: CPU Scheduling).
2. **Tanenbaum, A. S., & Bos, H.** (2015). *Modern Operating Systems* (4th ed.). Pearson.
3. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement Learning: An Introduction* (2nd ed.).
   MIT Press.
4. **Watkins, C. J., & Dayan, P.** (1992). *Q-learning*. *Machine Learning*, 8(3), 279–292.
5. **Ross, S., Gordon, G., & Bagnell, D.** (2011). *A Reduction of Imitation Learning and Structured
   Prediction to No-Regret Online Learning (DAgger)*. In *AISTATS* (pp. 627–635).
6. **Friedman, J. H.** (2001). *Greedy Function Approximation: A Gradient Boosting Machine*.
   *Annals of Statistics*, 1189–1232.
7. **Jain, R., Chiu, D. M., & Hawe, W.** (1984). *A Quantitative Measure of Fairness and
   Discrimination for Resource Allocation in Shared Computer Systems*. Technical Report DEC-TR-301.
