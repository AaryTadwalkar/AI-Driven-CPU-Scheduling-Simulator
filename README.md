# AI-Driven CPU Scheduling Simulator (AY 2026–27, Topic 1)

A reproducible, fully-tested, and scientifically validated CPU scheduling simulator developed
for Operating Systems (AI3002). The platform benchmarks classical operating system heuristics
against a strictly bounded 5-model AI roster across discrete hardware-latency accounting regimes,
accompanied by a high-throughput FastAPI backend and an interactive React dashboard.

---

## 1. Project Overview & Architectural Principles

Modern operating systems face diverse, heterogeneous workloads spanning interactive desktop
apps, batch processing, and high-frequency I/O bursts. Static heuristics such as FCFS, SJF,
SRTF, and Round Robin (RR) make rigid scheduling assumptions that fail to balance mean waiting
time, interactive response, starvation avoidance, and context switch penalties simultaneously.

This simulator models a single-core CPU executing discrete unit-step time ticks with:
- **Unit-Step Simulation Engine:** Discrete integer tick execution supporting exact preemption,
  idle-jump optimizations, and integer context-switch penalties ($c_s \in \{0, 1\}$).
- **Exact AI Latency Overhead Accounting:** One-time startup latency for workload-level meta-
  classifiers and per-decision discrete tick debt for process-level candidate rankers.
- **Pure NumPy Model Inference:** Zero heavy runtime dependencies (e.g. PyTorch/TensorFlow) during
  simulation loops, delivering microsecond inference latency.
- **Rigid Seed Partitioning:** Enforces strict isolation across training, validation, and held-out
  test splits with zero cross-contamination.

---

## 2. The Strict 5-Model AI Roster

In strict adherence to the project specification, the machine learning roster consists of exactly
five models categorized into three distinct operational tiers. Continuous deep RL (PPO), k-NN,
and exact Dynamic Programming were deliberately excluded due to sample inefficiency, unbounded
lookup overhead, and combinatorial state-space explosion:

1. **Model 1: Decision Tree (DT) Workload Classifier (Category 1)**
   - *Architecture:* Scikit-learn DecisionTreeClassifier with max depth $\le 6$.
   - *Operation:* Charged once at workload arrival ($\approx 299\,\mu\text{s}$). Classifies
     12 summary features into the optimal classical heuristic (FCFS, SJF, RR, SRTF).
2. **Model 2: Random Forest (RF) Workload Classifier (Category 1)**
   - *Architecture:* 100-tree ensemble classifier evaluating statistical variance reduction.
   - *Operation:* Charged once at workload arrival ($\approx 31,193\,\mu\text{s}$). Identifies
     whether ensemble accuracy gains justify substantial startup preemption latency.
3. **Model 3: Tabular Q-Learning Scheduler (Category 2)**
   - *Architecture:* Permutation-invariant 7-tuple aggregate state representation with Bellman
     TD-error updates across 4 high-level dispatch actions.
   - *Operation:* Charged per non-forced scheduling decision ($\approx 0.18\,\mu\text{s}$).
4. **Model 4: GBDT Candidate Ranker (Category 3)**
   - *Architecture:* Pairwise Gradient Boosted Decision Trees trained on Rollout Oracle regrets.
   - *Operation:* Charged per non-forced scheduling decision ($\approx 43\,\mu\text{s}$). Scores
     ready queue processes using 10 normalized candidate features.
5. **Model 5: Neural Candidate Scorer (Category 3)**
   - *Architecture:* 2-layer MLP (10 -> 32 -> 16 -> 1) trained via Imitation Learning (DAgger).
   - *Operation:* Charged per non-forced scheduling decision ($\approx 2.9\,\mu\text{s}$). Fast
     vectorized pure NumPy forward pass scoring all candidate processes.

---

## 3. Seed Partition Specification

All synthetic workloads and training datasets follow rigid seed separation:

| Split Name | Seed Range | Size (Workloads) | Purpose |
|---|---|---|---|
| **Train** | `0` to `99,999` | Up to 100,000 | Model training, Q-learning, DAgger rollouts |
| **Validation** | `100,000` to `100,499` | 500 | Hyperparameter tuning, early stopping |
| **Held-Out Test** | `200,000`–`200,999` | 1,000 | Final benchmark evaluation, hypotheses |

---

## 4. End-to-End Reproduction Sequence

To reproduce all datasets, trained checkpoints, latency benchmarks, and final comparison tables
from scratch, execute the following CLI commands in exact sequence:

```powershell
# 1. Synthesize baseline workload splits (Train, Val, Test across 9 scenarios)
python cli.py build-workloads --n-train 5000 --n-val 500 --n-test 1000

# 2. Build Category 1 supervised classification dataset
python cli.py build-dataset --n-workloads 5000

# 3. Train Category 1 Decision Tree recommender
python cli.py train-tree

# 4. Train Category 1 Random Forest ensemble
python cli.py train-tree --model rf

# 5. Train Category 2 Tabular Q-learning agents for Setting S0 and S1
python cli.py train-rl --episodes 20000 --setting both

# 6. Generate Rollout Oracle imitation dataset for process-level candidate ranking
python cli.py build-pid-data --n-workloads 5000

# 7. Train Model 4: GBDT Candidate Ranker for S0 and S1
python cli.py train-pid-gbdt --setting both

# 8. Train Model 5: Neural Candidate Scorer via DAgger for S0 and S1
python cli.py train-pid-nn --setting both --dagger-iters 2 --epochs 15

# 9. Benchmark empirical model inference latencies and construct cost profile
python cli.py bench-latency --reps 200

# 10. Execute Stage 18 Final Comparison across all 11 policies on held-out test split
python cli.py compare-all --n-workloads 1000 --tick-us 100.0
```

---

## 5. Directory Folder Map

```
scheduling_simulator_rl/
├── config.py                 # Central configurations, hyperparameters, and directory paths
├── cli.py                    # Unified CLI entry point supporting 12 subcommands
├── Makefile                  # Fast orchestration targets (smoke, test, lint, install)
├── pyproject.toml            # Python packaging and dependency specifications
├── src/schedsim/             # Core simulation package
│   ├── domain/               # Process dataclass, discrete unit-step Engine, Metrics, Timeline
│   ├── policies/             # Classical policies (FCFS, SJF, SRTF, RR), Selector, Registry
│   ├── workloads/            # 9 synthetic scenarios, generator, presets, split partitions
│   ├── ml/                   # Workload feature extraction, labeling, Decision Tree / RF
│   ├── rl/                   # Q-learning agent, 7-tuple state encoder, Gym-like env, policy
│   ├── pidsel/               # Rollout Oracle, GBDT ranker, Neural scorer, imitation data
│   ├── overhead/             # Latency profiler, discrete tick debt, break-even analyzer
│   ├── evaluation/           # Final comparison matrix, bootstrap CIs, publication figures
│   └── api/                  # FastAPI backend app, schemas, service layer, modular routes
├── tests/                    # 470+ unit, integration, and property test suites (100% pass)
├── data/                     # Generated CSV workloads, supervised datasets, oracle caches
├── models/                   # Serialized ML & RL checkpoints and latency_profile.json
├── results/                  # Comparison CSVs, markdown tables, and publication PNG figures
│   └── figures/              # 300 DPI publication plots (effective cost, heatmaps, Gantt)
├── screenshots/              # 17 high-resolution dashboard and evaluation figures
├── docs/                     # Detailed architectural specifications and REPORT_NOTES.md
└── frontend/                 # Interactive React + TypeScript + Vite + Tailwind dashboard
```

---

## 6. Execution Run Times & Hardware Specifications

Empirical execution benchmarks measured on standard reference hardware:
- **Processor:** AMD Ryzen 7 / Intel Core i7 (8 Cores, 16 Threads, 3.8 GHz base clock)
- **RAM:** 16 GB DDR4 / DDR5
- **Operating System:** Windows 11 (PowerShell 7 / Command Prompt)

| Pipeline Step | Scope / Configuration | Measured Wall Time |
|---|---|---|
| `python cli.py smoke` | End-to-end component validation | **0.38 seconds** |
| `pytest -q` | 476 comprehensive unit tests | **~6.5 minutes** |
| `train-tree` & `train-forest` | 5,000 workloads, 12 features | **~15 seconds** |
| `train-rl` | 20,000 episodes $\times$ 2 settings | **~2.5 minutes** |
| `train-pid-gbdt` | 30,000 state transitions $\times$ 2 settings | **~45 seconds** |
| `train-pid-nn` | 2 DAgger iterations, 15 epochs | **~1.8 minutes** |
| `bench-latency` | 200 repetitions per queue size | **~1.2 minutes** |
| `compare-all` | 1,000 test workloads $\times$ 11 policies $\times$ 2 | **~4.5 minutes** |

---

## 7. Operating the Project

### 7.1 Automated Makefile Targets
```powershell
make smoke    # Runs fast (< 1s) end-to-end verification across all components
make test     # Executes complete pytest test suite
make lint     # Runs ruff linter to assert zero style or formatting errors
```

### 7.2 CLI Commands
```powershell
# Display help and available subcommands
python cli.py --help

# Run single policy simulation with ASCII Gantt output
python cli.py simulate --policy SRTF --preset 1 --cs-cost 1

# Execute fast smoke test
python cli.py smoke

# Benchmark all 11 policies on held-out test split
python cli.py compare-all --n-workloads 1000 --tick-us 100.0
```

### 7.3 FastAPI Backend Server
Launch the high-performance REST API backend service on `http://127.0.0.1:8000`:
```powershell
uvicorn schedsim.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 7.4 React Interactive Dashboard
Launch the web interface providing Gantt scrubbing, multi-policy comparisons, and AI lab panels:
```powershell
cd frontend
npm install
npm run dev
```
Navigate to `http://localhost:5173` to interact with the simulator.
