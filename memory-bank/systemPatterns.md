# System Patterns & Architecture

## Core Architectural Patterns

### 1. Unified Unit-Step Simulation Loop
- Every policy (FCFS, SJF, SRTF, RR, RL) is evaluated using the exact same step-by-step engine contract:
  `View -> policy.select(view) -> Engine.step(pid) -> StepInfo`
- **Time Model:** Discrete integer tick intervals `[t, t+1)`.
- **Idle Jump:** When no process is ready, time advances directly to the next arrival, appending an `IDLE` segment.
- **Context Switch Semantics:**
  - A dispatch occurs on `step(pid)`. If `last_pid is not None` and `pid != last_pid` and `cs_cost > 0`, `cs_cost` overhead ticks (`CS`) are recorded before the 1 execution tick.
  - The first dispatch of a simulation run is free. Idle gaps do not reset `last_pid`. Switching away from a completed process counts as a context switch.
  - Overhead ticks count as waiting time for all ready processes (including the one about to run).

### 2. Permutation-Invariant State Abstraction for RL
- Per-process ID representations are avoided.
- Aggregate feature state vector:
  - Ready queue length bucket.
  - Current process status / remaining burst bucket.
  - Relative burst dispersion (e.g., ratio of shortest remaining to longest remaining).
  - Time elapsed since last context switch.
- Total discrete state space bounded to ~1,000–2,000 states.
- **Action Space:** Meta-rules (e.g., Shortest Remaining, Longest Remaining, Earliest Arrival, Continue Current).

### 3. Supervised Workload Classification
- **Feature Extraction:** Purely structural features derived before simulation (number of jobs, burst mean, burst CV, max/median burst ratio, arrival span, load ratio, simultaneous arrival fraction).
- **Composite Labeling:** Multi-objective normalized cost combining waiting time (50%), response time (30%), turnaround time (20%), and context switch cost.

### 4. Strict Seed Isolation
- **Train Split:** 0 – 99,999 (workloads for dataset generation and RL training episodes).
- **Validation Split:** 100,000 – 100,499 (hyperparameter tuning and validation curves).
- **Test Split:** 200,000 – 200,999 (final untouched evaluation benchmark).

## Code Style & Documentation Standards
- **File Docstrings:** 1. File purpose, 2. Overall design, 3. Elements in file, 4. Final output.
- **Class Docstrings:** 1. Class use, 2. Content, 3. Technologies used & why, 4. Key & overall design.
- **Function Docstrings:** 1. Function use, 2. How it works, 3. Concepts used, 4. Places imported & used, 5. Input/output state before and after.
- **Invariants:** Dataclasses for domain records, type hints everywhere, immutable where appropriate, zero unseeded randomness (`numpy.random.Generator` required).
