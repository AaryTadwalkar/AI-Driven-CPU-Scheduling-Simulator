# Academic Report Notes & Observations

Course: Operating System (AI3002), Topic 1: AI-Driven CPU Scheduling Simulator

## Key Themes to Address in Final Report

1. **Theoretical Foundations:**
   - Optimality of SJF/SRTF under waiting time vs. starvation vulnerabilities.
   - Response time vs. context switch overhead trade-offs in Round Robin.
   - The scheduling decision as a Markov Decision Process (MDP).

2. **Workload Characterization:**
   - Why synthetic distributions require explicit scenario labeling (convoy, bimodal, high load).
   - Limitations of synthetic benchmarks vs. production OS kernel traces.

3. **Empirical Results & Regret:**
   - Quantitative regret comparisons across held-out splits.
   - Decision Tree interpretability and feature importance (e.g., CV of bursts, load ratio).
   - Q-learning convergence and policy robustness against corner cases.

4. **Reproducibility Guarantee:**
   - Explicit seed partitioning across train, validation, and test workloads.
   - Unit-step engine conservation proofs: busy execution ticks == sum of bursts.
