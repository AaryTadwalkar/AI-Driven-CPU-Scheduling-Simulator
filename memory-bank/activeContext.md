# Active Context

## Current Work Focus
- Stages 19 & 20 (API & Dashboard Extension + Submission Pack) actively being implemented by subagent.
- Stage 18: Final Comparison (The Deliverable) — COMPLETE, APPROVED & VERIFIED.
- All 11 policies evaluated across Setting S0 ($c_s = 0$) and Setting S1 ($c_s = 1$) on 1,000 test split workloads:
  * Classical: FCFS, SJF, RR (q=2), SRTF
  * AI Category 1 (charged once at start): DT, RF
  * AI Category 2 (charged per decision): Q-learning
  * AI Category 3 (charged per decision): PID_GBDT, PID_NN
  * References: Best-Classical-Hindsight, Rollout Oracle
- Full test suite passing: 469 passed (100%), `ruff check .` clean (0 errors), 0 lines > 100 chars.

## Recent Changes (Stage 18 Implementation)
- `src/schedsim/evaluation/final_comparison.py`:
  - Evaluated run matrix across S0 ($c_s = 0$) and S1 ($c_s = 1$) on 1,000 test split workloads (seeds 200,000–200,999).
  - Pre-computed and cached Rollout Oracle on test split in `data/oracle_test_S0.pkl` and `data/oracle_test_S1.pkl`.
  - Computed 15-column comparison matrices with 95% bootstrap CIs (1,000 resamples).
  - Saved outputs: `results/final_comparison_S0.csv`, `results/final_comparison_S0.md`, `results/final_comparison_S1.csv`, `results/final_comparison_S1.md`.
  - Computed 1D linear interpolation break-even tick frequencies across sweep $[1, 10, 100, 1000, 10000]\,\mu\text{s}$.
  - Rendered 5 publication-quality figures at 300 DPI in `results/figures/`:
    1. `effective_cost_vs_tick.png`
    2. `gap_to_oracle_bars.png`
    3. `overhead_breakdown.png`
    4. `final_scenario_heatmap.png`
    5. `gantt_with_ai_stalls.png`
  - Evaluated pre-registered hypotheses H1–H4 in `docs/HYPOTHESES_RESULTS.md` with formal verdicts:
    * H1: CONFIRMED (SRTF optimal at S0; learned models degrade with overhead).
    * H2: CONFIRMED (Oracle beats SRTF at S1: 17.76 vs 17.90; Neural 17.77 and GBDT 17.78 capture gap).
    * H3: CONFIRMED (Identified break-even tick thresholds for Q-learning, Neural, and GBDT).
    * H4: REFUTED (Random Forest 100-tree ensemble latency 31,193 µs imposes 38.06% makespan overhead, refuting the < 0.1% hypothesis).
- `cli.py`:
  - Added `compare-all` subcommand with flags `--n-workloads`, `--tick-us`, `--skip-plots`, `--output-dir`.
- Tests:
  - Created `tests/test_stage18_comparison.py` with 11 acceptance tests covering all 7 criteria (100% pass in 22.8s).
- Full Test Suite:
  - 469 passed in 363.2s, 0 failures.

## Active Decisions & Considerations
- Strict user workspace rule followed: **never commit or push in git**.
- Bounded strictly to the 5 models in the roster. No extra models trained.
- Pure NumPy used for all simulation inference.
- Exact line length <= 100 characters enforced across all files.
