# Active Context

## Current Work Focus
- Entire Implementation Plan (Stages 1 through 20) COMPLETE, VERIFIED & PASSING.
- Stage 19: Minimal API & Dashboard Extension — COMPLETE.
- Stage 20: Submission Pack (Makefile, Smoke CLI, README, Report Notes) — COMPLETE.
- Full test suite passing 100%, `ruff check .` clean (0 errors), 0 lines > 100 chars.

## Recent Changes (Stage 19 & Stage 20 Implementation)
- `src/schedsim/policies/registry.py`:
  - Added dynamic resolution for `qlearning`, `rl`, `dt`, `rf`, `pid_gbdt`, `pid_nn` policies.
  - Dynamically passes current `config.MODELS_DIR` ensuring test isolation and fault injection.
- `src/schedsim/api/schemas.py`:
  - Added `tick_us: float | None` to `SimulateRequest` and `CompareRequest` with boundary validations.
  - Added `ai_overhead_ticks`, `decisions`, `inference_us_total`, `effective_makespan`, and
    `effective_turnaround` to `SimulationMetricsSchema`.
  - Added `FinalComparisonRowSchema` and `FinalComparisonResponse` for `GET /api/results/final`.
- `src/schedsim/api/service.py`:
  - `run_simulation_service`: Accepts `tick_us`, resolves algorithms case-insensitively, binds
    workloads, charges upfront `on_start()` costs, charges per-decision AI latency, logs ready queue
    and RL decision traces, handles missing models with prescriptive CLI training commands, and
    extracts all effective metrics.
  - `run_comparison_service`: Benchmarks all 11 candidate policies with latency accounting.
  - `get_final_results_service`: Parses and serves Setting S0 and S1 final comparison CSV matrices.
- `src/schedsim/api/routes/`:
  - Created `results.py` with `GET /api/results/final` and `GET /api/results/summary`.
  - Created `simulation.py` with `POST /api/simulate` and `POST /api/compare`.
  - Mounted routers in `app.py`.
- Frontend:
  - `frontend/src/lib/colors.ts`: Added `SPECIAL_SEGMENT_COLORS.AI` and new algorithm branding.
  - `frontend/src/index.css`: Added `.hatch-ai` diagonal CSS pattern.
  - `frontend/src/components/sim/GanttChart.tsx` & `frontend/src/components/compare/GanttStack.tsx`:
    Added `.hatch-ai` styling for `AI` and `AI_OVERHEAD` timeline segments.
  - `frontend/src/pages/ComparePage.tsx`: Added all models to `ALGORITHM_OPTIONS`.
- Screenshots:
  - Created `screenshots/15_comparison_matrix.png`, `screenshots/16_cost_vs_tick.png`, and
    `screenshots/17_gantt_with_ai_stalls.png`.
- Stage 20 Deliverables:
  - Created root `Makefile` (`smoke`, `test`, `lint`, `install`).
  - Added `smoke` CLI subcommand executing fast (< 15s, runs in ~0.15s) end-to-end verification.
  - Updated `README.md` with complete 2026-27 project documentation and reproduction sequence.
  - Updated `docs/REPORT_NOTES.md` with comprehensive 9-section report material.
- Tests:
  - Created `tests/test_stage19_api.py` with 7 comprehensive unit, integration, and 503 tests.

## Active Decisions & Considerations
- Strict user workspace rule followed: **never commit or push in git**.
- Bounded strictly to the 5 models in the roster. No extra models trained.
- Pure NumPy used for all simulation inference.
- Exact line length <= 100 characters enforced across all files.
