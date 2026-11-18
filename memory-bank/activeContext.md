# Active Context

## Current Work Focus
- Stage 1 execution: Domain core acceptance test suite completed.
- All Stage 1 tasks (Task 1: `Process`, Task 2: `Timeline`, Task 3: `Metrics`, Task 4: `Engine`, Task 5: Domain Core Acceptance Test Suite) complete with 100% test coverage.
- Next up: Stage 2 (Classical Policies: FCFS, SJF, SRTF, RR, registry, and CLI simulation).

## Recent Changes
- Implemented `tests/test_domain.py` verifying all Stage 1 acceptance criteria and domain invariants:
  1. Idle gap at start: Initial idle jump [0, 5) and completion at tick 7.
  2. Simultaneous arrivals: Ordered admission at tick 2, canonical sorting, and waited tracking.
  3. Context switch overhead with `cs_cost = 2`: Hand-calculated fixture verification (makespan=6, CS=1, overhead=2, busy=4, P1 wait=0, P2 wait=3).
  4. Penalty invariant: Step penalty sums exactly equal process waiting times (`sum(penalties) == sum(waiting)`).
  5. Timeline partition invariant: Gapless, non-overlapping partition of `[0, makespan)`.
  6. Conservation invariant: Total busy ticks strictly equals sum of process bursts across all configurations.
  7. Engine independence: Engine neutrality verified under adversarial/custom policies with zero heuristic bias.
  8. Hand-verified Section 4.6 reference fixtures: FCFS, SJF, and SRTF numerical agreement.
  9. Domain metric algebraic invariants and end-to-end `run()` helper verification.
- All 92 tests passing with 100% test coverage (`src/schedsim`); `ruff check .` clean.

## Next Steps
- Transition to Stage 2: Classical Policies (`FCFS`, `SJF`, `SRTF`, `RR(q)`), policy registry, and CLI simulator.

## Active Decisions & Considerations
- Strict user workspace rule followed: **never commit or push in git**.
- Multi-point docstrings on all files, classes, and functions maintained.
- All domain invariants confirmed mechanically.
