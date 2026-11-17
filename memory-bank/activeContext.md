# Active Context

## Current Work Focus
- Stage 0 completion and verification.
- Preparing repo and environment for Stage 1 (Domain core implementation: `Process`, `Engine`, `Metrics`, `Timeline`).

## Recent Changes
- Initialized local git repository (`git init`).
- Created Python 3.11 virtual environment (`.venv`).
- Structured repository layout according to Section 3.2.
- Created `pyproject.toml` with pinned dependencies (`numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `pydantic`, `fastapi`, `uvicorn`, `pytest`, `pytest-cov`, `ruff`).
- Created `config.py` with standard constants, labeling weights, and seed splits.
- Created `cli.py` skeleton supporting all primary subcommands.
- Initialized documentation stubs (`docs/PROGRESS.md`, `docs/DECISIONS.md`, `docs/REPORT_NOTES.md`).
- Established the Memory Bank core files (`projectbrief.md`, `productContext.md`, `systemPatterns.md`, `techContext.md`, `progress.md`, `activeContext.md`).

## Next Steps
- Execute `pytest -q` and `python cli.py --help` to confirm Stage 0 acceptance criteria.
- Proceed to Stage 1: Implement `Process` validation, unit-step `Engine`, `Timeline` log aggregation, and `Metrics` calculation.

## Active Decisions & Considerations
- Remember user workspace rule: **never commit or push in git**.
- Strict docstring rule: Every Python file, class, and function must have detailed multi-point docstrings covering usage, inner workings, design concepts, usage sites, and inputs/outputs.
- Seed discipline: Always use `numpy.random.default_rng(seed)` and respect defined split ranges.
