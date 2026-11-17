# Technical Context

## Core Technologies & Dependencies
- **Language:** Python 3.11.9
- **Numerical & Data Processing:**
  - `numpy`: Seeded random number generation, vectorized array operations.
  - `pandas`: Workload dataset management, metric aggregation, CSV serialization.
- **Machine Learning & Statistical Analysis:**
  - `scikit-learn`: DecisionTreeClassifier, feature scaling, cross-validation, confusion matrices.
  - `joblib`: Model serialization and persistence.
- **Visualization:**
  - `matplotlib`: Training convergence plots, regret histograms, comparative performance bar charts.
- **API & Data Validation:**
  - `pydantic` (v2): Strict domain schema definition and serialization.
  - `fastapi` & `uvicorn`: Asynchronous web server exposing simulation, recommendation, and evaluation endpoints.
- **Testing & Quality Assurance:**
  - `pytest` & `pytest-cov`: Unit testing, property-based invariants, and branch coverage.
  - `ruff`: High-performance linting and code formatting.
- **Frontend Dashboard (Stage 9):**
  - Node.js / Vite / React / TypeScript / Tailwind CSS / shadcn/ui / Recharts.

## Development Setup
- Virtual Environment: `.venv` located in project root.
- Python Command: `.\.venv\Scripts\python`
- Pytest Command: `.\.venv\Scripts\pytest -q`
- Ruff Command: `.\.venv\Scripts\ruff check .`

## Technical Constraints
- Single-core virtual CPU, discrete integer time steps.
- Tabular Q-learning only (no external deep learning / RL frameworks like Gym, Stable-Baselines3, PyTorch, or TensorFlow).
- Strictly zero Git commits or pushes executed by the agent (per user workspace rules).
