"""
1. What this file does:
   Defines the FastAPI application instance, lifespan lifecycle manager, CORS middleware,
   global exception handlers, and all REST API endpoints under the /api route prefix.
2. Overall design:
   Provides a high-performance, stateless HTTP service layer exposing the CPU scheduling
   simulator, preset workloads, on-demand generation, multi-algorithm comparisons, Decision Tree
   recommendations, and benchmark analytics to the React dashboard. Pre-loads AI models at startup,
   and surfaces descriptive HTTP 503 responses when optional model artifacts are missing.
3. Elements in the file:
   - lifespan: Async context manager pre-loading Q-learning agents and checking model files.
   - create_app: Application factory configuring FastAPI, CORS, routes, and exception handlers.
   - Endpoints: /api/health, /api/presets, /api/workloads/generate, /api/simulate, /api/compare,
     /api/recommend, /api/ml/model-info, /api/rl/training-log, /api/rl/policy-summary,
     /api/results/summary.
4. Final output of the file:
   Exports app and create_app for ASGI servers (Uvicorn) and integration test clients.
"""

import json
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from schedsim.api.schemas import (
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    GenerateWorkloadRequest,
    GenerateWorkloadResponse,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    SimulateRequest,
    SimulateResponse,
)
from schedsim.api.service import (
    ModelMissingException,
    generate_workload_service,
    processes_from_schemas,
    run_comparison_service,
    run_recommend_service,
    run_simulation_service,
)
from schedsim.rl.agent import QLearningAgent
from schedsim.workloads.presets import get_preset, list_presets


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    1. The use of function:
       Lifespan context manager loading machine learning and RL models at application startup.
    2. How does the specific function work:
       Checks paths for decision_tree.joblib, q_table_cs0.pkl, and q_table_cs1.pkl. Loads
       Q-learning agent instances into app.state for high-throughput in-memory inference.
    3. Concepts used in the function:
       FastAPI lifespan events, in-memory model caching, startup resource initialization.
    4. Places where this function is imported and used:
       Passed to FastAPI(lifespan=lifespan) during application creation.
    5. Final output of the function, before and after:
       Before: Uninitialized app.state.
       After: app.state populated with cached agents or None references.
    """
    # Decision tree path
    app.state.dt_path = config.MODELS_DIR / "decision_tree.joblib"

    # Tabular Q-learning agents
    p_cs0 = config.MODELS_DIR / "q_table_cs0.pkl"
    p_cs1 = config.MODELS_DIR / "q_table_cs1.pkl"

    app.state.q_agent_cs0 = QLearningAgent.load(p_cs0) if p_cs0.exists() else None
    app.state.q_agent_cs1 = QLearningAgent.load(p_cs1) if p_cs1.exists() else None

    yield

    app.state.q_agent_cs0 = None
    app.state.q_agent_cs1 = None


def create_app() -> FastAPI:
    """
    1. The use of function:
       Application factory assembling the complete FastAPI service with routes and handlers.
    2. How does the specific function work:
       Creates FastAPI app, configures CORS middleware for localhost:5173, registers global
       exception handlers for ModelMissingException, ValueError, and RequestValidationError,
       and attaches all /api endpoints.
    3. Concepts used in the function:
       Application factory pattern, middleware configuration, declarative routing.
    4. Places where this function is imported and used:
       Imported by main.py, test suite fixtures, and ASGI runner configurations.
    5. Final output of the function, before and after:
       Before: No application configured.
       After: Fully configured FastAPI application instance.
    """
    api = FastAPI(
        title="AI-Driven CPU Scheduling Simulator API",
        version="1.0.0",
        description="REST backend service for classical, ML, and RL scheduling simulations.",
        lifespan=lifespan,
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.exception_handler(ModelMissingException)
    async def model_missing_handler(_: Request, exc: ModelMissingException) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "Model Not Found", "detail": exc.detail},
        )

    @api.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Bad Request", "detail": str(exc)},
        )

    @api.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": "Validation Error", "detail": str(exc)},
        )

    # --------------------------------------------------------------------------
    # API Routes
    # --------------------------------------------------------------------------

    @api.get(
        "/api/health",
        response_model=HealthResponse,
        summary="Service Health and Model Status",
    )
    async def get_health() -> HealthResponse:
        """Returns API status, version, and availability flags for model checkpoints."""
        models_status = {
            "decision_tree": (config.MODELS_DIR / "decision_tree.joblib").exists(),
            "q_table_cs0": (config.MODELS_DIR / "q_table_cs0.pkl").exists(),
            "q_table_cs1": (config.MODELS_DIR / "q_table_cs1.pkl").exists(),
            "q_table": (config.MODELS_DIR / "q_table.pkl").exists(),
        }
        artifacts_status = {
            "evaluation_results": (config.RESULTS_DIR / "evaluation_results.csv").exists(),
            "summary_table": (config.RESULTS_DIR / "summary_table.csv").exists(),
            "rl_training_logs": (config.DATA_DIR / "rl_training_logs.csv").exists(),
        }
        return HealthResponse(
            status="ok",
            version="1.0.0",
            models=models_status,
            artifacts=artifacts_status,
        )

    @api.get(
        "/api/presets",
        summary="List Hand-crafted Preset Workloads",
    )
    async def get_presets() -> dict[str, Any]:
        """Lists preset workloads with descriptions and process schedules."""
        descriptions = list_presets()
        presets_data: dict[str, Any] = {}
        for name, desc in descriptions.items():
            procs = get_preset(name)
            presets_data[name] = {
                "name": name,
                "description": desc,
                "process_count": len(procs),
                "processes": [
                    {
                        "pid": p.pid,
                        "arrival": p.arrival,
                        "burst": p.burst,
                        "priority": p.priority,
                    }
                    for p in procs
                ],
            }
        return presets_data

    @api.post(
        "/api/workloads/generate",
        response_model=GenerateWorkloadResponse,
        responses={400: {"model": ErrorResponse}},
        summary="Generate Synthetic Workload",
    )
    async def post_generate_workload(
        req: GenerateWorkloadRequest,
    ) -> GenerateWorkloadResponse:
        """Synthesizes a reproducible workload given scenario, process count, and seed."""
        return generate_workload_service(scenario=req.scenario, n=req.n, seed=req.seed)

    @api.post(
        "/api/simulate",
        response_model=SimulateResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        summary="Simulate CPU Scheduling Policy",
    )
    async def post_simulate(req: SimulateRequest) -> SimulateResponse:
        """Executes simulation for a policy with ready-queue log and RL decision trace."""
        domain_procs = processes_from_schemas(req.processes)
        agent = api.state.q_agent_cs0 if req.cs_cost == 0 else api.state.q_agent_cs1
        return run_simulation_service(
            processes=domain_procs,
            algorithm=req.algorithm,
            quantum=req.quantum,
            cs_cost=req.cs_cost,
            rl_agent=agent,
        )

    @api.post(
        "/api/compare",
        response_model=CompareResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
        summary="Compare Multiple Scheduling Policies",
    )
    async def post_compare(req: CompareRequest) -> CompareResponse:
        """Benchmarks all candidate policies on a workload in a single round-trip."""
        domain_procs = processes_from_schemas(req.processes)
        agent = api.state.q_agent_cs0 if req.cs_cost == 0 else api.state.q_agent_cs1
        return run_comparison_service(
            processes=domain_procs,
            algorithms=req.algorithms,
            quantum=req.quantum,
            cs_cost=req.cs_cost,
            rl_agent=agent,
        )

    @api.post(
        "/api/recommend",
        response_model=RecommendResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        summary="Decision Tree Policy Recommendation",
    )
    async def post_recommend(req: RecommendRequest) -> RecommendResponse:
        """Predicts winning scheduling policy from workload features via Decision Tree."""
        domain_procs = processes_from_schemas(req.processes)
        return run_recommend_service(
            processes=domain_procs,
            dt_model_path=api.state.dt_path,
        )

    @api.get(
        "/api/ml/model-info",
        responses={503: {"model": ErrorResponse}},
        summary="Decision Tree Model Architecture & Test Metrics",
    )
    async def get_ml_model_info() -> dict[str, Any]:
        """Returns Decision Tree training metadata, feature importances, and confusion matrix."""
        meta_p = config.MODELS_DIR / "decision_tree_meta.json"
        test_p = config.RESULTS_DIR / "decision_tree_test_metrics.json"

        if not meta_p.exists():
            raise ModelMissingException("Train the model first: python cli.py train-tree")

        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        test_metrics = (
            json.loads(test_p.read_text(encoding="utf-8")) if test_p.exists() else None
        )

        return {
            "meta": meta,
            "test_metrics": test_metrics,
        }

    @api.get(
        "/api/rl/training-log",
        responses={503: {"model": ErrorResponse}},
        summary="Reinforcement Learning Learning-Curve Data",
    )
    async def get_rl_training_log(
        downsample: int = Query(1, ge=1, le=100, description="Downsample factor"),
    ) -> list[dict[str, Any]]:
        """Returns episodic training trajectories across seeds for dashboard learning curves."""
        log_p = config.DATA_DIR / "rl_training_logs.csv"
        if not log_p.exists():
            raise ModelMissingException("Train the model first: python cli.py train-rl")

        df = pd.read_csv(log_p)
        if downsample > 1:
            df = df.iloc[::downsample]
        return df.to_dict(orient="records")

    @api.get(
        "/api/rl/policy-summary",
        responses={503: {"model": ErrorResponse}},
        summary="Q-Table State Summary",
    )
    async def get_rl_policy_summary(
        cs_cost: int = Query(0, ge=0, le=1, description="Context switch cost variant"),
    ) -> dict[str, Any]:
        """Returns top visited state Q-values and action shares from JSON summary."""
        target_name = "q_table_summary_cs0.json" if cs_cost == 0 else "q_table_summary_cs1.json"
        summary_p = config.MODELS_DIR / target_name
        if not summary_p.exists():
            summary_p = config.MODELS_DIR / "q_table_summary.json"

        if not summary_p.exists():
            raise ModelMissingException("Train the model first: python cli.py train-rl")

        return json.loads(summary_p.read_text(encoding="utf-8"))

    @api.get(
        "/api/results/summary",
        responses={503: {"model": ErrorResponse}},
        summary="Final Benchmark Results and Report Tables",
    )
    async def get_results_summary() -> dict[str, Any]:
        """Returns held-out test split summary table, scenario breakdown, and limitations."""
        sum_p = config.RESULTS_DIR / "summary_table.csv"
        scen_p = config.RESULTS_DIR / "scenario_breakdown.csv"
        rep_p = config.DOCS_DIR / "REPORT_NOTES.md"

        if not sum_p.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Run evaluation first: python cli.py evaluate",
            )

        sum_df = pd.read_csv(sum_p)
        scen_df = pd.read_csv(scen_p) if scen_p.exists() else pd.DataFrame()
        notes = rep_p.read_text(encoding="utf-8") if rep_p.exists() else ""

        return {
            "summary_table": sum_df.to_dict(orient="records"),
            "scenario_breakdown": scen_df.to_dict(orient="records"),
            "report_notes": notes,
        }

    return api


app = create_app()

__all__ = ["app", "create_app", "lifespan"]
