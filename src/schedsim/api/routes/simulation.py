"""
1. What this file does:
   Defines the FastAPI APIRouter for simulation execution and multi-algorithm comparative
   benchmarking endpoints (/api/simulate and /api/compare).
2. Overall design:
   Mounts routes for discrete unit-step simulation with optional tick_us real-time accounting,
   playback ready queue logging, RL decision explainability traces, and batch comparative runs
   across classical, ML meta-selectors (DT/RF), Q-learning, GBDT ranker, and Neural scorers.
3. Elements in the file:
   - router: APIRouter instance grouping simulation execution endpoints.
   - post_simulate: Handler executing single policy simulation with latency overhead.
   - post_compare: Handler executing batch comparison across designated scheduling algorithms.
4. Final output of the file:
   Exports router for inclusion in the primary FastAPI application.
"""

from fastapi import APIRouter, Request

from schedsim.api.schemas import (
    CompareRequest,
    CompareResponse,
    ErrorResponse,
    SimulateRequest,
    SimulateResponse,
)
from schedsim.api.service import (
    processes_from_schemas,
    run_comparison_service,
    run_simulation_service,
)

router = APIRouter(tags=["simulation"])


@router.post(
    "/api/simulate",
    response_model=SimulateResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Simulate CPU Scheduling Policy",
)
async def post_simulate(req: SimulateRequest, request: Request) -> SimulateResponse:
    """
    1. The use of function:
       Executes simulation for a single policy with playback logging and RL decision traces.
    2. How does the specific function work:
       Extracts domain processes from request schema, retrieves cached Q-learning agent from
       application state if applicable, and invokes run_simulation_service with parameters.
    3. Concepts used in the function:
       FastAPI route handler, stateful agent retrieval, service delegation.
    4. Places where this function is imported and used:
       Mounted on router and queried by frontend Simulator page.
    5. Final output of the function, before and after:
       Before: SimulateRequest payload.
       After: SimulateResponse with metrics, timeline, and traces.
    """
    domain_procs = processes_from_schemas(req.processes)
    agent = (
        request.app.state.q_agent_cs0 if req.cs_cost == 0 else request.app.state.q_agent_cs1
    ) if hasattr(request.app.state, "q_agent_cs0") else None

    return run_simulation_service(
        processes=domain_procs,
        algorithm=req.algorithm,
        quantum=req.quantum,
        cs_cost=req.cs_cost,
        tick_us=req.tick_us,
        rl_agent=agent,
    )


@router.post(
    "/api/compare",
    response_model=CompareResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Compare Multiple Scheduling Policies",
)
async def post_compare(req: CompareRequest, request: Request) -> CompareResponse:
    """
    1. The use of function:
       Benchmarks designated scheduling algorithms on a single workload in one round-trip.
    2. How does the specific function work:
       Extracts domain processes from request schema, resolves cached Q-learning agent from
       application state, and calls run_comparison_service with all requested algorithms.
    3. Concepts used in the function:
       FastAPI route handler, comparative batch evaluation, service delegation.
    4. Places where this function is imported and used:
       Mounted on router and queried by frontend Compare page.
    5. Final output of the function, before and after:
       Before: CompareRequest payload.
       After: CompareResponse with dictionary of per-algorithm runs and summary rows.
    """
    domain_procs = processes_from_schemas(req.processes)
    agent = (
        request.app.state.q_agent_cs0 if req.cs_cost == 0 else request.app.state.q_agent_cs1
    ) if hasattr(request.app.state, "q_agent_cs0") else None

    return run_comparison_service(
        processes=domain_procs,
        algorithms=req.algorithms,
        quantum=req.quantum,
        cs_cost=req.cs_cost,
        tick_us=req.tick_us,
        rl_agent=agent,
    )


__all__ = ["post_compare", "post_simulate", "router"]
