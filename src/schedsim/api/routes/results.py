"""
1. What this file does:
   Defines the FastAPI APIRouter for benchmark results endpoints, including final comparison
   matrices (Setting S0 and S1) and held-out test split evaluation summary tables.
2. Overall design:
   Mounts endpoints under the /api/results route namespace. Routes HTTP requests to service
   layer helpers, translating disk CSV evaluation matrices and report notes into structured
   Pydantic JSON responses, with graceful HTTP 503 handling when evaluations have not been run.
3. Elements in the file:
   - router: APIRouter instance grouping result retrieval endpoints.
   - get_results_final: Handler for GET /api/results/final.
   - get_results_summary: Handler for GET /api/results/summary.
4. Final output of the file:
   Exports router for inclusion in the primary FastAPI application.
"""

from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status

import config
from schedsim.api.schemas import ErrorResponse, FinalComparisonResponse
from schedsim.api.service import get_final_results_service

router = APIRouter(prefix="/api/results", tags=["results"])


@router.get(
    "/final",
    response_model=FinalComparisonResponse,
    responses={503: {"model": ErrorResponse}},
    summary="Final Benchmark Comparison Results (S0 and S1)",
)
async def get_results_final() -> FinalComparisonResponse:
    """
    1. The use of function:
       Endpoint returning held-out test split final comparison matrices for S0 and S1.
    2. How does the specific function work:
       Delegates to get_final_results_service() to read final_comparison_S0.csv and
       final_comparison_S1.csv, returning structured FinalComparisonResponse.
    3. Concepts used in the function:
       FastAPI route handler, schema serialization, model missing propagation.
    4. Places where this function is imported and used:
       Mounted in router and queried by dashboard and external evaluators.
    5. Final output of the function, before and after:
       Before: HTTP GET request.
       After: JSON payload with S0 and S1 comparison tables.
    """
    return get_final_results_service()


@router.get(
    "/summary",
    responses={503: {"model": ErrorResponse}},
    summary="Final Benchmark Results and Report Tables",
)
async def get_results_summary() -> dict[str, Any]:
    """
    1. The use of function:
       Endpoint returning summary metrics, scenario breakdowns, and report notes.
    2. How does the specific function work:
       Reads summary_table.csv, scenario_breakdown.csv, and REPORT_NOTES.md from results and
       docs directories, packaging records into a dictionary.
    3. Concepts used in the function:
       FastAPI route handler, Pandas CSV serialization, markdown file reading.
    4. Places where this function is imported and used:
       Mounted in router and consumed by frontend Results page.
    5. Final output of the function, before and after:
       Before: HTTP GET request.
       After: Dictionary with summary_table, scenario_breakdown, and report_notes.
    """
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


__all__ = ["get_results_final", "get_results_summary", "router"]
