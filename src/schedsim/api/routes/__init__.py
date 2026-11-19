"""
1. What this file does:
   Initializes the API routes subpackage and exports modular APIRouter instances.
2. Overall design:
   Re-exports simulation_router and results_router for clean assembly within the FastAPI
   application factory.
3. Elements in the file:
   - results_router: Router for benchmark and comparison results.
   - simulation_router: Router for single-policy simulation and multi-policy comparison.
4. Final output of the file:
   Exposes results_router and simulation_router to callers.
"""

from schedsim.api.routes.results import router as results_router
from schedsim.api.routes.simulation import router as simulation_router

__all__ = ["results_router", "simulation_router"]
