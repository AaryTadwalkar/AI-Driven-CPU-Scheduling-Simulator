"""
1. What this file does:
   Initializes the API package for the FastAPI backend service.
2. Overall design:
   Re-exports the application instance `app`, application factory `create_app`, schemas,
   and service logic connecting the simulation engine with the React dashboard.
3. Elements in the file:
   - app: Default application instance.
   - create_app: Application factory function.
4. Final output of the file:
   Enables module importing from schedsim.api.
"""

from schedsim.api.app import app, create_app, lifespan

__all__ = ["app", "create_app", "lifespan"]
