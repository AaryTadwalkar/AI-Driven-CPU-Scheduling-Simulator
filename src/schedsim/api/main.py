"""
1. What this file does:
   Provides the main entry point and Uvicorn runner for the FastAPI scheduling API service.
2. Overall design:
   Re-exports the application instance `app` so `uvicorn schedsim.api.main:app` serves the API.
   Includes a direct execution handler running `uvicorn.run()` on port 8000 when invoked via CLI.
3. Elements in the file:
   - app: FastAPI application instance created by create_app().
   - main: Direct execution runner launching Uvicorn ASGI server.
4. Final output of the file:
   Allows running backend via `uvicorn schedsim.api.main:app` or `python -m schedsim.api.main`.
"""

import sys

import uvicorn

from schedsim.api.app import app, create_app


def main() -> None:
    """
    1. The use of function:
       Runs the Uvicorn ASGI web server serving the FastAPI application on port 8000.
    2. How does the specific function work:
       Launches uvicorn.run with host 0.0.0.0 and port 8000.
    3. Concepts used in the function:
       ASGI server execution, CLI application launcher.
    4. Places where this function is imported and used:
       Executed when running `python -m schedsim.api.main`.
    5. Final output of the function, before and after:
       Before: Offline server.
       After: Active HTTP listening socket on port 8000.
    """
    uvicorn.run("schedsim.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
    sys.exit(0)


__all__ = ["app", "create_app", "main"]
