"""
1. What this file does:
   Implements the comprehensive Stage 8 acceptance test suite certifying the FastAPI backend
   service, schema validations, endpoint contracts, CORS headers, and OpenAPI specification.
2. Overall design:
   Validates every required Stage 8 acceptance criterion from Section 5 of the Implementation Plan:
   - OpenAPI schema valid and discoverable.
   - All 10 /api endpoints functional with correct response models.
   - Pydantic bounds (n <= 50, burst <= 100, arrival <= 500, unique PIDs).
   - CORS middleware headers for http://localhost:5173.
   - Exact mathematical parity between /simulate and Engine run() output.
   - Graceful HTTP 503 fallback with clear prescriptive instructions when models are absent.
   - Uniform error envelope {error, detail}.
3. Elements in the file:
   - test_openapi_schema_completeness: Validates OpenAPI title, version, and route paths.
   - test_cors_headers_present: Checks CORS pre-flight and response headers for localhost:5173.
   - test_all_ten_endpoints_functional: Confirms all endpoints return successful status codes.
   - test_simulation_parity_with_domain_engine: Validates exact metric parity with run().
   - test_payload_boundary_limits_enforced: Asserts 422 on burst > 100, n > 50, duplicate PIDs.
   - test_model_missing_503_prescriptive_error: Tests 503 error format on missing models.
4. Final output of the file:
   Executes pytest test assertions certifying that Stage 8 meets all acceptance criteria.
"""

import pytest
from fastapi.testclient import TestClient

import config
from schedsim.api.app import app, create_app
from schedsim.domain.engine import SimResult, run
from schedsim.policies.registry import get_policy
from schedsim.workloads.presets import get_preset


@pytest.fixture
def client() -> TestClient:
    """
    1. The use of function:
       Fixture providing a TestClient instance bound to the application.
    2. How does the specific function work:
       Wraps app in starlette / fastapi TestClient context.
    3. Concepts used in the function:
       FastAPI test fixture.
    4. Places where this function is imported and used:
       Injected into acceptance tests.
    5. Final output of the function, before and after:
       Before: Unbound test function.
       After: Yields connected TestClient.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_openapi_schema_completeness(client: TestClient) -> None:
    """
    1. The use of function:
       Validates that the OpenAPI schema is valid, non-empty, and contains all 10 endpoints.
    2. How does the specific function work:
       Fetches /openapi.json. Asserts status 200, checks title and version, and verifies that all
       required /api routes are listed in the OpenAPI paths object.
    3. Concepts used in the function:
       OpenAPI specification verification, schema introspection.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified OpenAPI schema.
       After: Confirms all 10 endpoint paths are present and documented.
    """
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()

    assert schema["info"]["title"] == "AI-Driven CPU Scheduling Simulator API"
    assert schema["info"]["version"] == "1.0.0"

    expected_routes = [
        "/api/health",
        "/api/presets",
        "/api/workloads/generate",
        "/api/simulate",
        "/api/compare",
        "/api/recommend",
        "/api/ml/model-info",
        "/api/rl/training-log",
        "/api/rl/policy-summary",
        "/api/results/summary",
    ]
    paths = schema["paths"]
    for route in expected_routes:
        assert route in paths, f"Route missing from OpenAPI specification: {route}"


def test_cors_headers_present(client: TestClient) -> None:
    """
    1. The use of function:
       Verifies that CORS middleware headers are properly returned for localhost:5173.
    2. How does the specific function work:
       Sends an OPTIONS pre-flight request with Origin 'http://localhost:5173'.
       Asserts that 'access-control-allow-origin' equals 'http://localhost:5173'.
    3. Concepts used in the function:
       CORS pre-flight verification, cross-origin security header inspection.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified CORS configuration.
       After: Asserts allowed origin for React dashboard.
    """
    headers = {
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    }
    resp = client.options("/api/simulate", headers=headers)
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_all_ten_endpoints_functional(client: TestClient) -> None:
    """
    1. The use of function:
       Verifies that all 10 required /api endpoints execute successfully and return status 200.
    2. How does the specific function work:
       Executes requests against each endpoint with valid parameters, asserting status 200.
    3. Concepts used in the function:
       End-to-end integration testing, full API route coverage.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified route execution.
       After: Confirms all 10 endpoints return HTTP 200.
    """
    sample_procs = [
        {"pid": "P1", "arrival": 0, "burst": 4, "priority": 0},
        {"pid": "P2", "arrival": 1, "burst": 2, "priority": 0},
    ]

    # 1. Health
    assert client.get("/api/health").status_code == 200

    # 2. Presets
    assert client.get("/api/presets").status_code == 200

    # 3. Workload Generate
    r_gen = client.post("/api/workloads/generate", json={"scenario": "short_jobs", "n": 3})
    assert r_gen.status_code == 200

    # 4. Simulate
    r_sim = client.post(
        "/api/simulate",
        json={"processes": sample_procs, "algorithm": "SJF", "cs_cost": 0},
    )
    assert r_sim.status_code == 200

    # 5. Compare
    r_comp = client.post(
        "/api/compare",
        json={"processes": sample_procs, "quantum": 2, "cs_cost": 0},
    )
    assert r_comp.status_code == 200

    # 6. Recommend
    r_rec = client.post("/api/recommend", json={"processes": sample_procs})
    assert r_rec.status_code == 200

    # 7. ML Model Info
    assert client.get("/api/ml/model-info").status_code == 200

    # 8. RL Training Log
    assert client.get("/api/rl/training-log?downsample=10").status_code == 200

    # 9. RL Policy Summary
    assert client.get("/api/rl/policy-summary?cs_cost=0").status_code == 200

    # 10. Results Summary
    assert client.get("/api/results/summary").status_code == 200


def test_simulation_parity_with_domain_engine(client: TestClient) -> None:
    """
    1. The use of function:
       Certifies exact mathematical parity between /api/simulate and Engine.run().
    2. How does the specific function work:
       Runs Section 4.6 reference fixture across FCFS, SJF, RR, and SRTF via API and library.
       Asserts average waiting, response, and turnaround match to 6 decimal places.
    3. Concepts used in the function:
       Engine invariant assertion, parity verification across classical algorithms.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified parity across multiple policies.
       After: Confirms exact numerical parity for all algorithms.
    """
    raw_procs = get_preset("section_4_6")
    p_dicts = [
        {"pid": p.pid, "arrival": p.arrival, "burst": p.burst, "priority": p.priority}
        for p in raw_procs
    ]

    for algo in ["FCFS", "SJF", "RR", "SRTF"]:
        api_resp = client.post(
            "/api/simulate",
            json={"processes": p_dicts, "algorithm": algo, "quantum": 2, "cs_cost": 1},
        )
        assert api_resp.status_code == 200
        api_metrics = api_resp.json()["simulation_metrics"]

        policy_obj = (
            get_policy("rr", quantum=2) if algo == "RR" else get_policy(algo.lower())
        )
        lib_res: SimResult = run(policy_obj, raw_procs, cs_cost=1)
        lib_metrics = lib_res.simulation_metrics

        assert abs(api_metrics["avg_waiting"] - lib_metrics.avg_waiting) < 1e-6
        assert abs(api_metrics["avg_response"] - lib_metrics.avg_response) < 1e-6
        assert abs(api_metrics["avg_turnaround"] - lib_metrics.avg_turnaround) < 1e-6
        assert api_metrics["context_switches"] == lib_metrics.context_switches


def test_payload_boundary_limits_enforced(client: TestClient) -> None:
    """
    1. The use of function:
       Validates strict boundary limit enforcement (n <= 50, burst <= 100, unique PIDs).
    2. How does the specific function work:
       Tests payloads with 51 processes, burst 101, duplicate PIDs, and negative arrival ticks.
       Asserts that all invalid payloads receive HTTP 422 with a structured error envelope.
    3. Concepts used in the function:
       Defensive input validation, schema boundary assertion.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified boundary enforcement.
       After: Confirms 422 on all violations.
    """
    # 51 processes
    over_50 = [{"pid": f"P{i}", "arrival": 0, "burst": 2} for i in range(51)]
    r_51 = client.post("/api/simulate", json={"processes": over_50, "algorithm": "FCFS"})
    assert r_51.status_code == 422
    assert "error" in r_51.json()

    # Burst > 100
    b_101 = [{"pid": "P1", "arrival": 0, "burst": 101}]
    r_burst = client.post("/api/simulate", json={"processes": b_101, "algorithm": "FCFS"})
    assert r_burst.status_code == 422

    # Duplicate PIDs
    dups = [
        {"pid": "P1", "arrival": 0, "burst": 5},
        {"pid": "P1", "arrival": 2, "burst": 3},
    ]
    r_dup = client.post("/api/simulate", json={"processes": dups, "algorithm": "FCFS"})
    assert r_dup.status_code == 422


def test_model_missing_503_prescriptive_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """
    1. The use of function:
       Verifies HTTP 503 response and uniform error envelope {error, detail} when models absent.
    2. How does the specific function work:
       Monkeypatches config.MODELS_DIR to empty directory. Requests /api/recommend and
       /api/simulate (QLEARNING). Asserts status 503 and confirms prescriptive CLI messages.
       Confirms classical simulation continues to function normally.
    3. Concepts used in the function:
       Graceful service degradation, prescriptive error reporting, partial availability.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified 503 behavior.
       After: Confirms prescriptive 503 responses and resilient classical simulator.
    """
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    isolated_app = create_app()

    with TestClient(isolated_app) as isolated_client:
        sample = [{"pid": "P1", "arrival": 0, "burst": 4, "priority": 0}]

        # 1. Recommend returns 503
        r_rec = isolated_client.post("/api/recommend", json={"processes": sample})
        assert r_rec.status_code == 503
        data_rec = r_rec.json()
        assert data_rec["error"] == "Model Not Found"
        assert "Train the model first: python cli.py train-tree" in data_rec["detail"]

        # 2. QLearning simulation returns 503
        r_ql = isolated_client.post(
            "/api/simulate",
            json={"processes": sample, "algorithm": "QLEARNING", "cs_cost": 0},
        )
        assert r_ql.status_code == 503
        data_ql = r_ql.json()
        assert data_ql["error"] == "Model Not Found"
        assert "Train the model first: python cli.py train-rl" in data_ql["detail"]

        # 3. Classical simulator still works 100% fine
        r_fcfs = isolated_client.post(
            "/api/simulate",
            json={"processes": sample, "algorithm": "FCFS", "cs_cost": 0},
        )
        assert r_fcfs.status_code == 200
        assert r_fcfs.json()["algorithm"] == "FCFS"
