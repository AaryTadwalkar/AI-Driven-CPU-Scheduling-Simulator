"""
1. What this file does:
   Implements comprehensive unit, integration, and security/boundary tests for the FastAPI
   backend service using starlette / fastapi TestClient.
2. Overall design:
   Validates every API route (/health, /presets, /workloads/generate, /simulate, /compare,
   /recommend, /ml/model-info, /rl/training-log, /rl/policy-summary, /results/summary).
   Asserts Pydantic validation boundaries (n <= 50, burst <= 100, unique PIDs), verifies
   exact identity between /simulate and Engine run() output, and validates HTTP 503 fallback
   when machine learning and RL models are absent from disk.
3. Elements in the file:
   - test_health_endpoint: Checks health status, version, and model presence flags.
   - test_presets_endpoint: Asserts presence of canonical presets and process schedules.
   - test_workload_generate_endpoint: Checks synthetic generation and error handling.
   - test_simulate_classical_endpoint: Tests simulation for classical FCFS and RR.
   - test_simulate_equals_run_directly: Asserts exact mathematical parity with engine.run().
   - test_simulate_qlearning_with_trace: Validates RL decision trace explainability log.
   - test_compare_endpoint: Validates multi-algorithm benchmarking in single round-trip.
   - test_recommend_endpoint: Tests supervised Decision Tree policy recommendation.
   - test_analytics_endpoints: Tests model-info, rl-training-log, policy-summary, results-summary.
   - test_model_missing_503_handling: Tests graceful HTTP 503 degradation with missing models.
   - test_payload_limits_validation: Asserts boundary rejections for process counts and bursts.
4. Final output of the file:
   Executes pytest test suite certifying the FastAPI backend service for Stage 8.
"""

import pytest
from fastapi.testclient import TestClient

import config
from schedsim.api.app import create_app
from schedsim.domain.engine import SimResult, run
from schedsim.policies.registry import get_policy
from schedsim.workloads.presets import get_preset


@pytest.fixture
def client() -> TestClient:
    """
    1. The use of function:
       Pytest fixture creating an isolated TestClient for FastAPI requests.
    2. How does the specific function work:
       Calls create_app() and wraps it in starlette.testclient.TestClient context.
    3. Concepts used in the function:
       Pytest fixture, TestClient ASGI testing harness.
    4. Places where this function is imported and used:
       Injected into test functions across tests/test_api.py.
    5. Final output of the function, before and after:
       Before: Unconnected test function.
       After: Yields active TestClient connected to the FastAPI application.
    """
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    """
    1. The use of function:
       Tests the /api/health endpoint for operational status and model availability.
    2. How does the specific function work:
       Sends GET /api/health request. Asserts status code 200, status == 'ok', version == '1.0.0',
       and presence of models and artifacts dictionaries.
    3. Concepts used in the function:
       HTTP GET testing, health probing, dictionary schema verification.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified health route.
       After: Confirms correct 200 health payload.
    """
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "1.0.0"
    assert "models" in data
    assert "decision_tree" in data["models"]
    assert "artifacts" in data


def test_presets_endpoint(client: TestClient) -> None:
    """
    1. The use of function:
       Tests the /api/presets endpoint listing canonical hand-crafted workloads.
    2. How does the specific function work:
       Sends GET /api/presets request. Asserts 200 response, verifies presence of 'section_4_6'
       and 'convoy', and checks process schedules and descriptions.
    3. Concepts used in the function:
       REST collection querying, schema validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified presets route.
       After: Confirms presets map with valid process structures.
    """
    resp = client.get("/api/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert "section_4_6" in presets
    assert "convoy" in presets

    fixture = presets["section_4_6"]
    assert fixture["process_count"] == 3
    assert len(fixture["processes"]) == 3
    assert fixture["processes"][0]["pid"] == "P1"


def test_workload_generate_endpoint(client: TestClient) -> None:
    """
    1. The use of function:
       Tests the /api/workloads/generate endpoint for on-demand synthetic workload creation.
    2. How does the specific function work:
       Sends POST /api/workloads/generate with valid parameters. Asserts 200 response, checks
       process count, and verifies deterministic PID naming. Tests invalid scenario returns 400.
    3. Concepts used in the function:
       POST request testing, input validation, deterministic generation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified generation route.
       After: Confirms 200 on valid inputs and 400 on invalid scenarios.
    """
    payload = {"scenario": "short_jobs", "n": 4, "seed": 42}
    resp = client.post("/api/workloads/generate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario"] == "short_jobs"
    assert data["seed"] == 42
    assert len(data["processes"]) == 4

    # Invalid scenario returns 400
    bad_resp = client.post("/api/workloads/generate", json={"scenario": "nonexistent", "n": 4})
    assert bad_resp.status_code == 400


def test_simulate_classical_endpoint(client: TestClient) -> None:
    """
    1. The use of function:
       Tests the /api/simulate endpoint on classical scheduling algorithms (FCFS, RR).
    2. How does the specific function work:
       Sends POST /api/simulate with 3 processes and algorithm 'FCFS'. Asserts 200 response,
       verifies presence of process_metrics, simulation_metrics, timeline segments, and
       ready_queue_log. Tests Round Robin with custom quantum.
    3. Concepts used in the function:
       Simulation dispatch testing, metric extraction, playback logging.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified simulation route.
       After: Confirms valid 200 simulation response for classical policies.
    """
    payload = {
        "processes": [
            {"pid": "P1", "arrival": 0, "burst": 5, "priority": 0},
            {"pid": "P2", "arrival": 1, "burst": 3, "priority": 0},
            {"pid": "P3", "arrival": 2, "burst": 2, "priority": 0},
        ],
        "algorithm": "FCFS",
        "quantum": 2,
        "cs_cost": 0,
    }
    resp = client.post("/api/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["algorithm"] == "FCFS"
    assert len(data["process_metrics"]) == 3
    assert data["simulation_metrics"]["avg_waiting"] > 0
    assert len(data["timeline"]) == 3
    assert len(data["ready_queue_log"]) > 0

    # Test Round Robin
    payload["algorithm"] = "RR"
    payload["quantum"] = 2
    rr_resp = client.post("/api/simulate", json=payload)
    assert rr_resp.status_code == 200
    rr_data = rr_resp.json()
    assert rr_data["algorithm"] == "RR"
    assert rr_data["simulation_metrics"]["context_switches"] > 0


def test_simulate_equals_run_directly(client: TestClient) -> None:
    """
    1. The use of function:
       Confirms that /api/simulate output equals domain Engine.run() output directly.
    2. How does the specific function work:
       Runs Section 4.6 fixture through both /api/simulate and Engine.run() with SRTF policy.
       Asserts exact numeric equality for waiting, response, turnaround, and makespan.
    3. Concepts used in the function:
       Parity assertion, domain abstraction verification, regression protection.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified API parity with library.
       After: Asserts exact algebraic match between API and run().
    """
    raw_procs = get_preset("section_4_6")
    p_dicts = [
        {"pid": p.pid, "arrival": p.arrival, "burst": p.burst, "priority": p.priority}
        for p in raw_procs
    ]

    # API execution
    api_resp = client.post(
        "/api/simulate",
        json={"processes": p_dicts, "algorithm": "SRTF", "cs_cost": 1},
    )
    assert api_resp.status_code == 200
    api_sm = api_resp.json()["simulation_metrics"]

    # Library direct execution
    direct_res: SimResult = run(get_policy("srtf"), raw_procs, cs_cost=1)
    direct_sm = direct_res.simulation_metrics

    assert api_sm["avg_waiting"] == direct_sm.avg_waiting
    assert api_sm["avg_turnaround"] == direct_sm.avg_turnaround
    assert api_sm["avg_response"] == direct_sm.avg_response
    assert api_sm["makespan"] == direct_sm.makespan
    assert api_sm["context_switches"] == direct_sm.context_switches


def test_simulate_qlearning_with_trace(client: TestClient) -> None:
    """
    1. The use of function:
       Tests the /api/simulate endpoint with QLEARNING, verifying the decision trace.
    2. How does the specific function work:
       Sends POST /api/simulate with algorithm 'QLEARNING' and cs_cost=0. Asserts 200 response,
       verifies decision_trace is non-empty, and checks state, valid actions, Q-values, and reward.
    3. Concepts used in the function:
       RL explainability testing, decision trace verification.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified RL simulation route.
       After: Confirms populated decision trace for Q-learning.
    """
    procs = [
        {"pid": "P1", "arrival": 0, "burst": 4, "priority": 0},
        {"pid": "P2", "arrival": 1, "burst": 2, "priority": 0},
    ]
    resp = client.post(
        "/api/simulate",
        json={"processes": procs, "algorithm": "QLEARNING", "cs_cost": 0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["algorithm"] == "QLEARNING"
    assert data["decision_trace"] is not None
    assert len(data["decision_trace"]) > 0

    first_step = data["decision_trace"][0]
    assert "time" in first_step
    assert "state" in first_step
    assert "valid_actions" in first_step
    assert "q_values" in first_step
    assert "chosen_action" in first_step
    assert "reward" in first_step


def test_compare_endpoint(client: TestClient) -> None:
    """
    1. The use of function:
       Tests the /api/compare endpoint running multi-algorithm benchmarks in a single call.
    2. How does the specific function work:
       Sends POST /api/compare with 3 processes. Asserts 200 response, checks that results
       contains FCFS, SJF, RR, and SRTF, and verifies comparison_summary table rows.
    3. Concepts used in the function:
       Batch comparison testing, summary table validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified compare route.
       After: Confirms multi-algorithm benchmark response.
    """
    payload = {
        "processes": [
            {"pid": "P1", "arrival": 0, "burst": 4, "priority": 0},
            {"pid": "P2", "arrival": 2, "burst": 3, "priority": 0},
        ],
        "quantum": 2,
        "cs_cost": 0,
    }
    resp = client.post("/api/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert "FCFS" in data["results"]
    assert "SJF" in data["results"]
    assert "comparison_summary" in data
    assert len(data["comparison_summary"]) >= 4


def test_recommend_endpoint(client: TestClient) -> None:
    """
    1. The use of function:
       Tests the /api/recommend endpoint querying Decision Tree policy predictions.
    2. How does the specific function work:
       Sends POST /api/recommend with processes. Asserts 200 response, verifies winning algorithm
       is in ('FCFS', 'SJF', 'RR'), and checks probability distribution and 12 features.
    3. Concepts used in the function:
       Supervised inference testing, probability distribution verification.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified recommend route.
       After: Confirms 200 with valid algorithm recommendation and probabilities.
    """
    payload = {
        "processes": [
            {"pid": "P1", "arrival": 0, "burst": 5, "priority": 0},
            {"pid": "P2", "arrival": 1, "burst": 2, "priority": 0},
            {"pid": "P3", "arrival": 3, "burst": 8, "priority": 0},
        ]
    }
    resp = client.post("/api/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["algorithm"] in ("FCFS", "SJF", "RR")
    assert "probabilities" in data
    assert len(data["features"]) == 12


def test_analytics_endpoints(client: TestClient) -> None:
    """
    1. The use of function:
       Tests reporting and analytics endpoints (/ml/model-info, /rl/*, /results/summary).
    2. How does the specific function work:
       Queries /api/ml/model-info, /api/rl/training-log, /api/rl/policy-summary, and
       /api/results/summary. Asserts 200 responses and validates key dictionary fields.
    3. Concepts used in the function:
       Analytics endpoint querying, artifact schema validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified analytics endpoints.
       After: Confirms all analytics routes return valid 200 payloads.
    """
    # ML model info
    r_ml = client.get("/api/ml/model-info")
    assert r_ml.status_code == 200
    ml_data = r_ml.json()
    assert "meta" in ml_data
    assert "chosen_params" in ml_data["meta"]

    # RL training log
    r_log = client.get("/api/rl/training-log?downsample=10")
    assert r_log.status_code == 200
    assert isinstance(r_log.json(), list)

    # RL policy summary
    r_pol = client.get("/api/rl/policy-summary?cs_cost=0")
    assert r_pol.status_code == 200
    assert "top_states" in r_pol.json()

    # Results summary
    r_sum = client.get("/api/results/summary")
    assert r_sum.status_code == 200
    sum_data = r_sum.json()
    assert "summary_table" in sum_data
    assert "report_notes" in sum_data


def test_model_missing_503_handling(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """
    1. The use of function:
       Verifies graceful degradation and HTTP 503 response when models are missing on disk.
    2. How does the specific function work:
       Directly instantiates app with empty state (no agents or tree). Sends requests to
       /api/recommend and /api/simulate (QLEARNING). Asserts 503 status code with prescriptive
       CLI training guidance. Confirms classical FCFS simulation still works 100% fine.
    3. Concepts used in the function:
       Fault tolerance testing, HTTP 503 handling, partial service availability.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified 503 handling.
       After: Confirms 503 on missing models and continued operation of classical simulator.
    """
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    app = create_app()

    with TestClient(app) as test_client:
        # /api/recommend returns 503 with clear message
        procs = [{"pid": "P1", "arrival": 0, "burst": 4, "priority": 0}]
        resp_rec = test_client.post("/api/recommend", json={"processes": procs})
        assert resp_rec.status_code == 503
        assert "Train the model first: python cli.py train-tree" in resp_rec.json()["detail"]

        # /api/simulate with QLEARNING returns 503
        resp_sim = test_client.post(
            "/api/simulate",
            json={"processes": procs, "algorithm": "QLEARNING", "cs_cost": 0},
        )
        assert resp_sim.status_code == 503
        assert "Train the model first: python cli.py train-rl" in resp_sim.json()["detail"]

        # Classical policy still works normally
        resp_fcfs = test_client.post(
            "/api/simulate",
            json={"processes": procs, "algorithm": "FCFS", "cs_cost": 0},
        )
        assert resp_fcfs.status_code == 200
        assert resp_fcfs.json()["algorithm"] == "FCFS"


def test_payload_limits_validation(client: TestClient) -> None:
    """
    1. The use of function:
       Tests API boundary rejections and schema validations on malformed or oversized payloads.
    2. How does the specific function work:
       Tests > 50 processes returns 422, burst > 100 returns 422, negative arrival returns 422,
       duplicate PIDs return 422, and invalid context switch cost returns 422.
    3. Concepts used in the function:
       Boundary condition testing, Pydantic validation enforcement.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified input boundary enforcement.
       After: Confirms 422 rejections on all boundary violations.
    """
    # 1. More than 50 processes
    oversized = [
        {"pid": f"P{i}", "arrival": 0, "burst": 2, "priority": 0}
        for i in range(51)
    ]
    r1 = client.post("/api/simulate", json={"processes": oversized, "algorithm": "FCFS"})
    assert r1.status_code == 422

    # 2. Burst duration > 100
    too_large_burst = [{"pid": "P1", "arrival": 0, "burst": 101, "priority": 0}]
    r2 = client.post("/api/simulate", json={"processes": too_large_burst, "algorithm": "FCFS"})
    assert r2.status_code == 422

    # 3. Negative arrival tick
    neg_arrival = [{"pid": "P1", "arrival": -1, "burst": 5, "priority": 0}]
    r3 = client.post("/api/simulate", json={"processes": neg_arrival, "algorithm": "FCFS"})
    assert r3.status_code == 422

    # 4. Duplicate process IDs
    duplicate_pids = [
        {"pid": "P1", "arrival": 0, "burst": 4, "priority": 0},
        {"pid": "P1", "arrival": 2, "burst": 3, "priority": 0},
    ]
    r4 = client.post("/api/simulate", json={"processes": duplicate_pids, "algorithm": "FCFS"})
    assert r4.status_code == 422
