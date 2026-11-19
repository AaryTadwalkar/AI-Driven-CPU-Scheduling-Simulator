"""
1. What this file does:
   Implements unit, integration, and fault tolerance tests for Stage 19 FastAPI extensions,
   validating /api/simulate with new AI models, tick_us overhead accounting, /api/compare
   batch evaluation, /api/results/final endpoint retrieval, and HTTP 503 model-missing responses.
2. Overall design:
   Uses Starlette TestClient to probe the API application instance. Tests all 4 new learned
   models (DT, RF, PID_GBDT, PID_NN) and Q-learning under realistic hardware tick_us parameters,
   asserts effective turnaround and makespan metric calculations, validates final benchmark table
   ingestion, and verifies prescriptive error guidance when model checkpoints are missing.
3. Elements in the file:
   - client: Pytest fixture instantiating TestClient against create_app().
   - sample_workload_payload: Helper returning valid process JSON dictionary.
   - test_simulate_new_policies_with_tick_us: Tests /api/simulate with DT, RF, GBDT, NN.
   - test_simulate_qlearning_with_tick_us: Tests /api/simulate with QLEARNING and traces.
   - test_compare_new_policies: Tests /api/compare with new policies and effective metrics.
   - test_results_final_endpoint: Tests GET /api/results/final returning S0 and S1 tables.
   - test_model_missing_503_response: Tests HTTP 503 response on missing model checkpoints.
   - test_results_final_missing_503: Tests HTTP 503 response when CSV results are missing.
   - test_simulate_tick_us_validation: Tests boundary validation for invalid tick_us values.
4. Final output of the file:
   Executes pytest test suite certifying Stage 19 backend deliverables.
"""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import config
from schedsim.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    """
    1. The use of function:
       Pytest fixture instantiating TestClient against fresh create_app() application instance.
    2. How does the specific function work:
       Constructs FastAPI app via create_app() and yields initialized TestClient.
    3. Concepts used in the function:
       Pytest fixture, TestClient ASGI testing harness.
    4. Places where this function is imported and used:
       Injected into test functions across tests/test_stage19_api.py.
    5. Final output of the function, before and after:
       Before: Unconnected test function.
       After: Yields active TestClient connected to the FastAPI application.
    """
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _sample_payload() -> list[dict[str, Any]]:
    """
    1. The use of function:
       Returns a standardized 3-process workload dictionary payload for API testing.
    2. How does the specific function work:
       Constructs list of process dicts with valid PID, arrival, burst, and priority values.
    3. Concepts used in the function:
       Test fixture helper, input payload synthesis.
    4. Places where this function is imported and used:
       Internal helper in test_stage19_api.py.
    5. Final output of the function, before and after:
       Before: No payload.
       After: List of 3 valid process specification dicts.
    """
    return [
        {"pid": "P1", "arrival": 0, "burst": 5, "priority": 0},
        {"pid": "P2", "arrival": 1, "burst": 3, "priority": 0},
        {"pid": "P3", "arrival": 2, "burst": 2, "priority": 0},
    ]


def test_simulate_new_policies_with_tick_us(client: TestClient) -> None:
    """
    1. The use of function:
       Tests /api/simulate endpoint with DT, RF, PID_GBDT, and PID_NN under tick_us.
    2. How does the specific function work:
       Posts SimulateRequest for each new policy with tick_us=100.0. Asserts status 200,
       verifies presence and types of ai_overhead_ticks, decisions, inference_us_total,
       effective_makespan, and effective_turnaround in simulation_metrics.
    3. Concepts used in the function:
       HTTP POST testing, AI policy simulation, metric extraction verification.
    4. Places where this function is imported and used:
       Executed by pytest in Stage 19 test suite.
    5. Final output of the function, before and after:
       Before: Unverified API simulation for new models.
       After: Confirms 200 response and correct metric accounting for all 4 new models.
    """
    procs = _sample_payload()
    for algo in ("DT", "RF", "PID_GBDT", "PID_NN"):
        resp = client.post(
            "/api/simulate",
            json={
                "processes": procs,
                "algorithm": algo,
                "cs_cost": 1,
                "tick_us": 100.0,
            },
        )
        assert resp.status_code == 200, f"Failed for {algo}: {resp.text}"
        data = resp.json()
        assert data["algorithm"] == algo
        sm = data["simulation_metrics"]
        assert "ai_overhead_ticks" in sm
        assert "decisions" in sm
        assert "inference_us_total" in sm
        assert "effective_makespan" in sm
        assert "effective_turnaround" in sm
        assert sm["effective_makespan"] >= sm["makespan"]
        assert sm["effective_turnaround"] >= sm["avg_turnaround"]
        assert len(data["timeline"]) > 0
        assert len(data["ready_queue_log"]) > 0


def test_simulate_qlearning_with_tick_us(client: TestClient) -> None:
    """
    1. The use of function:
       Tests /api/simulate endpoint with QLEARNING and tick_us duration.
    2. How does the specific function work:
       Posts SimulateRequest for QLEARNING with cs_cost=1 and tick_us=50.0. Asserts status
       200, non-empty decision_trace, and valid effective metrics.
    3. Concepts used in the function:
       RL simulation testing, decision trace explainability, discrete tick accounting.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified Q-learning simulation under tick_us.
       After: Confirms 200 response with decision traces and effective metrics.
    """
    procs = _sample_payload()
    resp = client.post(
        "/api/simulate",
        json={
            "processes": procs,
            "algorithm": "QLEARNING",
            "cs_cost": 1,
            "tick_us": 50.0,
        },
    )
    assert resp.status_code == 200, f"Failed for QLEARNING: {resp.text}"
    data = resp.json()
    assert data["decision_trace"] is not None
    assert len(data["decision_trace"]) > 0
    sm = data["simulation_metrics"]
    assert sm["decisions"] > 0
    assert sm["effective_makespan"] >= sm["makespan"]


def test_compare_new_policies(client: TestClient) -> None:
    """
    1. The use of function:
       Tests /api/compare endpoint comparing classical and learned policies in single call.
    2. How does the specific function work:
       Posts CompareRequest with algorithms=['FCFS', 'DT', 'PID_NN', 'SRTF'] and tick_us=100.0.
       Asserts status 200, confirms all 4 algorithms in results, and verifies summary rows.
    3. Concepts used in the function:
       Batch multi-algorithm benchmarking, summary table verification.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified compare route with new policies.
       After: Confirms complete comparative benchmarking payload with effective metrics.
    """
    procs = _sample_payload()
    target_algos = ["FCFS", "DT", "PID_NN", "SRTF"]
    resp = client.post(
        "/api/compare",
        json={
            "processes": procs,
            "algorithms": target_algos,
            "cs_cost": 1,
            "tick_us": 100.0,
        },
    )
    assert resp.status_code == 200, f"Compare failed: {resp.text}"
    data = resp.json()
    for algo in target_algos:
        assert algo in data["results"]
    summary = data["comparison_summary"]
    assert len(summary) == len(target_algos)
    for row in summary:
        assert "algorithm" in row
        assert "effective_makespan" in row
        assert "effective_turnaround" in row
        assert "ai_overhead_ticks" in row


def test_results_final_endpoint(client: TestClient) -> None:
    """
    1. The use of function:
       Tests GET /api/results/final endpoint returning held-out test split comparison tables.
    2. How does the specific function work:
       Sends GET /api/results/final request. Asserts status 200, verifies presence of S0 and S1
       lists of 11 rows each, and checks mandatory columns (policy, avg_waiting, win_tie_loss).
    3. Concepts used in the function:
       REST final results retrieval, schema contract validation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified /api/results/final endpoint.
       After: Confirms 200 response with complete S0 and S1 benchmark matrices.
    """
    resp = client.get("/api/results/final")
    assert resp.status_code == 200
    data = resp.json()
    assert "S0" in data
    assert "S1" in data
    assert len(data["S0"]) == 11
    assert len(data["S1"]) == 11
    # Check first row has expected structure
    first_s0 = data["S0"][0]
    assert "policy" in first_s0
    assert "avg_waiting" in first_s0
    assert "effective_makespan" in first_s0
    assert "win_tie_loss" in first_s0


def test_model_missing_503_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    1. The use of function:
       Verifies HTTP 503 response with prescriptive CLI guidance when model files are missing.
    2. How does the specific function work:
       Sets config.MODELS_DIR to an empty tmp_path. Directly sends /api/simulate requests for
       DT, RF, PID_GBDT, PID_NN, and QLEARNING. Asserts status 503 and checks training guidance.
    3. Concepts used in the function:
       Fault injection testing, HTTP 503 handling, prescriptive CLI error messages.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified missing model handling for new policies.
       After: Confirms 503 error responses with precise CLI training instructions.
    """
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    app = create_app()

    with TestClient(app) as test_client:
        procs = _sample_payload()
        expectations = [
            ("DT", "python cli.py train-tree"),
            ("RF", "python cli.py train-forest"),
            ("PID_GBDT", "python cli.py train-pid-gbdt"),
            ("PID_NN", "python cli.py train-pid-nn"),
            ("QLEARNING", "python cli.py train-rl"),
        ]
        for algo, expected_cmd in expectations:
            resp = test_client.post(
                "/api/simulate",
                json={"processes": procs, "algorithm": algo, "cs_cost": 1},
            )
            assert resp.status_code == 503, f"Expected 503 for {algo}, got {resp.status_code}"
            detail = resp.json().get("detail", "")
            assert expected_cmd in detail, f"Expected '{expected_cmd}' in '{detail}'"


def test_results_final_missing_503(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    1. The use of function:
       Verifies HTTP 503 response when final comparison CSV files are absent on disk.
    2. How does the specific function work:
       Points config.RESULTS_DIR to empty tmp_path via monkeypatch. Sends GET /api/results/final
       and asserts status 503 with instruction to run 'python cli.py compare-all'.
    3. Concepts used in the function:
       Fault tolerance, missing artifact handling.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified missing results handling.
       After: Confirms 503 response with compare-all instruction.
    """
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    app = create_app()

    with TestClient(app) as test_client:
        resp = test_client.get("/api/results/final")
        assert resp.status_code == 503
        detail = resp.json().get("detail", "")
        assert "python cli.py compare-all" in detail


def test_simulate_tick_us_validation(client: TestClient) -> None:
    """
    1. The use of function:
       Tests input boundary validation on tick_us in /api/simulate and /api/compare.
    2. How does the specific function work:
       Sends tick_us=0.0 and tick_us=-5.0 (below ge=0.001 minimum boundary). Asserts 422
       Unprocessable Entity validation response.
    3. Concepts used in the function:
       Pydantic boundary validation testing.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: Unverified tick_us input boundaries.
       After: Confirms 422 rejection on non-positive or out-of-bounds tick durations.
    """
    procs = _sample_payload()
    for bad_tick in (0.0, -10.0, 200000.0):
        resp_sim = client.post(
            "/api/simulate",
            json={
                "processes": procs,
                "algorithm": "FCFS",
                "tick_us": bad_tick,
            },
        )
        assert resp_sim.status_code == 422

        resp_cmp = client.post(
            "/api/compare",
            json={
                "processes": procs,
                "tick_us": bad_tick,
            },
        )
        assert resp_cmp.status_code == 422
