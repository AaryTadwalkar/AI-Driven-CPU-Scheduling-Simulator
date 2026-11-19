"""
1. What this file does:
   Defines Pydantic request and response schemas for the FastAPI backend service.
2. Overall design:
   Enforces strict input validation, type coercion, and boundary assertions on processes,
   simulation parameters, and generation requests (limits: n <= 50, burst <= 100, arrival <= 500).
   Defines structured response models for simulation metrics, execution timelines, step playback,
   reinforcement learning decision traces, supervised recommendations, and system health.
3. Elements in the file:
   - ProcessSchema: Pydantic model for single process specifications.
   - SimulateRequest, SimulateResponse: Simulation execution contracts.
   - ReadyQueueLogEntry, DecisionTraceEntry: Step-level playback and RL trace contracts.
   - CompareRequest, CompareResponse: Multi-algorithm benchmarking contracts.
   - RecommendRequest, RecommendResponse: Decision Tree recommendation contracts.
   - GenerateWorkloadRequest, GenerateWorkloadResponse: Workload synthesis contracts.
   - HealthResponse, ErrorResponse: System diagnostics and uniform error schemas.
4. Final output of the file:
   Exports validated Pydantic models for API routing, validation, and OpenAPI documentation.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProcessSchema(BaseModel):
    """
    1. The use of class:
       Schema representing an individual process within a CPU scheduling workload.
    2. What the class contains:
       - pid: Unique alphanumeric process identifier string.
       - arrival: Non-negative integer arrival tick (0 to 500).
       - burst: Strictly positive integer CPU burst duration (1 to 100).
       - priority: Integer priority level (default 0).
    3. Technologies used and why:
       Pydantic v2 BaseModel with Field constraints ensuring runtime data integrity.
    4. Key design and overall design of the class:
       Strictly bounds inputs to avoid denial-of-service and memory exhaustion.
    """

    model_config = ConfigDict(frozen=True)

    pid: str = Field(..., min_length=1, max_length=16, description="Process ID (e.g. 'P1')")
    arrival: int = Field(..., ge=0, le=500, description="Arrival tick (0 <= arrival <= 500)")
    burst: int = Field(..., ge=1, le=100, description="CPU burst duration (1 <= burst <= 100)")
    priority: int = Field(0, description="Process priority level")


class SimulateRequest(BaseModel):
    """
    1. The use of class:
       Request payload for executing a single scheduling policy simulation.
    2. What the class contains:
       - processes: List of 1 to 50 ProcessSchema instances with unique PIDs.
       - algorithm: Target policy name ('FCFS', 'SJF', 'SRTF', 'RR', 'QLEARNING', 'RL').
       - quantum: Round Robin execution quantum (default 2, 1 to 50).
       - cs_cost: Context switch penalty in discrete ticks (default 0, 0 to 10).
    3. Technologies used and why:
       Pydantic v2 model_validator checking process count and PID uniqueness.
    4. Key design and overall design of the class:
       Defensive gateway preventing duplicate PIDs or unbounded execution payloads.
    """

    processes: list[ProcessSchema] = Field(
        ..., min_length=1, max_length=50, description="Workload processes (1 to 50)"
    )
    algorithm: str = Field(
        ...,
        description=(
            "Target policy (FCFS, SJF, SRTF, RR, QLEARNING, RL, DT, RF, PID_GBDT, PID_NN)"
        ),
    )
    quantum: int = Field(2, ge=1, le=50, description="Time quantum for Round Robin")
    cs_cost: int = Field(0, ge=0, le=10, description="Context switch cost in ticks")
    tick_us: float | None = Field(
        None, ge=0.001, le=100000.0, description="Optional microseconds per tick"
    )

    @model_validator(mode="after")
    def validate_unique_pids(self) -> "SimulateRequest":
        """
        1. The use of function:
           Ensures all process identifiers in the workload are strictly unique.
        2. How does the specific function work:
           Collects PIDs and asserts set cardinality matches list length.
        3. Concepts used in the function:
           Pydantic after-validator, uniqueness constraint verification.
        4. Places where this function is imported and used:
           Invoked automatically on SimulateRequest instantiation.
        5. Final output of the function, before and after:
           Before: Unverified process list.
           After: Confirmed unique PIDs or raises ValueError.
        """
        pids = [p.pid for p in self.processes]
        if len(pids) != len(set(pids)):
            raise ValueError("Process PIDs in workload must be strictly unique.")
        return self


class ProcessMetricsSchema(BaseModel):
    """
    1. The use of class:
       Schema representing final performance metrics for an individual process.
    2. What the class contains:
       pid, arrival, burst, completion, turnaround, waiting, and response times.
    3. Technologies used and why:
       Pydantic frozen model for immutable serialized output.
    4. Key design and overall design of the class:
       Matches domain ProcessMetrics dataclass for direct serialization.
    """

    model_config = ConfigDict(frozen=True)

    pid: str
    arrival: int
    burst: int
    completion: int
    turnaround: int
    waiting: int
    response: int


class SimulationMetricsSchema(BaseModel):
    """
    1. The use of class:
       Schema representing system-wide aggregate scheduling metrics.
    2. What the class contains:
       avg_waiting, avg_turnaround, avg_response, max_waiting, makespan, cpu_utilization,
       overhead_ticks, throughput, context_switches, and jain_fairness.
    3. Technologies used and why:
       Pydantic frozen model for standardized reporting.
    4. Key design and overall design of the class:
       Reflects domain SimulationMetrics summary attributes.
    """

    model_config = ConfigDict(frozen=True)

    avg_waiting: float
    avg_turnaround: float
    avg_response: float
    max_waiting: int
    makespan: int
    cpu_utilization: float
    overhead_ticks: int
    throughput: float
    context_switches: int
    jain_fairness: float
    ai_overhead_ticks: int = 0
    decisions: int = 0
    inference_us_total: float = 0.0
    effective_makespan: int = 0
    effective_turnaround: float = 0.0


class TimelineSegmentSchema(BaseModel):
    """
    1. The use of class:
       Schema representing an execution interval [start, end) on the CPU timeline.
    2. What the class contains:
       label (PID or 'CS' or 'IDLE'), start tick, end tick, and duration.
    3. Technologies used and why:
       Pydantic frozen model for Gantt rendering serialization.
    4. Key design and overall design of the class:
       Provides half-open time interval endpoints for frontend timeline visualizers.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    start: int
    end: int
    duration: int


class ReadyQueueLogEntry(BaseModel):
    """
    1. The use of class:
       Schema logging ready queue state at a discrete simulation tick for UI playback.
    2. What the class contains:
       - time: Discrete integer tick.
       - running_pid: Currently executing process ID, or None if idle / context-switching.
       - ready_pids: List of process IDs waiting in the ready queue.
    3. Technologies used and why:
       Pydantic model supporting dynamic playback sliders and step-through debuggers.
    4. Key design and overall design of the class:
       Decouples visual playback scrubbing from continuous simulation re-execution.
    """

    model_config = ConfigDict(frozen=True)

    time: int
    running_pid: str | None
    ready_pids: list[str]


class DecisionTraceEntry(BaseModel):
    """
    1. The use of class:
       Schema logging an RL agent decision step for explainability and inspection.
    2. What the class contains:
       time, state representation, valid action names, Q-values, chosen action, PID, reward.
    3. Technologies used and why:
       Pydantic model for AI explainability panels in the dashboard.
    4. Key design and overall design of the class:
       Enables step-by-step inspection of tabular Q-learning value estimates and action choices.
    """

    model_config = ConfigDict(frozen=True)

    time: int
    state: list[int]
    valid_actions: list[str]
    q_values: dict[str, float]
    chosen_action: str
    chosen_pid: str
    reward: float


class SimulateResponse(BaseModel):
    """
    1. The use of class:
       Complete response envelope for /api/simulate endpoint.
    2. What the class contains:
       algorithm, quantum, cs_cost, process_metrics, simulation_metrics, timeline,
       ready_queue_log, and optional decision_trace for RL policies.
    3. Technologies used and why:
       Pydantic v2 schema for unified API responses.
    4. Key design and overall design of the class:
       Comprehensive payload satisfying table views, Gantt charts, and RL explainability.
    """

    model_config = ConfigDict(frozen=True)

    algorithm: str
    quantum: int
    cs_cost: int
    process_metrics: list[ProcessMetricsSchema]
    simulation_metrics: SimulationMetricsSchema
    timeline: list[TimelineSegmentSchema]
    ready_queue_log: list[ReadyQueueLogEntry]
    decision_trace: list[DecisionTraceEntry] | None = None


class CompareRequest(BaseModel):
    """
    1. The use of class:
       Request payload for /api/compare endpoint comparing multiple policies.
    2. What the class contains:
       - processes: Workload processes (1 to 50).
       - algorithms: Optional list of algorithm names (defaults to all available).
       - quantum: Round Robin time quantum.
       - cs_cost: Context switch cost.
    3. Technologies used and why:
       Pydantic v2 model_validator enforcing unique PIDs.
    4. Key design and overall design of the class:
       Executes comparative benchmarking in a single client round-trip.
    """

    processes: list[ProcessSchema] = Field(..., min_length=1, max_length=50)
    algorithms: list[str] | None = Field(
        None, description="Algorithms to compare (default: all available)"
    )
    quantum: int = Field(2, ge=1, le=50)
    cs_cost: int = Field(0, ge=0, le=10)
    tick_us: float | None = Field(
        None, ge=0.001, le=100000.0, description="Optional microseconds per tick"
    )

    @model_validator(mode="after")
    def validate_unique_pids(self) -> "CompareRequest":
        """
        1. The use of function:
           Validates that processes in compare request have unique PIDs.
        2. How does the specific function work:
           Asserts set length matches list length.
        3. Concepts used in the function:
           Pydantic after-validator.
        4. Places where this function is imported and used:
           Invoked on CompareRequest initialization.
        5. Final output of the function, before and after:
           Before: Unverified process list.
           After: Validated request or raises ValueError.
        """
        pids = [p.pid for p in self.processes]
        if len(pids) != len(set(pids)):
            raise ValueError("Process PIDs in workload must be strictly unique.")
        return self


class CompareResponse(BaseModel):
    """
    1. The use of class:
       Response envelope for /api/compare endpoint.
    2. What the class contains:
       - results: Dictionary mapping policy name to full SimulateResponse.
       - comparison_summary: List of tabular row summaries for easy dashboard rendering.
    3. Technologies used and why:
       Pydantic model serializing multi-algorithm benchmark comparisons.
    4. Key design and overall design of the class:
       Facilitates multi-policy comparative tables and stacked Gantt charts in frontend.
    """

    results: dict[str, SimulateResponse]
    comparison_summary: list[dict[str, Any]]


class RecommendRequest(BaseModel):
    """
    1. The use of class:
       Request payload for /api/recommend endpoint.
    2. What the class contains:
       processes: Workload processes (1 to 50) with unique PIDs.
    3. Technologies used and why:
       Pydantic model with validation hook.
    4. Key design and overall design of the class:
       Input contract for Decision Tree policy classification.
    """

    processes: list[ProcessSchema] = Field(..., min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_unique_pids(self) -> "RecommendRequest":
        """
        1. The use of function:
           Ensures unique PIDs in recommend request.
        2. How does the specific function work:
           Checks set length equals list length.
        3. Concepts used in the function:
           Pydantic validator.
        4. Places where this function is imported and used:
           Invoked on RecommendRequest validation.
        5. Final output of the function, before and after:
           Before: Unverified processes.
           After: Confirmed unique PIDs or ValueError.
        """
        pids = [p.pid for p in self.processes]
        if len(pids) != len(set(pids)):
            raise ValueError("Process PIDs in workload must be strictly unique.")
        return self


class RecommendResponse(BaseModel):
    """
    1. The use of class:
       Response payload for /api/recommend endpoint.
    2. What the class contains:
       - algorithm: Recommended winning policy string ('FCFS', 'SJF', 'RR').
       - probabilities: Class probability distribution dictionary.
       - features: 12 workload-derived numerical feature values.
    3. Technologies used and why:
       Pydantic frozen model for supervised inference output.
    4. Key design and overall design of the class:
       Exposes prediction and explainability features for the frontend AI Lab.
    """

    model_config = ConfigDict(frozen=True)

    algorithm: str
    probabilities: dict[str, float]
    features: dict[str, float]


class GenerateWorkloadRequest(BaseModel):
    """
    1. The use of class:
       Request payload for /api/workloads/generate endpoint.
    2. What the class contains:
       - scenario: Scenario name string (e.g. 'uniform', 'burst_heavy', 'interactive').
       - n: Process count (1 to 50).
       - seed: Non-negative random seed.
    3. Technologies used and why:
       Pydantic model bounding workload parameters.
    4. Key design and overall design of the class:
       Standardized contract for on-demand synthetic workload synthesis.
    """

    scenario: str = Field(..., description="Workload scenario name")
    n: int = Field(5, ge=1, le=50, description="Process count (1 to 50)")
    seed: int = Field(42, ge=0, description="Random seed")


class GenerateWorkloadResponse(BaseModel):
    """
    1. The use of class:
       Response payload for /api/workloads/generate endpoint.
    2. What the class contains:
       workload_id, scenario, seed, and generated processes list.
    3. Technologies used and why:
       Pydantic model returning synthesized workload.
    4. Key design and overall design of the class:
       Reproducible workload data transfer object for simulator and editor.
    """

    model_config = ConfigDict(frozen=True)

    workload_id: str
    scenario: str
    seed: int
    processes: list[ProcessSchema]


class HealthResponse(BaseModel):
    """
    1. The use of class:
       Response schema for /api/health endpoint.
    2. What the class contains:
       status ('ok'), version, models status dict, and artifacts status dict.
    3. Technologies used and why:
       Pydantic model for backend health probing and UI status dot coloring.
    4. Key design and overall design of the class:
       Allows frontend to distinguish between full service, model missing (503), or offline.
    """

    model_config = ConfigDict(frozen=True)

    status: str
    version: str
    models: dict[str, bool]
    artifacts: dict[str, bool]


class ErrorResponse(BaseModel):
    """
    1. The use of class:
       Uniform error response schema across all API routes.
    2. What the class contains:
       error (short error identifier string) and detail (human-readable explanation).
    3. Technologies used and why:
       Pydantic model enforcing consistent API error contracts.
    4. Key design and overall design of the class:
       Standardizes error handling across HTTP 400, 422, 500, and 503 responses.
    """

    model_config = ConfigDict(frozen=True)

    error: str
    detail: str


class FinalComparisonRowSchema(BaseModel):
    """
    1. The use of class:
       Schema representing an individual policy row in final benchmark comparison matrix.
    2. What the class contains:
       policy name, raw waiting, turnaround, response, effective turnaround and makespan,
       fairness, context switches, decisions, inference latency, overhead %, gains, and W/T/L.
    3. Technologies used and why:
       Pydantic v2 frozen model for immutable serialization.
    4. Key design and overall design of the class:
       Exposes held-out test split evaluation metrics with 95% bootstrap confidence intervals.
    """

    model_config = ConfigDict(frozen=True)

    policy: str
    avg_waiting: str | float | None = None
    avg_turnaround: str | float | None = None
    avg_response: str | float | None = None
    effective_avg_turnaround: str | float | None = None
    effective_makespan: str | float | None = None
    max_waiting: str | float | None = None
    jain_fairness: str | float | None = None
    context_switches: str | float | None = None
    decisions: str | float | None = None
    inference_us_total: str | float | None = None
    overhead_pct_makespan: str | float | None = None
    gain_over_best_classical: str | float | None = None
    gap_to_oracle: str | float | None = None
    win_tie_loss: str | None = None


class FinalComparisonResponse(BaseModel):
    """
    1. The use of class:
       Response envelope for /api/results/final endpoint containing Setting S0 and S1 tables.
    2. What the class contains:
       S0 list of FinalComparisonRowSchema, S1 list of FinalComparisonRowSchema, and aliases.
    3. Technologies used and why:
       Pydantic v2 model for structured multi-setting benchmark reporting.
    4. Key design and overall design of the class:
       Returns full final comparison tables for the dashboard and external evaluators.
    """

    model_config = ConfigDict(frozen=True)

    S0: list[FinalComparisonRowSchema] = Field(default_factory=list)
    S1: list[FinalComparisonRowSchema] = Field(default_factory=list)
    s0: list[FinalComparisonRowSchema] | None = None
    s1: list[FinalComparisonRowSchema] | None = None


__all__ = [
    "CompareRequest",
    "CompareResponse",
    "DecisionTraceEntry",
    "ErrorResponse",
    "FinalComparisonResponse",
    "FinalComparisonRowSchema",
    "GenerateWorkloadRequest",
    "GenerateWorkloadResponse",
    "HealthResponse",
    "ProcessMetricsSchema",
    "ProcessSchema",
    "ReadyQueueLogEntry",
    "RecommendRequest",
    "RecommendResponse",
    "SimulateRequest",
    "SimulateResponse",
    "SimulationMetricsSchema",
    "TimelineSegmentSchema",
]
