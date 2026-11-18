"""
1. What this file does:
   Initializes the domain package and exports core entities like Process, Timeline, Metrics,
   Engine, Views, and simulation helpers.
2. Overall design:
   The domain module encapsulates core operating systems scheduling abstractions including
   workload process entities, discrete timeline execution tracking, metric evaluation, and
   the unit-step simulation engine.
3. Elements in the file:
   - CS_LABEL: Label constant for context switch intervals ("CS").
   - IDLE_LABEL: Label constant for CPU idle intervals ("IDLE").
   - Engine: Central discrete unit-step CPU scheduling simulation engine.
   - Process: Immutable process definition with input validation.
   - ProcessMetrics: Immutable per-process scheduling metrics record.
   - ProcView: Immutable process state snapshot for scheduling policies.
   - ScriptedPolicy: Pre-scheduled sequence or callable testing policy.
   - SimResult: Immutable container of process metrics, simulation metrics, and timeline.
   - SimulationMetrics: Immutable system-wide aggregate simulation metrics record.
   - StepInfo: Result tuple of single engine step (penalty, view, done).
   - Timeline: Execution history aggregator with segment merging and partition validation.
   - TimelineSegment: Immutable half-open execution interval.
   - View: Observable system state snapshot at discrete tick.
   - calculate_jain_fairness: Standalone calculator for Jain's fairness index over slowdowns.
   - calculate_process_metrics: Function deriving turnaround, waiting, and response times.
   - calculate_simulation_metrics: Function deriving aggregate simulation benchmark metrics.
   - run: Simulation loop helper executing a policy over a workload.
   - validate_processes: Workload validation helper enforcing collection invariants.
4. Final output of the file:
   Enables direct importing of domain models, constants, engine entities, and metric utilities
   from schedsim.domain.
"""

from schedsim.domain.engine import (
    Engine,
    ProcView,
    ScriptedPolicy,
    SimResult,
    StepInfo,
    View,
    run,
)
from schedsim.domain.metrics import (
    ProcessMetrics,
    SimulationMetrics,
    calculate_jain_fairness,
    calculate_process_metrics,
    calculate_simulation_metrics,
)
from schedsim.domain.process import Process, validate_processes
from schedsim.domain.timeline import CS_LABEL, IDLE_LABEL, Timeline, TimelineSegment

__all__ = [
    "CS_LABEL",
    "IDLE_LABEL",
    "Engine",
    "Process",
    "ProcessMetrics",
    "ProcView",
    "ScriptedPolicy",
    "SimResult",
    "SimulationMetrics",
    "StepInfo",
    "Timeline",
    "TimelineSegment",
    "View",
    "calculate_jain_fairness",
    "calculate_process_metrics",
    "calculate_simulation_metrics",
    "run",
    "validate_processes",
]
