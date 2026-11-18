/**
 * @file src/pages/SimulatorPage.tsx
 * 1. What this file does:
 *    Renders the unified CPU Scheduling Simulator workbench: integrates WorkloadEditor,
 *    AlgorithmPicker, MetricCards, GanttChart, PlaybackControls, ReadyQueueView,
 *    ProcessMetricsTable, and Q-Learning DecisionTrace.
 * 2. Overall design:
 *    Manages the simulation execution cycle via TanStack React Query mutations (`/api/simulate`
 *    and `/api/recommend`). Automatically runs an initial simulation on the Section 4.6
 *    canonical fixture on mount. Coordinates live playhead scrubbing with the Zustand playbackStore.
 * 3. Elements in the file:
 *    - `SimulatorPage`: Main React functional page component
 *    - Initial simulation auto-trigger on mount
 *    - Debounced Decision Tree recommendation query on workload change
 *    - 503 model-unavailable callout handling
 * 4. Final output of the file:
 *    Primary interactive simulator page mounted at route '/' in the application router.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Cpu, AlertTriangle, AlertCircle } from 'lucide-react';
import { WorkloadEditor } from '@/components/workload/WorkloadEditor';
import { AlgorithmPicker } from '@/components/sim/AlgorithmPicker';
import { GanttChart } from '@/components/sim/GanttChart';
import { PlaybackControls } from '@/components/sim/PlaybackControls';
import { MetricCards } from '@/components/sim/MetricCards';
import { ProcessMetricsTable } from '@/components/sim/ProcessMetricsTable';
import { ReadyQueueView } from '@/components/sim/ReadyQueueView';
import { DecisionTrace } from '@/components/ai/DecisionTrace';
import { useWorkloadStore } from '@/store/workloadStore';
import { usePlaybackStore } from '@/store/playbackStore';
import { useSimulateMutation, useRecommendMutation } from '@/api/hooks';
import type { SimulateResponse } from '@/api/client';

/**
 * Main simulator page component orchestrating workload configuration, scheduling, and timeline playback.
 *
 * 1. The use of function:
 *    Entry-point component for the CPU Scheduling Simulator page ('/').
 * 2. How does the specific function work:
 *    Synchronizes active workload from useWorkloadStore with algorithm parameters, executes
 *    simulation via useSimulateMutation, sets maxTick in playbackStore, and distributes
 *    simulation results to GanttChart, MetricCards, ReadyQueueView, and DecisionTrace.
 * 3. Concepts used in the function:
 *    React Query mutations, debounced effects, responsive CSS grid, playback state synchronization.
 * 4. Places where this function is imported and used:
 *    Imported by routes.tsx and App.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Standalone workload editor scaffold.
 *    After: Complete interactive simulation and discrete playback workbench.
 *
 * @returns JSX Element for the Simulator page.
 */
export const SimulatorPage: React.FC = () => {
  const processes = useWorkloadStore((s) => s.processes);
  const tick = usePlaybackStore((s) => s.tick);
  const setTick = usePlaybackStore((s) => s.setTick);
  const setMaxTick = usePlaybackStore((s) => s.setMaxTick);
  const resetPlayback = usePlaybackStore((s) => s.reset);

  const [algorithm, setAlgorithm] = useState<string>('FCFS');
  const [quantum, setQuantum] = useState<number>(2);
  const [csCost, setCsCost] = useState<number>(0);
  const [simulationData, setSimulationData] = useState<SimulateResponse | null>(null);

  const simulateMutation = useSimulateMutation();
  const recommendMutation = useRecommendMutation();

  const hasInitialRunRef = useRef<boolean>(false);

  /**
   * Executes CPU scheduling simulation against the active workload processes.
   *
   * 1. The use of function:
   *    Triggered by AlgorithmPicker or auto-run effect to generate scheduling output.
   * 2. How does the specific function work:
   *    Dispatches simulateMutation with processes, algorithm, quantum, and cs_cost,
   *    then updates simulationData, sets maxTick in playbackStore, and resets playhead to 0.
   * 3. Concepts used in the function:
   *    Mutation dispatch, callback state update, timeline range calibration.
   * 4. Places where this function is imported and used:
   *    Internal to SimulatorPage for run buttons and recommendation application.
   * 5. Final output of the function, before and after:
   *    Before: Stale or pending simulation state.
   *    After: Populated simulationData and updated playback limits.
   *
   * @param targetAlgo - Optional policy name to override the active state.
   */
  const handleRunSimulation = (targetAlgo?: string) => {
    if (processes.length === 0) return;
    const algoToRun = targetAlgo || algorithm;

    simulateMutation.mutate(
      {
        processes,
        algorithm: algoToRun,
        quantum,
        cs_cost: csCost,
      },
      {
        onSuccess: (data) => {
          setSimulationData(data);
          setMaxTick(data.simulation_metrics.makespan);
          resetPlayback();
        },
      }
    );
  };

  /**
   * Auto-run simulation on initial load with the canonical Section 4.6 default fixture.
   */
  useEffect(() => {
    if (!hasInitialRunRef.current && processes.length > 0) {
      hasInitialRunRef.current = true;
      handleRunSimulation('FCFS');
    }
  }, []);

  /**
   * Debounced Decision Tree policy recommendation on workload modifications (500ms delay).
   */
  useEffect(() => {
    if (processes.length === 0) return;
    const debounceTimer = setTimeout(() => {
      recommendMutation.mutate({ processes });
    }, 500);

    return () => clearTimeout(debounceTimer);
  }, [processes]);

  // Check for 503 Model Unavailable status (e.g. uncheckpointed Q-Learning model)
  const is503Error =
    simulateMutation.error?.status === 503 ||
    simulateMutation.error?.error === 'MODEL_UNAVAILABLE';

  return (
    <div className="space-y-6 pb-12">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <Cpu className="w-6 h-6 text-primary" />
            Simulator & Gantt Playback
          </h1>
          <p className="text-sm text-foreground-muted mt-1">
            Configure process workloads, select scheduling algorithms (FCFS, SJF, SRTF, RR,
            Q-Learning), and step through discrete execution ticks.
          </p>
        </div>
      </div>

      {/* Top Section: Workload Editor (2 cols) & Algorithm Settings (1 col) */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2">
          <WorkloadEditor />
        </div>

        <div className="space-y-6">
          <AlgorithmPicker
            algorithm={algorithm}
            onAlgorithmChange={(newAlgo) => {
              setAlgorithm(newAlgo);
            }}
            quantum={quantum}
            onQuantumChange={setQuantum}
            csCost={csCost}
            onCsCostChange={setCsCost}
            onRunSimulation={() => handleRunSimulation()}
            isSimulating={simulateMutation.isPending}
            recommendation={recommendMutation.data}
            isLoadingRecommendation={recommendMutation.isPending}
          />
        </div>
      </div>

      {/* Error / 503 Model Missing Banner */}
      {simulateMutation.error && (
        <div
          className={`p-4 rounded-card border text-xs space-y-2 ${
            is503Error
              ? 'border-warning/40 bg-warning/10 text-warning-foreground'
              : 'border-danger/40 bg-danger/10 text-danger'
          }`}
        >
          <div className="flex items-center gap-2 font-semibold">
            {is503Error ? (
              <AlertTriangle className="w-4 h-4 text-warning" />
            ) : (
              <AlertCircle className="w-4 h-4 text-danger" />
            )}
            <span>
              {is503Error
                ? 'Model Checkpoint Unavailable (HTTP 503)'
                : 'Simulation Execution Error'}
            </span>
          </div>
          <p className="text-foreground leading-relaxed">
            {is503Error
              ? 'The trained Q-learning policy checkpoint was not found. Please train and checkpoint the tabular RL agent by executing `python -m src.cli train-rl` in your terminal, or switch to classical algorithms (FCFS, SJF, SRTF, RR).'
              : simulateMutation.error.detail ||
                simulateMutation.error.message ||
                'An error occurred during simulation.'}
          </p>
        </div>
      )}

      {/* Simulation Results Section */}
      <div className="space-y-6">
        {/* Metric Cards KPI Summary */}
        <MetricCards
          metrics={simulationData?.simulation_metrics ?? null}
          isLoading={simulateMutation.isPending}
        />

        {/* Gantt Chart Timeline View */}
        <GanttChart
          timeline={simulationData?.timeline ?? []}
          makespan={simulationData?.simulation_metrics?.makespan ?? 0}
          playhead={tick}
          onSeek={setTick}
          title={`CPU Execution Timeline — ${simulationData?.algorithm ?? algorithm}`}
          algorithmName={simulationData?.algorithm ?? algorithm}
        />

        {/* Playback Transport & Scrubbing Controls */}
        <PlaybackControls />

        {/* Bottom Grid: Ready Queue View (1 col) & Process Metrics Table (2 cols) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <ReadyQueueView
              readyQueueLog={simulationData?.ready_queue_log ?? []}
              currentTick={tick}
            />
          </div>

          <div className="lg:col-span-2">
            <ProcessMetricsTable
              processMetrics={simulationData?.process_metrics ?? []}
            />
          </div>
        </div>

        {/* Reinforcement Learning Explainability Decision Trace */}
        {simulationData?.decision_trace && simulationData.decision_trace.length > 0 && (
          <DecisionTrace
            trace={simulationData.decision_trace}
            currentTick={tick}
            onSelectTick={setTick}
          />
        )}
      </div>
    </div>
  );
};

export default SimulatorPage;
