/**
 * @file src/pages/ComparePage.tsx
 * 1. What this file does:
 *    Renders the comprehensive multi-algorithm comparison dashboard page (Sub-stage F4).
 * 2. Overall design:
 *    Presents workload metadata header, configurable benchmarking controls (context switch
 *    cost 0/1, Round Robin quantum, policy selection checkboxes), and dispatches `/api/compare`.
 *    Renders ComparisonTable with best-per-metric badges, ComparisonBars grouped charts,
 *    and synchronized GanttStack timelines. Implements skeleton loading, empty state,
 *    and friendly HTTP 503 model missing diagnostics.
 * 3. Elements in the file:
 *    - `ALGORITHM_OPTIONS`: Available scheduling algorithms for multi-policy evaluation
 *    - `ComparePage`: Main React functional page component
 * 4. Final output of the file:
 *    Complete algorithm benchmarking view mounted at route '/compare'.
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Link } from 'react-router-dom';
import {
  BarChart3,
  Play,
  RotateCcw,
  Sliders,
  Cpu,
  ArrowRight,
} from 'lucide-react';
import { useWorkloadStore } from '@/store/workloadStore';
import { useCompareMutation } from '@/api/hooks';
import type { ComparisonCsvRow } from '@/lib/csv';
import { ComparisonTable } from '@/components/compare/ComparisonTable';
import { ComparisonBars } from '@/components/compare/ComparisonBars';
import { GanttStack } from '@/components/compare/GanttStack';
import { EmptyState } from '@/components/common/EmptyState';
import { ErrorState } from '@/components/common/ErrorState';
import { TableSkeleton, ChartSkeleton, GanttStackSkeleton } from '@/components/common/Skeletons';

const ALGORITHM_OPTIONS = [
  { id: 'FCFS', label: 'FCFS' },
  { id: 'SJF', label: 'SJF (Non-preemptive)' },
  { id: 'SRTF', label: 'SRTF (Preemptive SJF)' },
  { id: 'RR', label: 'Round Robin (RR)' },
  { id: 'QLEARNING', label: 'Q-Learning RL Agent' },
  { id: 'DT', label: 'Decision Tree (Category 1)' },
  { id: 'RF', label: 'Random Forest (Category 1)' },
  { id: 'PID_GBDT', label: 'GBDT Ranker (Category 3)' },
  { id: 'PID_NN', label: 'Neural Scorer (Category 3)' },
] as const;

/**
 * Multi-policy comparative benchmarking page component.
 *
 * 1. The use of function:
 *    Hosts the comparative performance analysis suite on route '/compare'.
 * 2. How does the specific function work:
 *    Binds to current workloadStore, executes batch simulation via useCompareMutation,
 *    and orchestrates ComparisonTable, ComparisonBars, and GanttStack subviews.
 * 3. Concepts used in the function:
 *    Auto-execution on mount, TanStack Query mutation management, conditional state rendering.
 * 4. Places where this function is imported and used:
 *    Imported by App.tsx and routes.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Static placeholder scaffold.
 *    After: Interactive multi-algorithm benchmarking dashboard.
 *
 * @returns JSX Element.
 */
export const ComparePage: React.FC = () => {
  const { processes } = useWorkloadStore();
  const compareMutation = useCompareMutation();

  // Benchmark parameters
  const [csCost, setCsCost] = useState<number>(0);
  const [quantum, setQuantum] = useState<number>(2);
  const [selectedAlgos, setSelectedAlgos] = useState<string[]>([
    'FCFS',
    'SJF',
    'SRTF',
    'RR',
    'QLEARNING',
  ]);

  // Track initial auto-run execution to avoid re-triggering on unrelated re-renders
  const hasAutoRunRef = useRef<boolean>(false);

  // Workload summary metrics
  const totalBurst = useMemo(
    () => processes.reduce((acc, p) => acc + p.burst, 0),
    [processes]
  );
  const maxArrival = useMemo(
    () => (processes.length > 0 ? Math.max(...processes.map((p) => p.arrival)) : 0),
    [processes]
  );

  /**
   * Executes the multi-policy comparison mutation on current parameters.
   *
   * 1. The use of function:
   *    Dispatches POST /api/compare with selected algorithms, quantum, and CS cost.
   * 2. How does the specific function work:
   *    Constructs CompareRequest payload and invokes compareMutation.mutate.
   * 3. Concepts used in the function:
   *    Mutation execution, async state handling.
   * 4. Places where this function is imported and used:
   *    Auto-run effect and "Run All Comparison" button click handler.
   * 5. Final output of the function, before and after:
   *    Before: Mutation idle.
   *    After: Mutation pending or resolved with benchmark results.
   */
  const handleRunComparison = () => {
    if (processes.length === 0 || selectedAlgos.length === 0) return;

    compareMutation.mutate({
      processes,
      algorithms: selectedAlgos,
      quantum,
      cs_cost: csCost,
    });
  };

  // Auto-execute comparison on initial mount if valid workload exists
  useEffect(() => {
    if (
      !hasAutoRunRef.current &&
      processes.length > 0 &&
      !compareMutation.data &&
      !compareMutation.isPending
    ) {
      hasAutoRunRef.current = true;
      handleRunComparison();
    }
  }, [processes.length]);

  /**
   * Toggles inclusion of an algorithm in the candidate comparison set.
   *
   * 1. The use of function:
   *    Allows user to select a subset of algorithms to benchmark.
   * 2. How does the specific function work:
   *    Adds algorithm ID if absent; removes if present (as long as >= 1 remains).
   * 3. Concepts used in the function:
   *    State immutability, array filtering.
   * 4. Places where this function is imported and used:
   *    Checkbox onChange handlers in the controls toolbar.
   * 5. Final output of the function, before and after:
   *    Before: Selected algorithms set.
   *    After: Updated selected algorithms set.
   *
   * @param id - Algorithm code string.
   */
  const toggleAlgorithm = (id: string) => {
    if (selectedAlgos.includes(id)) {
      if (selectedAlgos.length > 1) {
        setSelectedAlgos(selectedAlgos.filter((a) => a !== id));
      }
    } else {
      setSelectedAlgos([...selectedAlgos, id]);
    }
  };

  // Transform comparison summary rows into typed records
  const comparisonRows: ComparisonCsvRow[] = useMemo(() => {
    if (!compareMutation.data?.comparison_summary) return [];
    return (compareMutation.data.comparison_summary as Record<string, unknown>[]).map((row) => ({
      algorithm: String(row.algorithm || ''),
      avg_waiting: Number(row.avg_waiting || 0),
      avg_turnaround: Number(row.avg_turnaround || 0),
      avg_response: Number(row.avg_response || 0),
      cpu_utilization: Number(row.cpu_utilization || 0),
      jain_fairness: Number(row.jain_fairness || 0),
      context_switches: Number(row.context_switches || 0),
    }));
  }, [compareMutation.data]);

  return (
    <div className="space-y-6">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-primary" />
            Algorithm Comparison
          </h1>
          <p className="text-sm text-foreground-muted mt-1">
            Compare all classical and reinforcement learning algorithms side-by-side on identical workloads.
          </p>
        </div>

        {/* Workload Brief Pill */}
        <div className="flex items-center gap-3 p-2.5 rounded-card border border-border bg-surface shadow-subtle self-start sm:self-auto text-xs">
          <div className="flex items-center gap-1.5 text-foreground font-semibold">
            <Cpu className="w-4 h-4 text-primary" />
            <span>Active Workload:</span>
          </div>
          <div className="font-mono text-foreground-muted flex items-center gap-2">
            <span className="font-semibold text-foreground">{processes.length}</span> processes
            <span>•</span>
            <span className="font-semibold text-foreground">{totalBurst}</span> burst ticks
            <span>•</span>
            arrival ≤ <span className="font-semibold text-foreground">{maxArrival}</span>
          </div>
          <Link
            to="/"
            className="ml-2 inline-flex items-center gap-1 text-primary hover:underline font-medium"
            title="Modify workload in editor"
          >
            <span>Edit</span>
            <ArrowRight className="w-3 h-3" />
          </Link>
        </div>
      </div>

      {/* Controls Toolbar */}
      <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-4">
        <div className="flex items-center gap-2 text-xs font-semibold text-foreground pb-2 border-b border-border">
          <Sliders className="w-4 h-4 text-primary" />
          <span>Benchmark Execution Parameters</span>
        </div>

        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          {/* Algorithm Checkboxes */}
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs font-medium text-foreground-muted">Algorithms:</span>
            {ALGORITHM_OPTIONS.map((opt) => {
              const isChecked = selectedAlgos.includes(opt.id);
              return (
                <label
                  key={opt.id}
                  className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-control text-xs font-medium cursor-pointer border transition-colors select-none ${
                    isChecked
                      ? 'bg-primary/10 border-primary/30 text-primary'
                      : 'bg-surface border-border text-foreground-muted hover:text-foreground'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => toggleAlgorithm(opt.id)}
                    className="rounded text-primary focus:ring-primary h-3.5 w-3.5"
                  />
                  <span>{opt.label}</span>
                </label>
              );
            })}
          </div>

          {/* Context Switch and Quantum Inputs */}
          <div className="flex flex-wrap items-center gap-4">
            {/* Context Switch Cost Toggle */}
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-foreground-muted">
                Context Switch Cost:
              </span>
              <div className="flex items-center p-0.5 rounded-control bg-surface-muted border border-border text-xs">
                <button
                  type="button"
                  onClick={() => setCsCost(0)}
                  className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors ${
                    csCost === 0
                      ? 'bg-surface text-primary shadow-xs font-semibold'
                      : 'text-foreground-muted hover:text-foreground'
                  }`}
                  title="Zero context switch penalty (pure scheduling)"
                >
                  0 ticks
                </button>
                <button
                  type="button"
                  onClick={() => setCsCost(1)}
                  className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors ${
                    csCost === 1
                      ? 'bg-surface text-primary shadow-xs font-semibold'
                      : 'text-foreground-muted hover:text-foreground'
                  }`}
                  title="1-tick context switch overhead penalty"
                >
                  1 tick
                </button>
              </div>
            </div>

            {/* RR Quantum Input */}
            <div className="flex items-center gap-1.5">
              <label
                htmlFor="quantum-input"
                className="text-xs font-medium text-foreground-muted"
              >
                RR Quantum (q):
              </label>
              <input
                id="quantum-input"
                type="number"
                min={1}
                max={50}
                value={quantum}
                onChange={(e) => setQuantum(Math.max(1, parseInt(e.target.value, 10) || 1))}
                className="w-16 px-2 py-1 rounded-control text-xs font-mono border border-border bg-surface text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
              />
            </div>

            {/* Run Button */}
            <button
              type="button"
              onClick={handleRunComparison}
              disabled={compareMutation.isPending || processes.length === 0}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-control text-xs font-semibold bg-primary text-white hover:bg-primary-hover transition-colors shadow-sm disabled:opacity-50"
            >
              {compareMutation.isPending ? (
                <>
                  <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                  <span>Simulating Policies...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Run All Comparison</span>
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area: Empty, Error, Loading, or Data views */}
      {processes.length === 0 ? (
        <EmptyState
          title="No Workload Configured"
          description="A process workload is required to benchmark scheduling algorithms. Visit the Simulator page to load a canonical preset or synthesize synthetic processes."
          action={{
            label: 'Open Workload Editor',
            onClick: () => {
              window.location.hash = '#/';
            },
          }}
        />
      ) : compareMutation.isPending ? (
        <div className="space-y-6">
          <TableSkeleton rows={selectedAlgos.length || 5} />
          <ChartSkeleton />
          <GanttStackSkeleton rows={selectedAlgos.length || 5} />
        </div>
      ) : compareMutation.error ? (
        <ErrorState
          title="Comparison Benchmark Failed"
          error={compareMutation.error}
          onRetry={handleRunComparison}
        />
      ) : compareMutation.data ? (
        <div className="space-y-6">
          {/* Comparison Table */}
          <ComparisonTable rows={comparisonRows} />

          {/* Comparison Bar Charts */}
          <ComparisonBars rows={comparisonRows} />

          {/* Synchronized Stacked Gantt Charts */}
          <GanttStack results={compareMutation.data.results} />
        </div>
      ) : (
        <EmptyState
          title="Ready to Benchmark"
          description="Click 'Run All Comparison' above to simulate all candidate scheduling algorithms on the active workload."
          action={{
            label: 'Run Comparison',
            onClick: handleRunComparison,
          }}
        />
      )}
    </div>
  );
};

export default ComparePage;
