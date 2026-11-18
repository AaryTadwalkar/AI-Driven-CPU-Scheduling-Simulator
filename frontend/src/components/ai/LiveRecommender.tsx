/**
 * @file src/components/ai/LiveRecommender.tsx
 * 1. What this file does:
 *    Provides an interactive live testbed for the Decision Tree workload policy recommender.
 * 2. Overall design:
 *    Allows engineers to input or load processes from the active simulator workload, executes
 *    supervised inference via /api/recommend, displays the predicted algorithm badge, confidence
 *    score, class probability bars (FCFS, SJF, RR), and inspects all 12 extracted structural features.
 * 3. Elements in the file:
 *    - `LiveRecommenderProps`: Component props and optional callbacks
 *    - `LiveRecommender`: Primary React functional component
 * 4. Final output of the file:
 *    Interactive policy recommendation card consumed by AILabPage.
 */

import React, { useState, useEffect } from 'react';
import {
  Sparkles,
  Play,
  RotateCcw,
  ArrowRight,
  Plus,
  Trash2,
  ListFilter,
  AlertCircle,
} from 'lucide-react';
import type { ProcessSchema } from '@/api/client';
import { useRecommendMutation } from '@/api/hooks';
import { useWorkloadStore } from '@/store/workloadStore';
import { getAlgorithmColor } from '@/lib/colors';
import { formatPercent, formatNumber } from '@/lib/format';
import { formatFeatureLabel } from './FeatureImportance';

export interface LiveRecommenderProps {
  onApplyAlgorithm?: (algorithm: string) => void;
  className?: string;
}

const SAMPLE_WORKLOAD: ProcessSchema[] = [
  { pid: 'P1', arrival: 0, burst: 8, priority: 0 },
  { pid: 'P2', arrival: 1, burst: 4, priority: 0 },
  { pid: 'P3', arrival: 2, burst: 9, priority: 0 },
  { pid: 'P4', arrival: 3, burst: 5, priority: 0 },
];

/**
 * Interactive Decision Tree Live Recommender testbed component.
 *
 * 1. The use of function:
 *    Allows interactive testing of the Decision Tree model against custom or active workloads.
 * 2. How does the specific function work:
 *    Maintains local process state with options to load from central `workloadStore`. On submission,
 *    triggers `useRecommendMutation` to obtain predicted class, probabilities, and extracted features.
 * 3. Concepts used in the function:
 *    Supervised inference testbed, local state editing, class distribution bar visualization.
 * 4. Places where this function is imported and used:
 *    Imported by AILabPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Unchecked workload array.
 *    After: Interactive inference card with predicted algorithm badge, probability bars, and feature list.
 *
 * @param props - Component parameters including apply callback.
 * @returns JSX Element.
 */
export const LiveRecommender: React.FC<LiveRecommenderProps> = ({
  onApplyAlgorithm,
  className = '',
}) => {
  const activeSimulatorProcesses = useWorkloadStore((s) => s.processes);
  const [localProcesses, setLocalProcesses] = useState<ProcessSchema[]>(SAMPLE_WORKLOAD);
  const [validationError, setValidationError] = useState<string | null>(null);

  const recommendMutation = useRecommendMutation();

  /**
   * Dispatches recommendation mutation on component mount or button press.
   */
  const handleRunRecommendation = () => {
    if (localProcesses.length === 0) {
      setValidationError('Add at least one process to run recommendation.');
      return;
    }

    // Validate burst >= 1
    for (const p of localProcesses) {
      if (p.burst < 1 || isNaN(p.burst)) {
        setValidationError(`Process ${p.pid} burst must be an integer >= 1.`);
        return;
      }
      if (p.arrival < 0 || isNaN(p.arrival)) {
        setValidationError(`Process ${p.pid} arrival must be an integer >= 0.`);
        return;
      }
    }

    setValidationError(null);
    recommendMutation.mutate({ processes: localProcesses });
  };

  // Run initial recommendation with sample workload on mount
  useEffect(() => {
    recommendMutation.mutate({ processes: SAMPLE_WORKLOAD });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLoadFromSimulator = () => {
    if (activeSimulatorProcesses && activeSimulatorProcesses.length > 0) {
      setLocalProcesses([...activeSimulatorProcesses]);
      setValidationError(null);
      recommendMutation.mutate({ processes: activeSimulatorProcesses });
    }
  };

  const handleResetToSample = () => {
    setLocalProcesses(SAMPLE_WORKLOAD);
    setValidationError(null);
    recommendMutation.mutate({ processes: SAMPLE_WORKLOAD });
  };

  const handleUpdateProcess = (index: number, patch: Partial<ProcessSchema>) => {
    setLocalProcesses((prev) => {
      const next = [...prev];
      next[index] = { ...next[index], ...patch };
      return next;
    });
  };

  const handleAddProcess = () => {
    setLocalProcesses((prev) => {
      const nextNum = prev.length + 1;
      const newPid = `P${nextNum}`;
      const lastArrival = prev.length > 0 ? prev[prev.length - 1].arrival + 1 : 0;
      return [...prev, { pid: newPid, arrival: lastArrival, burst: 4, priority: 0 }];
    });
  };

  const handleRemoveProcess = (index: number) => {
    setLocalProcesses((prev) => prev.filter((_, i) => i !== index));
  };

  const result = recommendMutation.data;
  const isPending = recommendMutation.isPending;
  const predictedAlgo = result?.algorithm;
  const probabilities = result?.probabilities || {};
  const features = result?.features || {};

  // Find confidence (highest probability)
  const confidence = predictedAlgo && probabilities[predictedAlgo] !== undefined
    ? probabilities[predictedAlgo]
    : 0;

  return (
    <div
      className={`p-5 rounded-card border border-border bg-surface shadow-subtle flex flex-col space-y-5 ${className}`}
      data-testid="live-recommender"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-primary" />
            Decision Tree Live Policy Recommender
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Test the trained Decision Tree classifier in real time. Predicts winning meta-rule (FCFS, SJF, RR) from 12 workload features.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-center">
          <button
            type="button"
            onClick={handleLoadFromSimulator}
            className="px-2.5 py-1 text-xs rounded-control border border-border bg-surface hover:bg-surface-muted text-foreground transition-colors"
          >
            Load from Simulator ({activeSimulatorProcesses.length})
          </button>
          <button
            type="button"
            onClick={handleResetToSample}
            className="px-2 py-1 text-xs rounded-control border border-border bg-surface hover:bg-surface-muted text-foreground-muted hover:text-foreground"
            title="Reset to default sample"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {validationError && (
        <div className="p-3 rounded-control bg-danger/10 border border-danger/30 text-danger text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{validationError}</span>
        </div>
      )}

      {/* Grid: Process Editor & Recommendation Result */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: Interactive Mini Process Editor */}
        <div className="lg:col-span-5 space-y-3">
          <div className="flex items-center justify-between text-xs font-semibold text-foreground">
            <span>Workload Processes ({localProcesses.length})</span>
            <button
              type="button"
              onClick={handleAddProcess}
              className="inline-flex items-center gap-1 text-[11px] text-primary hover:underline"
            >
              <Plus className="w-3 h-3" /> Add Process
            </button>
          </div>

          <div className="max-h-60 overflow-y-auto space-y-1.5 pr-1">
            {localProcesses.map((p, idx) => (
              <div
                key={p.pid || idx}
                className="flex items-center gap-2 p-2 rounded-control bg-surface-muted/40 border border-border text-xs"
              >
                <input
                  type="text"
                  value={p.pid}
                  onChange={(e) => handleUpdateProcess(idx, { pid: e.target.value })}
                  className="w-12 px-1.5 py-1 font-mono font-semibold rounded border border-border bg-surface text-foreground"
                  aria-label={`Process PID ${idx + 1}`}
                />
                <div className="flex items-center gap-1 flex-1">
                  <span className="text-[10px] text-foreground-muted">Arr:</span>
                  <input
                    type="number"
                    min={0}
                    value={p.arrival}
                    onChange={(e) =>
                      handleUpdateProcess(idx, { arrival: Math.max(0, parseInt(e.target.value, 10) || 0) })
                    }
                    className="w-14 px-1.5 py-1 font-mono rounded border border-border bg-surface text-foreground"
                    aria-label={`Arrival time for ${p.pid}`}
                  />
                </div>
                <div className="flex items-center gap-1 flex-1">
                  <span className="text-[10px] text-foreground-muted">Burst:</span>
                  <input
                    type="number"
                    min={1}
                    value={p.burst}
                    onChange={(e) =>
                      handleUpdateProcess(idx, { burst: Math.max(1, parseInt(e.target.value, 10) || 1) })
                    }
                    className="w-14 px-1.5 py-1 font-mono rounded border border-border bg-surface text-foreground"
                    aria-label={`Burst time for ${p.pid}`}
                  />
                </div>
                {localProcesses.length > 1 && (
                  <button
                    type="button"
                    onClick={() => handleRemoveProcess(idx)}
                    className="p-1 text-foreground-muted hover:text-danger rounded"
                    aria-label={`Remove process ${p.pid}`}
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            ))}
          </div>

          <button
            type="button"
            onClick={handleRunRecommendation}
            disabled={isPending}
            className="w-full py-2 px-3 text-xs font-semibold rounded-control bg-primary hover:bg-primary-hover text-white flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            {isPending ? 'Classifying Workload...' : 'Recommend Optimal Policy'}
          </button>
        </div>

        {/* Right Column: Prediction, Probabilities & Apply */}
        <div className="lg:col-span-7 flex flex-col justify-between space-y-4 p-4 rounded-control border border-border bg-surface-muted/20">
          <div>
            <div className="text-xs font-semibold text-foreground-muted mb-2">
              Model Prediction
            </div>

            {recommendMutation.isError ? (
              <div className="p-3 rounded bg-danger/10 border border-danger/20 text-danger text-xs">
                Prediction failed: {(recommendMutation.error as Error)?.message || 'Service unavailable'}
              </div>
            ) : predictedAlgo ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-3">
                    <span
                      style={{ backgroundColor: getAlgorithmColor(predictedAlgo) }}
                      className="px-3 py-1 rounded text-white font-mono font-bold text-sm shadow-xs"
                    >
                      {predictedAlgo}
                    </span>
                    <div>
                      <div className="text-xs font-semibold text-foreground">
                        Recommended Policy
                      </div>
                      <div className="text-[11px] text-foreground-muted font-mono">
                        Confidence: <strong className="text-foreground">{formatPercent(confidence, 1)}</strong>
                      </div>
                    </div>
                  </div>

                  {onApplyAlgorithm && (
                    <button
                      type="button"
                      onClick={() => onApplyAlgorithm(predictedAlgo)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-control bg-surface border border-border hover:bg-surface-muted text-foreground transition-colors"
                    >
                      <span>Apply to Simulator</span>
                      <ArrowRight className="w-3.5 h-3.5 text-primary" />
                    </button>
                  )}
                </div>

                {/* Probability Distribution Bars */}
                <div className="space-y-2 pt-2 border-t border-border">
                  <div className="text-[11px] font-semibold text-foreground-muted">
                    Class Probability Distribution
                  </div>
                  {(['SJF', 'RR', 'FCFS'] as const).map((algo) => {
                    const prob = probabilities[algo] ?? 0;
                    const pct = prob * 100;
                    const isWinner = algo === predictedAlgo;
                    const barColor = getAlgorithmColor(algo);

                    return (
                      <div key={algo} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span
                            className={`font-mono font-medium ${
                              isWinner ? 'text-foreground font-bold' : 'text-foreground-muted'
                            }`}
                          >
                            {algo}
                          </span>
                          <span className="font-mono text-xs text-foreground font-semibold">
                            {formatPercent(prob, 1)}
                          </span>
                        </div>
                        <div className="h-2 w-full rounded-full bg-surface-muted overflow-hidden">
                          <div
                            style={{
                              width: `${Math.max(2, pct)}%`,
                              backgroundColor: barColor,
                            }}
                            className="h-full rounded-full transition-all duration-300"
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="py-6 text-center text-xs text-foreground-muted">
                Run recommendation to classify workload.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Extracted 12 Structural Features Grid */}
      {Object.keys(features).length > 0 && (
        <div className="pt-3 border-t border-border space-y-2">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
            <ListFilter className="w-3.5 h-3.5 text-primary" />
            <span>Extracted Workload Structural Features (12-Vector)</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
            {Object.entries(features).map(([key, val]) => (
              <div
                key={key}
                className="p-2 rounded-control border border-border bg-surface-muted/30 text-xs space-y-0.5"
                title={`${formatFeatureLabel(key)}: ${val}`}
              >
                <div className="text-[10px] text-foreground-muted truncate">
                  {formatFeatureLabel(key)}
                </div>
                <div className="font-mono font-semibold text-foreground text-xs">
                  {formatNumber(val, 2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default LiveRecommender;
