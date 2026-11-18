/**
 * @file src/components/sim/AlgorithmPicker.tsx
 * 1. What this file does:
 *    Renders the CPU algorithm configuration panel: algorithm selection, time quantum input,
 *    context switch cost toggle, simulation run trigger, and ML recommendation banner.
 * 2. Overall design:
 *    Provides clean UI controls for FCFS, SJF, SRTF, Round Robin, and Q-Learning.
 *    Shows a Decision Tree recommendation badge with confidence scores and one-click application.
 * 3. Elements in the file:
 *    - `ALGORITHM_OPTIONS`: metadata list for supported policies
 *    - `AlgorithmPickerProps`: TypeScript interface for component props
 *    - `AlgorithmPicker`: React functional component
 * 4. Final output of the file:
 *    Exported component mounted in SimulatorPage.
 */

import React from 'react';
import { Cpu, Play, Sparkles, Loader2, Clock, Zap } from 'lucide-react';
import type { RecommendResponse } from '@/api/client';

export interface AlgorithmOption {
  id: string;
  name: string;
  description: string;
  isAI?: boolean;
}

export const ALGORITHM_OPTIONS: readonly AlgorithmOption[] = [
  { id: 'FCFS', name: 'FCFS', description: 'First-Come First-Served' },
  { id: 'SJF', name: 'SJF', description: 'Shortest Job First (non-preemptive)' },
  { id: 'SRTF', name: 'SRTF', description: 'Shortest Remaining Time First' },
  { id: 'RR', name: 'Round Robin', description: 'Time-sliced preemptive' },
  { id: 'QLEARNING', name: 'Q-Learning', description: 'Tabular RL agent', isAI: true },
];

export interface AlgorithmPickerProps {
  algorithm: string;
  onAlgorithmChange: (algo: string) => void;
  quantum: number;
  onQuantumChange: (q: number) => void;
  csCost: number;
  onCsCostChange: (cost: number) => void;
  onRunSimulation: () => void;
  isSimulating?: boolean;
  recommendation?: RecommendResponse | null;
  isLoadingRecommendation?: boolean;
  error?: string | null;
}

/**
 * Renders the scheduling algorithm selection and execution configuration controls.
 *
 * 1. The use of function:
 *    Configures simulation execution parameters and triggers the simulation run.
 * 2. How does the specific function work:
 *    Displays radio/button selectors for algorithms, conditionally reveals quantum input for RR,
 *    toggles context switch penalty (0 vs 1), and renders the Decision Tree recommendation banner.
 * 3. Concepts used in the function:
 *    Controlled form inputs, conditional rendering, recommendation badges, accessible buttons.
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Unconfigured execution request.
 *    After: Interactive control panel with live algorithm parameters and execution trigger.
 *
 * @param props - Component properties conforming to AlgorithmPickerProps.
 * @returns JSX Element for the algorithm picker.
 */
export const AlgorithmPicker: React.FC<AlgorithmPickerProps> = ({
  algorithm,
  onAlgorithmChange,
  quantum,
  onQuantumChange,
  csCost,
  onCsCostChange,
  onRunSimulation,
  isSimulating = false,
  recommendation = null,
  isLoadingRecommendation = false,
  error = null,
}) => {
  const isRR = algorithm.toUpperCase() === 'RR';
  const recAlgo = recommendation?.algorithm;
  const recProb = recAlgo && recommendation?.probabilities?.[recAlgo]
    ? Math.round(recommendation.probabilities[recAlgo] * 100)
    : null;

  return (
    <div className="p-5 rounded-card border border-border bg-surface shadow-subtle space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <h3 className="font-semibold text-sm text-foreground flex items-center gap-2">
          <Cpu className="w-4 h-4 text-primary" />
          Algorithm & Simulation Settings
        </h3>
        <span className="text-xs text-foreground-muted font-mono">
          Policy: {algorithm}
        </span>
      </div>

      {/* Decision Tree Recommendation Banner */}
      {recommendation && recAlgo && (
        <div className="p-3 rounded-control border border-primary/30 bg-primary/5 flex items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-primary flex-shrink-0" />
            <div>
              <span className="font-medium text-foreground">
                ML Recommendation:
              </span>{' '}
              <span className="font-bold text-primary font-mono">{recAlgo}</span>
              {recProb !== null && (
                <span className="text-foreground-muted ml-1 font-mono">
                  ({recProb}% confidence)
                </span>
              )}
            </div>
          </div>
          {algorithm.toUpperCase() !== recAlgo.toUpperCase() ? (
            <button
              type="button"
              onClick={() => onAlgorithmChange(recAlgo)}
              className="px-2.5 py-1 rounded bg-primary text-white font-medium hover:bg-primary-hover transition-colors flex-shrink-0"
            >
              Use Recommended
            </button>
          ) : (
            <span className="px-2 py-0.5 rounded bg-success/10 text-success text-[11px] font-semibold">
              Selected ✓
            </span>
          )}
        </div>
      )}

      {isLoadingRecommendation && !recommendation && (
        <div className="p-2.5 rounded-control border border-border bg-surface-muted flex items-center gap-2 text-xs text-foreground-muted">
          <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
          Evaluating workload with Decision Tree...
        </div>
      )}

      {/* Algorithm Selection Grid */}
      <div className="space-y-2">
        <label className="text-xs font-medium text-foreground block">
          Scheduling Policy
        </label>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {ALGORITHM_OPTIONS.map((opt) => {
            const isSelected = algorithm.toLowerCase() === opt.id.toLowerCase();
            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => onAlgorithmChange(opt.id)}
                className={`p-3 rounded-control border text-left transition-all flex flex-col justify-between ${
                  isSelected
                    ? 'border-primary bg-primary/10 shadow-sm'
                    : 'border-border bg-surface hover:bg-surface-muted'
                }`}
              >
                <div className="flex items-center justify-between w-full">
                  <span className="font-semibold text-xs text-foreground font-mono">
                    {opt.name}
                  </span>
                  {opt.isAI && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-600 dark:text-purple-400 font-bold uppercase">
                      AI / RL
                    </span>
                  )}
                </div>
                <span className="text-[11px] text-foreground-muted mt-1 leading-tight line-clamp-1">
                  {opt.description}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Context Switch and Round Robin Options */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
        {/* Time Quantum (RR only) */}
        <div className={`space-y-1.5 ${!isRR ? 'opacity-50 pointer-events-none' : ''}`}>
          <label
            htmlFor="time-quantum-input"
            className="text-xs font-medium text-foreground flex items-center justify-between"
          >
            <span className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-primary" />
              Time Quantum (q)
            </span>
            <span className="text-[11px] text-foreground-muted font-mono">{quantum} ticks</span>
          </label>
          <input
            id="time-quantum-input"
            type="number"
            min={1}
            max={50}
            value={quantum}
            disabled={!isRR}
            onChange={(e) => {
              const val = parseInt(e.target.value, 10);
              onQuantumChange(Number.isNaN(val) ? 1 : Math.max(1, Math.min(50, val)));
            }}
            className="w-full px-3 py-1.5 rounded-control border border-border bg-surface text-foreground font-mono text-xs focus:outline-none focus:ring-1 focus:ring-primary"
          />
        </div>

        {/* Context Switch Cost Toggle */}
        <div className="space-y-1.5">
          <label className="text-xs font-medium text-foreground flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-primary" />
              Context Switch Cost
            </span>
            <span className="text-[11px] text-foreground-muted font-mono">{csCost} tick(s)</span>
          </label>
          <div className="grid grid-cols-2 gap-1.5">
            <button
              type="button"
              onClick={() => onCsCostChange(0)}
              className={`py-1.5 px-2 rounded-control border text-xs font-mono font-medium transition-colors ${
                csCost === 0
                  ? 'border-primary bg-primary text-white'
                  : 'border-border bg-surface-muted text-foreground hover:bg-surface'
              }`}
            >
              0 (Ideal)
            </button>
            <button
              type="button"
              onClick={() => onCsCostChange(1)}
              className={`py-1.5 px-2 rounded-control border text-xs font-mono font-medium transition-colors ${
                csCost === 1
                  ? 'border-primary bg-primary text-white'
                  : 'border-border bg-surface-muted text-foreground hover:bg-surface'
              }`}
            >
              1 (Overhead)
            </button>
          </div>
        </div>
      </div>

      {/* Error banner */}
      {error && (
        <div className="p-2.5 rounded-control border border-danger/30 bg-danger/10 text-danger text-xs">
          {error}
        </div>
      )}

      {/* Run Simulation Action */}
      <button
        type="button"
        disabled={isSimulating}
        onClick={onRunSimulation}
        className="w-full py-2.5 px-4 rounded-control bg-primary text-white font-semibold text-xs tracking-wide uppercase shadow-sm hover:bg-primary-hover disabled:opacity-60 transition-all flex items-center justify-center gap-2"
      >
        {isSimulating ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            Executing Simulation...
          </>
        ) : (
          <>
            <Play className="w-4 h-4 fill-white" />
            Run Simulation
          </>
        )}
      </button>
    </div>
  );
};
