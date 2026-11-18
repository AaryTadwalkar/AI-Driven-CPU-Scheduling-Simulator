/**
 * @file src/components/workload/GeneratorForm.tsx
 * 1. What this file does:
 *    Renders the synthetic workload generator form allowing users to parameterize and trigger
 *    random workload synthesis across all 9 canonical workload scenarios with seed reproducibility.
 * 2. Overall design:
 *    Connects to the backend `/api/workloads/generate` endpoint via TanStack Query mutation.
 *    Features scenario selection, process count slider (1-50), seed input with randomize dice,
 *    and displays an alert confirming the active seed for experiment reproducibility.
 * 3. Elements in the file:
 *    - `SCENARIOS`: Constant metadata cataloging all 9 workload scenarios with user descriptions
 *    - `GeneratorForm`: React functional component
 *    - Process count slider with live numeric badge
 *    - Seed input with instant randomize generator
 *    - "Generate Workload" mutation action button with spinner
 * 4. Final output of the file:
 *    Exported form component embedded within WorkloadEditor.
 */

import React, { useState } from 'react';
import { useGenerateWorkloadMutation } from '@/api/hooks';
import { useWorkloadStore } from '@/store/workloadStore';
import { Dices, Sparkles, Loader2, CheckCircle2, AlertCircle } from 'lucide-react';

export const SCENARIOS: readonly {
  id: string;
  name: string;
  description: string;
}[] = [
  {
    id: 'short_jobs',
    name: 'Short Jobs',
    description: 'Quick interactive jobs (bursts 1-4, low variance)',
  },
  {
    id: 'long_jobs',
    name: 'Long Jobs',
    description: 'Compute-intensive batch workloads (bursts 12-25)',
  },
  {
    id: 'mixed',
    name: 'Mixed Workload',
    description: 'Uniform mix of short interactive and long batch processes',
  },
  {
    id: 'bimodal',
    name: 'Bimodal',
    description: '80% short jobs (burst 1-3) and 20% long jobs (burst 15-25)',
  },
  {
    id: 'simultaneous',
    name: 'Simultaneous Arrivals',
    description: 'All processes arrive at tick 0 competing at once',
  },
  {
    id: 'staggered',
    name: 'Staggered Arrivals',
    description: 'Processes arrive with periodic spacing gaps',
  },
  {
    id: 'convoy',
    name: 'Convoy Effect',
    description: 'One long CPU-bound process followed by rapid short processes',
  },
  {
    id: 'equal_bursts',
    name: 'Equal Bursts',
    description: 'Identical CPU burst lengths with staggered arrivals',
  },
  {
    id: 'high_load',
    name: 'High Load',
    description: 'Dense arrival rates resulting in deep ready queue contention',
  },
];

/**
 * Workload generation form component providing scenario configuration and seeded synthesis.
 *
 * 1. The use of function:
 *    Enables rapid generation of diverse test workloads matching backend probability models.
 * 2. How does the specific function work:
 *    Captures scenario, process count (1-50), and seed from store; fires `useGenerateWorkloadMutation`;
 *    and on success updates the store's process list while confirming the seed used.
 * 3. Concepts used in the function:
 *    React Query mutations, reproducible pseudorandom seed generation, input range binding.
 * 4. Places where this function is imported and used:
 *    Imported by WorkloadEditor.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Previous workload in store.
 *    After: Synthesized workload populated in store with confirmation notification.
 *
 * @returns JSX Element for workload synthesis.
 */
export const GeneratorForm: React.FC = () => {
  const scenario = useWorkloadStore((s) => s.activeScenario);
  const setScenario = useWorkloadStore((s) => s.setScenario);
  const seed = useWorkloadStore((s) => s.seed);
  const setSeed = useWorkloadStore((s) => s.setSeed);
  const nProcesses = useWorkloadStore((s) => s.nProcesses);
  const setNProcesses = useWorkloadStore((s) => s.setNProcesses);
  const setProcesses = useWorkloadStore((s) => s.setProcesses);

  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const generateMutation = useGenerateWorkloadMutation();

  const handleRandomizeSeed = () => {
    const newSeed = Math.floor(Math.random() * 900000) + 100000;
    setSeed(newSeed);
    setToastMessage(null);
  };

  const handleGenerate = () => {
    setToastMessage(null);
    generateMutation.mutate(
      { scenario, n: nProcesses, seed },
      {
        onSuccess: (data) => {
          setProcesses(data.processes, null);
          setToastMessage(
            `Generated ${data.processes.length} processes for '${data.scenario}' using seed ${data.seed}`
          );
        },
      }
    );
  };

  return (
    <div className="space-y-4">
      {/* Scenario selector */}
      <div>
        <label
          htmlFor="scenario-select"
          className="text-xs font-semibold uppercase tracking-wider text-foreground-muted block mb-1.5"
        >
          Scenario Distribution
        </label>
        <select
          id="scenario-select"
          aria-label="Scenario Distribution"
          value={scenario}
          onChange={(e) => setScenario(e.target.value)}
          className="w-full bg-surface-muted text-foreground border border-border rounded-control px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary font-sans transition-colors cursor-pointer"
        >
          {SCENARIOS.map((sc) => (
            <option key={sc.id} value={sc.id}>
              {sc.name} — {sc.description}
            </option>
          ))}
        </select>
      </div>

      {/* Process count slider */}
      <div>
        <div className="flex items-center justify-between mb-1.5">
          <label
            htmlFor="process-count-slider"
            className="text-xs font-semibold uppercase tracking-wider text-foreground-muted"
          >
            Process Count (N)
          </label>
          <span className="font-mono text-xs font-bold text-primary px-2 py-0.5 bg-primary/10 rounded-control">
            {nProcesses} processes
          </span>
        </div>
        <input
          id="process-count-slider"
          aria-label="Process Count"
          type="range"
          min={1}
          max={50}
          step={1}
          value={nProcesses}
          onChange={(e) => setNProcesses(parseInt(e.target.value, 10))}
          className="w-full h-2 bg-surface-muted rounded-lg appearance-none cursor-pointer accent-primary"
        />
        <div className="flex justify-between text-[11px] text-foreground-muted font-mono mt-1">
          <span>1</span>
          <span>25</span>
          <span>50 max</span>
        </div>
      </div>

      {/* Seed input with randomize dice */}
      <div>
        <label
          htmlFor="seed-input"
          className="text-xs font-semibold uppercase tracking-wider text-foreground-muted block mb-1.5"
        >
          Random Seed (for reproducibility)
        </label>
        <div className="flex items-center gap-2">
          <input
            id="seed-input"
            aria-label="Random Seed"
            type="number"
            min={0}
            max={2147483647}
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
            className="flex-1 bg-surface-muted text-foreground border border-border rounded-control px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary"
          />
          <button
            type="button"
            onClick={handleRandomizeSeed}
            title="Generate random seed"
            className="px-3 py-2 bg-surface-muted hover:bg-surface text-foreground border border-border rounded-control text-xs font-medium flex items-center gap-1.5 transition-colors"
          >
            <Dices className="w-4 h-4 text-primary" />
            <span>Random</span>
          </button>
        </div>
      </div>

      {/* Action button */}
      <div>
        <button
          type="button"
          onClick={handleGenerate}
          disabled={generateMutation.isPending}
          className="w-full py-2.5 px-4 bg-primary hover:bg-primary-hover text-white rounded-control text-sm font-semibold flex items-center justify-center gap-2 transition-colors shadow-subtle disabled:opacity-60 disabled:cursor-not-allowed"
        >
          {generateMutation.isPending ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Generating Workload...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              <span>Generate Workload</span>
            </>
          )}
        </button>
      </div>

      {/* Seed reproducibility toast message */}
      {toastMessage && (
        <div className="p-2.5 bg-success/10 border border-success/30 rounded-control text-xs text-success flex items-start gap-2">
          <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Mutation error banner */}
      {generateMutation.isError && (
        <div className="p-2.5 bg-danger/10 border border-danger/30 rounded-control text-xs text-danger flex items-start gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>
            {generateMutation.error?.detail ||
              generateMutation.error?.message ||
              'Workload generation failed. Check API connectivity.'}
          </span>
        </div>
      )}
    </div>
  );
};

export default GeneratorForm;
