/**
 * @file src/components/results/ScenarioBreakdown.tsx
 * 1. What this file does:
 *    Renders scenario-stratified performance analysis across all 9 synthetic workload regimes.
 * 2. Overall design:
 *    Provides an interactive scenario selector (pill buttons / dropdown), scenario characteristics
 *    description, algorithm performance ranking table, and a Recharts bar chart comparing waiting
 *    times with context-switch cost toggles and CSV export.
 * 3. Elements in the file:
 *    - `SCENARIO_METADATA`: Metadata dictionary detailing description and characteristics for all 9 scenarios
 *    - `ScenarioBreakdownItem`: Record type for scenario breakdown data from /api/results/summary
 *    - `ScenarioBreakdownProps`: Interface for component props and customization
 *    - `ScenarioBreakdown`: Primary React functional component
 * 4. Final output of the file:
 *    Interactive scenario exploration and benchmark visualization card consumed by ResultsPage.
 */

import React, { useState, useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Layers, Download, CheckCircle, Info } from 'lucide-react';
import { getAlgorithmColor } from '@/lib/colors';
import { downloadScenarioBreakdownCsv, type ScenarioBreakdownCsvRow } from '@/lib/csv';
import { formatNumber } from '@/lib/format';

export interface ScenarioBreakdownItem {
  cs_cost: number;
  scenario: string;
  policy: string;
  avg_waiting: number;
}

export interface ScenarioBreakdownProps {
  scenarios?: ScenarioBreakdownItem[];
  className?: string;
}

export interface ScenarioInfo {
  name: string;
  label: string;
  description: string;
  characteristics: string;
}

export const SCENARIO_METADATA: Record<string, ScenarioInfo> = {
  short_jobs: {
    name: 'short_jobs',
    label: 'Short Jobs',
    description: 'Rapid completions with bursts drawn uniformly from [1, 4].',
    characteristics: 'Low convoy vulnerability; SRTF and SJF complete jobs with minimal queuing.',
  },
  long_jobs: {
    name: 'long_jobs',
    label: 'Long Jobs',
    description: 'Compute-intensive jobs with large bursts drawn from [15, 30].',
    characteristics: 'High execution times; preemption penalties become prominent at cs_cost=1.',
  },
  mixed: {
    name: 'mixed',
    label: 'Mixed Jobs',
    description: '70% short jobs (1–4) and 30% long compute jobs (12–25).',
    characteristics: 'High burst coefficient of variation; classic benchmark favoring SJF/SRTF.',
  },
  bimodal: {
    name: 'bimodal',
    label: 'Bimodal',
    description: 'Distinct two-cluster distribution with extreme small and large job sizes.',
    characteristics: 'Strongly polarizes FIFO vs SJF policies; tests queue reordering effectiveness.',
  },
  simultaneous: {
    name: 'simultaneous',
    label: 'Simultaneous Arrivals',
    description: 'All processes arrive synchronously at tick t=0.',
    characteristics: 'Maximal initial ready queue contention; demonstrates pure algorithm tie-breaking.',
  },
  staggered: {
    name: 'staggered',
    label: 'Staggered Arrivals',
    description: 'Processes arrive sequentially in regular spaced intervals.',
    characteristics: 'Smooth pipeline execution; lowers queue congestion and context switch overhead.',
  },
  convoy: {
    name: 'convoy',
    label: 'Convoy Effect',
    description: 'One long job arrives at t=0 followed immediately by a burst of short jobs.',
    characteristics: 'Devastates non-preemptive FCFS; SJF and Round Robin dramatically outperform.',
  },
  equal_bursts: {
    name: 'equal_bursts',
    label: 'Equal Bursts',
    description: 'All processes have identical burst lengths.',
    characteristics: 'Isolates pure arrival order and quantum behavior; eliminates SJF variance advantage.',
  },
  high_load: {
    name: 'high_load',
    label: 'High Load',
    description: 'Dense arrival rate approaching or exceeding single-core CPU throughput.',
    characteristics: 'Queue lengths build up rapidly; tests starvation resilience and preemption cost trade-offs.',
  },
};

const ALL_SCENARIOS = [
  'short_jobs',
  'long_jobs',
  'mixed',
  'bimodal',
  'simultaneous',
  'staggered',
  'convoy',
  'equal_bursts',
  'high_load',
];

/**
 * Scenario Breakdown Explorer component.
 *
 * 1. The use of function:
 *    Enables users to examine how different scheduling algorithms perform across 9 distinct synthetic workload regimes.
 * 2. How does the specific function work:
 *    Filters data by active scenario and context-switch cost. Renders ranked algorithm list,
 *    scenario characteristics panel, Recharts comparative bar chart, and CSV export.
 * 3. Concepts used in the function:
 *    Scenario stratification, comparative bar visualization, performance ranking, CSV download.
 * 4. Places where this function is imported and used:
 *    Imported by ResultsPage.tsx and test suites.
 * 5. Final output of the function, before and after:
 *    Before: Array of ScenarioBreakdownItem records.
 *    After: Interactive multi-scenario explorer with bar charts and rankings.
 *
 * @param props - Scenario items and styling classes.
 * @returns JSX Element.
 */
export const ScenarioBreakdown: React.FC<ScenarioBreakdownProps> = ({
  scenarios = [],
  className = '',
}) => {
  const [selectedScenario, setSelectedScenario] = useState('bimodal');
  const [selectedCost, setSelectedCost] = useState<0 | 1>(0);

  // Filter records by selected scenario and cost
  const filteredRecords = useMemo(() => {
    return scenarios
      .filter(
        (r) =>
          r.scenario === selectedScenario &&
          Number(r.cs_cost) === selectedCost
      )
      .sort((a, b) => a.avg_waiting - b.avg_waiting);
  }, [scenarios, selectedScenario, selectedCost]);

  const activeMeta = SCENARIO_METADATA[selectedScenario] || {
    name: selectedScenario,
    label: selectedScenario,
    description: 'Synthetic workload scenario.',
    characteristics: 'Comparative performance evaluation.',
  };

  const handleDownloadCsv = () => {
    const csvRows: ScenarioBreakdownCsvRow[] = scenarios.map((r) => ({
      cs_cost: r.cs_cost,
      scenario: r.scenario,
      policy: r.policy,
      avg_waiting: r.avg_waiting,
    }));

    downloadScenarioBreakdownCsv(csvRows, 'scenario_breakdown.csv');
  };

  return (
    <div
      className={`p-5 rounded-card border border-border bg-surface shadow-subtle flex flex-col space-y-4 ${className}`}
      data-testid="scenario-breakdown"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Layers className="w-4 h-4 text-primary" />
            Performance Breakdown Across 9 Workload Scenarios
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Evaluate algorithm specialization and trade-offs under diverse synthetic arrival and burst distributions.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-center">
          {/* Cost toggle button */}
          <div className="inline-flex rounded-control p-1 bg-surface-muted border border-border">
            <button
              type="button"
              onClick={() => setSelectedCost(0)}
              className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                selectedCost === 0
                  ? 'bg-surface text-foreground shadow-xs font-semibold'
                  : 'text-foreground-muted hover:text-foreground'
              }`}
            >
              Cost = 0
            </button>
            <button
              type="button"
              onClick={() => setSelectedCost(1)}
              className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                selectedCost === 1
                  ? 'bg-surface text-foreground shadow-xs font-semibold'
                  : 'text-foreground-muted hover:text-foreground'
              }`}
            >
              Cost = 1
            </button>
          </div>

          <button
            type="button"
            onClick={handleDownloadCsv}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-control bg-surface border border-border hover:bg-surface-muted text-foreground transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-primary" />
            <span>Download Scenario CSV</span>
          </button>
        </div>
      </div>

      {/* 9 Scenario Selector Pills */}
      <div className="flex flex-wrap gap-1.5">
        {ALL_SCENARIOS.map((scenKey) => {
          const meta = SCENARIO_METADATA[scenKey];
          const isSelected = selectedScenario === scenKey;

          return (
            <button
              key={scenKey}
              type="button"
              onClick={() => setSelectedScenario(scenKey)}
              className={`px-3 py-1.5 text-xs rounded-control border transition-all ${
                isSelected
                  ? 'bg-primary text-white border-primary font-semibold shadow-xs'
                  : 'bg-surface border-border text-foreground hover:bg-surface-muted'
              }`}
            >
              {meta ? meta.label : scenKey}
            </button>
          );
        })}
      </div>

      {/* Scenario Description Banner */}
      <div className="p-3.5 rounded-control bg-surface-muted/40 border border-border space-y-1 text-xs">
        <div className="flex items-center gap-1.5 font-semibold text-foreground">
          <Info className="w-3.5 h-3.5 text-primary" />
          <span>{activeMeta.label} — {activeMeta.description}</span>
        </div>
        <p className="text-foreground-muted leading-relaxed">
          {activeMeta.characteristics}
        </p>
      </div>

      {/* Grid: Bar Chart & Algorithm Ranking */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 pt-1">
        {/* Left: Recharts Bar Chart */}
        <div className="lg:col-span-7 h-[280px]" data-testid="scenario-bar-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={filteredRecords}
              margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
            >
              <XAxis
                dataKey="policy"
                stroke="var(--text-muted)"
                fontSize={11}
                tickLine={false}
              />
              <YAxis
                stroke="var(--text-muted)"
                fontSize={11}
                tickFormatter={(v: number) => `${v.toFixed(0)}`}
                label={{
                  value: 'Avg Waiting (ticks)',
                  angle: -90,
                  position: 'insideLeft',
                  fontSize: 10,
                  fill: 'var(--text-muted)',
                }}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload as ScenarioBreakdownItem;
                    return (
                      <div className="p-2.5 rounded-control border border-border bg-surface shadow-subtle text-xs space-y-1">
                        <div className="font-semibold text-foreground">{data.policy}</div>
                        <div className="text-foreground">
                          Avg Waiting:{' '}
                          <strong className="font-mono text-primary">
                            {formatNumber(data.avg_waiting, 2)} ticks
                          </strong>
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="avg_waiting" radius={[4, 4, 0, 0]}>
                {filteredRecords.map((entry) => (
                  <Cell
                    key={`cell-${entry.policy}`}
                    fill={getAlgorithmColor(entry.policy)}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Right: Algorithm Performance Ranking */}
        <div className="lg:col-span-5 space-y-2">
          <div className="text-xs font-semibold text-foreground pb-1 border-b border-border">
            Algorithm Performance Ranking (Lower is Better)
          </div>

          <div className="space-y-1.5">
            {filteredRecords.map((item, index) => {
              const isFirst = index === 0;
              const algoColor = getAlgorithmColor(item.policy);

              return (
                <div
                  key={item.policy}
                  className={`flex items-center justify-between p-2 rounded-control border text-xs transition-colors ${
                    isFirst
                      ? 'border-emerald-500/30 bg-emerald-500/5 font-semibold'
                      : 'border-border bg-surface hover:bg-surface-muted/30'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-foreground-muted w-4">
                      #{index + 1}
                    </span>
                    <span
                      style={{ backgroundColor: algoColor }}
                      className="w-2.5 h-2.5 rounded-full shrink-0"
                    />
                    <span className="font-mono text-foreground font-medium">
                      {item.policy}
                    </span>
                    {isFirst && (
                      <span className="inline-flex items-center gap-0.5 px-1.5 py-0.2 rounded text-[9px] font-semibold bg-emerald-500/10 text-emerald-500">
                        <CheckCircle className="w-2.5 h-2.5" /> Best
                      </span>
                    )}
                  </div>
                  <span className="font-mono font-bold text-foreground">
                    {formatNumber(item.avg_waiting, 2)} ticks
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScenarioBreakdown;
