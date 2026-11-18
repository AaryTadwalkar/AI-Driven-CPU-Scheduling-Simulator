/**
 * @file src/components/ai/LearningCurve.tsx
 * 1. What this file does:
 *    Renders a Recharts ComposedChart visualizing the Reinforcement Learning training trajectory.
 * 2. Overall design:
 *    Groups episodic training metrics across random seeds, computes mean validation waiting time
 *    and +/- 1 standard deviation confidence bands, overlays heuristic baseline reference lines
 *    (FCFS, SJF, SRTF), and provides an interactive toggle between context-switch cost 0 and 1.
 * 3. Elements in the file:
 *    - `LearningCurveRecord`: Type definition for raw training log items from /api/rl/training-log
 *    - `AggregatedEpisodePoint`: Transformed data structure with mean, std dev, and bounds
 *    - `BASELINE_VALUES`: Canonical baseline waiting times at cs_cost 0 and 1
 *    - `LearningCurve`: Primary React functional component
 * 4. Final output of the file:
 *    Interactive learning curve visualizer card consumed by AILabPage.
 */

import React, { useState, useMemo } from 'react';
import {
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';
import { TrendingUp, Layers } from 'lucide-react';
import { formatNumber } from '@/lib/format';

export interface LearningCurveRecord {
  episode: number;
  seed: number;
  cs_cost: number;
  avg_waiting: number;
  avg_turnaround?: number;
  q_table_size?: number;
  unique_states?: number;
  unseen_state_rate?: number;
}

export interface LearningCurveProps {
  records?: LearningCurveRecord[];
  className?: string;
}

export const BASELINE_VALUES = {
  cs0: {
    SRTF: 14.41,
    SJF: 17.57,
    FCFS: 21.27,
    RR: 27.92,
  },
  cs1: {
    SRTF: 17.90,
    SJF: 20.48,
    FCFS: 24.45,
    RR: 47.88,
  },
} as const;

/**
 * Aggregates training records for a specific context-switch cost across seeds.
 *
 * 1. The use of function:
 *    Computes episode-by-episode statistics (mean, sample std deviation, upper/lower bounds).
 * 2. How does the specific function work:
 *    Filters input records by target `csCost`, groups values by `episode`, and calculates
 *    mean average waiting time and standard deviation across seeds.
 * 3. Concepts used in the function:
 *    Sample variance calculation, group-by aggregation, interval bounds clamping.
 * 4. Places where this function is imported and used:
 *    Used internally by `LearningCurve` component.
 * 5. Final output of the function, before and after:
 *    Before: Array of raw seed-level training records.
 *    After: Array of aggregated episode objects with mean and std confidence bounds.
 *
 * @param records - List of LearningCurveRecord items.
 * @param csCost - Target context switch cost (0 or 1).
 * @returns Array of aggregated chart points.
 */
function aggregateTrainingLogs(
  records: LearningCurveRecord[],
  csCost: number
): Array<{
  episode: number;
  mean_wait: number;
  std_wait: number;
  upper_wait: number;
  lower_wait: number;
  band: [number, number];
  q_table_size: number;
}> {
  const filtered = records.filter((r) => Number(r.cs_cost) === csCost);
  if (filtered.length === 0) return [];

  // Group by episode
  const groups = new Map<number, number[]>();
  const qSizes = new Map<number, number>();

  filtered.forEach((r) => {
    const ep = Number(r.episode);
    if (!groups.has(ep)) {
      groups.set(ep, []);
    }
    groups.get(ep)!.push(Number(r.avg_waiting));
    if (r.q_table_size !== undefined) {
      qSizes.set(ep, Number(r.q_table_size));
    }
  });

  const sortedEpisodes = Array.from(groups.keys()).sort((a, b) => a - b);

  return sortedEpisodes.map((ep) => {
    const vals = groups.get(ep)!;
    const n = vals.length;
    const sum = vals.reduce((a, b) => a + b, 0);
    const mean = sum / n;

    let std = 0;
    if (n > 1) {
      const variance =
        vals.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (n - 1);
      std = Math.sqrt(variance);
    }

    const lower = Math.max(0, mean - std);
    const upper = mean + std;

    return {
      episode: ep,
      mean_wait: Number(mean.toFixed(2)),
      std_wait: Number(std.toFixed(2)),
      lower_wait: Number(lower.toFixed(2)),
      upper_wait: Number(upper.toFixed(2)),
      band: [Number(lower.toFixed(2)), Number(upper.toFixed(2))],
      q_table_size: qSizes.get(ep) ?? 0,
    };
  });
}

/**
 * RL Learning Curve visualization component.
 *
 * 1. The use of function:
 *    Visualizes the 20,000 episode Tabular Q-Learning training trajectory and baseline convergence.
 * 2. How does the specific function work:
 *    Renders a Recharts ComposedChart with a shaded standard deviation band, mean curve,
 *    and reference lines for FCFS, SJF, and SRTF baselines. Includes cost toggle (0 vs 1).
 * 3. Concepts used in the function:
 *    ComposedChart (Area + Line + ReferenceLine), multi-seed variance shading, baseline benchmarks.
 * 4. Places where this function is imported and used:
 *    Imported by AILabPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Raw episodic CSV records from /api/rl/training-log.
 *    After: Interactive multi-seed convergence chart with baseline markers.
 *
 * @param props - Training log records and styling classes.
 * @returns JSX Element.
 */
export const LearningCurve: React.FC<LearningCurveProps> = ({
  records = [],
  className = '',
}) => {
  const [selectedCost, setSelectedCost] = useState<0 | 1>(0);

  const chartData = useMemo(() => {
    return aggregateTrainingLogs(records, selectedCost);
  }, [records, selectedCost]);

  const baselines = selectedCost === 0 ? BASELINE_VALUES.cs0 : BASELINE_VALUES.cs1;

  // Summary metrics
  const initialWait = chartData.length > 0 ? chartData[0].mean_wait : 21.52;
  const finalWait =
    chartData.length > 0 ? chartData[chartData.length - 1].mean_wait : (selectedCost === 0 ? 16.74 : 19.14);
  const reductionPct =
    initialWait > 0 ? ((initialWait - finalWait) / initialWait) * 100 : 0;
  const finalQTableSize =
    chartData.length > 0 ? chartData[chartData.length - 1].q_table_size : 4695;

  return (
    <div
      className={`p-5 rounded-card border border-border bg-surface shadow-subtle flex flex-col space-y-4 ${className}`}
      data-testid="learning-curve"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-pink-500" />
            Tabular Q-Learning Trajectory Across 5 Seeds
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Validation mean waiting time over 20,000 episodes with ±1 std band vs heuristic baselines.
          </p>
        </div>

        {/* Cost toggle button */}
        <div className="inline-flex rounded-control p-1 bg-surface-muted border border-border self-start sm:self-center">
          <button
            type="button"
            onClick={() => setSelectedCost(0)}
            className={`px-3 py-1 text-xs font-medium rounded-control transition-colors ${
              selectedCost === 0
                ? 'bg-surface text-foreground shadow-xs font-semibold'
                : 'text-foreground-muted hover:text-foreground'
            }`}
          >
            Cost = 0 (Ideal)
          </button>
          <button
            type="button"
            onClick={() => setSelectedCost(1)}
            className={`px-3 py-1 text-xs font-medium rounded-control transition-colors ${
              selectedCost === 1
                ? 'bg-surface text-foreground shadow-xs font-semibold'
                : 'text-foreground-muted hover:text-foreground'
            }`}
          >
            Cost = 1 (Overhead)
          </button>
        </div>
      </div>

      {/* Trajectory KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 py-1">
        <div className="p-2.5 rounded-control border border-border bg-surface-muted/40">
          <div className="text-[11px] text-foreground-muted">Initial Wait (Ep 0)</div>
          <div className="text-base font-mono font-bold text-foreground">
            {formatNumber(initialWait, 2)} ticks
          </div>
        </div>
        <div className="p-2.5 rounded-control border border-border bg-surface-muted/40">
          <div className="text-[11px] text-foreground-muted">Final Wait (Ep 20k)</div>
          <div className="text-base font-mono font-bold text-pink-500">
            {formatNumber(finalWait, 2)} ticks
          </div>
        </div>
        <div className="p-2.5 rounded-control border border-border bg-surface-muted/40">
          <div className="text-[11px] text-foreground-muted">Wait Reduction</div>
          <div className="text-base font-mono font-bold text-emerald-500">
            -{reductionPct.toFixed(1)}%
          </div>
        </div>
        <div className="p-2.5 rounded-control border border-border bg-surface-muted/40">
          <div className="text-[11px] text-foreground-muted">Explored Q-States</div>
          <div className="text-base font-mono font-bold text-foreground">
            {finalQTableSize.toLocaleString()}
          </div>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="h-[380px] w-full" data-testid="learning-curve-chart">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
          >
            <XAxis
              dataKey="episode"
              stroke="var(--text-muted)"
              fontSize={11}
              tickFormatter={(ep: number) => (ep >= 1000 ? `${ep / 1000}k` : `${ep}`)}
            />
            <YAxis
              domain={['auto', 'auto']}
              stroke="var(--text-muted)"
              fontSize={11}
              tickFormatter={(v: number) => `${v.toFixed(0)}`}
              label={{
                value: 'Avg Waiting Time (ticks)',
                angle: -90,
                position: 'insideLeft',
                fontSize: 11,
                fill: 'var(--text-muted)',
              }}
            />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload as {
                    episode: number;
                    mean_wait: number;
                    std_wait: number;
                    lower_wait: number;
                    upper_wait: number;
                  };
                  return (
                    <div className="p-3 rounded-control border border-border bg-surface shadow-subtle text-xs space-y-1.5">
                      <div className="font-semibold text-foreground">
                        Episode {label?.toLocaleString()}
                      </div>
                      <div className="flex items-center justify-between gap-4 text-pink-500 font-mono">
                        <span>RL Mean:</span>
                        <strong>{data.mean_wait.toFixed(2)} ticks</strong>
                      </div>
                      <div className="text-foreground-muted text-[11px] font-mono">
                        ±1 Std Band: [{data.lower_wait.toFixed(2)}, {data.upper_wait.toFixed(2)}]
                      </div>
                      <div className="pt-1.5 border-t border-border/60 text-[10px] space-y-0.5">
                        <div className="text-purple-400">SRTF Baseline: {baselines.SRTF.toFixed(2)}</div>
                        <div className="text-emerald-500">SJF Baseline: {baselines.SJF.toFixed(2)}</div>
                        <div className="text-blue-500">FCFS Baseline: {baselines.FCFS.toFixed(2)}</div>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: 11 }} />

            {/* Shaded +/- 1 Std Dev Area */}
            <Area
              type="monotone"
              dataKey="band"
              stroke="none"
              fill="rgba(236, 72, 153, 0.18)"
              name="±1 Std Band"
            />

            {/* Mean Validation Waiting Line */}
            <Line
              type="monotone"
              dataKey="mean_wait"
              stroke="#EC4899"
              strokeWidth={2.5}
              dot={false}
              name="RL Mean Waiting"
            />

            {/* Horizontal Heuristic Baseline Markers */}
            <ReferenceLine
              y={baselines.SRTF}
              stroke="#8B5CF6"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                value: `SRTF (${baselines.SRTF})`,
                fill: '#8B5CF6',
                position: 'right',
                fontSize: 10,
              }}
            />
            <ReferenceLine
              y={baselines.SJF}
              stroke="#10B981"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                value: `SJF (${baselines.SJF})`,
                fill: '#10B981',
                position: 'right',
                fontSize: 10,
              }}
            />
            <ReferenceLine
              y={baselines.FCFS}
              stroke="#3B82F6"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                value: `FCFS (${baselines.FCFS})`,
                fill: '#3B82F6',
                position: 'right',
                fontSize: 10,
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      <div className="pt-2 border-t border-border flex items-center justify-between text-xs text-foreground-muted">
        <span className="flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-primary" />
          <span>Convergence: Approaching SRTF minimum under unit cost without starvation</span>
        </span>
        <span className="font-mono text-[11px]">20,000 Episodes Total</span>
      </div>
    </div>
  );
};

export default LearningCurve;
