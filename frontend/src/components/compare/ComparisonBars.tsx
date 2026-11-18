/**
 * @file src/components/compare/ComparisonBars.tsx
 * 1. What this file does:
 *    Renders interactive grouped bar charts comparing core latency metrics (Waiting,
 *    Turnaround, Response time) across scheduling algorithms using Recharts.
 * 2. Overall design:
 *    Transforms tabular comparison records into grouped bar series.
 *    Supports toggling between 'Group by Metric' (comparing all policies per latency metric)
 *    and 'Group by Algorithm' (comparing all three latency metrics per policy).
 *    Uses standardized algorithm palette colors and renders an engineering tabular tooltip.
 * 3. Elements in the file:
 *    - `ComparisonBarsProps`: Input interface for comparison data and optional view modes
 *    - `ComparisonBars`: Main React chart component
 *    - `CustomTooltip`: Tabular hover tooltip component for exact numeric inspection
 * 4. Final output of the file:
 *    Exported Recharts bar chart visualization mounted on ComparePage.
 */

import React, { useState, useMemo } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from 'recharts';
import { BarChart2 } from 'lucide-react';
import type { ComparisonCsvRow } from '@/lib/csv';
import { getAlgorithmColor } from '@/lib/colors';
import { formatNumber } from '@/lib/format';

export interface ComparisonBarsProps {
  rows: ComparisonCsvRow[];
  title?: string;
  defaultGrouping?: 'metric' | 'algorithm';
}

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number;
    color: string;
    dataKey: string;
  }>;
  label?: string;
}

/**
 * Custom engineering tooltip component for Recharts hover events.
 *
 * 1. The use of function:
 *    Displays precise tabular metrics when hovering over bar chart columns.
 * 2. How does the specific function work:
 *    Renders an accessible card listing each active bar's colored badge, series name,
 *    and formatted duration in ticks.
 * 3. Concepts used in the function:
 *    Recharts custom tooltip protocol, tabular typography, subtle surface styling.
 * 4. Places where this function is imported and used:
 *    Internal to ComparisonBars.tsx as the `<Tooltip content={<CustomTooltip />} />`.
 * 5. Final output of the function, before and after:
 *    Before: Generic unstyled SVG tooltip.
 *    After: Polished tabular card with exact latency values and units.
 *
 * @param props - Tooltip properties provided by Recharts runtime.
 * @returns JSX Element or null.
 */
export const CustomTooltip: React.FC<CustomTooltipProps> = ({ active, payload, label }) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  return (
    <div className="p-3 rounded-card border border-border bg-surface shadow-subtle text-xs space-y-2 min-w-[160px] select-none">
      <div className="font-semibold text-foreground border-b border-border pb-1">
        {label}
      </div>
      <div className="space-y-1 font-mono-numbers">
        {payload.map((entry, idx) => (
          <div key={`tooltip-${entry.dataKey || idx}`} className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-1.5 text-foreground-muted">
              <span
                className="w-2.5 h-2.5 rounded-sm shrink-0"
                style={{ backgroundColor: entry.color }}
              />
              <span className="truncate max-w-[120px] font-medium">{entry.name}</span>
            </div>
            <span className="font-semibold text-foreground">
              {formatNumber(entry.value, 2)} <span className="text-[10px] text-foreground-muted">ticks</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Interactive grouped bar chart comparing latency metrics across scheduling algorithms.
 *
 * 1. The use of function:
 *    Visualizes waiting, turnaround, and response times for multi-algorithm comparisons.
 * 2. How does the specific function work:
 *    Restructures ComparisonCsvRow items into Recharts series format based on grouping mode.
 *    Applies consistent algorithm colors and renders axes, gridlines, legend, and tooltip.
 * 3. Concepts used in the function:
 *    Recharts ResponsiveContainer, data pivoting, accessible SVG charting.
 * 4. Places where this function is imported and used:
 *    Imported by ComparePage.tsx and test suites.
 * 5. Final output of the function, before and after:
 *    Before: Array of comparison rows.
 *    After: Interactive grouped bar chart with hover tooltips and grouping toggles.
 *
 * @param props - Component configuration conforming to ComparisonBarsProps.
 * @returns JSX Element.
 */
export const ComparisonBars: React.FC<ComparisonBarsProps> = ({
  rows,
  title = 'Core Latency Comparison (ticks)',
  defaultGrouping = 'metric',
}) => {
  const [grouping, setGrouping] = useState<'metric' | 'algorithm'>(defaultGrouping);

  // Extract unique algorithm names
  const algorithms = useMemo(() => rows.map((r) => r.algorithm), [rows]);

  // Data pivoted by metric (X-axis: Waiting, Turnaround, Response; Bars: Algorithms)
  const metricGroupedData = useMemo(() => {
    const waitingRow: Record<string, string | number> = { metric: 'Avg Waiting Time' };
    const turnaroundRow: Record<string, string | number> = { metric: 'Avg Turnaround Time' };
    const responseRow: Record<string, string | number> = { metric: 'Avg Response Time' };

    rows.forEach((r) => {
      waitingRow[r.algorithm] = Number(r.avg_waiting);
      turnaroundRow[r.algorithm] = Number(r.avg_turnaround);
      responseRow[r.algorithm] = Number(r.avg_response);
    });

    return [waitingRow, turnaroundRow, responseRow];
  }, [rows]);

  // Data grouped by algorithm (X-axis: Algorithm; Bars: Waiting, Turnaround, Response)
  const algorithmGroupedData = useMemo(() => {
    return rows.map((r) => ({
      algorithm: r.algorithm,
      'Avg Waiting': Number(r.avg_waiting),
      'Avg Turnaround': Number(r.avg_turnaround),
      'Avg Response': Number(r.avg_response),
    }));
  }, [rows]);

  return (
    <div className="p-5 rounded-card border border-border bg-surface shadow-subtle space-y-4">
      {/* Chart Header and Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="font-semibold text-sm text-foreground flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-primary" />
            {title}
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Lower bar heights reflect superior latency performance (measured in discrete ticks).
          </p>
        </div>

        {/* Grouping Toggle */}
        <div className="flex items-center gap-1 p-0.5 rounded-control bg-surface-muted border border-border text-xs self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setGrouping('metric')}
            className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors ${
              grouping === 'metric'
                ? 'bg-surface text-primary shadow-xs font-semibold'
                : 'text-foreground-muted hover:text-foreground'
            }`}
          >
            By Metric
          </button>
          <button
            type="button"
            onClick={() => setGrouping('algorithm')}
            className={`px-2.5 py-1 rounded-[6px] font-medium transition-colors ${
              grouping === 'algorithm'
                ? 'bg-surface text-primary shadow-xs font-semibold'
                : 'text-foreground-muted hover:text-foreground'
            }`}
          >
            By Algorithm
          </button>
        </div>
      </div>

      {/* Chart Container */}
      <div className="w-full h-80 pt-2 pb-1" role="img" aria-label="Grouped latency comparison bar chart">
        <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={240}>
          {grouping === 'metric' ? (
            <BarChart
              data={metricGroupedData}
              margin={{ top: 12, right: 16, left: -10, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
              <XAxis
                dataKey="metric"
                tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={false}
                unit="t"
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ paddingTop: 12, fontSize: 12 }}
                iconType="circle"
                iconSize={8}
              />
              {algorithms.map((algo) => (
                <Bar
                  key={algo}
                  dataKey={algo}
                  name={algo}
                  fill={getAlgorithmColor(algo)}
                  radius={[4, 4, 0, 0]}
                  maxBarSize={48}
                />
              ))}
            </BarChart>
          ) : (
            <BarChart
              data={algorithmGroupedData}
              margin={{ top: 12, right: 16, left: -10, bottom: 20 }}
            >
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" />
              <XAxis
                dataKey="algorithm"
                tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: 'var(--text-muted)', fontSize: 11 }}
                axisLine={{ stroke: 'var(--border)' }}
                tickLine={false}
                unit="t"
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                wrapperStyle={{ paddingTop: 12, fontSize: 12 }}
                iconType="circle"
                iconSize={8}
              />
              <Bar
                dataKey="Avg Waiting"
                fill="#3B82F6"
                radius={[4, 4, 0, 0]}
                maxBarSize={36}
              />
              <Bar
                dataKey="Avg Turnaround"
                fill="#8B5CF6"
                radius={[4, 4, 0, 0]}
                maxBarSize={36}
              />
              <Bar
                dataKey="Avg Response"
                fill="#10B981"
                radius={[4, 4, 0, 0]}
                maxBarSize={36}
              />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ComparisonBars;
