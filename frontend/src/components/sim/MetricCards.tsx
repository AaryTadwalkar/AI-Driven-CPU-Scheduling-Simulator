/**
 * @file src/components/sim/MetricCards.tsx
 * 1. What this file does:
 *    Renders a responsive 6-card grid summarizing key CPU scheduling simulation metrics:
 *    Average Waiting Time, Turnaround Time, Response Time, CPU Utilization, Jain's Fairness Index,
 *    and Context Switch count.
 * 2. Overall design:
 *    Uses StatCard with tooltips explaining each OS scheduling metric and tabular typography.
 *    Supports optional baseline delta comparisons (e.g. against FCFS or reference baseline).
 * 3. Elements in the file:
 *    - `MetricCardsProps`: TypeScript interface for metrics payload and baseline
 *    - `MetricCards`: React functional component
 * 4. Final output of the file:
 *    Exported metric summary grid component rendered on the Simulator page.
 */

import React from 'react';
import { Clock, RotateCcw, Zap, Activity, Scale, Shuffle } from 'lucide-react';
import type { SimulationMetricsSchema } from '@/api/client';
import { formatNumber, formatPercent, formatDelta } from '@/lib/format';
import { StatCard } from '@/components/common/StatCard';

export interface MetricCardsProps {
  metrics: SimulationMetricsSchema | null;
  baselineMetrics?: SimulationMetricsSchema | null;
  isLoading?: boolean;
}

/**
 * Grid of metric cards presenting aggregate simulation performance.
 *
 * 1. The use of function:
 *    Displays high-level performance indicators for the active simulation run.
 * 2. How does the specific function work:
 *    Extracts values from SimulationMetricsSchema, applies precision formatters,
 *    and computes optional deltas when baselineMetrics are provided.
 * 3. Concepts used in the function:
 *    KPI dashboard design, tabular numeric formatting, contextual metric definitions.
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Raw simulation metrics payload or null state.
 *    After: Responsive 6-card performance dashboard with tooltips and units.
 *
 * @param props - Component parameters conforming to MetricCardsProps.
 * @returns JSX Element rendering the 6-card grid.
 */
export const MetricCards: React.FC<MetricCardsProps> = ({
  metrics,
  baselineMetrics,
  isLoading = false,
}) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
      {/* 1. Average Waiting Time */}
      <StatCard
        label="Avg Waiting"
        value={formatNumber(metrics?.avg_waiting, 2)}
        unit="ticks"
        icon={<Clock className="w-3.5 h-3.5" />}
        definition="Average time processes spend waiting in the ready queue before completing execution."
        delta={
          metrics && baselineMetrics
            ? formatDelta(metrics.avg_waiting, baselineMetrics.avg_waiting, true)
            : undefined
        }
        isLoading={isLoading}
      />

      {/* 2. Average Turnaround Time */}
      <StatCard
        label="Avg Turnaround"
        value={formatNumber(metrics?.avg_turnaround, 2)}
        unit="ticks"
        icon={<RotateCcw className="w-3.5 h-3.5" />}
        definition="Average total elapsed time from process arrival to final completion (TAT = Completion - Arrival)."
        delta={
          metrics && baselineMetrics
            ? formatDelta(metrics.avg_turnaround, baselineMetrics.avg_turnaround, true)
            : undefined
        }
        isLoading={isLoading}
      />

      {/* 3. Average Response Time */}
      <StatCard
        label="Avg Response"
        value={formatNumber(metrics?.avg_response, 2)}
        unit="ticks"
        icon={<Zap className="w-3.5 h-3.5" />}
        definition="Average duration from process arrival until its first CPU execution tick (Response = First Run - Arrival)."
        delta={
          metrics && baselineMetrics
            ? formatDelta(metrics.avg_response, baselineMetrics.avg_response, true)
            : undefined
        }
        isLoading={isLoading}
      />

      {/* 4. CPU Utilization */}
      <StatCard
        label="CPU Utilization"
        value={formatPercent(metrics?.cpu_utilization, 1)}
        icon={<Activity className="w-3.5 h-3.5" />}
        definition="Percentage of total elapsed simulation time during which the CPU was actively executing user burst ticks."
        delta={
          metrics && baselineMetrics
            ? formatDelta(metrics.cpu_utilization, baselineMetrics.cpu_utilization, false)
            : undefined
        }
        isLoading={isLoading}
      />

      {/* 5. Jain's Fairness Index */}
      <StatCard
        label="Jain's Fairness"
        value={formatNumber(metrics?.jain_fairness, 3)}
        icon={<Scale className="w-3.5 h-3.5" />}
        definition="Structural metric of service equity across processes, bounded in [0, 1]. A score of 1.0 indicates identical turnaround ratios."
        delta={
          metrics && baselineMetrics
            ? formatDelta(metrics.jain_fairness, baselineMetrics.jain_fairness, false)
            : undefined
        }
        isLoading={isLoading}
      />

      {/* 6. Context Switches */}
      <StatCard
        label="Context Switches"
        value={metrics ? metrics.context_switches.toString() : '—'}
        unit="switches"
        icon={<Shuffle className="w-3.5 h-3.5" />}
        definition="Total number of process context switches executed by the CPU scheduler during this workload."
        delta={
          metrics && baselineMetrics
            ? formatDelta(metrics.context_switches, baselineMetrics.context_switches, true)
            : undefined
        }
        isLoading={isLoading}
      />
    </div>
  );
};
