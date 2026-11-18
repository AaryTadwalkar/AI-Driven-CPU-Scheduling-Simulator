/**
 * @file src/components/results/SummaryTable.tsx
 * 1. What this file does:
 *    Renders the canonical benchmark evaluation summary table across 1,000 held-out test workloads.
 * 2. Overall design:
 *    Presents point estimates alongside 95% bootstrap confidence intervals [lower, upper] for
 *    waiting, turnaround, response, CPU utilization, Jain's fairness, context switches, and starvation.
 *    Highlights best-performing algorithms per metric, supports cost toggling (0 vs 1), and CSV exports.
 * 3. Elements in the file:
 *    - `SummaryTableRow`: Interface matching records returned in summary_table array
 *    - `SummaryTableProps`: Interface for component props and customization
 *    - `formatEstimateCi`: Helper formatting point estimate with 95% bootstrap interval
 *    - `SummaryTable`: Primary React functional component
 * 4. Final output of the file:
 *    Data-dense benchmark table card consumed by ResultsPage.
 */

import React, { useState, useMemo } from 'react';
import { Table, Download } from 'lucide-react';
import { getAlgorithmColor } from '@/lib/colors';
import { downloadSummaryTableCsv, type SummaryTableCsvRow } from '@/lib/csv';
import { formatNumber, formatPercent } from '@/lib/format';

export interface SummaryTableRow {
  cs_cost: number;
  policy: string;
  workload_count?: number;
  avg_waiting_mean: number;
  avg_waiting_ci_lower?: number;
  avg_waiting_ci_upper?: number;
  avg_turnaround_mean: number;
  avg_turnaround_ci_lower?: number;
  avg_turnaround_ci_upper?: number;
  avg_response_mean: number;
  avg_response_ci_lower?: number;
  avg_response_ci_upper?: number;
  cpu_utilization_mean: number;
  cpu_utilization_ci_lower?: number;
  cpu_utilization_ci_upper?: number;
  context_switches_mean: number;
  context_switches_ci_lower?: number;
  context_switches_ci_upper?: number;
  jain_fairness_mean: number;
  jain_fairness_ci_lower?: number;
  jain_fairness_ci_upper?: number;
  starvation_rate_pct?: number;
}

export interface SummaryTableProps {
  summaryData?: SummaryTableRow[];
  className?: string;
}

/**
 * Formats a point estimate with its 95% confidence interval brackets.
 *
 * 1. The use of function:
 *    Displays statistical rigor in a compact tabular-numeric format: '14.41 [13.16, 15.77]'.
 * 2. How does the specific function work:
 *    Extracts mean, lower bound, and upper bound, formatting numbers with specified precision.
 * 3. Concepts used in the function:
 *    Bootstrap confidence interval representation, tabular string alignment.
 * 4. Places where this function is imported and used:
 *    Used internally by `SummaryTable` cells.
 * 5. Final output of the function, before and after:
 *    Before: `mean = 14.41, lower = 13.16, upper = 15.77`
 *    After: JSX element displaying value and interval subtitle.
 *
 * @param mean - Point estimate.
 * @param lower - CI lower bound.
 * @param upper - CI upper bound.
 * @param isPct - Whether values are percentages.
 * @param decimals - Decimal places (default 2).
 * @returns JSX Element.
 */
export function formatEstimateCi(
  mean: number,
  lower?: number,
  upper?: number,
  isPct = false,
  decimals = 2
): React.ReactNode {
  const fmt = (v: number) => (isPct ? formatPercent(v, 1) : formatNumber(v, decimals));
  const mainStr = fmt(mean);

  if (lower === undefined || upper === undefined) {
    return <span className="font-mono font-medium">{mainStr}</span>;
  }

  return (
    <div className="flex flex-col items-end">
      <span className="font-mono font-semibold text-foreground">{mainStr}</span>
      <span className="font-mono text-[10px] text-foreground-muted">
        [{fmt(lower)}, {fmt(upper)}]
      </span>
    </div>
  );
}

/**
 * Canonical Benchmark Evaluation Summary Table component.
 *
 * 1. The use of function:
 *    Displays empirical evaluation across 1,000 held-out test workloads for all algorithms.
 * 2. How does the specific function work:
 *    Filters summary rows by active `cs_cost` (0 or 1), determines optimal performance winners
 *    per metric, highlights cells with green badges, and triggers CSV downloads.
 * 3. Concepts used in the function:
 *    Bootstrap confidence intervals, directional optimization (lower vs higher is better),
 *    tabular data rendering, CSV export.
 * 4. Places where this function is imported and used:
 *    Imported by ResultsPage.tsx and test suites.
 * 5. Final output of the function, before and after:
 *    Before: Array of raw summary rows from /api/results/summary.
 *    After: Publication-quality interactive comparison table with 95% CIs.
 *
 * @param props - Summary table rows and styling classes.
 * @returns JSX Element.
 */
export const SummaryTable: React.FC<SummaryTableProps> = ({
  summaryData = [],
  className = '',
}) => {
  const [selectedCost, setSelectedCost] = useState<0 | 1>(0);

  // Filter rows by cost variant
  const rows = useMemo(() => {
    return summaryData.filter((r) => Number(r.cs_cost) === selectedCost);
  }, [summaryData, selectedCost]);

  // Compute best values per metric for highlighting
  const bestValues = useMemo(() => {
    if (rows.length === 0) return {};

    const minWait = Math.min(...rows.map((r) => r.avg_waiting_mean));
    const minTat = Math.min(...rows.map((r) => r.avg_turnaround_mean));
    const minResp = Math.min(...rows.map((r) => r.avg_response_mean));
    const maxUtil = Math.max(...rows.map((r) => r.cpu_utilization_mean));
    const maxFairness = Math.max(...rows.map((r) => r.jain_fairness_mean));
    const minSwitches = Math.min(...rows.map((r) => r.context_switches_mean));
    const minStarvation = Math.min(
      ...rows.map((r) => r.starvation_rate_pct ?? 100)
    );

    return {
      minWait,
      minTat,
      minResp,
      maxUtil,
      maxFairness,
      minSwitches,
      minStarvation,
    };
  }, [rows]);

  const handleDownloadCsv = () => {
    const csvRows: SummaryTableCsvRow[] = summaryData.map((r) => ({
      cs_cost: r.cs_cost,
      policy: r.policy,
      workload_count: r.workload_count,
      avg_waiting_mean: r.avg_waiting_mean,
      avg_waiting_ci_lower: r.avg_waiting_ci_lower,
      avg_waiting_ci_upper: r.avg_waiting_ci_upper,
      avg_turnaround_mean: r.avg_turnaround_mean,
      avg_turnaround_ci_lower: r.avg_turnaround_ci_lower,
      avg_turnaround_ci_upper: r.avg_turnaround_ci_upper,
      avg_response_mean: r.avg_response_mean,
      avg_response_ci_lower: r.avg_response_ci_lower,
      avg_response_ci_upper: r.avg_response_ci_upper,
      cpu_utilization_mean: r.cpu_utilization_mean,
      cpu_utilization_ci_lower: r.cpu_utilization_ci_lower,
      cpu_utilization_ci_upper: r.cpu_utilization_ci_upper,
      context_switches_mean: r.context_switches_mean,
      context_switches_ci_lower: r.context_switches_ci_lower,
      context_switches_ci_upper: r.context_switches_ci_upper,
      jain_fairness_mean: r.jain_fairness_mean,
      jain_fairness_ci_lower: r.jain_fairness_ci_lower,
      jain_fairness_ci_upper: r.jain_fairness_ci_upper,
      starvation_rate_pct: r.starvation_rate_pct,
    }));

    downloadSummaryTableCsv(csvRows, `summary_table_cs${selectedCost}.csv`);
  };

  return (
    <div
      className={`p-5 rounded-card border border-border bg-surface shadow-subtle flex flex-col space-y-4 ${className}`}
      data-testid="summary-table"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Table className="w-4 h-4 text-primary" />
            Canonical Evaluation Summary (1,000 Test Workloads)
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Point estimates with 95% bootstrap confidence intervals [lower, upper] across 12,000 total runs.
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
            <span>Download Summary CSV</span>
          </button>
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto rounded-control border border-border">
        <table className="w-full text-xs text-left border-collapse" aria-label="Benchmark Summary Table">
          <thead>
            <tr className="border-b border-border bg-surface-muted/50 font-semibold text-foreground">
              <th className="p-3">Policy</th>
              <th className="p-3 text-right">Avg Waiting</th>
              <th className="p-3 text-right">Avg Turnaround</th>
              <th className="p-3 text-right">Avg Response</th>
              <th className="p-3 text-right">CPU Util (%)</th>
              <th className="p-3 text-right">Jain's Fairness</th>
              <th className="p-3 text-right">Context Switches</th>
              <th className="p-3 text-right">Starvation Rate</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const isBestWait = Math.abs(r.avg_waiting_mean - (bestValues.minWait ?? 0)) < 0.01;
              const isBestTat = Math.abs(r.avg_turnaround_mean - (bestValues.minTat ?? 0)) < 0.01;
              const isBestResp = Math.abs(r.avg_response_mean - (bestValues.minResp ?? 0)) < 0.01;
              const isBestUtil = Math.abs(r.cpu_utilization_mean - (bestValues.maxUtil ?? 0)) < 0.005;
              const isBestFair = Math.abs(r.jain_fairness_mean - (bestValues.maxFairness ?? 0)) < 0.005;
              const isBestSwitches = Math.abs(r.context_switches_mean - (bestValues.minSwitches ?? 0)) < 0.01;
              const isBestStarv =
                r.starvation_rate_pct !== undefined &&
                Math.abs(r.starvation_rate_pct - (bestValues.minStarvation ?? 0)) < 0.1;

              const algoColor = getAlgorithmColor(r.policy);
              const isRL = r.policy === 'RL' || r.policy === 'Q_LEARNING';
              const isDT = r.policy === 'DT' || r.policy === 'DECISION_TREE';

              return (
                <tr
                  key={`${r.cs_cost}-${r.policy}`}
                  className="border-b border-border/40 hover:bg-surface-muted/30 transition-colors"
                >
                  <td className="p-3">
                    <div className="flex items-center gap-2">
                      <span
                        style={{ backgroundColor: algoColor }}
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                      />
                      <span className="font-mono font-bold text-foreground">
                        {r.policy}
                      </span>
                      {isRL && (
                        <span className="px-1.5 py-0.2 rounded text-[9px] font-semibold bg-pink-500/10 text-pink-500 border border-pink-500/20">
                          Q-Agent
                        </span>
                      )}
                      {isDT && (
                        <span className="px-1.5 py-0.2 rounded text-[9px] font-semibold bg-primary/10 text-primary border border-primary/20">
                          Recommender
                        </span>
                      )}
                    </div>
                  </td>

                  {/* Avg Waiting */}
                  <td className={`p-3 text-right ${isBestWait ? 'bg-emerald-500/10' : ''}`}>
                    {formatEstimateCi(r.avg_waiting_mean, r.avg_waiting_ci_lower, r.avg_waiting_ci_upper)}
                  </td>

                  {/* Avg Turnaround */}
                  <td className={`p-3 text-right ${isBestTat ? 'bg-emerald-500/10' : ''}`}>
                    {formatEstimateCi(r.avg_turnaround_mean, r.avg_turnaround_ci_lower, r.avg_turnaround_ci_upper)}
                  </td>

                  {/* Avg Response */}
                  <td className={`p-3 text-right ${isBestResp ? 'bg-emerald-500/10' : ''}`}>
                    {formatEstimateCi(r.avg_response_mean, r.avg_response_ci_lower, r.avg_response_ci_upper)}
                  </td>

                  {/* CPU Utilization */}
                  <td className={`p-3 text-right ${isBestUtil ? 'bg-emerald-500/10' : ''}`}>
                    {formatEstimateCi(
                      r.cpu_utilization_mean,
                      r.cpu_utilization_ci_lower,
                      r.cpu_utilization_ci_upper,
                      true,
                      1
                    )}
                  </td>

                  {/* Jain's Fairness */}
                  <td className={`p-3 text-right ${isBestFair ? 'bg-emerald-500/10' : ''}`}>
                    {formatEstimateCi(
                      r.jain_fairness_mean,
                      r.jain_fairness_ci_lower,
                      r.jain_fairness_ci_upper,
                      false,
                      3
                    )}
                  </td>

                  {/* Context Switches */}
                  <td className={`p-3 text-right ${isBestSwitches ? 'bg-emerald-500/10' : ''}`}>
                    {formatEstimateCi(
                      r.context_switches_mean,
                      r.context_switches_ci_lower,
                      r.context_switches_ci_upper,
                      false,
                      1
                    )}
                  </td>

                  {/* Starvation Rate */}
                  <td className={`p-3 text-right ${isBestStarv ? 'bg-emerald-500/10' : ''}`}>
                    <span className="font-mono font-semibold text-foreground">
                      {r.starvation_rate_pct !== undefined ? `${r.starvation_rate_pct.toFixed(1)}%` : '—'}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="pt-2 border-t border-border flex flex-wrap items-center justify-between text-xs text-foreground-muted gap-2">
        <div className="flex items-center gap-2">
          <span className="inline-block w-2.5 h-2.5 rounded bg-emerald-500/20 border border-emerald-500/40" />
          <span>Green highlight indicates top-performing algorithm on that metric.</span>
        </div>
        <div className="font-mono text-[11px]">
          1,000 Resamples Bootstrap CI
        </div>
      </div>
    </div>
  );
};

export default SummaryTable;
