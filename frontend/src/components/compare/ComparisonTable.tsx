/**
 * @file src/components/compare/ComparisonTable.tsx
 * 1. What this file does:
 *    Renders the multi-algorithm benchmark comparison table with best-per-metric highlighting,
 *    baseline differential comparison (+/- deltas), and CSV export capability.
 * 2. Overall design:
 *    Evaluates minimums for lower-is-better metrics (waiting, turnaround, response, context switches)
 *    and maximums for higher-is-better metrics (CPU utilization, Jain's fairness index).
 *    Highlights winning cells with a subtle emerald pill badge.
 *    Supports choosing any algorithm as baseline to inspect relative gains or regressions.
 * 3. Elements in the file:
 *    - `ComparisonTableProps`: Component input parameters
 *    - `METRIC_CONFIG`: Metric column definitions with directionality and formatting rules
 *    - `ComparisonTable`: React functional component
 * 4. Final output of the file:
 *    Exported benchmark table component mounted on the ComparePage.
 */

import React, { useState, useMemo } from 'react';
import { Download, Award, GitCompare, Table as TableIcon } from 'lucide-react';
import type { ComparisonCsvRow } from '@/lib/csv';
import { downloadComparisonCsv } from '@/lib/csv';
import { formatNumber, formatPercent, formatDelta, formatAlgorithmName } from '@/lib/format';
import { getAlgorithmColor } from '@/lib/colors';

export interface ComparisonTableProps {
  rows: ComparisonCsvRow[];
  defaultBaseline?: string | null;
  title?: string;
}

interface MetricDef {
  key: keyof ComparisonCsvRow;
  label: string;
  lowerIsBetter: boolean;
  format: (val: number) => string;
}

const METRICS: MetricDef[] = [
  {
    key: 'avg_waiting',
    label: 'Avg Waiting',
    lowerIsBetter: true,
    format: (v) => formatNumber(v, 2),
  },
  {
    key: 'avg_turnaround',
    label: 'Avg Turnaround',
    lowerIsBetter: true,
    format: (v) => formatNumber(v, 2),
  },
  {
    key: 'avg_response',
    label: 'Avg Response',
    lowerIsBetter: true,
    format: (v) => formatNumber(v, 2),
  },
  {
    key: 'cpu_utilization',
    label: 'CPU Utilization',
    lowerIsBetter: false,
    format: (v) => formatPercent(v, 1),
  },
  {
    key: 'jain_fairness',
    label: "Jain's Fairness",
    lowerIsBetter: false,
    format: (v) => formatNumber(v, 3),
  },
  {
    key: 'context_switches',
    label: 'Context Switches',
    lowerIsBetter: true,
    format: (v) => v.toString(),
  },
];

/**
 * Multi-algorithm benchmark comparison table component.
 *
 * 1. The use of function:
 *    Displays tabular comparison across scheduling algorithms on the same workload.
 * 2. How does the specific function work:
 *    Calculates column extrema to assign best-in-class badges, resolves baseline deltas,
 *    and enables direct CSV export.
 * 3. Concepts used in the function:
 *    Directional metric optimality, relative difference calculation, tabular typography.
 * 4. Places where this function is imported and used:
 *    Imported by ComparePage.tsx and test suites.
 * 5. Final output of the function, before and after:
 *    Before: Raw comparison summary list.
 *    After: Interactive, accessible HTML table with highlights, deltas, and CSV export.
 *
 * @param props - Component configuration conforming to ComparisonTableProps.
 * @returns JSX Element.
 */
export const ComparisonTable: React.FC<ComparisonTableProps> = ({
  rows,
  defaultBaseline = null,
  title = 'Performance Benchmark Metrics',
}) => {
  const [baselineAlgo, setBaselineAlgo] = useState<string | null>(defaultBaseline);

  // Compute best value per metric
  const bestValues = useMemo(() => {
    const bests: Partial<Record<keyof ComparisonCsvRow, number>> = {};
    if (rows.length === 0) return bests;

    METRICS.forEach(({ key, lowerIsBetter }) => {
      const values = rows.map((r) => Number(r[key]));
      bests[key] = lowerIsBetter ? Math.min(...values) : Math.max(...values);
    });

    return bests;
  }, [rows]);

  // Find baseline row if selected
  const baselineRow = useMemo(() => {
    if (!baselineAlgo) return null;
    return rows.find((r) => r.algorithm === baselineAlgo) ?? null;
  }, [rows, baselineAlgo]);

  /**
   * Handles exporting the comparison data to a client-side CSV file.
   *
   * 1. The use of function:
   *    Triggers browser CSV download of currently visible comparison rows.
   * 2. How does the specific function work:
   *    Calls `downloadComparisonCsv(rows)`.
   * 3. Concepts used in the function:
   *    File download triggering via Blob URL.
   * 4. Places where this function is imported and used:
   *    Attached to the Export CSV button click handler.
   * 5. Final output of the function, before and after:
   *    Before: Click event.
   *    After: Downloaded 'algorithm_comparison.csv' file.
   */
  const handleExportCsv = () => {
    downloadComparisonCsv(rows, 'algorithm_comparison.csv');
  };

  return (
    <div className="p-5 rounded-card border border-border bg-surface shadow-subtle space-y-4">
      {/* Table Header and Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="font-semibold text-sm text-foreground flex items-center gap-2">
            <TableIcon className="w-4 h-4 text-primary" />
            {title}
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Green highlight indicates best-performing algorithm for each metric.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          {/* Baseline selector */}
          <div className="flex items-center gap-1.5 text-xs text-foreground-muted">
            <GitCompare className="w-3.5 h-3.5 text-primary" />
            <label htmlFor="baseline-select" className="font-medium">
              Baseline:
            </label>
            <select
              id="baseline-select"
              value={baselineAlgo || ''}
              onChange={(e) => setBaselineAlgo(e.target.value || null)}
              className="px-2 py-1 rounded-control text-xs border border-border bg-surface text-foreground hover:bg-surface-muted transition-colors focus:outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="">None (Absolute)</option>
              {rows.map((r) => (
                <option key={r.algorithm} value={r.algorithm}>
                  {r.algorithm}
                </option>
              ))}
            </select>
          </div>

          {/* Export CSV button */}
          <button
            type="button"
            onClick={handleExportCsv}
            disabled={rows.length === 0}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-control text-xs font-medium border border-border bg-surface text-foreground hover:bg-surface-muted transition-colors disabled:opacity-50 shadow-sm"
          >
            <Download className="w-3.5 h-3.5 text-primary" />
            <span>Export Table to CSV</span>
          </button>
        </div>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-border text-foreground-muted font-medium bg-surface-muted/50">
              <th scope="col" className="py-2.5 px-3">
                Algorithm
              </th>
              {METRICS.map((m) => (
                <th key={m.key} scope="col" className="py-2.5 px-3 text-right">
                  <div className="flex flex-col items-end">
                    <span>{m.label}</span>
                    <span className="text-[10px] text-foreground-muted/70 font-normal">
                      {m.lowerIsBetter ? '↓ lower is better' : '↑ higher is better'}
                    </span>
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {rows.map((row) => {
              const isBaseline = baselineRow?.algorithm === row.algorithm;
              const algoColor = getAlgorithmColor(row.algorithm);

              return (
                <tr
                  key={row.algorithm}
                  className={`hover:bg-surface-muted/40 transition-colors ${
                    isBaseline ? 'bg-primary/5 font-medium' : ''
                  }`}
                >
                  {/* Algorithm Column */}
                  <td className="py-3 px-3">
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: algoColor }}
                      />
                      <div className="flex flex-col">
                        <span className="font-semibold text-foreground">
                          {row.algorithm}
                        </span>
                        <span className="text-[11px] text-foreground-muted truncate max-w-[180px]">
                          {formatAlgorithmName(row.algorithm)}
                        </span>
                      </div>
                      {isBaseline && (
                        <span className="ml-1.5 px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold bg-primary/10 text-primary border border-primary/20">
                          Baseline
                        </span>
                      )}
                    </div>
                  </td>

                  {/* Metric Columns */}
                  {METRICS.map(({ key, lowerIsBetter, format }) => {
                    const val = Number(row[key]);
                    const bestVal = bestValues[key];
                    const isBest =
                      bestVal !== undefined && Math.abs(val - bestVal) < 0.0001;

                    // Relative delta if baseline is active
                    let deltaInfo = null;
                    if (baselineRow && !isBaseline) {
                      const baseVal = Number(baselineRow[key]);
                      deltaInfo = formatDelta(val, baseVal, lowerIsBetter);
                    }

                    return (
                      <td key={key} className="py-3 px-3 text-right font-mono-numbers">
                        <div className="flex flex-col items-end gap-0.5">
                          <div className="flex items-center gap-1.5 justify-end">
                            {isBest && (
                              <span
                                className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-success/15 text-success border border-success/30 shadow-xs"
                                title="Best result for this metric"
                              >
                                <Award className="w-3 h-3 text-success shrink-0" />
                                <span>Best</span>
                              </span>
                            )}
                            <span
                              className={`text-xs ${
                                isBest
                                  ? 'font-bold text-success'
                                  : 'font-semibold text-foreground'
                              }`}
                            >
                              {format(val)}
                            </span>
                          </div>

                          {deltaInfo && (
                            <span
                              className={`text-[10px] font-mono font-medium ${deltaInfo.colorClass}`}
                              title={`Delta vs ${baselineRow?.algorithm}: ${deltaInfo.text}`}
                            >
                              {deltaInfo.text}
                            </span>
                          )}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ComparisonTable;
