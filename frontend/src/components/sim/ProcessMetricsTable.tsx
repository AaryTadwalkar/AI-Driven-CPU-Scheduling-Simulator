/**
 * @file src/components/sim/ProcessMetricsTable.tsx
 * 1. What this file does:
 *    Renders a detailed tabular breakdown of per-process scheduling metrics: Arrival, Burst,
 *    Completion, Turnaround, Waiting, and Response times, with PID color badges and column averages.
 * 2. Overall design:
 *    Displays processes in an accessible table with monospace tabular numbers.
 *    Provides a summary footer row computing arithmetic averages across the workload.
 * 3. Elements in the file:
 *    - `ProcessMetricsTableProps`: Interface for per-process metrics array
 *    - `ProcessMetricsTable`: React functional component
 * 4. Final output of the file:
 *    Exported table component rendered on the Simulator page.
 */

import React from 'react';
import { Table } from 'lucide-react';
import type { ProcessMetricsSchema } from '@/api/client';
import { getProcessColor } from '@/lib/colors';
import { formatNumber } from '@/lib/format';

export interface ProcessMetricsTableProps {
  processMetrics: ProcessMetricsSchema[];
}

/**
 * Renders individual process turnaround, waiting, and response metrics in a formatted table.
 *
 * 1. The use of function:
 *    Provides granular process-level performance verification for OS scheduling policies.
 * 2. How does the specific function work:
 *    Iterates over ProcessMetricsSchema entries, attaches deterministic PID color dots,
 *    renders arrival, burst, completion, and computed delays, and renders an aggregate footer.
 * 3. Concepts used in the function:
 *    Tabular numbers, process lifecycle accounting (TAT = Comp - Arr, Wait = TAT - Burst).
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Array of ProcessMetricsSchema objects.
 *    After: Polished tabular UI with process color indicators and summary row.
 *
 * @param props - Component parameters conforming to ProcessMetricsTableProps.
 * @returns JSX Element rendering the process metrics table.
 */
export const ProcessMetricsTable: React.FC<ProcessMetricsTableProps> = ({
  processMetrics,
}) => {
  const n = processMetrics.length;
  const avgArrival = n > 0 ? processMetrics.reduce((s, p) => s + p.arrival, 0) / n : 0;
  const avgBurst = n > 0 ? processMetrics.reduce((s, p) => s + p.burst, 0) / n : 0;
  const avgComp = n > 0 ? processMetrics.reduce((s, p) => s + p.completion, 0) / n : 0;
  const avgTAT = n > 0 ? processMetrics.reduce((s, p) => s + p.turnaround, 0) / n : 0;
  const avgWait = n > 0 ? processMetrics.reduce((s, p) => s + p.waiting, 0) / n : 0;
  const avgResp = n > 0 ? processMetrics.reduce((s, p) => s + p.response, 0) / n : 0;

  return (
    <div className="p-5 rounded-card border border-border bg-surface shadow-subtle space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <h3 className="font-semibold text-sm text-foreground flex items-center gap-2">
          <Table className="w-4 h-4 text-primary" />
          Per-Process Lifecycle Metrics
        </h3>
        <span className="text-xs text-foreground-muted font-mono">
          {n} Process{n === 1 ? '' : 'es'}
        </span>
      </div>

      {/* Table Container */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-border text-foreground-muted bg-surface-muted/50">
              <th className="py-2.5 px-3 font-semibold">PID</th>
              <th className="py-2.5 px-3 font-semibold text-right">Arrival (t)</th>
              <th className="py-2.5 px-3 font-semibold text-right">Burst (B)</th>
              <th className="py-2.5 px-3 font-semibold text-right">Completion</th>
              <th className="py-2.5 px-3 font-semibold text-right">Turnaround</th>
              <th className="py-2.5 px-3 font-semibold text-right">Waiting</th>
              <th className="py-2.5 px-3 font-semibold text-right">Response</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {processMetrics.length === 0 ? (
              <tr>
                <td colSpan={7} className="py-8 text-center text-foreground-muted">
                  No simulation metrics available. Run a simulation to populate process stats.
                </td>
              </tr>
            ) : (
              processMetrics.map((p) => {
                const color = getProcessColor(p.pid);
                return (
                  <tr
                    key={p.pid}
                    className="hover:bg-surface-muted/40 transition-colors font-mono-numbers"
                  >
                    <td className="py-2.5 px-3 font-semibold text-foreground font-mono flex items-center gap-2">
                      <span
                        style={{ backgroundColor: color }}
                        className="w-2.5 h-2.5 rounded-full inline-block flex-shrink-0"
                      />
                      {p.pid}
                    </td>
                    <td className="py-2.5 px-3 text-right text-foreground-muted">
                      {p.arrival}
                    </td>
                    <td className="py-2.5 px-3 text-right text-foreground font-medium">
                      {p.burst}
                    </td>
                    <td className="py-2.5 px-3 text-right text-foreground-muted">
                      {p.completion}
                    </td>
                    <td className="py-2.5 px-3 text-right font-medium text-foreground">
                      {p.turnaround}
                    </td>
                    <td className="py-2.5 px-3 text-right font-medium text-primary">
                      {p.waiting}
                    </td>
                    <td className="py-2.5 px-3 text-right text-foreground-muted">
                      {p.response}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
          {n > 0 && (
            <tfoot>
              <tr className="border-t-2 border-border bg-surface-muted/30 font-semibold font-mono-numbers text-foreground">
                <td className="py-2.5 px-3 font-sans">Workload Mean</td>
                <td className="py-2.5 px-3 text-right text-foreground-muted">
                  {formatNumber(avgArrival, 1)}
                </td>
                <td className="py-2.5 px-3 text-right">
                  {formatNumber(avgBurst, 1)}
                </td>
                <td className="py-2.5 px-3 text-right text-foreground-muted">
                  {formatNumber(avgComp, 1)}
                </td>
                <td className="py-2.5 px-3 text-right text-foreground">
                  {formatNumber(avgTAT, 2)}
                </td>
                <td className="py-2.5 px-3 text-right text-primary">
                  {formatNumber(avgWait, 2)}
                </td>
                <td className="py-2.5 px-3 text-right text-foreground-muted">
                  {formatNumber(avgResp, 2)}
                </td>
              </tr>
            </tfoot>
          )}
        </table>
      </div>
    </div>
  );
};
