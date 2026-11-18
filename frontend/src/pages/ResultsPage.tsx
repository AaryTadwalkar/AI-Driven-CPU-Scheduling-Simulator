/**
 * @file src/pages/ResultsPage.tsx
 * 1. What this file does:
 *    Renders the Benchmark Results and Empirical Performance Evaluation page.
 * 2. Overall design:
 *    Displays executive summary KPI cards highlighting major research findings, mounts the
 *    canonical benchmark summary table with 95% bootstrap confidence intervals, embeds the
 *    9-scenario breakdown explorer with Recharts comparative bars, and includes the honest
 *    technical limitations accordion panel conforming to Section 8 of the report.
 * 3. Elements in the file:
 *    - `ResultsPage`: Primary React functional page component
 * 4. Final output of the file:
 *    Comprehensive evaluation dashboard mounted at route '/results' in React Router.
 */

import React from 'react';
import {
  FileText,
  Award,
  ShieldCheck,
  TrendingDown,
  Download,
  RefreshCw,
  Terminal,
} from 'lucide-react';
import { useResultsSummaryQuery } from '@/api/hooks';
import { SummaryTable, type SummaryTableRow } from '@/components/results/SummaryTable';
import { ScenarioBreakdown, type ScenarioBreakdownItem } from '@/components/results/ScenarioBreakdown';
import { LimitationsPanel } from '@/components/results/LimitationsPanel';
import { CardSkeleton, TableSkeleton } from '@/components/common/Skeletons';
import { downloadSummaryTableCsv, downloadScenarioBreakdownCsv } from '@/lib/csv';

/**
 * Benchmark Results and Evaluation page component.
 *
 * 1. The use of function:
 *    Top-level page component for route '/results'.
 * 2. How does the specific function work:
 *    Fetches benchmark evaluation results from /api/results/summary. Renders statistical findings,
 *    confidence interval tables, scenario breakdowns, and technical trade-offs.
 * 3. Concepts used in the function:
 *    Statistical evaluation reporting, bootstrap confidence intervals, scenario stratification.
 * 4. Places where this function is imported and used:
 *    Imported by App.tsx / routes.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Unrouted path or placeholder.
 *    After: Full-featured benchmark results view with export tools and limitations disclosure.
 *
 * @returns JSX Element.
 */
export const ResultsPage: React.FC = () => {
  const { data, isLoading, error, refetch } = useResultsSummaryQuery();

  const isMissingResults =
    (error && typeof error === 'object' && 'status' in error && (error as any).status === 503) ||
    (typeof (error as any)?.detail === 'string' &&
      (error as any).detail.includes('evaluate'));

  const summaryData = (data as { summary_table?: SummaryTableRow[] })?.summary_table || [];
  const scenarioData = (data as { scenario_breakdown?: ScenarioBreakdownItem[] })?.scenario_breakdown || [];

  const handleDownloadAll = () => {
    if (summaryData.length > 0) {
      downloadSummaryTableCsv(summaryData, 'summary_table.csv');
    }
    if (scenarioData.length > 0) {
      downloadScenarioBreakdownCsv(scenarioData, 'scenario_breakdown.csv');
    }
  };

  return (
    <div className="space-y-6" data-testid="results-page">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <FileText className="w-6 h-6 text-primary" />
            Benchmark Results & Empirical Evaluation
          </h1>
          <p className="text-sm text-foreground-muted mt-1">
            Empirical validation on 1,000 held-out test workloads (12,000 simulated executions) with 95% bootstrap confidence intervals.
          </p>
        </div>

        <button
          type="button"
          onClick={handleDownloadAll}
          disabled={isLoading || summaryData.length === 0}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-control bg-primary text-white hover:bg-primary-hover transition-colors disabled:opacity-50 self-start sm:self-center"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Download All Artifacts (CSV)</span>
        </button>
      </div>

      {/* HTTP 503 Missing Evaluation State */}
      {isMissingResults ? (
        <div className="p-6 rounded-card border border-amber-500/30 bg-surface shadow-subtle space-y-3">
          <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
            <Terminal className="w-4 h-4 text-amber-500" />
            <span>Benchmark Evaluation Artifacts Missing</span>
          </div>
          <p className="text-xs text-foreground-muted leading-relaxed">
            The evaluation suite has not been executed yet. Run the following command in your terminal to build test results:
          </p>
          <pre className="p-2.5 rounded bg-surface-muted border border-border text-xs font-mono text-primary select-all">
            python cli.py evaluate
          </pre>
          <button
            type="button"
            onClick={() => refetch()}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-control bg-primary text-white hover:bg-primary-hover transition-colors"
          >
            <RefreshCw className="w-3 h-3" /> Retry Connection
          </button>
        </div>
      ) : isLoading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <CardSkeleton />
            <CardSkeleton />
            <CardSkeleton />
          </div>
          <TableSkeleton rows={7} />
        </div>
      ) : (
        <>
          {/* Executive Summary Cards: 3 Major Findings */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Finding 1 */}
            <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-foreground-muted uppercase tracking-wider">
                  Finding 1: Supervised ML
                </span>
                <Award className="w-4 h-4 text-primary" />
              </div>
              <div className="text-xl font-bold font-mono text-foreground">
                5.4x Regret Reduction
              </div>
              <p className="text-xs text-foreground-muted leading-relaxed">
                Decision Tree meta-recommender achieves <strong className="text-foreground">0.0487 regret</strong> versus{' '}
                <span className="text-foreground font-semibold">0.2655 for Always-SJF</span> across 1,000 test workloads, proving adaptive policy selection substantially outperforms static heuristics.
              </p>
            </div>

            {/* Finding 2 */}
            <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-foreground-muted uppercase tracking-wider">
                  Finding 2: Fairness & RL
                </span>
                <ShieldCheck className="w-4 h-4 text-emerald-500" />
              </div>
              <div className="text-xl font-bold font-mono text-emerald-500">
                56% Starvation Reduction
              </div>
              <p className="text-xs text-foreground-muted leading-relaxed">
                Under zero-overhead preemption, the Q-learning agent cuts starvation incidence from{' '}
                <span className="text-danger font-semibold">35.8% (SRTF)</span> to{' '}
                <strong className="text-emerald-500">15.7%</strong> while remaining within 2.3 ticks of optimal theoretical waiting time.
              </p>
            </div>

            {/* Finding 3 */}
            <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-foreground-muted uppercase tracking-wider">
                  Finding 3: Preemption Cost
                </span>
                <TrendingDown className="w-4 h-4 text-pink-500" />
              </div>
              <div className="text-xl font-bold font-mono text-foreground">
                46% Switching Economy
              </div>
              <p className="text-xs text-foreground-muted leading-relaxed">
                When context switching costs 1 tick, Round Robin waiting inflates by <span className="text-danger font-semibold">71%</span>. The RL agent automatically adapts, cutting switches from <span className="text-foreground">13.6 to 7.4</span>.
              </p>
            </div>
          </div>

          {/* Canonical Summary Table */}
          <SummaryTable summaryData={summaryData} />

          {/* Scenario Stratification Breakdown */}
          <ScenarioBreakdown scenarios={scenarioData} />

          {/* Honest Technical Limitations & Discussion */}
          <LimitationsPanel />
        </>
      )}
    </div>
  );
};

export default ResultsPage;
