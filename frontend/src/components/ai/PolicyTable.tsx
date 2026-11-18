/**
 * @file src/components/ai/PolicyTable.tsx
 * 1. What this file does:
 *    Renders a searchable, paginated table inspecting top-visited MDP states and optimal Q-actions.
 * 2. Overall design:
 *    Parses the 7-tuple aggregate state representation, displays visits and action distribution,
 *    highlights the winning meta-action with color badges, and displays Q-values for all 4 rules
 *    with maximum value emphasis. Includes real-time search, action filter, and page controls.
 * 3. Elements in the file:
 *    - `TopStateItem`: Data contract for individual state entries from /api/rl/policy-summary
 *    - `PolicyTableProps`: Component props covering data payload, cost variant, and callbacks
 *    - `formatState7Tuple`: Helper rendering human-readable descriptions of the 7-tuple
 *    - `getActionBadgeClass`: Color map helper for meta-rule actions
 *    - `PolicyTable`: Primary React functional component
 * 4. Final output of the file:
 *    Interactive Q-table policy browser card consumed by AILabPage.
 */

import React, { useState, useMemo } from 'react';
import { Search, Filter, ChevronLeft, ChevronRight, Binary, Award } from 'lucide-react';

export interface TopStateItem {
  state: number[];
  state_str: string;
  visits: number;
  best_action: number;
  best_action_name: string;
  q_values: Record<string, number>;
  action_visits?: Record<string, number>;
}

export interface PolicyTableProps {
  topStates?: TopStateItem[];
  qTableSize?: number;
  uniqueStatesVisited?: number;
  csCost?: number;
  onCsCostChange?: (cost: number) => void;
  className?: string;
}

const META_ACTIONS = [
  'SHORTEST_REMAINING',
  'LONGEST_REMAINING',
  'EARLIEST_ARRIVAL',
  'CONTINUE_CURRENT',
] as const;

/**
 * Maps an action name to a distinctive semantic badge styling class.
 *
 * 1. The use of function:
 *    Visually differentiates the 4 scheduling meta-rules across the policy table.
 * 2. How does the specific function work:
 *    Matches action strings to corresponding Tailwind border, text, and background tokens.
 * 3. Concepts used in the function:
 *    Semantic badge token assignment.
 * 4. Places where this function is imported and used:
 *    Used internally by `PolicyTable` and action badges.
 * 5. Final output of the function, before and after:
 *    Before: 'SHORTEST_REMAINING'
 *    After: 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
 *
 * @param action - The meta-action string.
 * @returns Class string.
 */
export function getActionBadgeClass(action: string): string {
  switch (action) {
    case 'SHORTEST_REMAINING':
      return 'bg-emerald-500/10 text-emerald-500 border-emerald-500/30';
    case 'LONGEST_REMAINING':
      return 'bg-purple-500/10 text-purple-400 border-purple-500/30';
    case 'EARLIEST_ARRIVAL':
      return 'bg-blue-500/10 text-blue-500 border-blue-500/30';
    case 'CONTINUE_CURRENT':
      return 'bg-amber-500/10 text-amber-500 border-amber-500/30';
    default:
      return 'bg-surface-muted text-foreground border-border';
  }
}

/**
 * Formats a 7-tuple aggregate state into a readable string breakdown.
 *
 * 1. The use of function:
 *    Explains the semantic dimensions encoded in the 7-element state tuple.
 * 2. How does the specific function work:
 *    Maps array indices [0..6] to queue count, min burst, max burst, wait time,
 *    previous process ready, pending arrivals, and min burst flags.
 * 3. Concepts used in the function:
 *    Aggregate state decoding, compact tuple serialization.
 * 4. Places where this function is imported and used:
 *    Used internally by `PolicyTable`.
 * 5. Final output of the function, before and after:
 *    Before: `[3, 4, 4, 3, 1, 0, 1]`
 *    After: `Q:3 | MinB:4 | MaxB:4 | Wait:3 | PrevR:1 | Arr:0 | PrevShort:1`
 *
 * @param tuple - Array of 7 state integers.
 * @returns Formatted description.
 */
export function formatState7Tuple(tuple: number[]): string {
  if (!tuple || tuple.length < 7) return '(—)';
  const [q, minB, maxB, wait, prevR, arr, prevShort] = tuple;
  return `Q:${q} S:${minB} L:${maxB} W:${wait} PR:${prevR} A:${arr} PS:${prevShort}`;
}

/**
 * Q-Learning Policy and State Exploration Table component.
 *
 * 1. The use of function:
 *    Allows engineers to inspect high-frequency MDP states, visit counts, and converged Q-values.
 * 2. How does the specific function work:
 *    Filters `topStates` by text query and action category. Computes pagination slices and
 *    renders table rows with formatted Q-value cells, highlighting optimal values.
 * 3. Concepts used in the function:
 *    Client-side tabular pagination, real-time query filtering, Q-value maximization highlighting.
 * 4. Places where this function is imported and used:
 *    Imported by AILabPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Array of raw Q-table state summaries from /api/rl/policy-summary.
 *    After: Interactive, paginated policy inspection table.
 *
 * @param props - State list, metadata counts, cost variant, and callbacks.
 * @returns JSX Element.
 */
export const PolicyTable: React.FC<PolicyTableProps> = ({
  topStates = [],
  qTableSize = 4695,
  uniqueStatesVisited = 1327,
  csCost = 0,
  onCsCostChange,
  className = '',
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [actionFilter, setActionFilter] = useState('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

  // Filter states based on search query and action
  const filteredStates = useMemo(() => {
    return topStates.filter((item) => {
      const matchesAction =
        actionFilter === 'ALL' || item.best_action_name === actionFilter;
      const matchesQuery =
        !searchQuery ||
        item.state_str.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.best_action_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.visits.toString().includes(searchQuery);

      return matchesAction && matchesQuery;
    });
  }, [topStates, searchQuery, actionFilter]);

  // Pagination calculation
  const totalPages = Math.max(1, Math.ceil(filteredStates.length / pageSize));
  const validPage = Math.min(currentPage, totalPages);
  const startIndex = (validPage - 1) * pageSize;
  const currentRows = filteredStates.slice(startIndex, startIndex + pageSize);

  // Calculate total visits across all states for percentage computation
  const totalVisitsSum = useMemo(() => {
    return topStates.reduce((acc, s) => acc + s.visits, 0);
  }, [topStates]);

  return (
    <div
      className={`p-5 rounded-card border border-border bg-surface shadow-subtle flex flex-col space-y-4 ${className}`}
      data-testid="policy-table"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <Binary className="w-4 h-4 text-primary" />
            Top Visited States & Optimal Policy Actions
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Discretized 7-tuple aggregate state space: (Queue, Shortest, Longest, Wait, PrevReady, PendingArrivals, PrevShortest).
            Total Q-table entries: {qTableSize.toLocaleString()} ({uniqueStatesVisited.toLocaleString()} unique states visited).
          </p>
        </div>

        {onCsCostChange && (
          <div className="inline-flex rounded-control p-1 bg-surface-muted border border-border self-start sm:self-center">
            <button
              type="button"
              onClick={() => onCsCostChange(0)}
              className={`px-2.5 py-0.5 text-xs font-medium rounded transition-colors ${
                csCost === 0
                  ? 'bg-surface text-foreground shadow-xs font-semibold'
                  : 'text-foreground-muted hover:text-foreground'
              }`}
            >
              Cost = 0
            </button>
            <button
              type="button"
              onClick={() => onCsCostChange(1)}
              className={`px-2.5 py-0.5 text-xs font-medium rounded transition-colors ${
                csCost === 1
                  ? 'bg-surface text-foreground shadow-xs font-semibold'
                  : 'text-foreground-muted hover:text-foreground'
              }`}
            >
              Cost = 1
            </button>
          </div>
        )}
      </div>

      {/* Control Bar: Search & Action Filter */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-foreground-muted" />
          <input
            type="text"
            placeholder="Search state tuple or action..."
            value={searchQuery}
            onChange={(e) => {
              setSearchQuery(e.target.value);
              setCurrentPage(1);
            }}
            className="w-full pl-9 pr-3 py-1.5 text-xs rounded-control border border-border bg-surface text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            aria-label="Search States"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-3.5 h-3.5 text-foreground-muted" />
          <select
            value={actionFilter}
            onChange={(e) => {
              setActionFilter(e.target.value);
              setCurrentPage(1);
            }}
            className="px-2.5 py-1.5 text-xs rounded-control border border-border bg-surface text-foreground focus:outline-none focus:ring-1 focus:ring-primary"
            aria-label="Filter by Action"
          >
            <option value="ALL">All Actions ({topStates.length})</option>
            {META_ACTIONS.map((act) => (
              <option key={act} value={act}>
                {act}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto rounded-control border border-border">
        <table className="w-full text-xs text-left border-collapse" aria-label="Policy Summary Table">
          <thead>
            <tr className="border-b border-border bg-surface-muted/50 font-semibold text-foreground">
              <th className="p-2.5">State (7-Tuple)</th>
              <th className="p-2.5">Detailed Breakdown</th>
              <th className="p-2.5 text-right">Visits</th>
              <th className="p-2.5">Optimal Action</th>
              <th className="p-2.5 text-right" title="Shortest Remaining">Q(Shortest)</th>
              <th className="p-2.5 text-right" title="Longest Remaining">Q(Longest)</th>
              <th className="p-2.5 text-right" title="Earliest Arrival">Q(Arrival)</th>
              <th className="p-2.5 text-right" title="Continue Current">Q(Continue)</th>
            </tr>
          </thead>
          <tbody>
            {currentRows.length === 0 ? (
              <tr>
                <td colSpan={8} className="p-8 text-center text-foreground-muted">
                  No states matched the search criteria.
                </td>
              </tr>
            ) : (
              currentRows.map((row, idx) => {
                const visitPct =
                  totalVisitsSum > 0 ? (row.visits / totalVisitsSum) * 100 : 0;
                const badgeClass = getActionBadgeClass(row.best_action_name);

                return (
                  <tr
                    key={row.state_str || idx}
                    className="border-b border-border/40 hover:bg-surface-muted/30 transition-colors"
                  >
                    <td className="p-2.5 font-mono font-semibold text-foreground">
                      {row.state_str}
                    </td>
                    <td className="p-2.5 font-mono text-[11px] text-foreground-muted whitespace-nowrap">
                      {formatState7Tuple(row.state)}
                    </td>
                    <td className="p-2.5 text-right font-mono">
                      <span className="font-semibold text-foreground">
                        {row.visits.toLocaleString()}
                      </span>
                      <span className="text-[10px] text-foreground-muted block">
                        ({visitPct.toFixed(1)}%)
                      </span>
                    </td>
                    <td className="p-2.5">
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${badgeClass}`}
                      >
                        <Award className="w-2.5 h-2.5" />
                        {row.best_action_name}
                      </span>
                    </td>
                    {META_ACTIONS.map((act) => {
                      const qVal = row.q_values?.[act];
                      const isBest = row.best_action_name === act;
                      return (
                        <td
                          key={act}
                          className={`p-2.5 text-right font-mono ${
                            isBest
                              ? 'font-bold text-primary bg-primary/5'
                              : 'text-foreground-muted'
                          }`}
                        >
                          {qVal !== undefined ? qVal.toFixed(1) : '—'}
                        </td>
                      );
                    })}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2 text-xs text-foreground-muted">
        <div className="flex items-center gap-3">
          <span>
            Showing <strong className="text-foreground">{startIndex + 1}</strong> to{' '}
            <strong className="text-foreground">
              {Math.min(startIndex + pageSize, filteredStates.length)}
            </strong>{' '}
            of <strong className="text-foreground">{filteredStates.length}</strong> states
          </span>
          <select
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="px-2 py-1 text-xs rounded border border-border bg-surface text-foreground"
            aria-label="Rows per page"
          >
            <option value={10}>10 rows</option>
            <option value={25}>25 rows</option>
            <option value={50}>50 rows</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            disabled={validPage <= 1}
            className="p-1 rounded border border-border bg-surface hover:bg-surface-muted disabled:opacity-40 disabled:cursor-not-allowed text-foreground"
            aria-label="Previous Page"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="font-mono text-foreground font-semibold">
            {validPage} / {totalPages}
          </span>
          <button
            type="button"
            onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
            disabled={validPage >= totalPages}
            className="p-1 rounded border border-border bg-surface hover:bg-surface-muted disabled:opacity-40 disabled:cursor-not-allowed text-foreground"
            aria-label="Next Page"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default PolicyTable;
