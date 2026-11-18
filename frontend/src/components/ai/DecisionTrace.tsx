/**
 * @file src/components/ai/DecisionTrace.tsx
 * 1. What this file does:
 *    Renders an interactive step-by-step Decision Trace table for the Tabular Q-Learning scheduler,
 *    displaying state representations, valid actions, estimated Q-values, chosen action, and reward.
 * 2. Overall design:
 *    Highlights the row corresponding to the current playback tick. Clicking any decision row
 *    immediately synchronizes the playback store tick to that execution moment.
 * 3. Elements in the file:
 *    - `DecisionTraceProps`: TypeScript interface for trace entries and selection callbacks
 *    - `DecisionTrace`: React functional component
 * 4. Final output of the file:
 *    Exported explainability component rendered when Q-Learning is simulated.
 */

import React from 'react';
import { Brain, CheckCircle2, Navigation } from 'lucide-react';
import type { DecisionTraceEntry } from '@/api/client';
import { formatNumber } from '@/lib/format';

export interface DecisionTraceProps {
  trace: DecisionTraceEntry[];
  currentTick: number;
  onSelectTick: (tick: number) => void;
}

/**
 * Interactive tabular decision trace visualizer for Reinforcement Learning scheduling policies.
 *
 * 1. The use of function:
 *    Exposes RL agent decision-making transparency, state vector discretizations, and Q-value rankings.
 * 2. How does the specific function work:
 *    Renders each DecisionTraceEntry, highlights the active playhead tick, and attaches row-click
 *    callbacks to scrub the Gantt timeline to the corresponding simulation step.
 * 3. Concepts used in the function:
 *    XAI (Explainable AI), Q-value inspection, synchronized playback scrub, tabular formatting.
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx and AILabPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Array of raw RL agent decision logs.
 *    After: Interactive explainability table with live playhead synchronization and Q-value cards.
 *
 * @param props - Component parameters conforming to DecisionTraceProps.
 * @returns JSX Element rendering the decision trace table.
 */
export const DecisionTrace: React.FC<DecisionTraceProps> = ({
  trace,
  currentTick,
  onSelectTick,
}) => {
  if (trace.length === 0) {
    return null;
  }

  return (
    <div className="p-5 rounded-card border border-border bg-surface shadow-subtle space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-border">
        <div>
          <h3 className="font-semibold text-sm text-foreground flex items-center gap-2">
            <Brain className="w-4 h-4 text-purple-600 dark:text-purple-400" />
            Q-Learning Agent Decision Trace & Explainability
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Click any row to seek the Gantt timeline to that discrete decision tick.
          </p>
        </div>
        <span className="text-xs font-mono font-medium text-purple-600 dark:text-purple-400 bg-purple-500/10 px-2.5 py-1 rounded">
          {trace.length} Decisions Logged
        </span>
      </div>

      {/* Decision Table */}
      <div className="overflow-x-auto max-h-[360px] overflow-y-auto border border-border rounded-control">
        <table className="w-full text-left text-xs border-collapse">
          <thead className="sticky top-0 bg-surface-muted border-b border-border z-10 text-foreground-muted">
            <tr>
              <th className="py-2.5 px-3 font-semibold">Tick (t)</th>
              <th className="py-2.5 px-3 font-semibold">Discrete State Vector</th>
              <th className="py-2.5 px-3 font-semibold">Chosen Action</th>
              <th className="py-2.5 px-3 font-semibold">Chosen PID</th>
              <th className="py-2.5 px-3 font-semibold">Q-Values (Action Utilities)</th>
              <th className="py-2.5 px-3 font-semibold text-right">Reward</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {trace.map((entry, idx) => {
              const isSelected = entry.time === currentTick;
              const rewardFavorable = entry.reward >= 0;

              return (
                <tr
                  key={`${entry.time}-${idx}`}
                  onClick={() => onSelectTick(entry.time)}
                  className={`cursor-pointer transition-colors font-mono-numbers ${
                    isSelected
                      ? 'bg-purple-500/15 border-l-4 border-l-purple-600 dark:border-l-purple-400 font-medium'
                      : 'hover:bg-surface-muted/60'
                  }`}
                  title="Click to scrub playback to this tick"
                >
                  {/* Tick */}
                  <td className="py-2 px-3 font-mono font-bold text-foreground">
                    <span className="flex items-center gap-1.5">
                      {isSelected && (
                        <Navigation className="w-3 h-3 text-purple-600 fill-purple-600 rotate-90 flex-shrink-0" />
                      )}
                      t={entry.time}
                    </span>
                  </td>

                  {/* State Vector */}
                  <td className="py-2 px-3 font-mono text-[11px] text-foreground-muted max-w-[160px] truncate">
                    [{entry.state.join(', ')}]
                  </td>

                  {/* Chosen Action */}
                  <td className="py-2 px-3 font-mono">
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-purple-500/10 text-purple-700 dark:text-purple-300 font-semibold text-[11px]">
                      <CheckCircle2 className="w-3 h-3 flex-shrink-0" />
                      {entry.chosen_action}
                    </span>
                  </td>

                  {/* Chosen PID */}
                  <td className="py-2 px-3 font-mono font-bold text-foreground">
                    {entry.chosen_pid}
                  </td>

                  {/* Q-Values */}
                  <td className="py-2 px-3">
                    <div className="flex flex-wrap gap-1.5 text-[10px] font-mono">
                      {Object.entries(entry.q_values).map(([act, qVal]) => {
                        const isChosen = act === entry.chosen_action;
                        return (
                          <span
                            key={act}
                            className={`px-1.5 py-0.5 rounded border ${
                              isChosen
                                ? 'bg-purple-600 text-white font-bold border-purple-700'
                                : 'bg-surface-muted text-foreground-muted border-border'
                            }`}
                          >
                            {act}: {formatNumber(qVal, 2)}
                          </span>
                        );
                      })}
                    </div>
                  </td>

                  {/* Reward */}
                  <td className="py-2 px-3 text-right font-mono font-bold">
                    <span
                      className={
                        rewardFavorable ? 'text-success' : 'text-danger'
                      }
                    >
                      {entry.reward >= 0 ? `+${formatNumber(entry.reward, 1)}` : formatNumber(entry.reward, 1)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
