/**
 * @file src/components/sim/ReadyQueueView.tsx
 * 1. What this file does:
 *    Renders the discrete ready queue state and active CPU execution slot at any given
 *    simulation tick (synchronized with the playback playhead).
 * 2. Overall design:
 *    Inspects `ready_queue_log` from the backend simulation response for the entry matching
 *    `currentTick`. Shows the active process running on the CPU core and the waiting queue line.
 * 3. Elements in the file:
 *    - `ReadyQueueViewProps`: TypeScript interface for ready queue log and tick
 *    - `ReadyQueueView`: React functional component
 * 4. Final output of the file:
 *    Exported ready queue visualizer component rendered on the Simulator page.
 */

import React from 'react';
import { Cpu, ArrowRight, ListOrdered, PauseCircle } from 'lucide-react';
import type { ReadyQueueLogEntry } from '@/api/client';
import { getProcessColor } from '@/lib/colors';

export interface ReadyQueueViewProps {
  readyQueueLog: ReadyQueueLogEntry[];
  currentTick: number;
}

/**
 * Visualizes the ready queue line and currently executing process at the current playhead tick.
 *
 * 1. The use of function:
 *    Enables step-by-step inspection of CPU scheduling decisions and queue ordering over time.
 * 2. How does the specific function work:
 *    Locates the ReadyQueueLogEntry for the requested tick, renders the CPU Core socket
 *    (running process vs IDLE / Context Switching), and maps waiting PIDs with process badges.
 * 3. Concepts used in the function:
 *    Discrete-event simulation log lookup, CPU core status representation, FIFO queue visualization.
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Array of discrete tick log snapshots.
 *    After: Interactive visual representation of CPU and Ready Queue state at t = currentTick.
 *
 * @param props - Component parameters conforming to ReadyQueueViewProps.
 * @returns JSX Element rendering the ready queue and CPU socket.
 */
export const ReadyQueueView: React.FC<ReadyQueueViewProps> = ({
  readyQueueLog,
  currentTick,
}) => {
  // Find log entry for the current tick, falling back to closest previous entry or null
  const currentEntry = readyQueueLog.length > 0
    ? readyQueueLog.find((e) => e.time === currentTick) ||
      readyQueueLog[Math.min(currentTick, readyQueueLog.length - 1)]
    : null;

  const runningPid = currentEntry?.running_pid ?? null;
  const readyPids = currentEntry?.ready_pids ?? [];

  return (
    <div className="p-5 rounded-card border border-border bg-surface shadow-subtle space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <h3 className="font-semibold text-sm text-foreground flex items-center gap-2">
          <ListOrdered className="w-4 h-4 text-primary" />
          Ready Queue & CPU State
        </h3>
        <span className="text-xs text-foreground-muted font-mono">
          Snapshot at <strong className="text-primary font-bold">t = {currentTick}</strong>
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* CPU Core Execution Slot (1 Column) */}
        <div className="p-4 rounded-control border border-border bg-surface-muted flex flex-col justify-between space-y-3">
          <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-primary" />
            CPU Core
          </span>

          <div className="flex items-center justify-center py-2">
            {runningPid ? (
              <div className="flex flex-col items-center gap-1.5 animate-fadeIn">
                <div
                  style={{ backgroundColor: getProcessColor(runningPid) }}
                  className="w-12 h-12 rounded-full flex items-center justify-center text-white font-mono font-bold text-lg shadow-md"
                >
                  {runningPid}
                </div>
                <span className="text-xs font-semibold text-success flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-success animate-ping" />
                  RUNNING
                </span>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-1 text-foreground-muted py-1">
                <PauseCircle className="w-8 h-8 opacity-40" />
                <span className="text-xs font-mono font-medium">
                  {readyPids.length > 0 ? 'CONTEXT SWITCH' : 'IDLE'}
                </span>
                <span className="text-[10px] opacity-75">No process on CPU</span>
              </div>
            )}
          </div>

          <div className="text-[11px] text-foreground-muted text-center font-mono border-t border-border/60 pt-2">
            Status: {runningPid ? `Active: ${runningPid}` : readyPids.length > 0 ? 'Switching' : 'Idle'}
          </div>
        </div>

        {/* Ready Queue Line (2 Columns) */}
        <div className="sm:col-span-2 p-4 rounded-control border border-border bg-surface-muted flex flex-col justify-between space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-foreground-muted uppercase tracking-wider">
              Ready Queue (Waiting)
            </span>
            <span className="text-xs font-mono font-bold text-primary">
              {readyPids.length} process{readyPids.length === 1 ? '' : 'es'}
            </span>
          </div>

          <div className="py-2 min-h-[60px] flex items-center">
            {readyPids.length === 0 ? (
              <div className="w-full text-center py-3 text-xs text-foreground-muted font-sans">
                Queue is empty (no processes waiting in ready state at t = {currentTick}).
              </div>
            ) : (
              <div className="flex items-center gap-2 overflow-x-auto w-full py-1">
                <span className="text-[10px] uppercase font-bold text-foreground-muted mr-1 font-mono">
                  Head
                </span>
                {readyPids.map((pid, idx) => {
                  const color = getProcessColor(pid);
                  return (
                    <React.Fragment key={`${pid}-${idx}`}>
                      <div
                        style={{ borderColor: color }}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-control bg-surface border shadow-sm flex-shrink-0"
                      >
                        <span
                          style={{ backgroundColor: color }}
                          className="w-2.5 h-2.5 rounded-full"
                        />
                        <span className="text-xs font-mono font-bold text-foreground">
                          {pid}
                        </span>
                        <span className="text-[10px] text-foreground-muted font-mono">
                          #{idx + 1}
                        </span>
                      </div>
                      {idx < readyPids.length - 1 && (
                        <ArrowRight className="w-3.5 h-3.5 text-foreground-muted flex-shrink-0" />
                      )}
                    </React.Fragment>
                  );
                })}
                <span className="text-[10px] uppercase font-bold text-foreground-muted ml-1 font-mono">
                  Tail
                </span>
              </div>
            )}
          </div>

          <div className="text-[11px] text-foreground-muted font-mono border-t border-border/60 pt-2 flex justify-between">
            <span>Dispatched via scheduling policy order</span>
            <span>Tick: t={currentTick}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
