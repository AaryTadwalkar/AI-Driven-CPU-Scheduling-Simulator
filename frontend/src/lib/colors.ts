/**
 * @file src/lib/colors.ts
 * 1. What this file does:
 *    Provides color palettes, deterministic color assignments for process IDs (PIDs),
 *    special scheduling states (IDLE, CS), and standardized algorithm branding colors.
 * 2. Overall design:
 *    Implements Okabe-Ito extended colorblind-safe categorical palette for process segments,
 *    guaranteeing that the same PID receives the same color across all pages and charts.
 * 3. Elements in the file:
 *    - `PROCESS_PALETTE`: 12 categorical hex colors
 *    - `SPECIAL_SEGMENT_COLORS`: color definitions for IDLE and CS
 *    - `ALGORITHM_COLORS`: color map for FCFS, SJF, RR, SRTF, and Q-Learning
 *    - `getProcessColor`: function mapping PID string to a deterministic palette color
 *    - `getAlgorithmColor`: function mapping algorithm code to its brand color
 * 4. Final output of the file:
 *    Exported constants and lookup functions consumed by Gantt, charts, and metrics tables.
 */

export const PROCESS_PALETTE: readonly string[] = [
  '#0072B2', // 0: Deep Blue
  '#E69F00', // 1: Warm Amber/Orange
  '#009E73', // 2: Bluish Green
  '#D55E00', // 3: Vermilion
  '#CC79A7', // 4: Reddish Purple
  '#56B4E9', // 5: Sky Blue
  '#E5C494', // 6: Soft Sand / Tan
  '#4B0082', // 7: Indigo
  '#33A02C', // 8: Dark Green
  '#FB9A99', // 9: Light Coral
  '#A6CEE3', // 10: Light Steel Blue
  '#B15928', // 11: Rust Brown
];

export const SPECIAL_SEGMENT_COLORS = {
  IDLE: '#94A3B8', // Slate-400
  CS: '#D97706',   // Amber-600
  AI: '#EC4899',   // Pink-500 AI stall overhead
} as const;

export const ALGORITHM_COLORS: Record<string, string> = {
  FCFS: '#3B82F6',        // Blue
  SJF: '#10B981',         // Green
  RR: '#F97316',          // Orange
  SRTF: '#8B5CF6',        // Purple
  Q_LEARNING: '#EC4899',  // Pink/Magenta
  AI: '#EC4899',          // Alias for Q-Learning
  RL: '#EC4899',          // Alias for RL agent
  QLEARNING: '#EC4899',   // Alias for Q-Learning
  DT: '#06B6D4',          // Cyan
  RF: '#14B8A6',          // Teal
  PID_GBDT: '#E11D48',    // Rose
  PID_NN: '#8B5CF6',      // Violet
};

/**
 * Deterministically resolves a hex color for any process ID.
 *
 * 1. The use of function:
 *    Ensures consistent visual identity for a process across Gantt chart, metrics table,
 *    and ready queue visualizers.
 * 2. How does the specific function work:
 *    Parses trailing digits if PID starts with 'P' or 'p' (e.g. 'P1' -> index 0),
 *    falling back to a string hash modulo the palette size for arbitrary alphanumeric IDs.
 * 3. Concepts used in the function:
 *    Deterministic hashing, regex digit extraction, zero-based modulo indexing.
 * 4. Places where this function is imported and used:
 *    Imported by GanttChart, ProcessMetricsTable, ReadyQueueView, ProcessTable.
 * 5. Final output of the function, before and after:
 *    Before: 'P1' -> After: '#0072B2'
 *    Before: 'P2' -> After: '#E69F00'
 *    Before: 'IDLE' -> After: '#94A3B8'
 *    Before: 'CS' -> After: '#D97706'
 *
 * @param pid - The process identifier string (e.g. 'P1', 'P2') or special token ('IDLE', 'CS').
 * @returns Hex color string.
 */
export function getProcessColor(pid: string): string {
  const upper = pid.trim().toUpperCase();
  if (upper === 'IDLE') return SPECIAL_SEGMENT_COLORS.IDLE;
  if (upper === 'CS' || upper === 'CONTEXT_SWITCH') return SPECIAL_SEGMENT_COLORS.CS;

  // Check for common format P<number>
  const match = upper.match(/^P(\d+)$/);
  if (match) {
    const num = parseInt(match[1], 10);
    const index = Math.max(0, num - 1) % PROCESS_PALETTE.length;
    return PROCESS_PALETTE[index];
  }

  // Fallback hash for custom PIDs (e.g. 'worker-a')
  let hash = 0;
  for (let i = 0; i < upper.length; i++) {
    hash = (hash << 5) - hash + upper.charCodeAt(i);
    hash |= 0;
  }
  const index = Math.abs(hash) % PROCESS_PALETTE.length;
  return PROCESS_PALETTE[index];
}

/**
 * Resolves the official color for a scheduling algorithm.
 *
 * 1. The use of function:
 *    Used in comparison charts, badges, and legend items to color-code algorithms.
 * 2. How does the specific function work:
 *    Normalizes input algorithm name to uppercase and checks the ALGORITHM_COLORS dictionary,
 *    defaulting to primary color if unrecognized.
 * 3. Concepts used in the function:
 *    Key normalization, safe dictionary lookup with fallback.
 * 4. Places where this function is imported and used:
 *    Imported by ComparisonBars, ComparisonTable, GanttStack, StatCard.
 * 5. Final output of the function, before and after:
 *    Before: 'fcfs' -> After: '#3B82F6'
 *    Before: 'SJF' -> After: '#10B981'
 *
 * @param algorithm - Algorithm identifier string.
 * @returns Hex color code.
 */
export function getAlgorithmColor(algorithm: string): string {
  const norm = algorithm.trim().toUpperCase().replace(/-/g, '_');
  return ALGORITHM_COLORS[norm] ?? '#64748B';
}
