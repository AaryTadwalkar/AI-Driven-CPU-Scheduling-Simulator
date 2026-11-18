/**
 * @file src/lib/format.ts
 * 1. What this file does:
 *    Provides clean, standardized formatting helpers for metric numbers, percentages,
 *    deltas, algorithm display labels, and timestamps.
 * 2. Overall design:
 *    Lightweight, pure functions ensuring tabular, predictable number representations
 *    matching the engineering dashboard requirements.
 * 3. Elements in the file:
 *    - `formatNumber`: formats numbers with fixed decimal places
 *    - `formatPercent`: formats ratios or values into human-readable percentages
 *    - `formatDelta`: formats improvement/degradation differences with semantic colors
 *    - `formatMeanStd`: formats statistical mean ± std / CI
 *    - `formatAlgorithmName`: human-friendly algorithm title generator
 * 4. Final output of the file:
 *    Exported utility functions for rendering in tables, cards, tooltips, and badges.
 */

/**
 * Formats a numeric value with a specified number of decimal places.
 *
 * 1. The use of function:
 *    Renders metrics such as average waiting time, turnaround, throughput.
 * 2. How does the specific function work:
 *    Handles null/undefined/NaN by returning '—', otherwise uses toFixed(decimals).
 * 3. Concepts used in the function:
 *    Type guards, safe numeric coercion, tabular numeric strings.
 * 4. Places where this function is imported and used:
 *    Imported by MetricCards, ProcessMetricsTable, ComparisonTable, StatCard.
 * 5. Final output of the function, before and after:
 *    Before: `formatNumber(3.14159, 2)`
 *    After: `'3.14'`
 *
 * @param value - Number or undefined.
 * @param decimals - Decimal places (default 2).
 * @returns Formatted string.
 */
export function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—';
  }
  return value.toFixed(decimals);
}

/**
 * Formats a fraction (0.0 to 1.0 or 0 to 100) as a formatted percentage string.
 *
 * 1. The use of function:
 *    Renders CPU utilization, recommendation confidence, or accuracy.
 * 2. How does the specific function work:
 *    Detects whether value is normalized (<= 1) or already multiplied by 100, then formats with '%'.
 * 3. Concepts used in the function:
 *    Scale normalization, clamp boundary protection.
 * 4. Places where this function is imported and used:
 *    Imported by MetricCards, ProbabilityBars, RecommendationCard.
 * 5. Final output of the function, before and after:
 *    Before: `formatPercent(0.874, 1)`
 *    After: `'87.4%'`
 *
 * @param value - Normalized ratio or percentage.
 * @param decimals - Decimal places (default 1).
 * @returns Formatted percentage string.
 */
export function formatPercent(value: number | null | undefined, decimals = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—%';
  }
  const pct = value <= 1.0 ? value * 100 : value;
  return `${pct.toFixed(decimals)}%`;
}

/**
 * Formats difference between a test value and baseline, taking 'lower is better' into account.
 *
 * 1. The use of function:
 *    Renders comparison deltas in stat cards and benchmark tables.
 * 2. How does the specific function work:
 *    Computes signed difference, determines if the change is favorable or unfavorable based
 *    on lowerIsBetter, and returns sign, delta text, and semantic Tailwind text color.
 * 3. Concepts used in the function:
 *    Directional delta calculation, semantic CSS variable assignment.
 * 4. Places where this function is imported and used:
 *    Imported by StatCard, ComparisonTable.
 * 5. Final output of the function, before and after:
 *    Before: `formatDelta(3.5, 4.0, true)` (waiting time dropped from 4 to 3.5)
 *    After: `{ text: '-0.50 (-12.5%)', isFavorable: true, colorClass: 'text-success' }`
 *
 * @param current - Current value.
 * @param baseline - Baseline value to compare against.
 * @param lowerIsBetter - True if a smaller value represents improvement (e.g. waiting time).
 */
export function formatDelta(
  current: number,
  baseline: number,
  lowerIsBetter = true
): { text: string; isFavorable: boolean; colorClass: string } {
  const diff = current - baseline;
  if (Math.abs(diff) < 0.0001) {
    return { text: '0.00', isFavorable: true, colorClass: 'text-foreground-muted' };
  }
  const pctDiff = baseline !== 0 ? (diff / baseline) * 100 : 0;
  const sign = diff > 0 ? '+' : '';
  const text = `${sign}${diff.toFixed(2)} (${sign}${pctDiff.toFixed(1)}%)`;

  const isFavorable = lowerIsBetter ? diff < 0 : diff > 0;
  const colorClass = isFavorable ? 'text-success' : 'text-danger';

  return { text, isFavorable, colorClass };
}

/**
 * Formats a mean and standard deviation / confidence interval pair.
 *
 * 1. The use of function:
 *    Renders benchmark summary statistics (e.g. 14.25 ± 1.12).
 * 2. How does the specific function work:
 *    Formats both values to 2 decimal places and joins with the plus-minus symbol.
 * 3. Concepts used in the function:
 *    Statistical representation standards.
 * 4. Places where this function is imported and used:
 *    Imported by ResultsPage, ComparisonTable.
 * 5. Final output of the function, before and after:
 *    Before: `formatMeanStd(12.345, 0.678)`
 *    After: `'12.35 ± 0.68'`
 *
 * @param mean - Mean value.
 * @param std - Standard deviation or confidence margin.
 * @returns String in format `mean ± std`.
 */
export function formatMeanStd(mean: number, std: number): string {
  return `${formatNumber(mean, 2)} ± ${formatNumber(std, 2)}`;
}

/**
 * Formats an algorithm identifier code into an informative display title.
 *
 * 1. The use of function:
 *    Presents user-friendly algorithm names in dropdowns, cards, and chart headers.
 * 2. How does the specific function work:
 *    Looks up normalized uppercase key in dictionary of acronym expansions.
 * 3. Concepts used in the function:
 *    Dictionary lookup with default identity fallback.
 * 4. Places where this function is imported and used:
 *    Imported by AlgorithmPicker, ComparisonTable, GanttChart, Sidebar.
 * 5. Final output of the function, before and after:
 *    Before: `'SRTF'`
 *    After: `'Shortest Remaining Time First (SRTF)'`
 *
 * @param algorithm - Algorithm code (e.g. 'FCFS', 'SJF', 'RR', 'SRTF', 'Q_LEARNING').
 * @returns Human-friendly algorithm name.
 */
export function formatAlgorithmName(algorithm: string): string {
  const norm = algorithm.trim().toUpperCase().replace(/-/g, '_');
  const map: Record<string, string> = {
    FCFS: 'First-Come First-Served (FCFS)',
    SJF: 'Shortest Job First (SJF)',
    RR: 'Round Robin (RR)',
    SRTF: 'Shortest Remaining Time First (SRTF)',
    Q_LEARNING: 'Q-Learning RL Agent (AI)',
    AI: 'Q-Learning RL Agent (AI)',
    RL: 'Q-Learning RL Agent (AI)',
    QLEARNING: 'Q-Learning RL Agent (AI)',
  };
  return map[norm] ?? algorithm;
}
