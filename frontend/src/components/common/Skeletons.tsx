/**
 * @file src/components/common/Skeletons.tsx
 * 1. What this file does:
 *    Renders animated skeleton loaders matching the exact structural layout of tables,
 *    metrics cards, bar charts, and stacked Gantt timelines.
 * 2. Overall design:
 *    Uses CSS pulse animations with theme-aware surface-muted placeholders to prevent
 *    layout shift during asynchronous query/mutation fetches.
 * 3. Elements in the file:
 *    - `CardSkeleton`: Single metric or summary card placeholder
 *    - `TableSkeleton`: Tabular row and column wireframe loader
 *    - `ChartSkeleton`: Grouped chart canvas placeholder
 *    - `GanttStackSkeleton`: Multi-row Gantt timeline wireframe loader
 * 4. Final output of the file:
 *    Exported skeleton building blocks used across all async views in the dashboard.
 */

import React from 'react';

/**
 * Skeleton placeholder for statistical metric cards.
 *
 * 1. The use of function:
 *    Renders a placeholder for stat cards while awaiting API responses.
 * 2. How does the specific function work:
 *    Displays an animated pulse box matching standard card height and borders.
 * 3. Concepts used in the function:
 *    Content layout preservation, zero layout shifts.
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx, ComparePage.tsx, and ResultsPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Blank card area.
 *    After: Animated pulse card layout.
 *
 * @returns JSX Element.
 */
export const CardSkeleton: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div
    className={`p-4 rounded-card border border-border bg-surface animate-pulse space-y-3 ${className}`}
  >
    <div className="h-3.5 w-1/3 bg-surface-muted rounded" />
    <div className="h-7 w-1/2 bg-surface-muted rounded" />
    <div className="h-3 w-3/4 bg-surface-muted rounded" />
  </div>
);

/**
 * Skeleton placeholder for tabular data layouts.
 *
 * 1. The use of function:
 *    Provides animated wireframe for ComparisonTable or ProcessMetricsTable.
 * 2. How does the specific function work:
 *    Renders mock header cells and N pulsing table rows.
 * 3. Concepts used in the function:
 *    Repetitive structural mapping, CSS pulse animation.
 * 4. Places where this function is imported and used:
 *    Imported by ComparisonTable.tsx and ComparePage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Missing table contents.
 *    After: Structured tabular skeleton.
 *
 * @param rows - Number of wireframe rows to render (default 5).
 * @returns JSX Element.
 */
export const TableSkeleton: React.FC<{ rows?: number }> = ({ rows = 5 }) => (
  <div className="p-4 rounded-card border border-border bg-surface animate-pulse space-y-4">
    <div className="flex items-center justify-between pb-3 border-b border-border">
      <div className="h-4 w-40 bg-surface-muted rounded" />
      <div className="h-7 w-28 bg-surface-muted rounded" />
    </div>
    <div className="space-y-3">
      <div className="h-8 bg-surface-muted/60 rounded" />
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="h-10 bg-surface-muted/40 rounded" />
      ))}
    </div>
  </div>
);

/**
 * Skeleton placeholder for Recharts bar or line charts.
 *
 * 1. The use of function:
 *    Renders a placeholder canvas for ComparisonBars while benchmark finishes.
 * 2. How does the specific function work:
 *    Renders chart header and simulated vertical bar pillars with varying heights.
 * 3. Concepts used in the function:
 *    Visual cadence matching, wireframe simulation.
 * 4. Places where this function is imported and used:
 *    Imported by ComparisonBars.tsx and ComparePage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Blank chart section.
 *    After: Wireframe bars with pulse animation.
 *
 * @returns JSX Element.
 */
export const ChartSkeleton: React.FC = () => (
  <div className="p-5 rounded-card border border-border bg-surface animate-pulse space-y-4">
    <div className="flex items-center justify-between pb-3 border-b border-border">
      <div className="h-4 w-48 bg-surface-muted rounded" />
      <div className="h-4 w-32 bg-surface-muted rounded" />
    </div>
    <div className="h-64 flex items-end justify-around gap-4 pt-8 pb-4">
      {[45, 75, 30, 90, 60, 85, 40].map((h, i) => (
        <div
          key={i}
          style={{ height: `${h}%` }}
          className="w-12 bg-surface-muted rounded-t"
        />
      ))}
    </div>
    <div className="h-4 w-full bg-surface-muted/50 rounded" />
  </div>
);

/**
 * Skeleton placeholder for stacked Gantt timelines across multiple algorithms.
 *
 * 1. The use of function:
 *    Pre-allocates the vertical space and layout for GanttStack.
 * 2. How does the specific function work:
 *    Renders multiple stacked timeline rows with mock algorithm badges.
 * 3. Concepts used in the function:
 *    Synchronized height matching, realistic timeline layout.
 * 4. Places where this function is imported and used:
 *    Imported by GanttStack.tsx and ComparePage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Unloaded timeline area.
 *    After: Animated stack of timeline tracks.
 *
 * @param rows - Number of algorithm rows to render (default 5).
 * @returns JSX Element.
 */
export const GanttStackSkeleton: React.FC<{ rows?: number }> = ({ rows = 5 }) => (
  <div className="p-5 rounded-card border border-border bg-surface animate-pulse space-y-4">
    <div className="flex items-center justify-between pb-3 border-b border-border">
      <div className="h-4 w-52 bg-surface-muted rounded" />
      <div className="h-6 w-24 bg-surface-muted rounded" />
    </div>
    <div className="space-y-4">
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="space-y-1.5">
          <div className="flex items-center justify-between">
            <div className="h-3.5 w-24 bg-surface-muted rounded" />
            <div className="h-3 w-16 bg-surface-muted rounded" />
          </div>
          <div className="h-10 bg-surface-muted/50 rounded border border-border/50" />
        </div>
      ))}
    </div>
    <div className="h-5 w-full bg-surface-muted/60 rounded" />
  </div>
);
