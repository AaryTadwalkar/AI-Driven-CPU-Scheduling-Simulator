/**
 * @file src/components/common/StatCard.tsx
 * 1. What this file does:
 *    Renders a standardized statistical metric card with label, tabular numeric value,
 *    unit indicator, informational tooltip definition, and optional comparative delta badge.
 * 2. Overall design:
 *    Uses CSS variables and tabular monospace numbers (font-mono-numbers) for engineering clarity.
 *    Handles favorable vs unfavorable delta coloring cleanly.
 * 3. Elements in the file:
 *    - `StatCardProps`: Interface defining card props and optional baseline comparison
 *    - `StatCard`: React functional component
 * 4. Final output of the file:
 *    Exported component used by MetricCards, ComparePage, and ResultsPage.
 */

import React, { useState } from 'react';
import { HelpCircle } from 'lucide-react';

export interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  definition?: string;
  delta?: {
    text: string;
    isFavorable: boolean;
    colorClass: string;
  };
  icon?: React.ReactNode;
  isLoading?: boolean;
  className?: string;
}

/**
 * Single statistical card rendering a key performance indicator (KPI).
 *
 * 1. The use of function:
 *    Presents scheduling summary metrics with context, tooltips, and comparison diffs.
 * 2. How does the specific function work:
 *    Renders an accessible card container with title, icon, value, optional unit,
 *    and a hover-activated definition tooltip.
 * 3. Concepts used in the function:
 *    Tabular typography, accessible tooltip triggers, directional styling.
 * 4. Places where this function is imported and used:
 *    Imported by MetricCards.tsx and dashboard analytics pages.
 * 5. Final output of the function, before and after:
 *    Before: Raw numeric value.
 *    After: Structured engineering KPI card.
 *
 * @param props - Component parameters conforming to StatCardProps.
 * @returns JSX Element for the metric card.
 */
export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  unit,
  definition,
  delta,
  icon,
  isLoading = false,
  className = '',
}) => {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div
      className={`p-4 rounded-card border border-border bg-surface shadow-subtle flex flex-col justify-between relative transition-shadow hover:shadow-md ${className}`}
    >
      {/* Header: Label, Icon, and Info Tooltip */}
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className="font-medium text-foreground-muted truncate flex items-center gap-1.5">
          {icon && <span className="text-primary">{icon}</span>}
          {label}
        </span>
        {definition && (
          <div className="relative">
            <button
              type="button"
              onMouseEnter={() => setShowTooltip(true)}
              onMouseLeave={() => setShowTooltip(false)}
              onFocus={() => setShowTooltip(true)}
              onBlur={() => setShowTooltip(false)}
              aria-label={`Definition for ${label}`}
              className="text-foreground-muted hover:text-foreground transition-colors p-0.5 rounded"
            >
              <HelpCircle className="w-3.5 h-3.5" />
            </button>
            {showTooltip && (
              <div className="absolute right-0 top-6 z-30 w-52 p-2.5 rounded-control bg-surface-muted border border-border text-[11px] text-foreground shadow-lg leading-relaxed pointer-events-none">
                {definition}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main Metric Value */}
      <div className="mt-3 flex items-baseline gap-1.5">
        {isLoading ? (
          <div className="h-7 w-20 bg-surface-muted rounded animate-pulse" />
        ) : (
          <>
            <span className="text-2xl font-bold font-mono-numbers text-foreground tracking-tight">
              {value}
            </span>
            {unit && (
              <span className="text-xs text-foreground-muted font-normal font-sans">
                {unit}
              </span>
            )}
          </>
        )}
      </div>

      {/* Optional Comparison Delta */}
      {delta && !isLoading && (
        <div className="mt-2 text-xs font-mono flex items-center gap-1">
          <span className={`font-semibold ${delta.colorClass}`}>
            {delta.text}
          </span>
          <span className="text-[10px] text-foreground-muted font-sans">
            vs baseline
          </span>
        </div>
      )}
    </div>
  );
};
