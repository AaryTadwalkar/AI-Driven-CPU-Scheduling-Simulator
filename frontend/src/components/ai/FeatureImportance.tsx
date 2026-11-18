/**
 * @file src/components/ai/FeatureImportance.tsx
 * 1. What this file does:
 *    Renders a Recharts horizontal bar chart displaying ranked Gini feature importances for the Decision Tree.
 * 2. Overall design:
 *    Transforms raw feature importance weights into a sorted, descending dataset with human-readable
 *    feature descriptions, percentage values, and interactive tooltips.
 * 3. Elements in the file:
 *    - `FEATURE_LABEL_MAP`: Dictionary mapping raw feature keys to human-readable names
 *    - `formatFeatureLabel`: Helper resolving clean display labels
 *    - `FeatureImportanceProps`: Interface defining component inputs and style overrides
 *    - `FeatureImportance`: Primary React functional component
 * 4. Final output of the file:
 *    Visual Gini feature importance ranking card consumed by AILabPage.
 */

import React, { useMemo } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { BarChart3 } from 'lucide-react';
import { formatPercent } from '@/lib/format';

export interface FeatureImportanceProps {
  importances?: Record<string, number>;
  className?: string;
}

export const FEATURE_LABEL_MAP: Record<string, string> = {
  n: 'Process Count (n)',
  mean_burst: 'Mean Burst Time',
  std_burst: 'Burst Std. Dev',
  cv_burst: 'Burst Coeff. of Variation',
  max_burst: 'Max Burst Time',
  min_burst: 'Min Burst Time',
  max_over_median_burst: 'Max / Median Burst Ratio',
  mean_interarrival: 'Mean Interarrival Time',
  std_interarrival: 'Interarrival Std. Dev',
  arrival_span: 'Arrival Span (Window)',
  load_ratio: 'Load Ratio (Work / Span)',
  frac_simultaneous: 'Frac. Simultaneous Arrivals',
};

// Fallback values from Stage 6 model training if API log is still loading
const DEFAULT_IMPORTANCES: Record<string, number> = {
  max_burst: 0.4300,
  std_burst: 0.2991,
  cv_burst: 0.0772,
  frac_simultaneous: 0.0764,
  n: 0.0659,
  load_ratio: 0.0329,
  mean_interarrival: 0.0124,
  mean_burst: 0.0051,
  std_interarrival: 0.0006,
  max_over_median_burst: 0.0003,
  min_burst: 0.0,
  arrival_span: 0.0,
};

/**
 * Resolves a human-readable display label for a raw feature column key.
 *
 * 1. The use of function:
 *    Translates raw feature strings like 'cv_burst' into publication-grade labels.
 * 2. How does the specific function work:
 *    Looks up key in FEATURE_LABEL_MAP; falls back to capitalized snake_case formatting.
 * 3. Concepts used in the function:
 *    Dictionary lookup with string transformation fallback.
 * 4. Places where this function is imported and used:
 *    Imported by FeatureImportance.tsx and LiveRecommender.tsx.
 * 5. Final output of the function, before and after:
 *    Before: 'cv_burst' -> After: 'Burst Coeff. of Variation'
 *
 * @param key - Raw feature column name.
 * @returns Human-readable label string.
 */
export function formatFeatureLabel(key: string): string {
  if (FEATURE_LABEL_MAP[key]) {
    return FEATURE_LABEL_MAP[key];
  }
  return key
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

/**
 * Feature Importance bar chart component for Decision Tree explainability.
 *
 * 1. The use of function:
 *    Visualizes the ranked Gini impurity reduction contributed by each workload feature.
 * 2. How does the specific function work:
 *    Extracts key-value pairs from `importances`, sorts them descending by score,
 *    and feeds them into a vertical layout Recharts BarChart.
 * 3. Concepts used in the function:
 *    Horizontal bar rendering, Gini feature importance ranking, responsive SVG charting.
 * 4. Places where this function is imported and used:
 *    Imported by AILabPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Key-value dictionary of importances.
 *    After: Interactive horizontal bar chart sorted by relevance.
 *
 * @param props - Importances map and styling classes.
 * @returns JSX Element.
 */
export const FeatureImportance: React.FC<FeatureImportanceProps> = ({
  importances,
  className = '',
}) => {
  const chartData = useMemo(() => {
    const source =
      importances && Object.keys(importances).length > 0
        ? importances
        : DEFAULT_IMPORTANCES;

    return Object.entries(source)
      .map(([rawKey, value]) => ({
        rawKey,
        label: formatFeatureLabel(rawKey),
        importance: value,
        pct: value * 100,
      }))
      .sort((a, b) => b.importance - a.importance);
  }, [importances]);

  return (
    <div
      className={`p-5 rounded-card border border-border bg-surface shadow-subtle flex flex-col space-y-4 ${className}`}
      data-testid="feature-importance"
    >
      <div className="pb-3 border-b border-border">
        <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-primary" />
          Gini Feature Importance Ranking
        </h3>
        <p className="text-xs text-foreground-muted mt-0.5">
          Workload features ranked by normalized Gini impurity reduction during Decision Tree training.
        </p>
      </div>

      <div className="h-[360px] w-full" data-testid="feature-importance-chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 10, right: 30, left: 130, bottom: 5 }}
          >
            <XAxis
              type="number"
              domain={[0, 'auto']}
              tickFormatter={(v: number) => `${v.toFixed(1)}%`}
              stroke="var(--text-muted)"
              fontSize={11}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={140}
              stroke="var(--text-muted)"
              fontSize={11}
              tickLine={false}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload as {
                    rawKey: string;
                    label: string;
                    importance: number;
                    pct: number;
                  };
                  return (
                    <div className="p-2.5 rounded-control border border-border bg-surface shadow-subtle text-xs space-y-1">
                      <div className="font-semibold text-foreground">{data.label}</div>
                      <div className="text-foreground-muted">
                        Column: <code className="text-primary font-mono">{data.rawKey}</code>
                      </div>
                      <div className="text-foreground">
                        Gini Weight:{' '}
                        <strong className="font-mono text-primary">
                          {data.importance.toFixed(4)} ({formatPercent(data.importance, 1)})
                        </strong>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar dataKey="pct" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => {
                // Highlight top 3 dominant features with primary color, rest muted
                const fill =
                  index < 3 ? 'var(--primary)' : 'rgba(99, 102, 241, 0.4)';
                return <Cell key={`cell-${entry.rawKey}`} fill={fill} />;
              })}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="pt-2 border-t border-border flex items-center justify-between text-xs text-foreground-muted">
        <span>Dominant features: Max Burst, Burst Std. Dev, and CV</span>
        <span className="font-mono text-[11px]">Total 12 Structural Features</span>
      </div>
    </div>
  );
};

export default FeatureImportance;
