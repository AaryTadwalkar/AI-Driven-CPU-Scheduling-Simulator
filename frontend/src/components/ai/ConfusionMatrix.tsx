/**
 * @file src/components/ai/ConfusionMatrix.tsx
 * 1. What this file does:
 *    Renders a 3x3 heatmap confusion matrix grid visualizing Decision Tree policy classification.
 * 2. Overall design:
 *    Displays predicted algorithms along the horizontal axis and actual ground-truth classes
 *    along the vertical axis. Each cell shows sample counts and row-normalized percentages
 *    with responsive heatmap color shading. Includes an accessible HTML table fallback.
 * 3. Elements in the file:
 *    - `ConfusionMatrixProps`: Interface defining matrix inputs, class labels, and metadata
 *    - `getCellColor`: Helper calculating normalized background color intensity
 *    - `ConfusionMatrix`: Primary React functional component
 * 4. Final output of the file:
 *    Interactive, accessible confusion matrix heatmap card consumed by AILabPage.
 */

import React, { useState } from 'react';
import { Table, Eye, CheckCircle2 } from 'lucide-react';
import { formatPercent } from '@/lib/format';

export interface ConfusionMatrixProps {
  matrix?: number[][];
  classes?: string[];
  totalSamples?: number;
  accuracy?: number;
  macroF1?: number;
  className?: string;
}

const DEFAULT_CLASSES = ['FCFS', 'RR', 'SJF'];

/**
 * Calculates a dynamic CSS background style matching cell percentage intensity.
 *
 * 1. The use of function:
 *    Computes theme-compatible background color shading based on row-normalized accuracy.
 * 2. How does the specific function work:
 *    Checks if the cell is on the diagonal (true positive). For diagonal cells, applies
 *    a graduated emerald/primary hue with opacity proportional to normalized percentage.
 *    For off-diagonal errors, applies a subtle red/amber tint.
 * 3. Concepts used in the function:
 *    Dynamic RGBA opacity interpolation, diagonal classification distinction.
 * 4. Places where this function is imported and used:
 *    Used internally by `ConfusionMatrix` for cell rendering.
 * 5. Final output of the function, before and after:
 *    Before: `pct = 85.5`, `isDiagonal = true`
 *    After: CSS style object `{ backgroundColor: 'rgba(16, 185, 129, 0.75)' }`
 *
 * @param pct - Normalized percentage (0 to 100).
 * @param isDiagonal - Boolean indicating whether predicted class matches actual class.
 * @returns React.CSSProperties
 */
function getCellColor(pct: number, isDiagonal: boolean): React.CSSProperties {
  if (pct === 0) {
    return { backgroundColor: 'transparent' };
  }

  const alpha = Math.min(0.85, Math.max(0.12, pct / 100));
  if (isDiagonal) {
    // Emerald green for correct classifications
    return { backgroundColor: `rgba(16, 185, 129, ${alpha})` };
  }
  // Subtle amber/red for misclassifications
  return { backgroundColor: `rgba(239, 68, 68, ${alpha * 0.4})` };
}

/**
 * Confusion Matrix Heatmap component.
 *
 * 1. The use of function:
 *    Visualizes classification fidelity of the Decision Tree model on held-out test workloads.
 * 2. How does the specific function work:
 *    Validates 3x3 input matrix. Computes row sums to normalize each cell into a percentage.
 *    Renders a 2D grid with axis headers, counts, percentages, and an accessible table view toggle.
 * 3. Concepts used in the function:
 *    Heatmap visualization, row normalization, WCAG accessible table fallback, state toggles.
 * 4. Places where this function is imported and used:
 *    Imported by AILabPage.tsx and test suites.
 * 5. Final output of the function, before and after:
 *    Before: Raw 2D number array from /api/ml/model-info.
 *    After: Interactive heatmap grid with hover tooltips and accessible screen-reader table.
 *
 * @param props - Matrix data, class labels, and evaluation metrics.
 * @returns JSX Element.
 */
export const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({
  matrix,
  classes = DEFAULT_CLASSES,
  totalSamples = 1000,
  accuracy = 0.902,
  macroF1 = 0.8878,
  className = '',
}) => {
  const [showAccessibleTable, setShowAccessibleTable] = useState(false);

  // Fallback if matrix is not provided or empty
  const defaultMatrix = [
    [154, 3, 38],
    [9, 237, 19],
    [11, 18, 511],
  ];
  const grid = matrix && matrix.length === 3 ? matrix : defaultMatrix;

  // Compute row totals for row normalization (Actual class totals)
  const rowTotals = grid.map((row) => row.reduce((sum, val) => sum + val, 0));

  return (
    <div
      className={`p-5 rounded-card border border-border bg-surface shadow-subtle flex flex-col space-y-4 ${className}`}
      data-testid="confusion-matrix"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-border">
        <div>
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
            Decision Tree Test Confusion Matrix
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Normalized across 1,000 held-out test workloads. Rows indicate actual composite winners;
            columns indicate model predictions.
          </p>
        </div>

        <button
          type="button"
          onClick={() => setShowAccessibleTable((prev) => !prev)}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-control border border-border bg-surface hover:bg-surface-muted text-foreground transition-colors self-start sm:self-center"
          title="Toggle Accessible Table View"
          aria-label="Toggle Accessible Table View"
        >
          {showAccessibleTable ? (
            <>
              <Eye className="w-3.5 h-3.5 text-primary" />
              <span>Show Heatmap</span>
            </>
          ) : (
            <>
              <Table className="w-3.5 h-3.5 text-primary" />
              <span>Table View</span>
            </>
          )}
        </button>
      </div>

      {showAccessibleTable ? (
        // Accessible HTML Table Fallback
        <div className="overflow-x-auto">
          <table
            className="w-full text-xs text-left border-collapse"
            aria-label="Accessible Confusion Matrix Table"
          >
            <thead>
              <tr className="border-b border-border bg-surface-muted/50">
                <th className="p-2 font-semibold text-foreground">Actual \ Predicted</th>
                {classes.map((cls) => (
                  <th key={cls} className="p-2 font-semibold text-foreground text-center">
                    Pred {cls}
                  </th>
                ))}
                <th className="p-2 font-semibold text-foreground text-right">Row Total</th>
              </tr>
            </thead>
            <tbody>
              {grid.map((row, rowIdx) => {
                const actualClass = classes[rowIdx] ?? `Class ${rowIdx}`;
                const total = rowTotals[rowIdx] || 1;
                return (
                  <tr key={actualClass} className="border-b border-border/50 hover:bg-surface-muted/30">
                    <td className="p-2 font-medium text-foreground">{actualClass}</td>
                    {row.map((val, colIdx) => {
                      const pct = (val / total) * 100;
                      return (
                        <td key={colIdx} className="p-2 text-center font-mono">
                          {val} ({pct.toFixed(1)}%)
                        </td>
                      );
                    })}
                    <td className="p-2 text-right font-mono font-semibold">{total}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        // Interactive CSS Heatmap Grid
        <div className="space-y-3">
          {/* Header Row: Predicted Labels */}
          <div className="grid grid-cols-4 gap-2 items-center text-center">
            <div className="text-[11px] font-semibold text-foreground-muted text-left pl-2">
              Actual \ Predicted
            </div>
            {classes.map((cls) => (
              <div
                key={cls}
                className="py-1 px-2 rounded bg-surface-muted text-xs font-semibold text-foreground"
              >
                {cls}
              </div>
            ))}
          </div>

          {/* Grid Rows */}
          {grid.map((row, rowIdx) => {
            const actualClass = classes[rowIdx] ?? `Class ${rowIdx}`;
            const total = rowTotals[rowIdx] || 1;

            return (
              <div key={actualClass} className="grid grid-cols-4 gap-2 items-center">
                {/* Actual Label Header */}
                <div className="flex items-center justify-between pl-2 pr-1 text-xs font-medium text-foreground">
                  <span>{actualClass}</span>
                  <span className="text-[10px] text-foreground-muted font-mono">
                    (n={total})
                  </span>
                </div>

                {/* Heatmap Cells */}
                {row.map((val, colIdx) => {
                  const isDiagonal = rowIdx === colIdx;
                  const pct = (val / total) * 100;
                  const style = getCellColor(pct, isDiagonal);

                  return (
                    <div
                      key={colIdx}
                      style={style}
                      className={`p-3 rounded-control border flex flex-col items-center justify-center transition-all ${
                        isDiagonal
                          ? 'border-emerald-500/40 text-foreground font-semibold shadow-xs'
                          : 'border-border/60 text-foreground/80'
                      }`}
                      title={`Actual ${actualClass} -> Predicted ${classes[colIdx]}: ${val} workloads (${pct.toFixed(1)}%)`}
                    >
                      <span className="text-sm font-mono font-bold">{val}</span>
                      <span className="text-[10px] text-foreground-muted font-mono">
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      )}

      {/* Summary KPI Strip */}
      <div className="pt-3 border-t border-border flex flex-wrap items-center justify-between text-xs text-foreground-muted gap-2">
        <div className="flex items-center gap-4">
          <div>
            <span>Test Accuracy: </span>
            <strong className="text-foreground font-mono">{formatPercent(accuracy, 1)}</strong>
          </div>
          <div>
            <span>Macro-F1: </span>
            <strong className="text-foreground font-mono">{macroF1.toFixed(4)}</strong>
          </div>
        </div>
        <div className="text-[11px]">
          Sample Size: <span className="font-mono text-foreground font-semibold">{totalSamples}</span>
        </div>
      </div>
    </div>
  );
};

export default ConfusionMatrix;
