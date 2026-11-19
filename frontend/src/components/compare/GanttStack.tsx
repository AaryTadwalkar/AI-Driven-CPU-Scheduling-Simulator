/**
 * @file src/components/compare/GanttStack.tsx
 * 1. What this file does:
 *    Renders a synchronized stack of horizontal Gantt charts (one per scheduling policy)
 *    aligned to a shared maximum makespan time axis for direct visual comparison.
 * 2. Overall design:
 *    Computes the global maximum makespan across all compared algorithms to establish a
 *    uniform time scale. Renders a sticky/fixed algorithm identifier lane on the left and
 *    proportional execution blocks on the right. Includes hatched styling for IDLE/CS states,
 *    hover inspection tooltips, shared time axis ruler, and high-resolution PNG export.
 * 3. Elements in the file:
 *    - `GanttStackProps`: Input interface containing results dictionary keyed by algorithm
 *    - `GanttStack`: Main React functional component
 * 4. Final output of the file:
 *    Exported synchronized multi-timeline component mounted on ComparePage.
 */

import React, { useRef, useState, useMemo } from 'react';
import { Camera, Layers, HelpCircle } from 'lucide-react';
import { toPng } from 'html-to-image';
import type { SimulateResponse, TimelineSegmentSchema } from '@/api/client';
import { getProcessColor, getAlgorithmColor } from '@/lib/colors';
import { formatAlgorithmName } from '@/lib/format';

export interface GanttStackProps {
  results: Record<string, SimulateResponse>;
  title?: string;
}

/**
 * Synchronized stack of multi-algorithm CPU execution Gantt charts.
 *
 * 1. The use of function:
 *    Allows instant visual comparison of scheduling decisions, preemption, idle gaps,
 *    and context switch costs across all algorithms on identical workloads.
 * 2. How does the specific function work:
 *    Identifies the maximum makespan among all simulated policies, renders each policy
 *    as a horizontal lane scaled to that max time, places a shared time ruler along the base,
 *    and provides interactive segment hover details and PNG export.
 * 3. Concepts used in the function:
 *    Shared time domain scaling, proportional CSS flex/grid layout, DOM-to-image rasterization.
 * 4. Places where this function is imported and used:
 *    Imported by ComparePage.tsx and test suites.
 * 5. Final output of the function, before and after:
 *    Before: Disjoint simulation timeline segments.
 *    After: Unified, synchronized Gantt stack with uniform time ruler and tooltips.
 *
 * @param props - Component configuration conforming to GanttStackProps.
 * @returns JSX Element.
 */
export const GanttStack: React.FC<GanttStackProps> = ({
  results,
  title = 'Synchronized Execution Timelines',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredSeg, setHoveredSeg] = useState<{
    algo: string;
    segment: TimelineSegmentSchema;
  } | null>(null);
  const [isExporting, setIsExporting] = useState<boolean>(false);

  const algorithmKeys = useMemo(() => Object.keys(results), [results]);

  // Compute the global maximum makespan across all algorithms
  const globalMakespan = useMemo(() => {
    let max = 1;
    Object.values(results).forEach((res) => {
      if (res.simulation_metrics?.makespan && res.simulation_metrics.makespan > max) {
        max = res.simulation_metrics.makespan;
      }
      if (res.timeline && res.timeline.length > 0) {
        const lastEnd = res.timeline[res.timeline.length - 1].end;
        if (lastEnd > max) {
          max = lastEnd;
        }
      }
    });
    return max;
  }, [results]);

  // Extract all unique process identifiers for the shared legend
  const uniquePids = useMemo(() => {
    const pids = new Set<string>();
    Object.values(results).forEach((res) => {
      res.timeline?.forEach((seg) => {
        const up = seg.label.toUpperCase();
        if (up !== 'IDLE' && up !== 'CS' && up !== 'CONTEXT_SWITCH') {
          pids.add(seg.label);
        }
      });
    });
    return Array.from(pids).sort();
  }, [results]);

  // Generate tick marks for the shared bottom ruler
  const tickMarks = useMemo(() => {
    let step = 1;
    if (globalMakespan > 40) step = 10;
    else if (globalMakespan > 20) step = 5;
    else if (globalMakespan > 10) step = 2;

    const ticks: number[] = [];
    for (let t = 0; t <= globalMakespan; t += step) {
      ticks.push(t);
    }
    if (ticks[ticks.length - 1] !== globalMakespan) {
      ticks.push(globalMakespan);
    }
    return ticks;
  }, [globalMakespan]);

  /**
   * Captures the entire Gantt stack as a PNG image for reports.
   *
   * 1. The use of function:
   *    Exports high-resolution synchronized Gantt graphic.
   * 2. How does the specific function work:
   *    Calls html-to-image `toPng` on containerRef and triggers an anchor download.
   * 3. Concepts used in the function:
   *    Raster image export, DOM snapshotting.
   * 4. Places where this function is imported and used:
   *    Export PNG button click handler.
   * 5. Final output of the function, before and after:
   *    Before: Click event.
   *    After: Browser downloads 'gantt_stack_comparison.png'.
   */
  const handleExportPng = async () => {
    if (!containerRef.current || isExporting) return;
    try {
      setIsExporting(true);
      const dataUrl = await toPng(containerRef.current, {
        backgroundColor: '#FFFFFF',
        pixelRatio: 2,
      });
      const link = document.createElement('a');
      link.download = 'gantt_stack_comparison.png';
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Failed to export Gantt stack PNG:', err);
    } finally {
      setIsExporting(false);
    }
  };

  if (algorithmKeys.length === 0) {
    return null;
  }

  return (
    <div className="p-5 rounded-card border border-border bg-surface shadow-subtle space-y-4">
      {/* Header and Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="font-semibold text-sm text-foreground flex items-center gap-2">
            <Layers className="w-4 h-4 text-primary" />
            {title}
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            All timelines scaled to common makespan of{' '}
            <span className="font-mono font-semibold text-foreground">
              {globalMakespan} ticks
            </span>
            . Hover blocks for interval details.
          </p>
        </div>

        <button
          type="button"
          onClick={handleExportPng}
          disabled={isExporting}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-control text-xs font-medium border border-border bg-surface text-foreground hover:bg-surface-muted transition-colors disabled:opacity-50 self-start sm:self-auto shadow-sm"
        >
          <Camera className="w-3.5 h-3.5 text-primary" />
          <span>{isExporting ? 'Exporting...' : 'Export Stack PNG'}</span>
        </button>
      </div>

      {/* Main Stack Container */}
      <div
        ref={containerRef}
        className="p-4 rounded-control bg-surface border border-border space-y-4 overflow-x-auto"
      >
        <div className="min-w-[620px] select-none space-y-3">
          {algorithmKeys.map((algo) => {
            const sim = results[algo];
            const timeline = sim?.timeline || [];
            const makespan = sim?.simulation_metrics?.makespan || 0;
            const algoColor = getAlgorithmColor(algo);

            return (
              <div key={algo} className="flex items-center gap-3">
                {/* Left Algorithm Label Lane */}
                <div className="w-36 shrink-0 flex items-center justify-between pr-2 border-r border-border">
                  <div className="flex items-center gap-1.5 truncate">
                    <span
                      className="w-2.5 h-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: algoColor }}
                    />
                    <span
                      className="text-xs font-semibold text-foreground truncate"
                      title={formatAlgorithmName(algo)}
                    >
                      {algo}
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-foreground-muted shrink-0">
                    {makespan}t
                  </span>
                </div>

                {/* Right Horizontal Timeline Track */}
                <div className="relative flex-1 h-9 rounded-md border border-border/80 flex overflow-hidden bg-surface-muted/50 shadow-inner">
                  {timeline.map((seg, idx) => {
                    const upper = seg.label.toUpperCase();
                    const isIdle = upper === 'IDLE';
                    const isCS = upper === 'CS' || upper === 'CONTEXT_SWITCH';
                    const isAI = upper === 'AI' || upper === 'AI_OVERHEAD';
                    const widthPct = ((seg.end - seg.start) / globalMakespan) * 100;
                    const bgColor = isIdle || isCS || isAI ? undefined : getProcessColor(seg.label);

                    return (
                      <div
                        key={`${algo}-${seg.label}-${seg.start}-${seg.end}-${idx}`}
                        onMouseEnter={() => setHoveredSeg({ algo, segment: seg })}
                        onMouseLeave={() => setHoveredSeg(null)}
                        style={{
                          width: `${widthPct}%`,
                          backgroundColor: bgColor,
                        }}
                        className={`relative h-full flex items-center justify-center border-r border-black/15 transition-all text-[11px] font-mono font-bold text-white group cursor-pointer ${
                          isIdle ? 'hatch-idle text-foreground-muted' : ''
                        } ${isCS ? 'hatch-cs text-amber-900' : ''} ${
                          isAI ? 'hatch-ai text-pink-900' : ''
                        }`}
                        title={`${algo} — ${seg.label}: [${seg.start}, ${seg.end}) (${seg.duration} ticks)`}
                      >
                        <span className="truncate px-1 drop-shadow-sm pointer-events-none">
                          {seg.label}
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}

          {/* Shared Common Time Axis Ruler */}
          <div className="flex items-center gap-3 pt-1 border-t border-border">
            <div className="w-36 shrink-0 text-right pr-2 text-[10px] font-mono text-foreground-muted">
              Time (ticks) →
            </div>
            <div className="relative flex-1 h-5">
              {tickMarks.map((tick) => {
                const leftPct = (tick / globalMakespan) * 100;
                return (
                  <div
                    key={`tick-${tick}`}
                    style={{ left: `${leftPct}%` }}
                    className="absolute -translate-x-1/2 flex flex-col items-center"
                  >
                    <div className="w-px h-1.5 bg-border" />
                    <span className="text-[10px] font-mono text-foreground-muted">
                      {tick}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Hover Details Panel */}
        {hoveredSeg && (
          <div className="p-2.5 rounded bg-surface-muted border border-border text-xs flex items-center justify-between text-foreground">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-primary">{hoveredSeg.algo}</span>
              <span className="text-foreground-muted">Segment:</span>
              <span className="font-mono font-bold">{hoveredSeg.segment.label}</span>
            </div>
            <div className="font-mono text-xs text-foreground-muted space-x-3">
              <span>
                Interval: [{hoveredSeg.segment.start}, {hoveredSeg.segment.end})
              </span>
              <span>Duration: {hoveredSeg.segment.duration} ticks</span>
            </div>
          </div>
        )}

        {/* Color Legend */}
        <div className="pt-2 border-t border-border flex flex-wrap items-center gap-4 text-xs text-foreground-muted">
          <div className="flex items-center gap-1.5 font-medium text-foreground">
            <HelpCircle className="w-3.5 h-3.5 text-primary" />
            <span>Legend:</span>
          </div>

          {uniquePids.map((pid) => (
            <div key={pid} className="flex items-center gap-1.5 font-mono">
              <span
                className="w-3 h-3 rounded-sm"
                style={{ backgroundColor: getProcessColor(pid) }}
              />
              <span>{pid}</span>
            </div>
          ))}

          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm hatch-idle border border-border" />
            <span>IDLE</span>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm hatch-cs border border-amber-600/40" />
            <span>CS (Context Switch)</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default GanttStack;
