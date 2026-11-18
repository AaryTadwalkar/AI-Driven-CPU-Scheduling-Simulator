/**
 * @file src/components/sim/GanttChart.tsx
 * 1. What this file does:
 *    Renders an interactive horizontal Gantt chart timeline for CPU execution intervals,
 *    displaying process allocations, hatched IDLE and CS intervals, synchronized playhead,
 *    hover tooltips, and PNG export capability.
 * 2. Overall design:
 *    Computes proportional segment widths over total simulation makespan.
 *    Renders a synchronized vertical playhead corresponding to the playback store tick.
 *    Provides time ruler markings and allows direct playhead seeking via timeline clicks.
 * 3. Elements in the file:
 *    - `GanttChartProps`: Interface for timeline segments, playhead, and seek callbacks
 *    - `GanttChart`: React functional component
 * 4. Final output of the file:
 *    Exported Gantt timeline component rendered on Simulator and Compare pages.
 */

import React, { useRef, useState } from 'react';
import { Camera, HelpCircle, Layers } from 'lucide-react';
import { toPng } from 'html-to-image';
import type { TimelineSegmentSchema } from '@/api/client';
import { getProcessColor } from '@/lib/colors';

export interface GanttChartProps {
  timeline: TimelineSegmentSchema[];
  makespan?: number;
  playhead?: number;
  onSeek?: (tick: number) => void;
  title?: string;
  algorithmName?: string;
}

/**
 * Interactive CPU Scheduling Gantt chart component with playhead and PNG export.
 *
 * 1. The use of function:
 *    Visualizes the scheduling timeline, context switches, and CPU idle intervals.
 * 2. How does the specific function work:
 *    Calculates interval percentage widths against total makespan, applies process colors
 *    or hatch patterns, positions the vertical playhead indicator, and exports to PNG on request.
 * 3. Concepts used in the function:
 *    Proportional CSS layout, HTML to image rendering, playhead scrubbing, dynamic tooltips.
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx and ComparePage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Array of discrete TimelineSegmentSchema intervals.
 *    After: Interactive, zoomable visual Gantt timeline with playhead synchronization.
 *
 * @param props - Component properties conforming to GanttChartProps.
 * @returns JSX Element for the Gantt chart.
 */
export const GanttChart: React.FC<GanttChartProps> = ({
  timeline,
  makespan = 0,
  playhead = 0,
  onSeek,
  title = 'CPU Execution Timeline',
  algorithmName = 'Simulation',
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [hoveredSegment, setHoveredSegment] = useState<TimelineSegmentSchema | null>(null);
  const [isExporting, setIsExporting] = useState<boolean>(false);

  // Compute total duration (minimum 1 to prevent division by zero)
  const lastEnd = timeline.length > 0 ? timeline[timeline.length - 1].end : 0;
  const totalTicks = Math.max(makespan, lastEnd, 1);

  // Extract unique process labels for the legend
  const uniqueLabels = Array.from(new Set(timeline.map((s) => s.label.toUpperCase()))).sort();

  /**
   * Captures the chart container DOM node and triggers an image download.
   *
   * 1. The use of function:
   *    Exports high-resolution Gantt timeline images for reports or assignment deliverables.
   * 2. How does the specific function work:
   *    Calls html-to-image toPng on chartContainerRef, converts to data URL, and triggers an anchor click.
   * 3. Concepts used in the function:
   *    DOM image rasterization, dynamic link triggering, async error handling.
   * 4. Places where this function is imported and used:
   *    Internal export button click handler.
   * 5. Final output of the function, before and after:
   *    Before: Rendered DOM elements.
   *    After: Downloaded PNG file 'cpu-gantt-[algorithm].png'.
   */
  const handleExportPng = async () => {
    if (!chartContainerRef.current || isExporting) return;
    try {
      setIsExporting(true);
      const dataUrl = await toPng(chartContainerRef.current, {
        backgroundColor: '#FFFFFF',
        pixelRatio: 2,
      });
      const link = document.createElement('a');
      link.download = `cpu-gantt-${algorithmName.toLowerCase().replace(/[^a-z0-9]/g, '_')}.png`;
      link.href = dataUrl;
      link.click();
    } catch (err) {
      console.error('Failed to export Gantt chart image:', err);
    } finally {
      setIsExporting(false);
    }
  };

  /**
   * Calculates tick intervals for ruler markings based on total duration.
   */
  const getTickMarks = (): number[] => {
    let step = 1;
    if (totalTicks > 50) step = 5;
    else if (totalTicks > 25) step = 2;

    const ticks: number[] = [];
    for (let t = 0; t <= totalTicks; t += step) {
      ticks.push(t);
    }
    if (ticks[ticks.length - 1] !== totalTicks) {
      ticks.push(totalTicks);
    }
    return ticks;
  };

  const tickMarks = getTickMarks();
  const playheadPercent = Math.min(100, Math.max(0, (playhead / totalTicks) * 100));

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
            Total makespan: <span className="font-mono font-semibold">{totalTicks} ticks</span>
            {' • '}
            Playhead: <span className="font-mono font-semibold text-primary">t = {playhead}</span>
          </p>
        </div>

        <button
          type="button"
          onClick={handleExportPng}
          disabled={isExporting || timeline.length === 0}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-control border border-border bg-surface text-foreground text-xs font-medium hover:bg-surface-muted transition-colors disabled:opacity-50 self-start sm:self-auto"
          title="Export Gantt Chart as PNG"
        >
          <Camera className="w-3.5 h-3.5 text-primary" />
          {isExporting ? 'Exporting...' : 'Export PNG'}
        </button>
      </div>

      {/* Main Chart Box */}
      <div
        ref={chartContainerRef}
        className="p-4 rounded-control bg-surface border border-border space-y-4 overflow-x-auto"
      >
        {timeline.length === 0 ? (
          <div className="py-12 text-center text-xs text-foreground-muted">
            No simulation run yet. Select an algorithm and click "Run Simulation".
          </div>
        ) : (
          <div className="min-w-[540px] select-none">
            {/* Timeline Track Container */}
            <div
              className="relative h-14 rounded-md border border-border/80 flex overflow-hidden cursor-pointer shadow-inner bg-surface-muted"
              onClick={(e) => {
                if (!onSeek) return;
                const rect = e.currentTarget.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const ratio = Math.max(0, Math.min(1, clickX / rect.width));
                const seekTick = Math.round(ratio * totalTicks);
                onSeek(seekTick);
              }}
            >
              {/* Segment Blocks */}
              {timeline.map((seg, idx) => {
                const upper = seg.label.toUpperCase();
                const isIdle = upper === 'IDLE';
                const isCS = upper === 'CS' || upper === 'CONTEXT_SWITCH';
                const widthPct = ((seg.end - seg.start) / totalTicks) * 100;
                const bgColor = isIdle || isCS ? undefined : getProcessColor(seg.label);

                return (
                  <div
                    key={`${seg.label}-${seg.start}-${seg.end}-${idx}`}
                    onMouseEnter={() => setHoveredSegment(seg)}
                    onMouseLeave={() => setHoveredSegment(null)}
                    style={{
                      width: `${widthPct}%`,
                      backgroundColor: bgColor,
                    }}
                    className={`relative h-full flex flex-col items-center justify-center border-r border-black/15 transition-all text-xs font-mono font-bold text-white group ${
                      isIdle ? 'hatch-idle text-foreground-muted' : ''
                    } ${isCS ? 'hatch-cs text-amber-900' : ''}`}
                    title={`${seg.label}: [${seg.start}, ${seg.end}) (${seg.duration} ticks)`}
                  >
                    <span className="truncate px-1 drop-shadow-sm pointer-events-none text-xs">
                      {seg.label}
                    </span>
                    <span className="text-[10px] font-normal opacity-85 pointer-events-none">
                      {seg.duration > 1 ? `${seg.duration}t` : ''}
                    </span>
                  </div>
                );
              })}

              {/* Synchronized Playhead Marker */}
              <div
                style={{ left: `${playheadPercent}%` }}
                className="absolute top-0 bottom-0 w-0.5 bg-red-500 z-10 pointer-events-none transition-all duration-75 shadow-md"
              >
                <div className="absolute -top-1 -left-1.5 w-3.5 h-3.5 bg-red-500 rotate-45 rounded-sm shadow-sm" />
                <div className="absolute -bottom-1 -left-1.5 w-3.5 h-3.5 bg-red-500 rotate-45 rounded-sm shadow-sm" />
              </div>
            </div>

            {/* Time Ruler Axis */}
            <div className="relative h-6 mt-2">
              {tickMarks.map((t) => {
                const leftPercent = (t / totalTicks) * 100;
                return (
                  <div
                    key={t}
                    style={{ left: `${leftPercent}%` }}
                    className="absolute -translate-x-1/2 flex flex-col items-center pointer-events-none"
                  >
                    <div className="w-px h-2 bg-border" />
                    <span className="text-[10px] font-mono text-foreground-muted">
                      {t}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Hover Info & Legend */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs pt-1">
        {/* Active Hover Inspection */}
        <div className="min-h-[22px] flex items-center gap-2">
          {hoveredSegment ? (
            <div className="px-2.5 py-1 rounded bg-surface-muted border border-border text-foreground font-mono text-xs flex items-center gap-2">
              <span className="font-semibold text-primary">{hoveredSegment.label}</span>
              <span>
                t = [{hoveredSegment.start} → {hoveredSegment.end})
              </span>
              <span className="text-foreground-muted">
                duration: {hoveredSegment.duration} tick(s)
              </span>
            </div>
          ) : (
            <span className="text-foreground-muted flex items-center gap-1.5 text-[11px]">
              <HelpCircle className="w-3.5 h-3.5" />
              Hover over a block to view time bounds; click anywhere on the timeline to scrub.
            </span>
          )}
        </div>

        {/* Categorical Legend */}
        {uniqueLabels.length > 0 && (
          <div className="flex flex-wrap items-center gap-2">
            {uniqueLabels.map((lbl) => {
              const isIdle = lbl === 'IDLE';
              const isCS = lbl === 'CS' || lbl === 'CONTEXT_SWITCH';
              const color = isIdle || isCS ? undefined : getProcessColor(lbl);

              return (
                <div key={lbl} className="inline-flex items-center gap-1 text-[11px] font-mono">
                  <div
                    style={{ backgroundColor: color }}
                    className={`w-3 h-3 rounded-sm border border-black/10 ${
                      isIdle ? 'hatch-idle' : ''
                    } ${isCS ? 'hatch-cs' : ''}`}
                  />
                  <span className="text-foreground-muted">{lbl}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
