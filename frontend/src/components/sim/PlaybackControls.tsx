/**
 * @file src/components/sim/PlaybackControls.tsx
 * 1. What this file does:
 *    Provides interactive playback controls for the simulation Gantt chart: play/pause,
 *    single-tick stepping forward/backward, reset to t=0, jump to completion, speed toggles
 *    (1x, 2x, 4x), scrubber slider, and keyboard shortcuts (Space, Left/Right arrows).
 * 2. Overall design:
 *    Subscribes to `usePlaybackStore` for state and action dispatch.
 *    Attaches a global keydown event listener that ignores keystrokes when focus is on form elements.
 * 3. Elements in the file:
 *    - `SPEED_OPTIONS`: Array of playback speed multipliers [1, 2, 4]
 *    - `PlaybackControls`: React functional component
 * 4. Final output of the file:
 *    Exported playback toolbar rendered beneath the Gantt chart on SimulatorPage.
 */

import React, { useEffect } from 'react';
import {
  Play,
  Pause,
  SkipForward,
  ChevronLeft,
  ChevronRight,
  RotateCcw,
  Gauge,
} from 'lucide-react';
import { usePlaybackStore } from '@/store/playbackStore';

export const SPEED_OPTIONS = [1, 2, 4] as const;

/**
 * Interactive playback controller component connected to the global playback store.
 *
 * 1. The use of function:
 *    Provides scrubbing, transport buttons, speed selection, and hotkey listeners.
 * 2. How does the specific function work:
 *    Binds store actions to UI buttons, controls range input for scrubber ticks, and binds
 *    Space / Left / Right keyboard events to play/pause and single-tick stepping.
 * 3. Concepts used in the function:
 *    Zustand state subscription, DOM keyboard listener, event delegation, accessible button states.
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Uncontrolled time state.
 *    After: Responsive transport bar with continuous playback and step controls.
 *
 * @returns JSX Element for simulation playback controls.
 */
export const PlaybackControls: React.FC = () => {
  const tick = usePlaybackStore((s) => s.tick);
  const maxTick = usePlaybackStore((s) => s.maxTick);
  const playing = usePlaybackStore((s) => s.playing);
  const speed = usePlaybackStore((s) => s.speed);

  const setTick = usePlaybackStore((s) => s.setTick);
  const pause = usePlaybackStore((s) => s.pause);
  const togglePlay = usePlaybackStore((s) => s.togglePlay);
  const setSpeed = usePlaybackStore((s) => s.setSpeed);
  const stepForward = usePlaybackStore((s) => s.stepForward);
  const stepBackward = usePlaybackStore((s) => s.stepBackward);
  const reset = usePlaybackStore((s) => s.reset);

  /**
   * Attaches window keyboard listener for Space (play/pause), Left (step back), Right (step forward).
   */
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore shortcut keys when typing in input, textarea, or select elements
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return;
      }

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        stepBackward();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        stepForward();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [togglePlay, stepBackward, stepForward]);

  const isAtStart = tick === 0;
  const isAtEnd = tick >= maxTick;
  const isDisabled = maxTick === 0;

  return (
    <div className="p-4 rounded-card border border-border bg-surface shadow-subtle flex flex-col md:flex-row md:items-center justify-between gap-4">
      {/* Primary Transport Controls */}
      <div className="flex items-center gap-1.5 self-center sm:self-auto">
        {/* Reset to Tick 0 */}
        <button
          type="button"
          onClick={reset}
          disabled={isDisabled || isAtStart}
          aria-label="Reset to start"
          title="Reset to tick 0"
          className="p-2 rounded-control border border-border bg-surface text-foreground hover:bg-surface-muted disabled:opacity-40 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
        </button>

        {/* Step Backward */}
        <button
          type="button"
          onClick={stepBackward}
          disabled={isDisabled || isAtStart}
          aria-label="Step backward"
          title="Step back 1 tick (←)"
          className="p-2 rounded-control border border-border bg-surface text-foreground hover:bg-surface-muted disabled:opacity-40 transition-colors"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>

        {/* Play / Pause Toggle */}
        <button
          type="button"
          onClick={togglePlay}
          disabled={isDisabled}
          aria-label={playing ? 'Pause' : 'Play'}
          title={playing ? 'Pause (Space)' : 'Play (Space)'}
          className="px-4 py-2 rounded-control bg-primary text-white font-semibold shadow-sm hover:bg-primary-hover disabled:opacity-40 transition-all flex items-center gap-2"
        >
          {playing ? (
            <>
              <Pause className="w-4 h-4 fill-white" />
              <span className="text-xs uppercase tracking-wide">Pause</span>
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-white" />
              <span className="text-xs uppercase tracking-wide">Play</span>
            </>
          )}
        </button>

        {/* Step Forward */}
        <button
          type="button"
          onClick={stepForward}
          disabled={isDisabled || isAtEnd}
          aria-label="Step forward"
          title="Step forward 1 tick (→)"
          className="p-2 rounded-control border border-border bg-surface text-foreground hover:bg-surface-muted disabled:opacity-40 transition-colors"
        >
          <ChevronRight className="w-4 h-4" />
        </button>

        {/* Jump to End */}
        <button
          type="button"
          onClick={() => {
            pause();
            setTick(maxTick);
          }}
          disabled={isDisabled || isAtEnd}
          aria-label="Jump to end"
          title="Jump to end"
          className="p-2 rounded-control border border-border bg-surface text-foreground hover:bg-surface-muted disabled:opacity-40 transition-colors"
        >
          <SkipForward className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Scrubber Range Slider & Time Readout */}
      <div className="flex-1 flex items-center gap-3 px-2">
        <span className="text-xs font-mono font-bold text-foreground min-w-[36px] text-right">
          t={tick}
        </span>
        <input
          type="range"
          min={0}
          max={maxTick}
          value={tick}
          disabled={isDisabled}
          onChange={(e) => setTick(Number(e.target.value))}
          aria-label="Simulation tick scrubber"
          className="w-full h-2 bg-surface-muted rounded-lg appearance-none cursor-pointer accent-primary border border-border disabled:opacity-50"
        />
        <span className="text-xs font-mono text-foreground-muted min-w-[44px]">
          /{maxTick}t
        </span>
      </div>

      {/* Speed Multiplier Controls */}
      <div className="flex items-center gap-1 self-center sm:self-auto">
        <span className="text-[11px] text-foreground-muted flex items-center gap-1 mr-1">
          <Gauge className="w-3 h-3 text-primary" />
          Speed:
        </span>
        {SPEED_OPTIONS.map((rate) => (
          <button
            key={rate}
            type="button"
            onClick={() => setSpeed(rate)}
            className={`px-2 py-1 rounded text-xs font-mono font-medium transition-colors ${
              speed === rate
                ? 'bg-primary text-white font-bold'
                : 'border border-border bg-surface text-foreground-muted hover:bg-surface-muted'
            }`}
          >
            {rate}x
          </button>
        ))}
      </div>
    </div>
  );
};
