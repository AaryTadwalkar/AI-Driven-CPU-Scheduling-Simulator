/**
 * @file src/store/playbackStore.ts
 * 1. What this file does:
 *    Defines the Zustand store managing discrete simulation time ticks, playback state
 *    (play/pause), speed multipliers (1x, 2x, 4x), and step-by-step scrubbing.
 * 2. Overall design:
 *    Centralizes playback synchronization across Gantt chart playhead, ready queue visualizer,
 *    and reinforcement learning decision traces. Employs a module-scoped interval timer
 *    with automatic pausing upon reaching maxTick.
 * 3. Elements in the file:
 *    - `PlaybackState`: interface defining state variables and mutation actions
 *    - `usePlaybackStore`: Zustand hook providing reactive state and controls
 *    - `clearPlaybackTimer`: helper terminating active timer loops
 * 4. Final output of the file:
 *    Reactive hook consumed by PlaybackControls, GanttChart, ReadyQueueView, and DecisionTrace.
 */

import { create } from 'zustand';

export interface PlaybackState {
  tick: number;
  maxTick: number;
  playing: boolean;
  speed: number;
  setTick: (tick: number) => void;
  setMaxTick: (maxTick: number) => void;
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  setSpeed: (speed: number) => void;
  stepForward: () => void;
  stepBackward: () => void;
  reset: () => void;
}

let playbackTimerId: ReturnType<typeof setInterval> | null = null;

/**
 * Clears the active browser interval timer for simulation playback.
 *
 * 1. The use of function:
 *    Ensures interval timers do not leak or run concurrently when paused, reset, or retimed.
 * 2. How does the specific function work:
 *    Inspects playbackTimerId and invokes window.clearInterval if active, resetting it to null.
 * 3. Concepts used in the function:
 *    Resource disposal, browser timer management.
 * 4. Places where this function is imported and used:
 *    Internal to playbackStore actions and cleanup.
 * 5. Final output of the function, before and after:
 *    Before: Timer running.
 *    After: Timer cancelled, playbackTimerId null.
 */
export function clearPlaybackTimer(): void {
  if (playbackTimerId !== null) {
    clearInterval(playbackTimerId);
    playbackTimerId = null;
  }
}

/**
 * Zustand store hook for simulation timeline playback and stepping.
 *
 * 1. The use of function:
 *    Coordinates playhead position, speed rate, and scrubber bounds for simulation views.
 * 2. How does the specific function work:
 *    Initializes with tick=0, maxTick=0, playing=false, speed=1. Implements play/pause/step
 *    mutators that adjust state and advance or cancel internal timers.
 * 3. Concepts used in the function:
 *    Zustand state store, timer intervals, bounds clamping.
 * 4. Places where this function is imported and used:
 *    Imported by PlaybackControls, GanttChart, ReadyQueueView, DecisionTrace, SimulatorPage.
 * 5. Final output of the function, before and after:
 *    Before: Unsubscribed components.
 *    After: Subscribed reactive playback state and action dispatchers.
 */
export const usePlaybackStore = create<PlaybackState>((set, get) => ({
  tick: 0,
  maxTick: 0,
  playing: false,
  speed: 1,

  setTick: (tick: number) => {
    const { maxTick, playing } = get();
    const clamped = Math.max(0, Math.min(Math.floor(tick), maxTick));
    if (clamped >= maxTick && playing) {
      clearPlaybackTimer();
      set({ tick: clamped, playing: false });
    } else {
      set({ tick: clamped });
    }
  },

  setMaxTick: (maxTick: number) => {
    const validMax = Math.max(0, Math.floor(maxTick));
    const { tick } = get();
    const clampedTick = Math.min(tick, validMax);
    set({ maxTick: validMax, tick: clampedTick });
  },

  play: () => {
    clearPlaybackTimer();
    const { tick, maxTick, speed } = get();
    if (maxTick <= 0) return;

    if (tick >= maxTick) {
      set({ tick: 0, playing: true });
    } else {
      set({ playing: true });
    }

    const intervalMs = Math.max(50, Math.round(500 / (speed || 1)));
    playbackTimerId = setInterval(() => {
      const state = get();
      if (state.tick >= state.maxTick) {
        clearPlaybackTimer();
        set({ tick: state.maxTick, playing: false });
      } else {
        const nextTick = state.tick + 1;
        if (nextTick >= state.maxTick) {
          clearPlaybackTimer();
          set({ tick: nextTick, playing: false });
        } else {
          set({ tick: nextTick });
        }
      }
    }, intervalMs);
  },

  pause: () => {
    clearPlaybackTimer();
    set({ playing: false });
  },

  togglePlay: () => {
    const { playing } = get();
    if (playing) {
      get().pause();
    } else {
      get().play();
    }
  },

  setSpeed: (speed: number) => {
    const validSpeed = speed > 0 ? speed : 1;
    set({ speed: validSpeed });
    if (get().playing) {
      get().play();
    }
  },

  stepForward: () => {
    const { tick, maxTick, playing } = get();
    if (tick < maxTick) {
      const next = tick + 1;
      if (next >= maxTick && playing) {
        clearPlaybackTimer();
        set({ tick: next, playing: false });
      } else {
        set({ tick: next });
      }
    }
  },

  stepBackward: () => {
    const { tick } = get();
    if (tick > 0) {
      set({ tick: tick - 1 });
    }
  },

  reset: () => {
    clearPlaybackTimer();
    set({ tick: 0, playing: false });
  },
}));
