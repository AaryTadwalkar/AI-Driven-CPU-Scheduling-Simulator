/**
 * @file src/test/playback.test.ts
 * 1. What this file does:
 *    Unit test suite covering the Zustand playback store: tick clamping, stepping forward/backward,
 *    speed multipliers, play/pause interval timer progression, and automatic pausing at maxTick.
 * 2. Overall design:
 *    Uses Vitest fake timers to test time advancement and boundary constraints reliably
 *    without real-time delays.
 * 3. Elements in the file:
 *    - Store action tests (setTick, setMaxTick, stepForward, stepBackward, reset)
 *    - Timer progression tests (play, pause, togglePlay, setSpeed)
 * 4. Final output of the file:
 *    Automated test verification for playback state management in Sub-stage F3.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { usePlaybackStore, clearPlaybackTimer } from '../store/playbackStore';

describe('Playback Store (usePlaybackStore)', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    clearPlaybackTimer();
    usePlaybackStore.setState({
      tick: 0,
      maxTick: 10,
      playing: false,
      speed: 1,
    });
  });

  afterEach(() => {
    clearPlaybackTimer();
    vi.restoreAllMocks();
  });

  it('initializes with default settings', () => {
    const state = usePlaybackStore.getState();
    expect(state.tick).toBe(0);
    expect(state.maxTick).toBe(10);
    expect(state.playing).toBe(false);
    expect(state.speed).toBe(1);
  });

  it('clamps setTick within [0, maxTick]', () => {
    const { setTick } = usePlaybackStore.getState();
    setTick(5);
    expect(usePlaybackStore.getState().tick).toBe(5);

    setTick(-3);
    expect(usePlaybackStore.getState().tick).toBe(0);

    setTick(25);
    expect(usePlaybackStore.getState().tick).toBe(10);
  });

  it('clamps current tick when setMaxTick reduces bounds', () => {
    const { setTick, setMaxTick } = usePlaybackStore.getState();
    setTick(8);
    expect(usePlaybackStore.getState().tick).toBe(8);

    setMaxTick(4);
    expect(usePlaybackStore.getState().maxTick).toBe(4);
    expect(usePlaybackStore.getState().tick).toBe(4);
  });

  it('steps forward and backward within bounds', () => {
    const { stepForward, stepBackward } = usePlaybackStore.getState();
    expect(usePlaybackStore.getState().tick).toBe(0);

    stepBackward();
    expect(usePlaybackStore.getState().tick).toBe(0);

    stepForward();
    expect(usePlaybackStore.getState().tick).toBe(1);

    stepForward();
    expect(usePlaybackStore.getState().tick).toBe(2);

    stepBackward();
    expect(usePlaybackStore.getState().tick).toBe(1);
  });

  it('stops stepping forward when tick reaches maxTick', () => {
    const { setTick, stepForward } = usePlaybackStore.getState();
    setTick(10);
    expect(usePlaybackStore.getState().tick).toBe(10);

    stepForward();
    expect(usePlaybackStore.getState().tick).toBe(10);
  });

  it('resets tick to 0 and stops playing on reset()', () => {
    const { setTick, reset } = usePlaybackStore.getState();
    setTick(7);
    reset();
    const state = usePlaybackStore.getState();
    expect(state.tick).toBe(0);
    expect(state.playing).toBe(false);
  });

  it('advances tick over time when playing and pauses at maxTick', () => {
    const { play } = usePlaybackStore.getState();
    play();
    expect(usePlaybackStore.getState().playing).toBe(true);

    // Advance 500ms (1 step at 1x speed)
    vi.advanceTimersByTime(500);
    expect(usePlaybackStore.getState().tick).toBe(1);

    vi.advanceTimersByTime(1000);
    expect(usePlaybackStore.getState().tick).toBe(3);

    // Advance past maxTick (7 remaining ticks = 3500ms)
    vi.advanceTimersByTime(4000);
    const state = usePlaybackStore.getState();
    expect(state.tick).toBe(10);
    expect(state.playing).toBe(false);
  });

  it('toggles play/pause state', () => {
    const { togglePlay } = usePlaybackStore.getState();
    togglePlay();
    expect(usePlaybackStore.getState().playing).toBe(true);

    togglePlay();
    expect(usePlaybackStore.getState().playing).toBe(false);
  });

  it('respects speed multiplier during playback', () => {
    const { play, setSpeed } = usePlaybackStore.getState();
    setSpeed(2); // 2x speed -> 250ms per step
    play();

    vi.advanceTimersByTime(250);
    expect(usePlaybackStore.getState().tick).toBe(1);

    vi.advanceTimersByTime(500);
    expect(usePlaybackStore.getState().tick).toBe(3);
  });
});
