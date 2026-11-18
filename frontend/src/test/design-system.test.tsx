/**
 * @file src/test/design-system.test.tsx
 * 1. What this file does:
 *    Unit and component tests verifying the design system utilities, color mappings,
 *    formatting helpers, and theme switcher behavior.
 * 2. Overall design:
 *    Tests pure functions (cn, colors, format) with edge cases and renders ThemeToggle
 *    in jsdom to verify DOM class mutation and localStorage synchronization.
 * 3. Elements in the file:
 *    - describe blocks for cn, colors, format, and ThemeToggle
 * 4. Final output of the file:
 *    Vitest test suite asserting correctness of Sub-stage F1 deliverables.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { render, fireEvent } from '@testing-library/react';
import { cn } from '../lib/cn';
import { getProcessColor, getAlgorithmColor, PROCESS_PALETTE, SPECIAL_SEGMENT_COLORS } from '../lib/colors';
import { formatNumber, formatPercent, formatDelta, formatMeanStd, formatAlgorithmName } from '../lib/format';
import { ThemeToggle } from '../components/layout/ThemeToggle';

describe('Design System Utilities', () => {
  describe('cn (Tailwind Merge)', () => {
    it('combines class names and resolves conflicting utilities', () => {
      const result = cn('px-4 py-2', 'px-6', { 'text-white': true, 'text-black': false });
      expect(result).toBe('py-2 px-6 text-white');
    });
  });

  describe('colors palette mappings', () => {
    it('assigns deterministic colors to process IDs', () => {
      const p1Color = getProcessColor('P1');
      const p2Color = getProcessColor('P2');
      expect(p1Color).toBe(PROCESS_PALETTE[0]);
      expect(p2Color).toBe(PROCESS_PALETTE[1]);
      // Deterministic check: calling again returns the same color
      expect(getProcessColor('P1')).toBe(p1Color);
    });

    it('assigns special colors for IDLE and CS segments', () => {
      expect(getProcessColor('IDLE')).toBe(SPECIAL_SEGMENT_COLORS.IDLE);
      expect(getProcessColor('CS')).toBe(SPECIAL_SEGMENT_COLORS.CS);
      expect(getProcessColor('CONTEXT_SWITCH')).toBe(SPECIAL_SEGMENT_COLORS.CS);
    });

    it('returns official algorithm branding colors', () => {
      expect(getAlgorithmColor('FCFS')).toBe('#3B82F6');
      expect(getAlgorithmColor('SJF')).toBe('#10B981');
      expect(getAlgorithmColor('RR')).toBe('#F97316');
      expect(getAlgorithmColor('SRTF')).toBe('#8B5CF6');
      expect(getAlgorithmColor('Q_LEARNING')).toBe('#EC4899');
    });
  });

  describe('format helpers', () => {
    it('formats numbers with fixed decimal places and handles nulls', () => {
      expect(formatNumber(3.14159, 2)).toBe('3.14');
      expect(formatNumber(null)).toBe('—');
      expect(formatNumber(undefined)).toBe('—');
    });

    it('formats percentages correctly', () => {
      expect(formatPercent(0.852, 1)).toBe('85.2%');
      expect(formatPercent(85.2, 1)).toBe('85.2%');
      expect(formatPercent(null)).toBe('—%');
    });

    it('formats deltas and respects lowerIsBetter logic', () => {
      // Lower is better: reduced waiting time from 4.0 to 3.0 is favorable (green)
      const res1 = formatDelta(3.0, 4.0, true);
      expect(res1.isFavorable).toBe(true);
      expect(res1.colorClass).toBe('text-success');
      expect(res1.text).toContain('-1.00');

      // Higher is worse when lowerIsBetter: increased waiting time from 3.0 to 5.0 is unfavorable (red)
      const res2 = formatDelta(5.0, 3.0, true);
      expect(res2.isFavorable).toBe(false);
      expect(res2.colorClass).toBe('text-danger');
    });

    it('formats mean and std correctly', () => {
      expect(formatMeanStd(10.5, 1.25)).toBe('10.50 ± 1.25');
    });

    it('formats algorithm names nicely', () => {
      expect(formatAlgorithmName('FCFS')).toBe('First-Come First-Served (FCFS)');
      expect(formatAlgorithmName('Q_LEARNING')).toBe('Q-Learning RL Agent (AI)');
    });
  });

  describe('ThemeToggle component', () => {
    beforeEach(() => {
      localStorage.clear();
      document.documentElement.className = '';
    });

    it('toggles dark class on document.documentElement and updates localStorage', () => {
      const { getByRole } = render(<ThemeToggle />);
      const button = getByRole('button');

      // Initial state is light (matchMedia mock is false)
      expect(document.documentElement.classList.contains('dark')).toBe(false);

      // Click to toggle to dark
      fireEvent.click(button);
      expect(document.documentElement.classList.contains('dark')).toBe(true);
      expect(localStorage.getItem('theme')).toBe('dark');

      // Click to toggle back to light
      fireEvent.click(button);
      expect(document.documentElement.classList.contains('dark')).toBe(false);
      expect(localStorage.getItem('theme')).toBe('light');
    });
  });
});
