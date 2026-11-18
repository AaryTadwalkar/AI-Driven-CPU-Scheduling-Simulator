/**
 * @file src/test/simulator.test.tsx
 * 1. What this file does:
 *    Integration test suite verifying rendering and interaction of Sub-stage F3 simulator
 *    components: GanttChart, MetricCards, ReadyQueueView, AlgorithmPicker, and DecisionTrace.
 * 2. Overall design:
 *    Uses Vitest and React Testing Library to render components with canonical Section 4.6
 *    simulation fixtures, asserting visual elements, hatch styling, formatted metrics, and callbacks.
 * 3. Elements in the file:
 *    - Mock simulation response fixtures matching FastAPI backend contracts
 *    - describe blocks for GanttChart, MetricCards, ReadyQueueView, AlgorithmPicker, DecisionTrace
 * 4. Final output of the file:
 *    Automated verification of the Sub-stage F3 simulation and playback UI suite.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { GanttChart } from '../components/sim/GanttChart';
import { MetricCards } from '../components/sim/MetricCards';
import { ReadyQueueView } from '../components/sim/ReadyQueueView';
import { AlgorithmPicker } from '../components/sim/AlgorithmPicker';
import { DecisionTrace } from '../components/ai/DecisionTrace';
import type {
  TimelineSegmentSchema,
  SimulationMetricsSchema,
  ReadyQueueLogEntry,
  DecisionTraceEntry,
} from '../api/client';

// Canonical Section 4.6 simulation mock fixtures (P1: [0, 5), P2: [5, 8), P3: [8, 10))
const MOCK_TIMELINE: TimelineSegmentSchema[] = [
  { label: 'P1', start: 0, end: 5, duration: 5 },
  { label: 'CS', start: 5, end: 6, duration: 1 },
  { label: 'P2', start: 6, end: 9, duration: 3 },
  { label: 'IDLE', start: 9, end: 11, duration: 2 },
  { label: 'P3', start: 11, end: 13, duration: 2 },
];

const MOCK_METRICS: SimulationMetricsSchema = {
  avg_waiting: 2.33,
  avg_turnaround: 5.67,
  avg_response: 1.33,
  max_waiting: 4,
  makespan: 13,
  cpu_utilization: 0.769,
  overhead_ticks: 1,
  throughput: 0.23,
  context_switches: 1,
  jain_fairness: 0.942,
};

const MOCK_QUEUE_LOG: ReadyQueueLogEntry[] = [
  { time: 0, running_pid: 'P1', ready_pids: ['P2', 'P3'] },
  { time: 5, running_pid: null, ready_pids: ['P2', 'P3'] },
  { time: 6, running_pid: 'P2', ready_pids: ['P3'] },
  { time: 9, running_pid: null, ready_pids: [] },
  { time: 11, running_pid: 'P3', ready_pids: [] },
];

const MOCK_DECISION_TRACE: DecisionTraceEntry[] = [
  {
    time: 0,
    state: [3, 2, 5, 0],
    valid_actions: ['PICK_SHORTEST', 'CONTINUE', 'PICK_LONGEST'],
    q_values: { PICK_SHORTEST: 4.82, CONTINUE: 3.12, PICK_LONGEST: 1.05 },
    chosen_action: 'PICK_SHORTEST',
    chosen_pid: 'P1',
    reward: -1.0,
  },
  {
    time: 5,
    state: [2, 1, 3, 0],
    valid_actions: ['PICK_SHORTEST', 'PICK_LONGEST'],
    q_values: { PICK_SHORTEST: 6.2, PICK_LONGEST: 2.1 },
    chosen_action: 'PICK_SHORTEST',
    chosen_pid: 'P2',
    reward: 0.5,
  },
];

describe('Sub-stage F3 Simulator Components', () => {
  describe('GanttChart Component', () => {
    it('renders timeline segment labels and duration', () => {
      render(
        <GanttChart
          timeline={MOCK_TIMELINE}
          makespan={13}
          playhead={0}
          title="CPU Execution Timeline — FCFS"
          algorithmName="FCFS"
        />
      );

      expect(screen.getByText('CPU Execution Timeline — FCFS')).toBeInTheDocument();
      expect(screen.getByText('13 ticks')).toBeInTheDocument();
      expect(screen.getAllByText('P1').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('P2').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('P3').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('CS').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('IDLE').length).toBeGreaterThanOrEqual(1);
    });

    it('renders diagonal hatch classes for IDLE and CS intervals', () => {
      const { container } = render(
        <GanttChart
          timeline={MOCK_TIMELINE}
          makespan={13}
          playhead={2}
        />
      );

      const idleElements = container.querySelectorAll('.hatch-idle');
      const csElements = container.querySelectorAll('.hatch-cs');
      expect(idleElements.length).toBeGreaterThan(0);
      expect(csElements.length).toBeGreaterThan(0);
    });

    it('triggers onSeek callback on click', () => {
      const onSeekMock = vi.fn();
      const { container } = render(
        <GanttChart
          timeline={MOCK_TIMELINE}
          makespan={13}
          playhead={0}
          onSeek={onSeekMock}
        />
      );

      const track = container.querySelector('.cursor-pointer');
      expect(track).toBeInTheDocument();
      if (track) {
        fireEvent.click(track, { clientX: 200 });
        expect(onSeekMock).toHaveBeenCalled();
      }
    });
  });

  describe('MetricCards Component', () => {
    it('renders all 6 summary KPI cards with formatted values', () => {
      render(<MetricCards metrics={MOCK_METRICS} />);

      expect(screen.getByText('Avg Waiting')).toBeInTheDocument();
      expect(screen.getByText('2.33')).toBeInTheDocument();

      expect(screen.getByText('Avg Turnaround')).toBeInTheDocument();
      expect(screen.getByText('5.67')).toBeInTheDocument();

      expect(screen.getByText('Avg Response')).toBeInTheDocument();
      expect(screen.getByText('1.33')).toBeInTheDocument();

      expect(screen.getByText('CPU Utilization')).toBeInTheDocument();
      expect(screen.getByText('76.9%')).toBeInTheDocument();

      expect(screen.getByText('Jain\'s Fairness')).toBeInTheDocument();
      expect(screen.getByText('0.942')).toBeInTheDocument();

      expect(screen.getByText('Context Switches')).toBeInTheDocument();
      expect(screen.getByText('1')).toBeInTheDocument();
    });

    it('renders fallback placeholders when metrics are null', () => {
      render(<MetricCards metrics={null} />);
      const placeholders = screen.getAllByText('—');
      expect(placeholders.length).toBeGreaterThanOrEqual(5);
    });
  });

  describe('ReadyQueueView Component', () => {
    it('displays running process P1 and ready queue [P2, P3] at tick 0', () => {
      render(
        <ReadyQueueView
          readyQueueLog={MOCK_QUEUE_LOG}
          currentTick={0}
        />
      );

      expect(screen.getByText('RUNNING')).toBeInTheDocument();
      expect(screen.getByText('P1')).toBeInTheDocument();
      expect(screen.getByText('P2')).toBeInTheDocument();
      expect(screen.getByText('P3')).toBeInTheDocument();
      expect(screen.getByText('2 processes')).toBeInTheDocument();
    });

    it('displays IDLE when running_pid is null and queue is empty at tick 9', () => {
      render(
        <ReadyQueueView
          readyQueueLog={MOCK_QUEUE_LOG}
          currentTick={9}
        />
      );

      expect(screen.getByText('IDLE')).toBeInTheDocument();
      expect(screen.getByText(/Queue is empty/i)).toBeInTheDocument();
    });

    it('displays CONTEXT SWITCH when running_pid is null but queue has waiting processes at tick 5', () => {
      render(
        <ReadyQueueView
          readyQueueLog={MOCK_QUEUE_LOG}
          currentTick={5}
        />
      );

      expect(screen.getByText('CONTEXT SWITCH')).toBeInTheDocument();
      expect(screen.getByText('P2')).toBeInTheDocument();
    });
  });

  describe('AlgorithmPicker Component', () => {
    it('renders algorithm selection, quantum controls, and context switch buttons', () => {
      const onAlgorithmChange = vi.fn();
      const onQuantumChange = vi.fn();
      const onCsCostChange = vi.fn();
      const onRunSimulation = vi.fn();

      render(
        <AlgorithmPicker
          algorithm="FCFS"
          onAlgorithmChange={onAlgorithmChange}
          quantum={2}
          onQuantumChange={onQuantumChange}
          csCost={0}
          onCsCostChange={onCsCostChange}
          onRunSimulation={onRunSimulation}
        />
      );

      expect(screen.getByText('First-Come First-Served')).toBeInTheDocument();
      expect(screen.getByText('Round Robin')).toBeInTheDocument();
      expect(screen.getByText('Q-Learning')).toBeInTheDocument();

      // Click SJF
      fireEvent.click(screen.getByText('SJF'));
      expect(onAlgorithmChange).toHaveBeenCalledWith('SJF');

      // Click Context Switch toggle
      fireEvent.click(screen.getByText('1 (Overhead)'));
      expect(onCsCostChange).toHaveBeenCalledWith(1);

      // Click Run Simulation
      fireEvent.click(screen.getByText('Run Simulation'));
      expect(onRunSimulation).toHaveBeenCalledTimes(1);
    });

    it('displays Decision Tree recommendation banner and allows auto-selection', () => {
      const onAlgorithmChange = vi.fn();

      render(
        <AlgorithmPicker
          algorithm="FCFS"
          onAlgorithmChange={onAlgorithmChange}
          quantum={2}
          onQuantumChange={vi.fn()}
          csCost={0}
          onCsCostChange={vi.fn()}
          onRunSimulation={vi.fn()}
          recommendation={{
            algorithm: 'SJF',
            probabilities: { SJF: 0.82, FCFS: 0.12, RR: 0.06 },
            features: {},
          }}
        />
      );

      expect(screen.getByText(/ML Recommendation:/i)).toBeInTheDocument();
      expect(screen.getAllByText('SJF').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('(82% confidence)')).toBeInTheDocument();

      // Click Use Recommended button
      const useRecBtn = screen.getByText('Use Recommended');
      fireEvent.click(useRecBtn);
      expect(onAlgorithmChange).toHaveBeenCalledWith('SJF');
    });
  });

  describe('DecisionTrace Component', () => {
    it('renders RL state vector, valid actions, Q-values, and triggers tick seek', () => {
      const onSelectTick = vi.fn();

      render(
        <DecisionTrace
          trace={MOCK_DECISION_TRACE}
          currentTick={0}
          onSelectTick={onSelectTick}
        />
      );

      expect(screen.getByText(/Q-Learning Agent Decision Trace/i)).toBeInTheDocument();
      expect(screen.getByText('t=0')).toBeInTheDocument();
      expect(screen.getByText('t=5')).toBeInTheDocument();
      expect(screen.getAllByText('PICK_SHORTEST').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('PICK_SHORTEST: 4.82')).toBeInTheDocument();

      // Click on row for tick 5
      const rowTick5 = screen.getByText('t=5').closest('tr');
      expect(rowTick5).toBeInTheDocument();
      if (rowTick5) {
        fireEvent.click(rowTick5);
        expect(onSelectTick).toHaveBeenCalledWith(5);
      }
    });
  });
});
