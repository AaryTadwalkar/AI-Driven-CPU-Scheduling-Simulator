/**
 * @file src/test/compare.test.tsx
 * 1. What this file does:
 *    Comprehensive unit and integration test suite for Sub-stage F4 Compare Page components:
 *    ComparisonTable, ComparisonBars, GanttStack, ComparePage, and CSV export utilities.
 * 2. Overall design:
 *    Uses Vitest and React Testing Library to verify metric optimality highlighting,
 *    relative baseline deltas, CSV serialization, Recharts grouped bar charts, synchronized
 *    multi-algorithm Gantt stacks, and ComparePage lifecycle with mocked API responses.
 * 3. Elements in the file:
 *    - Canonical comparison mock fixtures matching FastAPI /api/compare contracts
 *    - `renderWithProviders`: Helper wrapping test elements in QueryClient and MemoryRouter
 *    - Test suites for CSV export, best-per-metric logic, ComparisonTable, ComparisonBars,
 *      GanttStack, and ComparePage integration.
 * 4. Final output of the file:
 *    Complete verification of the multi-policy benchmarking dashboard.
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ComparisonTable } from '../components/compare/ComparisonTable';
import { ComparisonBars, CustomTooltip } from '../components/compare/ComparisonBars';
import { GanttStack } from '../components/compare/GanttStack';
import { ComparePage } from '../pages/ComparePage';
import { exportComparisonToCsv, downloadComparisonCsv, type ComparisonCsvRow } from '../lib/csv';
import { useWorkloadStore } from '../store/workloadStore';
import * as apiClient from '../api/client';
import type { CompareResponse, SimulateResponse } from '../api/client';

// Canonical test mock data matching Section 4.6 comparison benchmark
const MOCK_COMPARISON_ROWS: ComparisonCsvRow[] = [
  {
    algorithm: 'FCFS',
    avg_waiting: 3.67,
    avg_turnaround: 7.0,
    avg_response: 3.67,
    cpu_utilization: 0.909,
    jain_fairness: 0.85,
    context_switches: 2,
  },
  {
    algorithm: 'SJF',
    avg_waiting: 2.33,
    avg_turnaround: 5.67,
    avg_response: 2.33,
    cpu_utilization: 0.909,
    jain_fairness: 0.94,
    context_switches: 2,
  },
  {
    algorithm: 'SRTF',
    avg_waiting: 2.0,
    avg_turnaround: 5.33,
    avg_response: 1.67,
    cpu_utilization: 0.95,
    jain_fairness: 0.96,
    context_switches: 3,
  },
  {
    algorithm: 'RR',
    avg_waiting: 4.0,
    avg_turnaround: 7.33,
    avg_response: 1.33,
    cpu_utilization: 0.85,
    jain_fairness: 0.92,
    context_switches: 4,
  },
  {
    algorithm: 'QLEARNING',
    avg_waiting: 2.1,
    avg_turnaround: 5.45,
    avg_response: 1.8,
    cpu_utilization: 0.92,
    jain_fairness: 0.95,
    context_switches: 2,
  },
];

const MOCK_SIMULATE_RESULTS: Record<string, SimulateResponse> = {
  FCFS: {
    algorithm: 'FCFS',
    quantum: 2,
    cs_cost: 0,
    process_metrics: [],
    simulation_metrics: {
      avg_waiting: 3.67,
      avg_turnaround: 7.0,
      avg_response: 3.67,
      max_waiting: 6,
      makespan: 11,
      cpu_utilization: 0.909,
      overhead_ticks: 0,
      throughput: 0.27,
      context_switches: 2,
      jain_fairness: 0.85,
    },
    timeline: [
      { label: 'P1', start: 0, end: 5, duration: 5 },
      { label: 'P2', start: 5, end: 8, duration: 3 },
      { label: 'P3', start: 8, end: 10, duration: 2 },
      { label: 'IDLE', start: 10, end: 11, duration: 1 },
    ],
    ready_queue_log: [],
    decision_trace: null,
  },
  SJF: {
    algorithm: 'SJF',
    quantum: 2,
    cs_cost: 0,
    process_metrics: [],
    simulation_metrics: {
      avg_waiting: 2.33,
      avg_turnaround: 5.67,
      avg_response: 2.33,
      max_waiting: 5,
      makespan: 10,
      cpu_utilization: 0.909,
      overhead_ticks: 0,
      throughput: 0.3,
      context_switches: 2,
      jain_fairness: 0.94,
    },
    timeline: [
      { label: 'P1', start: 0, end: 5, duration: 5 },
      { label: 'P3', start: 5, end: 7, duration: 2 },
      { label: 'P2', start: 7, end: 10, duration: 3 },
    ],
    ready_queue_log: [],
    decision_trace: null,
  },
  RR: {
    algorithm: 'RR',
    quantum: 2,
    cs_cost: 1,
    process_metrics: [],
    simulation_metrics: {
      avg_waiting: 4.0,
      avg_turnaround: 7.33,
      avg_response: 1.33,
      max_waiting: 7,
      makespan: 14,
      cpu_utilization: 0.85,
      overhead_ticks: 4,
      throughput: 0.21,
      context_switches: 4,
      jain_fairness: 0.92,
    },
    timeline: [
      { label: 'P1', start: 0, end: 2, duration: 2 },
      { label: 'CS', start: 2, end: 3, duration: 1 },
      { label: 'P2', start: 3, end: 5, duration: 2 },
      { label: 'CS', start: 5, end: 6, duration: 1 },
      { label: 'P3', start: 6, end: 8, duration: 2 },
      { label: 'CS', start: 8, end: 9, duration: 1 },
      { label: 'P1', start: 9, end: 12, duration: 3 },
      { label: 'CS', start: 12, end: 13, duration: 1 },
      { label: 'P2', start: 13, end: 14, duration: 1 },
    ],
    ready_queue_log: [],
    decision_trace: null,
  },
};

const MOCK_COMPARE_RESPONSE: CompareResponse = {
  results: MOCK_SIMULATE_RESULTS,
  comparison_summary: MOCK_COMPARISON_ROWS as unknown as Record<string, unknown>[],
};

/**
 * Helper to render components wrapped in QueryClientProvider and MemoryRouter.
 *
 * 1. The use of function:
 *    Isolates TanStack Query state and router history between test cases.
 * 2. How does the specific function work:
 *    Constructs a non-retrying QueryClient and wraps the target JSX element.
 * 3. Concepts used in the function:
 *    Test isolation, React context provisioning.
 * 4. Places where this function is imported and used:
 *    Used across all component integration tests in this file.
 * 5. Final output of the function, before and after:
 *    Before: Standalone React element.
 *    After: RenderResult mounted within testing providers.
 */
function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>
  );
}

describe('Sub-stage F4: Algorithm Comparison Suite', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    useWorkloadStore.getState().resetToDefault();
  });

  describe('CSV Export Utilities', () => {
    it('serializes comparison metrics into compliant CSV with headers', () => {
      const csv = exportComparisonToCsv(MOCK_COMPARISON_ROWS);
      const lines = csv.split('\n');

      expect(lines[0]).toBe(
        'Algorithm,Avg Waiting Time,Avg Turnaround Time,Avg Response Time,CPU Utilization (%),Jain Fairness,Context Switches'
      );
      expect(lines.length).toBe(MOCK_COMPARISON_ROWS.length + 1);

      // Verify first algorithm row values
      expect(lines[1]).toContain('"FCFS",3.67,7.00,3.67,90.9,0.850,2');
    });

    it('triggers client-side CSV download without throwing', () => {
      expect(() => {
        downloadComparisonCsv(MOCK_COMPARISON_ROWS, 'test_comparison.csv');
      }).not.toThrow();
    });
  });

  describe('ComparisonTable Component', () => {
    it('renders all algorithms and column metrics', () => {
      render(<ComparisonTable rows={MOCK_COMPARISON_ROWS} />);

      expect(screen.getByText('Performance Benchmark Metrics')).toBeInTheDocument();
      expect(screen.getAllByText('FCFS').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('SJF').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('SRTF').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('RR').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('QLEARNING').length).toBeGreaterThanOrEqual(1);

      // Check column headers
      expect(screen.getByText('Avg Waiting')).toBeInTheDocument();
      expect(screen.getByText('Avg Turnaround')).toBeInTheDocument();
      expect(screen.getByText('Avg Response')).toBeInTheDocument();
      expect(screen.getByText('CPU Utilization')).toBeInTheDocument();
      expect(screen.getByText("Jain's Fairness")).toBeInTheDocument();
      expect(screen.getByText('Context Switches')).toBeInTheDocument();
    });

    it('correctly awards Best badge based on metric directionality', () => {
      render(<ComparisonTable rows={MOCK_COMPARISON_ROWS} />);

      const bestBadges = screen.getAllByText('Best');
      expect(bestBadges.length).toBeGreaterThanOrEqual(6);

      // SRTF has lowest waiting time (2.00) -> should be highlighted
      expect(screen.getByText('2.00')).toBeInTheDocument();

      // RR has lowest response time (1.33) -> should be highlighted
      expect(screen.getByText('1.33')).toBeInTheDocument();

      // SRTF has highest CPU utilization (95.0%) -> should be highlighted
      expect(screen.getByText('95.0%')).toBeInTheDocument();
    });

    it('calculates and displays relative deltas when a baseline algorithm is selected', () => {
      render(
        <ComparisonTable rows={MOCK_COMPARISON_ROWS} defaultBaseline="FCFS" />
      );

      // Baseline badge displayed on FCFS row
      expect(screen.getByText('Baseline')).toBeInTheDocument();

      // Check that delta text is rendered for other algorithms
      // SJF waiting time dropped from 3.67 to 2.33 -> -1.34 (-36.5%)
      const deltaElements = screen.getAllByTitle(/Delta vs FCFS/i);
      expect(deltaElements.length).toBeGreaterThan(0);
    });

    it('triggers CSV download when Export Table button is clicked', () => {
      render(<ComparisonTable rows={MOCK_COMPARISON_ROWS} />);

      const exportBtn = screen.getByRole('button', { name: /Export Table to CSV/i });
      expect(exportBtn).toBeEnabled();
      expect(() => fireEvent.click(exportBtn)).not.toThrow();
    });
  });

  describe('ComparisonBars Component', () => {
    it('renders chart container and title', () => {
      render(<ComparisonBars rows={MOCK_COMPARISON_ROWS} />);

      expect(screen.getByText('Core Latency Comparison (ticks)')).toBeInTheDocument();
      expect(screen.getByRole('img', { name: /Grouped latency comparison bar chart/i })).toBeInTheDocument();
    });

    it('allows toggling between By Metric and By Algorithm grouping modes', () => {
      render(<ComparisonBars rows={MOCK_COMPARISON_ROWS} />);

      const byAlgoBtn = screen.getByRole('button', { name: /By Algorithm/i });
      fireEvent.click(byAlgoBtn);
      expect(byAlgoBtn).toHaveClass('text-primary');

      const byMetricBtn = screen.getByRole('button', { name: /By Metric/i });
      fireEvent.click(byMetricBtn);
      expect(byMetricBtn).toHaveClass('text-primary');
    });

    it('renders custom tabular tooltip content correctly', () => {
      const mockPayload = [
        { name: 'FCFS', value: 3.67, color: '#3B82F6', dataKey: 'FCFS' },
        { name: 'SJF', value: 2.33, color: '#10B981', dataKey: 'SJF' },
      ];

      render(<CustomTooltip active={true} payload={mockPayload} label="Avg Waiting Time" />);

      expect(screen.getByText('Avg Waiting Time')).toBeInTheDocument();
      expect(screen.getByText('FCFS')).toBeInTheDocument();
      expect(screen.getByText('3.67')).toBeInTheDocument();
      expect(screen.getByText('SJF')).toBeInTheDocument();
      expect(screen.getByText('2.33')).toBeInTheDocument();
    });
  });

  describe('GanttStack Component', () => {
    it('renders stacked horizontal timeline rows for each algorithm', () => {
      render(<GanttStack results={MOCK_SIMULATE_RESULTS} />);

      expect(screen.getByText('Synchronized Execution Timelines')).toBeInTheDocument();
      expect(screen.getByText('FCFS')).toBeInTheDocument();
      expect(screen.getByText('SJF')).toBeInTheDocument();
      expect(screen.getByText('RR')).toBeInTheDocument();
    });

    it('renders process labels, context switch hatch, and idle hatch', () => {
      render(<GanttStack results={MOCK_SIMULATE_RESULTS} />);

      // P1, P2, P3 should be rendered in timelines
      expect(screen.getAllByText('P1').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('P2').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('P3').length).toBeGreaterThanOrEqual(1);

      // Context switch (CS) and IDLE segments should be rendered
      expect(screen.getAllByText('CS').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('IDLE').length).toBeGreaterThanOrEqual(1);
    });

    it('renders shared time ruler and legend items', () => {
      render(<GanttStack results={MOCK_SIMULATE_RESULTS} />);

      expect(screen.getByText('Time (ticks) →')).toBeInTheDocument();
      expect(screen.getByText('Legend:')).toBeInTheDocument();
      expect(screen.getByText('CS (Context Switch)')).toBeInTheDocument();
    });
  });

  describe('ComparePage Integration', () => {
    it('auto-executes comparison on mount and displays benchmark results', async () => {
      const compareSpy = vi
        .spyOn(apiClient, 'comparePolicies')
        .mockResolvedValue(MOCK_COMPARE_RESPONSE);

      renderWithProviders(<ComparePage />);

      // Workload summary header
      expect(screen.getByText('Active Workload:')).toBeInTheDocument();

      // Mutation is dispatched automatically on mount
      await waitFor(() => {
        expect(compareSpy).toHaveBeenCalledTimes(1);
      });

      // Verify benchmark views rendered
      await waitFor(() => {
        expect(screen.getByText('Performance Benchmark Metrics')).toBeInTheDocument();
        expect(screen.getByText('Core Latency Comparison (ticks)')).toBeInTheDocument();
        expect(screen.getByText('Synchronized Execution Timelines')).toBeInTheDocument();
      });
    });

    it('updates context-switch cost and quantum inputs', async () => {
      const compareSpy = vi
        .spyOn(apiClient, 'comparePolicies')
        .mockResolvedValue(MOCK_COMPARE_RESPONSE);

      renderWithProviders(<ComparePage />);

      // Wait for initial auto-run to complete first
      await waitFor(() => {
        expect(screen.getByText('Performance Benchmark Metrics')).toBeInTheDocument();
      });

      // Toggle CS cost to 1 tick
      const cs1Btn = screen.getByRole('button', { name: /1 tick/i });
      fireEvent.click(cs1Btn);

      // Change quantum input to 4
      const quantumInput = screen.getByLabelText(/RR Quantum \(q\):/i);
      fireEvent.change(quantumInput, { target: { value: '4' } });

      // Click Run All Comparison
      const runBtn = screen.getByRole('button', { name: /Run All Comparison/i });
      fireEvent.click(runBtn);

      await waitFor(() => {
        expect(compareSpy).toHaveBeenLastCalledWith(
          expect.objectContaining({
            cs_cost: 1,
            quantum: 4,
          }),
          expect.anything()
        );
      });
    });

    it('displays friendly HTTP 503 error banner when RL model checkpoint is missing', async () => {
      const err503 = new apiClient.ApiError(
        503,
        'MODEL_UNAVAILABLE',
        'Model Not Found: Train the model first: python cli.py train-rl'
      );
      vi.spyOn(apiClient, 'comparePolicies').mockRejectedValue(err503);

      renderWithProviders(<ComparePage />);

      await waitFor(() => {
        expect(screen.getByText('Comparison Benchmark Failed')).toBeInTheDocument();
        expect(screen.getByText('HTTP 503')).toBeInTheDocument();
        expect(
          screen.getByText('Missing RL Checkpoint — Run Training Command')
        ).toBeInTheDocument();
        expect(screen.getByText('python cli.py train-rl')).toBeInTheDocument();
      });
    });
  });
});
