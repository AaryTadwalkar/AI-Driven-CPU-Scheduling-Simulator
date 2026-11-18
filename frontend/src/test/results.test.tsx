/**
 * @file src/test/results.test.tsx
 * 1. What this file does:
 *    Unit and integration test suite for Sub-stage F5 Results Page components and ResultsPage.
 * 2. Overall design:
 *    Uses Vitest and React Testing Library to verify SummaryTable (point estimates + 95% CIs,
 *    optimality highlights, cost toggles, CSV downloads), ScenarioBreakdown (9 synthetic scenarios,
 *    ranking, bar charts, CSV export), LimitationsPanel (scientific constraints accordion),
 *    and ResultsPage integration with mocked API and 503 error handling.
 * 3. Elements in the file:
 *    - Mock evaluation fixtures matching /api/results/summary schema
 *    - `renderWithProviders`: Helper wrapping elements with QueryClient and MemoryRouter
 *    - Test suites for SummaryTable, ScenarioBreakdown, LimitationsPanel, and ResultsPage
 * 4. Final output of the file:
 *    Automated verification of the benchmark results presentation components.
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SummaryTable, type SummaryTableRow } from '../components/results/SummaryTable';
import { ScenarioBreakdown, type ScenarioBreakdownItem } from '../components/results/ScenarioBreakdown';
import { LimitationsPanel } from '../components/results/LimitationsPanel';
import { ResultsPage } from '../pages/ResultsPage';
import * as apiClient from '../api/client';
import * as csvLib from '../lib/csv';

// Mock canonical summary table rows matching results/summary_table.csv
const MOCK_SUMMARY_ROWS: SummaryTableRow[] = [
  {
    cs_cost: 0,
    policy: 'FCFS',
    workload_count: 1000,
    avg_waiting_mean: 21.27,
    avg_waiting_ci_lower: 19.84,
    avg_waiting_ci_upper: 22.62,
    avg_turnaround_mean: 29.16,
    avg_turnaround_ci_lower: 27.3,
    avg_turnaround_ci_upper: 30.85,
    avg_response_mean: 21.27,
    avg_response_ci_lower: 19.84,
    avg_response_ci_upper: 22.62,
    cpu_utilization_mean: 0.989,
    cpu_utilization_ci_lower: 0.986,
    cpu_utilization_ci_upper: 0.991,
    context_switches_mean: 6.5,
    context_switches_ci_lower: 6.31,
    context_switches_ci_upper: 6.68,
    jain_fairness_mean: 0.719,
    jain_fairness_ci_lower: 0.71,
    jain_fairness_ci_upper: 0.729,
    starvation_rate_pct: 5.3,
  },
  {
    cs_cost: 0,
    policy: 'SRTF',
    workload_count: 1000,
    avg_waiting_mean: 14.41,
    avg_waiting_ci_lower: 13.16,
    avg_waiting_ci_upper: 15.77,
    avg_turnaround_mean: 22.3,
    avg_turnaround_ci_lower: 20.73,
    avg_turnaround_ci_upper: 23.96,
    avg_response_mean: 13.52,
    avg_response_ci_lower: 12.3,
    avg_response_ci_upper: 14.86,
    cpu_utilization_mean: 0.989,
    cpu_utilization_ci_lower: 0.986,
    cpu_utilization_ci_upper: 0.991,
    context_switches_mean: 7.09,
    context_switches_ci_lower: 6.89,
    context_switches_ci_upper: 7.28,
    jain_fairness_mean: 0.856,
    jain_fairness_ci_lower: 0.852,
    jain_fairness_ci_upper: 0.86,
    starvation_rate_pct: 35.8,
  },
  {
    cs_cost: 0,
    policy: 'RL',
    workload_count: 1000,
    avg_waiting_mean: 16.74,
    avg_waiting_ci_lower: 15.39,
    avg_waiting_ci_upper: 18.19,
    avg_turnaround_mean: 24.62,
    avg_turnaround_ci_lower: 22.96,
    avg_turnaround_ci_upper: 26.36,
    avg_response_mean: 9.59,
    avg_response_ci_lower: 8.81,
    avg_response_ci_upper: 10.38,
    cpu_utilization_mean: 0.989,
    cpu_utilization_ci_lower: 0.986,
    cpu_utilization_ci_upper: 0.991,
    context_switches_mean: 13.62,
    context_switches_ci_lower: 13.03,
    context_switches_ci_upper: 14.19,
    jain_fairness_mean: 0.854,
    jain_fairness_ci_lower: 0.849,
    jain_fairness_ci_upper: 0.858,
    starvation_rate_pct: 15.7,
  },
  // cs_cost = 1 rows
  {
    cs_cost: 1,
    policy: 'FCFS',
    workload_count: 1000,
    avg_waiting_mean: 24.45,
    avg_waiting_ci_lower: 22.99,
    avg_waiting_ci_upper: 25.82,
    avg_turnaround_mean: 32.34,
    avg_turnaround_ci_lower: 30.51,
    avg_turnaround_ci_upper: 34.08,
    avg_response_mean: 24.45,
    avg_response_ci_lower: 22.99,
    avg_response_ci_upper: 25.82,
    cpu_utilization_mean: 0.865,
    cpu_utilization_ci_lower: 0.86,
    cpu_utilization_ci_upper: 0.87,
    context_switches_mean: 6.5,
    context_switches_ci_lower: 6.31,
    context_switches_ci_upper: 6.68,
    jain_fairness_mean: 0.701,
    jain_fairness_ci_lower: 0.692,
    jain_fairness_ci_upper: 0.71,
    starvation_rate_pct: 1.8,
  },
  {
    cs_cost: 1,
    policy: 'RL',
    workload_count: 1000,
    avg_waiting_mean: 19.14,
    avg_waiting_ci_lower: 17.86,
    avg_waiting_ci_upper: 20.54,
    avg_turnaround_mean: 27.03,
    avg_turnaround_ci_lower: 25.38,
    avg_turnaround_ci_upper: 28.7,
    avg_response_mean: 17.52,
    avg_response_ci_lower: 16.25,
    avg_response_ci_upper: 18.89,
    cpu_utilization_mean: 0.851,
    cpu_utilization_ci_lower: 0.846,
    cpu_utilization_ci_upper: 0.856,
    context_switches_mean: 7.35,
    context_switches_ci_lower: 7.15,
    context_switches_ci_upper: 7.55,
    jain_fairness_mean: 0.828,
    jain_fairness_ci_lower: 0.822,
    jain_fairness_ci_upper: 0.833,
    starvation_rate_pct: 8.4,
  },
];

// Mock Scenario Breakdown records
const MOCK_SCENARIO_ITEMS: ScenarioBreakdownItem[] = [
  { cs_cost: 0, scenario: 'bimodal', policy: 'SRTF', avg_waiting: 5.39 },
  { cs_cost: 0, scenario: 'bimodal', policy: 'RL', avg_waiting: 6.91 },
  { cs_cost: 0, scenario: 'bimodal', policy: 'RR', avg_waiting: 10.75 },
  { cs_cost: 0, scenario: 'bimodal', policy: 'SJF', avg_waiting: 11.1 },
  { cs_cost: 0, scenario: 'bimodal', policy: 'FCFS', avg_waiting: 20.23 },
  // short_jobs
  { cs_cost: 0, scenario: 'short_jobs', policy: 'SRTF', avg_waiting: 3.12 },
  { cs_cost: 0, scenario: 'short_jobs', policy: 'SJF', avg_waiting: 3.45 },
  { cs_cost: 0, scenario: 'short_jobs', policy: 'FCFS', avg_waiting: 5.67 },
  // cs_cost = 1
  { cs_cost: 1, scenario: 'bimodal', policy: 'RL', avg_waiting: 7.82 },
  { cs_cost: 1, scenario: 'bimodal', policy: 'SRTF', avg_waiting: 8.41 },
];

/**
 * Helper to render components wrapped in TanStack QueryClient and React Router.
 *
 * 1. The use of function:
 *    Isolates TanStack Query state and MemoryRouter history between benchmark results test cases.
 * 2. How does the specific function work:
 *    Constructs a test QueryClient with retries disabled and mounts the component inside MemoryRouter.
 * 3. Concepts used in the function:
 *    Provider wrapping for testing, isolated client-side routing, test harness decoupling.
 * 4. Places where this function is imported and used:
 *    Used across all unit and integration test blocks in this file.
 * 5. Final output of the function, before and after:
 *    Before: Standalone unmounted React element.
 *    After: RenderResult with active test providers ready for DOM queries.
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

describe('Sub-stage F5: Results Page Component Suite', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe('SummaryTable Component', () => {
    it('renders canonical benchmark table with point estimates and 95% bootstrap CIs', () => {
      renderWithProviders(<SummaryTable summaryData={MOCK_SUMMARY_ROWS} />);

      expect(screen.getByTestId('summary-table')).toBeInTheDocument();
      expect(screen.getByText(/Canonical Evaluation Summary/i)).toBeInTheDocument();
      // SRTF waiting point estimate and CI
      expect(screen.getByText('14.41')).toBeInTheDocument();
      expect(screen.getByText('[13.16, 15.77]')).toBeInTheDocument();
      // Starvation check
      expect(screen.getByText('35.8%')).toBeInTheDocument();
    });

    it('toggles between context switch cost 0 and 1', () => {
      renderWithProviders(<SummaryTable summaryData={MOCK_SUMMARY_ROWS} />);

      const cost1Btn = screen.getByRole('button', { name: /cost = 1/i });
      fireEvent.click(cost1Btn);

      // Now at cs_cost=1, FCFS wait should be 24.45
      expect(screen.getAllByText('24.45').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('[22.99, 25.82]').length).toBeGreaterThanOrEqual(1);
    });

    it('triggers CSV download on button click', () => {
      const downloadSpy = vi.spyOn(csvLib, 'downloadSummaryTableCsv').mockImplementation(() => {});

      renderWithProviders(<SummaryTable summaryData={MOCK_SUMMARY_ROWS} />);

      const downloadBtn = screen.getByRole('button', { name: /download summary csv/i });
      fireEvent.click(downloadBtn);

      expect(downloadSpy).toHaveBeenCalledWith(
        expect.any(Array),
        'summary_table_cs0.csv'
      );
    });
  });

  describe('ScenarioBreakdown Component', () => {
    it('renders scenario selector pills and switches active scenario', () => {
      renderWithProviders(<ScenarioBreakdown scenarios={MOCK_SCENARIO_ITEMS} />);

      expect(screen.getByTestId('scenario-breakdown')).toBeInTheDocument();
      expect(screen.getByText(/Performance Breakdown Across 9 Workload Scenarios/i)).toBeInTheDocument();

      // Switch from default 'bimodal' to 'short_jobs'
      const shortJobsBtn = screen.getByRole('button', { name: /short jobs/i });
      fireEvent.click(shortJobsBtn);

      expect(screen.getByText(/Short Jobs — Rapid completions/i)).toBeInTheDocument();
    });

    it('triggers scenario CSV download', () => {
      const downloadSpy = vi.spyOn(csvLib, 'downloadScenarioBreakdownCsv').mockImplementation(() => {});

      renderWithProviders(<ScenarioBreakdown scenarios={MOCK_SCENARIO_ITEMS} />);

      const downloadBtn = screen.getByRole('button', { name: /download scenario csv/i });
      fireEvent.click(downloadBtn);

      expect(downloadSpy).toHaveBeenCalledWith(
        expect.any(Array),
        'scenario_breakdown.csv'
      );
    });
  });

  describe('LimitationsPanel Component', () => {
    it('renders all 4 limitation cards and toggles accordion expansion', () => {
      renderWithProviders(<LimitationsPanel />);

      expect(screen.getByTestId('limitations-panel')).toBeInTheDocument();
      expect(screen.getByText(/1. Discrete Integer Tick Granularity/i)).toBeInTheDocument();
      expect(screen.getByText(/2. Tabular Q-Learning State Abstraction Boundaries/i)).toBeInTheDocument();
      expect(screen.getByText(/3. Offline Feature Extraction Assumption for Decision Tree/i)).toBeInTheDocument();
      expect(screen.getByText(/4. Single-Core Virtual CPU vs Multi-Core SMP Preemption/i)).toBeInTheDocument();

      // Test Expand All and Collapse All buttons
      const expandAllBtn = screen.getByRole('button', { name: /expand all/i });
      fireEvent.click(expandAllBtn);
      expect(screen.getByText(/Discrete OS scheduling pedagogy/i)).toBeInTheDocument();

      const collapseAllBtn = screen.getByRole('button', { name: /collapse all/i });
      fireEvent.click(collapseAllBtn);
      expect(screen.queryByText(/Discrete OS scheduling pedagogy/i)).not.toBeInTheDocument();
    });
  });

  describe('ResultsPage Integration', () => {
    it('renders executive summary cards, summary table, scenario breakdown, and limitations', async () => {
      vi.spyOn(apiClient, 'getResultsSummary').mockResolvedValue({
        summary_table: MOCK_SUMMARY_ROWS,
        scenario_breakdown: MOCK_SCENARIO_ITEMS,
        report_notes: '# Report Notes',
      });

      renderWithProviders(<ResultsPage />);

      expect(screen.getByTestId('results-page')).toBeInTheDocument();
      expect(screen.getByText('Benchmark Results & Empirical Evaluation')).toBeInTheDocument();

      await waitFor(() => {
        // Executive findings
        expect(screen.getByText('5.4x Regret Reduction')).toBeInTheDocument();
        expect(screen.getByText('56% Starvation Reduction')).toBeInTheDocument();
        expect(screen.getByText('46% Switching Economy')).toBeInTheDocument();
        // Embedded sections
        expect(screen.getByTestId('summary-table')).toBeInTheDocument();
        expect(screen.getByTestId('scenario-breakdown')).toBeInTheDocument();
        expect(screen.getByTestId('limitations-panel')).toBeInTheDocument();
      });
    });

    it('handles HTTP 503 missing evaluation state with CLI guidance', async () => {
      const err = new apiClient.ApiError(503, 'Evaluation missing', 'Run evaluation first: python cli.py evaluate');
      vi.spyOn(apiClient, 'getResultsSummary').mockRejectedValue(err);

      renderWithProviders(<ResultsPage />);

      await waitFor(() => {
        expect(screen.getByText(/Benchmark Evaluation Artifacts Missing/i)).toBeInTheDocument();
        expect(screen.getByText(/python cli.py evaluate/i)).toBeInTheDocument();
      });
    });
  });
});
