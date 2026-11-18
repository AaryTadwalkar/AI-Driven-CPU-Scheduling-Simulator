/**
 * @file src/test/ai-lab.test.tsx
 * 1. What this file does:
 *    Unit and integration test suite for Sub-stage F5 AI Lab components and AILabPage.
 * 2. Overall design:
 *    Uses Vitest and React Testing Library to verify ConfusionMatrix, FeatureImportance,
 *    LearningCurve, PolicyTable, LiveRecommender, and AILabPage tabbed navigation and 503 states.
 * 3. Elements in the file:
 *    - Mock API fixtures for /api/ml/model-info, /api/rl/training-log, /api/rl/policy-summary, and /api/recommend
 *    - `renderWithProviders`: Helper wrapping elements with QueryClient and MemoryRouter
 *    - Test suites covering Decision Tree tab, Q-Learning tab, and HTTP 503 missing model states
 * 4. Final output of the file:
 *    Automated verification of the AI Lab interactive components.
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConfusionMatrix } from '../components/ai/ConfusionMatrix';
import { FeatureImportance, formatFeatureLabel } from '../components/ai/FeatureImportance';
import { LearningCurve } from '../components/ai/LearningCurve';
import { PolicyTable } from '../components/ai/PolicyTable';
import { LiveRecommender } from '../components/ai/LiveRecommender';
import { AILabPage } from '../pages/AILabPage';
import * as apiClient from '../api/client';

// Mock ML Model Info payload
const MOCK_ML_INFO = {
  meta: {
    model_type: 'DecisionTreeClassifier',
    feature_names: [
      'n',
      'mean_burst',
      'std_burst',
      'cv_burst',
      'max_burst',
      'min_burst',
      'max_over_median_burst',
      'mean_interarrival',
      'std_interarrival',
      'arrival_span',
      'load_ratio',
      'frac_simultaneous',
    ],
    classes: ['FCFS', 'RR', 'SJF'],
    chosen_params: {
      max_depth: 6,
      min_samples_leaf: 5,
      criterion: 'gini',
    },
    training_samples: 6000,
    validation_samples: 500,
  },
  test_metrics: {
    accuracy: 0.902,
    macro_f1: 0.8878,
    classes: ['FCFS', 'RR', 'SJF'],
    confusion_matrix: [
      [154, 3, 38],
      [9, 237, 19],
      [11, 18, 511],
    ],
    regret: {
      decision_tree: 0.0487,
      baselines: {
        FCFS: 0.5402,
        SJF: 0.2655,
        RR: 0.411,
      },
    },
    sample_count: 1000,
  },
  feature_importances: {
    max_burst: 0.43,
    std_burst: 0.2991,
    cv_burst: 0.0772,
    frac_simultaneous: 0.0764,
    n: 0.0659,
    load_ratio: 0.0329,
    mean_interarrival: 0.0124,
    mean_burst: 0.0051,
  },
};

// Mock RL Training Log records
const MOCK_RL_LOGS = [
  { episode: 0, seed: 0, cs_cost: 0, avg_waiting: 22.0, q_table_size: 100 },
  { episode: 0, seed: 1, cs_cost: 0, avg_waiting: 21.0, q_table_size: 100 },
  { episode: 5000, seed: 0, cs_cost: 0, avg_waiting: 18.0, q_table_size: 2000 },
  { episode: 5000, seed: 1, cs_cost: 0, avg_waiting: 17.5, q_table_size: 2000 },
  { episode: 20000, seed: 0, cs_cost: 0, avg_waiting: 16.5, q_table_size: 4500 },
  { episode: 20000, seed: 1, cs_cost: 0, avg_waiting: 16.9, q_table_size: 4500 },
  // cs_cost = 1
  { episode: 0, seed: 0, cs_cost: 1, avg_waiting: 25.0, q_table_size: 100 },
  { episode: 20000, seed: 0, cs_cost: 1, avg_waiting: 19.2, q_table_size: 4500 },
];

// Mock RL Policy Summary
const MOCK_POLICY_SUMMARY = {
  metadata: {
    episodes: 20000,
    seed: 3,
    cs_cost: 0,
    alpha: 0.2,
    gamma: 0.95,
    eps_decay: 0.9995,
  },
  q_table_size: 4695,
  unique_states_visited: 1327,
  top_states_count: 2,
  top_states: [
    {
      state: [3, 4, 4, 3, 1, 0, 1],
      state_str: '(3, 4, 4, 3, 1, 0, 1)',
      visits: 53956,
      best_action: 0,
      best_action_name: 'SHORTEST_REMAINING',
      q_values: {
        SHORTEST_REMAINING: -76.99,
        LONGEST_REMAINING: -81.3,
        EARLIEST_ARRIVAL: -80.35,
        CONTINUE_CURRENT: -81.34,
      },
    },
    {
      state: [1, 2, 5, 2, 1, 1, 0],
      state_str: '(1, 2, 5, 2, 1, 1, 0)',
      visits: 24100,
      best_action: 3,
      best_action_name: 'CONTINUE_CURRENT',
      q_values: {
        SHORTEST_REMAINING: -45.2,
        LONGEST_REMAINING: -52.1,
        EARLIEST_ARRIVAL: -48.0,
        CONTINUE_CURRENT: -40.1,
      },
    },
  ],
};

// Mock Recommendation Response
const MOCK_RECOMMENDATION = {
  algorithm: 'SJF',
  probabilities: {
    SJF: 0.74,
    RR: 0.18,
    FCFS: 0.08,
  },
  features: {
    n: 4,
    mean_burst: 6.5,
    std_burst: 2.1,
    cv_burst: 0.32,
    max_burst: 9,
    min_burst: 4,
  },
};

/**
 * Helper to render components wrapped in TanStack QueryClient and React Router.
 *
 * 1. The use of function:
 *    Isolates TanStack Query state and MemoryRouter history between AI Lab test cases.
 * 2. How does the specific function work:
 *    Initializes a test QueryClient with disabled query and mutation retries and mounts the target tree.
 * 3. Concepts used in the function:
 *    React Testing Library wrapper provider pattern, query cache lifecycle isolation.
 * 4. Places where this function is imported and used:
 *    Used across all unit and integration test assertions in this file.
 * 5. Final output of the function, before and after:
 *    Before: Raw unmounted React element.
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

describe('Sub-stage F5: AI Lab Component Suite', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  describe('ConfusionMatrix Component', () => {
    it('renders 3x3 heatmap grid with row totals and accuracy KPIs', () => {
      renderWithProviders(
        <ConfusionMatrix
          matrix={MOCK_ML_INFO.test_metrics.confusion_matrix}
          classes={MOCK_ML_INFO.test_metrics.classes}
          accuracy={0.902}
          macroF1={0.8878}
        />
      );

      expect(screen.getByTestId('confusion-matrix')).toBeInTheDocument();
      expect(screen.getByText(/Decision Tree Test Confusion Matrix/i)).toBeInTheDocument();
      // True positive for SJF: 511
      expect(screen.getByText('511')).toBeInTheDocument();
      // KPI check
      expect(screen.getByText('90.2%')).toBeInTheDocument();
      expect(screen.getByText('0.8878')).toBeInTheDocument();
    });

    it('toggles accessible text table view', () => {
      renderWithProviders(
        <ConfusionMatrix
          matrix={MOCK_ML_INFO.test_metrics.confusion_matrix}
          classes={MOCK_ML_INFO.test_metrics.classes}
        />
      );

      const toggleBtn = screen.getByRole('button', { name: /toggle accessible table view/i });
      fireEvent.click(toggleBtn);

      expect(screen.getByLabelText('Accessible Confusion Matrix Table')).toBeInTheDocument();
      expect(screen.getByText('Show Heatmap')).toBeInTheDocument();

      // Switch back
      fireEvent.click(toggleBtn);
      expect(screen.queryByLabelText('Accessible Confusion Matrix Table')).not.toBeInTheDocument();
    });
  });

  describe('FeatureImportance Component', () => {
    it('formats column keys to clean human-readable labels', () => {
      expect(formatFeatureLabel('cv_burst')).toBe('Burst Coeff. of Variation');
      expect(formatFeatureLabel('max_burst')).toBe('Max Burst Time');
      expect(formatFeatureLabel('n')).toBe('Process Count (n)');
    });

    it('renders horizontal bar chart container with feature rankings', () => {
      renderWithProviders(
        <FeatureImportance importances={MOCK_ML_INFO.feature_importances} />
      );

      expect(screen.getByTestId('feature-importance')).toBeInTheDocument();
      expect(screen.getByText(/Gini Feature Importance Ranking/i)).toBeInTheDocument();
      expect(screen.getByTestId('feature-importance-chart')).toBeInTheDocument();
    });
  });

  describe('LearningCurve Component', () => {
    it('renders composed chart and handles context switch cost toggle', () => {
      renderWithProviders(<LearningCurve records={MOCK_RL_LOGS} />);

      expect(screen.getByTestId('learning-curve')).toBeInTheDocument();
      expect(screen.getByText(/Tabular Q-Learning Trajectory Across 5 Seeds/i)).toBeInTheDocument();
      expect(screen.getByTestId('learning-curve-chart')).toBeInTheDocument();

      // Check cost toggle buttons
      const cost1Btn = screen.getByRole('button', { name: /cost = 1/i });
      fireEvent.click(cost1Btn);
      expect(cost1Btn).toHaveClass('font-semibold');
    });
  });

  describe('PolicyTable Component', () => {
    it('renders state 7-tuples, optimal action badges, and filters by search query', () => {
      renderWithProviders(
        <PolicyTable
          topStates={MOCK_POLICY_SUMMARY.top_states}
          qTableSize={MOCK_POLICY_SUMMARY.q_table_size}
        />
      );

      expect(screen.getByTestId('policy-table')).toBeInTheDocument();
      expect(screen.getByText('(3, 4, 4, 3, 1, 0, 1)')).toBeInTheDocument();
      expect(screen.getAllByText('SHORTEST_REMAINING').length).toBeGreaterThanOrEqual(1);

      // Search filtering
      const searchInput = screen.getByLabelText('Search States');
      fireEvent.change(searchInput, { target: { value: 'CONTINUE_CURRENT' } });

      // In table body, only CONTINUE_CURRENT row should be displayed
      expect(screen.getAllByText('CONTINUE_CURRENT').length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('LiveRecommender Component', () => {
    it('allows process editing and executes recommendation', async () => {
      vi.spyOn(apiClient, 'getRecommendation').mockResolvedValue(MOCK_RECOMMENDATION);

      renderWithProviders(<LiveRecommender />);

      expect(screen.getByTestId('live-recommender')).toBeInTheDocument();
      expect(screen.getByText(/Decision Tree Live Policy Recommender/i)).toBeInTheDocument();

      const recommendBtn = screen.getByRole('button', { name: /recommend optimal policy/i });
      fireEvent.click(recommendBtn);

      await waitFor(() => {
        expect(screen.getAllByText('SJF').length).toBeGreaterThanOrEqual(1);
        expect(screen.getAllByText('74.0%').length).toBeGreaterThanOrEqual(1);
      });

      // Extracted features section check
      expect(screen.getByText(/Extracted Workload Structural Features/i)).toBeInTheDocument();
    });
  });

  describe('AILabPage Integration', () => {
    it('renders Decision Tree tab by default and switches to Q-Learning tab', async () => {
      vi.spyOn(apiClient, 'getMLModelInfo').mockResolvedValue(MOCK_ML_INFO);
      vi.spyOn(apiClient, 'getRLTrainingLog').mockResolvedValue(MOCK_RL_LOGS);
      vi.spyOn(apiClient, 'getRLPolicySummary').mockResolvedValue(MOCK_POLICY_SUMMARY);
      vi.spyOn(apiClient, 'getRecommendation').mockResolvedValue(MOCK_RECOMMENDATION);

      renderWithProviders(<AILabPage />);

      expect(screen.getByTestId('ai-lab-page')).toBeInTheDocument();
      expect(screen.getByText('AI Scheduling Research Lab')).toBeInTheDocument();

      // Decision Tree KPIs
      await waitFor(() => {
        expect(screen.getByText('CART Decision Tree')).toBeInTheDocument();
        expect(screen.getByText('90.2%')).toBeInTheDocument();
      });

      // Switch to Q-Learning Tab
      const rlTabBtn = screen.getByRole('button', { name: /q-learning tabular agent/i });
      fireEvent.click(rlTabBtn);

      await waitFor(() => {
        expect(screen.getByText('20,000 Episodes')).toBeInTheDocument();
        expect(screen.getByText('7-Tuple Aggregate')).toBeInTheDocument();
      });
    });

    it('handles HTTP 503 missing model state with CLI instructions', async () => {
      const err = new apiClient.ApiError(503, 'Model not trained', 'Train the model first: python cli.py train-tree');
      vi.spyOn(apiClient, 'getMLModelInfo').mockRejectedValue(err);

      renderWithProviders(<AILabPage />);

      await waitFor(() => {
        expect(screen.getByText(/Decision Tree Model Checkpoint Not Trained/i)).toBeInTheDocument();
        expect(screen.getByText(/python cli.py train-tree/i)).toBeInTheDocument();
      });
    });
  });
});
