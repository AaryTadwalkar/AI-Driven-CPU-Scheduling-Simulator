/**
 * @file src/api/hooks.ts
 * 1. What this file does:
 *    Defines TanStack React Query hooks and mutations for all backend endpoints.
 * 2. Overall design:
 *    Wraps client.ts functions in `useQuery` and `useMutation` hooks with appropriate
 *    cache keys, stale times (several minutes for static model/results data), and retry policies.
 * 3. Elements in the file:
 *    - Query key factories
 *    - `useHealthQuery`: periodically polls /api/health
 *    - `usePresetsQuery`: caches /api/presets
 *    - `useGenerateWorkloadMutation`: mutation for /api/workloads/generate
 *    - `useSimulateMutation`: mutation for /api/simulate
 *    - `useCompareMutation`: mutation for /api/compare
 *    - `useRecommendMutation`: mutation for /api/recommend
 *    - `useMLModelInfoQuery`: caches /api/ml/model-info
 *    - `useRLTrainingLogQuery`: caches /api/rl/training-log
 *    - `useRLPolicySummaryQuery`: caches /api/rl/policy-summary
 *    - `useResultsSummaryQuery`: caches /api/results/summary
 * 4. Final output of the file:
 *    Type-safe, reactive React hooks consumed across all pages and navigation components.
 */

import { useQuery, useMutation } from '@tanstack/react-query';
import {
  getHealth,
  getPresets,
  generateWorkload,
  simulatePolicy,
  comparePolicies,
  getRecommendation,
  getMLModelInfo,
  getRLTrainingLog,
  getRLPolicySummary,
  getResultsSummary,
  type ApiError,
  type HealthResponse,
  type ProcessSchema,
  type GenerateWorkloadRequest,
  type GenerateWorkloadResponse,
  type SimulateRequest,
  type SimulateResponse,
  type CompareRequest,
  type CompareResponse,
  type RecommendRequest,
  type RecommendResponse,
} from './client';

export const QUERY_KEYS = {
  health: ['health'] as const,
  presets: ['presets'] as const,
  mlModelInfo: ['ml', 'model-info'] as const,
  rlTrainingLog: ['rl', 'training-log'] as const,
  rlPolicySummary: ['rl', 'policy-summary'] as const,
  resultsSummary: ['results', 'summary'] as const,
};

/**
 * React Query hook for polling backend service health and model checkpoint statuses.
 *
 * 1. The use of function:
 *    Powers the status indicator dot in TopBar and checks backend availability.
 * 2. How does the specific function work:
 *    Dispatches `getHealth()` every 10,000ms with a 15,000ms staleTime.
 * 3. Concepts used in the function:
 *    TanStack Query polling, automatic retry.
 * 4. Places where this function is imported and used:
 *    Imported by TopBar.tsx and AppShell.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Hook subscription.
 *    After: Query result containing HealthResponse or error state.
 */
export function useHealthQuery() {
  return useQuery<HealthResponse, ApiError>({
    queryKey: QUERY_KEYS.health,
    queryFn: getHealth,
    refetchInterval: 10000,
    staleTime: 5000,
    retry: 1,
  });
}

/**
 * React Query hook for fetching and caching hand-crafted preset workloads.
 *
 * 1. The use of function:
 *    Provides pre-built process sets (P1/P2/P3, burst variance, priority sets) for WorkloadEditor.
 * 2. How does the specific function work:
 *    Dispatches `getPresets()` with 10-minute cache stale time.
 * 3. Concepts used in the function:
 *    Static query caching, stale-while-revalidate.
 * 4. Places where this function is imported and used:
 *    Imported by PresetPicker.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Hook subscription.
 *    After: Query result with preset workload mappings.
 */
export function usePresetsQuery() {
  return useQuery<Record<string, { description: string; processes: ProcessSchema[] }>, ApiError>({
    queryKey: QUERY_KEYS.presets,
    queryFn: getPresets,
    staleTime: 1000 * 60 * 10,
  });
}

/**
 * React Query mutation for generating synthetic workloads.
 *
 * 1. The use of function:
 *    Triggered by GeneratorForm to synthesize reproducible random workloads by seed.
 * 2. How does the specific function work:
 *    Executes POST /api/workloads/generate with scenario and seed parameters.
 * 3. Concepts used in the function:
 *    useMutation hook.
 * 4. Places where this function is imported and used:
 *    Imported by GeneratorForm.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Mutation trigger with GenerateWorkloadRequest.
 *    After: Resolved GenerateWorkloadResponse.
 */
export function useGenerateWorkloadMutation() {
  return useMutation<GenerateWorkloadResponse, ApiError, GenerateWorkloadRequest>({
    mutationFn: generateWorkload,
  });
}

/**
 * React Query mutation for executing a single scheduling simulation policy.
 *
 * 1. The use of function:
 *    Runs simulation on user-specified workload to compute Gantt segments, metrics, and logs.
 * 2. How does the specific function work:
 *    Calls `simulatePolicy` with SimulateRequest payload.
 * 3. Concepts used in the function:
 *    TanStack useMutation hook.
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Mutation call with SimulateRequest.
 *    After: Resolved SimulateResponse.
 */
export function useSimulateMutation() {
  return useMutation<SimulateResponse, ApiError, SimulateRequest>({
    mutationFn: simulatePolicy,
  });
}

/**
 * React Query mutation for comparing multiple scheduling policies on the same workload.
 *
 * 1. The use of function:
 *    Runs all policies simultaneously for the Compare page.
 * 2. How does the specific function work:
 *    Calls `comparePolicies` with CompareRequest payload.
 * 3. Concepts used in the function:
 *    TanStack useMutation hook.
 * 4. Places where this function is imported and used:
 *    Imported by ComparePage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Mutation call with CompareRequest.
 *    After: Resolved CompareResponse with results map and comparison summary rows.
 */
export function useCompareMutation() {
  return useMutation<CompareResponse, ApiError, CompareRequest>({
    mutationFn: comparePolicies,
  });
}

/**
 * React Query mutation for obtaining Decision Tree algorithm recommendations.
 *
 * 1. The use of function:
 *    Predicts the optimal scheduling algorithm and class probabilities from workload features.
 * 2. How does the specific function work:
 *    Calls `getRecommendation` with RecommendRequest payload.
 * 3. Concepts used in the function:
 *    TanStack useMutation hook.
 * 4. Places where this function is imported and used:
 *    Imported by RecommendationCard.tsx and SimulatorPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Mutation call with RecommendRequest.
 *    After: Resolved RecommendResponse.
 */
export function useRecommendMutation() {
  return useMutation<RecommendResponse, ApiError, RecommendRequest>({
    mutationFn: getRecommendation,
  });
}

/**
 * React Query hook for fetching Decision Tree architecture details and test evaluation metrics.
 *
 * 1. The use of function:
 *    Populates confusion matrix, feature importances, and model hyperparameters in AI Lab.
 * 2. How does the specific function work:
 *    Queries /api/ml/model-info with 10-minute cache stale time.
 * 3. Concepts used in the function:
 *    Static query caching.
 * 4. Places where this function is imported and used:
 *    Imported by AILabPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Hook subscription.
 *    After: Query result with Decision Tree metadata.
 */
export function useMLModelInfoQuery() {
  return useQuery<Record<string, unknown>, ApiError>({
    queryKey: QUERY_KEYS.mlModelInfo,
    queryFn: getMLModelInfo,
    staleTime: 1000 * 60 * 10,
  });
}

/**
 * React Query hook for fetching Q-Learning training curves.
 *
 * 1. The use of function:
 *    Renders the learning curve chart comparing RL vs heuristic baselines.
 * 2. How does the specific function work:
 *    Queries /api/rl/training-log with 10-minute cache stale time.
 * 3. Concepts used in the function:
 *    Static query caching.
 * 4. Places where this function is imported and used:
 *    Imported by LearningCurve.tsx and AILabPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Hook subscription.
 *    After: Query result with training episode logs.
 */
export function useRLTrainingLogQuery() {
  return useQuery<any, ApiError>({
    queryKey: QUERY_KEYS.rlTrainingLog,
    queryFn: getRLTrainingLog,
    staleTime: 1000 * 60 * 10,
  });
}

/**
 * React Query hook for fetching Q-table policy summary and top-visited states.
 *
 * 1. The use of function:
 *    Populates the Q-Learning policy inspection table.
 * 2. How does the specific function work:
 *    Queries /api/rl/policy-summary with 10-minute cache stale time.
 * 3. Concepts used in the function:
 *    Static query caching.
 * 4. Places where this function is imported and used:
 *    Imported by PolicyTable.tsx and AILabPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Hook subscription.
 *    After: Query result with Q-table state summaries.
 */
export function useRLPolicySummaryQuery() {
  return useQuery<Record<string, unknown>, ApiError>({
    queryKey: QUERY_KEYS.rlPolicySummary,
    queryFn: getRLPolicySummary,
    staleTime: 1000 * 60 * 10,
  });
}

/**
 * React Query hook for fetching final test-set summary benchmarks.
 *
 * 1. The use of function:
 *    Populates the Results page benchmark tables, scenario breakdowns, and limitations.
 * 2. How does the specific function work:
 *    Queries /api/results/summary with 10-minute cache stale time.
 * 3. Concepts used in the function:
 *    Static query caching.
 * 4. Places where this function is imported and used:
 *    Imported by ResultsPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Hook subscription.
 *    After: Query result with final test evaluation summaries.
 */
export function useResultsSummaryQuery() {
  return useQuery<Record<string, unknown>, ApiError>({
    queryKey: QUERY_KEYS.resultsSummary,
    queryFn: getResultsSummary,
    staleTime: 1000 * 60 * 10,
  });
}
