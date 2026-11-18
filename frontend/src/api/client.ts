/**
 * @file src/api/client.ts
 * 1. What this file does:
 *    Defines the typed HTTP API client for communication with the FastAPI backend service.
 *    Provides custom ApiError abstraction and strongly typed request wrappers.
 * 2. Overall design:
 *    Lightweight fetch wrapper targeting VITE_API_URL (default: http://localhost:8000).
 *    Intercepts non-2xx responses and unrolls standardized ErrorResponse payloads into ApiError.
 * 3. Elements in the file:
 *    - `ApiError`: Custom Error subclass carrying status, error code, and detailed message.
 *    - `apiFetch`: Generic authenticated/typed request dispatcher.
 *    - API endpoint functions: `getHealth`, `getPresets`, `generateWorkload`, `simulatePolicy`,
 *      `comparePolicies`, `getRecommendation`, `getMLModelInfo`, `getRLTrainingLog`,
 *      `getRLPolicySummary`, `getResultsSummary`.
 * 4. Final output of the file:
 *    Exported ApiError class and typed endpoint dispatcher functions consumed by TanStack Query hooks.
 */

import type { components } from './types.generated';

export type ProcessSchema = components['schemas']['ProcessSchema'];
export type SimulateRequest = components['schemas']['SimulateRequest'];
export type SimulateResponse = components['schemas']['SimulateResponse'];
export type SimulationMetricsSchema = components['schemas']['SimulationMetricsSchema'];
export type ProcessMetricsSchema = components['schemas']['ProcessMetricsSchema'];
export type TimelineSegmentSchema = components['schemas']['TimelineSegmentSchema'];
export type ReadyQueueLogEntry = components['schemas']['ReadyQueueLogEntry'];
export type DecisionTraceEntry = components['schemas']['DecisionTraceEntry'];
export type CompareRequest = components['schemas']['CompareRequest'];
export type CompareResponse = components['schemas']['CompareResponse'];
export type RecommendRequest = components['schemas']['RecommendRequest'];
export type RecommendResponse = components['schemas']['RecommendResponse'];
export type GenerateWorkloadRequest = components['schemas']['GenerateWorkloadRequest'];
export type GenerateWorkloadResponse = components['schemas']['GenerateWorkloadResponse'];
export type HealthResponse = components['schemas']['HealthResponse'];
export type ErrorResponse = components['schemas']['ErrorResponse'];

export const API_BASE_URL = (
  import.meta.env.VITE_API_URL || 'http://localhost:8000'
).replace(/\/$/, '');

/**
 * Custom error class wrapping HTTP error statuses and FastAPI ErrorResponse models.
 *
 * 1. The use of class:
 *    Enables consistent, type-safe error inspection across React Query error handlers and UI banners.
 * 2. What the class contains:
 *    - status: HTTP response status code (e.g. 400, 422, 503).
 *    - error: Short error code identifier (e.g. 'VALIDATION_ERROR', 'MODEL_UNAVAILABLE').
 *    - detail: Detailed human-readable explanation message.
 * 3. Technologies used and why:
 *    TypeScript class extending native Error with Object.setPrototypeOf for correct prototype chaining.
 * 4. Key design and overall design of the class:
 *    Guarantees that error boundaries and toast notifications can always access .status and .detail.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly error: string;
  readonly detail: string;

  constructor(status: number, error: string, detail: string) {
    super(`[${status}] ${error}: ${detail}`);
    this.name = 'ApiError';
    this.status = status;
    this.error = error;
    this.detail = detail;
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/**
 * Low-level JSON fetch wrapper with uniform error unrolling.
 *
 * 1. The use of function:
 *    Dispatches network requests to the backend with proper headers and error handling.
 * 2. How does the specific function work:
 *    Concatenates endpoint to API_BASE_URL, sets Content-Type application/json, parses response,
 *    and on non-ok responses constructs an ApiError with fallback values.
 * 3. Concepts used in the function:
 *    Generic fetch parsing, async/await, HTTP status interception, typed promises.
 * 4. Places where this function is imported and used:
 *    Internal to this module for all endpoint dispatchers.
 * 5. Final output of the function, before and after:
 *    Before: Network request options.
 *    After: Resolved typed payload T or rejected ApiError.
 *
 * @param path - API path relative to root (e.g. '/api/health').
 * @param init - Optional RequestInit settings.
 * @returns Promise resolving to parsed payload T.
 */
async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path}`;
  const headers = new Headers(init?.headers);
  if (!headers.has('Content-Type') && init?.method && init.method !== 'GET') {
    headers.set('Content-Type', 'application/json');
  }

  let response: Response;
  try {
    response = await fetch(url, { ...init, headers });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Network unreachable';
    throw new ApiError(0, 'NETWORK_ERROR', `Failed to reach backend at ${url}: ${message}`);
  }

  if (!response.ok) {
    let errorCode = response.statusText || 'API_ERROR';
    let errorDetail = `Request failed with HTTP status ${response.status}`;
    try {
      const errorJson = (await response.json()) as Partial<ErrorResponse>;
      if (errorJson.error) errorCode = errorJson.error;
      if (errorJson.detail) errorDetail = errorJson.detail;
    } catch {
      // Body was not valid JSON, use status text
    }
    throw new ApiError(response.status, errorCode, errorDetail);
  }

  return (await response.json()) as T;
}

/**
 * Fetches the backend service health status, version, and model checkpoint availability.
 *
 * 1. The use of function:
 *    Called periodically and on startup to power the API status dot in TopBar.
 * 2. How does the specific function work:
 *    GET request to /api/health returning HealthResponse.
 * 3. Concepts used in the function:
 *    HTTP GET, health check polling.
 * 4. Places where this function is imported and used:
 *    Used by `useHealthQuery` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: Health query invocation.
 *    After: HealthResponse object { status: 'ok', version: '...', models: {...}, artifacts: {...} }.
 */
export async function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/api/health');
}

/**
 * Fetches preset workloads configured on the backend.
 *
 * 1. The use of function:
 *    Populates the preset selector in the WorkloadEditor.
 * 2. How does the specific function work:
 *    GET request to /api/presets returning dictionary of presets.
 * 3. Concepts used in the function:
 *    HTTP GET, preset caching.
 * 4. Places where this function is imported and used:
 *    Used by `usePresetsQuery` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: Presets query invocation.
 *    After: Record mapping preset names to workload definitions.
 */
export async function getPresets(): Promise<Record<string, { description: string; processes: ProcessSchema[] }>> {
  return apiFetch<Record<string, { description: string; processes: ProcessSchema[] }>>('/api/presets');
}

/**
 * Requests backend generation of a reproducible synthetic workload.
 *
 * 1. The use of function:
 *    Synthesizes burst-heavy, interactive, or uniform workloads on demand.
 * 2. How does the specific function work:
 *    POST request to /api/workloads/generate with scenario, n, and seed.
 * 3. Concepts used in the function:
 *    HTTP POST, payload serialization.
 * 4. Places where this function is imported and used:
 *    Used by `useGenerateWorkloadMutation` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: GenerateWorkloadRequest.
 *    After: GenerateWorkloadResponse with processes list.
 */
export async function generateWorkload(req: GenerateWorkloadRequest): Promise<GenerateWorkloadResponse> {
  return apiFetch<GenerateWorkloadResponse>('/api/workloads/generate', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/**
 * Dispatches a CPU scheduling simulation run for a single algorithm policy.
 *
 * 1. The use of function:
 *    Executes simulation to compute timeline segments, metrics, ready queue log, and decision trace.
 * 2. How does the specific function work:
 *    POST request to /api/simulate with processes, algorithm, quantum, cs_cost.
 * 3. Concepts used in the function:
 *    HTTP POST mutation, simulation dispatching.
 * 4. Places where this function is imported and used:
 *    Used by `useSimulateMutation` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: SimulateRequest payload.
 *    After: SimulateResponse payload.
 */
export async function simulatePolicy(req: SimulateRequest): Promise<SimulateResponse> {
  return apiFetch<SimulateResponse>('/api/simulate', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/**
 * Runs comparative benchmarking across multiple scheduling policies.
 *
 * 1. The use of function:
 *    Executes all policies on the same workload in a single round-trip for Compare page.
 * 2. How does the specific function work:
 *    POST request to /api/compare with processes and algorithm list.
 * 3. Concepts used in the function:
 *    HTTP POST, multi-algorithm evaluation.
 * 4. Places where this function is imported and used:
 *    Used by `useCompareMutation` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: CompareRequest payload.
 *    After: CompareResponse payload containing results per policy and summary rows.
 */
export async function comparePolicies(req: CompareRequest): Promise<CompareResponse> {
  return apiFetch<CompareResponse>('/api/compare', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/**
 * Evaluates the Decision Tree policy recommendation on a given workload.
 *
 * 1. The use of function:
 *    Predicts the best algorithm and class probabilities from extracted features.
 * 2. How does the specific function work:
 *    POST request to /api/recommend with processes.
 * 3. Concepts used in the function:
 *    HTTP POST, ML inference contract.
 * 4. Places where this function is imported and used:
 *    Used by `useRecommendMutation` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: RecommendRequest payload.
 *    After: RecommendResponse with winning algorithm, probabilities, and features.
 */
export async function getRecommendation(req: RecommendRequest): Promise<RecommendResponse> {
  return apiFetch<RecommendResponse>('/api/recommend', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/**
 * Retrieves Decision Tree model metadata, training configuration, and test metrics.
 *
 * 1. The use of function:
 *    Populates the Decision Tree tab in the AI Lab view.
 * 2. How does the specific function work:
 *    GET request to /api/ml/model-info.
 * 3. Concepts used in the function:
 *    HTTP GET, metadata serialization.
 * 4. Places where this function is imported and used:
 *    Used by `useMLModelInfoQuery` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: Query trigger.
 *    After: JSON payload with accuracy, confusion matrix, and feature importances.
 */
export async function getMLModelInfo(): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>('/api/ml/model-info');
}

/**
 * Retrieves Q-Learning training curves across seeds.
 *
 * 1. The use of function:
 *    Populates episodic learning curve charts in AI Lab.
 * 2. How does the specific function work:
 *    GET request to /api/rl/training-log.
 * 3. Concepts used in the function:
 *    HTTP GET, episodic curve data transfer.
 * 4. Places where this function is imported and used:
 *    Used by `useRLTrainingLogQuery` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: Query trigger.
 *    After: JSON payload with episodes and baseline reference points.
 */
export async function getRLTrainingLog(): Promise<any> {
  return apiFetch<any>('/api/rl/training-log');
}

/**
 * Retrieves Q-table policy summary and top-visited state inspection data.
 *
 * 1. The use of function:
 *    Populates policy inspection table in AI Lab.
 * 2. How does the specific function work:
 *    GET request to /api/rl/policy-summary.
 * 3. Concepts used in the function:
 *    HTTP GET, state-action summary serialization.
 * 4. Places where this function is imported and used:
 *    Used by `useRLPolicySummaryQuery` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: Query trigger.
 *    After: JSON payload with Q-table size, unseen-state rate, and top states.
 */
export async function getRLPolicySummary(): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>('/api/rl/policy-summary');
}

/**
 * Retrieves final test evaluation summary with confidence intervals.
 *
 * 1. The use of function:
 *    Populates final benchmark tables on Results page.
 * 2. How does the specific function work:
 *    GET request to /api/results/summary.
 * 3. Concepts used in the function:
 *    HTTP GET, evaluation metrics serialization.
 * 4. Places where this function is imported and used:
 *    Used by `useResultsSummaryQuery` hook in `hooks.ts`.
 * 5. Final output of the function, before and after:
 *    Before: Query trigger.
 *    After: JSON payload with summary table, scenario breakdown, and notes.
 */
export async function getResultsSummary(): Promise<Record<string, unknown>> {
  return apiFetch<Record<string, unknown>>('/api/results/summary');
}
