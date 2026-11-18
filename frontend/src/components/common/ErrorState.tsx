/**
 * @file src/components/common/ErrorState.tsx
 * 1. What this file does:
 *    Renders error banners and friendly diagnostics for API failures, HTTP 503 missing models,
 *    and client-side unexpected exceptions.
 * 2. Overall design:
 *    Inspects error payloads for HTTP 503 (model missing) to display tailored CLI training instructions.
 *    Provides clear title, detail messages, status codes, and an optional Retry button.
 * 3. Elements in the file:
 *    - `ErrorStateProps`: Error payload, retry callback, and custom heading configurations
 *    - `ErrorState`: React functional component
 * 4. Final output of the file:
 *    Exported error visualizer component used across all simulator pages and data views.
 */

import React from 'react';
import { AlertTriangle, RefreshCw, Terminal } from 'lucide-react';
import type { ApiError } from '@/api/client';

export interface ErrorStateProps {
  title?: string;
  error?: Error | ApiError | string | null;
  onRetry?: () => void;
  className?: string;
}

/**
 * Diagnostic error display card with HTTP 503 and CLI command guidance.
 *
 * 1. The use of function:
 *    Presents user-friendly API failure feedback with retry mechanics and actionable commands.
 * 2. How does the specific function work:
 *    Inspects the status code or error text. If HTTP 503 or model-missing error is detected,
 *    displays a highlighted bash snippet instructing the user to train the RL checkpoint.
 * 3. Concepts used in the function:
 *    Type checking for ApiError, defensive error string extraction, developer-centric UX.
 * 4. Places where this function is imported and used:
 *    Imported by ComparePage.tsx, SimulatorPage.tsx, and AILabPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Opaque network exception or unhandled promise rejection.
 *    After: Polished error panel displaying root cause and remediation instructions.
 *
 * @param props - Component parameters including error object and retry callback.
 * @returns Rendered JSX element.
 */
export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Simulation Benchmark Failed',
  error,
  onRetry,
  className = '',
}) => {
  const isApiError = error && typeof error === 'object' && 'status' in error;
  const status = isApiError ? (error as ApiError).status : undefined;
  const detail = isApiError
    ? (error as ApiError).detail
    : error instanceof Error
    ? error.message
    : typeof error === 'string'
    ? error
    : 'An unexpected error occurred while communicating with the server.';

  const isModelMissing =
    status === 503 ||
    (typeof detail === 'string' &&
      (detail.toLowerCase().includes('model') ||
        detail.toLowerCase().includes('train-rl') ||
        detail.toLowerCase().includes('checkpoint')));

  return (
    <div
      className={`p-6 rounded-card border border-danger/30 bg-surface shadow-subtle space-y-4 ${className}`}
    >
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-lg bg-danger/10 text-danger shrink-0 mt-0.5">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <div className="space-y-1 flex-1">
          <div className="flex items-center gap-2">
            <h4 className="text-sm font-semibold text-foreground">{title}</h4>
            {status && (
              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-danger/10 text-danger border border-danger/20">
                HTTP {status}
              </span>
            )}
          </div>
          <p className="text-xs text-foreground-muted leading-relaxed">{detail}</p>
        </div>
      </div>

      {isModelMissing && (
        <div className="p-3.5 rounded-control bg-surface-muted border border-border space-y-2">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-foreground">
            <Terminal className="w-3.5 h-3.5 text-primary" />
            <span>Missing RL Checkpoint — Run Training Command</span>
          </div>
          <p className="text-xs text-foreground-muted">
            The Q-learning policy checkpoint has not been trained yet. Execute the following CLI command in your backend environment to build the artifact:
          </p>
          <pre className="p-2 rounded bg-background border border-border text-xs font-mono text-primary select-all">
            python cli.py train-rl
          </pre>
        </div>
      )}

      {onRetry && (
        <div className="pt-1 flex justify-end">
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-control text-xs font-medium border border-border bg-surface text-foreground hover:bg-surface-muted transition-colors shadow-sm"
          >
            <RefreshCw className="w-3.5 h-3.5 text-primary" />
            <span>Retry Comparison</span>
          </button>
        </div>
      )}
    </div>
  );
};

export default ErrorState;
