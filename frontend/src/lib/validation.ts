/**
 * @file src/lib/validation.ts
 * 1. What this file does:
 *    Defines Zod validation schemas and error formatting helpers for scheduling workloads
 *    and individual process definitions matching FastAPI backend constraints.
 * 2. Overall design:
 *    Provides strictly typed schemas for Process and Workload lists (1 to 50 processes,
 *    bounded arrival 0-500, burst 1-100, unique PIDs). Unrolls Zod errors into structured
 *    record maps for inline UI input highlighting.
 * 3. Elements in the file:
 *    - `processSchema`: Single process Zod validator (pid, arrival, burst, priority)
 *    - `workloadListSchema`: Workload array Zod validator with uniqueness refinement
 *    - `validateProcess`: Helper for validating a single process object
 *    - `validateWorkloadList`: Helper returning both row-level and global error maps
 *    - `validateProcessField`: Helper for on-the-fly single field input validation
 * 4. Final output of the file:
 *    Exported schemas, inference types, and helper validation functions consumed by
 *    workloadStore and form components.
 */

import { z } from 'zod';
import type { ProcessSchema } from '@/api/client';

/**
 * Zod schema validating a single process configuration.
 *
 * Constraints:
 * - pid: string, 1 to 16 characters, alphanumeric with underscores and hyphens.
 * - arrival: integer between 0 and 500 inclusive.
 * - burst: integer between 1 and 100 inclusive.
 * - priority: non-negative integer (defaults to 0).
 */
export const processSchema = z.object({
  pid: z
    .string()
    .trim()
    .min(1, 'PID is required (at least 1 character)')
    .max(16, 'PID cannot exceed 16 characters')
    .regex(/^[a-zA-Z0-9_-]+$/, 'PID must contain only letters, numbers, hyphens, or underscores'),
  arrival: z.coerce
    .number({ invalid_type_error: 'Arrival must be a number' })
    .int('Arrival must be an integer')
    .min(0, 'Arrival tick must be >= 0')
    .max(500, 'Arrival tick cannot exceed 500'),
  burst: z.coerce
    .number({ invalid_type_error: 'Burst must be a number' })
    .int('Burst must be an integer')
    .min(1, 'CPU Burst must be >= 1')
    .max(100, 'CPU Burst cannot exceed 100'),
  priority: z.coerce
    .number({ invalid_type_error: 'Priority must be a number' })
    .int('Priority must be an integer')
    .min(0, 'Priority must be >= 0')
    .default(0),
});

export type ValidatedProcess = z.infer<typeof processSchema>;

/**
 * Zod schema validating a full workload list of processes.
 *
 * Constraints:
 * - Array length between 1 and 50 processes.
 * - All process IDs must be strictly unique.
 */
export const workloadListSchema = z
  .array(processSchema)
  .min(1, 'Workload must contain at least 1 process')
  .max(50, 'Workload cannot exceed 50 processes')
  .superRefine((processes, ctx) => {
    const seenPids = new Set<string>();
    const duplicatePids = new Set<string>();

    processes.forEach((p, index) => {
      const pid = p.pid.trim();
      if (!pid) return;
      if (seenPids.has(pid)) {
        duplicatePids.add(pid);
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: `Duplicate PID '${pid}' detected at row ${index + 1}`,
          path: [index, 'pid'],
        });
      } else {
        seenPids.add(pid);
      }
    });

    if (duplicatePids.size > 0) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: `PIDs must be unique. Duplicates: ${Array.from(duplicatePids).join(', ')}`,
        path: [],
      });
    }
  });

export type ValidatedWorkload = z.infer<typeof workloadListSchema>;

export interface ProcessValidationResult {
  success: boolean;
  data?: ProcessSchema;
  errors?: Record<string, string>;
}

export interface WorkloadValidationResult {
  success: boolean;
  data?: ProcessSchema[];
  globalError?: string;
  rowErrors: Record<number, Record<string, string>>;
}

/**
 * Validates a single process data object against processSchema.
 *
 * 1. The use of function:
 *    Validates user input during table cell editing or form entry.
 * 2. How does the specific function work:
 *    Executes safeParse, converts any Zod issues into a key-value error dictionary.
 * 3. Concepts used in the function:
 *    Zod safeParse, record transformation of issue paths.
 * 4. Places where this function is imported and used:
 *    Imported by workloadStore and ProcessTable.
 * 5. Final output of the function, before and after:
 *    Before: Raw object payload.
 *    After: Validation result with typed data or mapped field errors.
 *
 * @param input - Process attributes to validate.
 * @returns ProcessValidationResult
 */
export function validateProcess(input: unknown): ProcessValidationResult {
  const result = processSchema.safeParse(input);
  if (result.success) {
    return { success: true, data: result.data };
  }

  const errors: Record<string, string> = {};
  result.error.issues.forEach((issue) => {
    const field = issue.path[0];
    if (field && typeof field === 'string' && !errors[field]) {
      errors[field] = issue.message;
    }
  });

  return { success: false, errors };
}

/**
 * Validates a complete list of processes, collecting global and per-row error maps.
 *
 * 1. The use of function:
 *    Enforces workload integrity prior to API simulation dispatch or state persistence.
 * 2. How does the specific function work:
 *    Runs workloadListSchema.safeParse and categorizes issues by row index and field key.
 * 3. Concepts used in the function:
 *    Zod superRefine parsing, multi-tier error unrolling.
 * 4. Places where this function is imported and used:
 *    Imported by workloadStore, ProcessTable, and test suites.
 * 5. Final output of the function, before and after:
 *    Before: Array of raw processes.
 *    After: Detailed breakdown of global issues and row-specific field errors.
 *
 * @param processes - Array of processes to validate.
 * @returns WorkloadValidationResult
 */
export function validateWorkloadList(processes: unknown): WorkloadValidationResult {
  const result = workloadListSchema.safeParse(processes);
  const rowErrors: Record<number, Record<string, string>> = {};
  let globalError: string | undefined;

  if (result.success) {
    return { success: true, data: result.data, rowErrors };
  }

  result.error.issues.forEach((issue) => {
    if (issue.path.length === 0) {
      globalError = issue.message;
    } else if (typeof issue.path[0] === 'number') {
      const rowIndex = issue.path[0];
      const field = issue.path[1];
      if (!rowErrors[rowIndex]) {
        rowErrors[rowIndex] = {};
      }
      if (typeof field === 'string' && !rowErrors[rowIndex][field]) {
        rowErrors[rowIndex][field] = issue.message;
      } else if (!rowErrors[rowIndex]._row) {
        rowErrors[rowIndex]._row = issue.message;
      }
    } else {
      globalError = issue.message;
    }
  });

  return {
    success: false,
    globalError,
    rowErrors,
  };
}

/**
 * Validates a single process attribute field on-the-fly.
 *
 * 1. The use of function:
 *    Provides immediate inline error feedback when typing in an input field.
 * 2. How does the specific function work:
 *    Pulls the field's validator from processSchema.shape and parses the given value.
 * 3. Concepts used in the function:
 *    Dynamic Zod sub-schema validation.
 * 4. Places where this function is imported and used:
 *    Imported by ProcessTable inline text and number inputs.
 * 5. Final output of the function, before and after:
 *    Before: Raw field key and candidate value.
 *    After: Error string message if invalid, or null if valid.
 *
 * @param field - Field name ('pid' | 'arrival' | 'burst' | 'priority').
 * @param value - Candidate value.
 * @returns Error string or null if valid.
 */
export function validateProcessField(
  field: 'pid' | 'arrival' | 'burst' | 'priority',
  value: unknown
): string | null {
  const fieldValidator = processSchema.shape[field];
  if (!fieldValidator) return null;

  const result = fieldValidator.safeParse(value);
  if (result.success) {
    return null;
  }
  return result.error.issues[0]?.message ?? 'Invalid value';
}
