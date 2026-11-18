/**
 * @file src/lib/csv.ts
 * 1. What this file does:
 *    Provides CSV serialization, parsing, error auditing, and browser download helpers
 *    for scheduling simulator process workloads.
 * 2. Overall design:
 *    Implements header-aware and position-fallback CSV parsing that validates values
 *    against domain constraints (arrival >= 0, burst >= 1, unique PIDs, max 50 rows).
 *    Includes client-side file download triggering via Blob and temporary anchor element.
 * 3. Elements in the file:
 *    - `exportProcessesToCsv`: Serializes process list to formatted CSV text
 *    - `parseProcessesFromCsv`: Parses CSV text into ProcessSchema objects with detailed error logs
 *    - `downloadProcessesCsv`: Triggers automated CSV file download in the browser
 *    - `exportComparisonToCsv`: Serializes multi-policy comparison benchmark metrics to CSV text
 *    - `downloadComparisonCsv`: Triggers browser download for comparison benchmark CSV reports
 *    - `exportSummaryTableToCsv`: Serializes final test benchmark evaluation table to CSV text
 *    - `downloadSummaryTableCsv`: Triggers browser download for benchmark summary table CSV
 *    - `exportScenarioBreakdownToCsv`: Serializes per-scenario algorithm waiting times to CSV text
 *    - `downloadScenarioBreakdownCsv`: Triggers browser download for scenario breakdown CSV
 * 4. Final output of the file:
 *    Exported utility functions supporting workload and benchmark data portability.
 */

import type { ProcessSchema } from '@/api/client';
import { validateWorkloadList, processSchema } from './validation';

export interface CsvParseResult {
  processes: ProcessSchema[];
  errors: string[];
}

/**
 * Serializes an array of process definitions into a standardized CSV string.
 *
 * 1. The use of function:
 *    Enables users to export current workloads to external CSV files for sharing and analysis.
 * 2. How does the specific function work:
 *    Emits header row `pid,arrival,burst,priority` followed by comma-delimited process attributes.
 * 3. Concepts used in the function:
 *    CSV RFC 4180 compatibility, string template serialization.
 * 4. Places where this function is imported and used:
 *    Imported by ProcessTable and test suites.
 * 5. Final output of the function, before and after:
 *    Before: Array of ProcessSchema objects.
 *    After: Multiline CSV string.
 *
 * @param processes - Array of ProcessSchema objects.
 * @returns Serialized CSV string.
 */
export function exportProcessesToCsv(processes: ProcessSchema[]): string {
  const header = 'pid,arrival,burst,priority';
  const rows = processes.map((p) => {
    const priority = p.priority !== undefined ? p.priority : 0;
    return `${p.pid},${p.arrival},${p.burst},${priority}`;
  });

  return [header, ...rows].join('\n');
}

/**
 * Parses raw CSV text into a validated list of ProcessSchema instances.
 *
 * 1. The use of function:
 *    Allows importing process workloads from CSV text pasted or uploaded by the user.
 * 2. How does the specific function work:
 *    Cleans input, detects header column indices (falling back to standard 0,1,2,3 order),
 *    validates each row with Zod schema, checks overall workload uniqueness and bounds,
 *    and aggregates user-friendly error messages for invalid rows.
 * 3. Concepts used in the function:
 *    Header mapping, resilient parsing, Zod integration, duplicate detection.
 * 4. Places where this function is imported and used:
 *    Imported by ProcessTable modal/file input and test suites.
 * 5. Final output of the function, before and after:
 *    Before: Raw string from file or pasteboard.
 *    After: CsvParseResult containing validated processes and an array of error messages.
 *
 * @param csvText - Raw CSV string.
 * @returns CsvParseResult
 */
export function parseProcessesFromCsv(csvText: string): CsvParseResult {
  const errors: string[] = [];
  const processes: ProcessSchema[] = [];

  const rawLines = csvText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0 && !line.startsWith('#'));

  if (rawLines.length === 0) {
    return { processes: [], errors: ['CSV content is empty.'] };
  }

  // Detect header row: line contains named tokens or has non-numeric arrival/burst
  const firstLineCols = rawLines[0].split(',').map((c) => c.trim().toLowerCase());
  const hasNamedHeaderTokens =
    firstLineCols.includes('pid') ||
    firstLineCols.includes('arrival') ||
    firstLineCols.includes('burst') ||
    firstLineCols.includes('priority');
  const isHeaderLike =
    hasNamedHeaderTokens ||
    (firstLineCols.length > 1 && isNaN(Number(firstLineCols[1]))) ||
    (firstLineCols.length > 2 && isNaN(Number(firstLineCols[2])));

  let colMap = { pid: 0, arrival: 1, burst: 2, priority: 3 };
  let startRow = 0;

  if (isHeaderLike) {
    startRow = 1;
    colMap = {
      pid: firstLineCols.indexOf('pid'),
      arrival: firstLineCols.indexOf('arrival'),
      burst: firstLineCols.indexOf('burst'),
      priority: firstLineCols.indexOf('priority'),
    };

    if (colMap.pid === -1 || colMap.arrival === -1 || colMap.burst === -1) {
      errors.push('Missing required CSV headers: expected "pid", "arrival", and "burst".');
      return { processes: [], errors };
    }
  }

  const dataLines = rawLines.slice(startRow);
  if (dataLines.length === 0) {
    errors.push('No process data rows found in CSV.');
    return { processes: [], errors };
  }

  if (dataLines.length > 50) {
    errors.push(`Workload exceeds 50 processes maximum (found ${dataLines.length} rows).`);
  }

  dataLines.forEach((line, idx) => {
    const lineNumber = startRow + idx + 1;
    const parts = line.split(',').map((p) => p.trim());

    if (parts.length < 3) {
      errors.push(`Line ${lineNumber}: Insufficient columns (expected at least pid, arrival, burst)`);
      return;
    }

    const pidVal = parts[colMap.pid] ?? '';
    const arrivalRaw = parts[colMap.arrival];
    const burstRaw = parts[colMap.burst];
    const priorityRaw = colMap.priority !== -1 ? parts[colMap.priority] : '0';

    const arrivalVal = Number(arrivalRaw);
    const burstVal = Number(burstRaw);
    const priorityVal = priorityRaw !== undefined && priorityRaw !== '' ? Number(priorityRaw) : 0;

    const rowParse = processSchema.safeParse({
      pid: pidVal,
      arrival: arrivalVal,
      burst: burstVal,
      priority: priorityVal,
    });

    if (!rowParse.success) {
      const issueMsgs = rowParse.error.issues.map((i) => i.message).join(', ');
      errors.push(`Line ${lineNumber} (${pidVal || 'unknown'}): ${issueMsgs}`);
    } else {
      processes.push(rowParse.data);
    }
  });

  // Verify list-level constraints (uniqueness, total count)
  if (processes.length > 0) {
    const listValidation = validateWorkloadList(processes);
    if (!listValidation.success && listValidation.globalError) {
      if (!errors.includes(listValidation.globalError)) {
        errors.push(listValidation.globalError);
      }
    }
  }

  return { processes, errors };
}

/**
 * Triggers a browser download of the process workload as a CSV file.
 *
 * 1. The use of function:
 *    Invoked on clicking "Export CSV" to provide user with a physical `.csv` download.
 * 2. How does the specific function work:
 *    Converts processes to CSV text, builds a text/csv Blob, generates an object URL,
 *    and simulates a click on a temporary hidden link element.
 * 3. Concepts used in the function:
 *    Blob API, URL.createObjectURL, DOM event dispatch.
 * 4. Places where this function is imported and used:
 *    Imported by ProcessTable.
 * 5. Final output of the function, before and after:
 *    Before: In-memory ProcessSchema array.
 *    After: Browser triggers download prompt for the CSV file.
 *
 * @param processes - Array of ProcessSchema objects.
 * @param filename - Target filename (default: 'workload.csv').
 */
export function downloadProcessesCsv(processes: ProcessSchema[], filename = 'workload.csv'): void {
  if (
    typeof document === 'undefined' ||
    typeof window === 'undefined' ||
    typeof URL === 'undefined' ||
    typeof URL.createObjectURL !== 'function'
  ) {
    return;
  }

  const csvContent = exportProcessesToCsv(processes);
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

export interface ComparisonCsvRow {
  algorithm: string;
  avg_waiting: number;
  avg_turnaround: number;
  avg_response: number;
  cpu_utilization: number;
  jain_fairness: number;
  context_switches: number;
}

/**
 * Serializes benchmark comparison metrics into a structured CSV string.
 *
 * 1. The use of function:
 *    Transforms comparative benchmark summaries into downloadable CSV text.
 * 2. How does the specific function work:
 *    Generates header 'Algorithm,Avg Waiting Time,Avg Turnaround Time,Avg Response Time,CPU Utilization (%),Jain Fairness,Context Switches'
 *    and appends formatted rows for each algorithm.
 * 3. Concepts used in the function:
 *    CSV generation, tabular metric formatting, percentage normalization.
 * 4. Places where this function is imported and used:
 *    Imported by ComparisonTable.tsx and compare test suites.
 * 5. Final output of the function, before and after:
 *    Before: Array of ComparisonCsvRow objects.
 *    After: Multiline CSV string with headers and formatted values.
 *
 * @param rows - Array of comparison metric records.
 * @returns Serialized CSV string.
 */
export function exportComparisonToCsv(rows: ComparisonCsvRow[]): string {
  const headers = [
    'Algorithm',
    'Avg Waiting Time',
    'Avg Turnaround Time',
    'Avg Response Time',
    'CPU Utilization (%)',
    'Jain Fairness',
    'Context Switches',
  ];

  const dataLines = rows.map((r) => {
    const cpuPct = r.cpu_utilization <= 1.0 ? r.cpu_utilization * 100 : r.cpu_utilization;
    return [
      `"${r.algorithm}"`,
      r.avg_waiting.toFixed(2),
      r.avg_turnaround.toFixed(2),
      r.avg_response.toFixed(2),
      cpuPct.toFixed(1),
      r.jain_fairness.toFixed(3),
      r.context_switches.toString(),
    ].join(',');
  });

  return [headers.join(','), ...dataLines].join('\n');
}

/**
 * Triggers browser download of a comparative benchmark metrics CSV file.
 *
 * 1. The use of function:
 *    Provides "Export Table to CSV" capability on the Compare page.
 * 2. How does the specific function work:
 *    Serializes rows via exportComparisonToCsv, creates a temporary Blob URL,
 *    and triggers download via anchor click.
 * 3. Concepts used in the function:
 *    Blob construction, temporary object URL, DOM download dispatch.
 * 4. Places where this function is imported and used:
 *    Imported by ComparisonTable.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Comparison row data in memory.
 *    After: Browser downloads 'algorithm_comparison.csv'.
 *
 * @param rows - Array of ComparisonCsvRow objects.
 * @param filename - Target filename (default: 'algorithm_comparison.csv').
 */
export function downloadComparisonCsv(
  rows: ComparisonCsvRow[],
  filename = 'algorithm_comparison.csv'
): void {
  if (
    typeof document === 'undefined' ||
    typeof window === 'undefined' ||
    typeof URL === 'undefined' ||
    typeof URL.createObjectURL !== 'function'
  ) {
    return;
  }

  const csvContent = exportComparisonToCsv(rows);
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

export interface SummaryTableCsvRow {
  cs_cost: number;
  policy: string;
  workload_count?: number;
  avg_waiting_mean: number;
  avg_waiting_ci_lower?: number;
  avg_waiting_ci_upper?: number;
  avg_turnaround_mean: number;
  avg_turnaround_ci_lower?: number;
  avg_turnaround_ci_upper?: number;
  avg_response_mean: number;
  avg_response_ci_lower?: number;
  avg_response_ci_upper?: number;
  cpu_utilization_mean: number;
  cpu_utilization_ci_lower?: number;
  cpu_utilization_ci_upper?: number;
  context_switches_mean: number;
  context_switches_ci_lower?: number;
  context_switches_ci_upper?: number;
  jain_fairness_mean: number;
  jain_fairness_ci_lower?: number;
  jain_fairness_ci_upper?: number;
  starvation_rate_pct?: number;
}

export interface ScenarioBreakdownCsvRow {
  cs_cost: number;
  scenario: string;
  policy: string;
  avg_waiting: number;
}

/**
 * Serializes final test benchmark evaluation table rows to a standard CSV string.
 *
 * 1. The use of function:
 *    Exports benchmark summary metrics and 95% bootstrap confidence intervals for external report generation.
 * 2. How does the specific function work:
 *    Generates header row and maps each summary row to comma-delimited numeric values.
 * 3. Concepts used in the function:
 *    RFC 4180 CSV serialization, formatted statistical interval string mapping.
 * 4. Places where this function is imported and used:
 *    Imported by SummaryTable.tsx and ResultsPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Array of SummaryTableCsvRow objects.
 *    After: Multiline CSV string containing policy metrics and confidence intervals.
 *
 * @param rows - Array of SummaryTableCsvRow objects.
 * @returns Serialized CSV string.
 */
export function exportSummaryTableToCsv(rows: SummaryTableCsvRow[]): string {
  const headers = [
    'cs_cost',
    'policy',
    'workload_count',
    'avg_waiting_mean',
    'avg_waiting_ci_lower',
    'avg_waiting_ci_upper',
    'avg_turnaround_mean',
    'avg_turnaround_ci_lower',
    'avg_turnaround_ci_upper',
    'avg_response_mean',
    'avg_response_ci_lower',
    'avg_response_ci_upper',
    'cpu_utilization_mean',
    'cpu_utilization_ci_lower',
    'cpu_utilization_ci_upper',
    'context_switches_mean',
    'context_switches_ci_lower',
    'context_switches_ci_upper',
    'jain_fairness_mean',
    'jain_fairness_ci_lower',
    'jain_fairness_ci_upper',
    'starvation_rate_pct',
  ];

  const dataLines = rows.map((r) => [
    r.cs_cost,
    `"${r.policy}"`,
    r.workload_count ?? 1000,
    r.avg_waiting_mean.toFixed(3),
    (r.avg_waiting_ci_lower ?? 0).toFixed(3),
    (r.avg_waiting_ci_upper ?? 0).toFixed(3),
    r.avg_turnaround_mean.toFixed(3),
    (r.avg_turnaround_ci_lower ?? 0).toFixed(3),
    (r.avg_turnaround_ci_upper ?? 0).toFixed(3),
    r.avg_response_mean.toFixed(3),
    (r.avg_response_ci_lower ?? 0).toFixed(3),
    (r.avg_response_ci_upper ?? 0).toFixed(3),
    r.cpu_utilization_mean.toFixed(3),
    (r.cpu_utilization_ci_lower ?? 0).toFixed(3),
    (r.cpu_utilization_ci_upper ?? 0).toFixed(3),
    r.context_switches_mean.toFixed(3),
    (r.context_switches_ci_lower ?? 0).toFixed(3),
    (r.context_switches_ci_upper ?? 0).toFixed(3),
    r.jain_fairness_mean.toFixed(3),
    (r.jain_fairness_ci_lower ?? 0).toFixed(3),
    (r.jain_fairness_ci_upper ?? 0).toFixed(3),
    (r.starvation_rate_pct ?? 0).toFixed(1),
  ].join(','));

  return [headers.join(','), ...dataLines].join('\n');
}

/**
 * Triggers browser download for final test benchmark summary table CSV.
 *
 * 1. The use of function:
 *    Powers the "Download Summary CSV" button in SummaryTable.tsx.
 * 2. How does the specific function work:
 *    Serializes rows via exportSummaryTableToCsv, constructs a Blob URL, and triggers download.
 * 3. Concepts used in the function:
 *    DOM Anchor download dispatch, temporary Blob URL creation and revocation.
 * 4. Places where this function is imported and used:
 *    Imported by SummaryTable.tsx and ResultsPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: In-memory array of summary rows.
 *    After: Browser downloads 'summary_table.csv'.
 *
 * @param rows - Array of SummaryTableCsvRow objects.
 * @param filename - Target filename (default: 'summary_table.csv').
 */
export function downloadSummaryTableCsv(
  rows: SummaryTableCsvRow[],
  filename = 'summary_table.csv'
): void {
  if (
    typeof document === 'undefined' ||
    typeof window === 'undefined' ||
    typeof URL === 'undefined' ||
    typeof URL.createObjectURL !== 'function'
  ) {
    return;
  }

  const csvContent = exportSummaryTableToCsv(rows);
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

/**
 * Serializes per-scenario algorithm waiting times to CSV format.
 *
 * 1. The use of function:
 *    Exports scenario-stratified performance metrics for experimental validation.
 * 2. How does the specific function work:
 *    Maps scenario breakdown rows into CSV lines matching the backend scenario_breakdown.csv format.
 * 3. Concepts used in the function:
 *    CSV generation, tabular column alignment.
 * 4. Places where this function is imported and used:
 *    Imported by ScenarioBreakdown.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Array of ScenarioBreakdownCsvRow objects.
 *    After: Multiline CSV string of scenario breakdown data.
 *
 * @param rows - Array of ScenarioBreakdownCsvRow objects.
 * @returns Serialized CSV string.
 */
export function exportScenarioBreakdownToCsv(rows: ScenarioBreakdownCsvRow[]): string {
  const headers = ['cs_cost', 'scenario', 'policy', 'avg_waiting'];
  const dataLines = rows.map((r) => [
    r.cs_cost,
    `"${r.scenario}"`,
    `"${r.policy}"`,
    r.avg_waiting.toFixed(3),
  ].join(','));

  return [headers.join(','), ...dataLines].join('\n');
}

/**
 * Triggers browser download for scenario breakdown CSV.
 *
 * 1. The use of function:
 *    Powers the "Download Scenario CSV" button in ScenarioBreakdown.tsx.
 * 2. How does the specific function work:
 *    Serializes rows via exportScenarioBreakdownToCsv, creates a Blob, and clicks temporary anchor.
 * 3. Concepts used in the function:
 *    Client-side blob export and DOM anchor triggering.
 * 4. Places where this function is imported and used:
 *    Imported by ScenarioBreakdown.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Scenario data in memory.
 *    After: Browser downloads 'scenario_breakdown.csv'.
 *
 * @param rows - Array of ScenarioBreakdownCsvRow objects.
 * @param filename - Target filename (default: 'scenario_breakdown.csv').
 */
export function downloadScenarioBreakdownCsv(
  rows: ScenarioBreakdownCsvRow[],
  filename = 'scenario_breakdown.csv'
): void {
  if (
    typeof document === 'undefined' ||
    typeof window === 'undefined' ||
    typeof URL === 'undefined' ||
    typeof URL.createObjectURL !== 'function'
  ) {
    return;
  }

  const csvContent = exportScenarioBreakdownToCsv(rows);
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}
