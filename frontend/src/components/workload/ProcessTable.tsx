/**
 * @file src/components/workload/ProcessTable.tsx
 * 1. What this file does:
 *    Renders the interactive workload process table supporting inline cell editing,
 *    row addition and deletion, CSV import/export, and real-time validation error alerts.
 * 2. Overall design:
 *    Displays process attributes (PID, Arrival, Burst, Priority) with color-coded PID badges.
 *    Provides immediate inline feedback for out-of-bound or duplicate values. Features a modal
 *    dialog supporting both file-drop and direct text pasting for CSV import.
 * 3. Elements in the file:
 *    - `ProcessTable`: React functional component
 *    - Inline editable inputs with validation styling and tooltip error popovers
 *    - Action toolbar with "Add Process", "Import CSV", "Export CSV", and "Reset" buttons
 *    - CSV Import Modal supporting both file upload and raw text pasting
 *    - Metric summary footer (Process count, Total CPU burst, Max arrival)
 * 4. Final output of the file:
 *    Exported table component integrated into WorkloadEditor and Simulator views.
 */

import React, { useState, useRef } from 'react';
import { useWorkloadStore } from '@/store/workloadStore';
import { getProcessColor } from '@/lib/colors';
import { parseProcessesFromCsv, downloadProcessesCsv } from '@/lib/csv';
import { cn } from '@/lib/cn';
import {
  Plus,
  Trash2,
  Download,
  Upload,
  RotateCcw,
  AlertCircle,
  FileText,
  X,
  Check,
} from 'lucide-react';

/**
 * Interactive process table with inline editing, CSV import/export, and validation alerts.
 *
 * 1. The use of function:
 *    Serves as the primary user interface for viewing and manually adjusting scheduling workloads.
 * 2. How does the specific function work:
 *    Connects to `useWorkloadStore`, maps processes to table rows with inline change handlers,
 *    renders validation errors per field, and handles CSV importing/exporting workflows.
 * 3. Concepts used in the function:
 *    Controlled inputs, inline validation mapping, modal dialog pattern, Blob file handling.
 * 4. Places where this function is imported and used:
 *    Imported by WorkloadEditor.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Abstract store state.
 *    After: Responsive, editable data table with live error annotations.
 *
 * @returns JSX Element for the process table.
 */
export const ProcessTable: React.FC = () => {
  const processes = useWorkloadStore((s) => s.processes);
  const validationErrors = useWorkloadStore((s) => s.validationErrors);
  const globalError = useWorkloadStore((s) => s.globalError);
  const addProcess = useWorkloadStore((s) => s.addProcess);
  const updateProcess = useWorkloadStore((s) => s.updateProcess);
  const removeProcess = useWorkloadStore((s) => s.removeProcess);
  const resetToDefault = useWorkloadStore((s) => s.resetToDefault);
  const setProcesses = useWorkloadStore((s) => s.setProcesses);

  // CSV Import Modal state
  const [isImportModalOpen, setIsImportModalOpen] = useState(false);
  const [csvInputText, setCsvInputText] = useState('');
  const [importErrors, setImportErrors] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Summary statistics
  const totalBurst = processes.reduce((acc, p) => acc + (Number(p.burst) || 0), 0);
  const maxArrival = processes.length > 0
    ? Math.max(...processes.map((p) => Number(p.arrival) || 0))
    : 0;

  const handleExportCsv = () => {
    downloadProcessesCsv(processes, 'workload.csv');
  };

  const handleOpenImportModal = () => {
    setCsvInputText('');
    setImportErrors([]);
    setIsImportModalOpen(true);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      if (text) {
        setCsvInputText(text);
        setImportErrors([]);
      }
    };
    reader.readAsText(file);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleConfirmImport = () => {
    const { processes: parsedProcesses, errors } = parseProcessesFromCsv(csvInputText);
    if (errors.length > 0) {
      setImportErrors(errors);
      return;
    }
    if (parsedProcesses.length === 0) {
      setImportErrors(['No valid processes found in input.']);
      return;
    }

    setProcesses(parsedProcesses, null);
    setIsImportModalOpen(false);
    setCsvInputText('');
    setImportErrors([]);
  };

  return (
    <div className="space-y-4">
      {/* Table Action Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={addProcess}
            disabled={processes.length >= 50}
            className="px-3 py-1.5 bg-primary hover:bg-primary-hover text-white rounded-control text-xs font-semibold flex items-center gap-1.5 transition-colors shadow-subtle disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add Process</span>
          </button>

          <button
            type="button"
            onClick={resetToDefault}
            title="Reset to Section 4.6 default fixture (P1, P2, P3)"
            className="px-3 py-1.5 bg-surface-muted hover:bg-surface text-foreground border border-border rounded-control text-xs font-medium flex items-center gap-1.5 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5 text-foreground-muted" />
            <span>Reset Fixture</span>
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleOpenImportModal}
            className="px-3 py-1.5 bg-surface-muted hover:bg-surface text-foreground border border-border rounded-control text-xs font-medium flex items-center gap-1.5 transition-colors"
          >
            <Upload className="w-3.5 h-3.5 text-foreground-muted" />
            <span>Import CSV</span>
          </button>

          <button
            type="button"
            onClick={handleExportCsv}
            disabled={processes.length === 0}
            className="px-3 py-1.5 bg-surface-muted hover:bg-surface text-foreground border border-border rounded-control text-xs font-medium flex items-center gap-1.5 transition-colors disabled:opacity-50"
          >
            <Download className="w-3.5 h-3.5 text-foreground-muted" />
            <span>Export CSV</span>
          </button>
        </div>
      </div>

      {/* Global Validation Banner if any */}
      {globalError && (
        <div className="p-3 bg-danger/10 border border-danger/30 rounded-control text-xs text-danger flex items-start gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <span className="font-medium">{globalError}</span>
        </div>
      )}

      {/* Processes Table Container */}
      <div className="overflow-x-auto border border-border rounded-control bg-surface shadow-subtle">
        <table className="w-full text-left text-xs">
          <thead className="bg-surface-muted text-foreground-muted uppercase tracking-wider text-[11px] font-semibold border-b border-border">
            <tr>
              <th className="py-2.5 px-3 w-16">#</th>
              <th className="py-2.5 px-3 min-w-[120px]">PID</th>
              <th className="py-2.5 px-3 min-w-[110px]">Arrival Tick</th>
              <th className="py-2.5 px-3 min-w-[110px]">Burst Duration</th>
              <th className="py-2.5 px-3 min-w-[100px]">Priority</th>
              <th className="py-2.5 px-3 w-12 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border font-mono">
            {processes.map((p, idx) => {
              const rowErr = validationErrors[idx] || {};
              const pidColor = getProcessColor(p.pid);

              return (
                <tr key={`${p.pid}-${idx}`} className="hover:bg-surface-muted/50 transition-colors">
                  {/* Row index indicator */}
                  <td className="py-2 px-3 text-foreground-muted text-[11px]">
                    {idx + 1}
                  </td>

                  {/* PID input with color tag */}
                  <td className="py-2 px-3">
                    <div className="flex items-center gap-1.5">
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: pidColor }}
                        title={`Color for ${p.pid}`}
                      />
                      <div className="relative flex-1">
                        <input
                          type="text"
                          value={p.pid}
                          maxLength={16}
                          onChange={(e) => updateProcess(idx, { pid: e.target.value })}
                          className={cn(
                            'w-full px-2 py-1 text-xs font-mono font-medium rounded-control border bg-surface-muted text-foreground focus:outline-none focus:ring-1',
                            rowErr.pid
                              ? 'border-danger focus:ring-danger text-danger'
                              : 'border-border focus:ring-primary'
                          )}
                        />
                        {rowErr.pid && (
                          <div className="text-[10px] text-danger font-sans mt-0.5">
                            {rowErr.pid}
                          </div>
                        )}
                      </div>
                    </div>
                  </td>

                  {/* Arrival input */}
                  <td className="py-2 px-3">
                    <div className="relative">
                      <input
                        type="number"
                        min={0}
                        max={500}
                        value={Number.isNaN(p.arrival) ? '' : p.arrival}
                        onChange={(e) =>
                          updateProcess(idx, { arrival: parseInt(e.target.value, 10) })
                        }
                        className={cn(
                          'w-full px-2 py-1 text-xs font-mono rounded-control border bg-surface-muted text-foreground focus:outline-none focus:ring-1',
                          rowErr.arrival
                            ? 'border-danger focus:ring-danger text-danger'
                            : 'border-border focus:ring-primary'
                        )}
                      />
                      {rowErr.arrival && (
                        <div className="text-[10px] text-danger font-sans mt-0.5">
                          {rowErr.arrival}
                        </div>
                      )}
                    </div>
                  </td>

                  {/* Burst input */}
                  <td className="py-2 px-3">
                    <div className="relative">
                      <input
                        type="number"
                        min={1}
                        max={100}
                        value={Number.isNaN(p.burst) ? '' : p.burst}
                        onChange={(e) =>
                          updateProcess(idx, { burst: parseInt(e.target.value, 10) })
                        }
                        className={cn(
                          'w-full px-2 py-1 text-xs font-mono rounded-control border bg-surface-muted text-foreground focus:outline-none focus:ring-1',
                          rowErr.burst
                            ? 'border-danger focus:ring-danger text-danger'
                            : 'border-border focus:ring-primary'
                        )}
                      />
                      {rowErr.burst && (
                        <div className="text-[10px] text-danger font-sans mt-0.5">
                          {rowErr.burst}
                        </div>
                      )}
                    </div>
                  </td>

                  {/* Priority input */}
                  <td className="py-2 px-3">
                    <div className="relative">
                      <input
                        type="number"
                        min={0}
                        value={Number.isNaN(p.priority) ? 0 : p.priority ?? 0}
                        onChange={(e) =>
                          updateProcess(idx, { priority: parseInt(e.target.value, 10) })
                        }
                        className={cn(
                          'w-full px-2 py-1 text-xs font-mono rounded-control border bg-surface-muted text-foreground focus:outline-none focus:ring-1',
                          rowErr.priority
                            ? 'border-danger focus:ring-danger text-danger'
                            : 'border-border focus:ring-primary'
                        )}
                      />
                      {rowErr.priority && (
                        <div className="text-[10px] text-danger font-sans mt-0.5">
                          {rowErr.priority}
                        </div>
                      )}
                    </div>
                  </td>

                  {/* Delete button */}
                  <td className="py-2 px-3 text-center">
                    <button
                      type="button"
                      onClick={() => removeProcess(idx)}
                      disabled={processes.length <= 1}
                      title={
                        processes.length <= 1
                          ? 'Workload must contain at least 1 process'
                          : `Remove ${p.pid}`
                      }
                      className="p-1 hover:bg-danger/10 text-foreground-muted hover:text-danger rounded-control transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Summary Footer */}
      <div className="flex flex-wrap items-center justify-between text-xs text-foreground-muted font-sans px-1 pt-1 border-t border-border">
        <div className="flex items-center gap-4">
          <span>
            Total Processes:{' '}
            <strong className="text-foreground font-mono">{processes.length}</strong> / 50
          </span>
          <span>
            Sum of Bursts:{' '}
            <strong className="text-foreground font-mono">{totalBurst}</strong> ticks
          </span>
          <span>
            Latest Arrival:{' '}
            <strong className="text-foreground font-mono">t = {maxArrival}</strong>
          </span>
        </div>
        <div className="text-[11px] text-foreground-muted">
          Arrival: 0–500 • Burst: 1–100 • Unique PIDs
        </div>
      </div>

      {/* CSV Import Modal */}
      {isImportModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4">
          <div className="bg-surface border border-border rounded-card p-6 max-w-lg w-full shadow-lg space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-foreground flex items-center gap-2">
                <FileText className="w-5 h-5 text-primary" />
                Import Workload from CSV
              </h3>
              <button
                type="button"
                onClick={() => setIsImportModalOpen(false)}
                className="p-1 text-foreground-muted hover:text-foreground rounded-control"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-xs text-foreground-muted">
              Upload a <span className="font-mono">.csv</span> file or paste CSV rows below.
              Headers expected: <span className="font-mono font-medium">pid,arrival,burst,priority</span>.
            </p>

            <div>
              <input
                type="file"
                ref={fileInputRef}
                accept=".csv,text/csv"
                onChange={handleFileUpload}
                className="block w-full text-xs text-foreground-muted file:mr-4 file:py-2 file:px-4 file:rounded-control file:border-0 file:text-xs file:font-semibold file:bg-surface-muted file:text-foreground hover:file:bg-surface cursor-pointer"
              />
            </div>

            <div>
              <label
                htmlFor="csv-paste-textarea"
                className="text-xs font-semibold text-foreground-muted block mb-1"
              >
                Or Paste CSV Text:
              </label>
              <textarea
                id="csv-paste-textarea"
                rows={5}
                value={csvInputText}
                onChange={(e) => {
                  setCsvInputText(e.target.value);
                  setImportErrors([]);
                }}
                placeholder="pid,arrival,burst,priority&#10;P1,0,5,0&#10;P2,1,3,0&#10;P3,2,2,0"
                className="w-full bg-surface-muted text-foreground border border-border rounded-control p-2.5 text-xs font-mono focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>

            {importErrors.length > 0 && (
              <div className="p-3 bg-danger/10 border border-danger/20 rounded-control text-xs text-danger space-y-1">
                <div className="font-semibold flex items-center gap-1.5">
                  <AlertCircle className="w-3.5 h-3.5" />
                  Import Validation Issues:
                </div>
                <ul className="list-disc pl-5 space-y-0.5 text-[11px]">
                  {importErrors.map((err, idx) => (
                    <li key={idx}>{err}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-border">
              <button
                type="button"
                onClick={() => setIsImportModalOpen(false)}
                className="px-4 py-2 bg-surface-muted hover:bg-surface text-foreground border border-border rounded-control text-xs font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleConfirmImport}
                disabled={!csvInputText.trim()}
                className="px-4 py-2 bg-primary hover:bg-primary-hover text-white rounded-control text-xs font-semibold flex items-center gap-1.5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Check className="w-4 h-4" />
                <span>Parse and Load</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessTable;
