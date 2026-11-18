/**
 * @file src/components/workload/WorkloadEditor.tsx
 * 1. What this file does:
 *    Combines PresetPicker, GeneratorForm, and ProcessTable into a unified responsive card
 *    providing seamless switching between manual table editing, preset loading, and synthetic generation.
 * 2. Overall design:
 *    Offers a clean tabbed configuration header ('Table & Edit', 'Presets', 'Generator')
 *    over an interactive process table. Allows users to switch workloads rapidly without losing
 *    immediate visibility of the live process list, burst sums, and validation health.
 * 3. Elements in the file:
 *    - `WorkloadEditor`: React functional container component
 *    - Mode tabs navigation (Manual, Preset, Generator)
 *    - PresetPicker and GeneratorForm integration sections
 *    - ProcessTable primary workspace
 * 4. Final output of the file:
 *    Exported composite component mounted on SimulatorPage and ComparePage.
 */

import React, { useState } from 'react';
import { PresetPicker } from './PresetPicker';
import { GeneratorForm } from './GeneratorForm';
import { ProcessTable } from './ProcessTable';
import { useWorkloadStore } from '@/store/workloadStore';
import { cn } from '@/lib/cn';
import { Table, Bookmark, Sparkles, SlidersHorizontal } from 'lucide-react';

type EditorTab = 'table' | 'presets' | 'generator';

/**
 * Composite workload editing workbench uniting manual table, presets, and generator.
 *
 * 1. The use of function:
 *    Main entry point for defining the input process workload for simulation and benchmarks.
 * 2. How does the specific function work:
 *    Manages active tab state; renders selected configuration controls (PresetPicker or
 *    GeneratorForm) above the live ProcessTable.
 * 3. Concepts used in the function:
 *    Tabbed segmented interface, compound component architecture, responsive grid layout.
 * 4. Places where this function is imported and used:
 *    Imported by SimulatorPage.tsx and ComparePage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Standalone disconnected components.
 *    After: Unified, polished workload creation dashboard.
 *
 * @returns JSX Element for the complete workload editor.
 */
export const WorkloadEditor: React.FC = () => {
  const [activeTab, setActiveTab] = useState<EditorTab>('table');
  const processes = useWorkloadStore((s) => s.processes);
  const selectedPreset = useWorkloadStore((s) => s.selectedPreset);
  const globalError = useWorkloadStore((s) => s.globalError);
  const validationErrors = useWorkloadStore((s) => s.validationErrors);

  const hasRowErrors = Object.keys(validationErrors).length > 0;
  const isInvalid = Boolean(globalError) || hasRowErrors;

  return (
    <div className="rounded-card border border-border bg-surface shadow-subtle overflow-hidden">
      {/* Card Header & Mode Switcher */}
      <div className="p-4 border-b border-border bg-surface-muted/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="w-4 h-4 text-primary" />
            <h2 className="text-base font-bold text-foreground">Workload Editor</h2>
            <span
              className={cn(
                'text-[11px] font-mono px-2 py-0.5 rounded-control font-semibold',
                isInvalid
                  ? 'bg-danger/10 text-danger border border-danger/30'
                  : 'bg-success/10 text-success border border-success/30'
              )}
            >
              {isInvalid ? 'Validation Issues' : `${processes.length} Processes Ready`}
            </span>
            {selectedPreset && (
              <span className="text-[11px] font-mono px-2 py-0.5 bg-primary/10 text-primary border border-primary/20 rounded-control">
                Preset: {selectedPreset}
              </span>
            )}
          </div>
          <p className="text-xs text-foreground-muted mt-0.5">
            Configure arrival ticks, burst durations, and priorities manually, from presets, or via synthetic generation.
          </p>
        </div>

        {/* Segmented Mode Control Tabs */}
        <div className="flex items-center p-1 bg-surface-muted border border-border rounded-control self-start sm:self-auto">
          <button
            type="button"
            onClick={() => setActiveTab('table')}
            className={cn(
              'px-3 py-1 text-xs font-semibold rounded-control flex items-center gap-1.5 transition-all',
              activeTab === 'table'
                ? 'bg-surface text-primary shadow-subtle'
                : 'text-foreground-muted hover:text-foreground'
            )}
          >
            <Table className="w-3.5 h-3.5" />
            <span>Table</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('presets')}
            className={cn(
              'px-3 py-1 text-xs font-semibold rounded-control flex items-center gap-1.5 transition-all',
              activeTab === 'presets'
                ? 'bg-surface text-primary shadow-subtle'
                : 'text-foreground-muted hover:text-foreground'
            )}
          >
            <Bookmark className="w-3.5 h-3.5" />
            <span>Presets</span>
          </button>

          <button
            type="button"
            onClick={() => setActiveTab('generator')}
            className={cn(
              'px-3 py-1 text-xs font-semibold rounded-control flex items-center gap-1.5 transition-all',
              activeTab === 'generator'
                ? 'bg-surface text-primary shadow-subtle'
                : 'text-foreground-muted hover:text-foreground'
            )}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Generator</span>
          </button>
        </div>
      </div>

      {/* Auxiliary Controls (Preset picker / Generator form) */}
      {activeTab === 'presets' && (
        <div className="p-4 border-b border-border bg-surface-muted/20">
          <PresetPicker />
        </div>
      )}

      {activeTab === 'generator' && (
        <div className="p-4 border-b border-border bg-surface-muted/20">
          <GeneratorForm />
        </div>
      )}

      {/* Main Process Table Content */}
      <div className="p-4">
        <ProcessTable />
      </div>
    </div>
  );
};

export default WorkloadEditor;
