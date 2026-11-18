/**
 * @file src/components/workload/PresetPicker.tsx
 * 1. What this file does:
 *    Renders the preset workload selection dropdown and description preview card.
 * 2. Overall design:
 *    Fetches standard named presets from the backend (/api/presets) via TanStack Query.
 *    Allows one-click loading of canonical workloads (e.g. Section 4.6 fixture, convoy effect,
 *    simultaneous arrivals, bimodal bursts) directly into the global workload store.
 * 3. Elements in the file:
 *    - `PresetPicker`: React functional component
 *    - Dropdown selector with category and process count tags
 *    - Detail card presenting preset description and process count
 * 4. Final output of the file:
 *    Exported component integrated into WorkloadEditor.
 */

import React from 'react';
import { usePresetsQuery } from '@/api/hooks';
import { useWorkloadStore } from '@/store/workloadStore';
import { Bookmark, Loader2, Info } from 'lucide-react';

/**
 * Dropdown and detail card for selecting and loading canonical preset workloads.
 *
 * 1. The use of function:
 *    Lets users instantly explore textbook workloads and canonical edge cases.
 * 2. How does the specific function work:
 *    Subscribes to `usePresetsQuery()`, matches `selectedPreset` from `useWorkloadStore`,
 *    and triggers `loadPresetWorkload` on user selection change.
 * 3. Concepts used in the function:
 *    TanStack Query caching, Zustand store dispatch, accessible select element.
 * 4. Places where this function is imported and used:
 *    Imported by WorkloadEditor.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Unloaded or custom workload state.
 *    After: Pre-configured processes loaded with preset metadata display.
 *
 * @returns JSX Element for preset selection.
 */
export const PresetPicker: React.FC = () => {
  const { data: presets, isLoading, isError } = usePresetsQuery();
  const selectedPreset = useWorkloadStore((s) => s.selectedPreset);
  const loadPresetWorkload = useWorkloadStore((s) => s.loadPresetWorkload);

  const presetList = presets ? Object.entries(presets) : [];
  const currentPresetData = presets && selectedPreset ? presets[selectedPreset] : null;

  const handleSelectChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const key = e.target.value;
    if (!key || !presets || !presets[key]) return;
    loadPresetWorkload(key, presets[key].processes);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <label
          htmlFor="preset-select"
          className="text-xs font-semibold uppercase tracking-wider text-foreground-muted flex items-center gap-1.5"
        >
          <Bookmark className="w-3.5 h-3.5 text-primary" />
          Canonical Workload Presets
        </label>
        {isLoading && (
          <span className="flex items-center gap-1 text-xs text-foreground-muted">
            <Loader2 className="w-3 h-3 animate-spin text-primary" />
            Loading presets...
          </span>
        )}
      </div>

      <div className="relative">
        <select
          id="preset-select"
          aria-label="Canonical Workload Presets"
          value={selectedPreset || ''}
          onChange={handleSelectChange}
          disabled={isLoading || isError || presetList.length === 0}
          className="w-full bg-surface-muted text-foreground border border-border rounded-control px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary font-sans transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
        >
          <option value="" disabled>
            {isLoading ? 'Loading presets...' : '-- Select a preset workload --'}
          </option>
          {presetList.map(([key, item]) => (
            <option key={key} value={key}>
              {key} ({item.processes.length} processes)
            </option>
          ))}
        </select>
      </div>

      {currentPresetData && (
        <div className="p-3 bg-surface-muted border border-border rounded-control text-xs text-foreground space-y-1">
          <div className="flex items-start gap-1.5 font-medium">
            <Info className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
            <span>{currentPresetData.description}</span>
          </div>
          <div className="text-foreground-muted pl-5 font-mono text-[11px]">
            {currentPresetData.processes.length} processes • Preset Key:{' '}
            <span className="text-primary font-semibold">{selectedPreset}</span>
          </div>
        </div>
      )}

      {isError && (
        <div className="text-xs text-danger bg-danger/10 border border-danger/20 rounded-control p-2">
          Unable to fetch presets from backend API. You can still create or import workloads.
        </div>
      )}
    </div>
  );
};

export default PresetPicker;
