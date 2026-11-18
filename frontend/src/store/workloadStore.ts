/**
 * @file src/store/workloadStore.ts
 * 1. What this file does:
 *    Provides the central Zustand state management store for process workloads, active presets,
 *    synthetic generator parameters, and real-time validation error state.
 * 2. Overall design:
 *    Stores workload array with automatic localStorage persistence ('schedsim_workload_v1').
 *    Re-evaluates validation constraints on every mutation, updating inline row errors and
 *    global banner messages. Defaults to the canonical Section 4.6 fixture (P1, P2, P3).
 * 3. Elements in the file:
 *    - `DEFAULT_WORKLOAD`: Section 4.6 canonical benchmark processes
 *    - `WorkloadState`: Interface detailing state properties and action signatures
 *    - `useWorkloadStore`: Zustand custom hook with persist middleware
 *    - `getNextPid`: Helper calculating the next auto-incremented process identifier
 * 4. Final output of the file:
 *    Reactive store hook consumed by WorkloadEditor, PresetPicker, GeneratorForm, ProcessTable,
 *    and simulator pages.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type { ProcessSchema } from '@/api/client';
import { validateWorkloadList } from '@/lib/validation';

/**
 * Section 4.6 hand-verified canonical workload fixture.
 * P1 (arrival 0, burst 5), P2 (arrival 1, burst 3), P3 (arrival 2, burst 2).
 */
export const DEFAULT_WORKLOAD: ProcessSchema[] = [
  { pid: 'P1', arrival: 0, burst: 5, priority: 0 },
  { pid: 'P2', arrival: 1, burst: 3, priority: 0 },
  { pid: 'P3', arrival: 2, burst: 2, priority: 0 },
];

export interface WorkloadState {
  // State variables
  processes: ProcessSchema[];
  selectedPreset: string | null;
  activeScenario: string;
  seed: number;
  nProcesses: number;
  validationErrors: Record<number, Record<string, string>>;
  globalError: string | null;

  // Actions
  setProcesses: (processes: ProcessSchema[], presetName?: string | null) => void;
  addProcess: () => void;
  updateProcess: (index: number, patch: Partial<ProcessSchema>) => void;
  removeProcess: (index: number) => void;
  resetToDefault: () => void;
  loadPresetWorkload: (presetKey: string, processes: ProcessSchema[]) => void;
  setScenario: (scenario: string) => void;
  setSeed: (seed: number) => void;
  setNProcesses: (n: number) => void;
  validateCurrentWorkload: () => boolean;
}

/**
 * Calculates the next unique PID string by inspecting current process IDs.
 *
 * 1. The use of function:
 *    Supplies sensible default process identifiers when adding a new row (e.g. 'P4', 'P5').
 * 2. How does the specific function work:
 *    Extracts trailing integers from PIDs matching /P(\d+)/i, finds the maximum value,
 *    and increments by 1. Ensures the result is not already in the list.
 * 3. Concepts used in the function:
 *    Regular expression pattern extraction, numeric maximization, collision avoidance.
 * 4. Places where this function is imported and used:
 *    Invoked by `addProcess` in useWorkloadStore.
 * 5. Final output of the function, before and after:
 *    Before: Current processes `[ { pid: 'P1' }, { pid: 'P2' } ]`.
 *    After: `'P3'`.
 *
 * @param processes - Existing array of processes.
 * @returns Next PID string.
 */
export function getNextPid(processes: ProcessSchema[]): string {
  const existingPids = new Set(processes.map((p) => p.pid.toUpperCase()));
  let maxId = 0;

  processes.forEach((p) => {
    const match = p.pid.match(/^P(\d+)$/i);
    if (match && match[1]) {
      const num = parseInt(match[1], 10);
      if (!isNaN(num) && num > maxId) {
        maxId = num;
      }
    }
  });

  let candidate = `P${maxId + 1}`;
  let count = maxId + 1;
  while (existingPids.has(candidate.toUpperCase())) {
    count++;
    candidate = `P${count}`;
  }

  return candidate;
}

/**
 * Global Zustand store managing workload configuration, presets, generator, and validation.
 *
 * 1. The use of function:
 *    Primary state hook for workload configuration across the dashboard.
 * 2. How does the specific function work:
 *    Maintains state with automatic localStorage synchronization under 'schedsim_workload_v1'.
 * 3. Concepts used in the function:
 *    Zustand state management, persist middleware, reactive validation update loop.
 * 4. Places where this function is imported and used:
 *    Imported by WorkloadEditor, ProcessTable, PresetPicker, GeneratorForm, SimulatorPage.
 * 5. Final output of the function, before and after:
 *    Before: Hook subscription.
 *    After: Reactive state values and dispatch actions.
 */
export const useWorkloadStore = create<WorkloadState>()(
  persist(
    (set, get) => ({
      processes: DEFAULT_WORKLOAD,
      selectedPreset: 'section_4_6',
      activeScenario: 'short_jobs',
      seed: 42,
      nProcesses: 5,
      validationErrors: {},
      globalError: null,

      setProcesses: (processes: ProcessSchema[], presetName = null) => {
        const validation = validateWorkloadList(processes);
        set({
          processes,
          selectedPreset: presetName,
          validationErrors: validation.rowErrors,
          globalError: validation.globalError ?? null,
        });
      },

      addProcess: () => {
        const { processes } = get();
        if (processes.length >= 50) {
          set({ globalError: 'Workload cannot exceed 50 processes.' });
          return;
        }

        const nextPid = getNextPid(processes);
        const maxArrival = processes.length > 0
          ? Math.max(...processes.map((p) => (Number.isFinite(p.arrival) ? p.arrival : 0)))
          : -1;

        const newProcess: ProcessSchema = {
          pid: nextPid,
          arrival: maxArrival + 1,
          burst: 4,
          priority: 0,
        };

        const updated = [...processes, newProcess];
        const validation = validateWorkloadList(updated);

        set({
          processes: updated,
          selectedPreset: null,
          validationErrors: validation.rowErrors,
          globalError: validation.globalError ?? null,
        });
      },

      updateProcess: (index: number, patch: Partial<ProcessSchema>) => {
        const { processes } = get();
        if (index < 0 || index >= processes.length) return;

        const updated = processes.map((p, idx) => {
          if (idx !== index) return p;
          return { ...p, ...patch };
        });

        const validation = validateWorkloadList(updated);

        set({
          processes: updated,
          selectedPreset: null,
          validationErrors: validation.rowErrors,
          globalError: validation.globalError ?? null,
        });
      },

      removeProcess: (index: number) => {
        const { processes } = get();
        if (processes.length <= 1) {
          set({ globalError: 'Workload must contain at least 1 process.' });
          return;
        }

        const updated = processes.filter((_, idx) => idx !== index);
        const validation = validateWorkloadList(updated);

        set({
          processes: updated,
          selectedPreset: null,
          validationErrors: validation.rowErrors,
          globalError: validation.globalError ?? null,
        });
      },

      resetToDefault: () => {
        const validation = validateWorkloadList(DEFAULT_WORKLOAD);
        set({
          processes: DEFAULT_WORKLOAD,
          selectedPreset: 'section_4_6',
          validationErrors: validation.rowErrors,
          globalError: validation.globalError ?? null,
        });
      },

      loadPresetWorkload: (presetKey: string, processes: ProcessSchema[]) => {
        const validation = validateWorkloadList(processes);
        set({
          processes,
          selectedPreset: presetKey,
          validationErrors: validation.rowErrors,
          globalError: validation.globalError ?? null,
        });
      },

      setScenario: (activeScenario: string) => {
        set({ activeScenario });
      },

      setSeed: (seed: number) => {
        set({ seed });
      },

      setNProcesses: (nProcesses: number) => {
        const bounded = Math.max(1, Math.min(50, nProcesses));
        set({ nProcesses: bounded });
      },

      validateCurrentWorkload: () => {
        const { processes } = get();
        const validation = validateWorkloadList(processes);
        set({
          validationErrors: validation.rowErrors,
          globalError: validation.globalError ?? null,
        });
        return validation.success;
      },
    }),
    {
      name: 'schedsim_workload_v1',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        processes: state.processes,
        selectedPreset: state.selectedPreset,
        activeScenario: state.activeScenario,
        seed: state.seed,
        nProcesses: state.nProcesses,
      }),
    }
  )
);
