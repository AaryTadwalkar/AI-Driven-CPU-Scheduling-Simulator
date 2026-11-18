/**
 * @file src/test/workload.test.ts
 * 1. What this file does:
 *    Unit test suite covering Zod validation schemas, CSV import/export parsing utilities,
 *    and Zustand workload store state mutations.
 * 2. Overall design:
 *    Uses Vitest to assert boundary conditions for CPU burst, arrival ticks, PID formats,
 *    workload list uniqueness, CSV RFC parsing tolerance, and store action state transitions.
 * 3. Elements in the file:
 *    - describe blocks for Zod validation (`processSchema`, `workloadListSchema`, helpers)
 *    - describe blocks for CSV helpers (`exportProcessesToCsv`, `parseProcessesFromCsv`)
 *    - describe blocks for Zustand store (`useWorkloadStore`)
 * 4. Final output of the file:
 *    Automated test verification for Sub-stage F2.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  processSchema,
  workloadListSchema,
  validateProcess,
  validateWorkloadList,
  validateProcessField,
} from '../lib/validation';
import { exportProcessesToCsv, parseProcessesFromCsv, downloadProcessesCsv } from '../lib/csv';
import { useWorkloadStore, DEFAULT_WORKLOAD, getNextPid } from '../store/workloadStore';
import type { ProcessSchema } from '../api/client';

describe('Workload Validation Schemas', () => {
  describe('processSchema', () => {
    it('accepts valid process attributes', () => {
      const valid = { pid: 'P1', arrival: 0, burst: 5, priority: 2 };
      const result = processSchema.safeParse(valid);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.pid).toBe('P1');
        expect(result.data.arrival).toBe(0);
        expect(result.data.burst).toBe(5);
        expect(result.data.priority).toBe(2);
      }
    });

    it('defaults priority to 0 when omitted', () => {
      const valid = { pid: 'P1', arrival: 0, burst: 5 };
      const result = processSchema.safeParse(valid);
      expect(result.success).toBe(true);
      if (result.success) {
        expect(result.data.priority).toBe(0);
      }
    });

    it('enforces PID length (1 to 16 chars) and allowed characters', () => {
      expect(processSchema.safeParse({ pid: '', arrival: 0, burst: 1 }).success).toBe(false);
      expect(
        processSchema.safeParse({ pid: 'A'.repeat(17), arrival: 0, burst: 1 }).success
      ).toBe(false);
      expect(processSchema.safeParse({ pid: 'P-1_A', arrival: 0, burst: 1 }).success).toBe(true);
      expect(
        processSchema.safeParse({ pid: 'P 1 with spaces', arrival: 0, burst: 1 }).success
      ).toBe(false);
      expect(
        processSchema.safeParse({ pid: 'P1@special', arrival: 0, burst: 1 }).success
      ).toBe(false);
    });

    it('enforces arrival boundaries (0 <= arrival <= 500, integer)', () => {
      expect(processSchema.safeParse({ pid: 'P1', arrival: 0, burst: 1 }).success).toBe(true);
      expect(processSchema.safeParse({ pid: 'P1', arrival: 500, burst: 1 }).success).toBe(true);
      expect(processSchema.safeParse({ pid: 'P1', arrival: -1, burst: 1 }).success).toBe(false);
      expect(processSchema.safeParse({ pid: 'P1', arrival: 501, burst: 1 }).success).toBe(false);
      expect(processSchema.safeParse({ pid: 'P1', arrival: 2.5, burst: 1 }).success).toBe(false);
    });

    it('enforces burst boundaries (1 <= burst <= 100, integer)', () => {
      expect(processSchema.safeParse({ pid: 'P1', arrival: 0, burst: 1 }).success).toBe(true);
      expect(processSchema.safeParse({ pid: 'P1', arrival: 0, burst: 100 }).success).toBe(true);
      expect(processSchema.safeParse({ pid: 'P1', arrival: 0, burst: 0 }).success).toBe(false);
      expect(processSchema.safeParse({ pid: 'P1', arrival: 0, burst: 101 }).success).toBe(false);
      expect(processSchema.safeParse({ pid: 'P1', arrival: 0, burst: 4.2 }).success).toBe(false);
    });
  });

  describe('workloadListSchema', () => {
    it('accepts valid workload list within 1 to 50 processes', () => {
      const result = workloadListSchema.safeParse(DEFAULT_WORKLOAD);
      expect(result.success).toBe(true);
    });

    it('rejects empty workload list', () => {
      const result = workloadListSchema.safeParse([]);
      expect(result.success).toBe(false);
    });

    it('rejects workloads exceeding 50 processes', () => {
      const oversized: ProcessSchema[] = Array.from({ length: 51 }, (_, i) => ({
        pid: `P${i + 1}`,
        arrival: i,
        burst: 2,
        priority: 0,
      }));
      const result = workloadListSchema.safeParse(oversized);
      expect(result.success).toBe(false);
    });

    it('rejects workloads with duplicate PIDs', () => {
      const duplicates: ProcessSchema[] = [
        { pid: 'P1', arrival: 0, burst: 5, priority: 0 },
        { pid: 'P2', arrival: 1, burst: 3, priority: 0 },
        { pid: 'P1', arrival: 2, burst: 2, priority: 0 },
      ];
      const result = workloadListSchema.safeParse(duplicates);
      expect(result.success).toBe(false);
      if (!result.success) {
        const hasDuplicateMessage = result.error.issues.some((issue) =>
          issue.message.toLowerCase().includes('duplicate')
        );
        expect(hasDuplicateMessage).toBe(true);
      }
    });
  });

  describe('validation helpers', () => {
    it('validateProcess returns success or error map', () => {
      const valid = validateProcess({ pid: 'P1', arrival: 0, burst: 4 });
      expect(valid.success).toBe(true);
      expect(valid.data?.pid).toBe('P1');

      const invalid = validateProcess({ pid: '', arrival: -5, burst: 0 });
      expect(invalid.success).toBe(false);
      expect(invalid.errors?.pid).toBeDefined();
      expect(invalid.errors?.arrival).toBeDefined();
      expect(invalid.errors?.burst).toBeDefined();
    });

    it('validateWorkloadList maps row-level and global issues', () => {
      const workloadWithRowAndDupErrors: ProcessSchema[] = [
        { pid: 'P1', arrival: 0, burst: 5, priority: 0 },
        { pid: 'P1', arrival: -1, burst: 0, priority: 0 }, // duplicate PID + invalid arrival + burst
      ];
      const result = validateWorkloadList(workloadWithRowAndDupErrors);
      expect(result.success).toBe(false);
      expect(result.rowErrors[1]?.arrival).toBeDefined();
      expect(result.rowErrors[1]?.burst).toBeDefined();
      expect(result.globalError).toBeDefined();
    });

    it('validateProcessField validates single field immediately', () => {
      expect(validateProcessField('burst', 10)).toBeNull();
      expect(validateProcessField('burst', 0)).toContain('>= 1');
      expect(validateProcessField('arrival', -1)).toContain('>= 0');
      expect(validateProcessField('pid', 'Valid_PID')).toBeNull();
      expect(validateProcessField('pid', '')).toContain('required');
    });
  });
});

describe('CSV Utilities', () => {
  const sampleProcesses: ProcessSchema[] = [
    { pid: 'P1', arrival: 0, burst: 5, priority: 2 },
    { pid: 'P2', arrival: 1, burst: 3, priority: 1 },
  ];

  it('exportProcessesToCsv generates correct headers and comma-separated rows', () => {
    const csv = exportProcessesToCsv(sampleProcesses);
    const lines = csv.split('\n');
    expect(lines[0]).toBe('pid,arrival,burst,priority');
    expect(lines[1]).toBe('P1,0,5,2');
    expect(lines[2]).toBe('P2,1,3,1');
  });

  it('parseProcessesFromCsv parses standard CSV with headers', () => {
    const csv = 'pid,arrival,burst,priority\nP1,0,5,2\nP2,1,3,1\n';
    const { processes, errors } = parseProcessesFromCsv(csv);
    expect(errors).toHaveLength(0);
    expect(processes).toHaveLength(2);
    expect(processes[0]).toEqual({ pid: 'P1', arrival: 0, burst: 5, priority: 2 });
    expect(processes[1]).toEqual({ pid: 'P2', arrival: 1, burst: 3, priority: 1 });
  });

  it('parseProcessesFromCsv handles missing optional priority column', () => {
    const csv = 'pid,arrival,burst\nP1,0,5\nP2,1,3\n';
    const { processes, errors } = parseProcessesFromCsv(csv);
    expect(errors).toHaveLength(0);
    expect(processes).toHaveLength(2);
    expect(processes[0].priority).toBe(0);
    expect(processes[1].priority).toBe(0);
  });

  it('parseProcessesFromCsv ignores blank lines and comments', () => {
    const csv = `
      # This is a comment
      pid, arrival, burst, priority

      P1, 0, 5, 0
      # Another comment
      P2, 2, 4, 1
    `;
    const { processes, errors } = parseProcessesFromCsv(csv);
    expect(errors).toHaveLength(0);
    expect(processes).toHaveLength(2);
    expect(processes[0].pid).toBe('P1');
    expect(processes[1].pid).toBe('P2');
  });

  it('parseProcessesFromCsv catches missing required headers', () => {
    const badCsv = 'process_name,arr,time\nP1,0,5';
    const { processes, errors } = parseProcessesFromCsv(badCsv);
    expect(processes).toHaveLength(0);
    expect(errors.length).toBeGreaterThan(0);
    expect(errors[0]).toContain('Missing required CSV headers');
  });

  it('parseProcessesFromCsv reports malformed rows and out-of-range bounds', () => {
    const malformedCsv = `pid,arrival,burst,priority
P1,0,5,0
P2,-5,0,0
P3,not_a_number,10,0`;
    const { processes, errors } = parseProcessesFromCsv(malformedCsv);
    expect(processes).toHaveLength(1);
    expect(errors.length).toBeGreaterThanOrEqual(2);
  });

  it('parseProcessesFromCsv reports duplicate PIDs in CSV', () => {
    const duplicateCsv = `pid,arrival,burst,priority
P1,0,5,0
P1,2,4,0`;
    const { processes, errors } = parseProcessesFromCsv(duplicateCsv);
    expect(processes).toHaveLength(2);
    expect(errors.some((e) => e.toLowerCase().includes('unique'))).toBe(true);
  });

  it('downloadProcessesCsv executes without error in browser-like environment', () => {
    expect(() => downloadProcessesCsv(sampleProcesses)).not.toThrow();
  });
});

describe('Zustand Workload Store', () => {
  beforeEach(() => {
    // Reset store state to initial Section 4.6 default fixture
    useWorkloadStore.getState().resetToDefault();
  });

  it('initializes with Section 4.6 canonical fixture', () => {
    const state = useWorkloadStore.getState();
    expect(state.processes).toHaveLength(3);
    expect(state.processes[0]).toEqual({ pid: 'P1', arrival: 0, burst: 5, priority: 0 });
    expect(state.processes[1]).toEqual({ pid: 'P2', arrival: 1, burst: 3, priority: 0 });
    expect(state.processes[2]).toEqual({ pid: 'P3', arrival: 2, burst: 2, priority: 0 });
    expect(state.selectedPreset).toBe('section_4_6');
    expect(state.globalError).toBeNull();
  });

  it('getNextPid increments cleanly based on existing PIDs', () => {
    expect(getNextPid(DEFAULT_WORKLOAD)).toBe('P4');
    expect(getNextPid([{ pid: 'P10', arrival: 0, burst: 1, priority: 0 }])).toBe('P11');
    expect(getNextPid([])).toBe('P1');
  });

  it('addProcess appends a process with incremented PID and max arrival + 1', () => {
    useWorkloadStore.getState().addProcess();
    const state = useWorkloadStore.getState();
    expect(state.processes).toHaveLength(4);
    const added = state.processes[3];
    expect(added.pid).toBe('P4');
    expect(added.arrival).toBe(3); // max arrival (2) + 1
    expect(added.burst).toBe(4);
    expect(state.selectedPreset).toBeNull();
  });

  it('updateProcess updates specific fields and revalidates', () => {
    useWorkloadStore.getState().updateProcess(0, { burst: 8 });
    const state = useWorkloadStore.getState();
    expect(state.processes[0].burst).toBe(8);
    expect(state.validationErrors[0]?.burst).toBeUndefined();

    // Set invalid burst (0)
    useWorkloadStore.getState().updateProcess(0, { burst: 0 });
    const invalidState = useWorkloadStore.getState();
    expect(invalidState.validationErrors[0]?.burst).toBeDefined();
  });

  it('removeProcess removes row and prevents removing the last process', () => {
    useWorkloadStore.getState().removeProcess(2); // Remove P3
    expect(useWorkloadStore.getState().processes).toHaveLength(2);

    useWorkloadStore.getState().removeProcess(1); // Remove P2
    expect(useWorkloadStore.getState().processes).toHaveLength(1);

    // Attempting to remove the last process should be blocked
    useWorkloadStore.getState().removeProcess(0);
    expect(useWorkloadStore.getState().processes).toHaveLength(1);
    expect(useWorkloadStore.getState().globalError).toContain('at least 1 process');
  });

  it('loadPresetWorkload replaces processes and updates preset name', () => {
    const convoyPreset: ProcessSchema[] = [
      { pid: 'P1', arrival: 0, burst: 20, priority: 0 },
      { pid: 'P2', arrival: 1, burst: 2, priority: 0 },
    ];
    useWorkloadStore.getState().loadPresetWorkload('convoy', convoyPreset);
    const state = useWorkloadStore.getState();
    expect(state.processes).toEqual(convoyPreset);
    expect(state.selectedPreset).toBe('convoy');
    expect(state.globalError).toBeNull();
  });

  it('manages generator parameters (scenario, seed, nProcesses)', () => {
    useWorkloadStore.getState().setScenario('bimodal');
    useWorkloadStore.getState().setSeed(12345);
    useWorkloadStore.getState().setNProcesses(15);

    const state = useWorkloadStore.getState();
    expect(state.activeScenario).toBe('bimodal');
    expect(state.seed).toBe(12345);
    expect(state.nProcesses).toBe(15);

    // Bounds checking on nProcesses
    useWorkloadStore.getState().setNProcesses(100);
    expect(useWorkloadStore.getState().nProcesses).toBe(50);
    useWorkloadStore.getState().setNProcesses(-5);
    expect(useWorkloadStore.getState().nProcesses).toBe(1);
  });
});
