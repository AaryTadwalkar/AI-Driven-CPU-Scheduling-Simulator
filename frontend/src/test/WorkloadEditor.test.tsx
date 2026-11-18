/**
 * @file src/test/WorkloadEditor.test.tsx
 * 1. What this file does:
 *    Component integration tests for WorkloadEditor, PresetPicker, GeneratorForm, and ProcessTable.
 * 2. Overall design:
 *    Renders the composite WorkloadEditor in JSDOM using React Testing Library wrapped in a
 *    clean QueryClientProvider. Asserts tab navigation, row additions, reset actions, and
 *    form element updates.
 * 3. Elements in the file:
 *    - `renderWithQueryClient`: Test wrapper provisioning fresh TanStack QueryClient
 *    - describe blocks for WorkloadEditor rendering, tab switching, and process editing
 * 4. Final output of the file:
 *    Automated integration tests verifying UI reactivity and user interactions.
 */

import React from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { WorkloadEditor } from '../components/workload/WorkloadEditor';
import { useWorkloadStore } from '../store/workloadStore';

/**
 * Helper creating an isolated QueryClientProvider for component tests.
 *
 * 1. The use of function:
 *    Wraps tested React components with a clean QueryClient context to isolate test executions.
 * 2. How does the specific function work:
 *    Instantiates a fresh QueryClient with retries disabled and renders the UI inside QueryClientProvider.
 * 3. Concepts used in the function:
 *    React Testing Library render, TanStack QueryClient test isolation.
 * 4. Places where this function is imported and used:
 *    Used within test cases across WorkloadEditor.test.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Standalone React element without query context.
 *    After: RenderResult with active QueryClientProvider in JSDOM.
 */
function renderWithQueryClient(ui: React.ReactElement) {
  const testQueryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={testQueryClient}>
      {ui}
    </QueryClientProvider>
  );
}

describe('WorkloadEditor Component', () => {
  beforeEach(() => {
    useWorkloadStore.getState().resetToDefault();
  });

  it('renders WorkloadEditor with header, status badge, and mode tabs', () => {
    renderWithQueryClient(<WorkloadEditor />);

    expect(screen.getByText('Workload Editor')).toBeInTheDocument();
    expect(screen.getByText('3 Processes Ready')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /table/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /presets/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /generator/i })).toBeInTheDocument();
  });

  it('displays initial Section 4.6 processes in table (P1, P2, P3)', () => {
    renderWithQueryClient(<WorkloadEditor />);

    const pidInputs = screen.getAllByDisplayValue(/P[1-3]/);
    expect(pidInputs).toHaveLength(3);
    expect(screen.getByText(/Total Processes:/i)).toBeInTheDocument();
    expect(screen.getByText(/Sum of Bursts:/i)).toBeInTheDocument();
  });

  it('switches to Presets tab and displays preset selector', () => {
    renderWithQueryClient(<WorkloadEditor />);

    const presetsTabButton = screen.getByRole('button', { name: /presets/i });
    fireEvent.click(presetsTabButton);

    expect(screen.getByText(/Canonical Workload Presets/i)).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: /Canonical Workload Presets/i })).toBeInTheDocument();
  });

  it('switches to Generator tab and displays generator controls', () => {
    renderWithQueryClient(<WorkloadEditor />);

    const generatorTabButton = screen.getByRole('button', { name: /generator/i });
    fireEvent.click(generatorTabButton);

    expect(screen.getByText(/Scenario Distribution/i)).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: /Scenario Distribution/i })).toBeInTheDocument();
    expect(screen.getByRole('slider', { name: /Process Count/i })).toBeInTheDocument();
    expect(screen.getByRole('spinbutton', { name: /Random Seed/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Generate Workload/i })).toBeInTheDocument();
  });

  it('adds a process when clicking "Add Process"', () => {
    renderWithQueryClient(<WorkloadEditor />);

    const addButton = screen.getByRole('button', { name: /add process/i });
    fireEvent.click(addButton);

    // Should now have P1, P2, P3, P4
    expect(screen.getByDisplayValue('P4')).toBeInTheDocument();
    expect(screen.getByText('4 Processes Ready')).toBeInTheDocument();
  });

  it('resets to default fixture when clicking "Reset Fixture"', () => {
    renderWithQueryClient(<WorkloadEditor />);

    // First add a process
    const addButton = screen.getByRole('button', { name: /add process/i });
    fireEvent.click(addButton);
    expect(screen.getByDisplayValue('P4')).toBeInTheDocument();

    // Click Reset
    const resetButton = screen.getByRole('button', { name: /reset fixture/i });
    fireEvent.click(resetButton);

    expect(screen.queryByDisplayValue('P4')).not.toBeInTheDocument();
    expect(screen.getByText('3 Processes Ready')).toBeInTheDocument();
  });
});
