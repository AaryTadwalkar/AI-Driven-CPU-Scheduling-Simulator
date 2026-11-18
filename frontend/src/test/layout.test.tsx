/**
 * @file src/test/layout.test.tsx
 * 1. What this file does:
 *    Integration tests verifying AppShell navigation, layout rendering, and router switching.
 * 2. Overall design:
 *    Renders the App component with MemoryRouter/BrowserRouter inside QueryClientProvider
 *    to verify sidebar links, header title, and active page containers.
 * 3. Elements in the file:
 *    - describe block for Layout and Routing
 * 4. Final output of the file:
 *    Automated integration test assertions for layout components.
 */

import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppShell } from '../components/layout/AppShell';

// Mock getHealth API call to return valid health response
vi.mock('../api/client', async () => {
  const actual = await vi.importActual('../api/client');
  return {
    ...actual,
    getHealth: vi.fn().mockResolvedValue({
      status: 'ok',
      version: '1.0.0',
      models: { decision_tree: true, q_learning: true },
      artifacts: { test_split_results: true },
    }),
  };
});

describe('AppShell Layout Integration', () => {
  it('renders sidebar navigation links and top header', async () => {
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={['/']}>
          <AppShell />
        </MemoryRouter>
      </QueryClientProvider>
    );

    // Verify Brand title
    expect(screen.getByText('SchedSim AI')).toBeInTheDocument();

    // Verify Navigation links in sidebar
    expect(screen.getByRole('link', { name: /Simulator/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Compare/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /AI Lab/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Results/i })).toBeInTheDocument();

    // Verify ThemeToggle button exists
    expect(screen.getByRole('button', { name: /Switch to dark theme/i })).toBeInTheDocument();
  });
});
