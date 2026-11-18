/**
 * @file src/test/setup.ts
 * 1. What this file does:
 *    Configures the testing environment for Vitest and React Testing Library.
 * 2. Overall design:
 *    Imports jest-dom matchers and mocks DOM APIs (window.matchMedia) absent in jsdom.
 * 3. Elements in the file:
 *    - @testing-library/jest-dom extensions
 *    - window.matchMedia mock implementation
 * 4. Final output of the file:
 *    Configured testing runtime for all Vitest suites.
 */

import '@testing-library/jest-dom';
import React from 'react';
import { vi } from 'vitest';

// Mock window.matchMedia for JSDOM
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: (query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  }),
});

// Mock URL.createObjectURL and URL.revokeObjectURL for Blob downloads in JSDOM
if (typeof window.URL.createObjectURL === 'undefined') {
  Object.defineProperty(window.URL, 'createObjectURL', {
    writable: true,
    value: () => 'blob:http://localhost:3000/mock-uuid',
  });
}
if (typeof window.URL.revokeObjectURL === 'undefined') {
  Object.defineProperty(window.URL, 'revokeObjectURL', {
    writable: true,
    value: () => {},
  });
}

// Prevent JSDOM navigation error on programmatic anchor clicks
HTMLAnchorElement.prototype.click = () => {};

// Mock ResizeObserver for Recharts in JSDOM
if (typeof window.ResizeObserver === 'undefined') {
  class MockResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  window.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;
  global.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;
}

// Mock Recharts ResponsiveContainer to provide fixed dimensions in JSDOM
vi.mock('recharts', async (importOriginal) => {
  const original = await importOriginal<Record<string, unknown>>();
  return {
    ...original,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) =>
      React.createElement(
        'div',
        { className: 'recharts-responsive-container', style: { width: 800, height: 400 } },
        children
      ),
  };
});
