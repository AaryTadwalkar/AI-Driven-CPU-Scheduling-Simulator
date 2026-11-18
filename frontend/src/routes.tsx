/**
 * @file src/routes.tsx
 * 1. What this file does:
 *    Configures client-side routing definitions for the CPU Scheduling Simulator application.
 * 2. Overall design:
 *    Uses React Router nested routes under the AppShell layout element, mapping all four
 *    primary views (Simulator, Compare, AI Lab, Results) and wildcard redirect fallback.
 * 3. Elements in the file:
 *    - `AppRoutes`: Router route definition component
 * 4. Final output of the file:
 *    Route tree element consumed by BrowserRouter in App.tsx.
 */

import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { SimulatorPage } from '@/pages/SimulatorPage';
import { ComparePage } from '@/pages/ComparePage';
import { AILabPage } from '@/pages/AILabPage';
import { ResultsPage } from '@/pages/ResultsPage';

/**
 * Route table component declaring the navigation hierarchy.
 *
 * 1. The use of function:
 *    Renders the active route page component inside the AppShell.
 * 2. How does the specific function work:
 *    Matches browser URL pathname to child route elements and falls back to '/' for unmatched URLs.
 * 3. Concepts used in the function:
 *    React Router declarative Routes and Route elements, nested route layouts.
 * 4. Places where this function is imported and used:
 *    Imported by App.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Bare routing context.
 *    After: Active route branch mounted inside AppShell.
 *
 * @returns JSX Element for route definitions.
 */
export const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<AppShell />}>
        <Route index element={<SimulatorPage />} />
        <Route path="compare" element={<ComparePage />} />
        <Route path="ai-lab" element={<AILabPage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
};
