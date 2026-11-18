/**
 * @file src/App.tsx
 * 1. What this file does:
 *    Configures global application providers including TanStack QueryClientProvider
 *    and React Router BrowserRouter.
 * 2. Overall design:
 *    Creates a singleton QueryClient instance with robust defaults (no aggressive refetch on
 *    window focus, 1 retry, 5-minute cache time) and renders AppRoutes inside BrowserRouter.
 * 3. Elements in the file:
 *    - `queryClient`: Singleton TanStack QueryClient instance
 *    - `App`: Root React application component
 * 4. Final output of the file:
 *    The top-level provider tree rendered into the DOM by main.tsx.
 */

import React from 'react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppRoutes } from './routes';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 1000 * 30, // 30 seconds default stale time
    },
  },
});

/**
 * Root React application component wrapping the provider hierarchy.
 *
 * 1. The use of function:
 *    Entry component establishing React Query and React Router runtime contexts.
 * 2. How does the specific function work:
 *    Wraps child route definitions with QueryClientProvider and BrowserRouter.
 * 3. Concepts used in the function:
 *    React context providers, client-side routing, server state cache management.
 * 4. Places where this function is imported and used:
 *    Imported and mounted in main.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Uninitialized DOM root.
 *    After: Fully provisioned React application hierarchy.
 *
 * @returns JSX Element for the root application tree.
 */
export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </QueryClientProvider>
  );
};

export default App;
