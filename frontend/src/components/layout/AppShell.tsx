/**
 * @file src/components/layout/AppShell.tsx
 * 1. What this file does:
 *    Renders the master application layout shell containing the Sidebar, TopBar,
 *    and responsive scrollable main content outlet.
 * 2. Overall design:
 *    Flexbox two-column layout on desktop and column layout on mobile devices.
 *    Monitors backend reachability and surfaces a global warning banner if the service is offline.
 * 3. Elements in the file:
 *    - `AppShell`: Master layout wrapper component
 *    - Offline warning banner
 *    - Main content scroll viewport with responsive padding
 * 4. Final output of the file:
 *    Top-level layout component rendered across all application routes.
 */

import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { useHealthQuery } from '@/api/hooks';
import { API_BASE_URL } from '@/api/client';
import { AlertCircle } from 'lucide-react';

/**
 * Root structural layout shell providing navigation frame and responsive page viewport.
 *
 * 1. The use of function:
 *    Wraps every route with persistent navigation (Sidebar), header (TopBar), and offline alerts.
 * 2. How does the specific function work:
 *    Combines Sidebar and TopBar into a unified grid/flex hierarchy, checks `useHealthQuery()`
 *    for backend disconnects, and renders `<Outlet />` for child page components.
 * 3. Concepts used in the function:
 *    React Router Outlet, responsive viewport containers, conditional alert banners.
 * 4. Places where this function is imported and used:
 *    Imported by App.tsx to define the root layout route.
 * 5. Final output of the function, before and after:
 *    Before: Bare page component.
 *    After: Fully framed application with navigation, top bar, and responsive layout.
 *
 * @returns JSX Element for the overall application frame.
 */
export const AppShell: React.FC = () => {
  const { isError, isLoading, data: health } = useHealthQuery();
  const isOffline = isError || (!isLoading && !health);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-foreground">
      {/* Sidebar navigation */}
      <Sidebar />

      {/* Main layout column */}
      <div className="flex flex-col flex-1 min-w-0 h-full overflow-hidden">
        {/* Top Header */}
        <TopBar />

        {/* Global Offline Warning Banner if backend unreachable */}
        {isOffline && (
          <div
            role="alert"
            className="bg-danger/10 border-b border-danger/20 text-danger px-4 py-2.5 flex items-center justify-between text-xs sm:text-sm font-medium shrink-0 animate-in fade-in"
          >
            <div className="flex items-center gap-2 truncate">
              <AlertCircle className="w-4 h-4 shrink-0 text-danger" />
              <span className="truncate">
                Backend not reachable at <span className="font-mono font-semibold">{API_BASE_URL}</span>.
                Ensure FastAPI server is running with <code className="bg-danger/20 px-1 py-0.5 rounded text-xs font-mono">python -m uvicorn schedsim.api.main:app</code>.
              </span>
            </div>
          </div>
        )}

        {/* Scrollable page content area */}
        <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8 pb-20 md:pb-8">
          <div className="max-w-7xl mx-auto">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default AppShell;
