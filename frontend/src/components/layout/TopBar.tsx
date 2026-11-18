/**
 * @file src/components/layout/TopBar.tsx
 * 1. What this file does:
 *    Renders the application header top bar containing the current view title,
 *    live backend API health status dot indicator, and the theme mode toggle.
 * 2. Overall design:
 *    Subscribes to React Router route changes for dynamic title resolution,
 *    and queries /api/health via TanStack Query to reflect backend connection health
 *    (green = fully operational, amber = model checkpoints missing, red = service unreachable).
 * 3. Elements in the file:
 *    - `TopBar`: React functional component
 *    - Status indicator dot with tooltip
 *    - ThemeToggle component mount
 * 4. Final output of the file:
 *    Persistent top header bar integrated into the AppShell layout.
 */

import React from 'react';
import { useLocation } from 'react-router-dom';
import { ThemeToggle } from './ThemeToggle';
import { useHealthQuery } from '@/api/hooks';
import { API_BASE_URL } from '@/api/client';
import { cn } from '@/lib/cn';
import { AlertCircle, CheckCircle2, AlertTriangle } from 'lucide-react';

const ROUTE_TITLES: Record<string, { title: string; subtitle: string }> = {
  '/': {
    title: 'Simulator & Gantt Playback',
    subtitle: 'Configure workloads, run algorithms, and inspect CPU scheduling timelines',
  },
  '/compare': {
    title: 'Algorithm Comparison',
    subtitle: 'Benchmark FCFS, SJF, RR, SRTF, and Q-Learning side-by-side',
  },
  '/ai-lab': {
    title: 'AI Lab',
    subtitle: 'Supervised Decision Tree recommendation and Tabular Q-Learning policy evaluation',
  },
  '/results': {
    title: 'Benchmark Results',
    subtitle: 'Held-out test set evaluation with 95% confidence intervals and fairness metrics',
  },
};

/**
 * TopBar navigation and status component.
 *
 * 1. The use of function:
 *    Header component displaying current view metadata, server connectivity, and theme switcher.
 * 2. How does the specific function work:
 *    Derives title from `useLocation()`, checks `useHealthQuery()` to categorize connection
 *    status as green, amber, or red, and renders the interactive ThemeToggle.
 * 3. Concepts used in the function:
 *    React Router location hook, React Query polling, dynamic badge rendering.
 * 4. Places where this function is imported and used:
 *    Imported by AppShell.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Blank top region.
 *    After: Informative top navigation bar with live status dot.
 *
 * @returns JSX Element for the top header.
 */
export const TopBar: React.FC = () => {
  const location = useLocation();
  const { data: health, isLoading, isError } = useHealthQuery();

  const currentRoute = ROUTE_TITLES[location.pathname] || {
    title: 'CPU Scheduling AI Simulator',
    subtitle: 'OS Lab Assignment',
  };

  // Determine status color and label
  let statusState: 'green' | 'amber' | 'red' = 'green';
  let statusText = 'Connected';
  let StatusIcon = CheckCircle2;

  if (isError || (!isLoading && !health)) {
    statusState = 'red';
    statusText = 'Backend Offline';
    StatusIcon = AlertCircle;
  } else if (health) {
    const modelsReady = health.models ? Object.values(health.models).every(Boolean) : true;
    if (!modelsReady) {
      statusState = 'amber';
      statusText = 'Models Missing';
      StatusIcon = AlertTriangle;
    } else {
      statusState = 'green';
      statusText = `Connected (v${health.version || '1.0'})`;
      StatusIcon = CheckCircle2;
    }
  } else {
    statusState = 'amber';
    statusText = 'Connecting...';
    StatusIcon = AlertTriangle;
  }

  return (
    <header className="h-16 px-4 lg:px-6 bg-surface border-b border-border flex items-center justify-between z-20 shrink-0">
      <div className="flex flex-col overflow-hidden mr-4">
        <h2 className="text-base sm:text-lg font-semibold text-foreground truncate">
          {currentRoute.title}
        </h2>
        <p className="text-xs text-foreground-muted hidden sm:block truncate">
          {currentRoute.subtitle}
        </p>
      </div>

      <div className="flex items-center gap-3 shrink-0">
        {/* API Status Dot & Badge */}
        <div
          className={cn(
            'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border transition-colors',
            statusState === 'green' && 'bg-success/10 text-success border-success/20',
            statusState === 'amber' && 'bg-warning/10 text-warning border-warning/20',
            statusState === 'red' && 'bg-danger/10 text-danger border-danger/20'
          )}
          title={`Target API: ${API_BASE_URL}`}
        >
          <span
            className={cn(
              'w-2 h-2 rounded-full shrink-0',
              statusState === 'green' && 'bg-success animate-pulse',
              statusState === 'amber' && 'bg-warning',
              statusState === 'red' && 'bg-danger'
            )}
          />
          <StatusIcon className="w-3.5 h-3.5" />
          <span className="hidden md:inline font-mono text-[11px]">{statusText}</span>
        </div>

        {/* Theme Toggle */}
        <ThemeToggle />
      </div>
    </header>
  );
};
