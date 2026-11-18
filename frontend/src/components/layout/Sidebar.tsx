/**
 * @file src/components/layout/Sidebar.tsx
 * 1. What this file does:
 *    Renders the main navigational sidebar containing links to all four dashboard views
 *    (Simulator, Compare, AI Lab, Results) and responsive layout adaptability.
 * 2. Overall design:
 *    Uses React Router NavLink components with active state indicator styling.
 *    Adapts responsively: full-width 240px sidebar on large screens, compact icon-only sidebar
 *    on tablets, and horizontal bottom/drawer navigation on mobile devices.
 * 3. Elements in the file:
 *    - `NAV_ITEMS`: List of route navigation definitions with labels, paths, and icons
 *    - `Sidebar`: Primary navigational React component
 * 4. Final output of the file:
 *    Sidebar JSX structure integrated into AppShell.
 */

import React from 'react';
import { NavLink } from 'react-router-dom';
import { Cpu, BarChart3, Brain, FileText, Activity } from 'lucide-react';
import { cn } from '@/lib/cn';

interface NavItem {
  label: string;
  path: string;
  icon: React.ComponentType<{ className?: string }>;
  description: string;
}

const NAV_ITEMS: readonly NavItem[] = [
  {
    label: 'Simulator',
    path: '/',
    icon: Cpu,
    description: 'Workload & timeline playback',
  },
  {
    label: 'Compare',
    path: '/compare',
    icon: BarChart3,
    description: 'Multi-policy benchmark & stack',
  },
  {
    label: 'AI Lab',
    path: '/ai-lab',
    icon: Brain,
    description: 'Decision Tree & Q-Learning',
  },
  {
    label: 'Results',
    path: '/results',
    icon: FileText,
    description: 'Test metrics & confidence intervals',
  },
];

/**
 * Sidebar navigation component for routing across the CPU Scheduling Simulator views.
 *
 * 1. The use of function:
 *    Provides primary application navigation, displaying route links with icons and active highlights.
 * 2. How does the specific function work:
 *    Maps over `NAV_ITEMS` and creates `NavLink` elements. Highlights active links with primary colors
 *    and subtle background tints. Collapses labels on tablet viewports and docks to the bottom on mobile.
 * 3. Concepts used in the function:
 *    React Router NavLink, responsive flex/grid Tailwind classes, ARIA navigation semantics.
 * 4. Places where this function is imported and used:
 *    Imported by AppShell.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Raw route context.
 *    After: Rendered accessible navigation sidebar with active indicators.
 *
 * @returns JSX navigation element.
 */
export const Sidebar: React.FC = () => {
  return (
    <aside
      className={cn(
        'bg-surface border-r border-border z-30 transition-all duration-200',
        'fixed bottom-0 left-0 right-0 h-16 border-t md:border-t-0 md:static md:h-screen',
        'w-full md:w-20 lg:w-60 flex md:flex-col justify-between p-2 md:p-3 lg:p-4'
      )}
      aria-label="Main Navigation"
    >
      <div className="flex md:flex-col w-full">
        {/* Brand / Logo Header (hidden on mobile, compact on tablet, full on desktop) */}
        <div className="hidden md:flex items-center gap-2.5 px-2 py-3 mb-4 border-b border-border/60">
          <div className="flex items-center justify-center w-9 h-9 rounded-control bg-primary text-white shadow-subtle shrink-0">
            <Activity className="w-5 h-5 animate-pulse" />
          </div>
          <div className="hidden lg:flex flex-col overflow-hidden">
            <span className="font-semibold text-sm tracking-tight text-foreground truncate">
              SchedSim AI
            </span>
            <span className="text-[11px] text-foreground-muted truncate">
              CPU Scheduling Lab
            </span>
          </div>
        </div>

        {/* Navigation links */}
        <nav className="flex md:flex-col justify-around md:justify-start gap-1 w-full">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 px-3 py-2.5 rounded-control text-xs lg:text-sm font-medium transition-colors',
                    isActive
                      ? 'bg-primary/10 text-primary border border-primary/20 shadow-subtle'
                      : 'text-foreground-muted hover:text-foreground hover:bg-surface-muted border border-transparent'
                  )
                }
              >
                <Icon className="w-5 h-5 shrink-0" />
                <span className="hidden lg:inline truncate">{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* Footer info (desktop only) */}
      <div className="hidden lg:block pt-3 border-t border-border/60 text-[11px] text-foreground-muted">
        <p className="truncate">OS Lab · Assignment 9</p>
        <p className="text-[10px] text-foreground-muted/70">React 18 · TypeScript · Vite</p>
      </div>
    </aside>
  );
};
