/**
 * @file src/components/layout/ThemeToggle.tsx
 * 1. What this file does:
 *    Renders a button component allowing the user to toggle between light and dark themes.
 * 2. Overall design:
 *    Reads initial theme from localStorage or system prefers-color-scheme.
 *    Toggles the 'dark' class on document.documentElement and persists preference to localStorage.
 * 3. Elements in the file:
 *    - `ThemeToggle`: React functional component with Sun and Moon icons.
 * 4. Final output of the file:
 *    An accessible, interactive theme switcher button.
 */

import React, { useEffect, useState } from 'react';
import { Moon, Sun } from 'lucide-react';
import { cn } from '@/lib/cn';

/**
 * Interactive button component to switch between light and dark visual themes.
 *
 * 1. The use of function:
 *    Allows user to switch the entire application's color theme between Light and Dark modes.
 * 2. How does the specific function work:
 *    Initializes state from localStorage('theme') or window.matchMedia('(prefers-color-scheme: dark)').
 *    When clicked, toggles state, syncs the 'dark' class on <html>, and saves to localStorage.
 * 3. Concepts used in the function:
 *    React useState, useEffect, DOM classList manipulation, localStorage persistence, media queries.
 * 4. Places where this function is imported and used:
 *    Imported by TopBar.tsx in the application navigation header.
 * 5. Final output of the function, before and after:
 *    Before click (light mode): <html> has no 'dark' class, button displays Moon icon.
 *    After click (dark mode): <html> has 'dark' class, button displays Sun icon.
 *
 * @param props.className - Optional additional CSS class names.
 * @returns JSX button element.
 */
export const ThemeToggle: React.FC<{ className?: string }> = ({ className }) => {
  const [isDark, setIsDark] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    const saved = localStorage.getItem('theme');
    if (saved) return saved === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDark]);

  return (
    <button
      type="button"
      onClick={() => setIsDark((prev) => !prev)}
      className={cn(
        'p-2 rounded-control border border-border bg-surface text-foreground',
        'hover:bg-surface-muted transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-primary',
        className
      )}
      aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
      title={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
    >
      {isDark ? (
        <Sun className="w-4 h-4 text-warning" />
      ) : (
        <Moon className="w-4 h-4 text-foreground-muted" />
      )}
    </button>
  );
};
