/**
 * @file src/lib/cn.ts
 * 1. What this file does:
 *    Utility function for merging Tailwind CSS classes with clsx and twMerge.
 * 2. Overall design:
 *    Combines conditional class joining (clsx) with intelligent CSS conflict resolution (twMerge)
 *    so dynamic classes do not clash.
 * 3. Elements in the file:
 *    - `cn`: class merge helper function
 * 4. Final output of the file:
 *    A single deduplicated, conflict-free class string.
 */

import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merges conditional class names and resolves conflicting Tailwind CSS utilities.
 *
 * 1. The use of function:
 *    Used across all UI components to combine base styles with custom or conditional class names.
 * 2. How does the specific function work:
 *    Evaluates inputs with `clsx` to flatten arrays, objects, and falsy values into a raw string,
 *    then passes the result through `twMerge` to eliminate conflicting Tailwind properties.
 * 3. Concepts used in the function:
 *    Variadic arguments, AST-based Tailwind conflict resolution, conditional class maps.
 * 4. Places where this function is imported and used:
 *    Imported by AppShell, TopBar, Sidebar, ThemeToggle, buttons, tables, and status cards.
 * 5. Final output of the function, before and after:
 *    Before: `['px-4 py-2', isPrimary && 'px-6 bg-primary']`
 *    After: `'py-2 px-6 bg-primary'` (conflicting `px-4` is safely removed).
 *
 * @param inputs - Variable list of ClassValue items (strings, booleans, arrays, objects).
 * @returns Combined and deduplicated className string.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
