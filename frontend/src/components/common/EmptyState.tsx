/**
 * @file src/components/common/EmptyState.tsx
 * 1. What this file does:
 *    Renders a unified empty state placeholder when data, workloads, or simulation runs
 *    are absent.
 * 2. Overall design:
 *    Displays an icon, descriptive heading, contextual message, and optional action button
 *    inside a dashed border card adhering to the calm observability design language.
 * 3. Elements in the file:
 *    - `EmptyStateProps`: Configuration interface for icons, titles, messages, and actions
 *    - `EmptyState`: React functional component
 * 4. Final output of the file:
 *    Exported empty state component reused across Simulator, Compare, and AI Lab views.
 */

import React from 'react';
import { Inbox } from 'lucide-react';

export interface EmptyStateProps {
  title?: string;
  description?: string;
  icon?: React.ReactNode;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

/**
 * Standard empty state card component.
 *
 * 1. The use of function:
 *    Informs the user that no data is currently available and suggests next steps.
 * 2. How does the specific function work:
 *    Renders a centered icon, title, description, and optional call-to-action button.
 * 3. Concepts used in the function:
 *    Accessibility with clear hierarchy, graceful degradation, calm dashboard UX.
 * 4. Places where this function is imported and used:
 *    Imported by ComparePage.tsx, SimulatorPage.tsx, and ProcessTable.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Blank space or confusing absence of data.
 *    After: Structured UI explaining why content is empty with recovery guidance.
 *
 * @param props - Empty state parameters.
 * @returns Rendered JSX element.
 */
export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No Data Available',
  description = 'There is currently no information to display.',
  icon,
  action,
  className = '',
}) => {
  return (
    <div
      className={`p-8 rounded-card border border-dashed border-border bg-surface text-center flex flex-col items-center justify-center space-y-3 ${className}`}
    >
      <div className="p-3 rounded-full bg-surface-muted text-foreground-muted">
        {icon || <Inbox className="w-6 h-6 text-foreground-muted" />}
      </div>
      <div className="space-y-1 max-w-md">
        <h4 className="text-sm font-semibold text-foreground">{title}</h4>
        <p className="text-xs text-foreground-muted leading-relaxed">{description}</p>
      </div>
      {action && (
        <button
          type="button"
          onClick={action.onClick}
          className="mt-2 px-3 py-1.5 rounded-control text-xs font-medium bg-primary text-white hover:bg-primary-hover transition-colors shadow-sm"
        >
          {action.label}
        </button>
      )}
    </div>
  );
};

export default EmptyState;
