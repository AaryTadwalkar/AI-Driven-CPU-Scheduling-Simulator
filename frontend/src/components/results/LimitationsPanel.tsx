/**
 * @file src/components/results/LimitationsPanel.tsx
 * 1. What this file does:
 *    Renders an expandable accordion panel articulating honest technical limitations and trade-offs.
 * 2. Overall design:
 *    Organizes scientific constraints from docs/REPORT_NOTES.md into 4 interactive cards covering
 *    discrete tick granularity, tabular state abstraction, offline feature extraction, and single-core CPU semantics.
 * 3. Elements in the file:
 *    - `LimitationItem`: Interface defining limitation title, icon, summary, and academic details
 *    - `LIMITATIONS_LIST`: Curated list of the 4 core technical boundaries
 *    - `LimitationsPanelProps`: Component props and customization options
 *    - `LimitationsPanel`: Primary React functional component
 * 4. Final output of the file:
 *    Interactive technical limitations accordion card consumed by ResultsPage.
 */

import React, { useState } from 'react';
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Cpu,
  Clock,
  Layers,
  FileCode,
  BookOpen,
} from 'lucide-react';

export interface LimitationItem {
  id: string;
  title: string;
  badge: string;
  icon: React.ElementType;
  summary: string;
  details: string;
  mitigation: string;
}

export const LIMITATIONS_LIST: LimitationItem[] = [
  {
    id: 'discrete_granularity',
    title: '1. Discrete Integer Tick Granularity [t, t+1)',
    badge: 'Engine Semantics',
    icon: Clock,
    summary:
      'The scheduling engine operates on discrete unit steps where execution occurs strictly over [t, t+1).',
    details:
      'Continuous hardware timer interrupts, sub-microsecond preemption slices, and asynchronous I/O completion pulses are discretized into integer boundaries. When multiple processes arrive at identical tick t, deterministic tie-breaking (arrival order, then lexicographical PID) must resolve ties, which may create minor quantization artifacts not present in real interrupt-driven kernel dispatchers.',
    mitigation:
      'Matches standard discrete OS scheduling pedagogy (Silberschatz et al.) while guaranteeing 100% reproducible bitwise simulation replay across platforms.',
  },
  {
    id: 'tabular_state_abstraction',
    title: '2. Tabular Q-Learning State Abstraction Boundaries',
    badge: 'RL Architecture',
    icon: Layers,
    summary:
      'The RL agent compresses arbitrary ready queue shapes into a permutation-invariant 7-tuple state.',
    details:
      'By digitizing queue length (4 buckets), min burst (5), max burst (5), max wait (4), and binary flags into 3,200 aggregate states, the agent eliminates combinatorial state explosion and avoids seeing new PIDs. However, aggregating distinct queue distributions into shared discrete buckets means the agent cannot distinguish subtle intra-bucket burst variations.',
    mitigation:
      'Achieves an unseen state rate of only 0.07% on held-out validation workloads while maintaining strict convergence without deep neural network instabilities.',
  },
  {
    id: 'offline_feature_extraction',
    title: '3. Offline Feature Extraction Assumption for Decision Tree',
    badge: 'Supervised ML',
    icon: FileCode,
    summary:
      'The Decision Tree recommender calculates 12 structural features assuming the full workload profile is known a priori.',
    details:
      'Features such as arrival span, coefficient of variation in burst, and load ratio require knowledge of all process arrivals and bursts before scheduling starts. While appropriate for batch scheduling systems, supercomputing job queues, and container workload dispatchers, true online interactive OS kernels discover process arrival times and bursts dynamically at runtime.',
    mitigation:
      'Serves as a macro meta-policy selector for batch and container workloads; paired with the online per-tick Q-learning scheduler for interactive environments.',
  },
  {
    id: 'single_core_semantics',
    title: '4. Single-Core Virtual CPU vs Multi-Core SMP Preemption',
    badge: 'Hardware Reality',
    icon: Cpu,
    summary:
      'Execution assumes a single uniprocessor CPU without hardware cache thrashing or NUMA affinity.',
    details:
      'Context switches incur an integer penalty of 0 or 1 tick. In physical symmetric multiprocessing (SMP) hardware, context switches incur L1/L2 cache line invalidation, Translation Lookaside Buffer (TLB) flushes, inter-processor interrupts (IPI), and memory bus contention that non-linearly depend on working set sizes.',
    mitigation:
      'Parameterizing context-switch overhead (cs_cost = 0 vs 1) cleanly models the preemption penalty curve and demonstrates how Q-learning reduces switches by 46%.',
  },
];

export interface LimitationsPanelProps {
  className?: string;
}

/**
 * Honest Technical Limitations & Trade-offs component.
 *
 * 1. The use of function:
 *    Documents system boundaries, architectural trade-offs, and academic limitations for report defense.
 * 2. How does the specific function work:
 *    Maintains expanded/collapsed state for each limitation card, rendering detailed explanations
 *    and mitigations conforming to Section 8 of the project report.
 * 3. Concepts used in the function:
 *    Accordion disclosure pattern, scientific integrity, academic self-audit.
 * 4. Places where this function is imported and used:
 *    Imported by ResultsPage.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Static limitations notes buried in markdown.
 *    After: Interactive disclosure accordion with technical summaries and mitigations.
 *
 * @param props - Component parameters and classes.
 * @returns JSX Element.
 */
export const LimitationsPanel: React.FC<LimitationsPanelProps> = ({
  className = '',
}) => {
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>({
    discrete_granularity: true,
    tabular_state_abstraction: false,
    offline_feature_extraction: false,
    single_core_semantics: false,
  });

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  const handleExpandAll = () => {
    setExpandedIds({
      discrete_granularity: true,
      tabular_state_abstraction: true,
      offline_feature_extraction: true,
      single_core_semantics: true,
    });
  };

  const handleCollapseAll = () => {
    setExpandedIds({
      discrete_granularity: false,
      tabular_state_abstraction: false,
      offline_feature_extraction: false,
      single_core_semantics: false,
    });
  };

  return (
    <div
      className={`p-5 rounded-card border border-border bg-surface shadow-subtle flex flex-col space-y-4 ${className}`}
      data-testid="limitations-panel"
    >
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-border">
        <div>
          <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-500" />
            Engineering Trade-Offs & Honest Technical Limitations
          </h3>
          <p className="text-xs text-foreground-muted mt-0.5">
            Transparent scientific critique conforming to Section 1.4 of the Implementation Plan and Report Notes.
          </p>
        </div>

        <div className="flex items-center gap-2 self-start sm:self-center">
          <button
            type="button"
            onClick={handleExpandAll}
            className="px-2 py-1 text-xs rounded border border-border bg-surface hover:bg-surface-muted text-foreground transition-colors"
          >
            Expand All
          </button>
          <button
            type="button"
            onClick={handleCollapseAll}
            className="px-2 py-1 text-xs rounded border border-border bg-surface hover:bg-surface-muted text-foreground-muted hover:text-foreground transition-colors"
          >
            Collapse All
          </button>
        </div>
      </div>

      {/* Accordion Cards */}
      <div className="space-y-3">
        {LIMITATIONS_LIST.map((item) => {
          const isExpanded = !!expandedIds[item.id];
          const Icon = item.icon;

          return (
            <div
              key={item.id}
              className="rounded-control border border-border bg-surface-muted/20 overflow-hidden transition-all"
            >
              <button
                type="button"
                onClick={() => toggleExpand(item.id)}
                className="w-full p-3.5 text-left flex items-start sm:items-center justify-between gap-3 hover:bg-surface-muted/40 transition-colors"
                aria-expanded={isExpanded}
              >
                <div className="flex items-center gap-2.5">
                  <div className="p-1.5 rounded bg-primary/10 text-primary shrink-0">
                    <Icon className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-semibold text-foreground">
                        {item.title}
                      </span>
                      <span className="px-1.5 py-0.2 rounded text-[9px] font-semibold bg-surface border border-border text-foreground-muted">
                        {item.badge}
                      </span>
                    </div>
                    <p className="text-xs text-foreground-muted mt-0.5">
                      {item.summary}
                    </p>
                  </div>
                </div>

                <div className="text-foreground-muted shrink-0 mt-1 sm:mt-0">
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </div>
              </button>

              {isExpanded && (
                <div className="p-4 pt-1 border-t border-border/50 text-xs space-y-3 bg-surface/40">
                  <div className="space-y-1">
                    <div className="font-semibold text-foreground">
                      Technical Boundary & Physical Reality:
                    </div>
                    <p className="text-foreground-muted leading-relaxed">
                      {item.details}
                    </p>
                  </div>

                  <div className="p-2.5 rounded bg-emerald-500/5 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 space-y-0.5">
                    <div className="font-semibold text-[11px]">
                      Design Justification & Scientific Mitigation:
                    </div>
                    <p className="text-xs leading-relaxed">{item.mitigation}</p>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="pt-2 border-t border-border flex items-center justify-between text-xs text-foreground-muted">
        <span className="flex items-center gap-1.5">
          <BookOpen className="w-3.5 h-3.5 text-primary" />
          <span>Full citations & formal proofs detailed in REPORT_NOTES.md</span>
        </span>
        <span className="font-mono text-[11px]">Section 8 Documentation</span>
      </div>
    </div>
  );
};

export default LimitationsPanel;
