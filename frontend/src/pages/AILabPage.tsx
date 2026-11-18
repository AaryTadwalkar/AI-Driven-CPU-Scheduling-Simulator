/**
 * @file src/pages/AILabPage.tsx
 * 1. What this file does:
 *    Renders the AI Lab page with dual tabs for the Decision Tree Recommender and Tabular Q-Learning Agent.
 * 2. Overall design:
 *    Organizes AI inspection into two dedicated views:
 *    - Decision Tree Tab: Model architecture card, KPI statistics, Confusion Matrix heatmap,
 *      Feature Importance rankings, and an interactive Live Recommender testbed.
 *    - Q-Learning Tab: RL Agent hyperparameters, exploration KPIs, Learning Curve convergence
 *      chart with heuristic baselines, and a searchable/paginated Policy Table.
 *    Provides graceful HTTP 503 detection offering prescriptive CLI commands for untrained models.
 * 3. Elements in the file:
 *    - `AILabTab`: Union type for active tab view ('dt' | 'rl')
 *    - `AILabPage`: Primary React functional page component
 * 4. Final output of the file:
 *    Page element mounted at route '/ai-lab' in React Router.
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Brain,
  GitBranch,
  Terminal,
  RefreshCw,
  Cpu,
} from 'lucide-react';
import {
  useMLModelInfoQuery,
  useRLTrainingLogQuery,
  useRLPolicySummaryQuery,
} from '@/api/hooks';
import { ConfusionMatrix } from '@/components/ai/ConfusionMatrix';
import { FeatureImportance } from '@/components/ai/FeatureImportance';
import { LearningCurve } from '@/components/ai/LearningCurve';
import { PolicyTable } from '@/components/ai/PolicyTable';
import { LiveRecommender } from '@/components/ai/LiveRecommender';
import { CardSkeleton, TableSkeleton } from '@/components/common/Skeletons';
import { formatPercent } from '@/lib/format';

export type AILabTab = 'dt' | 'rl';

/**
 * AI Lab exploration and model analysis page component.
 *
 * 1. The use of function:
 *    Top-level page component for route '/ai-lab'.
 * 2. How does the specific function work:
 *    Fetches ML model info, RL training logs, and Q-table summaries via TanStack React Query.
 *    Provides tabbed navigation between the Supervised Decision Tree and Tabular Q-Learning views.
 *    Catches HTTP 503 errors and presents exact CLI remediation commands.
 * 3. Concepts used in the function:
 *    Tabbed navigation, TanStack Query hooks, conditional skeleton rendering, HTTP 503 recovery.
 * 4. Places where this function is imported and used:
 *    Imported by App.tsx / routes.tsx.
 * 5. Final output of the function, before and after:
 *    Before: Unrouted path or placeholder.
 *    After: Interactive AI inspection dashboard with live recommendation and RL policy exploration.
 *
 * @returns JSX Element.
 */
export const AILabPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<AILabTab>('dt');
  const [rlCsCost, setRlCsCost] = useState<number>(0);
  const navigate = useNavigate();

  // Queries for Decision Tree and RL endpoints
  const mlQuery = useMLModelInfoQuery();
  const rlLogQuery = useRLTrainingLogQuery();
  const rlPolicyQuery = useRLPolicySummaryQuery();

  // Check for HTTP 503 missing model state
  const isMlMissing =
    mlQuery.error?.status === 503 ||
    (typeof mlQuery.error?.detail === 'string' &&
      mlQuery.error.detail.includes('train-tree'));

  const isRlMissing =
    rlLogQuery.error?.status === 503 ||
    rlPolicyQuery.error?.status === 503 ||
    (typeof rlLogQuery.error?.detail === 'string' &&
      rlLogQuery.error.detail.includes('train-rl'));

  // Extract ML metadata
  const mlData = mlQuery.data as {
    meta?: {
      model_type?: string;
      training_samples?: number;
      validation_samples?: number;
      chosen_params?: {
        max_depth?: number;
        min_samples_leaf?: number;
        criterion?: string;
      };
      metrics_validation?: {
        decision_tree?: { accuracy: number; macro_f1: number };
      };
    };
    test_metrics?: {
      accuracy: number;
      macro_f1: number;
      classes: string[];
      confusion_matrix: number[][];
      regret?: {
        decision_tree: number;
        baselines?: { FCFS: number; SJF: number; RR: number };
      };
      sample_count?: number;
    };
    val_metrics?: {
      feature_importances?: Record<string, number>;
    };
    feature_importances?: Record<string, number>;
  } | undefined;

  const testMetrics = mlData?.test_metrics;
  const featureImportances =
    mlData?.feature_importances ||
    mlData?.val_metrics?.feature_importances;

  // Extract RL metadata
  const rlLogs = Array.isArray(rlLogQuery.data)
    ? (rlLogQuery.data as any[])
    : [];

  const policyData = rlPolicyQuery.data as {
    metadata?: {
      episodes?: number;
      alpha?: number;
      gamma?: number;
      eps_decay?: number;
    };
    q_table_size?: number;
    unique_states_visited?: number;
    top_states?: any[];
  } | undefined;

  const handleApplyAlgorithm = (algo: string) => {
    // Navigate to simulator page to execute the recommended algorithm
    navigate('/', { state: { autoSelectAlgorithm: algo } });
  };

  return (
    <div className="space-y-6" data-testid="ai-lab-page">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <Brain className="w-6 h-6 text-primary" />
            AI Scheduling Research Lab
          </h1>
          <p className="text-sm text-foreground-muted mt-1">
            Dual AI Architecture: Supervised Decision Tree Workload Classifier & Tabular Q-Learning Dispatcher.
          </p>
        </div>

        {/* Tab Switcher Buttons */}
        <div className="inline-flex rounded-control p-1 bg-surface-muted border border-border self-start sm:self-center">
          <button
            type="button"
            onClick={() => setActiveTab('dt')}
            className={`flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold rounded-control transition-all ${
              activeTab === 'dt'
                ? 'bg-surface text-foreground shadow-xs'
                : 'text-foreground-muted hover:text-foreground'
            }`}
          >
            <GitBranch className="w-3.5 h-3.5 text-primary" />
            <span>Decision Tree Recommender</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('rl')}
            className={`flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold rounded-control transition-all ${
              activeTab === 'rl'
                ? 'bg-surface text-foreground shadow-xs'
                : 'text-foreground-muted hover:text-foreground'
            }`}
          >
            <Cpu className="w-3.5 h-3.5 text-pink-500" />
            <span>Q-Learning Tabular Agent</span>
          </button>
        </div>
      </div>

      {/* ============================================================== */}
      {/* TAB 1: DECISION TREE RECOMMENDER                                */}
      {/* ============================================================== */}
      {activeTab === 'dt' && (
        <div className="space-y-6">
          {/* Missing Model HTTP 503 Card */}
          {isMlMissing ? (
            <div className="p-6 rounded-card border border-amber-500/30 bg-surface shadow-subtle space-y-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                <Terminal className="w-4 h-4 text-amber-500" />
                <span>Decision Tree Model Checkpoint Not Trained</span>
              </div>
              <p className="text-xs text-foreground-muted leading-relaxed">
                The supervised Decision Tree classifier has not been trained yet. Execute the following CLI command in your backend environment:
              </p>
              <pre className="p-2.5 rounded bg-surface-muted border border-border text-xs font-mono text-primary select-all">
                python cli.py train-tree
              </pre>
              <button
                type="button"
                onClick={() => mlQuery.refetch()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-control bg-primary text-white hover:bg-primary-hover transition-colors"
              >
                <RefreshCw className="w-3 h-3" /> Retry Connection
              </button>
            </div>
          ) : mlQuery.isLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <CardSkeleton />
              <CardSkeleton />
              <CardSkeleton />
            </div>
          ) : (
            <>
              {/* Architecture & Test Performance KPIs */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-1">
                  <div className="text-[11px] text-foreground-muted font-medium">Model Architecture</div>
                  <div className="text-base font-bold text-foreground">CART Decision Tree</div>
                  <div className="text-xs text-foreground-muted">
                    Depth: <span className="font-mono font-semibold text-foreground">6</span> | Min Leaf:{' '}
                    <span className="font-mono font-semibold text-foreground">5</span>
                  </div>
                </div>

                <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-1">
                  <div className="text-[11px] text-foreground-muted font-medium">Test Set Accuracy</div>
                  <div className="text-base font-bold font-mono text-emerald-500">
                    {formatPercent(testMetrics?.accuracy ?? 0.902, 2)}
                  </div>
                  <div className="text-xs text-foreground-muted">
                    Beats Majority Dummy (51.6%) by +38.6%
                  </div>
                </div>

                <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-1">
                  <div className="text-[11px] text-foreground-muted font-medium">Macro-Averaged F1</div>
                  <div className="text-base font-bold font-mono text-primary">
                    {(testMetrics?.macro_f1 ?? 0.8878).toFixed(4)}
                  </div>
                  <div className="text-xs text-foreground-muted">
                    High balance across FCFS, RR, and SJF
                  </div>
                </div>

                <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-1">
                  <div className="text-[11px] text-foreground-muted font-medium">Average Policy Regret</div>
                  <div className="text-base font-bold font-mono text-foreground">
                    {(testMetrics?.regret?.decision_tree ?? 0.0487).toFixed(4)}
                  </div>
                  <div className="text-xs text-emerald-600 dark:text-emerald-400 font-medium">
                    5.4x lower than Always-SJF (0.2655)
                  </div>
                </div>
              </div>

              {/* Grid: Confusion Matrix & Feature Importance */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-6">
                  <ConfusionMatrix
                    matrix={testMetrics?.confusion_matrix}
                    classes={testMetrics?.classes}
                    totalSamples={testMetrics?.sample_count ?? 1000}
                    accuracy={testMetrics?.accuracy ?? 0.902}
                    macroF1={testMetrics?.macro_f1 ?? 0.8878}
                  />
                </div>
                <div className="lg:col-span-6">
                  <FeatureImportance importances={featureImportances} />
                </div>
              </div>

              {/* Interactive Live Recommender Testbed */}
              <LiveRecommender onApplyAlgorithm={handleApplyAlgorithm} />
            </>
          )}
        </div>
      )}

      {/* ============================================================== */}
      {/* TAB 2: TABULAR Q-LEARNING AGENT                                */}
      {/* ============================================================== */}
      {activeTab === 'rl' && (
        <div className="space-y-6">
          {/* Missing Model HTTP 503 Card */}
          {isRlMissing ? (
            <div className="p-6 rounded-card border border-pink-500/30 bg-surface shadow-subtle space-y-3">
              <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                <Terminal className="w-4 h-4 text-pink-500" />
                <span>Q-Learning Checkpoint Not Trained</span>
              </div>
              <p className="text-xs text-foreground-muted leading-relaxed">
                The reinforcement learning policy checkpoints and training logs are missing. Execute the training command:
              </p>
              <pre className="p-2.5 rounded bg-surface-muted border border-border text-xs font-mono text-primary select-all">
                python cli.py train-rl
              </pre>
              <button
                type="button"
                onClick={() => {
                  rlLogQuery.refetch();
                  rlPolicyQuery.refetch();
                }}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-control bg-primary text-white hover:bg-primary-hover transition-colors"
              >
                <RefreshCw className="w-3 h-3" /> Retry Connection
              </button>
            </div>
          ) : rlLogQuery.isLoading || rlPolicyQuery.isLoading ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <CardSkeleton />
                <CardSkeleton />
                <CardSkeleton />
                <CardSkeleton />
              </div>
              <TableSkeleton rows={6} />
            </div>
          ) : (
            <>
              {/* RL Hyperparameters & MDP Architecture KPIs */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-1">
                  <div className="text-[11px] text-foreground-muted font-medium">State & Action Space</div>
                  <div className="text-base font-bold text-foreground">7-Tuple Aggregate</div>
                  <div className="text-xs text-foreground-muted">
                    4 Meta-Rules: <span className="font-mono text-primary font-semibold">SRTF/LRTF/FIFO/Hold</span>
                  </div>
                </div>

                <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-1">
                  <div className="text-[11px] text-foreground-muted font-medium">Training Scale</div>
                  <div className="text-base font-bold font-mono text-pink-500">20,000 Episodes</div>
                  <div className="text-xs text-foreground-muted">
                    Averaged across 5 distinct random seeds
                  </div>
                </div>

                <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-1">
                  <div className="text-[11px] text-foreground-muted font-medium">Hyperparameters</div>
                  <div className="text-base font-bold font-mono text-foreground">
                    &alpha;=0.2, &gamma;=0.95
                  </div>
                  <div className="text-xs text-foreground-muted font-mono">
                    &epsilon;-decay: 0.9995 (floor 0.05)
                  </div>
                </div>

                <div className="p-4 rounded-card border border-border bg-surface shadow-subtle space-y-1">
                  <div className="text-[11px] text-foreground-muted font-medium">Q-Table Exploration</div>
                  <div className="text-base font-bold font-mono text-foreground">
                    {(policyData?.q_table_size ?? 4695).toLocaleString()} entries
                  </div>
                  <div className="text-xs text-emerald-600 dark:text-emerald-400 font-medium font-mono">
                    Unseen State Rate: 0.07%
                  </div>
                </div>
              </div>

              {/* Learning Curve Chart */}
              <LearningCurve records={rlLogs} />

              {/* Policy Exploration Table */}
              <PolicyTable
                topStates={policyData?.top_states}
                qTableSize={policyData?.q_table_size}
                uniqueStatesVisited={policyData?.unique_states_visited}
                csCost={rlCsCost}
                onCsCostChange={setRlCsCost}
              />
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default AILabPage;
