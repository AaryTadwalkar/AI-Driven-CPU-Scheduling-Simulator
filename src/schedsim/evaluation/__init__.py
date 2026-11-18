"""
1. What this file does:
   Initializes the evaluation package for experimental benchmarking, statistical analysis,
   and publication-ready visualization generation.
2. Overall design:
   Aggregates core evaluation submodules into a unified public package interface:
   - run_experiments: Multi-policy execution across held-out splits.
   - stats: Bootstrap confidence intervals, win rates, fairness, and test metrics.
   - plots: Multi-panel bar charts, Gantt timelines, Decision Tree diagnostics, and heatmaps.
3. Elements in the file:
   - Experiment runners: EVAL_POLICIES, run_evaluation_experiments, run_workload_all_policies.
   - Statistical estimators: bootstrap_ci, compute_summary_table, compute_win_rates,
     compute_scenario_breakdown, evaluate_decision_tree_test_split, export_summary_tables.
   - Visualization functions: generate_all_plots, plot_comparison_bars, plot_gantt_fixtures,
     plot_dt_confusion_matrix, plot_dt_feature_importance, plot_fairness_starvation,
     plot_scenario_breakdown.
4. Final output of the file:
   Clean top-level import namespace for schedsim.evaluation.
"""

from schedsim.evaluation.plots import (
    generate_all_plots,
    plot_comparison_bars,
    plot_dt_confusion_matrix,
    plot_dt_feature_importance,
    plot_fairness_starvation,
    plot_gantt_fixtures,
    plot_scenario_breakdown,
)
from schedsim.evaluation.run_experiments import (
    EVAL_POLICIES,
    run_evaluation_experiments,
    run_workload_all_policies,
)
from schedsim.evaluation.stats import (
    bootstrap_ci,
    compute_scenario_breakdown,
    compute_summary_table,
    compute_win_rates,
    evaluate_decision_tree_test_split,
    export_summary_tables,
)

__all__ = [
    "EVAL_POLICIES",
    "bootstrap_ci",
    "compute_scenario_breakdown",
    "compute_summary_table",
    "compute_win_rates",
    "evaluate_decision_tree_test_split",
    "export_summary_tables",
    "generate_all_plots",
    "plot_comparison_bars",
    "plot_dt_confusion_matrix",
    "plot_dt_feature_importance",
    "plot_fairness_starvation",
    "plot_gantt_fixtures",
    "plot_scenario_breakdown",
    "run_evaluation_experiments",
    "run_workload_all_policies",
]
