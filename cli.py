"""
1. What this file does:
   Provides the Command Line Interface (CLI) entry point for the CPU Scheduling Simulator,
   orchestrating simulation runs, policy comparisons, workload generation, model training
   (Decision Tree and Q-learning), evaluation benchmarks, and ASCII Gantt chart rendering.
2. Overall design:
   Built on Python's standard `argparse` library using subcommands. Each subcommand represents
   a key project capability (simulate, compare, build-workloads / build-dataset, train-tree,
   train-rl, evaluate), routing CLI arguments to the respective domain, ML, or RL modules.
   Formats execution metrics into structured tables and multi-row wrapped ASCII Gantt charts.
3. Elements in the file:
   - render_ascii_gantt: Function rendering discrete timeline history into an ASCII Gantt chart.
   - format_results: Function composing process tables, summary metrics, and Gantt visualization.
   - handle_simulate: Handler executing policy simulation on preset workloads and printing output.
   - handle_build_workloads: Handler generating synthetic workloads and saving to CSV.
   - handle_build_dataset: Handler generating labeled supervised dataset and saving to CSV.
   - handle_build_pid_data: Handler generating process-level behavioral datasets (Stage 14).
   - handle_train_pid_gbdt: Handler training GBDT candidate rankers for S0/S1 (Stage 15).
   - handle_train_pid_nn: Handler training neural candidate scorers for S0/S1 (Stage 16).
   - handle_bench_latency: Handler benchmarking policy latencies and cost profile (Stage 17).
   - handle_train_tree: Handler fitting Decision Tree recommender and evaluating baselines.
   - handle_recommend: Handler running inference to recommend policy for a workload.
   - handle_train_rl: Handler executing tabular Q-learning training and hyperparameter search.
   - handle_evaluate: Handler executing evaluation across held-out splits.
   - handle_compare_all: Handler executing Stage 18 final comparison across all 11 policies.
   - build_parser: Function constructing the CLI argument parser and defining subcommands.
   - main: Main execution function parsing CLI arguments and dispatching requested subcommands.
4. Final output of the file:
   Parses command-line invocations, executes simulations or ML/RL tasks, and displays results.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import config
from schedsim.domain.engine import SimResult, run
from schedsim.domain.timeline import Timeline, TimelineSegment
from schedsim.evaluation import (
    compute_scenario_breakdown,
    compute_summary_table,
    evaluate_decision_tree_test_split,
    export_summary_tables,
    generate_all_plots,
    run_evaluation_experiments,
    run_final_comparison,
)
from schedsim.ml.dataset_builder import build_dataset, save_dataset_to_csv
from schedsim.ml.evaluate_tree import evaluate_decision_tree, recommend
from schedsim.ml.train_tree import train_decision_tree
from schedsim.pidsel.dataset import (
    generate_and_save_pid_datasets,
    generate_dataset_report,
)
from schedsim.pidsel.gbdt_ranker import (
    evaluate_gbdt_on_validation,
    train_gbdt,
)
from schedsim.policies.registry import get_policy
from schedsim.rl.evaluate import plot_action_distribution, plot_learning_curves
from schedsim.rl.train import run_hparam_search, train_final_agents, train_q_learning
from schedsim.workloads.io import load_workloads_from_csv, save_workloads_to_csv
from schedsim.workloads.presets import get_preset
from schedsim.workloads.splits import ALL_SPLITS, generate_split_workloads


def render_ascii_gantt(timeline: Timeline, max_width: int = 80) -> str:
    """
    1. The use of function:
       Renders discrete execution intervals from a Timeline into a chronological ASCII Gantt chart.
    2. How does the specific function work:
       Retrieves merged segments via `timeline.to_segments()`, calculates proportionate visual
       block widths bounded by min/max constraints and text bounds, wraps segments into lines
       respecting `max_width`, formats ASCII box borders and centered labels, and generates aligned
       tick markers below each boundary delimiter.
    3. Concepts used in the function:
       Interval visualization, ASCII graphic formatting, responsive terminal wrapping, alignment.
    4. Places where this function is imported and used:
       Called by `format_results` in `cli.py` and directly importable for tests and CLI reporting.
    5. Final output of the function, before and after:
       Before: Timeline object containing discrete execution history.
       After: Multi-line string depicting the chronological ASCII Gantt chart with time ticks.
    """
    if not isinstance(timeline, Timeline):
        raise TypeError(f"timeline must be a Timeline instance, got {type(timeline).__name__}.")

    segments = timeline.to_segments()
    if not segments:
        return "(empty timeline)"

    effective_max_width = max(max_width, 10)

    # Compute display widths and pack segments into rows
    row_segments: list[list[tuple[TimelineSegment, int]]] = []
    current_row: list[tuple[TimelineSegment, int]] = []
    current_width = 1  # Leading '+'

    for s in segments:
        lbl = s.label
        min_w = max(len(lbl) + 2, len(str(s.start)) + 1, len(str(s.end)) + 1, 4)
        target_w = max(min_w, min(min_w + (s.duration - 1) * 2, 24))
        if effective_max_width > 4:
            target_w = min(target_w, effective_max_width - 2)
            target_w = max(target_w, len(lbl) + 2)

        seg_cols = target_w + 1  # Block chars plus trailing '+'
        if current_row and (current_width + seg_cols > effective_max_width):
            row_segments.append(current_row)
            current_row = []
            current_width = 1

        current_row.append((s, target_w))
        current_width += seg_cols

    if current_row:
        row_segments.append(current_row)

    lines: list[str] = []
    for row_idx, row in enumerate(row_segments):
        top_line = "+" + "+".join("-" * w for _, w in row) + "+"
        label_line = "|" + "|".join(s.label.center(w) for s, w in row) + "|"
        bot_line = top_line

        # Calculate exact boundary column positions for tick alignment
        col_positions = [0]
        cur_pos = 0
        for _, w in row:
            cur_pos += w + 1
            col_positions.append(cur_pos)

        total_line_len = col_positions[-1] + len(str(row[-1][0].end)) + 1
        tick_chars = [" "] * total_line_len
        for i in range(len(row) + 1):
            tick_val = str(row[0][0].start if i == 0 else row[i - 1][0].end)
            pos = col_positions[i]
            for c_idx, ch in enumerate(tick_val):
                tick_chars[pos + c_idx] = ch

        tick_line = "".join(tick_chars).rstrip()

        if row_idx > 0:
            lines.append("")
        lines.extend([top_line, label_line, bot_line, tick_line])

    return "\n".join(lines)


def format_results(
    result: SimResult,
    policy_name: str,
    preset_name: str,
    cs_cost: int,
) -> str:
    """
    1. The use of function:
       Formats comprehensive simulation results including process metrics, aggregate stats, and
       Gantt chart.
    2. How does the specific function work:
       Extracts `process_metrics`, `simulation_metrics`, and `timeline` from `result`. Constructs a
       formatted process table (PID, Arrival, Burst, Completion, Turnaround, Waiting, Response),
       an aggregate metrics block (Avg Waiting, Avg Turnaround, Avg Response, Max Waiting, Makespan,
       CPU Utilization, Throughput, Context Switches, Jain's Fairness Index), and embeds the ASCII
       Gantt chart produced by `render_ascii_gantt`.
    3. Concepts used in the function:
       Tabular text formatting, string templating, metric reporting, visual composition.
    4. Places where this function is imported and used:
       Invoked by `handle_simulate` in `cli.py` and test suites evaluating CLI output fidelity.
    5. Final output of the function, before and after:
       Before: Raw SimResult object, policy name, preset name, and context switch cost.
       After: Multi-line terminal report string with tables, metrics, and Gantt chart.
    """
    if not isinstance(result, SimResult):
        raise TypeError(f"result must be a SimResult instance, got {type(result).__name__}.")

    sm = result.simulation_metrics
    sep = "=" * 80
    subsep = "-" * 80

    header_cols = (
        f"{'PID':<8}{'Arrival':>8}{'Burst':>7}"
        f"{'Completion':>12}{'Turnaround':>12}{'Waiting':>9}{'Response':>10}"
    )

    lines = [
        sep,
        (
            f"Simulation Results: Policy={policy_name}, "
            f"Preset={preset_name}, Context Switch Cost={cs_cost}"
        ),
        sep,
        "",
        "Process Execution Metrics:",
        subsep,
        header_cols,
        subsep,
    ]

    for p in result.process_metrics:
        row_str = (
            f"{p.pid:<8}{p.arrival:>8}{p.burst:>7}"
            f"{p.completion:>12}{p.turnaround:>12}{p.waiting:>9}{p.response:>10}"
        )
        lines.append(row_str)

    lines.extend(
        [
            subsep,
            "",
            "Aggregate Simulation Metrics:",
            subsep,
            f"Avg Waiting:               {sm.avg_waiting:.2f} ticks",
            f"Avg Turnaround:            {sm.avg_turnaround:.2f} ticks",
            f"Avg Response:              {sm.avg_response:.2f} ticks",
            f"Max Waiting:               {sm.max_waiting} ticks",
            f"Makespan:                  {sm.makespan} ticks",
            (
                f"CPU Utilization:           {sm.cpu_utilization:.4f} "
                f"({sm.cpu_utilization * 100:.2f}%)"
            ),
            f"Throughput:                {sm.throughput:.4f} processes/tick",
            f"Context Switches:          {sm.context_switches}",
            f"Jain's Fairness Index:     {sm.jain_fairness:.4f}",
            subsep,
            "",
            "Execution Timeline (Gantt Chart):",
            subsep,
            render_ascii_gantt(result.timeline, max_width=80),
            sep,
        ]
    )

    return "\n".join(lines)


def handle_simulate(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes the `simulate` CLI subcommand given parsed command-line arguments.
    2. How does the specific function work:
       Validates CLI arguments, retrieves requested policy via `get_policy(policy, ...)` with
       optional quantum configuration for Round Robin, loads workload via `get_preset(preset)`,
       executes the simulation engine via `run(policy, workload, cs_cost)`, formats output via
       `format_results`, prints the report to stdout, and handles errors with non-zero exit code.
    3. Concepts used in the function:
       Subcommand handler, exception handling, policy and workload resolution, simulation run.
    4. Places where this function is imported and used:
       Dispatched by `main()` when `command == 'simulate'` and tested in `test_cli_simulate.py`.
    5. Final output of the function, before and after:
       Before: Parsed argparse.Namespace with simulation options.
       After: Printed simulation report, returning 0 on success or non-zero (1) on error.
    """
    policy_arg = getattr(args, "policy", None)
    if not policy_arg:
        print("Simulation error: --policy is required for simulate.", file=sys.stderr)
        return 1

    preset_arg = getattr(args, "preset", "1")
    quantum_arg = getattr(args, "quantum", 2)
    cs_cost_arg = getattr(args, "cs_cost", 0)

    try:
        policy_name = policy_arg.strip()
        policy_kwargs = {}
        if policy_name.lower() in ("rr", "roundrobin"):
            policy_kwargs["quantum"] = quantum_arg

        policy = get_policy(policy_name, **policy_kwargs)
        workload = get_preset(preset_arg)
        result = run(policy, workload, cs_cost=cs_cost_arg)
    except (KeyError, ValueError, TypeError) as exc:
        print(f"Simulation error: {exc}", file=sys.stderr)
        return 1

    report = format_results(
        result=result,
        policy_name=policy_name,
        preset_name=preset_arg,
        cs_cost=cs_cost_arg,
    )
    print(report)
    return 0


def handle_build_workloads(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes the `build-workloads` (or `build-dataset`) CLI subcommand to generate synthetic
       workloads and serialize them to a CSV file.
    2. How does the specific function work:
       Validates command-line arguments (n, output, split, seed_offset), generates balanced
       workloads partitioned across the requested split via `generate_split_workloads()`, saves the
       resulting records using `save_workloads_to_csv()`, calculates elapsed wall-clock duration,
       and prints a detailed summary report to stdout. If generation or file I/O fails, writes
       error diagnostics to stderr and returns non-zero code 1.
    3. Concepts used in the function:
       Subcommand handling, dataset synthesis, performance profiling with time.perf_counter,
       file persistence, exception handling and exit code propagation.
    4. Places where this function is imported and used:
       Dispatched by `main()` in `cli.py` when `command` is `build-workloads` or `build-dataset`,
       and invoked directly by unit tests in `tests/test_cli_build_workloads.py`.
    5. Final output of the function, before and after:
       Before: Parsed argparse.Namespace containing generation configuration parameters.
       After: CSV file written to disk, formatted report printed to stdout, returns 0 on success
       or 1 on error.
    """
    n_arg = getattr(args, "n", 6000)
    output_arg = getattr(args, "output", "data/workloads.csv")
    split_arg = getattr(args, "split", "train")
    seed_offset_arg = getattr(args, "seed_offset", 0)

    if n_arg <= 0:
        print(f"Dataset generation error: --n must be >= 1, got {n_arg}.", file=sys.stderr)
        return 1

    if seed_offset_arg < 0:
        print(
            f"Dataset generation error: --seed-offset must be >= 0, got {seed_offset_arg}.",
            file=sys.stderr,
        )
        return 1

    try:
        t0 = time.perf_counter()
        workloads = generate_split_workloads(
            split=split_arg,
            n_workloads=n_arg,
            seed_offset=seed_offset_arg,
        )
        save_workloads_to_csv(workloads, path=output_arg)
        elapsed = time.perf_counter() - t0
    except (ValueError, TypeError, OSError) as exc:
        print(f"Dataset generation error: {exc}", file=sys.stderr)
        return 1

    total_procs = sum(len(w.processes) for w in workloads)
    start_seed = workloads[0].seed if workloads else 0
    end_seed = workloads[-1].seed if workloads else 0
    rate = len(workloads) / elapsed if elapsed > 0 else 0.0

    sep = "=" * 80
    lines = [
        sep,
        "Workload Dataset Generation Report",
        sep,
        f"Split:                     {split_arg}",
        f"Workloads Generated:       {len(workloads)}",
        f"Total Processes:           {total_procs}",
        f"Seed Range:                {start_seed} .. {end_seed}",
        f"Output Destination:        {output_arg}",
        f"Execution Time:            {elapsed:.2f}s ({rate:.1f} workloads/s)",
        sep,
    ]
    print("\n".join(lines))
    return 0


def handle_build_dataset(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes the `build-dataset` CLI subcommand to generate the supervised labeled ML dataset.
    2. How does the specific function work:
       Loads raw workloads from disk (or synthesizes balanced train/val workloads if missing),
       computes the 12 domain features and multi-metric composite labels, persists the DataFrame
       to CSV, and prints a formatted summary table of class distributions and ambiguity rates.
    3. Concepts used in the function:
       Supervised dataset synthesis, composite labeling, feature extraction, CLI reporting.
    4. Places where this function is imported and used:
       Dispatched by `main()` in `cli.py` when `command == 'build-dataset'`.
    5. Final output of the function, before and after:
       Before: Parsed CLI options for input workloads and output dataset CSV.
       After: Labeled dataset CSV saved to disk, summary table printed, returns 0 on success.
    """
    input_arg = getattr(args, "input", "data/workloads.csv")
    output_arg = getattr(args, "output", "data/scheduling_dataset.csv")
    n_arg = getattr(args, "n", None)
    cs_cost = getattr(args, "cs_cost", 1)
    quantum = getattr(args, "quantum", 2)

    try:
        input_path = Path(input_arg)
        if input_path.exists():
            workloads = load_workloads_from_csv(input_path)
            if n_arg is not None and n_arg > 0:
                workloads = workloads[:n_arg]
        else:
            n_train = n_arg if (n_arg is not None and n_arg > 0) else 6000
            workloads = generate_split_workloads(split="train", n_workloads=n_train)
            workloads.extend(generate_split_workloads(split="val", n_workloads=min(500, n_train)))

        t0 = time.perf_counter()
        df = build_dataset(workloads, cs_cost=cs_cost, quantum=quantum)
        save_dataset_to_csv(df, path=output_arg)
        elapsed = time.perf_counter() - t0
    except (ValueError, TypeError, OSError) as exc:
        print(f"Dataset building error: {exc}", file=sys.stderr)
        return 1

    total_samples = len(df)
    counts = df["label"].value_counts().to_dict()
    ambig_count = int(df["is_ambiguous"].sum()) if "is_ambiguous" in df.columns else 0
    ambig_pct = (ambig_count / total_samples * 100.0) if total_samples > 0 else 0.0

    sep = "=" * 80
    lines = [
        sep,
        "Supervised Scheduling Dataset Report",
        sep,
        f"Samples Generated:         {total_samples}",
        f"Output Destination:        {output_arg}",
        f"Execution Time:            {elapsed:.2f}s",
        "-" * 80,
        "Class Label Distribution:",
    ]
    for lbl in ["FCFS", "SJF", "RR"]:
        c = counts.get(lbl, 0)
        pct = (c / total_samples * 100.0) if total_samples > 0 else 0.0
        lines.append(f"  {lbl:<6} {c:>6} ({pct:>5.1f}%)")
    lines.extend([
        "-" * 80,
        f"Ambiguous Samples:         {ambig_count} ({ambig_pct:.1f}%)",
        sep,
    ])
    print("\n".join(lines))
    return 0


def handle_build_pid_data(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes `build-pid-data` CLI subcommand to generate process-level behavioral datasets.
    2. How does the specific function work:
       Loads train workloads, executes parallel simulation under the rollout oracle for S0 and S1,
       saves compressed NPZ datasets, compiles an analytical report, and prints execution stats.
    3. Concepts used in the function:
       Behavioral cloning dataset generation, rollout oracle search, comparative analysis report.
    4. Places where this function is imported and used:
       Dispatched by `main()` in `cli.py` when `command == 'build-pid-data'`.
    5. Final output of the function, before and after:
       Before: Parsed CLI options for dataset directory, capacity limits, and report path.
       After: Datasets and report persisted, summary printed, returns 0 on success.
    """
    data_dir = getattr(args, "data_dir", "data")
    report_path = getattr(args, "report", "results/pid_dataset_report.md")
    max_states = getattr(args, "max_states", config.PID_MAX_STATES)
    workers = getattr(args, "workers", None)

    try:
        t0 = time.perf_counter()
        s0_path, s1_path = generate_and_save_pid_datasets(
            data_dir=data_dir,
            max_states=max_states,
            n_workers=workers,
        )
        rep = generate_dataset_report(
            npz_s0_path=s0_path,
            npz_s1_path=s1_path,
            report_path=report_path,
        )
        elapsed = time.perf_counter() - t0
    except (ValueError, TypeError, OSError) as exc:
        print(f"PID dataset building error: {exc}", file=sys.stderr)
        return 1

    sep = "=" * 80
    lines = [
        sep,
        "Process-Level Behavioral Cloning Dataset Report (Stage 14)",
        sep,
        f"S0 Dataset:                {s0_path}",
        f"S1 Dataset:                {s1_path}",
        f"Analytical Report:         {rep}",
        f"Execution Time:            {elapsed:.2f}s",
        sep,
    ]
    print("\n".join(lines))
    return 0


def handle_train_pid_gbdt(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes `train-pid-gbdt` CLI subcommand to train GBDT candidate rankers for S0/S1.
    2. How does the specific function work:
       Parses setting ("S0", "S1", or "both"), data_dir, models_dir, and val_eval flag.
       Calls train_gbdt for designated setting(s), and if val_eval is requested,
       evaluates the model(s) on validation workloads and saves metrics to results.
       Prints execution time and summary metrics table.
    3. Concepts used in the function:
       CLI command handling, model training orchestration, validation benchmark evaluation.
    4. Places where this function is imported and used:
       Dispatched by `main()` in `cli.py` when `command == 'train-pid-gbdt'`.
    5. Final output of the function, before and after:
       Before: Parsed CLI namespace arguments.
       After: Models trained and saved, validation metrics compiled, returns 0 on success.
    """
    from typing import Any

    setting_arg = getattr(args, "setting", "both").upper().strip()
    data_dir = getattr(args, "data_dir", config.DATA_DIR)
    models_dir = getattr(args, "models_dir", config.MODELS_DIR)
    val_eval = getattr(args, "val_eval", False)
    max_states = getattr(args, "max_states", None)
    n_val_workloads = getattr(args, "n_val_workloads", 500)

    settings_to_train: list[str]
    if setting_arg == "BOTH":
        settings_to_train = ["S0", "S1"]
    elif setting_arg in ("S0", "S1"):
        settings_to_train = [setting_arg]
    else:
        print(f"Error: Unknown setting '{setting_arg}'. Must be S0, S1, or both.", file=sys.stderr)
        return 1

    try:
        t0 = time.perf_counter()
        summary_rows: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
        for s in settings_to_train:
            _, artifact = train_gbdt(
                setting=s,
                data_dir=data_dir,
                models_dir=models_dir,
                max_states=max_states,
            )
            val_metrics: dict[str, Any] | None = None
            if val_eval:
                val_metrics = evaluate_gbdt_on_validation(
                    artifact,
                    setting=s,
                    n_workloads=n_val_workloads,
                    data_dir=data_dir,
                )
            summary_rows.append((s, artifact, val_metrics))
        elapsed = time.perf_counter() - t0
    except (ValueError, TypeError, OSError, FileNotFoundError) as exc:
        print(f"GBDT training error: {exc}", file=sys.stderr)
        return 1

    sep = "=" * 80
    lines = [
        sep,
        "GBDT Candidate Ranker Training Report (Stage 15)",
        sep,
        f"Execution Time:            {elapsed:.2f}s",
    ]
    for s, art, val in summary_rows:
        lines.append(f"Setting:                   {s}")
        lines.append(f"Model Path:                {models_dir}/pid_gbdt_{s}.joblib")
        lines.append(f"Trees:                     {len(art['trees'])}")
        if val is not None:
            lines.append(f"Top-1 Agreement (Oracle):  {val['top1_agreement'] * 100:.2f}%")
            lines.append(f"Mean J_policy:             {val['mean_j_policy']:.4f}")
            lines.append(f"Gap vs Oracle:             {val['closed_loop_gap_oracle'] * 100:.3f}%")
            lines.append(f"Gap vs SRTF:               {val['closed_loop_gap_srtf'] * 100:.3f}%")
        lines.append("-" * 40)
    lines.append(sep)
    print("\n".join(lines))
    return 0


def handle_train_pid_nn(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes `train-pid-nn` CLI subcommand to train neural candidate scorer models for S0/S1.
    2. How does the specific function work:
       Parses CLI options: setting ("S0", "S1", or "both"), data_dir, models_dir, dagger_iters,
       epochs_per_iter, lr, max_states, and n_val_workloads. Invokes `train_neural_scorer()` for
       designated settings, saves pure-NumPy weights, metadata JSON, and multi-seed validation
       summaries to `results/pid_m2_val.csv`. Prints execution time and summary metrics.
    3. Concepts used in the function:
       CLI command handling, neural behavioral cloning, DAgger loop execution, validation reporting.
    4. Places where this function is imported and used:
       Dispatched by `main()` in `cli.py` when `command == 'train-pid-nn'`.
    5. Final output of the function, before and after:
       Before: Parsed CLI namespace arguments.
       After: Neural models trained and saved, validation metrics recorded, returns 0 on success.
    """
    from typing import Any

    from schedsim.pidsel.train_bc import train_neural_scorer

    setting_arg = getattr(args, "setting", "both").upper().strip()
    data_dir = getattr(args, "data_dir", config.DATA_DIR)
    models_dir = getattr(args, "models_dir", config.MODELS_DIR)
    dagger_iters = getattr(args, "dagger_iters", config.DAGGER_ITERS)
    epochs = getattr(args, "epochs", 15)
    lr = getattr(args, "lr", 1e-3)
    max_states = getattr(args, "max_states", 30000)
    n_val = getattr(args, "n_val_workloads", 200)

    settings_to_train: list[str]
    if setting_arg == "BOTH":
        settings_to_train = ["S0", "S1"]
    elif setting_arg in config.SETTINGS:
        settings_to_train = [setting_arg]
    else:
        valid = ", ".join(config.SETTINGS.keys())
        print(f"Error: Unknown setting '{setting_arg}'. Valid choices: {valid}, both.")
        return 1

    start_time = time.time()
    summary_rows: list[dict[str, Any]] = []

    for s in settings_to_train:
        print(f"\nTraining Neural Candidate Scorer for {s}...")
        meta = train_neural_scorer(
            setting=s,
            data_dir=data_dir,
            models_dir=models_dir,
            dagger_iters=dagger_iters,
            epochs_per_iter=epochs,
            lr=lr,
            max_train_states=max_states,
            n_val_workloads=n_val,
        )
        summary_rows.append(meta)

    elapsed = time.time() - start_time
    sep = "=" * 80
    lines = [
        sep,
        "Neural Candidate Scorer Training Report (Stage 16)",
        sep,
        f"Execution Time:            {elapsed:.2f}s",
    ]
    for meta in summary_rows:
        s = meta["setting"]
        best = meta["best_seed_metrics"]
        summ = meta["seeds_summary"]
        lines.append(f"Setting:                   {s}")
        lines.append(f"Model Path:                {models_dir}/pid_scorer_{s}_np.npz")
        lines.append(f"Best Seed:                 {meta['best_seed']}")
        lines.append(f"Top-1 Agreement:           {best['top1_agreement'] * 100:.2f}%")
        lines.append(f"Mean J_policy:             {best['mean_waiting_time']:.4f}")
        lines.append(f"Gap vs Oracle:             {best['gap_vs_oracle'] * 100:.3f}%")
        lines.append(f"Gap vs SRTF:               {best['gap_vs_srtf'] * 100:.3f}%")
        lines.append(
            f"Multi-Seed Waiting:        "
            f"{summ['waiting_mean']:.4f} +/- {summ['waiting_std']:.4f}"
        )
        lines.append("-" * 40)
    lines.append(sep)
    print("\n".join(lines))
    return 0


def handle_bench_latency(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes `bench-latency` CLI command benchmarking all scheduling policies and ML models.
    2. How does the specific function work:
       Parses benchmark options (n_warmup, n_timed, n_repeats, output, plot, skip_plot).
       Executes benchmark_all() across ready queue sizes (2, 3, 5, 8, 10) and workload sizes
       (3, 5, 8, 10, 15, 20). Persists structured JSON profile to disk, optionally generates
       latency vs n_ready plot, runs live-vs-table validation checks, and prints formatted tables.
    3. Concepts used in the function:
       Micro-benchmarking execution, JSON profile generation, tabular formatting, validation.
    4. Places where this function is imported and used:
       Dispatched by main() in cli.py when command == 'bench-latency'.
    5. Final output of the function, before and after:
       Before: Command arguments parsed.
       After: latency_profile.json and plot generated, summary printed, returns 0 on success.
    """
    import logging

    from schedsim.domain.process import Process
    from schedsim.overhead import (
        benchmark_all,
        plot_latency_profile,
        save_latency_profile,
        validate_live_vs_table,
    )
    from schedsim.pidsel.gbdt_ranker import GBDTRankerPolicy
    from schedsim.pidsel.scorer_net import NeuralCandidatePolicy

    logger = logging.getLogger(__name__)

    output_path = getattr(args, "output", config.MODELS_DIR / "latency_profile.json")
    plot_path = getattr(args, "plot", config.RESULTS_DIR / "figures" / "latency_vs_ready.png")
    n_warmup = getattr(args, "n_warmup", 200)
    n_timed = getattr(args, "n_timed", 2000)
    n_repeats = getattr(args, "n_repeats", 3)
    skip_plot = getattr(args, "skip_plot", False)

    print("Running CPU Scheduling Policy Hardware Latency Benchmarks...")
    t0 = time.time()

    profile = benchmark_all(
        n_warmup_decision=n_warmup,
        n_timed_decision=n_timed,
        n_warmup_startup=max(10, n_warmup // 4),
        n_timed_startup=max(50, n_timed // 4),
        n_repeats=n_repeats,
    )

    save_latency_profile(profile, output_path)
    print(f"Latency profile saved to: {output_path}")

    if not skip_plot:
        plot_latency_profile(profile, plot_path)
        print(f"Latency curve plot saved to: {plot_path}")

    # Validation check on 5-process sample workload
    val_procs = [
        Process(pid=f"P{i + 1}", arrival=i, burst=4 + (i % 3))
        for i in range(5)
    ]
    val_results = []
    for pol_cls in (NeuralCandidatePolicy, GBDTRankerPolicy):
        try:
            pol_inst = pol_cls.from_setting("S1")
            val_res = validate_live_vs_table(
                pol_inst, val_procs, cs_cost=1, tolerance=0.15
            )
            val_results.append(val_res)
        except Exception as exc:
            logger.warning(f"Validation check skipped for {pol_cls.__name__}: {exc}")

    elapsed = time.time() - t0
    sep = "=" * 80
    subsep = "-" * 80

    os_info = f"{profile['machine_info']['os']} {profile['machine_info']['os_release']}"
    lines = [
        sep,
        "Hardware Latency Profile and Cost Model Report (Stage 17)",
        sep,
        f"Benchmark Duration:         {elapsed:.2f}s",
        f"Warmup / Timed Iterations:  {n_warmup} / {n_timed} (repeats={n_repeats})",
        "",
        "Host Machine Information:",
        f"  CPU:                      {profile['machine_info']['cpu']}",
        f"  OS:                       {os_info}",
        f"  Python Version:           {profile['machine_info']['python_version']}",
        f"  Timestamp:                {profile['machine_info']['timestamp']}",
        "",
        "Per-Decision Dispatch Latency (median µs / p95 µs):",
        subsep,
        f"{'Policy':<14} {'n=2':<12} {'n=3':<12} {'n=5':<12} {'n=8':<12} {'n=10':<12}",
        subsep,
    ]

    for pol_name, points in profile["per_decision"].items():
        row_parts = [f"{pol_name:<14}"]
        for n_str in ("2", "3", "5", "8", "10"):
            if n_str in points:
                med = points[n_str]["median_us"]
                p95 = points[n_str]["p95_us"]
                row_parts.append(f"{med:.1f}/{p95:.1f}".center(12))
            else:
                row_parts.append("-".center(12))
        lines.append(" ".join(row_parts))

    lines.extend([
        subsep,
        "",
        "Category 1 Upfront Startup Latency (median µs / p95 µs):",
        subsep,
        (
            f"{'Model':<14} {'sz=3':<10} {'sz=5':<10} {'sz=8':<10} "
            f"{'sz=10':<10} {'sz=15':<10} {'sz=20':<10}"
        ),
        subsep,
    ])

    for model_name, points in profile["startup"].items():
        display_name = "Decision Tree" if model_name == "DT" else "Random Forest"
        row_parts = [f"{display_name:<14}"]
        for sz_str in ("3", "5", "8", "10", "15", "20"):
            if sz_str in points:
                med = points[sz_str]["median_us"]
                p95 = points[sz_str]["p95_us"]
                row_parts.append(f"{med:.1f}/{p95:.1f}".center(10))
            else:
                row_parts.append("-".center(10))
        lines.append(" ".join(row_parts))

    if val_results:
        lines.extend([
            subsep,
            "",
            "Live Timing vs Table Cost Validation Check:",
            subsep,
            (
                f"{'Policy':<20} {'Decisions':<10} {'Live (µs)':<12} "
                f"{'Table (µs)':<12} {'Gap (%)':<10} {'Status':<8}"
            ),
            subsep,
        ])
        for vr in val_results:
            status_str = "PASS" if vr["within_tolerance"] else "FLAGGED"
            gap_pct_str = f"{vr['percentage_gap']:.1f}%"
            lines.append(
                f"{vr['policy']:<20} {vr['decisions']:<10} {vr['live_us_total']:<12.1f} "
                f"{vr['table_us_total']:<12.1f} {gap_pct_str:<10} {status_str:<8}"
            )

    lines.extend([
        subsep,
        f"Artifact: {output_path}",
        sep,
    ])

    print("\n".join(lines))
    return 0


def handle_train_tree(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes the `train-tree` CLI subcommand to fit and benchmark the Decision Tree model.
    2. How does the specific function work:
       Loads labeled dataset from disk, executes cross-validated grid search and baseline model
       benchmarks via `train_decision_tree()`, performs validation evaluation via
       `evaluate_decision_tree()`, and prints a comprehensive performance and comparison report.
    3. Concepts used in the function:
       Model training dispatch, validation reporting, baseline comparison, CLI formatting.
    4. Places where this function is imported and used:
       Dispatched by `main()` in `cli.py` when `command == 'train-tree'`.
    5. Final output of the function, before and after:
       Before: Parsed CLI options for dataset path and output model destinations.
       After: Model joblib and metadata JSON saved, summary table printed, returns 0 on success.
    """
    dataset_arg = getattr(args, "dataset", "data/scheduling_dataset.csv")
    model_arg = getattr(args, "output_model", "models/decision_tree.joblib")
    meta_arg = getattr(args, "output_meta", "models/decision_tree_meta.json")
    metrics_arg = getattr(args, "output_metrics", "results/tree_metrics_val.json")
    seed = getattr(args, "seed", 42)

    try:
        t0 = time.perf_counter()
        meta = train_decision_tree(
            dataset=dataset_arg,
            output_model_path=model_arg,
            output_meta_path=meta_arg,
            seed=seed,
        )
        eval_metrics = evaluate_decision_tree(
            model_path=model_arg,
            dataset=dataset_arg,
            output_metrics_path=metrics_arg,
        )
        elapsed = time.perf_counter() - t0
    except (ValueError, TypeError, OSError, FileNotFoundError) as exc:
        print(f"Model training error: {exc}", file=sys.stderr)
        return 1

    params = meta.get("chosen_params", {})
    val_metrics = meta.get("metrics_validation", {})
    dt_acc = eval_metrics.get("accuracy", 0.0)
    dt_f1 = eval_metrics.get("macro_f1", 0.0)
    regret = eval_metrics.get("mean_regret", 0.0)

    sep = "=" * 80
    lines = [
        sep,
        "Decision Tree Training & Benchmark Report",
        sep,
        f"Training Samples:          {meta.get('training_samples', 0)}",
        f"Validation Samples:        {meta.get('validation_samples', 0)}",
        (
            f"Chosen Hyperparameters:    max_depth={params.get('max_depth')}, "
            f"min_samples_leaf={params.get('min_samples_leaf')}, "
            f"class_weight={params.get('class_weight')}"
        ),
        f"Training & Eval Time:      {elapsed:.2f}s",
        "-" * 80,
        "Validation Model Comparison:",
        f"  {'Model':<25} {'Accuracy':>10} {'Macro-F1':>10}",
        "-" * 80,
    ]
    model_labels = [
        ("Decision Tree", dt_acc, dt_f1),
        (
            "Majority Class Dummy",
            val_metrics.get("majority_class_dummy", {}).get("accuracy", 0.0),
            val_metrics.get("majority_class_dummy", {}).get("macro_f1", 0.0),
        ),
        (
            "Random Forest",
            val_metrics.get("random_forest", {}).get("accuracy", 0.0),
            val_metrics.get("random_forest", {}).get("macro_f1", 0.0),
        ),
        (
            "K-Nearest Neighbors",
            val_metrics.get("k_nearest_neighbors", {}).get("accuracy", 0.0),
            val_metrics.get("k_nearest_neighbors", {}).get("macro_f1", 0.0),
        ),
    ]
    for name, a, f in model_labels:
        lines.append(f"  {name:<25} {a:>10.4f} {f:>10.4f}")

    lines.extend([
        "-" * 80,
        f"Validation Mean Regret:    {regret:.4f}",
        (
            f"Unambiguous Accuracy:      "
            f"{eval_metrics.get('accuracy_unambiguous', 0.0):.4f}"
        ),
        f"Model Saved To:            {model_arg}",
        f"Metadata Saved To:         {meta_arg}",
        sep,
    ])
    print("\n".join(lines))
    return 0


def handle_recommend(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes the `recommend` CLI subcommand to recommend a scheduling policy for a workload.
    2. How does the specific function work:
       Loads the specified preset workload, extracts domain features, runs inference through
       `recommend()`, and prints the recommended policy, class probabilities, and feature values.
    3. Concepts used in the function:
       Inference dispatch, probability reporting, CLI result formatting.
    4. Places where this function is imported and used:
       Dispatched by `main()` in `cli.py` when `command == 'recommend'`.
    5. Final output of the function, before and after:
       Before: Parsed CLI options for preset identifier and model path.
       After: Recommended policy and probabilities printed, returns 0 on success.
    """
    preset_name = getattr(args, "preset", "1")
    model_arg = getattr(args, "model", "models/decision_tree.joblib")

    try:
        procs = get_preset(preset_name)
        rec = recommend(procs, model_path=model_arg)
    except (KeyError, ValueError, TypeError, FileNotFoundError) as exc:
        print(f"Recommendation error: {exc}", file=sys.stderr)
        return 1

    sep = "=" * 80
    lines = [
        sep,
        f"Policy Recommendation Report: Preset={preset_name} ({len(procs)} processes)",
        sep,
        f"Recommended Policy:        {rec['algorithm']}",
        "-" * 80,
        "Class Probabilities:",
    ]
    for policy_name, prob in sorted(rec["probabilities"].items()):
        lines.append(f"  {policy_name:<8} {prob:>6.3f} ({prob * 100.0:>5.1f}%)")

    lines.extend([
        "-" * 80,
        "Extracted Workload Features:",
    ])
    for feat_name, val in sorted(rec["features"].items()):
        lines.append(f"  {feat_name:<24} {val:>8.3f}")
    lines.append(sep)

    print("\n".join(lines))
    return 0


def handle_train_rl(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       CLI handler executing tabular Q-learning training, hyperparameter search, or evaluation.
    2. How does the specific function work:
       If --hparam-search is flagged, runs grid search over learning rates, discount factors,
       and epsilon decays, printing the results table. If --multi-seed is flagged, trains final
       agents across all canonical seeds for cs_cost=0 and cs_cost=1, saving checkpoints and
       summaries. Otherwise, executes training for the specified seed and cost, saves models,
       and outputs validation metrics.
    3. Concepts used in the function:
       Reinforcement learning CLI command dispatch, hyperparameter tuning, model artifact export.
    4. Places where this function is imported and used:
       Called by main() when 'train-rl' subcommand is selected.
    5. Final output of the function, before and after:
       Before: Parsed CLI namespace arguments.
       After: Trains model, writes checkpoints to disk, and returns integer process exit code 0.
    """
    t0 = time.time()
    sep = "=" * 80

    if getattr(args, "hparam_search", False):
        print(sep)
        print("Executing Q-Learning Hyperparameter Search on Validation Split...")
        print(sep)
        best_cfg, df = run_hparam_search(
            cs_cost=args.cs_cost,
            output_path=getattr(args, "output_results", None),
        )
        elapsed = time.time() - t0
        print(df.to_string(index=False))
        print("-" * 80)
        print(
            f"Optimal Hyperparameters: alpha={best_cfg['alpha']}, gamma={best_cfg['gamma']}, "
            f"eps_decay={best_cfg['eps_decay']} (Val Wait: {best_cfg['mean_val_waiting']:.4f})"
        )
        print("Saved hyperparameter search results to: results/rl_hparam_search.csv")
        print(f"Total time elapsed: {elapsed:.2f}s")
        print(sep)
        return 0

    if getattr(args, "multi_seed", False):
        print(sep)
        print("Training Canonical Multi-Seed Q-Learning Agents (cs_cost 0 and 1)...")
        print(sep)
        artifacts = train_final_agents(
            episodes=args.episodes,
            alpha=args.alpha,
            gamma=args.gamma,
            eps_decay=args.eps_decay,
            eval_every=args.eval_every,
        )
        elapsed = time.time() - t0
        print(f"Multi-seed training complete in {elapsed:.2f}s.")
        print("Artifacts generated:")
        for k, v in artifacts.items():
            print(f"  {k}: {v}")
        try:
            plot_learning_curves()
            plot_action_distribution()
            print("Generated figures in results/figures/")
        except Exception as exc:
            print(f"Note: Figure plotting skipped: {exc}")
        print(sep)
        return 0

    # Single-run mode
    print(sep)
    print(
        f"Training Q-Learning Agent: episodes={args.episodes}, seed={args.seed}, "
        f"cs_cost={args.cs_cost}"
    )
    print(
        f"Hyperparameters: alpha={args.alpha}, gamma={args.gamma}, "
        f"eps_decay={args.eps_decay}"
    )
    print(sep)

    log_path = getattr(args, "output_logs", None) or Path("data/rl_training_logs.csv")
    agent, logs = train_q_learning(
        episodes=args.episodes,
        seed=args.seed,
        cs_cost=args.cs_cost,
        alpha=args.alpha,
        gamma=args.gamma,
        eps_decay=args.eps_decay,
        eval_every=args.eval_every,
        save_logs=True,
        log_path=log_path,
    )

    elapsed = time.time() - t0
    final_metrics = logs[-1] if logs else {}

    model_p = Path(
        getattr(args, "output_model", None) or f"models/q_table_cs{args.cs_cost}.pkl"
    )
    summary_p = Path(
        getattr(args, "output_summary", None) or f"models/q_table_summary_cs{args.cs_cost}.json"
    )

    agent.save(model_p)
    with open(summary_p, "w", encoding="utf-8") as f:
        json.dump(agent.get_summary(top_n=500), f, indent=2)

    if args.cs_cost == 1:
        default_model = Path("models/q_table.pkl")
        default_summary = Path("models/q_table_summary.json")
        agent.save(default_model)
        with open(default_summary, "w", encoding="utf-8") as f:
            json.dump(agent.get_summary(top_n=500), f, indent=2)

    print(f"Training finished in {elapsed:.2f}s.")
    print(f"Final Validation Waiting Time: {final_metrics.get('avg_waiting', 'N/A')}")
    print(f"Final Validation Turnaround:   {final_metrics.get('avg_turnaround', 'N/A')}")
    print(f"Final Unseen State Rate:       {final_metrics.get('unseen_state_rate', 0.0):.4f}")
    print(f"Final Q-Table Size:            {agent.q_table_size} state-action entries")
    print(f"Unique States Visited:         {agent.unique_states_visited}")
    print(f"Saved model to:   {model_p}")
    print(f"Saved summary to: {summary_p}")
    print(sep)
    return 0


def handle_evaluate(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes complete multi-policy evaluation across held-out workloads, computes bootstrap CIs,
       exports summary tables, and generates publication figures.
    2. How does the specific function work:
       Parses CLI arguments (split, n_workloads, output paths), executes run_evaluation_experiments,
       computes summary table with 95% bootstrap CIs via compute_summary_table, writes summary CSV
       and Markdown tables, evaluates Decision Tree test accuracy and regret via
       evaluate_decision_tree_test_split, exports results/scenario_breakdown.csv, and renders all
       publication figures via generate_all_plots.
    3. Concepts used in the function:
       Held-out statistical benchmarking, bootstrap estimation, Markdown table generation,
       plotting orchestration.
    4. Places where this function is imported and used:
       Invoked by main() when subcommand is 'evaluate'.
    5. Final output of the function, before and after:
       Before: Raw CLI arguments.
       After: Complete evaluation CSV, summary tables, test JSON, and figure PNGs generated.
    """
    sep = "=" * 80
    print(sep)
    print("SCHEDSIM BENCHMARK & EVALUATION SUITE (Stage 7)")
    print(sep)
    print(f"Dataset Split:       {args.split}")
    print(f"Workload Count:      {args.n_workloads}")
    print("Evaluated Policies:  FCFS, SJF, RR(q=2), SRTF, Q-Learning (RL), Decision Tree (DT)")
    print("Context Switch Costs: 0, 1")
    print(sep)

    t0 = time.perf_counter()

    out_res = (
        Path(args.output_results)
        if args.output_results
        else (config.RESULTS_DIR / "evaluation_results.csv")
    )
    out_csv = (
        Path(args.output_summary_csv)
        if args.output_summary_csv
        else (config.RESULTS_DIR / "summary_table.csv")
    )
    out_md = (
        Path(args.output_summary_md)
        if args.output_summary_md
        else (config.RESULTS_DIR / "summary_table.md")
    )
    out_fig = (
        Path(args.output_figures_dir)
        if args.output_figures_dir
        else (config.RESULTS_DIR / "figures")
    )

    print("Running multi-policy simulations across workloads...")
    eval_df = run_evaluation_experiments(
        split=args.split,
        n_workloads=args.n_workloads,
        cs_costs=(0, 1),
        output_path=out_res,
        n_workers=args.workers,
    )
    print(f"  Generated {len(eval_df)} per-workload execution records -> {out_res}")

    print("Computing 1,000-resample bootstrap confidence intervals (95% CI)...")
    summary_df = compute_summary_table(eval_df, n_resamples=1000)
    export_summary_tables(summary_df, csv_path=out_csv, md_path=out_md)
    print(f"  Exported summary table -> {out_csv}")
    print(f"  Exported Markdown report table -> {out_md}")

    print("Computing per-scenario breakdown across policies...")
    scenario_df = compute_scenario_breakdown(eval_df, metric="avg_waiting")
    scenario_csv = out_res.parent / "scenario_breakdown.csv"
    scenario_df.to_csv(scenario_csv, index=False)
    print(f"  Exported scenario breakdown -> {scenario_csv}")

    print("Evaluating Decision Tree on held-out test split...")
    test_workloads = generate_split_workloads(
        split=args.split, n_workloads=min(args.n_workloads, 1000)
    )
    dt_test_metrics = evaluate_decision_tree_test_split(
        test_workloads, cs_cost=config.LABEL_CS_COST
    )
    dt_json_path = out_res.parent / "decision_tree_test_metrics.json"
    with open(dt_json_path, "w", encoding="utf-8") as f:
        json.dump(dt_test_metrics, f, indent=2)
    print(
        f"  Test Accuracy: {dt_test_metrics['accuracy']:.4f} | "
        f"Macro-F1: {dt_test_metrics['macro_f1']:.4f} | "
        f"Mean Regret: {dt_test_metrics['regret']['decision_tree']:.4f}"
    )
    print(f"  Saved Decision Tree test metrics -> {dt_json_path}")

    print("Rendering publication-quality evaluation figures (200 DPI)...")
    figs = generate_all_plots(
        summary_df=summary_df,
        scenario_df=scenario_df,
        dt_test_metrics=dt_test_metrics,
        output_dir=out_fig,
    )
    for fig_name, fig_path in figs.items():
        print(f"  Rendered [{fig_name}] -> {fig_path}")

    elapsed = time.perf_counter() - t0
    print(sep)
    print(f"Evaluation suite completed successfully in {elapsed:.2f}s.")
    print(sep)
    return 0


def handle_compare_all(args: argparse.Namespace) -> int:
    """
    1. The use of function:
       Executes Stage 18 final comparison across all 11 policies on S0 and S1 test splits.
    2. How does the specific function work:
       Extracts n_workloads, tick_us, skip_plots, and output_dir from parsed arguments.
       Invokes run_final_comparison(), which simulates classical, category 1/2/3 AI policies,
       and oracle/best-classical references, computes 15-column comparison matrices with bootstrap
       CIs, generates publication figures, and evaluates pre-registered hypotheses H1-H4.
       Prints summary Markdown tables to terminal and reports generated artifact paths.
    3. Concepts used in the function:
       CLI subcommand dispatching, multi-setting benchmark reporting, terminal formatting.
    4. Places where this function is imported and used:
       Dispatched by main() when subcommand is 'compare-all'.
    5. Final output of the function, before and after:
       Before: Raw CLI arguments.
       After: Executes comparison suite, prints tables and artifact paths, returns 0 on success.
    """
    sep = "=" * 80
    print(sep)
    print("SCHEDSIM FINAL COMPARISON SUITE (Stage 18: The Deliverable)")
    print(sep)
    print(f"Workload Count:      {args.n_workloads}")
    print(f"Default Tick (us):   {args.tick_us}")
    print(f"Skip Plots:          {args.skip_plots}")
    print(f"Output Directory:    {args.output_dir}")
    print(sep)

    out_dir = Path(args.output_dir) if args.output_dir else config.RESULTS_DIR

    results = run_final_comparison(
        n_workloads=args.n_workloads,
        tick_us=args.tick_us,
        skip_plots=args.skip_plots,
        output_dir=out_dir,
    )

    print("\n" + sep)
    print("FINAL COMPARISON RESULTS - SETTING S0 (cs_cost = 0)")
    print(sep)
    s0_md_path = out_dir / "final_comparison_S0.md"
    if s0_md_path.exists():
        print(s0_md_path.read_text(encoding="utf-8"))

    print("\n" + sep)
    print("FINAL COMPARISON RESULTS - SETTING S1 (cs_cost = 1)")
    print(sep)
    s1_md_path = out_dir / "final_comparison_S1.md"
    if s1_md_path.exists():
        print(s1_md_path.read_text(encoding="utf-8"))

    print("\n" + sep)
    print("GENERATED DELIVERABLES & ARTIFACTS")
    print(sep)
    print(f"  Comparison Tables: {out_dir / 'final_comparison_S0.csv'}, {s0_md_path}")
    print(f"                     {out_dir / 'final_comparison_S1.csv'}, {s1_md_path}")
    print(f"  Hypotheses Results:{results['hypotheses_path']}")
    if not args.skip_plots and results.get("figures"):
        print("  Figures:")
        for fig in results["figures"]:
            print(f"    - {fig}")
    print(sep)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """
    1. The use of function:
       Constructs and configures top-level argument parser and subcommands for schedsim CLI.
    2. How does the specific function work:
       Instantiates `argparse.ArgumentParser`, attaches global descriptive help, and creates
       subparsers for 'simulate', 'compare', 'build-dataset', 'train-tree', 'train-rl', and
       'evaluate' subcommands with their respective flags and arguments.
    3. Concepts used in the function:
       Declarative CLI parsing, subcommand dispatching, help formatting, argument routing.
    4. Places where this function is imported and used:
       Called directly inside `main()` in `cli.py` and importable by test suites.
    5. Final output of the function, before and after:
       Before: No parser configured.
       After: Returns a fully initialized `argparse.ArgumentParser` instance ready to parse.
    """
    parser = argparse.ArgumentParser(
        prog="schedsim",
        description="AI-Driven CPU Scheduling Simulator (FCFS, SJF, SRTF, RR, ML, RL)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # simulate subcommand
    sim_parser = subparsers.add_parser(
        "simulate", help="Run a simulation for a specific policy"
    )
    sim_parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Scheduling policy to simulate (e.g. FCFS, SJF, SRTF, RR)",
    )
    sim_parser.add_argument(
        "--preset", type=str, default="1", help="Preset workload identifier or name"
    )
    sim_parser.add_argument(
        "--quantum", type=int, default=2, help="Time quantum for Round Robin"
    )
    sim_parser.add_argument(
        "--cs-cost", type=int, default=0, help="Context switch cost in ticks"
    )

    # compare subcommand
    comp_parser = subparsers.add_parser(
        "compare", help="Compare multiple scheduling policies on a workload"
    )
    comp_parser.add_argument(
        "--preset", type=str, default="1", help="Preset workload identifier or name"
    )
    comp_parser.add_argument(
        "--cs-cost", type=int, default=0, help="Context switch cost in ticks"
    )

    # build-workloads subcommand
    workloads_parser = subparsers.add_parser(
        "build-workloads",
        aliases=["build-workload"],
        help="Generate synthetic workloads and CSV dataset",
    )
    workloads_parser.add_argument(
        "--n", type=int, default=6000, help="Number of workloads to generate"
    )
    workloads_parser.add_argument(
        "--output",
        type=str,
        default="data/workloads.csv",
        help="Output path for workloads CSV",
    )
    workloads_parser.add_argument(
        "--split",
        type=str,
        choices=list(ALL_SPLITS),
        default="train",
        help="Target dataset split partition (train, val, test)",
    )
    workloads_parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Starting seed offset within partition",
    )

    # build-dataset subcommand
    data_parser = subparsers.add_parser(
        "build-dataset",
        help="Generate labeled supervised dataset from workloads",
    )
    data_parser.add_argument(
        "--input",
        type=str,
        default="data/workloads.csv",
        help="Input path to raw workloads CSV",
    )
    data_parser.add_argument(
        "--output",
        type=str,
        default="data/scheduling_dataset.csv",
        help="Output path for labeled dataset CSV",
    )
    data_parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Number of workloads to evaluate (default: all)",
    )
    data_parser.add_argument(
        "--cs-cost",
        type=int,
        default=1,
        help="Context switch penalty in ticks",
    )
    data_parser.add_argument(
        "--quantum",
        type=int,
        default=2,
        help="Round Robin quantum for labeling",
    )

    # train-tree subcommand
    tree_parser = subparsers.add_parser(
        "train-tree", help="Train the Decision Tree policy recommender"
    )
    tree_parser.add_argument(
        "--dataset",
        type=str,
        default="data/scheduling_dataset.csv",
        help="Path to labeled scheduling dataset CSV",
    )
    tree_parser.add_argument(
        "--output-model",
        type=str,
        default="models/decision_tree.joblib",
        help="Output path for trained model joblib",
    )
    tree_parser.add_argument(
        "--output-meta",
        type=str,
        default="models/decision_tree_meta.json",
        help="Output path for model metadata JSON",
    )
    tree_parser.add_argument(
        "--output-metrics",
        type=str,
        default="results/tree_metrics_val.json",
        help="Output path for validation metrics JSON",
    )
    tree_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model training",
    )

    # recommend subcommand
    rec_parser = subparsers.add_parser(
        "recommend", help="Recommend optimal scheduling policy for a workload"
    )
    rec_parser.add_argument(
        "--preset",
        type=str,
        default="1",
        help="Preset workload identifier or name",
    )
    rec_parser.add_argument(
        "--model",
        type=str,
        default="models/decision_tree.joblib",
        help="Path to trained model joblib",
    )

    # train-rl subcommand
    rl_parser = subparsers.add_parser(
        "train-rl", help="Train the tabular Q-learning scheduling agent"
    )
    rl_parser.add_argument(
        "--episodes", type=int, default=config.RL_EPISODES, help="Number of training episodes"
    )
    rl_parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for training"
    )
    rl_parser.add_argument(
        "--cs-cost", type=int, default=1, choices=[0, 1], help="Context switch cost (0 or 1)"
    )
    rl_parser.add_argument(
        "--alpha", type=float, default=config.ALPHA, help="Learning rate alpha"
    )
    rl_parser.add_argument(
        "--gamma", type=float, default=config.GAMMA, help="Discount factor gamma"
    )
    rl_parser.add_argument(
        "--eps-decay", type=float, default=config.EPS_DECAY, help="Epsilon decay rate per episode"
    )
    rl_parser.add_argument(
        "--eval-every", type=int, default=config.RL_EVAL_EVERY, help="Validation frequency"
    )
    rl_parser.add_argument(
        "--hparam-search", action="store_true", help="Run hyperparameter search on validation split"
    )
    rl_parser.add_argument(
        "--multi-seed", action="store_true", help="Train across canonical seeds and save all models"
    )
    rl_parser.add_argument(
        "--output-model", type=str, default=None, help="Output path for saved model checkpoint"
    )
    rl_parser.add_argument(
        "--output-summary", type=str, default=None, help="Output path for JSON state summary"
    )
    rl_parser.add_argument(
        "--output-logs", type=str, default=None, help="Output path for training trajectory CSV"
    )
    rl_parser.add_argument(
        "--output-results", type=str, default=None, help="Output path for search results CSV"
    )

    # evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run full evaluation across held-out test workloads"
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Evaluation split to run",
    )
    eval_parser.add_argument(
        "--n-workloads",
        type=int,
        default=1000,
        help="Number of workloads to evaluate (default: 1000)",
    )
    eval_parser.add_argument(
        "--output-results",
        type=str,
        default=None,
        help="Path for evaluation CSV (default: results/evaluation_results.csv)",
    )
    eval_parser.add_argument(
        "--output-summary-csv",
        type=str,
        default=None,
        help="Output path for summary CSV table (default: results/summary_table.csv)",
    )
    eval_parser.add_argument(
        "--output-summary-md",
        type=str,
        default=None,
        help="Output path for summary Markdown table (default: results/summary_table.md)",
    )
    eval_parser.add_argument(
        "--output-figures-dir",
        type=str,
        default=None,
        help="Directory to save generated figures (default: results/figures)",
    )
    eval_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count)",
    )

    # build-pid-data sub-command (Stage 14)
    pid_data_parser = subparsers.add_parser(
        "build-pid-data",
        help="Build process-level behavioral cloning datasets and report (Stage 14)",
    )
    pid_data_parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to save NPZ datasets (default: data)",
    )
    pid_data_parser.add_argument(
        "--report",
        type=str,
        default="results/pid_dataset_report.md",
        help="Path for dataset analysis Markdown report (default: results/pid_dataset_report.md)",
    )
    pid_data_parser.add_argument(
        "--max-states",
        type=int,
        default=config.PID_MAX_STATES,
        help=f"Maximum states per setting (default: {config.PID_MAX_STATES})",
    )
    pid_data_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU count)",
    )

    # train-pid-gbdt sub-command (Stage 15)
    pid_gbdt_parser = subparsers.add_parser(
        "train-pid-gbdt",
        help="Train GBDT candidate ranker models for S0 and/or S1 (Stage 15)",
    )
    pid_gbdt_parser.add_argument(
        "--setting",
        type=str,
        default="both",
        choices=["S0", "S1", "both", "s0", "s1", "BOTH"],
        help="Setting to train: S0, S1, or both (default: both)",
    )
    pid_gbdt_parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing training datasets (default: data)",
    )
    pid_gbdt_parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts (default: models)",
    )
    pid_gbdt_parser.add_argument(
        "--val-eval",
        action="store_true",
        help="Run validation evaluation after training",
    )
    pid_gbdt_parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="Optional maximum states to train on (default: all)",
    )
    pid_gbdt_parser.add_argument(
        "--n-val-workloads",
        type=int,
        default=500,
        help="Number of validation workloads to evaluate (default: 500)",
    )

    # train-pid-nn sub-command (Stage 16)
    pid_nn_parser = subparsers.add_parser(
        "train-pid-nn",
        help="Train Neural candidate scorer models for S0 and/or S1 (Stage 16)",
    )
    pid_nn_parser.add_argument(
        "--setting",
        type=str,
        default="both",
        choices=["S0", "S1", "both", "s0", "s1", "BOTH"],
        help="Setting to train: S0, S1, or both (default: both)",
    )
    pid_nn_parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing training datasets (default: data)",
    )
    pid_nn_parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save trained model artifacts (default: models)",
    )
    pid_nn_parser.add_argument(
        "--dagger-iters",
        type=int,
        default=config.DAGGER_ITERS,
        help="Number of DAgger iterations (default: 2)",
    )
    pid_nn_parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Epochs per DAgger iteration (default: 15)",
    )
    pid_nn_parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 0.001)",
    )
    pid_nn_parser.add_argument(
        "--max-states",
        type=int,
        default=30000,
        help="Maximum initial states to train on (default: 30000)",
    )
    pid_nn_parser.add_argument(
        "--n-val-workloads",
        type=int,
        default=200,
        help="Number of validation workloads to evaluate (default: 200)",
    )

    bench_parser = subparsers.add_parser(
        "bench-latency",
        help="Run micro-benchmarks for all scheduling policies and Category 1 models.",
    )
    bench_parser.add_argument(
        "--output",
        type=Path,
        default=config.MODELS_DIR / "latency_profile.json",
        help="Target JSON output path for the latency profile.",
    )
    bench_parser.add_argument(
        "--plot",
        type=Path,
        default=config.RESULTS_DIR / "figures" / "latency_vs_ready.png",
        help="Target image output path for the latency vs n_ready plot.",
    )
    bench_parser.add_argument(
        "--n-warmup",
        type=int,
        default=200,
        help="Number of warmup iterations per policy/candidate sweep point (default: 200).",
    )
    bench_parser.add_argument(
        "--n-timed",
        type=int,
        default=2000,
        help="Number of timed iterations per policy/candidate sweep point (default: 2000).",
    )
    bench_parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of repeat measurements (default: 3).",
    )
    bench_parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip generating the latency plot figure.",
    )

    compare_all_parser = subparsers.add_parser(
        "compare-all",
        help="Run full final comparison across all 11 policies for S0 and S1 (Stage 18).",
    )
    compare_all_parser.add_argument(
        "--n-workloads",
        type=int,
        default=1000,
        help="Number of test workloads to evaluate (default: 1000).",
    )
    compare_all_parser.add_argument(
        "--tick-us",
        type=float,
        default=config.DEFAULT_TICK_US,
        help=(
            "Baseline hardware tick duration in microseconds "
            f"(default: {config.DEFAULT_TICK_US})."
        ),
    )
    compare_all_parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating publication figures.",
    )
    compare_all_parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RESULTS_DIR,
        help="Target output directory for tables and figures (default: results).",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    1. The use of function:
       Entry point execution for the CLI application.
    2. How does the specific function work:
       Invokes `build_parser()`, parses provided arguments (or `sys.argv[1:]` by default),
       verifies that a subcommand was provided, and dispatches to handler or prints help.
    3. Concepts used in the function:
       CLI argument validation, sub-command handling, exit code management, error handling.
    4. Places where this function is imported and used:
       Invoked when `cli.py` is executed as `__main__`, or called by integration tests.
    5. Final output of the function, before and after:
       Before: Raw command line input strings.
       After: Command executed, returns integer process exit code (0 for success).
    """
    parser = build_parser()
    try:
        parsed_args = parser.parse_args(args)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    if parsed_args.command is None:
        parser.print_help()
        return 0

    if parsed_args.command == "simulate":
        return handle_simulate(parsed_args)

    if parsed_args.command in ("build-workloads", "build-workload"):
        return handle_build_workloads(parsed_args)

    if parsed_args.command == "build-dataset":
        return handle_build_dataset(parsed_args)

    if parsed_args.command == "build-pid-data":
        return handle_build_pid_data(parsed_args)

    if parsed_args.command == "train-pid-gbdt":
        return handle_train_pid_gbdt(parsed_args)

    if parsed_args.command == "train-pid-nn":
        return handle_train_pid_nn(parsed_args)

    if parsed_args.command == "bench-latency":
        return handle_bench_latency(parsed_args)

    if parsed_args.command == "train-tree":
        return handle_train_tree(parsed_args)

    if parsed_args.command == "recommend":
        return handle_recommend(parsed_args)

    if parsed_args.command == "train-rl":
        return handle_train_rl(parsed_args)

    if parsed_args.command == "evaluate":
        return handle_evaluate(parsed_args)

    if parsed_args.command == "compare-all":
        return handle_compare_all(parsed_args)

    print(
        f"Command '{parsed_args.command}' received. Implementation active in subsequent stages."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
