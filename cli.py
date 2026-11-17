"""
1. What this file does:
   Provides the Command Line Interface (CLI) entry point for the CPU Scheduling Simulator,
   orchestrating simulation runs, policy comparisons, workload generation, model training (Decision Tree and Q-learning),
   and evaluation benchmarks.
2. Overall design:
   Built on Python's standard `argparse` library using subcommands. Each subcommand represents
   a key project capability (simulate, compare, build-dataset, train-tree, train-rl, evaluate),
   routing CLI arguments to the respective domain, ML, or RL modules.
3. Elements in the file:
   - build_parser: Function constructing the CLI argument parser and defining subcommands.
   - main: Main execution function parsing CLI arguments and executing requested actions.
4. Final output of the file:
   Parses command-line invocations, displays help/documentation, and executes requested simulation or training workflows.
"""

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    """
    1. The use of function:
       Constructs and configures the top-level argument parser and subcommands for the schedsim CLI.
    2. How does the specific function work:
       Instantiates `argparse.ArgumentParser`, attaches global descriptive help, and creates subparsers
       for 'simulate', 'compare', 'build-dataset', 'train-tree', 'train-rl', and 'evaluate' subcommands
       with their respective flags and arguments.
    3. Concepts used in the function:
       Declarative CLI parsing, subcommand dispatching, help message formatting, and optional argument routing.
    4. Places where this function is imported and used:
       Called directly inside `main()` in `cli.py` and importable by test suites to verify CLI argument grammar.
    5. Final output of the function, before and after:
       Before: No parser configured.
       After: Returns a fully initialized `argparse.ArgumentParser` instance ready to parse sys.argv.
    """
    parser = argparse.ArgumentParser(
        prog="schedsim",
        description="AI-Driven CPU Scheduling Simulator (FCFS, SJF, SRTF, RR, Decision Tree, Q-learning)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # simulate subcommand
    sim_parser = subparsers.add_parser("simulate", help="Run a simulation for a specific policy")
    sim_parser.add_argument("--policy", type=str, choices=["FCFS", "SJF", "SRTF", "RR"], help="Scheduling policy to simulate")
    sim_parser.add_argument("--preset", type=str, default="1", help="Preset workload identifier or name")
    sim_parser.add_argument("--quantum", type=int, default=2, help="Time quantum for Round Robin")
    sim_parser.add_argument("--cs-cost", type=int, default=0, help="Context switch cost in ticks")

    # compare subcommand
    comp_parser = subparsers.add_parser("compare", help="Compare multiple scheduling policies on a workload")
    comp_parser.add_argument("--preset", type=str, default="1", help="Preset workload identifier or name")
    comp_parser.add_argument("--cs-cost", type=int, default=0, help="Context switch cost in ticks")

    # build-dataset subcommand
    data_parser = subparsers.add_parser("build-dataset", help="Generate synthetic workloads and labeled dataset")
    data_parser.add_argument("--n", type=int, default=6000, help="Number of workloads to generate")
    data_parser.add_argument("--output", type=str, default="data/workloads.csv", help="Output path for workloads CSV")

    # train-tree subcommand
    tree_parser = subparsers.add_parser("train-tree", help="Train the Decision Tree policy recommender")
    tree_parser.add_argument("--max-depth", type=int, default=4, help="Maximum depth of the decision tree")

    # train-rl subcommand
    rl_parser = subparsers.add_parser("train-rl", help="Train the tabular Q-learning scheduling agent")
    rl_parser.add_argument("--episodes", type=int, default=20000, help="Number of training episodes")
    rl_parser.add_argument("--seed", type=int, default=0, help="Random seed for training")

    # evaluate subcommand
    eval_parser = subparsers.add_parser("evaluate", help="Run full evaluation across held-out test workloads")
    eval_parser.add_argument("--split", type=str, choices=["val", "test"], default="test", help="Evaluation split to run")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """
    1. The use of function:
       Entry point execution for the CLI application.
    2. How does the specific function work:
       Invokes `build_parser()`, parses provided arguments (or `sys.argv[1:]` by default),
       verifies that a subcommand was provided, and prints appropriate feedback or dispatches execution.
    3. Concepts used in the function:
       CLI argument validation, sub-command handling, exit code management, error handling.
    4. Places where this function is imported and used:
       Invoked when `cli.py` is executed as `__main__`, or called by end-to-end integration tests.
    5. Final output of the function, before and after:
       Before: Raw command line input strings.
       After: Command executed, returns integer process exit code (0 for success, non-zero for error/help).
    """
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    print(f"Command '{parsed_args.command}' received. Implementation active in subsequent stages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
