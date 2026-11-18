"""
1. What this file does:
   Provides basic smoke and scaffolding tests to verify repo structure, imports, and CLI config.
2. Overall design:
   Validates that core project modules (schedsim, config, cli) can be imported without errors
   and verifies that CLI argument parsing behaves as expected for Stage 0.
3. Elements in the file:
   - test_imports: Verifies core package and configuration imports.
   - test_cli_help: Verifies CLI parser construction and subcommand discovery.
4. Final output of the file:
   Pytest assertion passes confirming functional environment and scaffolding setup.
"""

import argparse

import config
import schedsim
from cli import build_parser


def test_imports() -> None:
    """
    1. The use of function:
       Validates that schedsim package and config constants can be loaded into memory.
    2. How does the specific function work:
       Checks package version and validates that config constants exist and have valid values.
    3. Concepts used in the function:
       Module importing, basic assertion testing.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested imports.
       After: Asserts package version is '0.1.0' and TIME_QUANTUM > 0.
    """
    assert schedsim.__version__ == "0.1.0"
    assert config.TIME_QUANTUM == 2
    assert config.MAX_PROCESSES == 12


def test_cli_parser() -> None:
    """
    1. The use of function:
       Verifies that the CLI argument parser constructs with all planned subcommands.
    2. How does the specific function work:
       Calls build_parser() and inspects the subparser choices for expected subcommands.
    3. Concepts used in the function:
       CLI argument validation, subparser inspection.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Unverified CLI parser.
       After: Asserts all required subcommands exist in the parser.
    """
    parser = build_parser()
    subparsers_actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparsers_actions) == 1
    choices = subparsers_actions[0].choices
    for cmd in ["simulate", "compare", "build-dataset", "train-tree", "train-rl", "evaluate"]:
        assert cmd in choices
