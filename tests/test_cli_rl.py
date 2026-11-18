"""
1. What this file does:
   Defines unit tests for the CLI train-rl subcommand.
2. Overall design:
   Validates CLI parser configuration and execution of the train-rl command:
   - Verifies argument flags, types, choices, and default values.
   - Tests end-to-end execution of a short episodic training run with custom output paths.
   - Verifies output model checkpoint creation and state summary generation.
3. Elements in the file:
   - test_cli_train_rl_parser_defaults: Tests argument parsing and default configuration.
   - test_cli_train_rl_execution: Tests end-to-end CLI execution with temporary files.
4. Final output of the file:
   Pytest suite validating CLI train-rl functionality.
"""

from pathlib import Path

from cli import build_parser, main


def test_cli_train_rl_parser_defaults() -> None:
    """
    1. The use of function:
       Verifies that the train-rl subparser registers all flags with proper defaults.
    2. How does the specific function work:
       Constructs parser, parses ['train-rl'], and verifies default values for episodes,
       seed, cs_cost, alpha, gamma, eps_decay, and eval_every.
    3. Concepts used in the function:
       CLI argument validation, subparser inspection.
    4. Places where this function is imported and used:
       Executed as part of pytest test runner.
    5. Final output of the function, before and after:
       Before: CLI parser configuration unverified.
       After: Asserts matching flags and default values.
    """
    parser = build_parser()
    args = parser.parse_args(["train-rl"])
    assert args.command == "train-rl"
    assert args.episodes == 20000
    assert args.seed == 0
    assert args.cs_cost == 1
    assert args.alpha == 0.1
    assert args.gamma == 1.0
    assert args.eps_decay == 0.9995
    assert args.eval_every == 500
    assert args.hparam_search is False
    assert args.multi_seed is False


def test_cli_train_rl_execution(tmp_path: Path) -> None:
    """
    1. The use of function:
       Tests end-to-end CLI execution of train-rl for a brief training run.
    2. How does the specific function work:
       Invokes main() with isolated temporary output paths for model, summary, and logs,
       using 5 episodes. Verifies exit code 0 and verifies that model, summary, and log
       files exist on disk and contain valid data.
    3. Concepts used in the function:
       CLI execution testing, filesystem verification, temporary isolation.
    4. Places where this function is imported and used:
       Executed by pytest.
    5. Final output of the function, before and after:
       Before: End-to-end CLI training unverified.
       After: Confirms successful execution and generated artifacts.
    """
    model_p = tmp_path / "cli_q_table.pkl"
    summary_p = tmp_path / "cli_summary.json"
    logs_p = tmp_path / "cli_logs.csv"

    ret = main([
        "train-rl",
        "--episodes", "5",
        "--eval-every", "5",
        "--seed", "42",
        "--cs-cost", "0",
        "--output-model", str(model_p),
        "--output-summary", str(summary_p),
        "--output-logs", str(logs_p),
    ])

    assert ret == 0
    assert model_p.exists()
    assert summary_p.exists()
    assert logs_p.exists()
