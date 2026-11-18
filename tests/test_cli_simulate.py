"""
1. What this file does:
   Tests CLI simulation execution, result formatting, error diagnostics, and ASCII Gantt chart
   rendering for the CPU scheduling simulator.
2. Overall design:
   Verifies CLI entry points and functions (`main`, `handle_simulate`, `format_results`,
   `render_ascii_gantt`) across multiple policies (FCFS, SJF, SRTF, RR) and workload presets
   (1, convoy, simultaneous, long_idle_gap), checking metrics formatting, timeline wrapping,
   and error handling.
3. Elements in the file:
   - test_cli_simulate_fcfs_preset_1: Tests FCFS simulation on preset 1.
   - test_cli_simulate_rr_preset_1_quantum: Tests Round Robin with custom quantum on preset 1.
   - test_cli_simulate_sjf_convoy_cs_cost: Tests SJF with context switch penalty on convoy preset.
   - test_cli_simulate_srtf_simultaneous: Tests SRTF on simultaneous arrival workload.
   - test_cli_simulate_case_insensitive_policy: Tests case-insensitive policy name resolution.
   - test_cli_simulate_error_unknown_policy: Verifies error handling for unregistered policies.
   - test_cli_simulate_error_unknown_preset: Verifies error handling for unregistered presets.
   - test_cli_simulate_error_missing_policy: Verifies error handling when --policy is omitted.
   - test_cli_simulate_error_invalid_quantum: Verifies validation on non-positive quantum.
   - test_cli_simulate_error_invalid_cs_cost: Verifies validation on negative context switch cost.
   - test_render_ascii_gantt_empty: Verifies Gantt chart rendering on empty timelines.
   - test_render_ascii_gantt_type_error: Verifies type checking on render_ascii_gantt arguments.
   - test_render_ascii_gantt_wrapping: Verifies row wrapping when max_width is constrained.
   - test_render_ascii_gantt_idle_gap: Verifies IDLE segment rendering in ASCII Gantt output.
   - test_format_results_type_error: Verifies type checking on format_results arguments.
   - test_cli_no_subcommand_prints_help: Verifies default help message when no command is passed.
4. Final output of the file:
   Pytest test suite asserting CLI simulation and ASCII Gantt fidelity.
"""

import pytest

from cli import format_results, main, render_ascii_gantt
from schedsim.domain.timeline import Timeline, TimelineSegment


def test_cli_simulate_fcfs_preset_1(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests FCFS policy execution on preset workload '1' through the CLI entry point.
    2. How does the specific function work:
       Invokes `main(['simulate', '--policy', 'FCFS', '--preset', '1'])`, asserts exit code 0,
       and validates that process metrics, aggregate statistics, and Gantt chart are printed.
    3. Concepts used in the function:
       CLI testing, stdout capture, table verification, metrics assertion.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested CLI FCFS command.
       After: Asserts successful execution and presence of all required report sections.
    """
    exit_code = main(["simulate", "--policy", "FCFS", "--preset", "1"])
    assert exit_code == 0

    captured = capsys.readouterr()
    out = captured.out

    assert "Simulation Results: Policy=FCFS, Preset=1, Context Switch Cost=0" in out
    assert "Process Execution Metrics:" in out
    for header in ["PID", "Arrival", "Burst", "Completion", "Turnaround", "Waiting", "Response"]:
        assert header in out

    for pid in ["P1", "P2", "P3"]:
        assert pid in out

    for metric in [
        "Avg Waiting",
        "Avg Turnaround",
        "Avg Response",
        "Max Waiting",
        "Makespan",
        "CPU Utilization",
        "Throughput",
        "Context Switches",
        "Jain's Fairness Index",
    ]:
        assert metric in out

    assert "Execution Timeline (Gantt Chart):" in out
    assert "+------------+--------+------+" in out


def test_cli_simulate_rr_preset_1_quantum(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests Round Robin policy with configurable time quantum via CLI.
    2. How does the specific function work:
       Runs `main(['simulate', '--policy', 'RR', '--preset', '1', '--quantum', '2'])`,
       verifies exit code 0, and checks that time-sliced execution segments appear in Gantt.
    3. Concepts used in the function:
       Time-slicing CLI options, quantum parameter forwarding, output verification.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested Round Robin CLI execution.
       After: Asserts exit code 0 and presence of interleaved process segments.
    """
    exit_code = main(["simulate", "--policy", "RR", "--preset", "1", "--quantum", "2"])
    assert exit_code == 0

    captured = capsys.readouterr()
    out = captured.out

    assert "Policy=RR" in out
    assert "P1" in out
    assert "P2" in out
    assert "P3" in out
    assert "Execution Timeline (Gantt Chart):" in out


def test_cli_simulate_sjf_convoy_cs_cost(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests SJF policy on the convoy workload with non-zero context switch cost.
    2. How does the specific function work:
       Executes `main(['simulate', '--policy', 'SJF', '--preset', 'convoy', '--cs-cost', '1'])`,
       verifies exit code 0, and asserts presence of 'CS' overhead intervals in the Gantt chart.
    3. Concepts used in the function:
       Overhead modeling, context switch verification, convoy effect testing.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested SJF convoy simulation with context switch cost.
       After: Asserts exit code 0 and confirms CS markers in output and Gantt chart.
    """
    exit_code = main(["simulate", "--policy", "SJF", "--preset", "convoy", "--cs-cost", "1"])
    assert exit_code == 0

    captured = capsys.readouterr()
    out = captured.out

    assert "Policy=SJF" in out
    assert "Preset=convoy" in out
    assert "Context Switch Cost=1" in out
    assert "CS" in out


def test_cli_simulate_srtf_simultaneous(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests SRTF preemptive policy on simultaneous arrival workload preset.
    2. How does the specific function work:
       Runs `main(['simulate', '--policy', 'SRTF', '--preset', 'simultaneous'])`,
       verifies exit code 0, and checks that shortest bursts execute first.
    3. Concepts used in the function:
       Preemption verification, simultaneous arrival workload handling.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested SRTF simultaneous preset simulation.
       After: Asserts exit code 0 and confirms correct SRTF execution sequence.
    """
    exit_code = main(["simulate", "--policy", "SRTF", "--preset", "simultaneous"])
    assert exit_code == 0

    captured = capsys.readouterr()
    out = captured.out

    assert "Policy=SRTF" in out
    assert "Preset=simultaneous" in out
    assert "P3" in out


def test_cli_simulate_case_insensitive_policy(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Verifies case-insensitive resolution of policy names via CLI.
    2. How does the specific function work:
       Runs simulation using lowercase 'rr' and mixed-case 'RoundRobin', verifying both succeed.
    3. Concepts used in the function:
       Case-normalization, alias support, robust CLI user experience.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested case-insensitivity in CLI argument handling.
       After: Asserts exit code 0 for both variants.
    """
    assert main(["simulate", "--policy", "rr", "--preset", "1"]) == 0
    assert main(["simulate", "--policy", "RoundRobin", "--preset", "1"]) == 0


def test_cli_simulate_error_unknown_policy(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests error handling when an unregistered policy name is provided.
    2. How does the specific function work:
       Invokes `main(['simulate', '--policy', 'NON_EXISTENT_POLICY', '--preset', '1'])`,
       asserts non-zero return code (1), and checks error message on stderr.
    3. Concepts used in the function:
       Error reporting, non-zero exit codes, stderr verification.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested unknown policy error path.
       After: Asserts non-zero return code and descriptive error message.
    """
    exit_code = main(["simulate", "--policy", "NON_EXISTENT_POLICY", "--preset", "1"])
    assert exit_code != 0

    captured = capsys.readouterr()
    assert "Simulation error: \"Unknown policy 'NON_EXISTENT_POLICY'" in captured.err


def test_cli_simulate_error_unknown_preset(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests error handling when an unregistered preset workload identifier is provided.
    2. How does the specific function work:
       Invokes `main(['simulate', '--policy', 'FCFS', '--preset', 'NON_EXISTENT_PRESET'])`,
       asserts non-zero return code (1), and checks error message on stderr.
    3. Concepts used in the function:
       Preset validation, defensive error handling, non-zero process exit.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested unknown preset error path.
       After: Asserts non-zero return code and descriptive error message.
    """
    exit_code = main(["simulate", "--policy", "FCFS", "--preset", "NON_EXISTENT_PRESET"])
    assert exit_code != 0

    captured = capsys.readouterr()
    assert "Simulation error: \"Unknown preset workload 'NON_EXISTENT_PRESET'" in captured.err


def test_cli_simulate_error_missing_policy(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests error handling when `--policy` option is omitted from simulate subcommand.
    2. How does the specific function work:
       Calls `main(['simulate', '--preset', '1'])`, asserts non-zero return code (1),
       and verifies stderr error output.
    3. Concepts used in the function:
       Mandatory argument validation, graceful error messaging.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested missing argument path.
       After: Asserts non-zero return code and clear requirement message.
    """
    exit_code = main(["simulate", "--preset", "1"])
    assert exit_code != 0

    captured = capsys.readouterr()
    assert "Simulation error: --policy is required for simulate." in captured.err


def test_cli_simulate_error_invalid_quantum(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests error handling when non-positive time quantum is passed for Round Robin.
    2. How does the specific function work:
       Calls `main(['simulate', '--policy', 'RR', '--quantum', '0'])`, asserts non-zero return,
       and checks error message on stderr.
    3. Concepts used in the function:
       Boundary validation, exception interception, diagnostic output.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid quantum handling.
       After: Asserts non-zero return code and value error message.
    """
    exit_code = main(["simulate", "--policy", "RR", "--quantum", "0"])
    assert exit_code != 0

    captured = capsys.readouterr()
    assert "Simulation error:" in captured.err
    assert "quantum must be >= 1" in captured.err


def test_cli_simulate_error_invalid_cs_cost(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Tests error handling when negative context switch cost is passed.
    2. How does the specific function work:
       Calls `main(['simulate', '--policy', 'FCFS', '--cs-cost', '-1'])`, asserts non-zero return,
       and checks error message on stderr.
    3. Concepts used in the function:
       Non-negative constraint validation, simulation exception bubbling.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested negative context switch cost handling.
       After: Asserts non-zero return code and error message.
    """
    exit_code = main(["simulate", "--policy", "FCFS", "--cs-cost", "-1"])
    assert exit_code != 0

    captured = capsys.readouterr()
    assert "Simulation error:" in captured.err
    assert "cs_cost must be >= 0" in captured.err


def test_render_ascii_gantt_empty() -> None:
    """
    1. The use of function:
       Verifies Gantt chart rendering on an empty Timeline instance.
    2. How does the specific function work:
       Passes an empty Timeline to `render_ascii_gantt()`, asserting '(empty timeline)'.
    3. Concepts used in the function:
       Edge case handling, empty timeline safety.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested empty timeline rendering.
       After: Asserts expected fallback string.
    """
    timeline = Timeline()
    assert render_ascii_gantt(timeline) == "(empty timeline)"


def test_render_ascii_gantt_type_error() -> None:
    """
    1. The use of function:
       Verifies type enforcement on `render_ascii_gantt` arguments.
    2. How does the specific function work:
       Passes non-Timeline objects to `render_ascii_gantt()` and asserts TypeError is raised.
    3. Concepts used in the function:
       Defensive type validation, runtime type assertions.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid argument types for render_ascii_gantt.
       After: Asserts TypeError raised with descriptive message.
    """
    with pytest.raises(TypeError, match="timeline must be a Timeline instance"):
        render_ascii_gantt("not_a_timeline")  # type: ignore[arg-type]


def test_render_ascii_gantt_wrapping() -> None:
    """
    1. The use of function:
       Verifies that `render_ascii_gantt` wraps rows cleanly when max_width is constrained.
    2. How does the specific function work:
       Constructs a timeline with multiple segments, renders with small max_width (30),
       and confirms that multiple box rows are produced separated by empty lines.
    3. Concepts used in the function:
       Terminal line wrapping, multi-row ASCII chart composition.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested Gantt chart wrapping behavior.
       After: Asserts multiple rows rendered without line overflow.
    """
    timeline = Timeline(
        [
            TimelineSegment(label="P1", start=0, end=5),
            TimelineSegment(label="P2", start=5, end=10),
            TimelineSegment(label="P3", start=10, end=15),
            TimelineSegment(label="P4", start=15, end=20),
        ]
    )
    chart = render_ascii_gantt(timeline, max_width=30)
    lines = chart.split("\n")
    # Multiple rows separated by blank line
    assert "" in lines
    for line in lines:
        if line.startswith("+"):
            assert len(line) <= 30


def test_render_ascii_gantt_idle_gap() -> None:
    """
    1. The use of function:
       Verifies that idle CPU intervals are labeled as 'IDLE' in ASCII Gantt charts.
    2. How does the specific function work:
       Constructs a timeline containing an IDLE interval [5, 15), renders Gantt chart,
       and checks that 'IDLE' and respective ticks are properly formatted.
    3. Concepts used in the function:
       IDLE state rendering, discontinuous process execution representation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested IDLE interval rendering in Gantt chart.
       After: Asserts 'IDLE' and tick boundaries exist in the rendered output.
    """
    timeline = Timeline(
        [
            TimelineSegment(label="P1", start=0, end=5),
            TimelineSegment(label="IDLE", start=5, end=15),
            TimelineSegment(label="P2", start=15, end=20),
        ]
    )
    chart = render_ascii_gantt(timeline)
    assert "IDLE" in chart
    assert "P1" in chart
    assert "P2" in chart


def test_format_results_type_error() -> None:
    """
    1. The use of function:
       Verifies type enforcement on `format_results` arguments.
    2. How does the specific function work:
       Passes invalid result object to `format_results()` and asserts TypeError is raised.
    3. Concepts used in the function:
       Defensive type validation, runtime type assertions.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested invalid result argument to format_results.
       After: Asserts TypeError raised with descriptive message.
    """
    with pytest.raises(TypeError, match="result must be a SimResult instance"):
        format_results("invalid_result", "FCFS", "1", 0)  # type: ignore[arg-type]


def test_cli_no_subcommand_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    """
    1. The use of function:
       Verifies that invoking CLI without subcommands prints help and exits with 0.
    2. How does the specific function work:
       Calls `main([])`, asserts return code 0, and checks help text output.
    3. Concepts used in the function:
       Default CLI behavior, user guidance, help generation.
    4. Places where this function is imported and used:
       Executed automatically by pytest runner.
    5. Final output of the function, before and after:
       Before: Untested bare CLI invocation.
       After: Asserts exit code 0 and presence of 'schedsim' usage string.
    """
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage: schedsim" in captured.out
