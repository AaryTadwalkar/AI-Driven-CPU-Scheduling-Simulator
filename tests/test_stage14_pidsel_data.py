"""
1. What this file does:
   Comprehensive unit and acceptance test suite for Stage 14 process-level candidate feature
   extraction, dataset construction, and analytical report generation.
2. Overall design:
   Validates the 13 feature definitions, pure NumPy normalized ranking with tie-breaking,
   observable feature extraction, lack of NaNs/infinities, permutation equivariance, neural batch
   padding, exclusion of single-candidate trivial states, deterministic seed reproducibility,
   S0 SRTF dominance, S1 continuation avoidance, and Markdown report structural integrity.
3. Elements in the file:
   - test_feature_names_tuple_and_dimension: Asserts exactly 13 canonical feature names in order.
   - test_compute_normalized_rank_mathematical_properties: Validates rank formulas and boundaries.
   - test_extract_candidate_features_no_nan_or_inf: Verifies numerical stability on diverse states.
   - test_extract_candidate_features_permutation_equivariance: Tests invariant row permutation.
   - test_pad_and_stack_features_tensor_padding: Verifies shape, masking, and padding values.
   - test_build_pid_dataset_no_single_candidate_states: Enforces len(ready) >= 2 on all states.
   - test_build_pid_dataset_seed_reproducibility: Verifies identical array outputs from same seeds.
   - test_s0_oracle_srtf_dominance_in_dataset: Asserts S0 oracle picks shortest remaining job.
   - test_s1_oracle_balances_continuation: Asserts S1 oracle balances continuation vs preemption.
   - test_generate_dataset_report_content_and_structure: Validates Markdown report sections.
   - test_input_validation_and_error_handling: Checks defensive TypeError and ValueError guards.
   - test_cli_build_pid_data_command: Tests end-to-end `build-pid-data` CLI command execution.
4. Final output of the file:
   Passes pytest test suite 100% without warnings or ruff violations.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cli import main
from schedsim.domain.engine import ProcView, View
from schedsim.domain.process import Process
from schedsim.pidsel.dataset import (
    _analyze_dataset,
    build_pid_dataset,
    generate_dataset_report,
)
from schedsim.pidsel.features import (
    FEATURE_NAMES,
    compute_normalized_rank,
    extract_candidate_features,
    pad_and_stack_features,
)
from schedsim.workloads.splits import generate_split_workloads


def test_feature_names_tuple_and_dimension() -> None:
    """
    1. The use of function:
       Validates that FEATURE_NAMES defines exactly the 13 specified features in canonical order.
    2. How does the specific function work:
       Checks tuple type, exact length of 13, and exact match against expected feature name strings.
    3. Concepts used in the function:
       Constant integrity assertion, schema validation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Unverified FEATURE_NAMES constant.
       After: Confirms FEATURE_NAMES contains exactly the 13 required names.
    """
    expected = (
        "rem",
        "burst",
        "rem_burst_ratio",
        "waited",
        "is_last",
        "rank_rem",
        "rank_waited",
        "rem_min_ratio",
        "rem_mean_ratio",
        "n_ready",
        "sum_rem",
        "cs_cost",
        "pending_arrivals",
    )
    assert isinstance(FEATURE_NAMES, tuple)
    assert len(FEATURE_NAMES) == 13
    assert FEATURE_NAMES == expected


def test_compute_normalized_rank_mathematical_properties() -> None:
    """
    1. The use of function:
       Tests the pure NumPy normalized fractional ranking function under various boundary cases.
    2. How does the specific function work:
       Tests empty arrays, single elements, strictly increasing values, tied values (all tied,
       partial ties), and asserts outputs fall strictly within [0.0, 1.0] with symmetric midpoints.
    3. Concepts used in the function:
       Fractional ranking, symmetric tie handling, boundary normalization.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Raw input arrays of varying configurations.
       After: Asserts computed ranks match exact algebraic expectations.
    """
    # Empty array
    assert len(compute_normalized_rank(np.array([]))) == 0

    # Single element
    single = compute_normalized_rank(np.array([42.0]))
    assert np.allclose(single, [0.0])

    # Distinct values
    distinct = compute_normalized_rank(np.array([10.0, 20.0, 30.0]))
    assert np.allclose(distinct, [0.0, 0.5, 1.0])

    # Two tied values
    tied_2 = compute_normalized_rank(np.array([5.0, 5.0]))
    assert np.allclose(tied_2, [0.5, 0.5])

    # Three tied values
    tied_3 = compute_normalized_rank(np.array([7.0, 7.0, 7.0]))
    assert np.allclose(tied_3, [0.5, 0.5, 0.5])

    # Partial ties: [1, 2, 2, 5]
    # For 1: 0 less, 1 equal -> 0.0
    # For 2: 1 less, 2 equal -> (1 + 0.5 * 1) / 3 = 0.5
    # For 5: 3 less, 1 equal -> 3.0 / 3 = 1.0
    partial = compute_normalized_rank(np.array([1.0, 2.0, 2.0, 5.0]))
    assert np.allclose(partial, [0.0, 0.5, 0.5, 1.0])

    # Bounds check
    random_vals = np.array([9.0, 1.0, 4.0, 7.0, 1.0, 9.0])
    ranks = compute_normalized_rank(random_vals)
    assert (ranks >= 0.0).all() and (ranks <= 1.0).all()


def test_extract_candidate_features_no_nan_or_inf() -> None:
    """
    1. The use of function:
       Tests that extract_candidate_features produces no NaNs or infinities across diverse states.
    2. How does the specific function work:
       Constructs synthetic views with edge-case attributes (zero waited time, burst equal to
       remaining, single candidate, multiple candidates) and verifies finite float32 outputs.
    3. Concepts used in the function:
       Numerical stability verification, float finite checks.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Synthetic View instances with boundary values.
       After: Confirms shape (n, 13) and np.isfinite(features).all().
    """
    # Empty view
    empty_view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    empty_feats, empty_pids = extract_candidate_features(empty_view)
    assert empty_feats.shape == (0, 13)
    assert empty_pids == []

    # Ready processes with zero waited time and equal remaining
    p1 = ProcView(pid="P1", arrival=0, burst=10, remaining=10, waited=0)
    p2 = ProcView(pid="P2", arrival=0, burst=5, remaining=5, waited=0)
    view = View(time=0, ready=(p1, p2), last_pid=None, newly_arrived=(), pending_arrivals=4)

    feats, pids = extract_candidate_features(view, cs_cost=1)
    assert feats.shape == (2, 13)
    assert feats.dtype == np.float32
    assert pids == ["P1", "P2"]
    assert np.isfinite(feats).all()
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

    # Check is_last column (feature index 4)
    view_with_last = View(time=1, ready=(p1, p2), last_pid="P1", newly_arrived=())
    feats_last, _ = extract_candidate_features(view_with_last, cs_cost=0)
    assert feats_last[0, 4] == 1.0
    assert feats_last[1, 4] == 0.0

    # Check pending_arrivals capping at 10 (feature index 12)
    view_pending_25 = View(
        time=2,
        ready=(p1, p2),
        last_pid=None,
        newly_arrived=(),
        pending_arrivals=25,
    )
    feats_pending, _ = extract_candidate_features(view_pending_25)
    assert feats_pending[0, 12] == 10.0
    assert feats_pending[1, 12] == 10.0


def test_extract_candidate_features_permutation_equivariance() -> None:
    """
    1. The use of function:
       Verifies that shuffling the ready queue permutes the feature rows by the exact same order.
    2. How does the specific function work:
       Creates a View with 4 distinct ready processes. Generates several permutations of the ready
       tuple, extracts features from each permuted view, and asserts features[perm] == permuted.
    3. Concepts used in the function:
       Permutation equivariance, algebraic symmetry testing.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Canonical and permuted View instances.
       After: Confirms max absolute difference is 0.0 across all permutations.
    """
    p1 = ProcView(pid="P1", arrival=0, burst=10, remaining=8, waited=2)
    p2 = ProcView(pid="P2", arrival=1, burst=6, remaining=3, waited=5)
    p3 = ProcView(pid="P3", arrival=2, burst=12, remaining=12, waited=0)
    p4 = ProcView(pid="P4", arrival=3, burst=4, remaining=1, waited=7)
    candidates = [p1, p2, p3, p4]

    canonical_view = View(
        time=10,
        ready=tuple(candidates),
        last_pid="P2",
        newly_arrived=(),
        pending_arrivals=3,
    )
    canonical_feats, canonical_pids = extract_candidate_features(canonical_view, cs_cost=1)
    assert canonical_pids == ["P1", "P2", "P3", "P4"]

    permutations = [
        [1, 2, 3, 0],
        [3, 0, 2, 1],
        [2, 3, 1, 0],
        [3, 2, 1, 0],
    ]

    for perm in permutations:
        permuted_ready = tuple(candidates[i] for i in perm)
        permuted_view = View(
            time=10,
            ready=permuted_ready,
            last_pid="P2",
            newly_arrived=(),
            pending_arrivals=3,
        )
        permuted_feats, permuted_pids = extract_candidate_features(permuted_view, cs_cost=1)

        expected_pids = [canonical_pids[i] for i in perm]
        assert permuted_pids == expected_pids

        expected_feats = canonical_feats[perm]
        diff = np.max(np.abs(permuted_feats - expected_feats))
        assert diff < 1e-6, f"Permutation equivariance violated for perm {perm}: diff {diff}"


def test_pad_and_stack_features_tensor_padding() -> None:
    """
    1. The use of function:
       Tests pad_and_stack_features for tensor padding and boolean validity mask creation.
    2. How does the specific function work:
       Passes a batch of 2D feature matrices with lengths 2, 4, and 3. Verifies resulting 3D tensor
       shape, boolean mask values, zero-padding in non-valid slots, and max_candidates override.
    3. Concepts used in the function:
       Tensor padding, attention masking, neural batch preparation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Sequence of candidate feature matrices of varying row counts.
       After: Confirms uniform shape (B, max_c, 13) and exact boolean mask alignment.
    """
    d = len(FEATURE_NAMES)
    f1 = np.ones((2, d), dtype=np.float32)
    f2 = np.ones((4, d), dtype=np.float32) * 2.0
    f3 = np.ones((3, d), dtype=np.float32) * 3.0

    batch = [f1, f2, f3]
    padded, mask = pad_and_stack_features(batch)

    assert padded.shape == (3, 4, d)
    assert mask.shape == (3, 4)
    assert mask.dtype == bool

    # Mask validation
    assert mask[0].tolist() == [True, True, False, False]
    assert mask[1].tolist() == [True, True, True, True]
    assert mask[2].tolist() == [True, True, True, False]

    # Content validation
    assert np.allclose(padded[0, :2, :], 1.0)
    assert np.allclose(padded[0, 2:, :], 0.0)
    assert np.allclose(padded[1, :4, :], 2.0)
    assert np.allclose(padded[2, :3, :], 3.0)
    assert np.allclose(padded[2, 3:, :], 0.0)

    # Custom max_candidates
    padded_custom, mask_custom = pad_and_stack_features(batch, max_candidates=6)
    assert padded_custom.shape == (3, 6, d)
    assert mask_custom.shape == (3, 6)
    assert mask_custom[0].tolist() == [True, True, False, False, False, False]

    # Error when max_candidates is smaller than observed
    with pytest.raises(ValueError, match="smaller than maximum observed"):
        pad_and_stack_features(batch, max_candidates=2)


def test_build_pid_dataset_no_single_candidate_states() -> None:
    """
    1. The use of function:
       Validates that build_pid_dataset never records single-candidate trivial states.
    2. How does the specific function work:
       Generates a small batch of synthetic workloads, runs build_pid_dataset, and asserts that
       for every state, mask.sum() >= 2. Also asserts regret of winning candidate is strictly 0.0.
    3. Concepts used in the function:
       Non-forced decision filtering, behavioral cloning data integrity.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Workload definitions with mixture of 1-candidate and multi-candidate steps.
       After: Confirms every row in dataset contains at least 2 ready candidates.
    """
    workloads = generate_split_workloads("train", n_workloads=15)
    ds = build_pid_dataset(workloads=workloads, cs_cost=0, max_states=200, n_workers=1)

    features = ds["features"]
    mask = ds["mask"]
    costs = ds["costs"]
    regrets = ds["regrets"]
    best_idx = ds["best_candidate_idx"]
    oracle_pids = ds["oracle_pids"]

    n_states = len(best_idx)
    assert n_states > 0
    assert features.shape[0] == n_states
    assert mask.shape[0] == n_states
    assert costs.shape[0] == n_states
    assert regrets.shape[0] == n_states
    assert len(oracle_pids) == n_states

    for i in range(n_states):
        n_candidates = int(mask[i].sum())
        assert n_candidates >= 2, f"State {i} has only {n_candidates} candidates (expected >= 2)."

        b = int(best_idx[i])
        assert 0 <= b < n_candidates
        assert regrets[i, b] == 0.0
        assert costs[i, b] == np.min(costs[i, :n_candidates])
        assert np.isfinite(features[i, :n_candidates, :]).all()


def test_build_pid_dataset_seed_reproducibility() -> None:
    """
    1. The use of function:
       Verifies bitwise reproducibility when building datasets from identical workload seeds.
    2. How does the specific function work:
       Generates 8 workloads twice from the same seed partition. Builds datasets independently
       and asserts identical feature matrices, masks, costs, regrets, and winning indices.
    3. Concepts used in the function:
       Deterministic simulation, reproducible serialization.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Two independently simulated dataset runs on identical workloads.
       After: Confirms np.array_equal across all dataset tensor fields.
    """
    ws1 = generate_split_workloads("train", n_workloads=8, seed_offset=50)
    ws2 = generate_split_workloads("train", n_workloads=8, seed_offset=50)

    ds1 = build_pid_dataset(ws1, cs_cost=1, max_states=100, n_workers=1)
    ds2 = build_pid_dataset(ws2, cs_cost=1, max_states=100, n_workers=1)

    assert ds1["meta"] == ds2["meta"]
    assert np.array_equal(ds1["features"], ds2["features"])
    assert np.array_equal(ds1["mask"], ds2["mask"])
    assert np.array_equal(ds1["costs"], ds2["costs"])
    assert np.array_equal(ds1["regrets"], ds2["regrets"])
    assert np.array_equal(ds1["best_candidate_idx"], ds2["best_candidate_idx"])
    assert np.array_equal(ds1["oracle_pids"], ds2["oracle_pids"])


def test_s0_oracle_srtf_dominance_in_dataset() -> None:
    """
    1. The use of function:
       Verifies that the rollout oracle at S0 (cs_cost=0) chooses the shortest remaining burst job.
    2. How does the specific function work:
       Builds a dataset with cs_cost=0 on 20 train workloads. Compares the oracle's choice against
       the minimum remaining burst candidate in each state and asserts agreement >= 99%.
    3. Concepts used in the function:
       SRTF optimality theorem validation, empirical policy verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Behavioral cloning dataset for S0.
       After: Asserts SRTF match proportion is >= 99%.
    """
    workloads = generate_split_workloads("train", n_workloads=20)
    ds = build_pid_dataset(workloads=workloads, cs_cost=0, max_states=250, n_workers=2)

    features = ds["features"]
    mask = ds["mask"]
    best_idx = ds["best_candidate_idx"]
    n_states = len(best_idx)

    srtf_matches = 0
    for i in range(n_states):
        k = int(mask[i].sum())
        b = int(best_idx[i])
        rem_vals = features[i, :k, 0]
        if rem_vals[b] == np.min(rem_vals):
            srtf_matches += 1

    srtf_pct = (srtf_matches / n_states) * 100.0
    assert srtf_pct >= 99.0, f"S0 SRTF match rate was {srtf_pct:.2f}% (expected >= 99.0%)."


def test_s1_oracle_balances_continuation() -> None:
    """
    1. The use of function:
       Verifies that at S1 (cs_cost=1), the oracle continues running processes to avoid
       switch costs.
    2. How does the specific function work:
       Constructs a workload where P1 (burst 4) is running at tick 1 (remaining 3) when P2
       (burst 2) arrives. At S0 (cs=0), the oracle preempts P1 for P2 (SRTF choice). At S1
       (cs=1), preempting incurs 2 switch stalls, so the oracle continues P1 to achieve lower
       cost (4 vs 5). Verifies continuation-only rate is 0.0 at S0 and positive at S1.
    3. Concepts used in the function:
       Context switch penalty avoidance, scheduling hysteresis, empirical trade-off verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Comparative dataset runs for S0 and S1 on preemption trade-off workload.
       After: Confirms S1 chooses continuation over preemption, elevating Continuation Only.
    """
    p1 = Process("P1", arrival=0, burst=4, priority=1)
    p2 = Process("P2", arrival=1, burst=2, priority=1)
    workload = [p1, p2]

    ds_s0 = build_pid_dataset([workload], cs_cost=0, n_workers=1)
    ds_s1 = build_pid_dataset([workload], cs_cost=1, n_workers=1)

    with tempfile.TemporaryDirectory() as td:
        p0 = Path(td) / "s0.npz"
        p1 = Path(td) / "s1.npz"
        np.savez_compressed(p0, **ds_s0)
        np.savez_compressed(p1, **ds_s1)

        a0 = _analyze_dataset(p0)
        a1 = _analyze_dataset(p1)

        # At S0, oracle never picks continuation if it is not SRTF
        assert a0["exclusive_pcts"]["Continuation Only"] == 0.0
        # At S1, oracle picks continuation to avoid switch stall penalties
        assert a1["exclusive_pcts"]["Continuation Only"] > 0.0
        # First decision: S0 picks P2 (shorter), S1 picks P1 (continuation)
        assert ds_s0["oracle_pids"][0] == "P2"
        assert ds_s1["oracle_pids"][0] == "P1"


def test_generate_dataset_report_content_and_structure() -> None:
    """
    1. The use of function:
       Tests the automated Markdown dataset report generation and verifies required sections.
    2. How does the specific function work:
       Generates mock/small S0 and S1 NPZ files inside a temporary directory, invokes
       generate_dataset_report, and checks for presence of required Markdown headers, tables,
       and statistical breakdowns.
    3. Concepts used in the function:
       Report automation testing, substring verification, table structure validation.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Temporary NPZ files for S0 and S1.
       After: Verified Markdown report containing all required documentation sections.
    """
    workloads = generate_split_workloads("train", n_workloads=10)
    ds_s0 = build_pid_dataset(workloads, cs_cost=0, max_states=80, n_workers=1)
    ds_s1 = build_pid_dataset(workloads, cs_cost=1, max_states=80, n_workers=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        s0_path = Path(tmpdir) / "pid_bc_S0.npz"
        s1_path = Path(tmpdir) / "pid_bc_S1.npz"
        report_path = Path(tmpdir) / "pid_dataset_report.md"

        np.savez_compressed(s0_path, **ds_s0)
        np.savez_compressed(s1_path, **ds_s1)

        result_path = generate_dataset_report(s0_path, s1_path, report_path)
        assert result_path.exists()
        assert result_path == report_path

        text = report_path.read_text(encoding="utf-8")
        assert "# Process-Level Behavioral Cloning Dataset Report" in text
        assert "## 1. Executive Summary" in text
        assert "## 2. Dataset Dimensions and Candidate Distribution" in text
        assert "## 3. Oracle Dispatch Behavior Breakdown" in text
        assert "## 4. Analytical Findings & Theoretical Grounding" in text
        assert "Shortest Remaining (SRTF)" in text
        assert "Currently Running (Continuation)" in text
        assert "Earliest Arrival (FCFS)" in text
        assert "Mutually Exclusive Decision Partition" in text


def test_input_validation_and_error_handling() -> None:
    """
    1. The use of function:
       Tests defensive validation and error handling across Stage 14 features and dataset APIs.
    2. How does the specific function work:
       Passes invalid types, negative costs, invalid shapes, and non-existent paths to assert
       that appropriate TypeError, ValueError, and FileNotFoundError exceptions are raised.
    3. Concepts used in the function:
       Defensive boundary testing, exception assertion.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Invalid parameter sets.
       After: Asserts expected exceptions raised with informative error messages.
    """
    # compute_normalized_rank 2D error
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        compute_normalized_rank(np.ones((2, 2)))

    # extract_candidate_features type errors
    with pytest.raises(TypeError, match="view must be a View instance"):
        extract_candidate_features("not_a_view")  # type: ignore[arg-type]

    dummy_view = View(time=0, ready=(), last_pid=None, newly_arrived=())
    with pytest.raises(ValueError, match="cs_cost must be a non-negative int"):
        extract_candidate_features(dummy_view, cs_cost=-1)

    with pytest.raises(ValueError, match="pending_arrivals must be a non-negative int"):
        extract_candidate_features(dummy_view, pending_arrivals=-5)

    # pad_and_stack_features type errors
    with pytest.raises(TypeError, match="must be a Sequence"):
        pad_and_stack_features(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a np.ndarray"):
        pad_and_stack_features(["bad_element"])  # type: ignore[list-item]

    # build_pid_dataset type errors
    with pytest.raises(TypeError, match="workloads must be a Sequence"):
        build_pid_dataset("not_a_sequence", cs_cost=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="workloads cannot be empty"):
        build_pid_dataset([], cs_cost=0)

    with pytest.raises(ValueError, match="cs_cost must be a non-negative int"):
        build_pid_dataset([("bad",)], cs_cost=-1)

    with pytest.raises(ValueError, match="max_states must be a positive int"):
        build_pid_dataset([("bad",)], cs_cost=0, max_states=0)

    # _analyze_dataset file not found
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        _analyze_dataset("non_existent_file.npz")


def test_cli_build_pid_data_command(tmp_path: Path) -> None:
    """
    1. The use of function:
       Tests the `build-pid-data` CLI subcommand entry point end-to-end.
    2. How does the specific function work:
       Invokes `main()` with `--data-dir`, `--max-states 50`, `--report`, and `--workers 1`.
       Asserts process return code is 0 and verifies that `pid_bc_S0.npz`, `pid_bc_S1.npz`, and the
       Markdown report file are generated and non-empty.
    3. Concepts used in the function:
       CLI integration testing, end-to-end execution, artifact persistence verification.
    4. Places where this function is imported and used:
       Executed by pytest runner during test suite execution.
    5. Final output of the function, before and after:
       Before: Empty temporary target directory.
       After: Both S0 and S1 NPZ files and report markdown file exist and are validated.
    """
    report_file = tmp_path / "report.md"
    exit_code = main(
        [
            "build-pid-data",
            "--data-dir",
            str(tmp_path),
            "--max-states",
            "50",
            "--report",
            str(report_file),
            "--workers",
            "1",
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "pid_bc_S0.npz").exists()
    assert (tmp_path / "pid_bc_S1.npz").exists()
    assert report_file.exists()
    assert report_file.stat().st_size > 0

