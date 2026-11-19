"""
1. What this file does:
   Implements Stage 16: Model 5 - Neural Candidate Scorer (Deep Learning) for process-level
   scheduling (PID selection). Defines the permutation-equivariant PyTorch neural network
   architecture, parameter export to pure-NumPy float32 weights, pure-NumPy vectorized forward
   inference, and the Policy-compliant NeuralCandidatePolicy adapter.
2. Overall design:
   Scores candidate processes at each non-forced scheduling decision point (len(ready) >= 2) using
   a Deep Sets style architecture: candidate features are mapped through candidate-wise MLP phi,
   mean-pooled across valid candidates to obtain invariant context ctx, concatenated [z, ctx],
   and scored through MLP head. Masked candidates are assigned -inf and never chosen. Pure NumPy
   inference executes in simulation without requiring PyTorch at runtime, with canonical
   multi-tier tie-breaking.
3. Elements in the file:
   - Scorer: PyTorch nn.Module architecture (~3.6k parameters) with candidate-wise phi and head.
   - export_numpy_weights: Extracts model weights and biases as flat NumPy float32 arrays.
   - numpy_score: Vectorized pure-NumPy forward pass matching PyTorch within 1e-5.
   - NeuralCandidatePolicy: Policy subclass providing fast pure-NumPy inference for Engine.run().
4. Final output of the file:
   Exports Scorer, export_numpy_weights, numpy_score, and NeuralCandidatePolicy for training,
   simulation, benchmarking, and policy registry dispatch.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import config
from config import MODELS_DIR
from schedsim.domain.engine import View
from schedsim.domain.process import Process, validate_processes
from schedsim.pidsel.features import extract_candidate_features
from schedsim.policies.base import Policy

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


_BaseModule: type = nn.Module if (_TORCH_AVAILABLE and nn is not None) else object


class Scorer(_BaseModule):
    """
    1. The use of class:
       PyTorch neural network scoring ready process candidates at discrete scheduling states.
    2. What the class contains:
       - d: Input candidate feature dimension (default 13).
       - h: Hidden layer dimension (default 32).
       - phi: Candidate-wise feature embedding MLP: Linear(d, h) -> ReLU -> Linear(h, h) -> ReLU.
       - head: Scoring MLP: Linear(2*h, h) -> ReLU -> Linear(h, 1).
       - forward: Forward pass over padded candidates with attention masking and mean-pool context.
    3. Technologies used and why:
       - PyTorch nn.Module for automatic differentiation, GPU/CPU tensor ops, and Adam optimization.
       - Deep Sets permutation-equivariant architecture guaranteeing invariant candidate pooling.
    4. Key design and overall design of the class:
       Applies identical weights to each candidate, pools context symmetrically, and outputs
       unbounded logits for valid candidates while setting invalid/padded candidates to -inf.
    """

    def __init__(self, d: int = 13, h: int = 32) -> None:
        """
        1. The use of function:
           Initializes the candidate embedding layers and scoring head layers.
        2. How does the specific function work:
           Validates dimensions d and h. Validates PyTorch availability. Instantiates nn.Sequential
           phi mapping (d -> h -> h) and head mapping (2*h -> h -> 1).
        3. Concepts used in the function:
           Multi-layer perceptron, parameter initialization, deep sets representation.
        4. Places where this function is imported and used:
           Used by behavioral cloning training pipeline in train_bc.py and unit tests.
        5. Final output of the function, before and after:
           Before: Uninitialized Scorer instance.
           After: Fully parameterized PyTorch neural network module with ~3.6k parameters.
        """
        if not _TORCH_AVAILABLE or nn is None:
            raise ImportError("PyTorch is required to instantiate and train Scorer.")

        super().__init__()
        if isinstance(d, bool) or not isinstance(d, int) or d <= 0:
            raise ValueError(f"d must be a positive int, got {d}.")
        if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
            raise ValueError(f"h must be a positive int, got {h}.")

        self.d = d
        self.h = h

        self.phi = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, x: Any, mask: Any) -> Any:
        """
        1. The use of function:
           Computes candidate logits for a padded batch of candidate feature tensors.
        2. How does the specific function work:
           Maps x [B, N, d] through candidate-wise MLP phi -> z [B, N, h]. Unqueezes mask -> m
           [B, N, 1]. Computes symmetric mean-pooled context ctx = sum(z*m)/sum(m). Concatenates
           [z, ctx] -> [B, N, 2*h]. Maps through head -> [B, N]. Fills masked positions with -inf.
        3. Concepts used in the function:
           Permutation equivariance, mean-pool context aggregation, attention masking.
        4. Places where this function is imported and used:
           Invoked during forward training passes and validation evaluation.
        5. Final output of the function, before and after:
           Before: Tensors x of shape [B, N, d] and mask of shape [B, N] bool.
           After: Tensor of shape [B, N] with candidate logits (-inf at invalid positions).
        """
        z = self.phi(x)
        m = mask.unsqueeze(-1)
        ctx = (z * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1.0)
        s = self.head(torch.cat([z, ctx.expand_as(z)], -1)).squeeze(-1)
        return s.masked_fill(~mask, float("-inf"))


def export_numpy_weights(model: Scorer) -> dict[str, np.ndarray]:
    """
    1. The use of function:
       Extracts all weights and biases from a trained PyTorch Scorer model as NumPy float32 arrays.
    2. How does the specific function work:
       Validates model attributes phi and head. Extracts phi[0].weight, phi[0].bias, phi[2].weight,
       phi[2].bias, head[0].weight, head[0].bias, head[2].weight, head[2].bias into a dictionary
       of detached CPU float32 NumPy arrays.
    3. Concepts used in the function:
       Model parameter extraction, detachment, pure-NumPy export.
    4. Places where this function is imported and used:
       Called by train_neural_scorer in train_bc.py and tested in Stage 16 test suite.
    5. Final output of the function, before and after:
       Before: PyTorch Scorer model instance.
       After: Dictionary mapping parameter names to flat NumPy float32 arrays.
    """
    if not hasattr(model, "phi") or not hasattr(model, "head"):
        raise TypeError(
            f"model must be an instance of Scorer with phi and head, got {type(model).__name__}."
        )

    phi = model.phi
    head = model.head

    return {
        "phi_w0": phi[0].weight.detach().cpu().numpy().astype(np.float32),
        "phi_b0": phi[0].bias.detach().cpu().numpy().astype(np.float32),
        "phi_w1": phi[2].weight.detach().cpu().numpy().astype(np.float32),
        "phi_b1": phi[2].bias.detach().cpu().numpy().astype(np.float32),
        "head_w0": head[0].weight.detach().cpu().numpy().astype(np.float32),
        "head_b0": head[0].bias.detach().cpu().numpy().astype(np.float32),
        "head_w1": head[2].weight.detach().cpu().numpy().astype(np.float32),
        "head_b1": head[2].bias.detach().cpu().numpy().astype(np.float32),
    }


def numpy_score(
    weights: dict[str, np.ndarray],
    x: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    1. The use of function:
       Computes candidate scores using a vectorized pure-NumPy implementation of the Scorer network.
    2. How does the specific function work:
       Validates weights dictionary and array dimensions. Promotes 2D inputs to 3D batch form if
       needed. Computes phi MLP layers with ReLU activations: np.maximum(0.0, x @ W.T + b).
       Calculates mean-pooled invariant context across valid candidates. Concatenates features and
       context, executes head MLP, and masks out invalid candidates with -np.inf.
    3. Concepts used in the function:
       Pure-NumPy tensor linear algebra, vectorized ReLU, symmetric context pooling, mask handling.
    4. Places where this function is imported and used:
       Imported by NeuralCandidatePolicy, validation evaluators, and Stage 16 test suite.
    5. Final output of the function, before and after:
       Before: Weights dictionary, feature array x [B, N, d] (or [N, d]), mask [B, N] (or [N]).
       After: Float32 score array of shape [B, N] (or [N]), matching PyTorch within 1e-5.
    """
    if not isinstance(weights, dict):
        raise TypeError(f"weights must be a dict, got {type(weights).__name__}.")

    required_keys = (
        "phi_w0",
        "phi_b0",
        "phi_w1",
        "phi_b1",
        "head_w0",
        "head_b0",
        "head_w1",
        "head_b1",
    )
    for k in required_keys:
        if k not in weights:
            raise KeyError(f"Missing required weight key '{k}'.")

    x_arr = np.asarray(x, dtype=np.float32)
    mask_arr = np.asarray(mask, dtype=bool)

    squeeze = False
    if x_arr.ndim == 2:
        if mask_arr.ndim != 1:
            raise ValueError(
                f"mask shape {mask_arr.shape} incompatible with 2D x shape {x_arr.shape}."
            )
        x_arr = x_arr[np.newaxis, :, :]
        mask_arr = mask_arr[np.newaxis, :]
        squeeze = True
    elif x_arr.ndim == 3:
        if mask_arr.ndim != 2:
            raise ValueError(
                f"mask shape {mask_arr.shape} incompatible with 3D x shape {x_arr.shape}."
            )
    else:
        raise ValueError(f"x must be 2D or 3D array, got shape {x_arr.shape}.")

    # phi MLP: Linear -> ReLU -> Linear -> ReLU
    z0 = np.maximum(0.0, x_arr @ weights["phi_w0"].T + weights["phi_b0"])
    z = np.maximum(0.0, z0 @ weights["phi_w1"].T + weights["phi_b1"])

    # Mean-pooled context over valid candidates
    m = mask_arr[..., np.newaxis].astype(np.float32)
    m_sum = np.maximum(1.0, np.sum(m, axis=1, keepdims=True))
    ctx = np.sum(z * m, axis=1, keepdims=True) / m_sum
    ctx_exp = np.broadcast_to(ctx, z.shape)

    # head MLP: Linear -> ReLU -> Linear
    cat = np.concatenate([z, ctx_exp], axis=-1)
    h0 = np.maximum(0.0, cat @ weights["head_w0"].T + weights["head_b0"])
    s = np.squeeze(h0 @ weights["head_w1"].T + weights["head_b1"], axis=-1)

    s = np.where(mask_arr, s, -np.inf)
    if squeeze:
        return s[0]
    return s


class NeuralCandidatePolicy(Policy):
    """
    1. The use of class:
       Policy-compliant scheduling adapter utilizing trained neural candidate scorer weights to
       score and dispatch ready processes at each non-forced scheduling decision point.
    2. What the class contains:
       - cs_cost: Integer context switch cost penalty.
       - weights: Pure-NumPy float32 weight dictionary for inference without PyTorch.
       - select: Evaluates candidate features, scores candidates, breaks ties, returns PID.
       - from_cs_cost: Factory constructor loading model matching cs_cost.
       - from_setting: Factory constructor loading model matching benchmark setting.
       - on_start: AI startup latency accounting hook (returns 0.0 µs).
       - decision_cost_us: Per-decision AI latency accounting hook (returns 0.0 µs).
       - reset: Resets policy state between simulation episodes.
    3. Technologies used and why:
       - Policy ABC inheritance for seamless plug-in into Engine.run() and CLI simulation.
       - Pure-NumPy inference eliminating all PyTorch runtime latency and dependencies.
    4. Key design and overall design of the class:
       Provides fast process-level candidate ranking with deterministic tie-breaking.
    """

    name: str = "NeuralCandidatePolicy"

    def __init__(
        self,
        weights: dict[str, np.ndarray] | None = None,
        cs_cost: int = 0,
        setting: str | None = None,
        models_dir: Path | str = MODELS_DIR,
        name: str = "NeuralCandidatePolicy",
    ) -> None:
        """
        1. The use of function:
           Initializes NeuralCandidatePolicy from exported weights dictionary or stored artifacts.
        2. How does the specific function work:
           Validates inputs. If weights dictionary is passed, stores it. Else resolves setting or
           cs_cost and loads stored .npz weights from models_dir.
        3. Concepts used in the function:
           Constructor overloading, defensive argument validation, lazy artifact resolution.
        4. Places where this function is imported and used:
           Instantiated by simulation engines, validation runners, and policy registry factory.
        5. Final output of the function, before and after:
           Before: Uninitialized policy reference.
           After: Fully configured NeuralCandidatePolicy ready for simulation execution.
        """
        super().__init__(name=name)
        if isinstance(cs_cost, bool) or not isinstance(cs_cost, int) or cs_cost < 0:
            raise ValueError(f"cs_cost must be a non-negative int, got {cs_cost}.")
        self.cs_cost = int(cs_cost)

        if weights is not None:
            if not isinstance(weights, dict):
                raise TypeError(f"weights must be a dict, got {type(weights).__name__}.")
            self.weights: dict[str, np.ndarray] = {
                k: np.asarray(v, dtype=np.float32) for k, v in weights.items()
            }
        else:
            resolved_setting = setting or f"S{cs_cost}"
            inst = self.from_setting(resolved_setting, models_dir=models_dir, name=name)
            self.weights = inst.weights
            self.cs_cost = inst.cs_cost

        self._processes: tuple[Process, ...] | None = None

    @classmethod
    def from_cs_cost(
        cls,
        cs_cost: int = 0,
        models_dir: Path | str = MODELS_DIR,
        name: str | None = None,
    ) -> "NeuralCandidatePolicy":
        """
        1. The use of function:
           Factory method instantiating NeuralCandidatePolicy with the trained model for cs_cost.
        2. How does the specific function work:
           Resolves model file path matching cs_cost (pid_scorer_S{cs_cost}_np.npz), loads weights
           dictionary from NPZ archive, and instantiates NeuralCandidatePolicy.
        3. Concepts used in the function:
           Factory pattern, artifact persistence loading, setting encapsulation.
        4. Places where this function is imported and used:
           Used by simulation runners, CLI handlers, and unit tests.
        5. Final output of the function, before and after:
           Before: cs_cost integer and optional models directory.
           After: Configured NeuralCandidatePolicy instance.
        """
        dir_path = Path(models_dir)
        target_path = dir_path / f"pid_scorer_S{cs_cost}_np.npz"
        if not target_path.exists():
            raise FileNotFoundError(
                f"Trained neural scorer model for cs_cost={cs_cost} not found at {target_path}."
            )

        loaded = np.load(target_path)
        weights = {k: loaded[k] for k in loaded.files}
        policy_name = name or f"PID_NN_S{cs_cost}"
        return cls(weights=weights, cs_cost=cs_cost, name=policy_name)

    @classmethod
    def from_setting(
        cls,
        setting: str = "S0",
        models_dir: Path | str = MODELS_DIR,
        name: str | None = None,
    ) -> "NeuralCandidatePolicy":
        """
        1. The use of function:
           Factory method instantiating NeuralCandidatePolicy with the trained model for a setting.
        2. How does the specific function work:
           Validates setting name against config.SETTINGS, resolves corresponding cs_cost,
           and delegates to cls.from_cs_cost().
        3. Concepts used in the function:
           Setting resolution, classmethod factory pattern.
        4. Places where this function is imported and used:
           Used by multi-setting benchmarks, CLI dispatch, and acceptance tests.
        5. Final output of the function, before and after:
           Before: Setting name string.
           After: Configured NeuralCandidatePolicy instance.
        """
        setting_key = setting.upper().strip()
        if setting_key not in config.SETTINGS:
            valid = ", ".join(config.SETTINGS.keys())
            raise ValueError(f"Unknown setting '{setting}'. Must be one of: {valid}.")
        cs_cost = config.SETTINGS[setting_key].cs_cost
        policy_name = name or f"PID_NN_{setting_key}"
        return cls.from_cs_cost(
            cs_cost=cs_cost,
            models_dir=models_dir,
            name=policy_name,
        )

    def bind_workload(self, processes: Sequence[Process]) -> None:
        """
        1. The use of function:
           Binds a workload process sequence to track processes across the simulation episode.
        2. How does the specific function work:
           Validates processes using validate_processes and stores defensive immutable tuple.
        3. Concepts used in the function:
           Workload binding lifecycle hook, defensive copying.
        4. Places where this function is imported and used:
           Invoked by engine runner run() before simulation execution loops.
        5. Final output of the function, before and after:
           Before: Unbound policy or previous workload state.
           After: Policy storing bound workload process tuple.
        """
        validate_processes(processes)
        self._processes = tuple(processes)

    def reset(self) -> None:
        """
        1. The use of function:
           Resets policy internal state between simulation runs.
        2. How does the specific function work:
           Stateless scorer; keeps weights intact and resets cached workload processes.
        3. Concepts used in the function:
           Lifecycle hook, test isolation.
        4. Places where this function is imported and used:
           Called by Engine.run() between simulation episodes.
        5. Final output of the function, before and after:
           Before: Active policy state.
           After: Reset policy state.
        """
        self._processes = None

    def on_start(self, processes: Sequence[Process]) -> float:
        """
        1. The use of function:
           Lifecycle hook returning upfront model loading latency in microseconds.
        2. How does the specific function work:
           Binds workload processes and returns 0.0 µs baseline latency placeholder.
        3. Concepts used in the function:
           Lifecycle accounting, latency profiling hook.
        4. Places where this function is imported and used:
           Called by Engine.run() prior to simulation start.
        5. Final output of the function, before and after:
           Before: Workload process sequence.
           After: Returns float latency in microseconds (0.0).
        """
        self.bind_workload(processes)
        return 0.0

    def decision_cost_us(self, view: View) -> float:
        """
        1. The use of function:
           Returns the estimated decision latency in microseconds from the latency profile.
        2. How does the specific function work:
           Returns 0.0 if view is None or ready count is <= 1 (forced decision). Otherwise
           queries get_interpolated_cost_us("Neural", len(view.ready)).
        3. Concepts used in the function:
           Per-decision overhead accounting, latency profile lookup, interpolation.
        4. Places where this function is imported and used:
           Called by Engine.step() after each non-forced scheduling decision.
        5. Final output of the function, before and after:
           Before: Current simulation state View.
           After: Returns float latency in microseconds.
        """
        if view is None or (config.FORCED_DECISION_FREE and len(view.ready) <= 1):
            return 0.0
        from schedsim.overhead import get_interpolated_cost_us

        return get_interpolated_cost_us("Neural", len(view.ready))

    def select(self, view: View) -> str:
        """
        1. The use of function:
           Selects the next process PID to execute using pure-NumPy neural candidate scoring.
        2. How does the specific function work:
           Validates ready queue is non-empty. For forced decisions (len <= 1), returns the single
           ready PID immediately. For non-forced decisions, extracts candidate feature matrix X,
           adds batch dimension (1, N, 13) and mask (1, N), scores candidates with numpy_score,
           breaks ties canonically (highest score -> smaller remaining burst -> earlier arrival
           -> pid string), and returns chosen PID.
        3. Concepts used in the function:
           Fast candidate scoring, deterministic multi-tier tie-breaking, forced-decision bypass.
        4. Places where this function is imported and used:
           Invoked at each discrete simulation step by Engine.step() and run().
        5. Final output of the function, before and after:
           Before: Simulation View snapshot.
           After: Returns selected PID string to execute on CPU.
        """
        if not view.ready:
            raise ValueError("Cannot select process: ready queue is empty.")

        if len(view.ready) <= 1:
            return view.ready[0].pid

        x_cands, candidate_pids = extract_candidate_features(
            view, cs_cost=self.cs_cost, pending_arrivals=view.pending_arrivals
        )
        x_batch = x_cands[np.newaxis, :, :]
        mask = np.ones((1, len(candidate_pids)), dtype=bool)
        scores = numpy_score(self.weights, x_batch, mask)[0]

        proc_map = {p.pid: p for p in view.ready}
        best_pair = min(
            zip(candidate_pids, scores, strict=True),
            key=lambda pair: (
                -float(pair[1]),
                proc_map[pair[0]].remaining,
                proc_map[pair[0]].arrival,
                pair[0],
            ),
        )
        return best_pair[0]


__all__ = [
    "NeuralCandidatePolicy",
    "Scorer",
    "export_numpy_weights",
    "numpy_score",
]
