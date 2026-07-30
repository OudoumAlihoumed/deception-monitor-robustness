"""
Load Goldowsky-Dill / ApolloResearch published residual probes.

Source: https://github.com/ApolloResearch/deception-detection
Example weights: example_results/instructed_pairs/detector.pt

Apollo fit call (experiment.py):
    detector.fit(deceptive_fit_acts, honest_fit_acts)
so class 1 = deceptive (higher score = more deceptive) — matches our convention.

Recipe (paper): logistic regression, L2 with reg_coeff=10 (C=0.1), StandardScaler,
Apollo ``detect_layers`` index 22 (= HF ``hidden_states[22]`` / ``layers[21]`` output)
of Llama-3.3-70B-Instruct in **fp16 (no quantization)**, mean over tokens for a
per-response score.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


@dataclass
class ApolloProbeBundle:
    """Published Apollo linear probe (single layer)."""

    layers: list[int]
    direction: np.ndarray  # (d,)
    scaler_mean: np.ndarray  # (d,)
    scaler_scale: np.ndarray  # (d,)
    normalize: bool
    reg_coeff: float
    source_path: str
    # Apollo: higher = more deceptive when fit(deceptive, honest)
    positive_class: str = "deceptive"

    @property
    def layer(self) -> int:
        return int(self.layers[0])

    @property
    def dim(self) -> int:
        return int(self.direction.size)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """Signed logit / direction projection (higher = more deceptive)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {X.shape[1]}")
        if self.normalize:
            scale = np.maximum(self.scaler_scale, 1e-8)
            Xs = (X - self.scaler_mean) / scale
        else:
            Xs = X
        return Xs @ self.direction.astype(np.float64)

    def score_prob(self, X: np.ndarray) -> np.ndarray:
        """Map logit to (0,1) for recall@FPR thresholds."""
        z = np.clip(self.decision_scores(X), -40, 40)
        return 1.0 / (1.0 + np.exp(-z))


def load_apollo_detector(path: Union[str, Path]) -> ApolloProbeBundle:
    """
    Load pickle detector.pt from Apollo example_results.

    Restores published ``scaler_mean`` / ``scaler_scale`` from the file.
    They are **never** refit on our activations — absorbed into adapter weights.
    """
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    layers = list(data["layers"])
    directions = data["directions"]
    if hasattr(directions, "detach"):
        directions = directions.detach().cpu().numpy()
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim == 2:
        direction = directions[0]
    else:
        direction = directions.ravel()

    if "scaler_mean" not in data or "scaler_scale" not in data:
        raise AssertionError(
            f"Apollo detector missing published scaler_mean/scaler_scale in {path}. "
            "Refitting on our activations is forbidden."
        )
    mean = data["scaler_mean"]
    scale = data["scaler_scale"]
    if mean is not None and hasattr(mean, "detach"):
        mean = mean.detach().cpu().numpy()
    if scale is not None and hasattr(scale, "detach"):
        scale = scale.detach().cpu().numpy()
    if mean is not None and np.asarray(mean).ndim == 2:
        mean = np.asarray(mean, dtype=np.float64)[0]
    if scale is not None and np.asarray(scale).ndim == 2:
        scale = np.asarray(scale, dtype=np.float64)[0]
    mean = np.asarray(mean, dtype=np.float64).ravel()
    scale = np.asarray(scale, dtype=np.float64).ravel()
    if mean.size == 0 or scale.size == 0:
        raise AssertionError("Apollo scaler_mean/scale empty — refuse to invent identity scaler")

    return ApolloProbeBundle(
        layers=layers,
        direction=direction.astype(np.float64).ravel(),
        scaler_mean=mean,
        scaler_scale=scale,
        normalize=bool(data.get("normalize", True)),
        reg_coeff=float(data.get("reg_coeff", 10.0)),
        source_path=str(path),
    )


def verify_scaler_absorbed(bundle: ApolloProbeBundle, adapter, atol: float = 1e-5) -> dict:
    """Confirm adapter Linear weights encode published mean/std (not identity / not refit)."""
    mean = bundle.scaler_mean.astype(np.float64)
    scale = np.maximum(bundle.scaler_scale.astype(np.float64), 1e-8)
    w = bundle.direction.astype(np.float64)
    w_eff = w / scale
    b_eff = -float((w / scale) @ mean)
    got_w = adapter.linear.weight.detach().cpu().numpy().ravel().astype(np.float64)
    got_b = float(adapter.linear.bias.detach().cpu().numpy().ravel()[0])
    if not np.allclose(got_w, w_eff, atol=atol):
        raise AssertionError("Apollo adapter weight ≠ published direction/scale absorption")
    if abs(got_b - b_eff) > atol:
        raise AssertionError("Apollo adapter bias ≠ published mean absorption")
    # Sanity: scaler is not a trivial identity (would suggest missing restore)
    if float(np.max(np.abs(mean))) < 1e-12 and float(np.max(np.abs(scale - 1.0))) < 1e-12:
        raise AssertionError(
            "Published scaler looks like identity zeros/ones — check detector.pt restore"
        )
    return {
        "scaler_absorbed": True,
        "normalize": bundle.normalize,
        "mean_l2": float(np.linalg.norm(mean)),
        "scale_mean": float(np.mean(scale)),
        "refit_on_ours": False,
    }


def apollo_to_torch_adapter(bundle: ApolloProbeBundle, device: str = "cpu"):
    """
    Absorb StandardScaler into a Linear(d→1) so score_torch(x) = decision_scores(x)
    on raw activations (same trick as our probe_adapter).
    """
    import torch
    import torch.nn as nn

    mean = bundle.scaler_mean.astype("float32")
    scale = np.maximum(bundle.scaler_scale.astype("float32"), 1e-8)
    w = bundle.direction.astype("float32")
    w_eff = w / scale
    b_eff = -float((w / scale) @ mean)  # Apollo LR fit_intercept=False on scaled space

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(bundle.dim, 1)
            with torch.no_grad():
                self.linear.weight.copy_(
                    torch.tensor(w_eff, dtype=torch.float32).unsqueeze(0)
                )
                self.linear.bias.copy_(
                    torch.tensor([b_eff], dtype=torch.float32)
                )
            self.apollo_layer = bundle.layer
            self.source_path = bundle.source_path
            self.scaler_absorbed = True
            self.refit_scaler = False

        def score_torch(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.linear(x.float()).squeeze(-1)

        def decision_numpy(self, X: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.linear.weight.device)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                return self.score_torch(t).detach().cpu().numpy().astype(np.float64)

    ad = Adapter().to(device)
    verify_scaler_absorbed(bundle, ad)
    return ad


def assert_apollo_adapter_live(
    bundle: ApolloProbeBundle,
    adapter,
    X: np.ndarray,
    *,
    atol: float = 1e-4,
    min_grad_norm: float = 1e-8,
    model: object = None,
) -> dict:
    """
    A5+A7 for published Apollo adapters: call after *every* rebuild.

    Grad path uses a dummy Parameter (safe before LoRA). Cleans up in finally.
    If ``model`` is passed, also zero its grads so synthetic .backward()
    cannot leak into the first training step.
    """
    import torch

    from src.monitors.probe_adapter import assert_probe_term_has_grad

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_check = X[: min(8, len(X))]
    scaler_report = verify_scaler_absorbed(bundle, adapter)
    ref = bundle.decision_scores(X_check)
    got = adapter.decision_numpy(X_check)
    max_abs = float(np.max(np.abs(ref - got)))
    if not np.allclose(ref, got, atol=atol):
        raise AssertionError(
            f"Apollo train≡eval mismatch after adapter rebuild: "
            f"max|ref-adapter|={max_abs:.6g} (atol={atol})"
        )
    device = str(adapter.linear.weight.device)
    pooled = torch.tensor(X_check[:4], dtype=torch.float32, device=device)
    try:
        grad_report = assert_probe_term_has_grad(
            adapter, pooled, min_grad_norm=min_grad_norm
        )
    finally:
        try:
            adapter.zero_grad(set_to_none=True)
        except Exception:
            pass
        if model is not None:
            try:
                model.zero_grad(set_to_none=True)
            except Exception:
                pass
        del pooled
    return {
        "passed": True,
        "max_abs_diff": max_abs,
        "atol": atol,
        "n": int(X_check.shape[0]),
        "grad": grad_report,
        "device": device,
        "scaler": scaler_report,
        "grads_cleared": True,
    }


def default_apollo_probe_path(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / "third_party" / "apollo_probes" / "instructed_pairs_detector.pt"
