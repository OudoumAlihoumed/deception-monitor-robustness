"""$0 gates for probe adapter: train≡eval, sign, grad-flow, pad pooling."""

from __future__ import annotations

import numpy as np
import pytest

from src.monitors.linear_probe import ActivationExample, LinearProbe
from src.monitors.probe_adapter import (
    assert_lambda_worked,
    assert_logit_separates_held_out,
    assert_positive_class_is_deceptive,
    assert_probe_term_has_grad,
    assert_train_eval_probe_match,
    make_torch_probe_adapter,
    mean_pool_masked,
    sklearn_decision_scores,
)


def _fit_toy_probe(dim: int = 16, n: int = 40, seed: int = 0) -> tuple[LinearProbe, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # Class-conditional Gaussians so logistic separates cleanly
    X0 = rng.normal(0.0, 1.0, size=(n // 2, dim))
    X1 = rng.normal(1.5, 1.0, size=(n // 2, dim))
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    examples = [
        ActivationExample(activation=X[i], session_id=f"s{i}", label=int(y[i]), layer=22)
        for i in range(n)
    ]
    probe = LinearProbe(method="logistic", C=1.0, random_state=seed).fit(examples)
    return probe, X, y


def test_positive_class_is_deceptive():
    probe, _, _ = _fit_toy_probe()
    assert_positive_class_is_deceptive(probe)
    assert list(probe.clf.classes_)[1] == 1


def test_train_eval_match_and_sign():
    probe, X, y = _fit_toy_probe()
    adapter = make_torch_probe_adapter(probe, dim=X.shape[1], device="cpu")
    report = assert_train_eval_probe_match(probe, adapter, X, atol=1e-4)
    assert report["passed"]
    sep = assert_logit_separates_held_out(probe, X, y)
    assert sep["gap"] > 0


def test_train_eval_on_padded_batch_pool():
    """Pad tokens must not silently diverge train/eval pooling."""
    import torch

    probe, X, y = _fit_toy_probe(dim=8, n=20)
    adapter = make_torch_probe_adapter(probe, dim=8, device="cpu")

    # Build a padded batch from two sequences of different lengths
    seq_a = torch.tensor(X[0:1].repeat(5, axis=0), dtype=torch.float32)  # (5, 8) — wrong
    # Fix: use tile properly
    seq_a = torch.tensor(np.tile(X[0], (5, 1)), dtype=torch.float32)
    seq_b = torch.tensor(np.tile(X[1], (3, 1)), dtype=torch.float32)
    # Pad seq_b to length 5
    pad = torch.zeros(2, 8)
    seq_b_pad = torch.cat([seq_b, pad], dim=0)
    batch = torch.stack([seq_a, seq_b_pad], dim=0)  # (2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.float32)

    pooled = mean_pool_masked(batch, mask).detach().cpu().numpy()
    # Unmasked mean would pollute second row with zeros
    pooled_bad = batch.float().mean(dim=1).numpy()
    assert not np.allclose(pooled[1], pooled_bad[1], atol=1e-5)

    assert_train_eval_probe_match(probe, adapter, pooled, atol=1e-4)


def test_probe_term_has_grad():
    import torch

    probe, X, _ = _fit_toy_probe()
    adapter = make_torch_probe_adapter(probe, dim=X.shape[1], device="cpu")
    pooled = torch.tensor(X[:4], dtype=torch.float32)
    report = assert_probe_term_has_grad(adapter, pooled)
    assert report["grad_norm"] > 1e-8


def test_sigmoid_softplus_is_numerically_weak():
    """Document why A1–A3 matters: softplus(σ) barely moves."""
    import torch
    import torch.nn.functional as F

    s = torch.linspace(0.01, 0.99, 50)
    span = float((F.softplus(s).max() - F.softplus(s).min()).item())
    # softplus on (0,1) spans < 0.7; logit softplus spans much more
    assert span < 0.7
    logits = torch.linspace(-8, 8, 50)
    span_logit = float((F.softplus(logits).max() - F.softplus(logits).min()).item())
    assert span_logit > 5.0


def test_lambda_worked_gate():
    b0 = np.array([2.0, 2.5, 3.0])
    b3 = np.array([0.5, 0.2, 0.0])
    assert_lambda_worked(b0, b3, min_drop=0.1)
    with pytest.raises(AssertionError):
        assert_lambda_worked(b0, b0, min_drop=0.1)


def test_flipped_classes_rejected(monkeypatch):
    probe, X, y = _fit_toy_probe()
    # Force flipped class order if possible
    probe.clf.classes_ = np.array([1, 0])
    with pytest.raises(AssertionError, match="positive class must be deceptive"):
        assert_positive_class_is_deceptive(probe)
