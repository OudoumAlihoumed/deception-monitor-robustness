"""Apollo adapter live asserts (A5+A7 after every rebuild)."""

from __future__ import annotations

import numpy as np

from src.monitors.apollo_probe import (
    ApolloProbeBundle,
    apollo_to_torch_adapter,
    assert_apollo_adapter_live,
)


def _toy_bundle(dim: int = 32) -> ApolloProbeBundle:
    rng = np.random.default_rng(0)
    direction = rng.normal(size=dim).astype(np.float64)
    direction /= np.linalg.norm(direction) + 1e-8
    return ApolloProbeBundle(
        layers=[22],
        direction=direction,
        scaler_mean=rng.normal(size=dim).astype(np.float64) * 0.1,
        scaler_scale=np.ones(dim, dtype=np.float64),
        normalize=True,
        reg_coeff=10.0,
        source_path="toy",
    )


def test_assert_apollo_adapter_live_passes():
    bundle = _toy_bundle()
    adapter = apollo_to_torch_adapter(bundle, device="cpu")
    X = np.random.default_rng(1).normal(size=(12, bundle.dim)).astype(np.float64)
    report = assert_apollo_adapter_live(bundle, adapter, X)
    assert report["passed"]
    assert report["max_abs_diff"] < 1e-4
    assert report.get("grads_cleared") is True
    # Item 7: synthetic .backward() must not leave grads for train step 0
    for p in adapter.parameters():
        assert p.grad is None, "adapter.grad leaked after assert_apollo_adapter_live"


def test_assert_clears_model_grads_when_passed():
    """Pass model= so LoRA params cannot keep synthetic grads from the assert."""
    import torch
    import torch.nn as nn

    bundle = _toy_bundle()
    adapter = apollo_to_torch_adapter(bundle, device="cpu")
    model = nn.Linear(4, 4)
    model.weight.grad = torch.ones_like(model.weight)  # stale
    X = np.random.default_rng(3).normal(size=(8, bundle.dim)).astype(np.float64)
    report = assert_apollo_adapter_live(bundle, adapter, X, model=model)
    assert report["passed"]
    assert model.weight.grad is None
    for p in adapter.parameters():
        assert p.grad is None


def test_assert_apollo_adapter_live_catches_mismatch():
    bundle = _toy_bundle()
    adapter = apollo_to_torch_adapter(bundle, device="cpu")
    import torch

    with torch.no_grad():
        adapter.linear.weight.add_(1.0)
    X = np.random.default_rng(2).normal(size=(8, bundle.dim)).astype(np.float64)
    try:
        assert_apollo_adapter_live(bundle, adapter, X)
        assert False, "expected AssertionError"
    except AssertionError as e:
        msg = str(e).lower()
        assert (
            "mismatch" in msg
            or "train" in msg
            or "absorption" in msg
            or "adapter weight" in msg
        )
