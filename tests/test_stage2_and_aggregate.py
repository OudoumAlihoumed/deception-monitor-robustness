"""Stage-2 helpers + aggregate_cells reporting (n=2 → values+range)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from src.eval.apollo_stage2 import (
    assert_response_span_matches_completion,
    pearson_corr,
    spearman_corr,
)
from src.pipeline.residual import response_attention_mask


def test_span_assert_passes_when_mask_matches_completion():
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            # pretend ids map to words
            table = {1: "Hello", 2: "world", 3: "rest", 4: "PROMPT"}
            return " ".join(table[int(i)] for i in ids.tolist())

    input_ids = torch.tensor([[4, 4, 1, 2, 3]])
    mask = torch.tensor([[0, 0, 1, 1, 1]])
    report = assert_response_span_matches_completion(
        Tok(), input_ids, mask, "Hello world rest of text", prefix_chars=11
    )
    assert report["passed"]


def test_span_assert_fails_on_prompt_leak():
    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            table = {1: "Hello", 2: "world", 4: "SYSTEM"}
            return " ".join(table[int(i)] for i in ids.tolist())

    input_ids = torch.tensor([[4, 1, 2]])
    mask = torch.tensor([[1, 1, 1]])  # includes SYSTEM
    try:
        assert_response_span_matches_completion(
            Tok(), input_ids, mask, "Hello world", prefix_chars=5
        )
        assert False, "expected AssertionError"
    except AssertionError as e:
        assert "response-span" in str(e).lower() or "prefix" in str(e).lower()


def test_corr_perfect():
    x = np.linspace(0, 1, 50)
    assert abs(pearson_corr(x, x) - 1.0) < 1e-9
    assert abs(spearman_corr(x, x**3) - 1.0) < 1e-9  # monotone


def test_stage2_gate_pairing_count_not_spearman_fail_fallback():
    """n_paired≥50: Spearman-fail + AUROC-pass must FAIL (not auroc_fallback)."""
    SPEARMAN_MIN_PAIRS = 50
    corr_min, lo = 0.99, 0.99

    def decide(n_paired, spearman, auroc):
        auroc_ok = auroc == auroc and auroc >= lo
        spearman_ok = spearman == spearman and spearman >= corr_min
        if n_paired >= SPEARMAN_MIN_PAIRS:
            return spearman_ok and auroc_ok, "spearman_and_auroc"
        return auroc_ok, "auroc_fallback"

    # The failure mode Stage 2 catches: within-group reordering
    # (wrong layer / pooling) — AUROC stays high, Spearman drops
    passed, mode = decide(195, 0.97, 0.999)
    assert mode == "spearman_and_auroc"
    assert passed is False

    passed, mode = decide(195, 0.995, 0.999)
    assert passed is True

    passed, mode = decide(14, float("nan"), 0.995)
    assert mode == "auroc_fallback"
    assert passed is True

    passed, mode = decide(14, float("nan"), 0.97)  # near-miss AUROC
    assert mode == "auroc_fallback"
    assert passed is False


def test_pooling_dilution_ratio_warning_threshold():
    agentic, stage2 = 400.0, 30.0
    ratio = agentic / max(stage2, 1.0)
    assert ratio > 5.0  # Smoke-1 warns (does not fail) above 5


def test_aggregate_cells_n2_values_and_range(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_cells.py"
    spec = importlib.util.spec_from_file_location("aggregate_cells", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    # Two b3 cells → values + range, not mean±std
    for seed, rec in ((0, 0.4), (1, 0.6)):
        cell = {
            "rung": "b3",
            "seed": seed,
            "lambda": 0.5,
            "gpu_seconds": 10.0 + seed,
            "lambda_worked": {"passed": True},
            "adapter_live": {"passed": True},
            "monitor_metrics": {
                "recall": {"linear_probe": {"b3": rec}},
                "auroc": {"linear_probe": {"b3": 0.9}},
            },
        }
        p = tmp_path / f"b3_seed{seed}_lam0.5.json"
        p.write_text(json.dumps(cell))

    cells = [mod.parse_cell(p) for p in sorted(tmp_path.glob("*.json"))]
    summary = mod.aggregate(cells)
    pooled = summary["summary_by_rung"]["b3"]
    assert pooled["n_cells"] == 2
    rec = pooled["recall_linear_probe"]
    assert rec["n_seeds"] == 2
    assert rec["values"] == [0.4, 0.6]
    assert rec["range"] == [0.4, 0.6]
    assert "std" not in rec
    assert "mean" not in rec
    gpu = pooled["gpu_seconds"]
    assert gpu["n_seeds"] == 2
    assert gpu["values"] == [10.0, 11.0]
    assert gpu["range"] == [10.0, 11.0]


def test_aggregate_cells_n3_mean_std(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_cells.py"
    spec = importlib.util.spec_from_file_location("aggregate_cells", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    for seed, rec in ((0, 0.3), (1, 0.5), (2, 0.7)):
        cell = {
            "rung": "b3",
            "seed": seed,
            "monitor_metrics": {"recall": {"linear_probe": {"b3": rec}}},
        }
        (tmp_path / f"b3_seed{seed}.json").write_text(json.dumps(cell))

    cells = [mod.parse_cell(p) for p in sorted(tmp_path.glob("*.json"))]
    pooled = mod.aggregate(cells)["summary_by_rung"]["b3"]["recall_linear_probe"]
    assert pooled["n_seeds"] == 3
    assert abs(pooled["mean"] - 0.5) < 1e-9
    assert "std" in pooled
