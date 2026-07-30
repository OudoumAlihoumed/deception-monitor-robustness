#!/usr/bin/env python3
"""
Stage 1 — Apollo probe arithmetic gate (CPU, $0).

Scores **Apollo's precomputed activations** through our adapter / published
scaler. Confirms: weights, scaler absorption, sign convention, adapter math.

This does **not** validate our extraction path (hook index, pooling). For that
see Stage 2: ``modal run scripts/modal_app.py::apollo_extraction_gate``.

Pass: AUROC in [0.96, 0.999] on Apollo eval acts.
Fail (~0.7): wrong probe file / scaler / sign — not layer-off-by-one.

Usage:
  PYTHONPATH=. python scripts/apollo_offline_auroc_gate.py
  PYTHONPATH=. python scripts/apollo_offline_auroc_gate.py \\
      --acts path/to/apollo_eval_X.npy --labels path/to/apollo_eval_y.npy
  PYTHONPATH=. python scripts/apollo_offline_auroc_gate.py --require-auroc \\
      --acts ... --labels ...

Without --acts/--labels: structural-only (adapter≡published). That is Stage-0,
not Stage-1 AUROC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    correct = 0.0
    for p in pos:
        correct += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return correct / (len(pos) * len(neg))


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from src.monitors.apollo_probe import (
        apollo_to_torch_adapter,
        assert_apollo_adapter_live,
        default_apollo_probe_path,
        load_apollo_detector,
        verify_scaler_absorbed,
    )

    ap = argparse.ArgumentParser(description="Stage 1: Apollo acts → our adapter AUROC")
    ap.add_argument("--detector", type=Path, default=None)
    ap.add_argument("--acts", type=Path, default=None, help="(N,d) Apollo precomputed acts")
    ap.add_argument("--labels", type=Path, default=None, help="(N,) 1=deceptive")
    ap.add_argument("--lo", type=float, default=0.96)
    ap.add_argument("--hi", type=float, default=0.999)
    ap.add_argument(
        "--require-auroc",
        action="store_true",
        help="Exit non-zero if --acts/--labels missing (refuse structural soft-pass)",
    )
    args = ap.parse_args()

    path = args.detector or default_apollo_probe_path(ROOT)
    bundle = load_apollo_detector(path)
    adapter = apollo_to_torch_adapter(bundle, device="cpu")
    scaler = verify_scaler_absorbed(bundle, adapter)
    X_dummy = np.random.default_rng(0).normal(size=(8, bundle.dim)).astype(np.float64)
    live = assert_apollo_adapter_live(bundle, adapter, X_dummy)

    report = {
        "stage": 1,
        "stage_name": "apollo_precomputed_acts_adapter_auroc",
        "validates": [
            "published_weights",
            "scaler_absorption",
            "sign_convention",
            "adapter_arithmetic",
        ],
        "does_not_validate": [
            "hook_placement",
            "layer_index",
            "response_only_pooling",
        ],
        "next_stage": (
            "Stage 2 Modal: apollo_extraction_gate — Apollo transcripts through "
            "our model + layer-22 hook + response-only pool"
        ),
        "detector": str(path),
        "layer": bundle.layer,
        "dim": bundle.dim,
        "reg_coeff": bundle.reg_coeff,
        "scaler": scaler,
        "adapter_live": {"passed": live["passed"], "max_abs_diff": live["max_abs_diff"]},
        "auroc": None,
        "gate": None,
        "note": None,
    }

    if args.acts is None or args.labels is None:
        report["note"] = (
            "Structural / Stage-0 PASS only (scaler absorbed, adapter≡published). "
            "Stage-1 AUROC needs Apollo precomputed --acts/--labels "
            "(export from deception-detection or data.apolloresearch.ai/dd/). "
            "Stage-2 (extraction) is a separate Modal GPU job."
        )
        report["gate"] = "structural_only"
        print(json.dumps(report, indent=2))
        if args.require_auroc:
            print("FAIL — --require-auroc set but no --acts/--labels", file=sys.stderr)
            return 1
        print("STRUCTURAL PASS — not Stage-1 AUROC; not Stage-2 extraction.")
        return 0

    X = np.load(args.acts)
    y = np.load(args.labels).astype(np.int64)
    if X.shape[1] != bundle.dim:
        raise SystemExit(f"acts dim {X.shape[1]} != probe dim {bundle.dim}")
    scores = bundle.decision_scores(X)
    scores_ad = adapter.decision_numpy(X)
    if not np.allclose(scores, scores_ad, atol=1e-4):
        raise SystemExit("adapter scores diverge from bundle on Apollo eval acts")
    auc = auroc(scores, y)
    report["auroc"] = float(auc)
    report["n"] = int(len(y))
    report["n_pos"] = int(np.sum(y == 1))
    report["n_neg"] = int(np.sum(y == 0))
    if auc < args.lo:
        report["gate"] = "FAIL"
        report["note"] = (
            f"Stage-1 AUROC={auc:.4f} < {args.lo}: weights/scaler/sign likely wrong "
            "(not an extraction bug — Stage 2 not reached)."
        )
        print(json.dumps(report, indent=2))
        return 1
    report["gate"] = "PASS"
    report["note"] = (
        f"Stage-1 AUROC={auc:.4f} — adapter arithmetic OK. "
        "Next: Stage-2 extraction gate on Modal before smoke."
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
