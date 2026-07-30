"""Fixed-threshold recall@FPR (Apollo chat-calib style)."""

from __future__ import annotations

import numpy as np

from src.monitors.surface import _recall_at_fpr, threshold_at_fpr


def test_threshold_at_fpr_quantile():
    neg = np.linspace(0.0, 1.0, 101)
    thr = threshold_at_fpr(neg, 0.01)
    assert abs(thr - 0.99) < 1e-9


def test_recall_uses_calib_not_eval_negatives():
    # Eval negatives are all high → naive quantile would set thr≈1.0
    scores = np.array([0.7, 0.8, 0.95, 0.96, 0.97])
    labels = np.array([1, 1, 0, 0, 0])
    calib = np.linspace(0.0, 0.5, 200)  # chat calib → thr near 0.495
    out = _recall_at_fpr(scores, labels, 0.01, calib_neg_scores=calib)
    naive = _recall_at_fpr(scores, labels, 0.01)
    assert out["threshold"] < 0.6
    assert naive["threshold"] > 0.9
    assert out["recall"] == 1.0  # both positives above chat thr
    assert naive["recall"] == 0.0  # thr above both positives


def test_fixed_threshold_across_rungs():
    scores_b0 = np.array([0.9, 0.8, 0.1, 0.05])
    scores_b = np.array([0.2, 0.15, 0.1, 0.05])  # positives collapsed
    labels = np.array([1, 1, 0, 0])
    thr = 0.5
    r0 = _recall_at_fpr(scores_b0, labels, 0.01, threshold=thr)
    rb = _recall_at_fpr(scores_b, labels, 0.01, threshold=thr)
    assert r0["recall"] == 1.0
    assert rb["recall"] == 0.0
    assert r0["threshold"] == rb["threshold"] == thr
