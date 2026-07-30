"""Out-of-fold scoring for surface / CoT monitors (no train/test leakage)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


def kfold_indices(n: int, n_splits: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) covering every index exactly once in test."""
    if n < 2:
        idx = np.arange(n)
        return [(idx, idx)]
    n_splits = int(max(2, min(n_splits, n)))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = np.array_split(order, n_splits)
    out = []
    for i in range(n_splits):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        if len(train) == 0:
            train = test
        out.append((train, test))
    return out


def cross_fit_scores(
    examples: Sequence[Any],
    fit_fn: Callable[[list], Any],
    score_fn: Callable[[Any, list], np.ndarray],
    *,
    n_splits: int = 5,
    seed: int = 0,
    labels: Sequence[int] | None = None,
) -> np.ndarray:
    """
    5-fold cross-fitting: each example scored by a model that never trained on it.

    If a fold's train set lacks both classes, fall back to fitting on all other
    examples that have both classes when possible; else score with a constant.
    """
    n = len(examples)
    scores = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return scores
    if labels is None:
        labels = [int(getattr(ex, "label", 0) or 0) for ex in examples]
    y = np.asarray(labels, dtype=np.int64)

    for train_idx, test_idx in kfold_indices(n, n_splits=n_splits, seed=seed):
        y_tr = y[train_idx]
        if len(set(y_tr.tolist())) < 2:
            # Try expanding train with any leftover that restores both classes
            alt = [i for i in range(n) if i not in set(test_idx.tolist())]
            if len(set(y[alt].tolist())) >= 2:
                train_idx = np.asarray(alt, dtype=np.int64)
            else:
                scores[test_idx] = 0.5
                continue
        model = fit_fn([examples[i] for i in train_idx])
        fold_scores = np.asarray(score_fn(model, [examples[i] for i in test_idx]), dtype=np.float64)
        scores[test_idx] = fold_scores
    if np.isnan(scores).any():
        scores = np.nan_to_num(scores, nan=0.5)
    return scores
