"""
Session-clustered bootstrap for means and paired deltas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    std: float
    n_bootstrap: int
    confidence_level: float

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "std": self.std,
            "n_bootstrap": self.n_bootstrap,
            "confidence_level": self.confidence_level,
        }


def _unique_preserve(ids: Sequence) -> list:
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def session_clustered_indices(
    session_ids: Sequence,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Resample clusters (sessions) with replacement; return row indices
    for all observations belonging to sampled sessions.
    """
    session_ids = np.asarray(session_ids)
    unique = _unique_preserve(session_ids.tolist())
    unique = np.asarray(unique)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    indices = []
    for s in sampled:
        indices.extend(np.where(session_ids == s)[0].tolist())
    return np.asarray(indices, dtype=np.int64)


def bootstrap_mean(
    values: np.ndarray,
    session_ids: Sequence,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Session-clustered bootstrap CI for a mean."""
    values = np.asarray(values, dtype=np.float64)
    session_ids = np.asarray(session_ids)
    rng = np.random.default_rng(seed)
    estimate = float(np.mean(values))
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = session_clustered_indices(session_ids, rng)
        stats[b] = float(np.mean(values[idx]))
    alpha = 1.0 - confidence_level
    lo, hi = np.quantile(stats, [alpha / 2, 1.0 - alpha / 2])
    return BootstrapResult(
        estimate=estimate,
        ci_low=float(lo),
        ci_high=float(hi),
        std=float(np.std(stats, ddof=1)),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def bootstrap_paired_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
    session_ids: Sequence,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
    statistic: Optional[Callable[[np.ndarray], float]] = None,
) -> BootstrapResult:
    """
    Session-clustered bootstrap for paired delta: mean(a - b) by default,
    or a custom statistic of the paired differences.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("values_a and values_b must have the same shape")
    session_ids = np.asarray(session_ids)
    if len(session_ids) != len(a):
        raise ValueError("session_ids length must match values")
    diff = a - b
    stat_fn = statistic or (lambda x: float(np.mean(x)))
    estimate = stat_fn(diff)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = session_clustered_indices(session_ids, rng)
        stats[i] = stat_fn(diff[idx])
    alpha = 1.0 - confidence_level
    lo, hi = np.quantile(stats, [alpha / 2, 1.0 - alpha / 2])
    return BootstrapResult(
        estimate=float(estimate),
        ci_low=float(lo),
        ci_high=float(hi),
        std=float(np.std(stats, ddof=1)),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


def bootstrap_recall_delta(
    scores_b0: np.ndarray,
    labels: np.ndarray,
    scores_b: np.ndarray,
    session_ids: Sequence,
    fpr_target: float = 0.01,
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """
    Bootstrap Δdet = recall@fpr(b0) - recall@fpr(b) with session clustering.
    Thresholds are recomputed within each bootstrap replicate on negatives.
    """
    from src.monitors.surface import _recall_at_fpr

    scores_b0 = np.asarray(scores_b0, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    session_ids = np.asarray(session_ids)
    rng = np.random.default_rng(seed)

    def _delta(idx: np.ndarray) -> float:
        r0 = _recall_at_fpr(scores_b0[idx], labels[idx], fpr_target)["recall"]
        rb = _recall_at_fpr(scores_b[idx], labels[idx], fpr_target)["recall"]
        return float(r0 - rb)

    all_idx = np.arange(len(labels))
    estimate = _delta(all_idx)
    stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = session_clustered_indices(session_ids, rng)
        stats[i] = _delta(idx)
    alpha = 1.0 - confidence_level
    lo, hi = np.quantile(stats, [alpha / 2, 1.0 - alpha / 2])
    return BootstrapResult(
        estimate=float(estimate),
        ci_low=float(lo),
        ci_high=float(hi),
        std=float(np.std(stats, ddof=1)),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )


# ---------------------------------------------------------------------------
# Simple mixed-effects helpers
# ---------------------------------------------------------------------------

@dataclass
class SimpleMixedEffectResult:
    """
    Lightweight random-intercept summary: overall mean and between-session variance.
    Not a full REML fit — suitable for diagnostics / paper appendix helpers.
    """

    overall_mean: float
    between_session_var: float
    within_session_var: float
    n_sessions: int
    n_obs: int

    def as_dict(self) -> dict:
        return {
            "overall_mean": self.overall_mean,
            "between_session_var": self.between_session_var,
            "within_session_var": self.within_session_var,
            "n_sessions": self.n_sessions,
            "n_obs": self.n_obs,
            "icc": self.icc,
        }

    @property
    def icc(self) -> float:
        denom = self.between_session_var + self.within_session_var
        return float(self.between_session_var / denom) if denom > 0 else 0.0


def simple_random_intercept(
    values: np.ndarray,
    session_ids: Sequence,
) -> SimpleMixedEffectResult:
    """
    Method-of-moments style variance components for y_ij = μ + u_i + e_ij.
    """
    values = np.asarray(values, dtype=np.float64)
    session_ids = np.asarray(session_ids)
    unique = _unique_preserve(session_ids.tolist())
    session_means = []
    within_ss = 0.0
    within_df = 0
    for s in unique:
        mask = session_ids == s
        v = values[mask]
        session_means.append(float(np.mean(v)))
        if len(v) > 1:
            within_ss += float(np.sum((v - np.mean(v)) ** 2))
            within_df += len(v) - 1
    session_means_arr = np.asarray(session_means, dtype=np.float64)
    overall = float(np.mean(values))
    between_var = float(np.var(session_means_arr, ddof=1)) if len(session_means_arr) > 1 else 0.0
    within_var = float(within_ss / within_df) if within_df > 0 else 0.0
    return SimpleMixedEffectResult(
        overall_mean=overall,
        between_session_var=between_var,
        within_session_var=within_var,
        n_sessions=len(unique),
        n_obs=len(values),
    )
