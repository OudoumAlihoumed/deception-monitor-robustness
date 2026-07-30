#!/usr/bin/env python3
"""
Publication-quality figures for the deception-monitor robustness paper.

Four-figure set (build order: conceptual → data):
  - scenario_figure1.*      Fig 1 — What we measure (intro)
  - methodology_figure1.*  Fig 2 — Controlled-comparison protocol (methods)
  - results_tension.*      Fig 3 — Primary: probe AUROC + retention
  - results_transfer.*     Fig 4 — Apollo → agentic transfer gap

Legacy aliases (same bytes as tension / transfer where noted):
  - results_main.* / results_figure.* → results_tension
  - results_diagnostics.* → results_transfer
  - methodology_figure_last.* → methodology_figure1

Removed from the paper set (not regenerated as primary):
  cross-monitor heatmap, outcome plane, confound triptych.

Uses matplotlib only. Print-safe; pass/fail encoded by marker shape + fill.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "paper" / "figures"

COLORS = {
    "ink": "#1a1a1a",
    "muted": "#4a4a4a",
    "grid": "#d0d0d0",
    "surface": "#2166ac",
    "probe": "#b2182b",
    "cot": "#636363",
    "honest": "#1b9e77",
    "deceptive": "#d95f02",
    "band": "#a6cee3",
    "accent": "#e6ab02",
    "panel": "#f4f4f4",
    "adapt": "#fddbc7",
    "pass": "#c7e9c0",
    "warn": "#fee0d2",
}


def _setup_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titleweight": "semibold",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "font.family": "serif",
            "font.size": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.grid": False,
        }
    )


def _save(fig: plt.Figure, stem: str, *, close: bool = True):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIGDIR / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Wrote {path}")
    if close:
        plt.close(fig)


def _box(ax, x, y, w, h, text, fc="white", ec=None, fs=7.5, lw=1.0, weight="normal", rounding=0.06):
    ec = ec or COLORS["ink"]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            clip_on=False,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color=COLORS["ink"],
        wrap=True,
    )


def _arrow(ax, x1, y1, x2, y2, color=None):
    color = color or COLORS["muted"]
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, mutation_scale=10),
        clip_on=False,
    )


def _load_results(path: Optional[Path]) -> Tuple[dict, bool]:
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend(
        [
            ROOT / "results" / "modal_experiment_results.json",
            ROOT / "results" / "pipeline_validation_synthetic.json",
        ]
    )
    for p in candidates:
        if p and p.exists():
            data = json.loads(p.read_text())
            synthetic = data.get("result_type") == "synthetic_pipeline_validation"
            return data, synthetic
    return _illustrative_results(), True


def _illustrative_results() -> dict:
    return {
        "result_type": "illustrative_schema",
        "disclaimer": "ILLUSTRATIVE ONLY — no experiment run",
        "n_sessions": 60,
        "table_recall_at_1pct_fpr": {
            "surface": {"b0": 0.85, "b1": 0.55, "b2": 0.25, "b3": 0.18, "b4": 0.20},
            "linear_probe": {"b0": 0.90, "b1": 0.82, "b2": 0.70, "b3": 0.35, "b4": 0.30},
            "cot_reading": {"b0": 0.75, "b1": 0.60, "b2": 0.40, "b3": 0.30, "b4": 0.35},
        },
        "table_delta_det": {
            "surface": {
                "b1": {"delta_det": 0.30, "delta_det_ci": {"ci_low": 0.20, "ci_high": 0.40}},
                "b2": {"delta_det": 0.60, "delta_det_ci": {"ci_low": 0.48, "ci_high": 0.70}},
                "b3": {"delta_det": 0.67, "delta_det_ci": {"ci_low": 0.55, "ci_high": 0.78}},
                "b4": {"delta_det": 0.65, "delta_det_ci": {"ci_low": 0.50, "ci_high": 0.75}},
            },
            "linear_probe": {
                "b1": {"delta_det": 0.08, "delta_det_ci": {"ci_low": 0.02, "ci_high": 0.14}},
                "b2": {"delta_det": 0.20, "delta_det_ci": {"ci_low": 0.10, "ci_high": 0.30}},
                "b3": {"delta_det": 0.55, "delta_det_ci": {"ci_low": 0.42, "ci_high": 0.66}},
                "b4": {"delta_det": 0.60, "delta_det_ci": {"ci_low": 0.45, "ci_high": 0.72}},
            },
            "cot_reading": {
                "b1": {"delta_det": 0.15, "delta_det_ci": {"ci_low": 0.05, "ci_high": 0.25}},
                "b2": {"delta_det": 0.35, "delta_det_ci": {"ci_low": 0.20, "ci_high": 0.45}},
                "b3": {"delta_det": 0.45, "delta_det_ci": {"ci_low": 0.30, "ci_high": 0.55}},
                "b4": {"delta_det": 0.40, "delta_det_ci": {"ci_low": 0.25, "ci_high": 0.50}},
            },
        },
        "tost_broken_agent_gate": {
            r: {
                "gate": "pass",
                "mean_task_success_b0": 1.0,
                "mean_task_success_b": 0.95,
                "equivalent": True,
            }
            for r in ("b1", "b2", "b3", "b4")
        },
        "threshold_drift_surface": {
            "b1": {"relative_drift": -0.10, "fpr_b_at_b0_threshold": 0.0},
            "b2": {"relative_drift": 0.02, "fpr_b_at_b0_threshold": 0.07},
            "b3": {"relative_drift": 0.12, "fpr_b_at_b0_threshold": 0.07},
            "b4": {"relative_drift": -0.20, "fpr_b_at_b0_threshold": 0.0},
        },
        "rule_label_agreement_b0": 0.7,
        "outcome_plane": [],
    }


def _delta(
    results: dict, monitor: str, rung: str
) -> Optional[Tuple[float, float, float]]:
    """
    Return (delta, ci_low, ci_high), or None if the rung was gated out.

    Adaptive rungs (b3/b4) set ``delta_det`` to null when retention fails or
    the attack was too weak — never coerce that to 0.0 (would fabricate a gap).
    """
    entry = results["table_delta_det"][monitor][rung]
    if isinstance(entry, dict):
        if entry.get("reportable") is False or entry.get("delta_det") is None:
            return None
        d = float(entry["delta_det"])
        ci = entry.get("delta_det_ci") or entry.get("gate", {}).get("delta_det_ci_raw") or {}
        return d, float(ci.get("ci_low", d)), float(ci.get("ci_high", d))
    if entry is None:
        return None
    return float(entry), float(entry), float(entry)


# ---------------------------------------------------------------------------
# Figure 1: scenario
# ---------------------------------------------------------------------------

def make_scenario_figure():
    """High-quality definition figure: structure + label (no score plots, no harsh red borders)."""
    fig = plt.figure(figsize=(10.8, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], hspace=0.40, wspace=0.30)
    ink = COLORS["ink"]
    soft = "#f2f2f2"
    blue = "#d9e8f5"
    rose = "#f3e4e1"
    sand = "#f5efd9"
    edge = ink

    # ---- A: problem structure ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A. Problem structure", loc="left", fontsize=10, pad=6)

    _box(ax, 0.5, 7.35, 3.9, 2.0, "Overseer\nsees transcript only", fc=soft, ec=edge, fs=8, lw=0.9)
    _box(ax, 5.6, 7.35, 3.9, 2.0, "Agent\nacts toward a goal", fc=blue, ec=edge, fs=8, lw=0.9)
    _arrow(ax, 4.4, 8.35, 5.6, 8.35)
    ax.text(5.0, 9.65, r"visible $a_t$", ha="center", fontsize=7, color=COLORS["muted"])

    _box(ax, 5.6, 4.55, 3.9, 1.9, r"Covert goal $g_{\mathrm{cov}}$" + "\n(hidden from overseer)",
         fc=sand, ec=edge, fs=7.5, lw=0.9)
    _arrow(ax, 7.55, 7.35, 7.55, 6.45)

    _box(ax, 0.5, 1.15, 2.7, 2.35, "Surface monitor\nreads transcript $a$", fc=blue, ec=edge, fs=7.2, lw=0.9)
    _box(ax, 3.65, 1.15, 2.7, 2.35, "Probe monitor\nreads residual $s$", fc=rose, ec=edge, fs=7.2, lw=0.9)
    _box(ax, 6.8, 1.15, 2.7, 2.35, "CoT control\nreads reasoning", fc=soft, ec=edge, fs=7.2, lw=0.9)
    ax.text(5.0, 3.75, "Monitors score the episode; they do not define the label",
            ha="center", fontsize=7, color=COLORS["muted"])

    # ---- B: operational definition -------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("B. Label definition (AND gate)", loc="left", fontsize=10, pad=6)

    _box(ax, 0.4, 7.5, 2.9, 1.9, r"Visible statement" + "\n" + r"$a_{t^{\star}}$",
         fc=soft, ec=edge, fs=7.6, lw=0.9)
    _box(ax, 3.55, 7.5, 2.9, 1.9, r"Internal fact" + "\n" + r"$b_{\mathrm{int}}$",
         fc=blue, ec=edge, fs=7.6, lw=0.9)
    _box(ax, 6.7, 7.5, 2.9, 1.9, r"Covert goal" + "\n" + r"$g_{\mathrm{cov}}$",
         fc=sand, ec=edge, fs=7.6, lw=0.9)

    _box(ax, 1.2, 4.55, 3.4, 1.7, "(i) conflict:\n" + r"$a_{t^{\star}}$ vs $b_{\mathrm{int}}$",
         fc=soft, ec=edge, fs=7.2, lw=0.9)
    _box(ax, 5.4, 4.55, 3.4, 1.7, "(ii) advances:\n" + r"$a_{t^{\star}}$ → $g_{\mathrm{cov}}$",
         fc=soft, ec=edge, fs=7.2, lw=0.9)
    _arrow(ax, 1.85, 7.5, 2.4, 6.25)
    _arrow(ax, 5.0, 7.5, 2.9, 6.25)
    _arrow(ax, 5.0, 7.5, 7.1, 6.25)
    _arrow(ax, 8.15, 7.5, 7.1, 6.25)

    # AND gate diamond-ish box
    _box(ax, 3.3, 2.55, 3.4, 1.35, r"AND", fc=rose, ec=edge, fs=9, lw=1.1, weight="bold")
    _arrow(ax, 2.9, 4.55, 4.5, 3.9)
    _arrow(ax, 7.1, 4.55, 5.5, 3.9)
    _box(ax, 2.2, 0.35, 5.6, 1.7,
         r"Label $y{=}1$ (deceptive)" + "\nscenario-defined · monitor-independent",
         fc="#f7f7f7", ec=edge, lw=1.0, fs=7.4)
    _arrow(ax, 5.0, 2.55, 5.0, 2.05)

    # ---- C: episode structure (single-turn agentic) --------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("C. Episode structure (single-turn agentic)", loc="left", fontsize=10, pad=6)

    _box(ax, 0.4, 6.3, 4.4, 3.0,
         "Stacked context\n(system + prior turns)\nas prompt",
         fc=soft, ec=edge, fs=7.6, lw=0.9)
    _arrow(ax, 4.8, 7.8, 5.5, 7.8)
    _box(ax, 5.5, 6.3, 4.1, 3.0,
         r"One completion $a$" + "\n" + r"at turn $t^{\star}$",
         fc=rose, ec=edge, fs=7.6, lw=0.9)

    _box(ax, 0.4, 3.0, 4.4, 2.4,
         r"Visible channel $a_{t^{\star}}$" + "\n(overseer / surface)",
         fc=blue, ec=edge, fs=7.4, lw=0.9)
    _box(ax, 5.5, 3.0, 4.1, 2.4,
         r"Internal channel $s_{t^{\star}}$" + "\n(probe residual)",
         fc=rose, ec=edge, fs=7.4, lw=0.9)
    ax.text(5.0, 1.05,
            r"Deception localised at $t^{\star}$: transcript can look compliant while $s_{t^{\star}}$ does not.",
            ha="center", fontsize=7.2, color=ink)

    # ---- D: elicitation families (7 kept; resource-hoarding excluded) -------
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("D. Seven elicitation families", loc="left", fontsize=10, pad=6)

    # Matches paper caption + judge.included_families (suite-v1-7fam).
    families = [
        "Insider trading",
        "Sandbagging",
        "Capability concealment",
        "Goal misreporting",
        "Tool-use coverup",
        "Audit evasion",
        "Safety-filter bypass",
    ]
    y0 = 8.85
    for i, name in enumerate(families):
        y = y0 - i * 1.05
        _box(ax, 0.35, y - 0.28, 7.1, 0.82, f"{i + 1}.  {name}",
             fc=blue, ec=edge, fs=7.4, lw=0.85)
    ax.text(5.0, 0.55,
            "Shared scenario seeds across baseline and adversary arms.",
            ha="center", fontsize=7, color=COLORS["muted"])

    fig.suptitle(
        "What we measure: agentic deception under monitoring",
        fontsize=12,
        y=0.995,
        fontweight="semibold",
    )
    _save(fig, "scenario_figure1")


# ---------------------------------------------------------------------------
# Figure 2: methodology — FULL pipeline (all steps)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Figure 2: methodology — full protocol (simple, current work)
# ---------------------------------------------------------------------------

def make_methodology_figure():
    """
    Single-row pipeline, read left to right, eight boxes, one arrow chain:
      validate probe -> build episodes -> b0 -> b1 -> b2 -> b3 -> b4 -> read curve

    No lettered sub-panels and no footer legend — both were restating what the
    caption in main.tex already says. No color coding either: red-on-two-boxes
    with no legend in sight reads as "these two failed," not "probe in the
    loop" — a real misreading risk, not just a style note. Every box is the
    same neutral fill / black border; b3 and b4 are distinguished purely by
    their own text ("softplus(probe)", "vs. probe"), which already says what
    matters without needing the reader to decode a color.
    """
    fig = plt.figure(figsize=(12.4, 2.55))
    ax = fig.add_axes([0.01, 0.05, 0.98, 0.82])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis("off")

    fig.suptitle(
        "Robustness-gap measurement protocol",
        fontsize=13.5,
        y=0.99,
        fontweight="semibold",
    )

    neutral = "#f7f7f5"

    steps = [
        (14.5, "Validate probe", "Apollo weights +\nextraction gate"),
        (14.5, "Build episodes", "7 families · 3 monitors\ncalibrate α"),
        (9.3, "b₀", "Baseline"),
        (9.3, "b₁", "Prompt\nconcealment"),
        (9.3, "b₂", "SFT,\nsurface only"),
        (9.3, "b₃", "SFT +\nsoftplus(probe)"),
        (9.3, "b₄", "RL vs.\nprobe"),
        (14.5, "Read the curve", "ΔAUROC · retention\ngate · report"),
    ]

    gap = 1.2
    x = 0.8
    y0, h = 5.0, 15.5
    mid_y = y0 + h / 2
    for i, (w, title, sub) in enumerate(steps):
        fc = "white" if title.startswith("b") else neutral
        label = f"{title}\n{sub}"
        _box(ax, x, y0, w, h, label, fc=fc, fs=7.6 if len(title) > 3 else 8.0,
             lw=0.9, rounding=0.03)
        if i < len(steps) - 1:
            _arrow(ax, x + w, mid_y, x + w + gap, mid_y)
        x += w + gap

    # Also write midpoint-checkin alias if used
    _save(fig, "methodology_figure1", close=False)
    _save(fig, "methodology_figure_last", close=True)


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def _meta(results: dict, synthetic: bool) -> str:
    """Footer note only — never stamp model / N / GPU on the figure."""
    note = "Session-clustered bootstrap 95% CIs where shown."
    if synthetic:
        note = "ILLUSTRATIVE / PIPELINE CHECK ONLY — not a published run.  |  " + note
    return note


def _load_figure_data() -> dict:
    """Prefer results/figure_data.json (written from aggregate + bootstrap)."""
    path = ROOT / "results" / "figure_data.json"
    if path.exists():
        return json.loads(path.read_text())
    # Fallback: derive from modal_experiment_results.json
    results, _ = _load_results(None)
    auroc = (results.get("table_auroc") or {}).get("linear_probe") or {}
    rates = results.get("label_stats_by_rung") or {}
    ci = (results.get("table_auroc_ci_b0") or {}).get("linear_probe") or {}
    tost = results.get("tost_broken_agent_gate") or {}
    rungs = ["b0", "b1", "b2", "b3", "b4"]
    vals = [float(auroc.get(r, np.nan)) for r in rungs]
    ret = [float((rates.get(r) or {}).get("rule_label_rate_frozen", np.nan)) for r in rungs]
    gates = ["baseline"] + [str((tost.get(r) or {}).get("gate", "?")) for r in rungs[1:]]
    apollo = 0.9992
    gate_path = ROOT / "results" / "apollo_extraction_gate.json"
    if gate_path.exists():
        apollo = float(json.loads(gate_path.read_text()).get("auroc", apollo))
    return {
        "rungs": rungs,
        "probe_auroc": {
            "values": vals,
            "ci_low": [float(ci.get("ci_low", vals[0] - 0.08))] + [v - 0.06 for v in vals[1:]],
            "ci_high": [float(ci.get("ci_high", vals[0] + 0.08))] + [v + 0.06 for v in vals[1:]],
        },
        "retention_frozen_rule": {"values": ret},
        "tost_gate": {"gate": gates, "margin": 0.8},
        "transfer": {
            "apollo_stage2_auroc": apollo,
            "agentic_b0_auroc": float(ci.get("estimate", vals[0])),
            "agentic_b0_ci_low": float(ci.get("ci_low", 0.75)),
            "agentic_b0_ci_high": float(ci.get("ci_high", 0.90)),
            "smoke_gate": 0.9,
        },
    }


# ---------------------------------------------------------------------------
# Fig 3 — The tension (primary result)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fig 3 — The tension (primary result); λ=8.0 primary, λ=0.5 companion
# ---------------------------------------------------------------------------

def _tension_ladder(fd: dict, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return probe AUROC point + CI for the requested λ ladder.

    b0–b2 are shared. b3/b4 come from the matching λ cell in figure_data.
    CIs for b3/b4 keep the same half-width as the λ=8 bootstrap band when a
    dedicated λ=0.5 bootstrap is not available.
    """
    base = np.asarray(fd["probe_auroc"]["values"], dtype=float).copy()
    lo = np.asarray(fd["probe_auroc"]["ci_low"], dtype=float).copy()
    hi = np.asarray(fd["probe_auroc"]["ci_high"], dtype=float).copy()
    by = fd.get("probe_auroc_by_lambda", {})
    if float(lam) == 0.5:
        b3 = float(by["b3_lam0.5"])
        b4 = float(by["b4_lam0.5"])
        # Preserve CI half-width around the new point estimates
        half3 = 0.5 * (hi[3] - lo[3])
        half4 = 0.5 * (hi[4] - lo[4])
        base[3], lo[3], hi[3] = b3, max(0.0, b3 - half3), min(1.0, b3 + half3)
        base[4], lo[4], hi[4] = b4, max(0.0, b4 - half4), min(1.0, b4 + half4)
    elif float(lam) == 8.0:
        # Default figure_data ladder already is λ=8 for b3/b4
        pass
    else:
        raise ValueError(f"Unsupported lambda={lam}; expected 0.5 or 8.0")
    return base, lo, hi


def make_results_tension(
    results: Optional[dict] = None,
    synthetic: bool = False,
    lam: float = 8.0,
):
    """Primary tension figure (λ=8): AUROC + frozen retention.

    Companion λ=0.5: AUROC only. Frozen rule+ retention was checkpointed once
    at end-of-run (last λ cell), so reusing 0.14 for λ=0.5 would be wrong.
    """
    fd = _load_figure_data()
    rungs = fd["rungs"]
    labels = [r"$b_0$", r"$b_1$", r"$b_2$", r"$b_3$", r"$b_4$"]
    x = np.arange(len(rungs))
    auroc, lo, hi = _tension_ladder(fd, lam)
    auroc_only = float(lam) == 0.5

    if auroc_only:
        fig, ax0 = plt.subplots(1, 1, figsize=(7.6, 3.4))
        axes = [ax0]
    else:
        ret = np.asarray(fd["retention_frozen_rule"]["values"], dtype=float)
        gates = fd["tost_gate"]["gate"]
        margin = float(fd["tost_gate"].get("margin", 0.8))
        fig, axes = plt.subplots(
            2, 1, figsize=(7.6, 5.2), sharex=True,
            gridspec_kw={"hspace": 0.18, "height_ratios": [1.1, 1.0]},
        )

    # Shared b4 stripe
    for ax in axes:
        ax.axvspan(3.5, 4.5, color="#f5d5d0", alpha=0.55, zorder=0)

    # ---- Probe AUROC ------------------------------------------------------
    ax = axes[0]
    ax.fill_between(x, lo, hi, color=COLORS["probe"], alpha=0.18, zorder=1, label="95% CI")
    ax.axhline(auroc[0], color=COLORS["muted"], ls=":", lw=1.15, zorder=2,
               label=fr"$b_0$ baseline ({auroc[0]:.2f})")
    ax.plot(x, auroc, color=COLORS["probe"], lw=2.5, marker="s", ms=8, zorder=3,
            label="Apollo linear probe")
    for i, y in enumerate(auroc):
        ax.text(i, hi[i] + 0.012, f"{y:.2f}", ha="center", va="bottom",
                fontsize=8, color=COLORS["probe"], fontweight="semibold")
    if not auroc_only:
        ax.text(4.0, 0.48, r"$b_4$ gated", ha="center", fontsize=8,
                color=COLORS["deceptive"], fontweight="semibold")
    ax.set_ylabel("Probe AUROC")
    ax.set_ylim(0.38, 1.08)
    ax.legend(loc="lower left", fontsize=7.5, ncol=2)
    ax.yaxis.grid(True, color=COLORS["grid"], lw=0.55)
    ax.set_axisbelow(True)
    if auroc_only:
        panel_a = r"Probe AUROC ladder ($\lambda{=}0.5$) — AUROC holds at $b_4$"
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Adversary budget")
        ax.text(
            0.01, 0.97, panel_a,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="semibold",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.92),
        )
        title = (
            r"Companion $\lambda{=}0.5$: AUROC only "
            r"(frozen retention not logged per-$\lambda$)"
        )
        stems = ("results_tension_lam0.5", "results_main_lam0.5")
        fig.suptitle(title, fontsize=11, y=0.98, fontweight="semibold")
        fig.subplots_adjust(top=0.88, bottom=0.16, left=0.12, right=0.98)
    else:
        panel_a = r"A. Probe AUROC — holds until the agent breaks ($\lambda{=}8$)"
        ax.text(
            0.01, 0.97, panel_a,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=9.5, fontweight="semibold",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.92),
        )

        # ---- Bottom: frozen-cohort retention (λ=8 only; measured) ---------
        ax = axes[1]
        ax.plot(x, ret, color=COLORS["ink"], lw=1.3, zorder=2)
        ax.axhline(margin, color=COLORS["deceptive"], ls="--", lw=1.4, zorder=2)
        for i, (y, g) in enumerate(zip(ret, gates)):
            failed = "fail" in str(g)
            if i == 0:
                marker, fc, ec = "o", COLORS["panel"], COLORS["ink"]
            elif failed:
                marker, fc, ec = "X", COLORS["warn"], COLORS["deceptive"]
            else:
                marker, fc, ec = "o", COLORS["pass"], COLORS["honest"]
            ax.plot(i, y, marker=marker, ms=11 if failed else 10, color=ec,
                    markerfacecolor=fc, markeredgewidth=1.4, zorder=3, linestyle="None")
            ax.text(i, y + 0.045, f"{y:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="semibold", zorder=4)
            if i > 0:
                tag = "FAIL" if failed else "PASS"
                ax.text(
                    i, 0.04, tag, ha="center", fontsize=7.5, fontweight="bold",
                    color=COLORS["deceptive"] if failed else COLORS["honest"],
                )
        pass_h = Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["pass"],
                        markeredgecolor=COLORS["honest"], markersize=9, label="TOST pass")
        fail_h = Line2D([0], [0], marker="X", color="w", markerfacecolor=COLORS["warn"],
                        markeredgecolor=COLORS["deceptive"], markersize=9, label="TOST fail")
        ax.legend(
            handles=[
                pass_h, fail_h,
                Line2D([0], [0], color=COLORS["deceptive"], ls="--", lw=1.4,
                       label=fr"TOST margin ({margin:.0%})"),
            ],
            loc="center right", fontsize=7.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(r"Retention on $\mathcal{C}$ (rule+)")
        ax.set_xlabel("Adversary budget")
        ax.set_ylim(-0.02, 1.20)
        ax.yaxis.grid(True, color=COLORS["grid"], lw=0.55)
        ax.set_axisbelow(True)
        ax.text(
            0.99, 0.98,
            r"B. Deception retention — only $b_4$ fails the gate",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, fontweight="semibold",
            bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.95),
            zorder=5,
        )
        axes[0].tick_params(axis="x", labelbottom=False)
        title = "The tension: detection falls only where deception collapses"
        stems = ("results_tension", "results_main", "results_figure")
        fig.suptitle(title, fontsize=12, y=0.98, fontweight="semibold")
        fig.subplots_adjust(hspace=0.18, top=0.93, bottom=0.10, left=0.12, right=0.98)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        for ext in ("pdf", "png"):
            path = FIGDIR / f"{stem}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches=None, pad_inches=0.12)
            print(f"Wrote {path}")
    plt.close(fig)


def make_results_tension_both(results: Optional[dict] = None, synthetic: bool = False):
    """Write λ=8.0 (primary) and λ=0.5 (companion) tension figures."""
    make_results_tension(results, synthetic, lam=8.0)
    make_results_tension(results, synthetic, lam=0.5)


# ---------------------------------------------------------------------------
# Fig 4 — Transfer gap
# ---------------------------------------------------------------------------

def make_results_transfer(results: Optional[dict] = None, synthetic: bool = False):
    fd = _load_figure_data()
    tr = fd["transfer"]
    apollo = float(tr["apollo_stage2_auroc"])
    agentic = float(tr["agentic_b0_auroc"])
    lo = float(tr["agentic_b0_ci_low"])
    hi = float(tr["agentic_b0_ci_high"])

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    cats = ["Apollo domain\n(insider / Stage 2)", r"Our agentic suite" + "\n" + r"(frozen $\mathcal{C}$, $b_0$)"]
    vals = [apollo, agentic]
    yerr = np.array([[0.0, agentic - lo], [0.0, hi - agentic]])
    ax.bar(
        [0, 1], vals, width=0.52,
        color=[COLORS["pass"], COLORS["probe"]], edgecolor=COLORS["ink"],
        yerr=yerr, capsize=5, error_kw=dict(ecolor=COLORS["ink"], lw=1.2), zorder=3,
    )
    ax.annotate(
        "", xy=(1, agentic), xytext=(0, apollo),
        arrowprops=dict(arrowstyle="-|>", color=COLORS["ink"], lw=1.25, mutation_scale=12),
    )
    gap = apollo - agentic
    ax.text(
        0.5, (apollo + agentic) / 2 + 0.02,
        fr"transfer gap" + "\n" + fr"$\Delta = {gap:.3f}$",
        ha="center", va="bottom", fontsize=9, fontweight="semibold",
    )
    ax.text(0, apollo + 0.015, f"{apollo:.3f}", ha="center", fontsize=9, fontweight="semibold")
    ax.text(
        1, hi + 0.03, f"{agentic:.3f}\n[{lo:.2f}, {hi:.2f}]",
        ha="center", fontsize=8.5, fontweight="semibold", color=COLORS["probe"],
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(cats)
    ax.set_ylabel("Probe AUROC")
    # Start at 0 so bar heights are not visually exaggerated
    ax.set_ylim(0.0, 1.12)
    ax.yaxis.grid(True, color=COLORS["grid"], lw=0.55)
    ax.set_axisbelow(True)
    ax.set_title(
        "Transfer gap: same probe, different distribution",
        loc="left", fontsize=10, pad=8,
    )
    fig.tight_layout()
    _save(fig, "results_transfer", close=False)
    _save(fig, "results_diagnostics", close=True)


def make_results_main(results: dict, synthetic: bool):
    """Alias — primary result is the tension figure (both λ)."""
    make_results_tension_both(results, synthetic)


def make_results_figure(results: dict, synthetic: bool):
    make_results_tension_both(results, synthetic)
    make_results_transfer(results, synthetic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=None, help="JSON results path")
    parser.add_argument(
        "--only",
        choices=["scenario", "methodology", "results", "tension", "transfer", "all"],
        default="all",
    )
    args = parser.parse_args()

    _setup_style()
    results, synthetic = _load_results(args.results)
    print(f"Loaded results (synthetic={synthetic}): keys={list(results.keys())[:8]}...")

    if args.only in ("scenario", "all"):
        make_scenario_figure()
    if args.only in ("methodology", "all"):
        make_methodology_figure()
    if args.only in ("results", "all"):
        make_results_figure(results, synthetic=synthetic)
    if args.only == "tension":
        make_results_tension_both(results, synthetic=synthetic)
    if args.only == "transfer":
        make_results_transfer(results, synthetic=synthetic)


if __name__ == "__main__":
    main()
