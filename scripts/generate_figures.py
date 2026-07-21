#!/usr/bin/env python3
"""
Publication-quality figures for the deception-monitor robustness paper.

Outputs (PDF + PNG) under paper/figures/:
  - scenario_figure1.*         operational definition + trajectory bands
  - methodology_figure1.*     full protocol pipeline (all steps)
  - results_main.*            primary: recall ladder + probe Δdet (2 panels)
  - results_cross_monitor.*   cross-monitor: heatmap + gaps + plane (3)
  - results_diagnostics.*     confounds: TOST + drift + stepwise path (3)
  - results_figure.*          alias of results_main (legacy)

Uses matplotlib only. Scientific style (print-safe, no purple AI aesthetic).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def _box(ax, x, y, w, h, text, fc="white", ec=None, fs=7.5, lw=1.0, weight="normal"):
    ec = ec or COLORS["ink"]
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
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


def _load_results(path: Path | None) -> tuple[dict, bool]:
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


def _delta(results: dict, monitor: str, rung: str) -> tuple[float, float, float]:
    """Return (delta, ci_low, ci_high)."""
    entry = results["table_delta_det"][monitor][rung]
    if isinstance(entry, dict):
        d = float(entry["delta_det"])
        ci = entry.get("delta_det_ci", {})
        return d, float(ci.get("ci_low", d)), float(ci.get("ci_high", d))
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
    ax.text(5.0, 8.85, r"visible $a_t$", ha="center", fontsize=7, color=COLORS["muted"])

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
    ax.set_title("B. Operational definition (label)", loc="left", fontsize=10, pad=6)

    _box(ax, 0.5, 7.4, 9.0, 2.0,
         r"Episode $\tau=(x_{1:T},\,a_{1:T},\,s_{1:T})$" + "\n"
         "observations · utterances · residual states",
         fc=soft, ec=edge, fs=8, lw=0.9)
    _box(ax, 0.5, 4.55, 2.8, 1.9, r"Statement" + "\n" + r"$a_{t^{\star}}$",
         fc=soft, ec=edge, fs=8, lw=0.9)
    _box(ax, 3.6, 4.55, 2.8, 1.9, r"Internal fact" + "\n" + r"$b_{\mathrm{int}}$",
         fc=blue, ec=edge, fs=8, lw=0.9)
    _box(ax, 6.7, 4.55, 2.8, 1.9, r"Covert goal" + "\n" + r"$g_{\mathrm{cov}}$",
         fc=sand, ec=edge, fs=8, lw=0.9)
    _arrow(ax, 1.9, 4.55, 5.0, 3.55)
    _arrow(ax, 5.0, 4.55, 5.0, 3.55)
    _arrow(ax, 8.1, 4.55, 5.0, 3.55)
    # Soft fill, ink border — no red outline
    _box(ax, 0.7, 0.55, 8.6, 2.7,
         r"Label $y{=}1$ (deceptive) if and only if" + "\n"
         r"(i) $a_{t^{\star}}$ conflicts with $b_{\mathrm{int}}$" + "\n"
         r"(ii) $a_{t^{\star}}$ advances $g_{\mathrm{cov}}$" + "\n"
         "Scenario-defined · independent of every monitor $D$",
         fc="#f7f7f7", ec=edge, lw=1.0, fs=7.6)

    # ---- C: episode structure ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("C. Episode structure (multi-turn)", loc="left", fontsize=10, pad=6)

    turns = [
        (0.45, "t = 1", "setup", soft),
        (2.75, "t = 2", "tool / act", soft),
        (5.05, r"$t = t^{\star}$", "deceptive turn", rose),
        (7.35, "t = T", "close", soft),
    ]
    for x, code, name, fc in turns:
        _box(ax, x, 5.7, 2.05, 3.0, f"{code}\n{name}", fc=fc, ec=edge, fs=7.4, lw=0.9)
    for i in range(3):
        _arrow(ax, turns[i][0] + 2.05, 7.2, turns[i + 1][0], 7.2)

    _box(ax, 0.45, 2.85, 4.35, 2.0,
         r"Visible channel: $a_t$" + "\n(overseer / surface)",
         fc=blue, ec=edge, fs=7.5, lw=0.9)
    _box(ax, 5.2, 2.85, 4.35, 2.0,
         r"Internal channel: $s_t$" + "\n(probe)",
         fc=rose, ec=edge, fs=7.5, lw=0.9)
    ax.text(5.0, 1.35,
            "Deception is a property of one or more turns, not a single answer.",
            ha="center", fontsize=7.4, color=ink)
    ax.text(5.0, 0.45,
            r"At $t^{\star}$ the transcript can look compliant while $s_{t^{\star}}$ does not.",
            ha="center", fontsize=7, color=COLORS["muted"])

    # ---- D: scenario families ------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("D. Elicitation suite (8 families)", loc="left", fontsize=10, pad=6)

    families = [
        ("Insider trading", "Scheurer-style"),
        ("Sandbagging", "Benton-style"),
        ("Capability concealment", ""),
        ("Goal misreporting", ""),
        ("Tool-use coverup", ""),
        ("Audit evasion", ""),
        ("Safety-filter bypass", ""),
        ("Resource hoarding", ""),
    ]
    y0 = 8.55
    for i, (name, tag) in enumerate(families):
        y = y0 - i * 0.98
        fc = blue if tag else soft
        _box(ax, 0.45, y - 0.32, 6.3, 0.78, name, fc=fc, ec=edge, fs=7.4, lw=0.85)
        if tag:
            ax.text(7.0, y + 0.05, tag, va="center", fontsize=6.8, color=COLORS["muted"])
    ax.text(5.0, 0.4,
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

def make_methodology_figure():
    """Full end-to-end pipeline; Step 11 laid out without overlapping text."""
    fig = plt.figure(figsize=(11.2, 8.2))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.92])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.suptitle(
        "Robustness-gap protocol: end-to-end evaluation pipeline",
        fontsize=13,
        y=0.98,
        fontweight="semibold",
    )

    # ---- Row 1: data & monitors ------------------------------------------------
    ax.text(2, 96.5, "Steps 1–5 · Construct paired episodes and score with three monitors",
            fontsize=9, fontweight="bold", color=COLORS["ink"], ha="left")

    _box(ax, 2, 84, 14, 10,
         "1. Scenarios\n8 agentic families\nfixed seeds",
         fc=COLORS["panel"], fs=7.2)
    _box(ax, 20, 84, 16, 10,
         "2. Paired policies\nbaseline  ↔  adversary\nshared seeds",
         fc=COLORS["band"], fs=7.2)
    _box(ax, 40, 88, 17, 6, "3a. Surface\n(black-box features)", fc="#deebf7", fs=7)
    _box(ax, 40, 80.5, 17, 6, "3b. Linear probe\n(residual layer 40)", fc="#fee0d2", fs=7)
    _box(ax, 40, 73, 17, 6, "3c. CoT reading\n(reasoning control)", fc="#f0f0f0", fs=7)
    _box(ax, 61, 80, 16, 10,
         "4. Calibrate\nthreshold @ 1% FPR\non benign controls",
         fc=COLORS["accent"], fs=7.2)
    _box(ax, 81, 80, 16, 10,
         "5. Score recall\n@ fixed FPR\nper rung $b$",
         fc=COLORS["pass"], fs=7.2)

    _arrow(ax, 16, 89, 20, 89)
    _arrow(ax, 36, 89, 40, 89)
    _arrow(ax, 57, 89, 61, 85)
    _arrow(ax, 77, 85, 81, 85)

    # ---- Row 2: budget ladder --------------------------------------------------
    ax.text(2, 68.5, "Step 6 · Adversary budget ladder  (horizontal axis of $\\Delta\\mathrm{det}$)",
            fontsize=9, fontweight="bold", color=COLORS["ink"], ha="left")

    rungs = [
        (2, "b₀", "Baseline\nno adversary", COLORS["panel"], False),
        (21, "b₁", "Prompt\nconcealment", COLORS["panel"], False),
        (40, "b₂", "SFT-agnostic\nsurface only", COLORS["panel"], False),
        (59, "b₃", "SFT-aware\n+ softplus(probe)", COLORS["adapt"], True),
        (78, "b₄", "RL-aware\nREINFORCE", COLORS["adapt"], True),
    ]
    for x, code, name, fc, adaptive in rungs:
        _box(ax, x, 54, 17, 12, f"{code}\n{name}", fc=fc, fs=7, lw=1.4 if adaptive else 1.0)
    for i in range(4):
        x1 = rungs[i][0] + 17
        x2 = rungs[i + 1][0]
        _arrow(ax, x1, 60, x2, 60)
    ax.text(50, 51.2,
            "Budget ↑ : prompt tokens → LoRA SFT steps → RL steps   |   pink = adaptive (probe in objective)",
            ha="center", fontsize=7, color=COLORS["muted"])

    _box(ax, 55, 44.5, 21, 5.5,
         r"$\mathcal{L}_{b_3}=\mathrm{CE}+\lambda\,\mathrm{softplus}(D)$",
         fc="white", fs=6.5, ec=COLORS["probe"])
    _box(ax, 78, 44.5, 19, 5.5,
         r"$R=\alpha\,\mathrm{dec}-\lambda D-\beta\,\mathrm{surf}$",
         fc="white", fs=6.5, ec=COLORS["probe"])

    # ---- Row 3: gap + confounds + stats ---------------------------------------
    ax.text(2, 40.5, "Steps 7–10 · Robustness gap, confound gates, and inference",
            fontsize=9, fontweight="bold", color=COLORS["ink"], ha="left")

    _box(ax, 2, 26, 28, 12,
         "7. Robustness gap (primary)\n"
         r"$\Delta\mathrm{det}_D(b)=\mathrm{recall}@1\%(0)$"
         "\n"
         r"$-\,\mathrm{recall}@1\%(b)$"
         "\nReport the curve over $b$",
         fc="#fff7bc", fs=7.2)
    _box(ax, 34, 26, 20, 12,
         "8. Broken-agent gate\nTOST equivalence:\nretain ≥80% baseline\ndeception-success",
         fc=COLORS["pass"], fs=7.2)
    _box(ax, 58, 26, 18, 12,
         "9. Threshold drift\nFPR at $b_0$ threshold\non adversarial\nbenign controls",
         fc=COLORS["warn"], fs=7.2)
    _box(ax, 80, 26, 17, 12,
         "10. Inference\nsession-clustered\nbootstrap 95% CIs\npaired seeds",
         fc=COLORS["band"], fs=7.2)

    _arrow(ax, 16, 54, 16, 38)
    _arrow(ax, 30, 32, 34, 32)
    _arrow(ax, 54, 32, 58, 32)
    _arrow(ax, 76, 32, 80, 32)

    # ---- Row 4: Step 11 — primary contrast (no results curve) ----------------
    ax.text(2, 21.8, "Step 11 · Primary scientific contrast",
            fontsize=9, fontweight="bold", color=COLORS["ink"], ha="left")

    _box(ax, 2, 4.0, 46, 16.0,
         "Primary contrast:  $b_2$ (transfer)  vs  $b_3$ (adaptive)\n\n"
         r"$\Delta\mathrm{det}_D(b)=\mathrm{recall}@1\%(0)-\mathrm{recall}@1\%(b)$"
         "\n\n"
         "Positive gap = detection worsened under budget $b$\n"
         "Retention gate + clustered CIs apply at every rung",
         fc=COLORS["panel"], fs=8.0)

    _box(ax, 52, 4.0, 45, 16.0,
         "What Step 11 asks\n\n"
         "Does putting the probe in the loss ($b_3$)\n"
         "open a larger gap than surface-only SFT ($b_2$)?\n\n"
         "The curve is reported in Results;\n"
         "this panel defines the contrast only.",
         fc="#eef6fb", fs=8.0)

    _save(fig, "methodology_figure1")


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def _meta(results: dict, synthetic: bool) -> tuple[str, str]:
    n = results.get("n_sessions", results.get("n_test", "?"))
    model = results.get("model", "Qwen3-32B")
    if isinstance(model, str) and "/" in model:
        model = model.split("/")[-1]
    tag = f"{model} · N={n} · Modal H200"
    if synthetic:
        tag += "  [synthetic]"
    note = (
        r"Error bars: session-clustered bootstrap 95% CIs.  "
        r"$\Delta\mathrm{det}(b)=\mathrm{recall}@1\%\mathrm{FPR}(0)-\mathrm{recall}@1\%\mathrm{FPR}(b)$."
    )
    if synthetic:
        note = results.get("disclaimer", "SYNTHETIC") + "  |  " + note
    return tag, note


def _plane(results: dict) -> list[dict]:
    plane = results.get("outcome_plane") or []
    if plane:
        return plane
    deltas = results.get("table_delta_det", {})
    out = []
    for r in ("b1", "b2", "b3", "b4"):
        if r in deltas.get("surface", {}) and r in deltas.get("linear_probe", {}):
            ds, _, _ = _delta(results, "surface", r)
            dp, _, _ = _delta(results, "linear_probe", r)
            out.append({"rung": r, "delta_det_surface": ds, "delta_det_probe": dp})
    return out


# ---------------------------------------------------------------------------
# Results Fig 1 (main): recall ladder + probe Δdet  — 2 panels
# ---------------------------------------------------------------------------

def make_results_main(results: dict, synthetic: bool):
    recalls = results.get("table_recall_at_1pct_fpr", {})
    deltas = results.get("table_delta_det", {})
    rungs_all = ["b0", "b1", "b2", "b3", "b4"]
    rungs_adv = ["b1", "b2", "b3", "b4"]
    mon_keys = [
        ("surface", COLORS["surface"], "Surface (black-box)"),
        ("linear_probe", COLORS["probe"], "Linear probe (white-box)"),
        ("cot_reading", COLORS["cot"], "CoT reading (control)"),
    ]
    tag, note = _meta(results, synthetic)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # A: recall curves
    ax = axes[0]
    x = np.arange(len(rungs_all))
    styles = {
        "surface": dict(lw=2.0, ms=7, marker="o", ls="-"),
        "linear_probe": dict(lw=2.6, ms=8, marker="s", ls="-"),
        "cot_reading": dict(lw=1.8, ms=7, marker="^", ls="--"),
    }
    for key, color, label in mon_keys:
        if key not in recalls:
            continue
        ys = [recalls[key].get(r, np.nan) for r in rungs_all]
        st = styles[key]
        ax.plot(x, ys, color=color, label=label, **st)
        if key in deltas:
            for i, r in enumerate(rungs_all):
                if r == "b0" or r not in deltas[key]:
                    continue
                _, lo_d, hi_d = _delta(results, key, r)
                r0 = recalls[key]["b0"]
                lo, hi = r0 - hi_d, r0 - lo_d
                ax.vlines(i, lo, hi, color=color, lw=1.2, alpha=0.7)
                ax.hlines([lo, hi], i - 0.08, i + 0.08, color=color, lw=0.9, alpha=0.7)
    if "linear_probe" in recalls:
        ax.annotate(
            r"peak gap at $b_2$",
            xy=(2, recalls["linear_probe"].get("b2", 0.2)),
            xytext=(3.05, 0.78),
            fontsize=8,
            color=COLORS["probe"],
            arrowprops=dict(arrowstyle="->", color=COLORS["probe"], lw=1.1),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(rungs_all)
    ax.set_ylabel(r"Recall @ 1% FPR")
    ax.set_xlabel("Adversary budget")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=7.5)
    ax.set_title("A. Detection recall across the budget ladder", loc="left")
    ax.yaxis.grid(True, color=COLORS["grid"], lw=0.6)
    ax.set_axisbelow(True)
    ax.text(
        0.02, 0.02,
        "Probe & CoT overlap on $b_0$–$b_2$ (identical recall); diverge at $b_3$.",
        transform=ax.transAxes, fontsize=6.5, color=COLORS["muted"], style="italic",
    )

    # B: probe Δdet
    ax = axes[1]
    vals, los, his = [], [], []
    for r in rungs_adv:
        d, lo, hi = _delta(results, "linear_probe", r)
        vals.append(d)
        los.append(d - lo)
        his.append(hi - d)
    ax.bar(
        np.arange(len(rungs_adv)),
        vals,
        color=[COLORS["probe"] if r != "b2" else "#67000d" for r in rungs_adv],
        edgecolor=COLORS["ink"],
        width=0.7,
        yerr=np.vstack([los, his]),
        capsize=3.5,
        error_kw=dict(ecolor=COLORS["ink"], lw=1.1),
    )
    ax.axhline(0, color=COLORS["grid"], lw=0.9)
    ax.set_xticks(np.arange(len(rungs_adv)))
    ax.set_xticklabels(rungs_adv)
    ax.set_ylabel(r"$\Delta\mathrm{det}$ (probe)")
    ax.set_xlabel("Adversary budget")
    ax.set_title(r"B. Probe robustness gap $\Delta\mathrm{det}(b)$", loc="left")
    for i, v in enumerate(vals):
        ax.text(i, v + his[i] + 0.035, f"{v:.2f}", ha="center", fontsize=8, fontweight="semibold")
    ax.set_ylim(min(-0.2, min(vals) - 0.3), max(vals) + max(his) + 0.22)

    fig.suptitle(f"Primary result · {tag}", fontsize=12, y=1.02, fontweight="semibold")
    fig.text(0.5, -0.02, note, ha="center", fontsize=7, color=COLORS["muted"], style="italic")
    fig.tight_layout()
    _save(fig, "results_main", close=False)
    _save(fig, "results_figure", close=True)  # legacy alias = primary figure


# ---------------------------------------------------------------------------
# Results Fig 2: cross-monitor comparison  — 3 panels
# ---------------------------------------------------------------------------

def make_results_cross_monitor(results: dict, synthetic: bool):
    recalls = results.get("table_recall_at_1pct_fpr", {})
    deltas = results.get("table_delta_det", {})
    plane = _plane(results)
    rungs_all = ["b0", "b1", "b2", "b3", "b4"]
    rungs_adv = ["b1", "b2", "b3", "b4"]
    mon_keys = [
        ("surface", COLORS["surface"], "Surface"),
        ("linear_probe", COLORS["probe"], "Probe"),
        ("cot_reading", COLORS["cot"], "CoT"),
    ]
    tag, note = _meta(results, synthetic)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7))

    # A: heatmap
    ax = axes[0]
    mat, row_labels = [], []
    for key, _, label in mon_keys:
        if key not in recalls:
            continue
        mat.append([recalls[key].get(r, np.nan) for r in rungs_all])
        row_labels.append(label)
    mat = np.array(mat, dtype=float)
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(rungs_all)))
    ax.set_xticklabels(rungs_all)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="white" if val > 0.45 else COLORS["ink"], fontweight="semibold",
            )
    ax.set_title("A. Recall heatmap", loc="left")
    ax.set_xlabel("Budget")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    # B: grouped Δdet
    ax = axes[1]
    width = 0.25
    xpos = np.arange(len(rungs_adv))
    for mi, (key, color, label) in enumerate(mon_keys):
        if key not in deltas:
            continue
        ys, yerr_lo, yerr_hi = [], [], []
        for r in rungs_adv:
            d, lo, hi = _delta(results, key, r)
            ys.append(d)
            yerr_lo.append(d - lo)
            yerr_hi.append(hi - d)
        ax.bar(
            xpos + (mi - 1) * width, ys, width=width, color=color,
            edgecolor=COLORS["ink"], linewidth=0.6, label=label,
            yerr=np.vstack([yerr_lo, yerr_hi]), capsize=2,
            error_kw=dict(ecolor=COLORS["ink"], lw=0.8),
        )
    ax.axhline(0, color=COLORS["grid"], lw=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(rungs_adv)
    ax.set_ylabel(r"$\Delta\mathrm{det}$")
    ax.set_xlabel("Budget")
    ax.set_title("B. Gap by monitor family", loc="left")
    ax.legend(loc="best", fontsize=7)
    ax.yaxis.grid(True, color=COLORS["grid"], lw=0.5)
    ax.set_axisbelow(True)

    # C: outcome plane
    ax = axes[2]
    ax.axhline(0, color=COLORS["grid"], lw=0.8)
    ax.axvline(0, color=COLORS["grid"], lw=0.8)
    ax.fill_between([0.2, 1.0], 0.2, 1.0, color="#f0f0f0", alpha=0.7, zorder=0)
    ax.text(0.55, 0.55, "both\nbrittle", ha="center", va="center", fontsize=7.5, color=COLORS["muted"])
    markers = {"b1": "o", "b2": "s", "b3": "D", "b4": "^"}
    xs, ys = [], []
    for pt in plane:
        r = pt["rung"]
        xs.append(pt["delta_det_surface"])
        ys.append(pt["delta_det_probe"])
        ax.scatter(
            pt["delta_det_surface"], pt["delta_det_probe"], s=95,
            marker=markers.get(r, "o"),
            color=COLORS["probe"] if r == "b2" else COLORS["ink"],
            zorder=3, edgecolors="white", linewidths=0.7,
        )
        ax.annotate(r, (pt["delta_det_surface"], pt["delta_det_probe"]),
                    textcoords="offset points", xytext=(6, 5), fontsize=8.5)
    if len(xs) >= 2:
        ax.plot(xs, ys, color=COLORS["muted"], lw=1.2, ls="--", zorder=2)
    ax.set_xlabel(r"$\Delta\mathrm{det}$ surface")
    ax.set_ylabel(r"$\Delta\mathrm{det}$ probe")
    ax.set_title("C. Outcome plane", loc="left")
    ax.set_xlim(-0.35, 0.85)
    ax.set_ylim(-0.2, 0.85)

    fig.suptitle(f"Cross-monitor comparison · {tag}", fontsize=12, y=1.04, fontweight="semibold")
    fig.text(0.5, -0.04, note + "  Negative gap = recall rose vs baseline.",
             ha="center", fontsize=7, color=COLORS["muted"], style="italic")
    fig.tight_layout()
    _save(fig, "results_cross_monitor")


# ---------------------------------------------------------------------------
# Results Fig 3: diagnostics / confounds  — 3 panels
# ---------------------------------------------------------------------------

def make_results_diagnostics(results: dict, synthetic: bool):
    recalls = results.get("table_recall_at_1pct_fpr", {})
    tost = results.get("tost_broken_agent_gate", {})
    drift = results.get("threshold_drift_surface", {})
    rungs_all = ["b0", "b1", "b2", "b3", "b4"]
    rungs_adv = ["b1", "b2", "b3", "b4"]
    tag, note = _meta(results, synthetic)
    xpos = np.arange(len(rungs_adv))

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7))

    # A: TOST
    ax = axes[0]
    succ0 = [float(tost.get(r, {}).get("mean_task_success_b0", np.nan)) for r in rungs_adv]
    succb = [float(tost.get(r, {}).get("mean_task_success_b", np.nan)) for r in rungs_adv]
    ax.bar(xpos - 0.18, succ0, width=0.35, color=COLORS["panel"], edgecolor=COLORS["ink"], label="Baseline")
    ax.bar(xpos + 0.18, succb, width=0.35, color=COLORS["pass"], edgecolor=COLORS["ink"], label="Adversary")
    for i, r in enumerate(rungs_adv):
        gate = tost.get(r, {}).get("gate", "?")
        ax.text(
            i, 1.06, gate.upper(), ha="center", fontsize=7.5, fontweight="bold",
            color=COLORS["honest"] if gate == "pass" else COLORS["deceptive"],
        )
    ax.axhline(0.8, color=COLORS["deceptive"], ls="--", lw=1.1, label="80% margin")
    ax.set_xticks(xpos)
    ax.set_xticklabels(rungs_adv)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Deception-success proxy")
    ax.set_xlabel("Budget")
    ax.set_title("A. Broken-agent gate (TOST)", loc="left")
    ax.legend(loc="lower right", fontsize=7)

    # B: threshold drift
    ax = axes[1]
    rel = [float(drift.get(r, {}).get("relative_drift", np.nan)) for r in rungs_adv]
    fpr = [float(drift.get(r, {}).get("fpr_b_at_b0_threshold", np.nan)) for r in rungs_adv]
    ax2 = ax.twinx()
    ax.bar(xpos - 0.15, rel, width=0.3, color=COLORS["surface"], edgecolor=COLORS["ink"],
           label="Rel. threshold drift")
    ax2.plot(xpos + 0.15, fpr, "o-", color=COLORS["deceptive"], lw=1.8, ms=7, label=r"FPR @ $b_0$ thr.")
    ax.axhline(0, color=COLORS["grid"], lw=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(rungs_adv)
    ax.set_ylabel("Relative threshold drift")
    ax2.set_ylabel(r"FPR at fixed $b_0$ threshold")
    ax.set_xlabel("Budget")
    ax.set_title("B. Surface threshold drift", loc="left")
    ax.spines["right"].set_visible(True)
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=6.5)

    # C: stepwise probe path
    ax = axes[2]
    probe_rec = [recalls.get("linear_probe", {}).get(r, np.nan) for r in rungs_all]
    ax.plot(np.arange(len(rungs_all)), probe_rec, color=COLORS["probe"], lw=2.4, marker="o", ms=8, zorder=3)
    for i in range(len(rungs_all) - 1):
        ax.annotate(
            "",
            xy=(i + 1, probe_rec[i + 1]),
            xytext=(i, probe_rec[i]),
            arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], lw=1.1),
        )
        d = probe_rec[i] - probe_rec[i + 1]
        mid_y = (probe_rec[i] + probe_rec[i + 1]) / 2
        ax.text(i + 0.5, mid_y + 0.045, f"{d:+.2f}", ha="center", fontsize=7.5, color=COLORS["muted"])
    ax.set_xticks(np.arange(len(rungs_all)))
    ax.set_xticklabels(rungs_all)
    ax.set_ylabel("Probe recall")
    ax.set_xlabel("Budget")
    ax.set_ylim(0, 0.75)
    ax.set_title("C. Probe recall path (stepwise $\\Delta$)", loc="left")
    ax.yaxis.grid(True, color=COLORS["grid"], lw=0.6)

    agree = results.get("rule_label_agreement_b0", "—")
    fig.suptitle(f"Confound diagnostics · {tag}", fontsize=12, y=1.04, fontweight="semibold")
    fig.text(
        0.5, -0.04,
        note + f"  Rule-label agreement @ $b_0$ = {agree}.",
        ha="center", fontsize=7, color=COLORS["muted"], style="italic",
    )
    fig.tight_layout()
    _save(fig, "results_diagnostics")


def make_results_figure(results: dict, synthetic: bool):
    """Write the three separated result figures."""
    make_results_main(results, synthetic)
    # remake main without double-close: make_results_main already saves both names
    make_results_cross_monitor(results, synthetic)
    make_results_diagnostics(results, synthetic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=None, help="JSON results path")
    parser.add_argument(
        "--only",
        choices=["scenario", "methodology", "results", "all"],
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


if __name__ == "__main__":
    main()
