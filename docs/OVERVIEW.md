# Local overview — what to open on your laptop

This file is the map of the project **on your machine**.  
GitHub keeps some empty folders with `.gitkeep` only (placeholders). Your **paper, results, and tests live here locally**.

---

## Start here (review)

| What | Open this path |
|------|----------------|
| **Paper PDF (for review)** | [`paper/main.pdf`](../paper/main.pdf) |
| **Paper source** | [`paper/main.tex`](../paper/main.tex) |
| **References** | [`paper/references.bib`](../paper/references.bib) |
| **Repo README** | [`README.md`](../README.md) |
| **Methodology (code-faithful)** | [`docs/METHODOLOGY.md`](METHODOLOGY.md) |
| **Math of $b_0$–$b_4$ (deep)** | [`docs/MATH_LADDER.md`](MATH_LADDER.md) |
| **Mentor checklist (decisions)** | [`docs/MENTOR_CHECKLIST.md`](MENTOR_CHECKLIST.md) |
| **Midpoint check-in** | [`paper/midpoint_checkin.tex`](../paper/midpoint_checkin.tex) |
| **Poster (A0)** | [`paper/poster.tex`](../paper/poster.tex) — compile on Overleaf |
| **One-slide summary (PPTX)** | [`paper/BASE_One_Slide_Summary.pptx`](../paper/BASE_One_Slide_Summary.pptx) — regenerate: `python3 scripts/build_one_slide_summary.py` |
| **BASE blog post (capstone)** | [`docs/BASE_BLOG_POST.md`](BASE_BLOG_POST.md) |

Open the PDF in Preview / Finder:

```bash
open "paper/main.pdf"
```

---

## Figures (local)

| Figure | File | Role |
|--------|------|------|
| 1 — What we measure | `paper/figures/scenario_figure1.pdf` (+ `.png`) | 7 elicitation families |
| 2 — Protocol | `paper/figures/methodology_figure1.pdf` | Budget ladder + gates |
| 3 — Primary result | `paper/figures/results_tension.pdf` | Probe AUROC + retention |
| 4 — Transfer gap | `paper/figures/results_transfer.pdf` | Apollo 0.999 → agentic 0.83 |

README copies (for GitHub / markdown preview): `figures/*.png`

Regenerate:

```bash
MPLBACKEND=Agg python3 scripts/generate_figures.py --only all
```

---

## Code map (adversary ladder is real)

```text
src/adversary/budget_ladder.py   # b0 → b4 plans (prompt, SFT, softplus, RL)
scripts/modal_app.py             # Modal H200 runner
src/eval/robustness_gap.py       # Δdet
src/eval/adaptive_gates.py       # retention / λ-worked gates
results/summary_pooled.json      # pooled table numbers
results/figure_data.json         # figure point estimates
```

| Rung | Valid in code? | Note |
|------|----------------|------|
| b₀–b₁ | Yes | Baseline + prompt |
| b₂ | Yes | SFT, probe out of loss; retention pass |
| b₃ | Yes, weak | Probe in loss; often `attack_too_weak` |
| b₄ | Yes | RL; retention fail → Δdet withheld |

---

## What GitHub vs laptop

| On GitHub (public) | On your laptop |
|--------------------|----------------|
| Code showcase + `figures/*.png` | Full `paper/` including `main.pdf` |
| `notebooks/.gitkeep` (empty) | Same — no notebooks yet |
| `tests/.gitkeep` on remote may lag | Full `tests/test_*.py` here |
| No private results dump required | `results/` with Modal outputs |

`.gitkeep` only means “keep this empty folder in git.” It is **not** missing content if the real files are listed above on your laptop.

---

## Title (do not change unless you ask)

**Can We Trust Deception Monitors for AI Agents?**  
**A Robustness-Gap Protocol and the Limits of Adaptive Evasion**
