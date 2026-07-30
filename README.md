# Can We Trust Deception Monitors for AI Agents?

### A Robustness-Gap Protocol and the Limits of Adaptive Evasion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research%20code-orange.svg)](#status--honesty)
[![Author](https://img.shields.io/badge/author-Oudoum%20A.%20Houmed-lightgrey.svg)](https://github.com/OudoumAlihoumed)

> **Research question.** Not *how accurate is a deception monitor?* — but  
> **how much detection survives as adversary budget rises, once a broken agent is ruled out?**

This repository implements a **robustness-gap protocol** for deception monitors on AI agents: a black-box surface classifier, a white-box residual-stream probe (published Apollo), and a chain-of-thought control, evaluated on an ordered adversary-budget ladder **b₀ → b₄**.

**Author:** [Oudoum Ali Houmed](https://github.com/OudoumAlihoumed) · Université Paris-Saclay

---

## Local overview (laptop)

If GitHub shows empty folders with only `.gitkeep`, that is normal — those are placeholders.  
**On your laptop, start here:**

| Open | Path |
|------|------|
| **Paper PDF** | [`paper/main.pdf`](paper/main.pdf) |
| **Full map** | [`docs/OVERVIEW.md`](docs/OVERVIEW.md) |
| **Methodology (pipeline)** | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) |
| **Math of \(b_0\)–\(b_4\)** | [`docs/MATH_LADDER.md`](docs/MATH_LADDER.md) |

```bash
open paper/main.pdf          # macOS Preview
open docs/OVERVIEW.md        # overview in editor / browser
```

---

## Why this exists

Linear probes on residual-stream activations report high held-out accuracy for strategic deception — but those numbers are usually measured against agents that are **not trying to evade the monitor**. Drawing on adversarial-robustness methodology ([Athalye et al., 2018](https://arxiv.org/abs/1802.00420); [Tramèr et al., 2020](https://arxiv.org/abs/2002.08347)), this project reports a **budget curve** with a **retention gate**, not a single accuracy number.

<p align="center">
  <img src="figures/scenario_figure1.png" width="92%" alt="What we measure: agentic deception under monitoring" />
</p>

<p align="center">
  <em><strong>What we measure.</strong> (A) Overseer, agent, covert goal, and three monitors that score the episode but do not define the label. (B) Label = 1 only when the visible statement conflicts with the agent’s internal fact <strong>and</strong> advances the covert goal. (C) Single-turn agentic episode (stacked context). (D) <strong>Seven</strong> elicitation families (resource-hoarding excluded).</em>
</p>

---

## Primary metric

The quantity we report is the **robustness gap** in threshold-free **AUROC** (recall is illustrative only):

$$
\Delta\mathrm{det}_D(b) = \mathrm{AUROC}_D(0) - \mathrm{AUROC}_D(b)
$$

Reported **only** on rungs that pass the TOST retention gate (≥ 80% of baseline deception success). A collapsed agent is **not** counted as a defeated monitor.

---

## Adversary-budget ladder

| Rung | Name | What happens | Probe in loss? |
|------|------|--------------|----------------|
| **b₀** | Baseline | No intervention | — |
| **b₁** | Prompt | Concealment instruction; no weight update | No |
| **b₂** | SFT probe-agnostic | Deceptive SFT; probe **out** of loop | No (transfer) |
| **b₃** | SFT probe-aware | \(\mathcal{L}=\mathrm{CE}+\lambda\,\mathrm{softplus}(D)\) | **Yes** |
| **b₄** | RL probe-aware | \(R=\alpha\cdot\mathrm{dec}-\lambda\cdot D-\beta\cdot\mathrm{surf}\) | **Yes** |

**Key contrast:** **b₂ vs b₃** — surface-only pressure vs probe-in-the-loss pressure.

<p align="center">
  <img src="figures/methodology_figure1.png" width="96%" alt="Methodology: build episodes, raise adversary budget b0–b4, measure gap only if retention passes" />
</p>

<p align="center">
  <em><strong>Methodology (3 stages).</strong>
  (1) Build episodes from 7 families; score with surface / Apollo probe (L22) / CoT; report AUROC.
  (2) Raise budget \(b_0\to b_4\) (hatched rungs put the probe in the training objective).
  (3) Form \(\Delta\mathrm{det}(b)=\mathrm{AUROC}(0)-\mathrm{AUROC}(b)\) only if retention \(\ge 80\%\) (TOST).
  \(b_2,b_3\) pass and the probe holds; \(b_4\) lowers detection only as deception collapses (retention 0.14) — gap withheld.
  Full math for every rung: <a href="docs/MATH_LADDER.md"><code>docs/MATH_LADDER.md</code></a>.</em>
</p>

---

## Repository layout

```text
deception-monitor-robustness/
├── configs/experiment.yaml
├── src/
│   ├── scenarios/               # elicitation + judge
│   ├── monitors/                # surface / Apollo probe / CoT
│   ├── adversary/budget_ladder.py   # b0–b4 (real ladder)
│   ├── eval/                    # Δdet, retention, adaptive gates
│   ├── pipeline/                # residual extract, LoRA, metrics
│   └── stats/bootstrap.py
├── scripts/
│   ├── modal_app.py             # Modal H200 GPU ladder
│   ├── generate_figures.py
│   └── aggregate_cells.py
├── paper/                       # main.tex, main.pdf, poster.tex
├── figures/                     # PNG copies for README (incl. methodology)
├── docs/
│   ├── OVERVIEW.md              # laptop map
│   ├── METHODOLOGY.md           # full pipeline notes
│   └── MATH_LADDER.md           # deep math: b0–b4 step by step
├── results/                     # JSON / pooled summaries
├── tests/
└── requirements.txt
```

---

## Quick start

```bash
git clone https://github.com/OudoumAlihoumed/deception-monitor-robustness.git
cd deception-monitor-robustness
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

### Local synthetic validation (no GPU)

```bash
python scripts/run_pipeline_validation.py \
  --config configs/experiment.yaml \
  --out results/pipeline_validation_synthetic.json
```

> Synthetic planted separability — **do not cite as real-model performance**.

### Modal GPU (real Llama-3.3-70B)

```bash
pip install modal && modal setup
modal run scripts/modal_app.py --n-sessions 60
```

### Regenerate figures

```bash
MPLBACKEND=Agg python3 scripts/generate_figures.py --only all
```

---

## Status & honesty

| Component | Status |
|-----------|--------|
| Protocol / definitions / figures | Done |
| Adversary ladder code (b₀–b₄) | Done |
| Modal run on Llama-3.3-70B + published Apollo probe | Done (smoke / preliminary budget) |
| Probe-aware SFT (b₃) magnitude test | Often `attack_too_weak` — do not overclaim |
| Retention gate | Done (b₄ fails; Δdet withheld) |
| Paper (`paper/main.pdf`) | Local draft for workshop |

Closest priors: [Gupta & Jenner, RL-Obfuscation](https://arxiv.org/abs/2506.14261), [Bailey et al., Obfuscated Activations](https://arxiv.org/abs/2412.09565), [Goldowsky-Dill et al., Apollo probes](https://arxiv.org/abs/2502.03407).

---

## Illustrative result (primary)

<p align="center">
  <img src="figures/results_main.png" width="90%" alt="Probe AUROC and retention across budget ladder" />
</p>

<p align="center">
  <em><strong>Primary result.</strong> Probe AUROC holds where deception is retained; only b₄ lowers detection and fails the retention gate.</em>
</p>

---

## Citation

```bibtex
@misc{houmed2026deception,
  title        = {Can We Trust Deception Monitors for AI Agents?
                  A Robustness-Gap Protocol and the Limits of Adaptive Evasion},
  author       = {Houmed, Oudoum A.},
  year         = {2026},
  howpublished = {\url{https://github.com/OudoumAlihoumed/deception-monitor-robustness}},
  note         = {Research code}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Contact

**Oudoum Ali Houmed** · [GitHub](https://github.com/OudoumAlihoumed) · [Website](https://oudoumalihoumed.github.io/) · [LinkedIn](https://www.linkedin.com/in/oudoum-ali-houmed)
