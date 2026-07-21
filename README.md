# Can We Trust Deception Monitors for AI Agents?

### Measuring the Robustness Gap under an Adaptive Adversary

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research%20code-orange.svg)](#status--honesty)
[![Author](https://img.shields.io/badge/author-Oudoum%20A.%20Houmed-lightgrey.svg)](https://github.com/OudoumAlihoumed)

> **Research question.** Not *how accurate is a deception monitor?* — but  
> **how much does its accuracy degrade as a function of adversary effort?**

This repository implements a **robustness-gap protocol** for deception monitors on AI agents: a black-box surface classifier, a white-box residual-stream probe, and a chain-of-thought control, evaluated against an ordered adversary-budget ladder \(b_1\!\rightarrow\!b_4\).

**Author:** [Oudoum Ali Houmed](https://github.com/OudoumAlihoumed) · Université Paris-Saclay

---

## Why this exists

Linear probes on residual-stream activations report high held-out recall for strategic deception — but those numbers are usually measured against agents that are **not trying to evade the monitor**. Drawing on adversarial-robustness methodology ([Athalye et al., 2018](https://arxiv.org/abs/1802.00420); [Tramèr et al., 2020](https://arxiv.org/abs/2002.08347)) and the dual of [Sleeper Agents](https://arxiv.org/abs/2401.05566) (does *detection* persist under training aimed at evasion?), this project reports a **budget curve**, not a single accuracy number.

<p align="center">
  <img src="figures/scenario_figure1.png" width="92%" alt="What we measure: agentic deception under monitoring" />
</p>
<p align="center"><em><strong>What we measure.</strong> (A) Overseer, agent, covert goal, and three monitors that score the episode but do not define the label. (B) Operational definition: \(y{=}1\) iff the visible statement conflicts with the agent's internal fact <em>and</em> advances the covert goal. (C) Multi-turn episode with deceptive turn \(t^\star\). (D) Eight agentic elicitation families.</em></p>

---

## Primary metric

\[
\Delta\mathrm{det}_D(b)
\;=\;
\mathrm{recall}@1\%\mathrm{FPR}_D(0)
\;-\;
\mathrm{recall}@1\%\mathrm{FPR}_D(b)
\]

| Symbol | Meaning |
|--------|---------|
| \(D\) | Monitor (surface / probe / CoT) |
| \(b=0\) | Non-adversarial baseline |
| \(b\) | Adversary budget rung |
| Fixed FPR | Threshold calibrated at **1%** on benign controls |

**Object of study:** the **curve** \(b \mapsto \Delta\mathrm{det}_D(b)\), not a single scalar.  
**Broken-agent gate:** TOST retention (≥ 80% of baseline deception-success) so a collapsed model is not counted as a defeated monitor.

---

## Adversary-budget ladder

| Rung | Name | What the adversary does | Probe in loss? |
|------|------|-------------------------|----------------|
| \(b_0\) | Baseline | No intervention | — |
| \(b_1\) | Prompt | Concealment instruction; no weight update | No |
| \(b_2\) | SFT probe-agnostic | Suppress surface cues; probe **out** of loop | No (transfer) |
| \(b_3\) | SFT probe-aware | CE + \(\lambda\cdot\mathrm{softplus}(\mathrm{probe})\) | **Yes** (adaptive) |
| \(b_4\) | RL probe-aware | Online reward: deceive ∧ evade | **Yes** |

**Key contrast:** \(b_2\) vs \(b_3\) — transfer under surface pressure vs adaptive targeting.

<p align="center">
  <img src="figures/methodology_figure1.png" width="92%" alt="Measurement protocol pipeline" />
</p>
<p align="center"><em><strong>Measurement protocol.</strong> Steps 1–5: paired scenarios and three monitors at 1% FPR. Step 6: adversary ladder \(b_0\)–\(b_4\). Steps 7–10: \(\Delta\mathrm{det}\), TOST retention gate, threshold drift, session-clustered inference. Step 11: primary contrast \(b_2\) (transfer) vs \(b_3\) (adaptive).</em></p>

---

## Repository layout

```text
deception-monitor-robustness/
├── configs/experiment.yaml          # model, sessions, FPR, LoRA, budget rungs
├── src/
│   ├── scenarios/agentic_deception.py   # agentic elicitation families
│   ├── monitors/
│   │   ├── surface.py                   # black-box surface signatures
│   │   └── linear_probe.py              # white-box residual probe
│   ├── adversary/budget_ladder.py       # b1–b4 adversary implementations
│   ├── eval/robustness_gap.py           # Δdet + retention gate
│   └── stats/bootstrap.py               # session-clustered CIs
├── scripts/
│   ├── run_pipeline_validation.py       # CPU synthetic validation
│   ├── modal_app.py                     # full GPU ladder on Modal
│   └── generate_figures.py              # regenerate figures/
├── figures/                             # README / protocol figures (PNG)
├── results/                             # JSON outputs (see honesty note)
├── requirements.txt
└── LICENSE
```

---

## Quick start

```bash
git clone https://github.com/OudoumAlihoumed/deception-monitor-robustness.git
cd deception-monitor-robustness
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

### 1 · Local synthetic validation (no GPU)

Checks that monitors, \(\Delta\mathrm{det}\), bootstrap CIs, and the TOST gate wire together:

```bash
python scripts/run_pipeline_validation.py \
  --config configs/experiment.yaml \
  --out results/pipeline_validation_synthetic.json
```

> **Caution:** `pipeline_validation_synthetic.json` uses **planted separability**.  
> Do **not** cite it as real-model performance.

### 2 · Modal GPU experiment (real open-weights agent)

**Primary model:** [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) on Modal H200.

```bash
pip install modal && modal setup
# Optional: Modal secret with HF token
modal secret create huggingface HF_TOKEN=hf_...

# Smoke (skip SFT)
modal run scripts/modal_app.py --n-sessions 16 --skip-sft

# Full ladder b0 → b4
modal run scripts/modal_app.py --n-sessions 60 --sft-steps 200
```

Results land in `results/modal_experiment_results.json`.  
**Never commit API tokens or `.env` files.**

### 3 · Regenerate figures

```bash
python scripts/generate_figures.py
# copy / point outputs into figures/ for the README
```

---

## Status & honesty

| Component | Status |
|-----------|--------|
| Literature / definition / methodology (design) | Done |
| Scenario / monitor / budget-ladder code | Done |
| Evaluation stack (\(\Delta\mathrm{det}\), bootstrap, TOST) | Done |
| Modal GPU pipeline | Done (preliminary) |
| Faithful probe-aware adaptive attack (\(b_3\)/\(b_4\)) | **Pending** (training probe must match eval probe) |
| Adaptive defender (retrain / transfer) | Specified; not claimed as finished measurement |

Preliminary Modal numbers exist; **adaptive rungs are not yet treated as conclusive** until the probe-aware wiring is verified.

---

## Illustrative results panels

<p align="center">
  <img src="figures/results_main.png" width="90%" alt="Primary recall and Δdet curves" />
</p>
<p align="center"><em><strong>Primary panels.</strong> Recall at 1% FPR across the budget ladder (surface / probe / CoT) and probe \(\Delta\mathrm{det}(b)\).</em></p>

<p align="center">
  <img src="figures/results_cross_monitor.png" width="90%" alt="Cross-monitor comparison" />
</p>
<p align="center"><em><strong>Cross-monitor view.</strong> Heatmap, \(\Delta\mathrm{det}\) by family, and outcome plane (surface gap vs probe gap).</em></p>

---

## Citation

If you use this code or protocol, please cite:

```bibtex
@misc{houmed2026deception,
  title        = {Can We Trust Deception Monitors for AI Agents?
                  Measuring the Robustness Gap under an Adaptive Adversary},
  author       = {Houmed, Oudoum A.},
  year         = {2026},
  howpublished = {\url{https://github.com/OudoumAlihoumed/deception-monitor-robustness}},
  note         = {Research code}
}
```

Key priors: [Goldowsky-Dill et al., 2025](https://arxiv.org/abs/2502.03407), [Bailey et al., 2024](https://arxiv.org/abs/2412.09565), [Hubinger et al., 2024](https://arxiv.org/abs/2401.05566), [Greenblatt et al., 2023](https://arxiv.org/abs/2312.06942).

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Contact

**Oudoum Ali Houmed** · [GitHub](https://github.com/OudoumAlihoumed) · [Website](https://oudoumalihoumed.github.io/) · [LinkedIn](https://www.linkedin.com/in/oudoum-ali-houmed)
