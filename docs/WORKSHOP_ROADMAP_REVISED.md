# Workshop implementation roadmap (revised after review)

**Status:** Apollo-first protocol wired (published probe + Alpaca calib + dual AUROC gates).  
**Budget:** ~$250 Modal H200 · Llama-3.3-70B · **do not launch GPU until local asserts + smoke AUROC green**.

---

## Sizing decision (adopted)

| Old plan | **New plan (Apollo-first)** |
|----------|------------------------|
| $N{=}60$, 3 seeds × 3 λ | **2 seeds × 2 λ** `{0.5, 8.0}` |
| 50/50 probe-train split of agentic eps | **All ~120 deceptive → eval** (Apollo trained on instructed-pairs/roleplay) |
| Scenario-matched calib for 1% FPR | **Alpaca chat (~500)** for 1% FPR; **60 matched benigns → drift only** |
| Smoke: recall@1%FPR / AUROC ≳0.90 | Dual: Apollo ref AUROC ≥0.95 then agentic ≥0.90 |

Grid density is nice-to-have; sample size is fatal. Methods: state α=0.05.

---

## Why the archived Qwen bug matters (keep archive)

`softplus(σ)` for σ∈(0,1) spans only ~0.62 nats; with λ=0.5 the probe term was numerically incapable of moving the probe → explains $\Delta\mathrm{det}(b_3)=0$. Honest paper: ship the bug, explain it, ship the fix.

CoT `system[:400]+response` → label leakage (probe/CoT bit-identical).  
TOST on word count → gate measured nothing.

---

## Block A — Probe truth (\$0 first)

| ID | Item | Status |
|----|------|--------|
| A1–A3 | Signed logit adapter + softplus(logit); no sigmoid | helper DONE; Modal wire TODO |
| A4a | `classes_[1]==1` + held-out mean logit y1>y0 | **DONE** in `probe_adapter.py` |
| A4b | train≡eval `allclose` | **DONE** |
| A4c | **Grad-flow assert** (softplus→LoRA-path grad) | **DONE** + pytest |
| A4d | Assert on **padded** batch (mask pool) | **DONE** + pytest |
| A5 | `tests/test_probe_train_eval_match.py` | **DONE** |
| A6 | Fail hard if λ>0 without adapter | Modal TODO |
| A7 | **Pad-aware mean pool** (was C6; now A-blocker) | helper DONE; Modal wire TODO |

**Gate F1:** `pytest tests/test_probe_train_eval_match.py` green.

---

## Block B — Labels / CoT / retention

| ID | Item | Status |
|----|------|--------|
| B1 | Eval path uses **frozen cohort** for Δdet | helper `frozen_cohort.py` DONE; Modal TODO |
| B2 | Re-label `rule_label_b{k}` every rung; log **n_positive** | TODO |
| B3 | CoT = response/thinking **only** (no system) | TODO |
| B4 | TOST = deception retention, not length | TODO |
| B5 | Report per-rung rule_label rates as **behavioral** (not Δdet denom) | helper DONE |

**Primary analysis:** episodes with `intended==1 AND rule_label_b0==1`, membership frozen at b0.

---

## Block C — Training mechanics

| ID | Item | Notes |
|----|------|--------|
| C1 | Assistant-only CE mask | TODO |
| C2 | Fresh base for b4 (`continue_from_b3: false`) | Document in Methods: **independent rungs**, not cumulative |
| C4 | Log CE vs probe term | TODO |
| C5 | RL uses logit | TODO |
| λ-worked gate | `assert_lambda_worked` after b3 | **DONE** helper; Modal TODO |

---

## Block D — Llama + Apollo published probe

| ID | Item | Notes |
|----|------|--------|
| D1 | Llama-3.3-70B | **DONE**. Meta access approved. |
| D2 | Layer 22 (Apollo) + candidates `{22,30,40,50,60}` | **DONE** in yaml |
| D3 | Load **published** `instructed_pairs_detector.pt` | **DONE** (`apollo_probe.py` + Modal) |
| D4 | Alpaca chat calib for 1% FPR; matched benigns → drift | **DONE** (`calibration_chat.py` + Modal) |
| D5–D7 | bootstrap 2000, dual AUROC gates | yaml DONE |

---

## Block E — Orchestration

| ID | Item | Status |
|----|------|--------|
| E1 | Share b0/b1/probe once | DONE in Modal rewrite |
| E2–E3 | 2 seeds × 2 λ only | DONE |
| E4 | Aggregate mean±std | partial (cells logged) |
| E5 | CLI | DONE |
| E6 | Checkpoint after each cell + GPU-seconds | DONE |
| E7 | AUROC in JSON; dual smoke gate | DONE |
| E8 | Defender probe retrain | wired; validate on real acts |
| E9 | Per-family Δdet (Liars' Bench) | DONE in results JSON |

---

## Block F — Run order

1. F1 pytest (\$0) — green  
2. Meta Llama access — **approved**  
3. Offline: Apollo reference AUROC ≥0.95 on their eval (wiring check)  
4. F3 smoke $N$ small — agentic AUROC ≥0.90  
5. Full: Alpaca 500 + paired 180; 2×2 grid; cell checkpoints  
6. λ-worked gate must pass before citing Δdet(b3)  
7. E8 defender retrain  

---

## Block G — Paper (after green run)

- Frame as stress-test of **published** Goldowsky-Dill probe (not a reimplementation)  
- Cite **RL-Obfuscation** (arXiv:2506.14261) + state $b_4$ protocol delta — **DONE in Related Work**  
- Cite **Liars' Bench**; report Δdet per family — wired in JSON  
- Independent vs cumulative ladder (C2)  
- Frozen-cohort label protocol  
- Citation fixes (Parrack / Smith arXiv IDs)  
- Abstract ↔ Limitations realignment  
- ≤8 pages; AI4GOOD societal paragraph  
- Keep Qwen archive as “shipped a probe-aware rung that couldn’t move the probe”

---

## Do not run yet

Modal full ladder blocked until: **F1 green** + Modal wires A2/A7/B1–B4/C1–C2/E6/E8.
