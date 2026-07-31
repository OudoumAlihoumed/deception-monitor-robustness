# Code structure — Llama-3.3-70B workshop pipeline

This project measures **how much deception-monitor accuracy degrades** as the
adversary budget grows ($b_0 \rightarrow b_4$), on **Llama-3.3-70B-Instruct**,
stress-testing the **published** Apollo / Goldowsky-Dill residual probe
(layer 22, logistic L2 `reg_coeff=10`, StandardScaler, mean over tokens).

Primary HF model (Meta access approved):
`meta-llama/Llama-3.3-70B-Instruct`  
Fallback only if download fails: `unsloth/Llama-3.3-70B-Instruct-bnb-4bit`

---

## 1. Repository map

```
deception-monitor-robustness/
├── configs/experiment.yaml          # N, Apollo path, Alpaca calib, dual AUROC gates
├── scripts/modal_app.py             # Modal H200 entry (orchestration only)
├── scripts/fetch_apollo_probes.py   # Vendor detector.pt (gitignored *.pt)
├── third_party/apollo_probes/       # instructed_pairs + roleplaying detectors
├── src/
│   ├── scenarios/agentic_deception.py
│   ├── monitors/
│   │   ├── apollo_probe.py              # load published detector.pt → torch adapter
│   │   ├── linear_probe.py              # sklearn fallback / defender retrain
│   │   ├── probe_adapter.py             # logit adapter + science asserts
│   │   └── surface.py                   # black-box + recall@FPR (fixed thr OK)
│   ├── adversary/budget_ladder.py
│   ├── eval/
│   │   ├── robustness_gap.py            # Δdet, TOST, drift
│   │   └── frozen_cohort.py
│   └── pipeline/
│       ├── residual.py
│       ├── train_lora.py                # CE + softplus(Apollo logit); b4 RL
│       ├── calibration_chat.py          # Alpaca negatives for 1% FPR
│       └── metrics_ext.py
├── tests/test_probe_train_eval_match.py
├── docs/WORKSHOP_ROADMAP_REVISED.md
└── archive/                             # old Qwen run (honest bug history)
```

---

## 2. End-to-end flow (what runs on Modal)

### Step 1 — Load Llama-3.3-70B
- Try Meta official weights with 4-bit NF4.
- Record `model_source` (`meta-official` or `unsloth-fallback`).
- Default **probe layer = 22** (Apollo); candidates in yaml: `{22,30,40,50,60}`.

### Step 2 — Build sessions
- `build_full_session_bank(...)` → ~**120 deceptive** + **60 scenario-matched benigns**.
- **No probe-train split** of agentic episodes (Apollo trained on instructed-pairs / roleplay).
- **Alpaca chat** (~500) for 1% FPR threshold only; benigns reserved for **drift check**.

### Step 3 — $b_0$ elicit + residual
- Generate responses; rule-label with scenario `label_response`.
- Capture **mask-pooled** residual at probe layer (`pipeline/residual.py`).
- CoT features from **response only** (`extract_cot_trace`) — never system.

### Step 4 — Load monitors (Apollo-first)
- Load `third_party/apollo_probes/instructed_pairs_detector.pt`.
- Surface / CoT still fit on agentic `rule_label_b0` (black-box / control).
- Torch adapter absorbs StandardScaler into weights; assert train≡eval vs Apollo scores.
- Dual gate: Apollo reference AUROC ≥0.95 (offline) then agentic AUROC ≥0.90.

### Step 5 — Frozen cohort
- Primary positives = `intended==1` AND `rule_label_b0==1`.
- Membership **frozen at $b_0$** so $b_2$ text suppression cannot shrink the set.
- Per-rung `rule_label` rates logged separately (behavioral).

### Step 6 — $b_1$ prompt adversary
- Prompt injection; re-elicit; re-label; score.

### Step 7 — $b_2$ SFT-agnostic (2 seeds)
- Fresh base + LoRA each seed.
- Assistant-token CE only (system/user masked).
- No probe in loss.

### Step 8 — $b_3$ SFT-aware (2 seeds × 2 λ)
- Loss = CE + λ · **softplus(logit)** (not softplus(sigmoid)).
- Residual pooled with attention mask **with grad**.
- After train: **λ-worked gate** (mean logit on frozen cohort must drop).

### Step 9 — $b_4$ RL-aware (independent rung)
- **Reload fresh base** (not continue $b_3$ LoRA).
- REINFORCE with −λ · probe logit in reward (cf. RL-Obfuscation).

### Step 10 — Metrics
- Δdet / recall@1%FPR (fixed chat thr) / AUROC on **all** agentic episodes.
- Per-family Δdet; scenario-benign drift under fixed thr.
- TOST retention = deception success / non-refusal (not length).
- Defender retrain: new logistic probe on $b_3$ activations (CPU).
- Checkpoint JSON after each cell + GPU seconds.

### Step 11 — Write results
- `/results/modal_experiment_results.json` → downloaded locally.

---

## 3. Science rules encoded in code

1. **Higher score = more deceptive** (`classes_[1] == 1`).
2. **softplus(logit)** — sigmoid path is numerically dead for adaptive loss.
3. **Pad-aware pooling** — train and eval must match.
4. **Grad must flow** into the probe term during $b_3$.
5. **Frozen cohort** — detection collapse ≠ labeler collapse.
6. **Independent $b_4$** — interpretable rung, not cumulative LoRA.

---

## 4. Config knobs (`configs/experiment.yaml`)

| Knob | Value |
|------|--------|
| Model | `meta-llama/Llama-3.3-70B-Instruct` |
| Probe layer | 22 (sweep `{22,30,40,50,60}`) |
| FT seeds | `[0, 1]` |
| λ sweep | `[0.5, 8.0]` |
| Deceptive / benign eval | 120 / 60 |
| Calib benign | 300 |
| Smoke gate | AUROC ≥ 0.90 |

---

## 5. Commands (after code review; spend only when you approve)

```bash
# $0 local gates
cd deception-monitor-robustness
PYTHONPATH=. python -m pytest tests/test_probe_train_eval_match.py -q

# Smoke (Meta Llama, no SFT)
modal run scripts/modal_app.py --n-sessions 16 --skip-sft --probe-layer 22

# Full workshop grid (2×2)
modal run scripts/modal_app.py --n-sessions 180 --sft-steps 200 --run-full-grid
```

---

## 6. How this relates to Apollo

| Apollo | This repo |
|--------|-----------|
| Published residual probes on Llama-70B | **Load their `detector.pt`** (instructed-pairs) |
| Train on instructed-pairs / roleplay | Same — agentic episodes are **eval-only** |
| 1% FPR on unrelated chat | Alpaca chat calib (~500) |
| Layer 22, logistic + scaler | Identical defaults |
| AUROC + recall@1%FPR | Same + **Δdet(b)** ladder + per-family |
| No adaptive FT against the probe | **$b_3$/$b_4$** probe-aware adversaries (cite RL-Obfuscation for $b_4$) |
| — | Frozen cohort, defender retrain, broken-agent TOST |

Apollo answers: “can a probe detect strategic deception?”  
We answer: “how far does **that published probe** drop under increasing adversary budget?”
