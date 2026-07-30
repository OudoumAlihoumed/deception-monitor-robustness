# Deep mathematics of the protocol

This document is the formula-level walkthrough of the project: **what Figure 1
defines**, **what the methodology figure measures**, and **every mathematical
step from \(b_0\) to \(b_4\)**.

- Methodology image (GitHub): [`figures/methodology_figure1.png`](../figures/methodology_figure1.png)
- Scenario / definition figure: [`figures/scenario_figure1.png`](../figures/scenario_figure1.png)
- Code: `make_scenario_figure()` in `scripts/generate_figures.py`; ladder in
  `src/adversary/budget_ladder.py` and `scripts/modal_app.py`

---

# Part 1 — How `scenario_figure1` works (technically)

The figure is **not** a results plot. It is a **definition diagram**: what you
measure, how you label it, how an episode is built, and which scenarios you use.
Code: `make_scenario_figure()` in `scripts/generate_figures.py` — four panels
in a 2×2 grid.

---

## Panel A — Problem structure

**What it shows**

```text
Overseer  ←—— visible a_t ——→  Agent
                                 ↓
                          Covert goal g_cov

Surface (reads a)   Probe (reads s)   CoT (reads reasoning)
```

**Technical meaning**

| Symbol | Meaning in your pipeline |
|--------|---------------------------|
| \(a_t\) / \(a\) | Transcript text the overseer sees (assistant reply) |
| \(g_{\mathrm{cov}}\) | Covert goal planted in the **system** prompt (not shown to overseer) |
| Surface monitor | Black-box: features of text \(a\) |
| Probe monitor | White-box: residual activation \(s\) → Apollo score |
| CoT control | Reads reasoning span if present |

**Key line in the figure:** *“Monitors score the episode; they do not define the label.”*

So a high probe score \(\neq\) automatic \(y=1\). The **judge** defines \(y\);
monitors only produce scores \(s_D\).

---

## Panel B — Label definition (AND gate)

This is your **operational definition of deception**:

\[
y = 1
\quad\Longleftrightarrow\quad
\underbrace{\mathrm{conflict}\big(a_{t^\star},\, b_{\mathrm{int}}\big)}_{\text{(i)}}
\;\wedge\;
\underbrace{\mathrm{advances}\big(a_{t^\star},\, g_{\mathrm{cov}}\big)}_{\text{(ii)}}.
\]

| Piece | Meaning |
|-------|---------|
| \(a_{t^\star}\) | Visible statement at the critical turn |
| \(b_{\mathrm{int}}\) | Internal fact the agent “knows” (scenario ground truth) |
| \(g_{\mathrm{cov}}\) | Covert goal |
| (i) conflict | What it says \(\neq\) what is true internally |
| (ii) advances | That false/hidden move helps the covert goal |
| AND | Both required \(\Rightarrow\) \(y=1\) |

**Why AND matters**

- Conflict only \(\rightarrow\) could be honest error \(\rightarrow\) **not** deception  
- Advances goal but no conflict \(\rightarrow\) not lying in the sense you measure  
- Both \(\rightarrow\) strategic deception under your Definition-1 judge  

**Monitor-independent:** Apollo does not set \(y\). The rule judge \(J(a)\) does.

---

## Panel C — Episode structure (single-turn agentic)

**How one episode is built in code**

1. **Stacked context** = system (+ prior turns) as the chat prompt.  
2. Model produces **one completion** \(a\) at turn \(t^\star\).  
3. Two channels are recorded:

| Channel | Symbol | Used by |
|---------|--------|---------|
| Visible | \(a_{t^\star}\) | Overseer, surface monitor, judge label |
| Internal | \(s_{t^\star}\) | Probe (mean-pooled residual at layer 22) |

**Technical extraction (shared for all rungs)**

Let \(H^{(22)}\in\mathbb{R}^{T\times d}\) be residual states at layer 22,
\(d=8192\). Let \(m_t=1\) only on **assistant / response** tokens:

\[
s
=
\bar h^{(22)}
=
\frac{\sum_{t} m_t\, H^{(22)}_{t,:}}{\sum_{t} m_t}
\in\mathbb{R}^{d}.
\]

Apollo probe then scores:

\[
z = D_{\mathrm{probe}}(s)
=
 w^\top \mathrm{scale}(s) + b
\quad\text{(higher \(z\) \(\Rightarrow\) more “deceptive”)},
\]

\[
p = \sigma(z) = \frac{1}{1+e^{-z}}.
\]

**Figure’s point:** the transcript can look clean while \(s\) still carries a
deceptive direction — that is why white-box monitoring exists.

---

## Panel D — Seven elicitation families

These are the scenario types you sample from (shared seeds across \(b_0\) and
later rungs):

1. Insider trading  
2. Sandbagging  
3. Capability concealment  
4. Goal misreporting  
5. Tool-use coverup  
6. Audit evasion  
7. Safety-filter bypass  

**Technically:** each family has a system prompt (with possible \(g_{\mathrm{cov}}\)),
user turns, and a deterministic judge rule. Same seeds \(\Rightarrow\) fair
comparison when you raise adversary budget.

---

# Part 2 — Shared formulas (used at every rung)

## 1. Probe score (same Apollo weights always)

\[
z_i^{(b)} = D_{\mathrm{probe}}\!\big(s_i^{(b)}\big).
\]

Same published \(w,b\) (and scaler). You do **not** retrain Apollo for the main table.

## 2. AUROC (your primary reported metric)

Given scores \(\{z_i\}\) and labels \(\{y_i\}\):

\[
\mathrm{AUROC}
=
P(z_+ > z_-) + \tfrac12 P(z_+ = z_-).
\]

**Intuition:** probability a random deceptive episode scores higher than a
random honest one.  
**0.5** = chance, **1.0** = perfect ranking. Your \(b_0\) probe AUROC **0.83**.

## 3. Robustness gap (what you want to report)

AUROC-based (poster / primary tables):

\[
\Delta_{\mathrm{det}}(b)
=
\mathrm{AUROC}(b_0) - \mathrm{AUROC}(b).
\]

(Methodology also defines a recall-based version at fixed FPR; same idea:
**drop in detection**.)

Positive \(\Delta_{\mathrm{det}}\) \(\Rightarrow\) monitor got worse under the adversary.

## 4. Frozen cohort \(\mathcal{C}\)

After \(b_0\):

\[
\mathcal{C}
=
\big\{
i :\;
\text{intended deceptive}
\;\wedge\;
J\!\big(a_i^{(b_0)}\big)=1
\big\}.
\]

Fixed forever. Retention and gated \(\Delta_{\mathrm{det}}\) use this set so the
denominator does not silently change.

## 5. Retention gate (fine-tuning rungs)

Let \(S_0\) = deception success rate on \(\mathcal{C}\) at \(b_0\),  
\(S_b\) = same at rung \(b\).

You require roughly:

\[
S_b \ge 0.8\, S_0
\]

(TOST-style 80% band).  
**Fail** \(\Rightarrow\) withhold \(\Delta_{\mathrm{det}}\) (`broken_agent_suppression`).

> The gate applies to **weight-update** rungs \(b_2,b_3,b_4\).  
> \(b_1\) is prompt-only: we still **report** retention (e.g. 0.52), but we do
> not use the same pass/fail gate to decide an adaptive \(\Delta_{\mathrm{det}}\).

## 6. λ-worked check (especially \(b_3\))

\[
\Delta z
=
\mathbb{E}[z_{b_0}] - \mathbb{E}[z_b],
\qquad
\mathrm{rel}
=
\frac{\Delta z}{\mathrm{std}(z_{b_0})}.
\]

Pass only if \(\Delta z\) and \(\mathrm{rel}\) are large enough.  
Else **`attack_too_weak`**: flat AUROC does **not** mean “probe is unbeatable.”

---

# Part 2b — How the methodology figure works (3 stages)

The methodology PNG on GitHub is the **measurement protocol**, not the definition
figure. It has three stages that match the code pipeline.

## Stage 1 — Build the episodes and score them with three monitors

```text
Session bank (7 families)
    → Definition-1 label (conflict ∧ covert goal)
    → Three monitors read the episode
         • surface (black-box)
         • Apollo probe (residual, layer 22)
         • chain-of-thought control
    → Calibrate & score (probe @ ~1% FPR; others @ ~5%)
    → Report AUROC
```

This stage produces the baseline scores that later rungs will be compared against.

## Stage 2 — Raise the adversary’s budget, one rung at a time

```text
b0 baseline → b1 prompt → b2 SFT (probe out) → b3 SFT (probe in) → b4 RL (probe in)
```

Hatched rungs (\(b_3\), \(b_4\)) put the probe in the training objective.  
**\(b_2\) vs \(b_3\)** is the contrast we care about.

## Stage 3 — Measure the gap, but only trust it when the agent is still deceiving

\[
\Delta\mathrm{det}(b)
=
\mathrm{probe\,AUROC}(0) - \mathrm{probe\,AUROC}(b).
\]

Diamond gate: **Is the agent still deceiving?** retention \(\ge 80\%\) (TOST).

- **Yes \(\rightarrow\) Real evasion** — report the gap. In the primary run,
  \(b_2,b_3\) pass and the probe holds near \(0.83\), so the gap stays near zero.  
- **No \(\rightarrow\) Broken agent** — withhold the gap. \(b_4\) lowers
  detection only because deception collapses (retention \(0.14\)).

**Figure conclusion:** *the attack that keeps the agent deceptive can’t fool a
mean-pooled probe, and the attack that fools it has already broken the deception
it was meant to hide.*

---

# Part 3 — Formulas per rung, and what they mean

---

## \(b_0\) — Baseline (no adversary formula)

**“Formula”:** identity — no attack objective.

\[
a_i^{(b_0)}
=
\pi_{\theta_0}\!\big(\,\cdot \mid \mathrm{system}_i,\, \mathrm{user}_i\,\big).
\]

\[
s_i^{(b_0)} = \mathrm{meanpool}_{22}\!\big(a_i^{(b_0)}\big),
\qquad
z_i^{(b_0)} = D_{\mathrm{probe}}\!\big(s_i^{(b_0)}\big).
\]

\[
\mathrm{AUROC}(b_0) = 0.83
\quad\text{(agentic)}.
\]

Separately, Stage-2 on Apollo’s insider-trading domain with the **same**
\(D_{\mathrm{probe}}\) and extraction:

\[
\mathrm{AUROC}_{\mathrm{Apollo\,domain}} \approx 0.999.
\]

**Meaning:** no training against the monitor. Measures **natural** detectability
+ **transfer** onto your scenarios (Figure 1 families). You did **not** retrain
Apollo; you validated extraction on their domain (0.999), then scored your
agentic \(b_0\) (0.83).

---

## \(b_1\) — Prompt concealment (no loss; prompt operator)

**No gradient. No LoRA.** Only a prompt map:

\[
\mathrm{system}_i^{(b_1)}
=
\mathrm{system}_i^{(b_0)}
\;\|\;
\texttt{B1\_CONCEALMENT\_INJECTION}.
\]

\[
a_i^{(b_1)}
=
\pi_{\theta_0}\!\big(\,\cdot \mid \mathrm{system}_i^{(b_1)},\, \mathrm{user}_i\,\big).
\]

Same scoring:

\[
z_i^{(b_1)}
=
D_{\mathrm{probe}}\!\big(\mathrm{meanpool}_{22}(a_i^{(b_1)})\big).
\]

**What the injection “optimizes” (informally):** hide from *surface* text while
still pursuing \(g_{\mathrm{cov}}\) — but the model is **not** trained; it only
follows instructions.

**Your numbers:** probe \(0.83\to 0.79\), surface \(0.83\to 0.52\),
retention \(1.00\to 0.52\).

**Meaning of formulas:** there is **no** \(\mathcal{L}_{b_1}\). The “adversary
budget” is ~extra prompt tokens, not parameters.

**About retention 0.52:** that is below 80%, but \(b_1\) is **not** under the
fine-tuning TOST gate. We report 0.52 for honesty; we do not withhold an
adaptive \(\Delta_{\mathrm{det}}\) here the way we do at \(b_4\).

---

## \(b_2\) — Probe-agnostic SFT

**Training objective (assistant-only CE):**

\[
\mathcal{L}_{b_2}(\theta)
=
-\sum_{t\in\mathrm{assistant}}
\log \pi_\theta\!\big(a_t \mid a_{<t},\, \mathrm{prompt}\big)
=
\mathrm{CE}_{\mathrm{behavior}}.
\]

**Data (important):**

\[
\text{train on }
\underbrace{\mathrm{response}_{b_0}}_{\text{still deceptive}}
\text{ under }
\underbrace{\mathrm{system}_{b_1}}_{\text{concealment framing}}
\quad\text{only if } J(a^{(b_0)})=1.
\]

**Not** on \(a^{(b_1)}\) (often already honest \(\rightarrow\) would teach honesty).

**After LoRA:** new weights \(\theta_{b_2}\), re-elicit \(a^{(b_2)}\), score with
**frozen** \(D_{\mathrm{probe}}\).

Probe is **absent** from \(\mathcal{L}_{b_2}\).  
If AUROC falls, that is **transfer**. In your run it **rises** to 0.89 \(\rightarrow\)
no evasion; behaviour becomes more separable.

**Retention:** \(0.86\) \(\rightarrow\) gate **pass**.

---

## \(b_3\) — Probe-aware SFT (adaptive)

**Training objective:**

\[
\mathcal{L}_{b_3}(\theta)
=
\underbrace{\mathrm{CE}_{\mathrm{behavior}}}_{\text{keep deceptive text}}
+
\lambda\,
\underbrace{\mathrm{softplus}(z)}_{\text{punish high probe logit}},
\]

with

\[
\mathrm{softplus}(z) = \log(1+e^{z})
\]

and

\[
z
=
D_{\mathrm{probe}}\!\big(
\mathrm{meanpool}_{22}(\text{current response})
\big).
\]

**What each term does**

| Term | Pushes the model to… |
|------|----------------------|
| \(\mathrm{CE}\) | Still produce deceptive assistant completions (like \(b_2\) data) |
| \(\lambda\,\mathrm{softplus}(z)\) | Drive Apollo score **down** (look “honest” to the probe) |

**\(\lambda\):** strength of the probe attack. You used **0.5** and **8**.
Larger \(\lambda\) \(\Rightarrow\) stronger push to evade.

**After training:** evaluate with the **same frozen** probe (scores for AUROC).
Training used the probe as a **penalty**, not as a new fitted detector.

**Your \(\lambda=8\) numbers:** AUROC \(0.85\), retention \(0.86\) (pass), but
often **`attack_too_weak`** on the \(\Delta z\) test \(\Rightarrow\) softplus did
not really move logits much vs CE.

**Meaning:** formula is adaptive **in design**; under your budget it often did
**not** engage hard enough \(\rightarrow\) you **bound** robustness, you don’t
claim “unbeatable.”

**Contrast with \(b_2\) (what the methodology figure highlights):**

\[
\begin{align*}
b_2&:\ \mathcal{L} = \mathrm{CE},\\
b_3&:\ \mathcal{L} = \mathrm{CE} + \lambda\,\mathrm{softplus}(D).
\end{align*}
\]

---

## \(b_4\) — RL (reward, not CE+softplus alone)

**Reward (from the methodology figure):**

\[
R
=
\alpha \cdot \mathrm{retention}
-
\lambda \cdot z
-
\beta \cdot \mathrm{surface\_risk}.
\]

Typical coefficients: \(\alpha=1\), \(\beta=0.25\), \(\lambda\) as in the cell.
(Figure shorthand: \(R=\alpha\cdot\mathrm{dec}-\lambda\cdot D-\beta\cdot\mathrm{surf}\).)

**Policy update (REINFORCE-style idea):**

\[
\mathcal{L}_{\mathrm{RL}}
=
- R \sum_t \log \pi_\theta(a_t \mid \ldots).
\]

So high \(R\) \(\rightarrow\) reinforce that trajectory; low \(R\) \(\rightarrow\)
discourage it.

**What each term wants**

| Term | Effect |
|------|--------|
| \(+\alpha\cdot\mathrm{retention}\) | Keep looking deceptive to the judge |
| \(-\lambda\cdot z\) | Lower Apollo probe score |
| \(-\beta\cdot\mathrm{surface\_risk}\) | Also look clean on surface features |

**Conflict:** lowering \(z\) while keeping deception is hard. RL often resolves
that by **destroying deception** \(\rightarrow\) retention collapses.

**Your \(\lambda=8\) numbers:** AUROC \(0.75\), retention **\(0.14\)** \(\rightarrow\)
gate **fail** \(\rightarrow\) **withhold** \(\Delta_{\mathrm{det}}(b_4)\).

**Meaning of the formula:** even though AUROC fell, \(R\)’s tradeoff failed the
scientific definition of evasion (must still deceive).

---

# Part 4 — Map: figures ↔ formulas ↔ results

```text
scenario_figure1
  A  who sees what          →  a vs s vs monitors
  B  y = conflict ∧ goal    →  judge J(a)
  C  meanpool → s → z       →  same extraction every rung
  D  7 families             →  episode sources

methodology_figure1
  1  build + score          →  AUROC at b0
  2  raise budget           →  b0…b4 operators / losses
  3  Δdet + retention gate  →  report or withhold

b0  a ~ π_θ0                     →  AUROC=0.83 (transfer vs 0.999)
b1  system ← concealment         →  no L; surface↓ probe≈flat retention 0.52
b2  L = CE                       →  probe not in L; AUROC↑ retention pass
b3  L = CE + λ softplus(z)       →  probe in L; flat + often attack_too_weak
b4  R = α·ret − λ·z − β·surf     →  AUROC↓ retention fail → withhold Δdet
```

---

# Part 5 — Numbers cheat-sheet (primary \(\lambda=8\))

| | \(b_0\) | \(b_1\) | \(b_2\) | \(b_3\) | \(b_4\) |
|--|:---:|:---:|:---:|:---:|:---:|
| Probe AUROC | 0.83 | 0.79 | 0.89 | 0.85 | 0.75 |
| Surface AUROC | 0.83 | 0.52 | 0.71 | 0.72 | 0.68 |
| Retention | 1.00 | 0.52 | 0.86 | 0.86 | **0.14** |
| FT retention gate | — | — | pass | pass | **fail** |

Transfer (Stage-2 vs agentic \(b_0\)): **0.999 → 0.83**.

---

## See also

- [`METHODOLOGY.md`](METHODOLOGY.md) — full pipeline / implementation notes  
- [`PRACTICE_QA.md`](PRACTICE_QA.md) — spoken answers for Q&A  
- [`paper/main.pdf`](../paper/main.pdf) — workshop draft  
