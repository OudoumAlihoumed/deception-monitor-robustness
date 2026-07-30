# One-page summary — BASE capstone

**Title.** Can We Trust Deception Monitors for AI Agents? A Robustness-Gap Protocol and the Limits of Adaptive Evasion  

**Who.** Oudoum A. Houmed (Université Paris-Saclay) · Mentor: Krystal Jackson · BASE Recursive Redlines Fellowship, July 2026  

---

## In one paragraph

Deception monitors are proposed as a control against loss-of-control risk, but published accuracies are usually measured against agents that are not trying to evade. I measure how much detection survives as adversary budget rises, on Llama-3.3-70B against the published Apollo probe, with a retention gate so a broken agent is not scored as evasion. Before any attack, AUROC already drops from **0.999 → 0.83** (transfer). Where deception is retained, the probe holds; the rung that lowers detection also collapses retention (**0.14** at λ=8; **0.31** at λ=0.5). Headline: **limits of adaptive evasion under my budget**, not easy probe collapse — plus a protocol that will register a real gap when one opens without breaking the agent.

---

## Bullet summary

| | |
|--|--|
| **Question** | How much detection survives as adversary budget rises? |
| **Method** | Robustness gap Δdet(b) on ladder b₀→b₄; AUROC; TOST retention ≥80% |
| **Setup** | Llama-3.3-70B · Apollo probe (L22) · Modal H200 · N=60 · \|C\|=29 |
| **Key result 1** | Transfer: 0.999 → 0.83 before any attack |
| **Key result 2** | b₂/b₃: probe holds while retention passes (softplus often `attack_too_weak`) |
| **Key result 3** | b₄: AUROC falls but retention fails → Δdet withheld |
| **Claim** | Limits of evasion under this budget; protocol, not an attack recipe |
| **Future tests** | Transfer + adaptive ladder + retention gate + check loss engaged + scale |

---

## Numbers to remember

- Transfer: **0.999 → 0.83**  
- Primary λ=8 probe AUROC: 0.83 / 0.79 / 0.89 / 0.85 / **0.75**  
- Retention at b₄: **0.14** (λ=8), **0.31** (λ=0.5)  

---

## One sentence for Slack / intro

I stress-tested Apollo’s deception probe on agentic agents under a rising attack budget: transfer already hurts (0.999→0.83), usable evasion is limited where the agent still deceives, and I withhold gaps when retention collapses.
