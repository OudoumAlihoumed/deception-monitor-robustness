# Practice Q&A — BASE capstone presentation

*Use this out loud. Short answer first, then one extra sentence if they dig.*

---

## A. Core story (most likely)

### Q1. In one sentence, what did you find?

**A:** Under my budget, usable adaptive evasion against the Apollo probe is limited: the probe already loses a lot to transfer, it holds where the agent still deceives, and the rung that moves detection also breaks the agent.

---

### Q2. So are deception monitors trustworthy or not?

**A:** Not on a single near-ceiling number from the training domain. They can still be useful — but only after you test transfer onto agentic data, adaptive pressure, and retention. My result is not “never trust them”; it’s “don’t trust the headline number alone.”

---

### Q3. What’s new compared to Apollo / Goldowsky-Dill?

**A:** They show a strong probe on their domain. I keep that probe fixed and ask how much of that strength survives on agentic episodes and under a budgeted adversary — with a retention gate so a broken agent is not scored as evasion. I’m stress-testing their detector, not replacing it.

---

### Q4. Why is this about loss-of-control?

**A:** If we lean on monitors as a control layer for agents, a false sense of safety is dangerous. An agent that can hide while still pursuing a covert goal is exactly the near-term shape of loss-of-control risk. So the monitor has to be tested against an agent that has a reason to escape.

---

## B. Method and definitions

### Q5. How do you define deception? Isn’t that circular with the monitor?

**A:** No — the label is monitor-independent. $y=1$ only if the visible statement conflicts with the agent’s internal fact **and** that advances a covert goal. The probe and surface classifier only score; they don’t define the label.

---

### Q6. What is Δdet / the robustness gap?

**A:** It’s the drop in AUROC from baseline as adversary budget rises: $\Delta_{\mathrm{det}}(b) = \mathrm{AUROC}(0) - \mathrm{AUROC}(b)$. I only report it on rungs where retention still passes. If the agent stops deceiving, I withhold the gap.

---

### Q7. Why AUROC and not recall at 1% FPR?

**A:** AUROC is threshold-free and stable across rungs. Recall at fixed FPR is useful for ops, but it can jump with threshold choice. For the curve over budget, AUROC is cleaner as the primary metric. (You can still report recall as secondary in the paper.)

---

### Q8. What is the retention gate, and why 80%?

**A:** Fine-tuning can “beat” the monitor by destroying the behaviour you care about. Retention asks: is the agent still deceiving at least ~80% as often as baseline on a frozen cohort? If not, that rung is not successful evasion — it’s a broken agent. The 80% TOST-style bar is a pre-registered practical threshold; the important idea is the gate itself.

---

### Q9. Walk me through b0 to b4.

**A:**  
- **b0:** baseline, no attack.  
- **b1:** prompt concealment, no weight updates.  
- **b2:** SFT, probe *not* in the loss.  
- **b3:** SFT + probe-aware softplus penalty.  
- **b4:** RL with retention / probe / surface terms.  

Primary scientific contrast is **b2 vs b3**: does putting the probe in the loss open a gap while deception is retained?

---

## C. Results (numbers they will ask)

### Q10. What are the key numbers I should remember?

**A:** Transfer **0.999 → 0.83**. Primary λ=8 probe AUROC: **0.83 / 0.79 / 0.89 / 0.85 / 0.75**. Retention at b4: **0.14** (λ=8), **0.31** (λ=0.5). I withhold Δdet at b4.

---

### Q11. Why does AUROC go *up* at b2 (0.89)?

**A:** SFT can make deceptive behaviour more stereotyped or more separable in activation space, so the probe can look *better*, not worse. That’s still consistent with “no usable evasion”: the agent is still deceiving and the probe still separates. I don’t over-interpret a small bump as a big discovery — the important point is it didn’t collapse.

---

### Q12. Why does surface AUROC collapse at b1 but the probe doesn’t?

**A:** Prompt concealment targets the transcript — that’s what the surface monitor reads. The probe reads residual-stream activations, so it’s harder to wipe with wording alone. That contrast is meaningful: black-box and white-box monitors fail differently under the same concealment.

---

### Q13. Why withhold b4 if AUROC fell to 0.75?

**A:** Because retention collapsed to 0.14. The agent mostly stopped deceiving. If I reported that as “we beat the monitor,” I’d be confusing a degraded agent with successful evasion. The gate exists exactly for this case.

---

### Q14. Is the flat b2–b3 result meaningful, or just a weak attack?

**A:** Both, honestly. Softplus often logged `attack_too_weak`, so I **bound** robustness rather than settle it. Under this budget, usable evasion did not open while retention passed. A stronger curriculum might still open a gap — and the same protocol would catch it. I’m not claiming “probes are forever safe.”

---

### Q15. Why are λ=8 and λ=0.5 both interesting?

**A:** λ=8 is my primary cell: detection only moves where retention fails. λ=0.5 is the companion: AUROC can stay high through b4 (0.90) but retention still fails (0.31). So the gate — not AUROC alone — decides what I may report. Same scientific story, two budgets.

---

### Q16. Is N=60 / |C|=29 enough?

**A:** Enough for a fellowship draft with clear limitations; not enough for a final definitive claim. Variance across seeds is unmeasured. I state that in the paper. The protocol is the reusable part; the point estimates need multi-seed follow-up.

---

## D. Compute and limitations

### Q17. Did compute constrain your conclusions?

**A:** Yes. Modal H200 let me run 70B end-to-end, but only one fine-tuning seed, two λ values, and limited search for a harder probe-aware schedule. Softplus never fully engaged. So I report **limits under my budget**, not a universal robustness theorem.

---

### Q18. What would more compute buy you?

**A:** A harder λ / curriculum that actually engages the probe, several seeds, maybe a second model family and true multi-turn episodes — always under the same retention gate. The question I’d re-run first: can b3 open a gap **while** retention still passes?

---

### Q19. Why Llama-3.3-70B and not a closed frontier model?

**A:** I need white-box access for the residual-stream probe and for probe-aware training. Open weights make the experiment reproducible for the safety community. Closed models would block the white-box part of the protocol.

---

### Q20. Single-turn vs multi-turn — does that weaken the claim?

**A:** It limits generality. Episodes are agentic in content but single-turn completion with stacked context. CoT control is weaker in that setup. Transfer and probe results still stand for this distribution; multi-turn is the clear next stress test.

---

## E. Literature and comparison

### Q21. How does this compare to Bailey (latent adversarial training)?

**A:** Bailey shows probes can be attacked in latent space. I ask a measurement question on a *published* probe with a *budget ladder* and a *retention gate* on agentic episodes. Related threat model; different deliverable: a protocol and a measured curve, including the null.

---

### Q22. How does this compare to Gupta & Jenner (RL-obfuscation)?

**A:** They show RL can obfuscate representations. My b4 is in that spirit, but I refuse to score obfuscation that destroys deception. If RL breaks retention, I gate it out. So I’m stricter about what counts as successful evasion.

---

### Q23. Isn’t this just “adversarial robustness applied to probes”?

**A:** Yes — that’s intentional. The contribution is applying that discipline to deception monitors for agents: budgeted adaptive attacker, deployment distribution, and a behaviour retention gate. The field often skips those three.

---

## F. Impact, ethics, “why publish”

### Q24. Aren’t you teaching people how to evade monitors?

**A:** No. I release a measurement protocol and gated results, not an evasion recipe or public evasive checkpoints. The point is to help monitor designers and evaluators test honesty under pressure — same spirit as adaptive evaluation in adversarial ML.

---

### Q25. What’s the takeaway for someone deploying a probe tomorrow?

**A:** Three checks: (1) measure on *your* agentic distribution — expect transfer loss; (2) try a budgeted adaptive ladder; (3) never treat a broken agent as a win for the attacker or a win for the monitor without saying so. Keep a retention gate.

---

### Q26. What should future testing always include?

**A:** Transfer gap on deployment data; budgeted adaptive ladder; retention gate; a log that the probe-aware loss actually engaged; then scale (seeds, multi-turn, other models) under the same gates.

---

## G. Personal / fellowship

### Q27. What did the fellowship deliver for you?

**A:** From literature and protocol design to a full Modal implementation, measured curves, a workshop draft, and a public write-up. The scientific contribution is the robustness-gap protocol with the retention gate.

---

### Q28. If you had two more weeks, what would you do?

**A:** One harder probe-aware cell under the same retention gate — higher effective λ or a curriculum — and maybe a second seed. Writing is secondary to that scientific check.

---

### Q29. What surprised you most?

**A:** I expected a sharp drop once the adversary became probe-aware. Instead, transfer hurt first, and usable evasion under my budget was limited by agent collapse at b4, not by a clean detection drop at b3.

---

## H. Hard / hostile questions (practice these)

### Q30. “Your attack failed — so you have no result.”

**A:** A null under a clear protocol *is* a result. Transfer is not a null — 0.999 to 0.83 is real. For adaptive SFT, I report a bound and the softplus weakness honestly. Overclaiming a collapse I didn’t measure would be worse science.

---

### Q31. “Retention 0.52 at b1 — isn’t your gate inconsistent?”

**A:** The fine-tuning TOST gate applies to b2–b4, where weight updates can destroy behaviour. b1 is prompt-only; I still report retention for honesty, but I don’t treat it the same as a failed SFT/RL rung. Happy to clarify that on the table.

---

### Q32. “0.83 AUROC is still good — why worry?”

**A:** Relative to 0.999 it’s a big drop, and it’s *before* any attack. If your threat model includes adaptive agents, 0.83 is a ceiling, not a floor. Worry about the evaluation story people tell, not only the absolute number.

---

### Q33. “Could a defender just retrain the probe?”

**A:** Maybe — that’s an adaptive defender, and it’s valuable follow-up. My paper measures the *frozen published* probe under attack. Retraining might recover detection; it doesn’t remove the need to test transfer and retention in the first place.

---

## Quick drill (answer in ≤15 seconds)

| Prompt | Your line |
|--------|-----------|
| Main finding? | Limits of evasion under my budget; transfer hurts first; b4 gated. |
| Key number? | 0.999 → 0.83; retention 0.14 at b4. |
| Why withhold b4? | Agent stopped deceiving. |
| Weak attack? | Softplus often too weak — I bound, I don’t oversell. |
| What’s new? | Budgeted ladder + retention gate on a published probe, agentic data. |
| Next step? | Harder b3 under the same gate. |

---

## How to practise

1. Read each **A** out loud once without looking.  
2. Have a friend ask only the **Q** from sections A, C, H.  
3. If you ramble past ~45 seconds, stop and restart with the first sentence only.  
4. End every hard answer with: *“That’s under my budget / with the retention gate.”*
