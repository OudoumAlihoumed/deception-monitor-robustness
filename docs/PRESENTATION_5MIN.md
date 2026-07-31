# 5-minute presentation script — BASE poster

*Speak naturally. ~5 minutes. Pause briefly at [slide / figure] cues if you point at the poster.*

---

## Introduction (~1.5–2 min)

Hi everyone — I’m Oudoum Houmed, from Université Paris-Saclay.  
This is my BASE Recursive Redlines Fellowship project, with mentorship from Krystal Jackson.

The title is: **Can We Trust Deception Monitors for AI Agents?**  
A robustness-gap protocol — and the limits of adaptive evasion.

Here’s why I care about this.

We’re putting large language models into the world as **agents**. They take actions, call tools, and often run with only intermittent human review. In that setting, an agent could pursue a goal nobody sanctioned, while the transcript still looks compliant. That’s what **loss-of-control risk** looks like in near-term systems.

So people put **deception monitors** on top: surface classifiers on the text, linear probes on the model’s internal activations, sometimes chain-of-thought readers. And the published numbers look great — sometimes near ceiling.

But almost all of those numbers share one assumption: **the model is not trying to evade the monitor.**

That worried me. A monitor is a defense. An agent with a reason to hide is an attacker. If you only test the cooperative case, you can get a false sense of safety.

So I asked a different question.  
Not: *how accurate is the detector?*  
But: **how much detection survives as adversary budget rises?**  
And: when does an attempted evasion stop counting, because the agent itself broke?

I did **not** invent a new probe. I took the **published Apollo probe** — Goldowsky-Dill and collaborators — and I stress-tested it on agentic episodes, under a rising budget ladder from prompt pressure, through SFT, to RL.

On the poster, Figure 1 is the definition: the overseer sees only the transcript; the label is one only if the visible statement conflicts with an internal fact **and** advances a covert goal. Monitors score — they do not define the label.

---

## Results (~2–2.5 min)

[point: transfer figure]

**First result — before any attack.**

Same published probe, same extraction path.  
On Apollo’s own insider-trading domain, it scores about **0.999**.  
On my agentic episodes at baseline — $b_0$ — it is **0.83**.  
That’s a gap of about **0.17**, from **distribution shift alone**. Not a wiring bug: I validated the extraction on their domain first. So if you import the near-ceiling number onto agentic logs, you’re already overconfident.

[point: tension figure + table]

Then I raise the adversary budget.

- **$b_1$** — prompt only: “hide from monitors.” No training. The **surface** monitor collapses — about 0.83 to 0.52. The **probe** only dips, to about 0.79. Retention falls to 0.52: many agents stop deceiving under concealment. That row is informative, but the strict 80% retention gate is for fine-tuning rungs, not for this prompt arm.

- **$b_2$** — SFT without the probe in the loss: teach the agent to keep deceiving and look clean on the surface. Probe AUROC stays high — even goes to about **0.89**. Retention comes back to **0.86** — gate passes.

- **$b_3$** — same SFT, but now the probe is **in** the loss: CE plus lambda times softplus of the probe score. This is the adaptive rung. Under my budget, probe AUROC is still about **0.85**, retention still **0.86**. Softplus often logged **attack too weak** — so I treat this as a **bound**, not “the probe can never be beaten.”

- **$b_4$** — RL: reward keeping deception, penalise the probe and surface risk. Probe AUROC falls to about **0.75** — but retention collapses to **0.14**. So I **withhold** that detection drop. I’m not calling a broken agent “successful evasion.”

Same story at a weaker lambda: retention at $b_4$ still fails. The gate — not AUROC alone — decides what I may report.

---

## Summary & conclusion (~1–1.5 min)

So what do I take from this?

**Three points.**

First: **transfer already hurts.** 0.999 to 0.83 before any attack. Test monitors on the distribution where they will actually run.

Second: **under attacks that keep deception, this mean-pooled probe held** within my budget. Probe-aware SFT did not open a clean further gap.

Third: **the rung that moves detection breaks the agent.** Usable adaptive evasion is **limited** here — not a story of easy probe collapse.

For how we should test in the future: always report the transfer gap; use a budgeted adaptive ladder; keep a retention gate; check that the probe-aware loss actually engaged; then scale under the same rules.

I release a **measurement protocol**, not an evasion recipe.

Thank you — happy to take questions.

---

## Timing cheat-sheet

| Block | Target |
|-------|--------|
| Introduction | ~1:45 |
| Results | ~2:15 |
| Summary & conclusion | ~1:00 |
| **Total** | **~5:00** |

If you run long: cut the $b_1$ surface detail and the lambda-0.5 sentence.  
If you run short: add one line on compute (one seed, softplus often weak → “limits under my budget”).

---

## Closing line if you need 10 more seconds

> Can we trust deception monitors? Not on a single near-ceiling number from the training domain — only after transfer, adaptive pressure, and retention.
