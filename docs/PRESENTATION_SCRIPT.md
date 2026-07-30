# Talk script — BASE blog / capstone presentation

*Speak this in first person. ~8–10 minutes. Pause at [slide] markers if you use the poster or one-slide summary.*

---

## Opening (about 1 min)

Hi everyone — I’m Oudoum Houmed, from Université Paris-Saclay.

This work is my BASE Recursive Redlines Fellowship project, with mentorship from Krystal Jackson.

The title is: **Can We Trust Deception Monitors for AI Agents?**  
Subtitle: a robustness-gap protocol, and the limits of adaptive evasion.

Here’s the short version of what I’ll say.  
People publish very high numbers for deception monitors. I asked a different question: **how much of that detection still works when the agent is under pressure to hide?**  
What I found is not “the probe collapses.” It’s more careful than that. Under my budget, usable evasion hits a limit — and the attack that moves detection also breaks the agent.

---

## Why this matters (about 1.5 min)

[slide: problem / Figure 1]

We’re deploying LLMs as agents. They take actions, call tools, run with only intermittent human review. In that setting, an agent could pursue a goal nobody sanctioned, while the transcript still looks compliant. That’s the concrete shape of loss-of-control risk for near-term systems.

So we put monitors on top: surface classifiers, linear probes on activations, chain-of-thought readers. And the published numbers look great — sometimes near ceiling.

But almost all of those numbers share one assumption: **the model is not trying to evade the monitor.**

That’s the same mistake adversarial robustness warned us about for years. A monitor is a defense. An agent with a reason to hide is an attacker. If you only test the cooperative case, you can get a false sense of safety.

I didn’t invent a new probe. I took the published Apollo probe — Goldowsky-Dill and collaborators — and I stress-tested it on agentic episodes, under a rising adversary budget.

---

## What was missing in the literature (about 45 sec)

Probes already look strong on their own domain. Other work shows probes *can* be attacked — latent adversarial training, RL-obfuscation. What was missing for me was a clean measurement: **how fast does a published probe degrade as budget rises**, on agentic data, **with a retention gate** so a broken agent is not counted as “we beat the monitor.”

That’s the gap this project fills.

---

## What I measured (about 2 min)

[slide: protocol / Figure 2]

First I fix the definition. Deception is not “the monitor said so.”  
Label is 1 only if the visible statement conflicts with the agent’s internal fact **and** that helps a covert goal. The monitor scores; it does not define the label. Seven elicitation families.

Then I measure a **robustness gap**: how much AUROC drops as adversary budget goes up — from prompt, to SFT without the probe in the loss, to SFT with the probe in the loss, to RL.

And every fine-tuning rung has a **retention gate**: if training stops the agent from deceiving, I withhold the gap. I don’t give the attacker credit for breaking the agent.

Setup, briefly: Llama-3.3-70B on Modal H200, published Apollo probe at layer 22, sixty sessions, frozen deceptive cohort of twenty-nine.

---

## What I found (about 2.5 min)

[slide: transfer + tension + table]

**First result — before any attack.**  
Same probe, same extraction path: AUROC **0.999** on Apollo’s domain, **0.83** on my agentic episodes. That’s a transfer gap of about 0.17. Distribution shift, not a wiring bug. And that already caps everything that comes after. So if you deploy this probe on agentic logs and you only remember the 0.999 number, you’re already overconfident.

**Then the ladder, primary cell λ=8.**  
Through prompt and SFT — b1, b2, b3 — the probe stays near baseline: 0.79, 0.89, 0.85. Retention at b2 and b3 still passes.  
At b4, RL, probe AUROC falls to 0.75 — but retention collapses to **0.14**. So I withhold that gap. I’m not calling that evasion.

Same story at λ=0.5: AUROC can stay high, but retention at b4 still fails — 0.31.

**Why do the numbers move?**

- Transfer is the big real drop, and it’s meaningful for deployers.  
- Prompt hits the surface monitor hard; the probe only dips — black-box and white-box don’t fail the same way.  
- Probe-aware SFT did not open a further gap while the agent still deceived. Softplus often logged `attack_too_weak`, so I treat that as a bound, not “probes are forever safe.”  
- RL moves detection only when the agent breaks. That’s meaningful as a warning against scoring a broken agent as success.

---

## Compute — honest (about 45 sec)

I had fellowship GPU on Modal. That was enough for a full 70B ladder. It was not enough for many seeds, a huge λ search, or a long curriculum. One seed, N=60, two λ values, and the probe-aware penalty often stayed weak.  

So my claim is **limits of evasion under my budget** — not “no attack will ever work.” More compute could still open a real gap while retention passes. If it does, this same protocol will report it.

---

## How we should test next (about 1.5 min)

[slide: takeaways]

I went in expecting a sharp drop once the adversary became probe-aware. What I got is tighter: under attacks that keep deception, this probe held; the attack that moves detection breaks the agent.

For future testing, five things I’d push:

1. Always measure on the **agentic / deployment** distribution — report the transfer gap.  
2. Use a **budgeted adaptive ladder**, not one static accuracy.  
3. Keep a **retention gate** — never score a broken agent as a defeated monitor.  
4. Check that the probe-aware loss **actually engaged** before you claim robustness.  
5. Then scale: harder curriculum, multi-seed, multi-turn, second model — under the same gates.

I release a measurement instrument, not an evasion recipe.

---

## Close (about 30 sec)

So: can we trust deception monitors for AI agents?  
Not on a single near-ceiling number from the training domain.  
Trust them only after you’ve tested transfer, adaptive pressure, and retention.

Thank you — happy to take questions.  
Mentor: Krystal Jackson. BASE Recursive Redlines Fellowship.

---

## Optional Q&A one-liners

- **“Is the probe robust?”** — Under my budget, yes where deception is retained. Softplus was often weak, so I bound that claim; I don’t oversell it.  
- **“Why withhold b4?”** — Retention 0.14. The agent stopped deceiving. That’s not evasion.  
- **“What’s next?”** — Harder probe-aware schedule under the same retention gate; multi-seed; multi-turn.
