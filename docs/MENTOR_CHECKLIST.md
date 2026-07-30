# Mentor checklist — what I need from you
*Updated 27 July 2026 · fellowship ends Fri 31 July*

Please read this with `paper/midpoint_checkin.tex` and `paper/main.pdf`.

---

## Where things stand (honest)

I finished the design work earlier in the fellowship. Since then I also finished the main implementation: Llama-3.3-70B on Modal, against the published Apollo probe, with the full budget ladder and the retention gate. The workshop draft is written (`paper/main.pdf`, about 10 pages, five figures).

What the numbers say, without dressing them up:

- Before any attack, the probe already drops from **0.999** on Apollo’s domain to **0.83** on my agentic episodes.
- Where the agent still deceives (`b2`, `b3`), the probe holds.
- At `b4`, retention fails (**0.14** at λ=8, **0.31** at λ=0.5), so I withhold Δdet — I do not call that evasion.
- Softplus at `b3` often logs `attack_too_weak`. I report that; I do not hide it.
- I only ran **one** fine-tuning seed. A stronger λ schedule / second seed is still open.

So the story is not “the probe collapses under adaptive attack.” It is closer to: **usable evasion hits a limit** — pressure that keeps deception does not move this mean-pooled probe much, and pressure that moves detection breaks the agent.

---

## What I need you to decide (please tick)

### 1. Is this enough to deposit for the fellowship?
- [ ] **Yes — deposit now** with the honest draft (`main.pdf` + midpoint), limitations clear.
- [ ] **Not yet** — you want one more GPU pass (stronger λ / more steps) first.
- [ ] Deposit the **design only**; treat all adaptive numbers as preliminary.

My own preference is the first option, with Limitations explicit about `attack_too_weak` and the single seed.

### 2. Framing
- [ ] Keep the current title (*…A Robustness-Gap Protocol and the Limits of Adaptive Evasion*).
- [ ] Push back toward a big `b3` gap story (I would rather not — the data do not support that).

### 3. If there is still compute budget
- [ ] Writing + figures only (no more GPU).
- [ ] One harder `b3`/`b4` cell under the same retention gate.
- [ ] A second seed instead of a harder λ.

### 4. Where to aim publicly after deposit
- [ ] NeurIPS 2026 workshop first (contribution ~29 Aug) — I was looking at **AI4GOOD** and **EIML3**.
- [ ] arXiv first, then workshop.
- [ ] Fellowship deposit only for now; decide later.

### 5. Midpoint vs workshop draft
- [ ] Midpoint = progress + questions for you; `main.pdf` = the scientific draft.
- [ ] Treat `main.pdf` as the main deliverable and keep midpoint short.

---

## Already done (no decision needed)

Scenario + methodology figures, transfer figure, tension figures (λ=8 and λ=0.5), retention gate, code under `scripts/modal_app.py` and cells in `results/checkpoints/cells/`.

---

## Files for the meeting

| File | What it is |
|------|------------|
| this checklist | decisions I need from you |
| `paper/midpoint_checkin.tex` | fellowship midpoint template |
| `paper/poster.tex` | A0 poster (compile on Overleaf) |
| `paper/main.pdf` | workshop draft to read |
| `docs/OVERVIEW.md` | map of the project on my laptop |

```bash
open "paper/main.pdf"
```

---

## The four questions I will ask you in the midpoint

1. Is this honest “limits of evasion” result acceptable as the fellowship scientific result?  
2. With the days left: polish the writing, or one more GPU cell?  
3. NeurIPS workshop first, or arXiv first?  
4. Is one seed and \(N{=}60\) / \(|\mathcal{C}|{=}29\) enough for deposit if I state the limits clearly?
