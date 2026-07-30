# Poster — fellowship / midpoint

**File:** `poster.tex`  
**Theme:** `beamerthemesharelatex.sty` (included)  
**Figures:** `figures/*.png` (all five project figures)

## Compile on Overleaf (recommended)

1. New project → **ShareLaTeX poster** template (A0 portrait), **or** upload:
   - `poster.tex`
   - `beamerthemesharelatex.sty`
   - `figures/` folder (scenario, methodology, transfer, both tension PNGs)
2. Menu → Compiler: **pdfLaTeX**
3. Compile

Overleaf includes `beamerposter` by default; local Basic TeX may not.

## Local compile (if `beamerposter` is installed)

```bash
cd paper
pdflatex poster.tex
pdflatex poster.tex
open poster.pdf
```

## Content

- **Title:** *Can We Trust Deception Monitors for AI Agents? A Robustness-Gap Protocol and the Limits of Adaptive Evasion*
- **Author:** Oudoum A. Houmed, Université Paris-Saclay
- **Figures:** scenario, methodology, transfer, tension (λ=8), tension (λ=0.5)
- **Table:** main AUROC + retention ladder (λ=8 primary)
