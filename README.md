# fibre-filtration-stokes

Regularized Stokeslet simulations of breathability and aerosol capture in random fibrous filters
(face masks and air filters).

**Research question.** Real filters are random, polydisperse, layered networks of micrometre fibres, but
design still uses 1950s cell models that assume regular arrays of identical fibres. How does real
structure change (1) permeability / pressure drop and (2) single-fibre and overall capture efficiency,
including the most penetrating particle size (MPPS)? Which structures give the best quality factor
q_F = -ln(P) / dP?

## Status

| Stage | Content | Status |
|---|---|---|
| 1 | Literature check, classical baselines, project set-up | Done (thesis/code review pending) |
| 2 | Theory and model formulation | — |
| 3 | 2D solver, validation, convergence | — |
| 4 | 3D solver and permeability | — |
| 5 | Particle transport and capture efficiency | — |
| 6 | Design study and sensitivity analysis | — |
| 7 | Discussion, journal template, reference check | — |

## Folder layout

```
src/classical_models.py              cell-model permeability and single-fibre efficiency formulas
scripts/stage1_classical_baselines.py tables and figures of the classical baselines
scripts/make_bib.py                   builds paper/refs.bib from the literature table
checks/check_stage1.py                all Stage 1 checks (and proof that each check can fail)
literature/literature_table.csv       every reference, its role, and how it was verified
results/stage1/                       CSV outputs and summary numbers
paper/                                LaTeX draft (kept out of GitHub)
```

## How to run (Windows, Python 3.13)

```
cd "C:\Users\Bernard Frimpong\Desktop\Filtration_paper"
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src\classical_models.py
python scripts\stage1_classical_baselines.py
python scripts\make_bib.py
python checks\check_stage1.py
```

## Stage 1 results (classical baseline)

For an uncharged microfibre layer (fibre diameter 3 µm, solid fraction 0.10, thickness 0.5 mm,
face velocity 0.10 m/s, NaCl particles):

- Kuwabara permeability 2.81e-12 m², pressure drop 323 Pa
- MPPS 0.21 µm, penetration 0.36 at the MPPS
- quality factor at the MPPS 3.2e-3 1/Pa

These are the numbers the regularized Stokeslet simulations will be compared against.

## Scope

Mechanical filtration only (no electret charge); microfibres (diameter about 1 µm or more, so gas slip
on fibres is neglected); Stokes flow (fibre Reynolds number much less than 1).

## Use of AI tools

Claude Opus 5.5 (Anthropic), through the Claude app, was used to search and summarise literature and to
draft and test code and text. The code is run and reviewed by the author; references not yet confirmed
are flagged in `literature/literature_table.csv` and will be checked before submission.
