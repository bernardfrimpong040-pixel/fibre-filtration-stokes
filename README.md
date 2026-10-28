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
| 1 | Literature check, classical baselines, project set-up | Done |
| 2 | Theory, model formulation, our own 2D kernels | Done |
| 3 | 2D solver, validation, convergence | — |
| 4 | 3D solver and permeability | — |
| 5 | Particle transport and capture efficiency | — |
| 6 | Design study and sensitivity analysis | — |
| 7 | Discussion, journal template, reference check | — |

## Folder layout

```
src/classical_models.py              cell-model permeability and single-fibre efficiency formulas
src/rs_kernels.py                    our 2D Gaussian regularized Stokeslet (velocity, pressure)
src/periodic.py                      our doubly periodic version (Ewald summation)
scripts/stage1_classical_baselines.py tables and figures of the classical baselines
scripts/stage2_kernel_checks.py      numerical evidence for the Stage 2 propositions, with figures
scripts/make_bib.py                   builds paper/refs.bib from the literature table
checks/check_stage1.py                all Stage 1 checks (and proof that each check can fail)
checks/check_stage2.py                all Stage 2 checks, plus a re-run of the Stage 1 checks
literature/literature_table.csv       every reference, its role, and how it was verified
results/stage1/, results/stage2/     CSV outputs and summary numbers
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
python scripts\stage2_kernel_checks.py
python checks\check_stage2.py
```

## Stage 1 results (classical baseline)

For an uncharged microfibre layer (fibre diameter 3 µm, solid fraction 0.10, thickness 0.5 mm,
face velocity 0.10 m/s, NaCl particles):

- Kuwabara permeability 2.81e-12 m², pressure drop 323 Pa
- MPPS 0.21 µm, penetration 0.36 at the MPPS
- quality factor at the MPPS 3.2e-3 1/Pa

These are the numbers the regularized Stokeslet simulations will be compared against.

## Stage 2 results (model and kernels)

- 2D Gaussian regularized Stokeslet derived in closed form and coded; it satisfies the forced Stokes
  equations (finite-difference residual below 1e-5 of the largest term).
- Far from a blob the flow is a point force plus a dipole of strength eps^2/2 (Proposition 2);
  one blob acts like a cylinder of radius 0.749 eps (Proposition 3).
- Doubly periodic kernel by Ewald summation: independent of the splitting width and equal to the plain
  Fourier series to machine precision (also for elongated cells).
- One blob per periodic cell reproduces Hasimoto's constant (1.4762 vs 1.476 at alpha = 1e-4) and stays
  within 0.8% of the cylinder permeability (full square-array series) at alpha = 0.1.
- Net flux out of a fibre is always zero (Proposition 1), so leakage is measured by the positive-part
  and chord fluxes: 31% for 8 tiny blobs, about 1e-6 when the blob width matches the spacing.

## Scope

Mechanical filtration only (no electret charge); microfibres (diameter about 1 µm or more, so gas slip
on fibres is neglected); Stokes flow (fibre Reynolds number much less than 1).

## Use of AI tools

Claude Opus 5.5 (Anthropic), through the Claude app and Claude Code, was used to search and summarise
literature, draft derivations, draft and test code and text, run the scripts that produce the results, and
review code and derivations. All numbers come from the scripts in this repository, which the author runs and
checks; references not yet confirmed are flagged in `literature/literature_table.csv` and will be checked
before submission. See the Methods section of the paper for the full statement.
