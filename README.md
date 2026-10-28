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
| 3 | 2D solver, validation, convergence, random 2D arrays | Done |
| 4 | 3D solver and permeability | — |
| 5 | Particle transport and capture efficiency | — |
| 6 | Design study and sensitivity analysis | — |
| 7 | Discussion, journal template, reference check | — |

## Folder layout

```
src/classical_models.py              cell-model permeability and single-fibre efficiency formulas
src/rs_kernels.py                    our 2D Gaussian regularized Stokeslet (velocity, pressure)
src/periodic.py                      our doubly periodic version (Ewald summation)
src/geometry.py                      fibre layouts: square, hexagonal, random mono-/polydisperse
src/solver2d.py                      our 2D regularized Stokeslet solver (dense LU or GMRES)
src/reference_bie.py                 independent exact reference: spectral boundary integral (circles only)
scripts/stage1_classical_baselines.py tables and figures of the classical baselines
scripts/stage2_kernel_checks.py      numerical evidence for the Stage 2 propositions, with figures
scripts/stage3_validation.py         validation against Sangani & Acrivos, convergence, leakage, timing
scripts/stage3_random_arrays.py      experiment (a): 480 random layouts, each checked against the reference
scripts/stage3_random_summary.py     statistics and figure for experiment (a)
scripts/stage3_cv_test.py            why 40 layouts: control test with nearly identical fibres
scripts/make_bib.py                   builds paper/refs.bib from the literature table
checks/check_stage1.py                all Stage 1 checks (and proof that each check can fail)
checks/check_stage2.py                all Stage 2 checks, plus a re-run of the Stage 1 checks
checks/check_stage3.py                all Stage 3 checks, plus a re-run of the Stage 1 and 2 checks
literature/literature_table.csv       every reference, its role, and how it was verified
results/stage1/ ... results/stage3/  CSV outputs and summary numbers
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
python scripts\stage3_validation.py
python scripts\stage3_random_arrays.py
python scripts\stage3_random_summary.py
python scripts\stage3_cv_test.py
python checks\check_stage3.py
```

`stage3_random_arrays.py` takes about 80-90 minutes on one core; it saves every result as it goes and
continues where it stopped if interrupted. To use two cores, run `python scripts\stage3_random_arrays.py 0.05 0.3`
and `python scripts\stage3_random_arrays.py 0.1 0.2` in two windows at the same time.

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

## Stage 3 results (2D solver and random arrays)

- An independent exact reference (spectral boundary integral) reproduces the Sangani & Acrivos (1982)
  square-array drag to 2e-4 for solid fractions 0.05-0.5; it also corrected the hexagonal dilute
  constant used in Stage 1 (1.4975, not 1.490).
- Proposition 6: a ring of blobs acts like a no-slip cylinder of radius R + eps/(2 sqrt(pi)). Pulling
  the blobs in by that amount turns first-order into second-order convergence; with Richardson
  extrapolation the error is 0.1% at eps = 0.1a for regular arrays.
- Experiment (a), 480 random layouts of 16 fibres, each checked against the reference (RS error:
  mean 0.07%, worst 1.0%): random arrays are 39% more permeable than regular arrays at alpha = 0.05 and
  18% at 0.1; the advantage disappears by 0.2-0.3. Cell models therefore underestimate random media.
- Fibre-size spread (CV 0.5) raises k by 34-50%; scaling by the r.m.s. radius (Proposition 7)
  removes about 60% of that, and the first logarithmic correction explains the rest in dilute media.
- k varies by 17-39% between 16-fibre layouts, so 40 layouts per case are used.

## Scope

Mechanical filtration only (no electret charge); microfibres (diameter about 1 µm or more, so gas slip
on fibres is neglected); Stokes flow (fibre Reynolds number much less than 1).

## Use of AI tools

Claude Opus 5.5 (Anthropic), through the Claude app and Claude Code, was used to search and summarise
literature, draft derivations, draft and test code and text, run the scripts that produce the results, and
review code and derivations. All numbers come from the scripts in this repository, which the author runs and
checks; references not yet confirmed are flagged in `literature/literature_table.csv` and will be checked
before submission. See the Methods section of the paper for the full statement.
