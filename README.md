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
| 4 | 3D solver, validation, permeability of random 3D networks | Done |
| 5 | Particle transport and capture efficiency (2D) | Done |
| 6 | Design study and sensitivity analysis | — |
| 7 | Discussion, journal template, reference check | — |

## Folder layout

```
src/classical_models.py              cell-model permeability and single-fibre efficiency formulas
src/rs_kernels.py                    our 2D Gaussian regularized Stokeslet (velocity, pressure)
src/periodic.py                      our doubly periodic version (Ewald summation)
src/geometry.py                      fibre layouts: square, hexagonal, random mono-/polydisperse
src/solver2d.py                      our 2D regularized Stokeslet solver (dense LU or GMRES)
src/reference_bie.py                 independent exact references: spectral boundary integrals for flow across
                                     and along parallel circular fibres
src/rs_kernels3d.py                  our 3D Gaussian regularized Stokeslet (velocity, pressure)
src/periodic3d.py                    our triply periodic version (Ewald summation)
src/geometry3d.py                    periodic straight fibres in 3D: lattices, random networks, blob points
src/flowfield2d.py                   fast exact copy of the 2D RS flow (FFT + B-spline) and the near-wall
                                     reconstruction around each fibre
src/particles2d.py                   particle tracker: Brownian motion, interception, inertia; capture
src/solver3d.py                      our 3D solver: surface method with pressure balance, centreline model,
                                     permeability tensor
scripts/stage1_classical_baselines.py tables and figures of the classical baselines
scripts/stage2_kernel_checks.py      numerical evidence for the Stage 2 propositions, with figures
scripts/stage3_validation.py         validation against Sangani & Acrivos, convergence, leakage, timing
scripts/stage3_random_arrays.py      experiment (a): 480 random layouts, each checked against the reference
scripts/stage3_random_summary.py     statistics and figure for experiment (a)
scripts/stage3_cv_test.py            why 40 layouts: control test with nearly identical fibres
scripts/stage4_validation.py         3D kernel checks; lattices with exact answers; crossing lattice; timing
scripts/stage4_networks.py           experiment (b): random 3D networks (layered, cubic, aligned) + 2D inputs
scripts/stage4_summary.py            statistics, orientation rule, figures for experiment (b)
scripts/stage5_validation.py         flow next to fibres, tracker against exact answers and theory
scripts/stage5_capture.py            experiment (c): capture by square, hexagonal and random arrays
scripts/stage5_summary.py            efficiencies, MPPS, penetration, quality factor, figures
scripts/stage6_design.py             experiments (d) design study and (e) fibre-size spread; sensitivity runs
scripts/stage6_summary.py            quality factors, elasticities, layered filters (Proposition 10), figures
scripts/make_bib.py                   builds paper/refs.bib from the literature table
checks/check_stage1.py                all Stage 1 checks (and proof that each check can fail)
checks/check_stage2.py                all Stage 2 checks, plus a re-run of the Stage 1 checks
checks/check_stage3.py                all Stage 3 checks, plus a re-run of the Stage 1 and 2 checks
checks/check_stage4.py                all Stage 4 checks, plus a re-run of the Stage 1-3 checks
checks/check_stage5.py                all Stage 5 checks, plus a re-run of the Stage 1-4 checks
checks/check_stage6.py                all Stage 6 checks, plus a re-run of the Stage 1-5 checks
literature/literature_table.csv       every reference, its role, and how it was verified
results/stage1/ ... results/stage6/  CSV outputs and summary numbers
paper/                                LaTeX draft (kept out of GitHub)
```

## How to run (Windows, Python 3.13)

```
cd "C:\Users\Bernard Frimpong\Desktop\fibre-filtration-stokes"
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
python scripts\stage4_validation.py
python scripts\stage4_networks.py
python scripts\stage4_summary.py
python checks\check_stage4.py
python scripts\stage5_validation.py
python scripts\stage5_capture.py
python scripts\stage5_summary.py
python checks\check_stage5.py
python scripts\stage6_design.py
python scripts\stage6_summary.py
python checks\check_stage6.py
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

## Stage 4 results (3D solver and random 3D networks)

- Our own 3D Gaussian regularized Stokeslet and triply periodic Ewald sum: independent of the splitting width
  (6e-16) and equal to the plain Fourier series (4e-15); satisfies the forced Stokes equations.
- Proposition 8: in a regularized fibre the inside is fluid, and the mean pressure gradient drives pipe flow inside
  fibres that point along it. A known body force equal to the mean pressure gradient inside each fibre removes this
  (no extra unknowns). Without it, flow along fibres is off by up to 5.5%; with it, Richardson extrapolation gives
  0.24% or better on all exact test lattices (across, along and at 45 degrees to the fibres) for solid fraction <= 0.2.
- Darcy's law is used as a matrix (permeability tensor from three flow directions); the tensors come out
  symmetric although nothing imposes it.
- Experiment (b), 104 random networks (crossed layers, three-axis "cubic", and parallel fibres as an accuracy check),
  solid fractions 0.1 and 0.15: an orientation rule that averages exact 2D resistances predicts them to about 11%;
  flow through a layered mat is 6-11% easier than the rule says; Jackson-James overestimates the cubic networks by
  about 15%; Kuwabara underestimates the layered mat by a factor 1.2-1.3.
- The cheap centreline model (one blob per cross-section) is 2-4% off on average and up to 12% on single networks,
  so the surface method is used from now on.
- Cost: dense solves, about 1-3 minutes per network on one core; larger cells will need fast summation.

`stage4_networks.py` takes about 3-4 hours on one core (results are saved as they come and the script continues where
it stopped). To use two cores, run `python scripts\stage4_networks.py 0.1` and `python scripts\stage4_networks.py 0.15`
in two windows at the same time. The results are included, so `python checks\check_stage4.py` can be run directly (about 3-5 minutes).

## Stage 5 results (particle capture, 2D)

- The regularized Stokeslet flow slips along the fibre surface (about 3% of the mean speed at eps = 0.05a, first
  order in eps), exactly where particles are caught. Proposition 9 replaces the flow in a thin ring around each fibre by
  the exact no-slip Stokes flow that matches the RS flow farther out: the error next to the fibre drops from 9e-3 to
  4e-4 of the mean speed.
- The tracker reproduces the exact interception answer, straight-line motion of heavy particles, and the boundary-layer
  (Leveque) theory for Brownian capture within its statistical error.
- Even a small particle inertia matters: air curving round the shoulders of a fibre throws particles outward, which
  lowers interception by about 11% at the Stokes number of a 0.25 um particle and lowers total capture there by 9-17%
  (more than the kinetic boundary layer of Brownian motion alone would).
  So every particle is followed with Brownian motion, interception and inertia together (full Langevin equation).
- Filter of Stage 1 (d_f = 3 um, alpha = 0.1, 0.1 m/s, 0.5 mm): the classical formulas over-predict capture near the
  most penetrating size (penetration 0.465 on the hexagonal array instead of 0.396, or 0.357 with Hinds' recipe; MPPS
  about 0.28 instead of 0.21 um, a shift that is only just resolved). Without inertia, diffusion plus interception agree with the classical E_D + E_R + E_DR; the
  difference is that inertia lowers capture in the simulation but the classical impaction correlation raises it.
- Random arrays of identical fibres catch about as much as the hexagonal array at the MPPS (ratio 0.94, interval
  0.86-1.08) but have about 21% lower pressure drop, so their quality factor is about 19% higher (interval 13-32%).
  A fibre-size spread lowers the pressure drop again; its effect on capture and on the quality factor is not resolved
  with six cells.

`stage5_capture.py` takes about 9 hours on one core (it saves as it goes, so it can be stopped and restarted). To use
two cores, run `python scripts\stage5_capture.py hexagonal random` and `python scripts\stage5_capture.py square random_poly`
in two windows (about 4.5 hours each). `stage5_validation.py` takes about 40 minutes. The results are included, so
`python checks\check_stage5.py` can be run directly (about 5 minutes).

## Stage 6 results (design study and sensitivity)

- For a given design the quality factor does not depend on the filter thickness: q_F = alpha eta k d_f / (pi mu U),
  so the worst particle size is where the single-fibre efficiency is smallest.
- Proposition 10: for independent layers in series, the stack's quality factor at every particle size is the
  pressure-drop-weighted mean of its layers'. A stack can only raise the worst case, and only if the second layer
  beats the first at the first layer's MPPS. A fine (1.5 um) + coarse (6 um) stack gains about 3%, not resolved
  from zero.
- Over eight designs (fibre diameter, face velocity, solid fraction, NaCl or oil aerosol) the computed q_F at the
  MPPS on the hexagonal array is only 0.74-0.90 of the classical value, but the classical formulas get the trends with fibre diameter and face velocity
  about right; q_F falls with solid fraction faster than they predict. The test aerosol makes no measurable difference.
- Random arrays of identical fibres have 9-24% higher q_F than the hexagonal array at solid fraction 0.1 or less;
  at 0.15 the comparison is not resolved.
- A wide spread of fibre sizes (CV = 0.5) raises q_F by about 13% (16 cells per spread).
- Blob width and near-wall settings change capture by at most about 2%; larger cells show no effect of cell size
  and no correlation between successive cell lengths (the test of the independent-layer model).

`stage6_design.py` takes about 13 hours on one core (it saves as it goes). To use two cores, run
`python scripts\stage6_design.py random1 poly` and `python scripts\stage6_design.py hexagonal random2 sensitivity`
in two windows. The results are included, so `python checks\check_stage6.py` can be run directly (about 10 minutes).

## Scope

Mechanical filtration only (no electret charge); microfibres (diameter about 1 µm or more, so gas slip
on fibres is neglected); Stokes flow (fibre Reynolds number much less than 1).

## Use of AI tools

Claude Opus 5.5 (Anthropic), through the Claude app and Claude Code, was used to search and summarise
literature, draft derivations, draft and test code and text, run the scripts that produce the results, and
review code and derivations. All numbers come from the scripts in this repository, which were run by the AI tool
and are re-run and checked by the author; references not yet confirmed are flagged in `literature/literature_table.csv` and will be checked
before submission. See the Methods section of the paper for the full statement.
