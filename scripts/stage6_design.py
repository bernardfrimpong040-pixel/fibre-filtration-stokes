"""
stage6_design.py
================
Stage 6: the design study (experiment d), polydispersity (experiment e) and sensitivity runs.
Every particle is followed as in Stage 5 (Brownian motion, interception and inertia; stage5_capture.model).

Groups (each writes its own CSV in results/stage6/, appended as it goes; rerun to continue):
  hexagonal   : experiment (d) on the hexagonal array, every design in DESIGNS
  random1, random2 : experiment (d) on random cells of 16 identical fibres (cells 0-2 and 3-5 of
                6 per solid fraction; alpha = 0.1: the same 6 cells as Stage 5), every design
  poly        : experiment (e): 16 random cells for each fibre-size spread CV = 0, 0.25, 0.5,
                the Stage 1 filter, the five grid sizes around the MPPS (0.169-0.483 um)
  sensitivity : blob width and near-wall settings (hexagonal array), and cell size (random cells
                of 32 fibres, two cells long, and of 36 fibres, 1.5 cells wide and long, at 0.22 and
                0.483 um)

Run from the project folder:   python scripts/stage6_design.py                (all groups; about 13 hours)
Two cores:                     python scripts/stage6_design.py random1 poly
                               python scripts/stage6_design.py hexagonal random2 sensitivity
Then:                          python scripts/stage6_summary.py
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import geometry as geo                                           # noqa: E402
import classical_models as cm                                    # noqa: E402
import particles2d as pt                                         # noqa: E402
import stage5_capture as sc                                      # noqa: E402  (particle model of Stage 5)
from solver2d import solve_with_rule                             # noqa: E402
from flowfield2d import StreamFunctionField2D, NearWallField2D   # noqa: E402
from reference_bie import reference_k                            # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "stage6")
RHO_DEHS = 912.0          # density of DEHS oil test aerosol (EN 149 tests), kg/m^3

# ---- experiment (d): one factor at a time around the filter of Stage 1 ----
# name: (fibre diameter m, face velocity m/s, solid fraction, particle density kg/m^3)
DESIGNS = {
    "base":     (3.0e-6, 0.10, 0.10, cm.RHO_NACL),
    "d_f=1.5":  (1.5e-6, 0.10, 0.10, cm.RHO_NACL),
    "d_f=6":    (6.0e-6, 0.10, 0.10, cm.RHO_NACL),
    "U=0.05":   (3.0e-6, 0.05, 0.10, cm.RHO_NACL),
    "U=0.2":    (3.0e-6, 0.20, 0.10, cm.RHO_NACL),
    "alpha=0.05": (3.0e-6, 0.10, 0.05, cm.RHO_NACL),
    "alpha=0.15": (3.0e-6, 0.10, 0.15, cm.RHO_NACL),
    "DEHS":     (3.0e-6, 0.10, 0.10, RHO_DEHS),
}
# one common size grid (factor 1.3 apart); each design runs the window of it that brackets its MPPS
# (the classical MPPS times about 1.3, Section 8), because the smallest sizes cost the most time steps.
# Every window contains 0.169-0.371 um, so any two designs can be stacked as layers (Proposition 10).
SIZES_UM = [0.1, 0.13, 0.169, 0.22, 0.286, 0.371, 0.483, 0.627]
DESIGN_SIZES_UM = {
    "base":       SIZES_UM[1:7],        # 0.13-0.483 um
    "d_f=1.5":    SIZES_UM[0:7],        # 0.1-0.483 um (smaller MPPS)
    "d_f=6":      SIZES_UM[1:8],        # 0.13-0.627 um (larger MPPS)
    "U=0.05":     SIZES_UM[2:8],        # 0.169-0.627 um
    "U=0.2":      SIZES_UM[0:6],        # 0.1-0.371 um
    "alpha=0.05": SIZES_UM[1:7],
    "alpha=0.15": SIZES_UM[1:7],
    "DEHS":       SIZES_UM[2:8],
}
N_RANDOM = 6
RANDOM_SPLIT = {"random1": range(0, 3), "random2": range(3, 6)}    # cells per window (two cores)
N_PARTICLES = {"hexagonal": 40000, "random": 4000}

# ---- experiment (e): fibre-size spread ----
POLY_CVS = [0.0, 0.25, 0.5]
N_POLY = 16
POLY_SIZES_UM = [0.169, 0.22, 0.286, 0.371, 0.483]   # the grid sizes around the MPPS of the Stage 1 filter
N_POLY_PARTICLES = 4000

# ---- sensitivity ----
SENS_SIZES_UM = [0.22, 0.483]
N_SENS_PARTICLES = 160000   # hexagonal variants: 4x the usual number, so that changes of 2-3% are resolved
CELL_SIZES_UM = [0.22, 0.483]   # diffusion-dominated and interception-dominated sizes
N_BIG = 4

EPS = sc.EPS              # 0.05 a_mean, as in Stage 5
MIN_GAP = sc.MIN_GAP      # 0.5 a_mean

DESIGN_COLUMNS = ["design", "structure", "seed", "alpha", "cv", "n_fibres", "Lx", "d_f_um", "U", "rho_p",
                  "d_p_um", "r_p", "Pe", "St", "D", "h", "n_particles", "f_caught", "n_stuck", "eta",
                  "k_over_a2", "seconds"]
CHECK_COLUMNS = ["structure", "seed", "alpha", "n_fibres", "r_p", "f_exact", "f_tracker", "n_particles"]
SENS_COLUMNS = ["case", "structure", "seed", "n_fibres", "aspect", "eps", "r_out_gap", "n_modes", "d_p_um",
                "n_particles", "f_caught", "n_stuck", "eta", "k_over_a2", "seconds"]


def design_cells(structure, alpha, which=range(N_RANDOM)):
    """(seed, layout) pairs: the hexagonal array, or random cells of 16 identical fibres (numbers in
    `which` out of N_RANDOM). At alpha = 0.1 the random cells are those of Stage 5 (same seeds), so the
    base design can be compared with Stage 5 directly."""
    if structure == "hexagonal":
        return [(0, geo.hexagonal_array(alpha))]
    out = []
    for r in which:
        seed = 700000 + r if alpha == 0.1 else 760000 + 1000 * int(round(alpha * 100)) + r
        out.append((seed, geo.random_array(alpha, 16, np.random.default_rng(seed), min_gap=MIN_GAP)))
    return out


def build(layout, eps=EPS, r_out_gap=0.3, n_modes=64):
    """Flow field with near-wall reconstruction (Stage 5) and the exact permeability k / a_mean^2."""
    s = solve_with_rule(layout, eps)
    W = NearWallField2D(StreamFunctionField2D(s, grid_factor=4), r_out_gap=r_out_gap, n_modes=n_modes)
    return W, reference_k(layout, 64)


def interception_check(W, layout, structure, seed, writer, checked):
    """Pure interception (r_p = 0.1a, 10000 particles) against the exact stream-function answer
    (written once per cell: `checked` holds the (structure, seed, alpha) already in the file)."""
    ex = pt.interception_exact(W, layout, 0.1)
    f = pt.track_pass(W, layout, 0.1, 0.0, 0.0, 0.04, 10000, np.random.default_rng(seed + 1))["caught"].mean()
    alpha = round(float(geo.solid_fraction(layout)), 4)
    if (structure, str(seed), str(alpha)) not in checked:
        writer.writerow([structure, seed, alpha, len(layout["radii"]), 0.1, ex, f, 10000])
        checked.add((structure, str(seed), str(alpha)))
    return ex, f


def done(path, keys):
    """The rows already in a results file, as tuples of the given columns (to continue a run)."""
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {tuple(r[k] for k in keys) for r in csv.DictReader(fh)}


def open_csv(path, columns):
    new = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(columns)
    return fh, w


def track(W, lay, d_um, d_f, U, rho, n, seed):
    """One pass of n particles of diameter d_um; returns (row values from r_p on, eta)."""
    r_p, Pe, Stk, St, D = sc.particle_parameters(d_um, d_f, U, rho)
    _, _, St_m, h = sc.model(d_um, d_f, U, rho)
    t1 = time.perf_counter()
    res = pt.track_pass(W, lay, r_p, D, St_m, h, n, np.random.default_rng(seed))
    f = res["caught"].mean()
    eta = pt.efficiency_from_pass(f, lay)
    return [r_p, Pe, St, D, h, n, f, int(res["stuck"].sum()), eta], time.perf_counter() - t1


def run_design(group):
    """Experiment (d) for one group (hexagonal, random1 or random2): every cell, design and size."""
    structure = "hexagonal" if group == "hexagonal" else "random"
    which = RANDOM_SPLIT.get(group, range(N_RANDOM))
    path = os.path.join(OUT_DIR, f"design_{group}.csv")
    finished = done(path, ["design", "seed", "d_p_um"])
    fh, w = open_csv(path, DESIGN_COLUMNS)
    chk_path = os.path.join(OUT_DIR, f"interception_checks_design_{group}.csv")
    checked = done(chk_path, ["structure", "seed", "alpha"])
    fc, wc = open_csv(chk_path, CHECK_COLUMNS)
    for alpha in sorted({v[2] for v in DESIGNS.values()}):
        names = [nm for nm, v in DESIGNS.items() if v[2] == alpha]
        for seed, lay in design_cells(structure, alpha, which):
            todo = [(nm, d) for nm in names for d in DESIGN_SIZES_UM[nm] if (nm, str(seed), str(d)) not in finished]
            if not todo:
                continue
            t0 = time.perf_counter()
            W, k = build(lay)
            ex, f = interception_check(W, lay, structure, seed, wc, checked)
            fc.flush()
            print(f"{structure} alpha = {alpha} seed {seed}: field ready in {time.perf_counter() - t0:.0f}s, "
                  f"k/a^2 = {k:.4f}, interception check {f:.4f} vs exact {ex:.4f}", flush=True)
            for nm, d in todo:
                d_f, U, a_, rho = DESIGNS[nm]
                vals, sec = track(W, lay, d, d_f, U, rho, N_PARTICLES[structure],
                                  seed * 100 + SIZES_UM.index(d))
                w.writerow([nm, structure, seed, alpha, 0.0, len(lay["radii"]), lay["Lx"], d_f * 1e6, U, rho, d]
                           + vals + [k, sec])
                fh.flush()
                print(f"   {nm:10s} d_p = {d:5.3f} um: caught {vals[6]:.4f}, eta = {vals[8]:.4f} ({sec:.0f}s)",
                      flush=True)
    fh.close()
    fc.close()
    print("done:", path, flush=True)


def run_poly():
    """Experiment (e): N_POLY random cells for each fibre-size spread, the filter of Stage 1."""
    path = os.path.join(OUT_DIR, "poly.csv")
    finished = done(path, ["seed", "d_p_um"])
    fh, w = open_csv(path, DESIGN_COLUMNS)
    chk_path = os.path.join(OUT_DIR, "interception_checks_poly.csv")
    checked = done(chk_path, ["structure", "seed", "alpha"])
    fc, wc = open_csv(chk_path, CHECK_COLUMNS)
    d_f, U, alpha, rho = DESIGNS["base"]
    for cv in POLY_CVS:
        for r in range(N_POLY):
            seed = 800000 + 1000 * int(round(cv * 100)) + r
            todo = [d for d in POLY_SIZES_UM if (str(seed), str(d)) not in finished]
            if not todo:
                continue
            lay = geo.random_array(alpha, 16, np.random.default_rng(seed), cv=cv, min_gap=MIN_GAP)
            t0 = time.perf_counter()
            W, k = build(lay)
            ex, f = interception_check(W, lay, f"poly cv={cv}", seed, wc, checked)
            fc.flush()
            print(f"poly CV = {cv} seed {seed}: field ready in {time.perf_counter() - t0:.0f}s, k/a^2 = {k:.4f}, "
                  f"interception check {f:.4f} vs exact {ex:.4f}", flush=True)
            for d in todo:
                vals, sec = track(W, lay, d, d_f, U, rho, N_POLY_PARTICLES, seed * 100 + POLY_SIZES_UM.index(d))
                w.writerow(["base", "random", seed, alpha, cv, 16, lay["Lx"], d_f * 1e6, U, rho, d] + vals + [k, sec])
                fh.flush()
                print(f"   d_p = {d:5.3f} um: caught {vals[6]:.4f}, eta = {vals[8]:.4f} ({sec:.0f}s)", flush=True)
    fh.close()
    fc.close()
    print("done:", path, flush=True)


def run_sensitivity():
    """(1) blob width eps and (2) near-wall settings on the hexagonal array, with the SAME particles for
    every variant (common random numbers, so differences are not hidden by noise); (3) cell size:
    random cells of 32 fibres two cells long (aspect 2) and of 36 fibres (aspect 1)."""
    path = os.path.join(OUT_DIR, "sensitivity.csv")
    finished = done(path, ["case", "seed", "d_p_um"])
    fh, w = open_csv(path, SENS_COLUMNS)
    d_f, U, alpha, rho = DESIGNS["base"]
    hexl = geo.hexagonal_array(alpha)
    variants = [("eps=0.1", 0.1, 0.3, 64), ("eps=0.05", 0.05, 0.3, 64), ("eps=0.025", 0.025, 0.3, 64),
                ("r_out=1.2a", 0.05, 0.2, 64), ("r_out=1.4a", 0.05, 0.4, 64), ("modes=32", 0.05, 0.3, 32)]
    for case, eps, gap, modes in variants:
        todo = [d for d in SENS_SIZES_UM if (case, "0", str(d)) not in finished]
        if not todo:
            continue
        W, k = build(hexl, eps, gap, modes)
        for d in todo:
            vals, sec = track(W, hexl, d, d_f, U, rho, N_SENS_PARTICLES, 4242 + SENS_SIZES_UM.index(d))
            w.writerow([case, "hexagonal", 0, 1, 1.0, eps, gap, modes, d, vals[5], vals[6], vals[7], vals[8], k, sec])
            fh.flush()
            print(f"sensitivity {case:10s} d_p = {d:5.3f} um: eta = {vals[8]:.4f}, k/a^2 = {k:.4f}", flush=True)
    for case, n_f, aspect in [("32 fibres, 2 cells long", 32, 2.0), ("36 fibres, 1.5 x 1.5 cells", 36, 1.0)]:
        for r in range(N_BIG):
            seed = 900000 + 100 * n_f + r
            todo = [d for d in CELL_SIZES_UM if (case, str(seed), str(d)) not in finished]
            if not todo:
                continue
            lay = geo.random_array(alpha, n_f, np.random.default_rng(seed), min_gap=MIN_GAP, aspect=aspect)
            t0 = time.perf_counter()
            W, k = build(lay)
            for d in todo:
                # the first size keeps the seed used before; later sizes get their own
                vals, sec = track(W, lay, d, d_f, U, rho, N_POLY_PARTICLES, seed + 10 * CELL_SIZES_UM.index(d))
                w.writerow([case, "random", seed, n_f, aspect, EPS, 0.3, 64, d, vals[5], vals[6], vals[7],
                            vals[8], k, sec])
                fh.flush()
                print(f"sensitivity {case} seed {seed} d_p = {d} um: k/a^2 = {k:.4f}, eta = {vals[8]:.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
    fh.close()
    print("done:", path, flush=True)


GROUPS = {"hexagonal": lambda: run_design("hexagonal"), "random1": lambda: run_design("random1"),
          "random2": lambda: run_design("random2"), "poly": run_poly, "sensitivity": run_sensitivity}


def main():
    names = sys.argv[1:] or list(GROUPS)
    for nm in names:
        if nm not in GROUPS:
            raise ValueError(f"group must be one of {list(GROUPS)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for nm in names:
        GROUPS[nm]()


if __name__ == "__main__":
    main()
