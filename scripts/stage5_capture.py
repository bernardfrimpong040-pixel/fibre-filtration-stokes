"""
stage5_capture.py
=================
Experiment (c): aerosol capture by 2D fibre arrays, from the flow to the filter.

The filter of Stage 1: fibre diameter d_f = 3 um, solid fraction alpha = 0.1, face velocity
U0 = 0.1 m/s, thickness 0.5 mm, NaCl particles of diameter 0.02-1 um, air at 20 C.
Structures (fibres across the flow):
    square      the regular square array             (1 cell)
    hexagonal   the regular hexagonal array          (1 cell)
    random      16 identical fibres, random          (6 cells)
    random_poly 16 fibres, radius spread CV = 0.5    (6 cells)
Random cells keep gaps >= 0.5 a_mean so that the near-wall reconstruction (Section 8.2) fits
around every fibre; their seeds are 700000 + 1000 CV x 100 + r.

For every cell: the RS flow (eps = 0.05 a_mean), the fast field with near-wall reconstruction,
the exact permeability (reference solver), and a pure-interception check of the tracker against the
exact stream-function answer. Then for every particle size, one pass of n particles (Brownian motion
interception and inertia together; see model()) gives the caught fraction f and
the single-fibre efficiency eta (Eq. 8.6).

Results: results/stage5/capture_<structure>.csv (one row per cell and size; appended as it goes,
rerun to continue), results/stage5/interception_checks_<structure>.csv, and (with the hexagonal
array) results/stage5/mechanisms_hexagonal.csv: each mechanism alone and all together, near the MPPS.
Run from the project folder:   python scripts/stage5_capture.py          (about 9 hours on one core; see README for running it in two windows)
Two cores:                     python scripts/stage5_capture.py square hexagonal random
                               python scripts/stage5_capture.py random_poly
Then:                          python scripts/stage5_summary.py
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import geometry as geo                                           # noqa: E402
import classical_models as cm                                    # noqa: E402
import particles2d as pt                                         # noqa: E402
from solver2d import solve_with_rule                             # noqa: E402
from flowfield2d import StreamFunctionField2D, NearWallField2D   # noqa: E402
from reference_bie import reference_k                            # noqa: E402

# ---- the filter of Stage 1 ----
D_F = 3.0e-6            # fibre diameter (m)
ALPHA = 0.1
U0 = 0.1                # face velocity (m/s)
THICKNESS = 0.5e-3      # m
SIZES_UM = [0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 1.0]

EPS = 0.05
MIN_GAP = 0.5
N_RANDOM = 6
STRUCTURES = {"square": None, "hexagonal": None, "random": 0.0, "random_poly": 0.5}
N_PARTICLES = {"square": 40000, "hexagonal": 40000, "random": 4000, "random_poly": 4000}
OUT_DIR = os.path.join(ROOT, "results", "stage5")
COLUMNS = ["structure", "seed", "alpha", "cv", "n_fibres", "Lx", "d_p_um", "r_p", "Pe", "Stk", "St", "St_model",
           "D", "h", "n_particles", "f_caught", "n_stuck", "eta", "k_over_a2", "seconds"]


def layouts(structure):
    """(seed, layout) pairs of a structure. Lengths in units of the mean fibre radius."""
    if structure == "square":
        return [(0, geo.square_array(ALPHA))]
    if structure == "hexagonal":
        return [(0, geo.hexagonal_array(ALPHA))]
    cv = STRUCTURES[structure]
    out = []
    for r in range(N_RANDOM):
        seed = 700000 + 1000 * int(round(cv * 100)) + r
        out.append((seed, geo.random_array(ALPHA, 16, np.random.default_rng(seed), cv=cv, min_gap=MIN_GAP)))
    return out


def particle_parameters(d_p_um):
    """Dimensionless parameters (lengths in a_mean = d_f/2, speeds in U0, time in a/U0)."""
    d_p = d_p_um * 1e-6
    Pe = float(cm.peclet(d_p, D_F, U0))
    Stk = float(cm.stokes_number(d_p, D_F, U0))
    r_p = d_p / D_F                    # particle radius / fibre radius
    D = 2.0 / Pe                       # D_p / (a U0)
    St = 2.0 * Stk                     # tau U0 / a
    return r_p, Pe, Stk, St, D


def time_step(D):
    """h = min(0.04, 0.005 delta^2 / D) with the diffusion boundary-layer thickness
    delta = (D / gamma)^(1/3), gamma ~ 4 U/a: the Brownian step sqrt(2 D h) stays below delta/10."""
    return min(0.04, 0.005 * (D / 4.0) ** (2.0 / 3.0) / D)


def model(d_p_um):
    """The particle model used for one size (main.tex Section 8.3): Brownian motion, interception AND
    inertia for every size (the full Langevin equation, Eq. 8.1). Inertia matters even for small
    Stokes numbers: air curving round a fibre throws particles outward, which lowers capture by about
    20% near the MPPS (measured in Section 8.5). The step h is the diffusion step of time_step(D),
    and at most St/8 so that the particle's velocity relaxation is resolved.
    Returns r_p, D, the St used and the step h."""
    r_p, Pe, Stk, St, D = particle_parameters(d_p_um)
    return r_p, D, St, min(time_step(D), St / 8.0)


def done(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {(r["seed"], r["d_p_um"]) for r in csv.DictReader(fh)}


def run(structure):
    path = os.path.join(OUT_DIR, f"capture_{structure}.csv")
    chk_path = os.path.join(OUT_DIR, f"interception_checks_{structure}.csv")
    finished = done(path)
    new_file = not os.path.exists(path)
    n = N_PARTICLES[structure]
    with open(path, "a", newline="") as fh, open(chk_path, "a", newline="") as fc:
        w, wc = csv.writer(fh), csv.writer(fc)
        if new_file:
            w.writerow(COLUMNS)
        if os.path.getsize(chk_path) == 0:
            wc.writerow(["structure", "seed", "r_p", "f_exact", "f_tracker", "n_particles"])
        for seed, lay in layouts(structure):
            todo = [d for d in SIZES_UM if (str(seed), str(d)) not in finished]
            if not todo:
                continue
            t0 = time.perf_counter()
            s = solve_with_rule(lay, EPS)
            W = NearWallField2D(StreamFunctionField2D(s, grid_factor=4))
            k = reference_k(lay, 64)
            # tracker check on this cell: pure interception, r_p = 0.1a
            ex = pt.interception_exact(W, lay, 0.1)
            chk = pt.track_pass(W, lay, 0.1, 0.0, 0.0, 0.04, 10000, np.random.default_rng(seed + 1))
            wc.writerow([structure, seed, 0.1, ex, chk["caught"].mean(), 10000])
            fc.flush()
            print(f"{structure} seed {seed}: field ready in {time.perf_counter() - t0:.0f}s, k/a^2 = {k:.4f}, "
                  f"interception check {chk['caught'].mean():.4f} vs exact {ex:.4f}", flush=True)
            for d in todo:
                t1 = time.perf_counter()
                r_p, Pe, Stk, St, D = particle_parameters(d)
                _, _, St_m, h = model(d)
                rng = np.random.default_rng(seed * 100 + SIZES_UM.index(d))
                res = pt.track_pass(W, lay, r_p, D, St_m, h, n, rng)
                f = res["caught"].mean()
                eta = pt.efficiency_from_pass(f, lay)
                a = lay["radii"]
                w.writerow([structure, seed, ALPHA, STRUCTURES[structure] or 0.0, len(a), lay["Lx"], d, r_p, Pe, Stk,
                            St, St_m, D, h, n, f, int(res["stuck"].sum()), eta, k, time.perf_counter() - t1])
                fh.flush()
                print(f"   d_p = {d:4.2f} um: caught {f:.4f}, eta = {eta:.4f} ({time.perf_counter() - t1:.0f}s)",
                      flush=True)
    print("done:", path)


MECH_SIZES_UM = [0.12, 0.25, 0.5]


def mechanisms():
    """Which mechanism catches what? On the hexagonal array, for three sizes around the MPPS:
    diffusion alone (point particles, no inertia), interception alone (no diffusion, no inertia),
    diffusion and interception together without inertia, and all three together (the model used
    in experiment c), with the classical formulas for each (Stage 1).
    -> results/stage5/mechanisms_hexagonal.csv"""
    path = os.path.join(OUT_DIR, "mechanisms_hexagonal.csv")
    if os.path.exists(path):
        return
    lay = geo.hexagonal_array(ALPHA)
    W = NearWallField2D(StreamFunctionField2D(solve_with_rule(lay, EPS), grid_factor=4))
    n = N_PARTICLES["hexagonal"]
    rows = []
    for d in MECH_SIZES_UM:
        r_p, D, St_m, h = model(d)
        h0 = time_step(D)                       # step for the runs without inertia
        out = []
        for rp_, D_, St_, h_ in [(0.0, D, 0.0, h0), (r_p, 0.0, 0.0, h0), (r_p, D, 0.0, h0), (r_p, D, St_m, h)]:
            res = pt.track_pass(W, lay, rp_, D_, St_, h_, n, np.random.default_rng(int(d * 1000)))
            out.append(pt.efficiency_from_pass(res["caught"].mean(), lay))
        dp = d * 1e-6
        rows.append([d] + out + [float(cm.eta_diffusion(dp, D_F, U0, ALPHA)), float(cm.eta_interception(dp, D_F, ALPHA)),
                                 float(cm.eta_diffusion_interception(dp, D_F, U0, ALPHA)),
                                 float(cm.eta_total(dp, D_F, U0, ALPHA))])
        print(f"mechanisms at d_p = {d} um: diffusion {out[0]:.4f}, interception {out[1]:.4f}, "
              f"both without inertia {out[2]:.4f}, all {out[3]:.4f}", flush=True)
    np.savetxt(path, np.array(rows), delimiter=",", comments="", fmt="%.10e",
               header="d_p_um,eta_diffusion_only,eta_interception_only,eta_diffusion_interception,eta_all,"
                      "classical_E_D,classical_E_R,classical_E_DR,classical_total")


def main():
    names = sys.argv[1:] or list(STRUCTURES)
    for nm in names:
        if nm not in STRUCTURES:
            raise ValueError(f"structure must be one of {list(STRUCTURES)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for nm in names:
        run(nm)
    if "hexagonal" in names:
        mechanisms()


if __name__ == "__main__":
    main()
