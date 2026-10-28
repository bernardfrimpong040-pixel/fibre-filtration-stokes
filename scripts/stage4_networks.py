"""
stage4_networks.py
==================
Experiment (b): permeability of RANDOM 3D fibre networks.

Part 1 (2D inputs, a few minutes). Exact permeabilities of random 2D arrays from our reference
solvers: across the fibres (k_perp, flow along x and along y) and along them (k_par), at the
same solid fractions and minimum gap as the 3D networks, with 4 and with 16 fibres per cell.
They are the inputs of the orientation rule (Eq. 7.13) that the 3D results are compared with.
-> results/stage4/inputs_2d_alpha<alpha>.csv
Also a placement check: 100 networks of parallel fibres placed by the 3D code, solved exactly in 2D,
must look like the 2D layouts statistically. -> results/stage4/placement_check_alpha<alpha>.csv

Part 2 (3D networks, the long part). For each solid fraction alpha in {0.1, 0.15} we draw random
periodic networks of straight fibres (radius a = 1, gaps >= 0.6 a) of three kinds:
    aligned  4 fibres, all along x                 (exact answer known: a check on random geometry)
    layered  4 fibres, 2 along x and 2 along y      (crossed layers, like a nonwoven mat)
    cubic    3 fibres, one along each axis          (isotropic orientation tensor <n n> = I/3)
and compute k with our 3D surface method (Richardson with eps and 2 eps; eps = 0.3 at alpha = 0.1,
0.25 at alpha = 0.15, chosen from the 2D accuracy test in main.tex Section 7.5) and with the cheap
centreline model. The LU factorization is reused, so we get k for flow along x, y and z:
    aligned: k_z (across the fibres; compared with the exact 2D value of the same layout)
    layered: k_z (through the mat) and k_x, k_y (in the plane of the mat)
    cubic  : k_x, k_y, k_z (equal on average by symmetry)
-> results/stage4/networks_alpha<alpha>_<structure>.csv   (one row per network, appended as it goes; rerun to continue)

Run from the project folder:   python scripts/stage4_networks.py          (about 2-3 hours on one core)
To use two cores, run two copies at the same time, one per solid fraction:
                               python scripts/stage4_networks.py 0.1
                               python scripts/stage4_networks.py 0.15
(or split by structure, e.g. "python scripts/stage4_networks.py 0.15 cubic"; every
solid-fraction/structure pair has its own results file, so copies never write to the same file;
the short 2D part is repeated by each copy only if it is not finished yet, so start one copy first
and the second after "placement check done" appears).
Then:                          python scripts/stage4_summary.py           (statistics + figures, seconds)
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import geometry as geo                                    # noqa: E402
import geometry3d as g3                                   # noqa: E402
from solver3d import NetworkSolver3D                      # noqa: E402
from reference_bie import reference_k, reference_k_axial  # noqa: E402

ALPHAS = [0.1, 0.15]
EPS = {0.1: 0.3, 0.15: 0.25}          # fine blob width for each solid fraction (coarse = 2 x)
MIN_GAP = 0.6
KINDS = {"aligned": (4, 12), "layered": (4, 20), "cubic": (3, 20)}    # (fibres, realizations)
KIND_ID = {"aligned": 0, "layered": 1, "cubic": 2}
N2D = {4: 200, 16: 40}                # 2D inputs: fibres per cell -> number of layouts
OUT_DIR = os.path.join(ROOT, "results", "stage4")
# one file per solid fraction, so two copies of the script never write to the same file


def out_2d(alpha):
    return os.path.join(OUT_DIR, f"inputs_2d_alpha{alpha}.csv")


def out_3d(alpha, kind):
    return os.path.join(OUT_DIR, f"networks_alpha{alpha}_{kind}.csv")


def out_check(alpha):
    return os.path.join(OUT_DIR, f"placement_check_alpha{alpha}.csv")


COLS_2D = ["alpha", "n_fibres", "seed", "min_gap", "k_perp_x", "k_perp_y", "k_par", "seconds"]
COLUMNS = ["kind", "alpha", "seed", "n_fibres", "L", "min_gap", "eps",
           "kx_R", "ky_R", "kz_R", "kz_eps", "kz_2eps", "kx_centre", "ky_centre", "kz_centre",
           "kz_exact", "k_par_exact", "K_asymmetry", "N_blobs", "seconds"]


def seed_2d(alpha, n, r):
    return 500000 + 10000 * (n == 16) + 1000 * int(round(alpha * 100)) + r


def seed_3d(kind, alpha, r):
    """A fixed, documented seed for every network, so results can be reproduced exactly."""
    return 100000 * (KIND_ID[kind] + 1) + 1000 * int(round(alpha * 100)) + r


def done(path, key_cols):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {tuple(row[c] for c in key_cols) for row in csv.DictReader(fh)}


def part1(alpha):
    """Exact 2D permeabilities of random arrays: the inputs of the orientation rule."""
    path = out_2d(alpha)
    finished = done(path, ["alpha", "n_fibres", "seed"])
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(COLS_2D)
        for n, n_lay in N2D.items():
            for r in range(n_lay):
                seed = seed_2d(alpha, n, r)
                if (str(alpha), str(n), str(seed)) in finished:
                    continue
                t0 = time.perf_counter()
                lay = geo.random_array(alpha, n, np.random.default_rng(seed), cv=0.0, min_gap=MIN_GAP)
                kx = reference_k(lay, 64, U=(1.0, 0.0))
                ky = reference_k(lay, 64, U=(0.0, 1.0))
                kp = reference_k_axial(lay, 64)
                w.writerow([alpha, n, seed, geo.min_surface_gap(lay), kx, ky, kp, time.perf_counter() - t0])
                fh.flush()
        print(f"2D inputs done for alpha = {alpha}", flush=True)


N_CHECK = 100          # extra aligned networks for the placement check (geometry + exact 2D solve only)


def placement_check(alpha):
    """Are the 3D networks placed like the 2D layouts? 100 aligned networks are placed by the 3D
    code (random_network) and their cross-sections solved EXACTLY with the 2D reference (no 3D solve).
    Their k must match the 2D inputs statistically. (Seeds 900000 + 1000 alpha + r.)"""
    path = out_check(alpha)
    finished = done(path, ["alpha", "seed"])
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["alpha", "seed", "k_perp_1", "k_perp_2"])
        for r in range(N_CHECK):
            seed = 900000 + 1000 * int(round(alpha * 100)) + r
            if (str(alpha), str(seed)) in finished:
                continue
            net = g3.random_network(alpha, 4, np.random.default_rng(seed), kind="aligned", min_gap=MIN_GAP,
                                    balanced=True)
            L = net["L"]
            c = np.array([[f["point"][1] % L, f["point"][2] % L] for f in net["fibres"]])
            lay = dict(centres=c, radii=np.ones(len(c)), Lx=L, Ly=L)
            w.writerow([alpha, seed, reference_k(lay, 64, U=(1.0, 0.0)), reference_k(lay, 64, U=(0.0, 1.0))])
            fh.flush()
    print(f"placement check done for alpha = {alpha}", flush=True)


def solve_all_directions(net, method, eps):
    """k/a^2 along x, y and z, i.e. 1/(K^-1)_cc, from the permeability tensor (one factorization)."""
    s = NetworkSolver3D(net, method, eps)
    ks = [s.dimensionless_permeability(n) for n in np.eye(3)]
    return np.array(ks), s


def exact_aligned(net):
    """Aligned network (all fibres along x): its cross-section in the (y, z) plane is a random 2D
    array, so the exact k for flow along z (2D flow along the second axis) and along the fibres."""
    L = net["L"]
    c = np.array([[f["point"][1] % L, f["point"][2] % L] for f in net["fibres"]])
    lay = dict(centres=c, radii=np.array([f["radius"] for f in net["fibres"]]), Lx=L, Ly=L)
    return reference_k(lay, 128, U=(0.0, 1.0)), reference_k_axial(lay, 128)


def part2(alpha, kind):
    path = out_3d(alpha, kind)
    finished = done(path, ["kind", "alpha", "seed"])
    new_file = not os.path.exists(path)
    n_f, n_real = KINDS[kind]
    count = len(finished)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(COLUMNS)
        eps = EPS[alpha]
        for r in range(n_real):
            seed = seed_3d(kind, alpha, r)
            if (kind, str(alpha), str(seed)) in finished:
                continue
            t0 = time.perf_counter()
            rng = np.random.default_rng(seed)
            net = g3.random_network(alpha, n_f, rng, kind=kind, min_gap=MIN_GAP, balanced=True)
            k1, s1 = solve_all_directions(net, "surface", eps)
            k2, _ = solve_all_directions(net, "surface", 2 * eps)
            kR = (4 * k1 - k2) / 3
            kc, _ = solve_all_directions(net, "centreline", None)
            kz_ex, kpar_ex = exact_aligned(net) if kind == "aligned" else (np.nan, np.nan)
            dt = time.perf_counter() - t0
            w.writerow([kind, alpha, seed, n_f, net["L"], g3.min_surface_gap(net), eps,
                        kR[0], kR[1], kR[2], k1[2], k2[2], kc[0], kc[1], kc[2],
                        kz_ex, kpar_ex, np.max(np.abs(s1.K - s1.K.T)) / np.max(np.abs(s1.K)), s1.N, dt])
            fh.flush()
            count += 1
            extra = f" (exact {kz_ex:.4f}, error {100 * (kR[2] / kz_ex - 1):+.2f}%)" if kind == "aligned" else ""
            print(f"[{count}/{n_real}] {kind} alpha={alpha} seed={seed}: k_z/a^2 = {kR[2]:.4f}{extra} "
                  f"{dt:.0f}s", flush=True)
    print("done:", path)


def main():
    # optional arguments: solid fractions (numbers) and/or structures (words), e.g. "0.15 cubic"
    args = sys.argv[1:]
    alphas = [float(a) for a in args if a[0].isdigit()] or ALPHAS
    kinds = [k for k in args if not k[0].isdigit()] or list(KINDS)
    for a in alphas:
        if a not in EPS:
            raise ValueError(f"alpha must be one of {ALPHAS}")
    for k in kinds:
        if k not in KINDS:
            raise ValueError(f"structure must be one of {list(KINDS)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    for a in alphas:
        part1(a)
        placement_check(a)
        for k in kinds:
            part2(a, k)


if __name__ == "__main__":
    main()
