"""
stage3_random_arrays.py
=======================
Experiment (a): permeability of RANDOM 2D fibre arrays, mono- and polydisperse.

For every combination of
    solid fraction  alpha in {0.05, 0.1, 0.2, 0.3}
    radius spread   cv    in {0, 0.25, 0.5}      (cv = std/mean of fibre radius, lognormal)
we draw 40 random layouts of 16 fibres (gaps >= 0.2 a_mean) and compute k with
  * our regularized Stokeslet solver (eps = 0.1 a_mean, Richardson with 0.2 a_mean), and
  * the independent reference solver (spectral boundary integral),
so every realization carries its own accuracy check.

Results are appended to results/stage3/random_arrays.csv one realization at a time.
If the script is stopped, running it again continues where it stopped.

Why 40: k varies by about 25% from one 16-fibre layout to the next, and a first run with
10 layouts per case was too noisy to separate the effect of polydispersity from chance
(main.tex, Section 6).

Run from the project folder:   python scripts/stage3_random_arrays.py     (about 80 minutes on one core)
To use two cores, run two copies at the same time, each on some solid fractions:
                               python scripts/stage3_random_arrays.py 0.05 0.3
                               python scripts/stage3_random_arrays.py 0.1 0.2
Then:                          python scripts/stage3_random_summary.py    (statistics + figure, seconds)
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import geometry as geo                       # noqa: E402
from solver2d import richardson_k            # noqa: E402
from reference_bie import reference_k        # noqa: E402

ALPHAS = [0.05, 0.1, 0.2, 0.3]
CVS = [0.0, 0.25, 0.5]
N_REAL = 40
N_FIBRES = 16
MIN_GAP = 0.2
EPS = 0.1
OUT = os.path.join(ROOT, "results", "stage3", "random_arrays.csv")
COLUMNS = ["alpha", "cv", "seed", "n_fibres", "a_mean", "a_rms", "a_min", "a_max", "min_gap",
           "k_rs_richardson", "k_rs_eps", "k_reference", "rs_rel_error", "seconds"]


def seed_for(alpha, cv, r):
    """A fixed, documented seed for every realization, so results can be reproduced exactly."""
    return 10000 * int(round(alpha * 100)) + 100 * int(round(cv * 100)) + r


def done_keys():
    if not os.path.exists(OUT):
        return set()
    with open(OUT, newline="") as fh:
        return {(float(r["alpha"]), float(r["cv"]), int(r["seed"])) for r in csv.DictReader(fh)}


def main():
    # optional command-line list of solid fractions to run (default: all)
    alphas = [float(a) for a in sys.argv[1:]] or ALPHAS
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    finished = done_keys()
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(COLUMNS)
        total = len(alphas) * len(CVS) * N_REAL
        count = sum(1 for key in finished if key[0] in alphas)
        for alpha in alphas:
            for cv in CVS:
                for r in range(N_REAL):
                    seed = seed_for(alpha, cv, r)
                    if (alpha, cv, seed) in finished:
                        continue
                    t0 = time.perf_counter()
                    rng = np.random.default_rng(seed)
                    lay = geo.random_array(alpha, N_FIBRES, rng, cv=cv, min_gap=MIN_GAP)
                    a = lay["radii"]
                    kR, k1, _ = richardson_k(lay, EPS)            # k / a_mean^2
                    k_ref = reference_k(lay, 64)
                    dt = time.perf_counter() - t0
                    w.writerow([alpha, cv, seed, N_FIBRES, a.mean(), np.sqrt(np.mean(a**2)), a.min(), a.max(),
                                geo.min_surface_gap(lay), kR, k1, k_ref, kR / k_ref - 1, dt])
                    fh.flush()
                    count += 1
                    print(f"[{count}/{total}] alpha={alpha} cv={cv} seed={seed}: k/a^2 = {kR:.5f} "
                          f"(reference {k_ref:.5f}, error {100 * (kR / k_ref - 1):+.2f}%) {dt:.1f}s", flush=True)
    print("done:", OUT)


if __name__ == "__main__":
    main()
