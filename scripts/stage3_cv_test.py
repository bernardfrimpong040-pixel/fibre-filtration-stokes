"""
stage3_cv_test.py
=================
Why experiment (a) uses 40 layouts per case (main.tex, Section 6).

1. A first analysis used only the first 10 layouts of each case (these are the seeds
   r = 0..9 of results/stage3/random_arrays.csv, so they are a subset of the deposited data).
   We recompute what that analysis showed: polydisperse/monodisperse ratios.
2. A control: fibres with almost identical radii (CV = 0.01) must give the same k as
   identical fibres (CV = 0). We compare 40 layouts of each at alpha = 0.1, using the exact
   reference solver (64 points per fibre). They should agree within the 95% intervals.

Output: results/stage3/cv_test.txt
Run from the project folder:   python scripts/stage3_cv_test.py     (about 10 minutes)
"""

import csv
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import geometry as geo                       # noqa: E402
from reference_bie import reference_k        # noqa: E402

OUT = os.path.join(ROOT, "results", "stage3", "cv_test.txt")
IN = os.path.join(ROOT, "results", "stage3", "random_arrays.csv")


def ci95(x):
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def main():
    lines = []
    # ---- 1. what 10 layouts per case would have shown ----
    with open(IN, newline="") as fh:
        rows = list(csv.DictReader(fh))
    def sample(alpha, cv, first=None):
        x = [float(r["k_rs_richardson"]) for r in rows
             if float(r["alpha"]) == alpha and float(r["cv"]) == cv and (first is None or int(r["seed"]) % 100 < first)]
        return np.array(x)
    for n in (10, 40):
        ratios = [sample(a, cv, n).mean() / sample(a, 0.0, n).mean() for a in (0.05, 0.1, 0.2, 0.3) for cv in (0.25, 0.5)]
        lines.append(f"first {n} layouts per case: k_poly/k_mono (mean radius) ranges {min(ratios):.3f}-{max(ratios):.3f}")

    # ---- 2. control: CV = 0 against CV = 0.01 ----
    res = {}
    for cv in (0.0, 0.01):
        k = np.array([reference_k(geo.random_array(0.1, 16, np.random.default_rng(5000 + r), cv=cv, min_gap=0.2), 64)
                      for r in range(40)])
        res[cv] = k
        print(f"CV = {cv}: done", flush=True)
    for n in (10, 40):
        a, b = res[0.0][:n], res[0.01][:n]
        lines.append(f"control at alpha = 0.1, first {n} layouts: CV = 0 gives k/a^2 = {a.mean():.3f} +- {ci95(a):.3f}, "
                     f"CV = 0.01 gives {b.mean():.3f} +- {ci95(b):.3f}; ratio {b.mean() / a.mean():.3f}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
