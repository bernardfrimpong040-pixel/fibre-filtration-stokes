"""
stage5_summary.py
=================
Statistics and figures for experiment (c), from results/stage5/capture_*.csv.

For every structure and particle size:
  single-fibre efficiency eta (regular arrays: one cell, interval from the binomial error of the
  caught fraction; random arrays: mean over cells with a Student-t 95% interval),
  penetration of the 0.5 mm filter  P = exp(-4 alpha eta L / (pi d_f))   (Eq. 8.6; eta = caught
  flux / (U d_f), the convention of the Kuwabara-based formulas),
  pressure drop  dp = mu U0 L / k  (k = exact permeability; for random cells the cells are layers
  in series, so their RESISTANCES 1/k are averaged),
  quality factor q_F = -ln P / dp,
and the most penetrating particle size (MPPS: minimum of eta, from a parabola in ln d_p through the
three sizes around the minimum). Compared with the classical formulas of Stage 1.

build() returns everything (the check script calls it); main() writes the files and the figure.
Outputs: results/stage5/capture_summary.csv, results/stage5/capture_summary.txt,
         paper/figures/stage5_capture.pdf
Run from the project folder:   python scripts/stage5_summary.py
"""

import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import classical_models as cm        # noqa: E402
import stage5_capture as sc          # noqa: E402

RES = os.path.join(ROOT, "results", "stage5")
OUT_CSV = os.path.join(RES, "capture_summary.csv")
OUT_TXT = os.path.join(RES, "capture_summary.txt")
FIG = os.path.join(ROOT, "paper", "figures", "stage5_capture.pdf")
ORDER = ["square", "hexagonal", "random", "random_poly"]
LABEL = {"square": "square array", "hexagonal": "hexagonal array", "random": "random, identical fibres",
         "random_poly": "random, CV = 0.5"}


def load():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "capture_*.csv"))):
        if p.endswith("capture_summary.csv"):
            continue
        with open(p, newline="") as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        raise FileNotFoundError("no capture results: run scripts/stage5_capture.py first")
    return rows


def penetration(eta):
    """P = exp(-4 alpha eta L / (pi d_f)) with eta = caught flux / (U d_f) (Eq. 8.6)."""
    return np.exp(-4 * sc.ALPHA * eta * sc.THICKNESS / (np.pi * sc.D_F))


def mpps(d, eta):
    """Minimum of eta(d_p): parabola in ln d through the smallest value and its two neighbours."""
    d, eta = np.asarray(d), np.asarray(eta)
    i = int(np.clip(np.argmin(eta), 1, len(d) - 2))
    x = np.log(d[i - 1:i + 2])
    c = np.polyfit(x, np.log(eta[i - 1:i + 2]), 2)
    if c[0] <= 0:
        return d[np.argmin(eta)], eta.min()
    xm = -c[1] / (2 * c[0])
    return float(np.exp(xm)), float(np.exp(np.polyval(c, xm)))


N_BOOT = 2000
BOOT_SEED = 2026          # fixed seed: the bootstrap intervals are reproducible


def cell_arrays(rows, st):
    """Per-cell arrays of one structure: sizes, eta, caught fraction f, particle numbers n
    (cells x sizes) and k/a^2 (cells)."""
    sel = [r for r in rows if r["structure"] == st]
    seeds = sorted({r["seed"] for r in sel})
    ds = [d for d in sc.SIZES_UM if any(float(r["d_p_um"]) == d for r in sel)]

    def grid(col):
        return np.array([[float(next(r[col] for r in sel if r["seed"] == s_ and float(r["d_p_um"]) == d))
                          for d in ds] for s_ in seeds])
    K = np.array([float(next(r["k_over_a2"] for r in sel if r["seed"] == s_)) for s_ in seeds])
    return np.array(ds), grid("eta"), grid("f_caught"), grid("n_particles"), K


def resample(cells_st, rng):
    """One bootstrap draw of a structure: random cells are resampled with replacement; a single cell
    (regular array) is redrawn from its binomial error: f* = Binomial(n, f) / n at every size."""
    ds, E, Fc, Nn, K = cells_st
    if len(K) > 1:
        idx = rng.integers(0, len(K), len(K))
        return E[idx].mean(axis=0), 1.0 / np.mean(1.0 / K[idx])
    f, n = Fc[0], Nn[0]
    fs = np.minimum(rng.binomial(n.astype(int), f) / n, 1 - 1e-12)
    c = E[0] / -np.log(1 - f)
    return c * -np.log(1 - fs), K[0]


def build():
    rows = load()
    lines, table, curves = [], [], {}
    a_m = sc.D_F / 2
    for st in ORDER:
        sel = [r for r in rows if r["structure"] == st]
        if not sel:
            continue
        seeds = sorted({r["seed"] for r in sel})
        k_cells = {r["seed"]: float(r["k_over_a2"]) for r in sel}
        k_eff = 1.0 / np.mean([1.0 / k_cells[s] for s in seeds])          # layers in series
        dp = cm.MU_AIR * sc.U0 * sc.THICKNESS / (k_eff * a_m**2)
        ds, em, ci = [], [], []
        for d in sc.SIZES_UM:
            rr = [r for r in sel if float(r["d_p_um"]) == d]
            if not rr:
                continue
            eta = np.array([float(r["eta"]) for r in rr])
            if len(rr) == 1:
                # one cell: eta = c (-ln(1 - f)), so its error is c se_f / (1 - f), with the binomial
                # error se_f = sqrt(f (1 - f) / n) of the caught fraction f
                f, n = float(rr[0]["f_caught"]), float(rr[0]["n_particles"])
                se_f = np.sqrt(f * (1 - f) / n)
                c = eta[0] / -np.log(1 - f)
                half = 1.96 * c * se_f / (1 - f)
            else:
                half = stats.t.ppf(0.975, len(eta) - 1) * eta.std(ddof=1) / np.sqrt(len(eta))
            ds.append(d)
            em.append(eta.mean())
            ci.append(half)
            P = penetration(eta.mean())
            table.append([st, d, len(rr), eta.mean(), half, P, dp, -np.log(P) / dp,
                          float(cm.eta_total(d * 1e-6, sc.D_F, sc.U0, sc.ALPHA))])
        d_m, eta_m = mpps(ds, em)
        P_m = penetration(eta_m)
        curves[st] = (np.array(ds), np.array(em), np.array(ci), d_m, eta_m, P_m, dp, k_eff)
        stuck = sum(int(r["n_stuck"]) for r in sel)
        lines.append(f"{LABEL[st]} ({len(seeds)} cell{'s' if len(seeds) > 1 else ''}): k/a^2 = {k_eff:.4f}, "
                     f"pressure drop {dp:.1f} Pa; MPPS {d_m:.3f} um, eta there {eta_m:.4f}, penetration {P_m:.3f}, "
                     f"q_F at the MPPS {-np.log(P_m) / dp:.2e} 1/Pa; particles left in the cell at the time limit: {stuck}")
    # classical baseline: the Stage 1 recipe (Hinds' penetration law, with 1/(1 - alpha)), and the same
    # formulas with the convention of this paper (Eq. 8.6), which the Kuwabara check supports
    d_c, P_c = cm.most_penetrating_size(sc.D_F, sc.U0, sc.ALPHA, sc.THICKNESS)
    k_c = float(cm.k_kuwabara(sc.ALPHA, a_m))
    dp_c = cm.pressure_drop(k_c, sc.U0, sc.THICKNESS)
    lines.append(f"classical (Stage 1, Kuwabara-based formulas): MPPS {d_c * 1e6:.3f} um, penetration {P_c:.3f}, "
                 f"pressure drop {dp_c:.1f} Pa, q_F {-np.log(P_c) / dp_c:.2e} 1/Pa")
    P_c2 = float(penetration(cm.eta_total(d_c, sc.D_F, sc.U0, sc.ALPHA)))
    lines.append(f"classical formulas with the convention of Eq. 8.6: MPPS {d_c * 1e6:.3f} um, penetration {P_c2:.3f}, "
                 f"pressure drop {dp_c:.1f} Pa, q_F {-np.log(P_c2) / dp_c:.2e} 1/Pa")
    # bootstrap: MPPS intervals and ratios between structures (cells resampled; a regular array's
    # single cell redrawn from its binomial error)
    rng = np.random.default_rng(BOOT_SEED)
    cells = {st: cell_arrays(rows, st) for st in curves}

    def at_mpps(em, k_eff, ds):
        dp = cm.MU_AIR * sc.U0 * sc.THICKNESS / (k_eff * a_m**2)
        d_m, eta_m = mpps(ds, em)
        return eta_m, dp, -np.log(penetration(eta_m)) / dp, d_m

    draws = {st: [at_mpps(*resample(cells[st], rng), cells[st][0]) for _ in range(N_BOOT)] for st in curves}
    for st in curves:
        lo, hi = np.percentile([b[3] for b in draws[st]], [2.5, 97.5])
        lines.append(f"{LABEL[st]}: MPPS 95% interval [{lo:.3f}, {hi:.3f}] um")

    def compare(a, b, name):
        A0 = at_mpps(curves[a][1], curves[a][7], curves[a][0])
        B0 = at_mpps(curves[b][1], curves[b][7], curves[b][0])
        boot = [[A[i] / B[i] for i in range(3)] for A, B in zip(draws[a], draws[b])]
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
        r = [A0[i] / B0[i] for i in range(3)]
        lines.append(f"{name}: eta at the MPPS {r[0]:.3f} [{lo[0]:.3f}, {hi[0]:.3f}], pressure drop {r[1]:.3f} "
                     f"[{lo[1]:.3f}, {hi[1]:.3f}], q_F at the MPPS {r[2]:.3f} [{lo[2]:.3f}, {hi[2]:.3f}]")
    if "hexagonal" in curves and "random" in curves:
        compare("random", "hexagonal", "random / hexagonal")
    if "random" in curves and "random_poly" in curves:
        compare("random_poly", "random", "random CV = 0.5 / random identical")
    for st in curves:
        ds, em = curves[st][0], curves[st][1]
        cl = np.array([float(cm.eta_total(d * 1e-6, sc.D_F, sc.U0, sc.ALPHA)) for d in ds])
        lines.append(f"{LABEL[st]}: eta / classical eta ranges {np.min(em / cl):.2f}-{np.max(em / cl):.2f} over "
                     f"{ds.min():g}-{ds.max():g} um")
    # mechanisms near the MPPS (hexagonal array)
    mp = os.path.join(RES, "mechanisms_hexagonal.csv")
    if os.path.exists(mp):
        M = np.loadtxt(mp, delimiter=",", skiprows=1, ndmin=2)
        hexl = sc.layouts("hexagonal")[0][1]
        c_hex = np.pi * 2.0 / (4 * sc.ALPHA * hexl["Lx"])          # eta = c (-ln(1 - f)), Eq. (8.6)
        n_hex = sc.N_PARTICLES["hexagonal"]

        def se(eta):                                              # binomial standard error of eta
            f = 1 - np.exp(-eta / c_hex)
            return c_hex * np.sqrt(f * (1 - f) / n_hex) / (1 - f)
        for m in M:
            # columns: d, diffusion alone, interception alone, both (no inertia), all, classical E_D, E_R, E_DR, total
            extra = m[3] - m[1] - m[2]
            se_x = np.sqrt(se(m[1]) ** 2 + se(m[2]) ** 2 + se(m[3]) ** 2)
            lines.append(f"hexagonal, d_p = {m[0]:g} um: diffusion alone {m[1]:.4f}, interception alone {m[2]:.4f}, "
                         f"sum {m[1] + m[2]:.4f}, both without inertia {m[3]:.4f}, extra from acting together "
                         f"{extra:.4f} +- {2 * se_x:.4f} (2 s.e.), with inertia {m[4]:.4f} (change {m[4] / m[3] - 1:+.3f}); "
                         f"classical E_D {m[5]:.4f}, E_R {m[6]:.4f}, E_DR {m[7]:.4f}, total {m[8]:.4f}")
    # the inlet line of every run: its clearance from the capture circles (negative would mean it cuts one)
    import particles2d as pt
    clear = min(pt.inlet_line(lay, sc.particle_parameters(d)[0])[1]
                for st in curves for _, lay in sc.layouts(st) for d in sc.SIZES_UM)
    lines.append(f"inlet lines of all runs: smallest clearance from a capture circle {clear:.3f} a_mean")
    # tracker checks on every cell (pure interception)
    chk = []
    for p in sorted(glob.glob(os.path.join(RES, "interception_checks_*.csv"))):
        with open(p, newline="") as fh:
            chk += list(csv.DictReader(fh))
    nf = {(r["structure"], r["seed"]): int(r["n_fibres"]) for r in rows}
    dev = [abs(float(c["f_tracker"]) - float(c["f_exact"])) * float(c["n_particles"]) / nf[(c["structure"], c["seed"])]
           for c in chk]
    lines.append(f"interception check on all {len(chk)} cells (10000 particles, r_p = 0.1a): largest difference "
                 f"from the exact answer {max(dev):.2f} particles per fibre")
    header = ["structure", "d_p_um", "n_cells", "eta", "eta_ci95", "penetration", "pressure_drop_Pa", "q_F",
              "eta_classical"]
    return table, lines, curves, header


def main():
    table, lines, curves, header = build()
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(table)
    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    dd = np.logspace(np.log10(0.02), 0, 200)
    for st, mk in zip(ORDER, ["o", "s", "^", "v"]):
        if st not in curves:
            continue
        ds, em, ci = curves[st][:3]
        h = ax[0].errorbar(ds, em, yerr=ci, fmt=mk + "-", ms=4, capsize=2, label=LABEL[st])
        ax[1].plot(ds, penetration(em), mk + "-", ms=4, color=h[0].get_color(), label=LABEL[st])
    ax[0].plot(dd, cm.eta_total(dd * 1e-6, sc.D_F, sc.U0, sc.ALPHA), "k--", lw=1, label="classical (Stage 1)")
    ax[1].plot(dd, cm.penetration(dd * 1e-6, sc.D_F, sc.U0, sc.ALPHA, sc.THICKNESS), "k--", lw=1,
               label="classical (Stage 1 recipe)")
    ax[1].plot(dd, penetration(cm.eta_total(dd * 1e-6, sc.D_F, sc.U0, sc.ALPHA)), "k:", lw=1,
               label="classical, convention of Eq. 8.6")
    ax[1].legend(fontsize=7)
    for a_ in ax:
        a_.set_xscale("log")
        a_.set_xlabel("particle diameter $d_p$ ($\\mu$m)")
        a_.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
        a_.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1"])
        a_.minorticks_off()
    ax[0].set_yscale("log")
    ax[0].set_ylabel("single-fibre efficiency $\\eta$")
    ax[0].set_title("(a) capture by one fibre (95% CI)", fontsize=9)
    ax[1].set_ylabel("penetration $P$ of the 0.5 mm filter")
    ax[1].set_title("(b) penetration ($d_f = 3\\,\\mu$m, $\\alpha = 0.1$, $U_0 = 0.1$ m/s)", fontsize=9)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)


if __name__ == "__main__":
    main()
