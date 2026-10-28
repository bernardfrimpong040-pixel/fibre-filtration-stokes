"""
stage3_random_summary.py
========================
Statistics and figure for experiment (a), from results/stage3/random_arrays.csv.

For each (alpha, cv):
  mean of k / a_mean^2 and of k / a_rms^2 over realizations, standard deviation,
  95% confidence interval of the mean (Student t), and the worst RS-vs-reference error.
Compared with: the reference square and hexagonal arrays, Kuwabara, Happel, Davies.

Also tests Proposition 7 (main.tex): for polydisperse fibres, k scales with the
ROOT-MEAN-SQUARE radius, k / a_rms^2 = K_mono(alpha) to leading order, and its first-order
correction k_poly / k_mono(a_rms) = 1 + var(ln a) / H, with H = 4 alpha K_mono.

build() computes everything and returns it (the check script calls it to confirm that
random_summary.txt is up to date); main() also writes the files and the figure.

Outputs: results/stage3/random_summary.csv, results/stage3/random_summary.txt,
         paper/figures/stage3_random.pdf
Run from the project folder:   python scripts/stage3_random_summary.py
"""

import csv
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
import classical_models as cm        # noqa: E402
import geometry as geo               # noqa: E402
from reference_bie import reference_k   # noqa: E402

IN = os.path.join(ROOT, "results", "stage3", "random_arrays.csv")
OUT_CSV = os.path.join(ROOT, "results", "stage3", "random_summary.csv")
OUT_TXT = os.path.join(ROOT, "results", "stage3", "random_summary.txt")
FIG = os.path.join(ROOT, "paper", "figures", "stage3_random.pdf")


def load():
    with open(IN, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


N_BOOT = 4000
BOOT_SEED = 2026          # fixed seed: the bootstrap intervals are reproducible


def mean_ci(x):
    """Mean, standard deviation and half-width of the 95% confidence interval of the mean
    (Student t). Used for the error bars."""
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    return m, sd, stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)


def ratio_ci(x, y, rng):
    """Ratio of means mean(x)/mean(y) with a 95% bootstrap percentile interval.
    k varies strongly between layouts and its distribution is skewed, so we resample
    the layouts rather than assume a normal distribution."""
    bx = rng.choice(x, (N_BOOT, len(x))).mean(axis=1)
    by = rng.choice(y, (N_BOOT, len(y))).mean(axis=1)
    r = bx / by
    return x.mean() / y.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def var_log_radius(d, sel):
    """Mean over layouts of the variance of ln(a) among the 16 fibres. The CSV stores the
    seeds, so we rebuild each layout (placement only, no flow solve) to get its radii."""
    out = []
    for alpha, cv, seed, n in zip(d["alpha"][sel], d["cv"][sel], d["seed"][sel], d["n_fibres"][sel]):
        lay = geo.random_array(alpha, int(n), np.random.default_rng(int(seed)), cv=cv, min_gap=0.2)
        out.append(np.var(np.log(lay["radii"])))
    return float(np.mean(out))


def build():
    """Compute every statistic. Returns (rows, lines, samples, R)."""
    rng = np.random.default_rng(BOOT_SEED)
    d = load()
    alphas, cvs = sorted(set(d["alpha"])), sorted(set(d["cv"]))
    rows, lines = [], []
    table, samples = {}, {}
    for alpha in alphas:
        k_sq = reference_k(geo.square_array(alpha), 64)
        k_hex = reference_k(geo.hexagonal_array(alpha), 64)
        for cv in cvs:
            sel = (d["alpha"] == alpha) & (d["cv"] == cv)
            if not np.any(sel):
                continue
            k_mean_norm = d["k_rs_richardson"][sel]                     # k / a_mean^2
            k_rms_norm = k_mean_norm * (d["a_mean"][sel] / d["a_rms"][sel]) ** 2   # k / a_rms^2
            # third candidate: the "area-weighted" radius a_s = <a^2>/<a> = a_rms^2 / a_mean
            k_s_norm = k_mean_norm * (d["a_mean"][sel] / d["a_rms"][sel]) ** 4
            m1, sd1, ci1 = mean_ci(k_mean_norm)
            m2, sd2, ci2 = mean_ci(k_rms_norm)
            m3, _, ci3 = mean_ci(k_s_norm)
            err = np.abs(d["rs_rel_error"][sel]).max()
            table[(alpha, cv)] = (m1, ci1, m2, ci2, m3, ci3)
            samples[(alpha, cv)] = (k_mean_norm, k_rms_norm, k_s_norm,
                                    float(np.mean((d["a_rms"][sel] / d["a_mean"][sel]) ** 2)),
                                    var_log_radius(d, sel) if cv > 0 else 0.0)
            rows.append([alpha, cv, int(sel.sum()), m1, sd1, ci1, m2, sd2, ci2, k_sq, k_hex,
                         float(cm.k_kuwabara(alpha, 1.0)), float(cm.k_happel(alpha, 1.0)),
                         float(cm.k_davies(alpha, 1.0)), err])
    header = ["alpha", "cv", "n_real", "k_amean_mean", "k_amean_sd", "k_amean_ci95", "k_arms_mean", "k_arms_sd",
              "k_arms_ci95", "k_square", "k_hex", "k_kuwabara", "k_happel", "k_davies", "max_rs_error"]
    R = np.array(rows)

    # ---- the numbers quoted in main.tex ----
    mono = R[R[:, 1] == 0]
    for r in mono:
        x = samples[(r[0], 0.0)][0]
        rr, lo, hi = ratio_ci(x, np.array([r[9]]), rng)
        lines.append(f"alpha = {r[0]:.2f}: random monodisperse k/a^2 = {r[3]:.4f} +- {r[5]:.4f} (95% CI, n = {int(r[2])}), "
                     f"median {np.median(x):.4f}, sd/mean {r[4] / r[3]:.2f}; square {r[9]:.4f}, hexagonal {r[10]:.4f}, "
                     f"Kuwabara {r[11]:.4f}, Davies {r[13]:.4f}; random/square = {rr:.3f} [{lo:.3f}, {hi:.3f}]")
    all_err = np.abs(d["rs_rel_error"])
    lines.append(f"largest RS-vs-reference error over all {int(R[:, 2].sum())} realizations = {100 * R[:, 14].max():.2f}%; "
                 f"mean {100 * all_err.mean():.2f}%, 95th percentile {100 * np.percentile(all_err, 95):.2f}%")
    sd_poly = [r[4] / r[3] for r in R if r[1] > 0]
    lines.append(f"spread between layouts (sd/mean): monodisperse {min(r[4] / r[3] for r in mono):.2f}-"
                 f"{max(r[4] / r[3] for r in mono):.2f}, polydisperse {min(sd_poly):.2f}-{max(sd_poly):.2f}")
    # Proposition 7: polydisperse / monodisperse ratio of k/a_mean^2, with bootstrap intervals
    n_inside = n_total = 0
    for alpha in alphas:
        for cv in cvs[1:]:
            if (alpha, cv) not in samples or (alpha, 0.0) not in samples:
                continue
            rr, lo, hi = ratio_ci(samples[(alpha, cv)][0], samples[(alpha, 0.0)][0], rng)
            rr2, lo2, hi2 = ratio_ci(samples[(alpha, cv)][1], samples[(alpha, 0.0)][0], rng)
            pred = samples[(alpha, cv)][3]               # realised <a^2>/a_mean^2 (not exactly 1 + CV^2)
            H = 4 * alpha * samples[(alpha, 0.0)][0].mean()
            first = 1 + samples[(alpha, cv)][4] / H      # first-order correction, for the a_rms-scaled ratio
            inside = lo <= pred <= hi
            n_inside += inside
            n_total += 1
            lines.append(f"  alpha = {alpha:.2f}, CV = {cv:g}: k_poly/k_mono (mean radius) = {rr:.3f} [{lo:.3f}, {hi:.3f}], "
                         f"Prop. 7 predicts {pred:.3f} -> {'inside' if inside else 'OUTSIDE'}; "
                         f"(r.m.s. radius) = {rr2:.3f} [{lo2:.3f}, {hi2:.3f}]; H = {H:.2f}, "
                         f"first-order correction predicts {first:.2f}")
    lines.append(f"Proposition 7: prediction <a^2>/a_mean^2 inside the 95% interval in {n_inside} of {n_total} cases")
    return rows, lines, samples, R, header, rng


def main():
    rows, lines, samples, R, header, rng = build()
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    alphas = sorted(set(R[:, 0]))
    cvs = sorted(set(R[:, 1]))
    mono = R[R[:, 1] == 0]

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    al = np.linspace(0.04, 0.32, 80)
    ax = axes[0]
    for cv, mk, colr in zip(cvs, ["o", "s", "^"], ["C0", "C1", "C2"]):
        sub = R[R[:, 1] == cv]
        ax.errorbar(sub[:, 0] * (1 + 0.03 * cvs.index(cv)), sub[:, 3], yerr=sub[:, 5], fmt=mk,
                    color=colr, ms=5, capsize=3, label=f"random, CV = {cv:g}")
    ax.plot(mono[:, 0], mono[:, 9], "k--", lw=1, label="square array (exact)")
    ax.plot(mono[:, 0], mono[:, 10], "k-.", lw=1, label="hexagonal array (exact)")
    ax.plot(al, cm.k_kuwabara(al, 1.0), color="0.5", lw=1, label="Kuwabara")
    ax.plot(al, cm.k_davies(al, 1.0), color="C3", lw=1, ls=":", label="Davies (empirical, 3D filters)")
    ax.set_yscale("log")
    ax.set_xlabel("solid fraction $\\alpha$")
    ax.set_ylabel("$k/\\bar a^{2}$")
    ax.set_title("(a) permeability of random 2D arrays (mean $\\pm$ 95% CI)")
    ax.legend(frameon=False, fontsize=7.5)
    ax = axes[1]
    for cv, mk, colr in zip(cvs[1:], ["s", "^"], ["C1", "C2"]):
        pts = []
        for alpha in alphas:
            if (alpha, cv) in samples and (alpha, 0.0) in samples:
                pts.append((alpha,) + ratio_ci(samples[(alpha, cv)][0], samples[(alpha, 0.0)][0], rng))
        pts = np.array(pts)
        shift = 1 + 0.03 * cvs.index(cv)
        ax.errorbar(pts[:, 0] * shift, pts[:, 1], yerr=[pts[:, 1] - pts[:, 2], pts[:, 3] - pts[:, 1]],
                    fmt=mk, color=colr, ms=5, capsize=3, label=f"CV = {cv:g} (bootstrap 95% CI)")
        pred = np.mean([samples[(a_, cv)][3] for a_ in alphas])
        ax.axhline(pred, color=colr, ls="--", lw=1)
        ax.text(0.31, pred + 0.01, f"Prop. 7: $\\langle a^2\\rangle/\\bar a^2$ = {pred:.3f}", color=colr, fontsize=7, ha="right")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("solid fraction $\\alpha$")
    ax.set_ylabel("$k_{\\rm poly}/k_{\\rm mono}$ (both scaled by $\\bar a^2$)")
    ax.set_title("(b) effect of fibre-size spread vs Proposition 7")
    # error bars in panel (b) use a different bootstrap draw from the table; both are 95% intervals
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)
    print("Wrote", OUT_CSV, OUT_TXT, FIG)


if __name__ == "__main__":
    main()
