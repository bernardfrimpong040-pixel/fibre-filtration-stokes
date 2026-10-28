"""
stage4_summary.py
=================
Statistics and figures for experiment (b), from results/stage4/networks_alpha*.csv and
results/stage4/inputs_2d_alpha*.csv (written by stage4_networks.py).

For each solid fraction:
  * accuracy: 3D surface method against the exact answer for the aligned networks;
  * permeability of each structure (mean, 95% confidence interval of the mean, Student t):
        layered, through the mat (k_z) and in its plane (k_x, k_y pooled);
        cubic, all three directions pooled (equal by symmetry);
  * the ORIENTATION RULE (main.tex Eq. 7.13): average the flow RESISTANCE of a fibre over the
    fibre directions, the resistance across a fibre and along it being those of a random array of
    parallel fibres at the TOTAL solid fraction (the averaging behind Jackson & James's formula),
        1/k_U = (1 - c_U)/k_perp + c_U/k_par ,    c_U = <(n . U_hat)^2> ,
    with k_perp and k_par the exact permeabilities of random 2D arrays at that solid fraction. The ratio (3D result)/(rule) gets a bootstrap 95% interval (the 3D networks and
    the 2D layouts are both resampled);
  * the centreline model against the surface method, network by network;
  * comparison with Jackson-James (isotropic), Kuwabara (2D cell model), Davies (empirical)
    and the regular cubic lattice (from stage4_validation.py).

build() computes everything and returns it (the check script calls it to confirm that
networks_summary.txt is up to date); main() also writes the files and the figures.

Also draws the convergence figure of the lattice validation (from lattice_validation.csv,
written by stage4_validation.py).

Outputs: results/stage4/networks_summary.csv, results/stage4/networks_summary.txt,
         paper/figures/stage4_networks.pdf, paper/figures/stage4_geometry.pdf,
         paper/figures/stage4_convergence.pdf
Run from the project folder:   python scripts/stage4_summary.py
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
import geometry3d as g3              # noqa: E402

RES = os.path.join(ROOT, "results", "stage4")
OUT_CSV = os.path.join(RES, "networks_summary.csv")
OUT_TXT = os.path.join(RES, "networks_summary.txt")
FIG = os.path.join(ROOT, "paper", "figures", "stage4_networks.pdf")
FIG_GEO = os.path.join(ROOT, "paper", "figures", "stage4_geometry.pdf")
FIG_CONV = os.path.join(ROOT, "paper", "figures", "stage4_convergence.pdf")
N_BOOT = 4000
BOOT_SEED = 2026          # fixed seed: the bootstrap intervals are reproducible


def load(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(RES, pattern))):
        with open(path, newline="") as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        raise FileNotFoundError(f"no files {pattern} in {RES}: run scripts/stage4_networks.py first")
    out = {}
    for key in rows[0]:
        try:
            out[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            out[key] = np.array([r[key] for r in rows])
    return out


def mean_ci(x):
    """Mean, standard deviation and half-width of the 95% confidence interval (Student t)."""
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1)
    return m, sd, stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)


def rule(k_perp, k_par, c):
    """Orientation rule, Eq. (7.13): resistances of the fibre families add."""
    return 1.0 / ((1 - c) / k_perp + c / k_par)


def ratio_to_rule(x, kperp_s, kpar_s, c, rng):
    """mean(x) / rule(mean k_perp, mean k_par) with a bootstrap 95% interval. Each resample draws
    networks (x) and 2D layouts (k_perp, k_par together, as they come from the same layout)."""
    pred = rule(kperp_s.mean(), kpar_s.mean(), c)
    i3 = rng.integers(0, len(x), (N_BOOT, len(x)))
    i2 = rng.integers(0, len(kperp_s), (N_BOOT, len(kperp_s)))
    r = x[i3].mean(axis=1) / rule(kperp_s[i2].mean(axis=1), kpar_s[i2].mean(axis=1), c)
    return x.mean() / pred, np.percentile(r, 2.5), np.percentile(r, 97.5), pred


def build():
    """Compute every statistic. Returns (rows, lines, data, header)."""
    rng = np.random.default_rng(BOOT_SEED)
    d3 = load("networks_alpha*.csv")
    d2 = load("inputs_2d_alpha*.csv")
    dpc = load("placement_check_alpha*.csv")
    cubic3 = np.loadtxt(os.path.join(RES, "cubic3.csv"), delimiter=",", skiprows=1, ndmin=2)
    rows, lines, data = [], [], {}
    for alpha in sorted(set(d3["alpha"])):
        s2 = {n: (d2["alpha"] == alpha) & (d2["n_fibres"] == n) for n in (4, 16)}
        # k_perp: flow along x and along y are two samples of the same (isotropic) statistic;
        # average them per layout so each layout counts once
        kperp = {n: 0.5 * (d2["k_perp_x"][s] + d2["k_perp_y"][s]) for n, s in s2.items()}
        kpar = {n: d2["k_par"][s] for n, s in s2.items()}
        sel = {kind: (d3["alpha"] == alpha) & (d3["kind"] == kind) for kind in ("aligned", "layered", "cubic")}
        al = sel["aligned"]
        err = d3["kz_R"][al] / d3["kz_exact"][al] - 1
        lines.append(f"alpha = {alpha}: aligned networks ({al.sum()}): 3D surface method vs exact, "
                     f"largest error {100 * np.max(np.abs(err)):.2f}%, mean {100 * np.mean(np.abs(err)):.2f}%")
        for n in (4, 16):
            mp, _, cp = mean_ci(kperp[n])
            ma, _, ca = mean_ci(kpar[n])
            lines.append(f"  2D random arrays, {n:2d} fibres ({len(kperp[n])} layouts): k_perp = {mp:.4f} +- {cp:.4f}, "
                         f"k_par = {ma:.4f} +- {ca:.4f}, k_par/k_perp = {ma / mp:.2f}")
        mz, _, cz = mean_ci(d3["kz_R"][al])
        # placement check: 100 networks of parallel fibres placed by the 3D code, solved exactly in 2D
        pc = dpc["alpha"] == alpha
        kpc = 0.5 * (dpc["k_perp_1"][pc] + dpc["k_perp_2"][pc])
        mp, _, cp = mean_ci(kpc)
        bx = rng.choice(kpc, (N_BOOT, len(kpc))).mean(axis=1)
        by = rng.choice(kperp[4], (N_BOOT, len(kperp[4]))).mean(axis=1)
        lo, hi = np.percentile(bx / by, [2.5, 97.5])
        lines.append(f"  placement check ({len(kpc)} parallel-fibre networks placed by the 3D code, exact 2D k): "
                     f"k_perp = {mp:.4f} +- {cp:.4f}; ratio to the 2D layouts {mp / kperp[4].mean():.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append(f"  the {al.sum()} aligned 3D networks: k_z = {mz:.4f} +- {cz:.4f}")
        lay, cub = sel["layered"], sel["cubic"]
        cases = [
            ("layered, through the mat", d3["kz_R"][lay], 0.0, d3["kz_centre"][lay] / d3["kz_R"][lay] - 1),
            ("layered, in the plane", 0.5 * (d3["kx_R"][lay] + d3["ky_R"][lay]), 0.5,
             np.concatenate([d3["kx_centre"][lay] / d3["kx_R"][lay], d3["ky_centre"][lay] / d3["ky_R"][lay]]) - 1),
            ("cubic, any direction", (d3["kx_R"][cub] + d3["ky_R"][cub] + d3["kz_R"][cub]) / 3, 1 / 3,
             np.concatenate([d3[f"k{c}_centre"][cub] / d3[f"k{c}_R"][cub] for c in "xyz"]) - 1),
        ]
        for name, x, c, cl_err in cases:
            m, sd, ci = mean_ci(x)
            r4, lo4, hi4, p4 = ratio_to_rule(x, kperp[4], kpar[4], c, rng)
            r16, lo16, hi16, p16 = ratio_to_rule(x, kperp[16], kpar[16], c, rng)
            data[(alpha, name)] = (x, m, ci, p4, r4, lo4, hi4)
            rows.append([alpha, name, len(x), m, sd, ci, p4, r4, lo4, hi4, p16, r16, lo16, hi16,
                         np.mean(cl_err), np.max(np.abs(cl_err))])
            lines.append(f"  {name:25s} ({len(x)}): k/a^2 = {m:.4f} +- {ci:.4f} (sd/mean {sd / m:.2f}); "
                         f"rule (4-fibre 2D) {p4:.4f}: ratio {r4:.3f} [{lo4:.3f}, {hi4:.3f}]; "
                         f"rule (16-fibre 2D) {p16:.4f}: ratio {r16:.3f} [{lo16:.3f}, {hi16:.3f}]; "
                         f"centreline error mean {100 * np.mean(cl_err):+.1f}%, worst {100 * np.max(np.abs(cl_err)):.1f}%")
        # anisotropy of the layered mat, network by network
        aniso = 0.5 * (d3["kx_R"][lay] + d3["ky_R"][lay]) / d3["kz_R"][lay]
        ma, _, ca = mean_ci(aniso)
        lines.append(f"  layered anisotropy k_in-plane / k_through = {ma:.2f} +- {ca:.2f}; rule predicts "
                     f"{rule(kperp[4].mean(), kpar[4].mean(), 0.5) / kperp[4].mean():.2f}")
        # classical models and the regular cubic lattice
        k_cub = data[(alpha, "cubic, any direction")][1]
        lat = cubic3[np.isclose(cubic3[:, 0], alpha)]
        lat_txt = f"; regular cubic lattice {lat[0, 1]:.4f} (random/lattice {k_cub / lat[0, 1]:.3f})" if len(lat) else ""
        lines.append(f"  classical: Jackson-James {float(cm.k_jackson_james_3d(alpha, 1.0)):.4f} "
                     f"(cubic random / JJ = {k_cub / float(cm.k_jackson_james_3d(alpha, 1.0)):.3f}), "
                     f"Kuwabara {float(cm.k_kuwabara(alpha, 1.0)):.4f} "
                     f"(layered through / Kuwabara = {data[(alpha, 'layered, through the mat')][1] / float(cm.k_kuwabara(alpha, 1.0)):.3f}), "
                     f"Davies {float(cm.k_davies(alpha, 1.0)):.4f}" + lat_txt)
    lines.append(f"all {len(d3['kind'])} networks: permeability tensors symmetric to "
                 f"{np.max(d3['K_asymmetry']):.1e} (largest |K - K^T|/|K|)")
    header = ["alpha", "structure", "n_networks", "k_mean", "k_sd", "k_ci95", "rule_4", "ratio_4", "ratio_4_lo",
              "ratio_4_hi", "rule_16", "ratio_16", "ratio_16_lo", "ratio_16_hi", "centreline_err_mean",
              "centreline_err_max"]
    return rows, lines, data, header


def convergence_figure():
    """Error of k against blob width for the three lattices at alpha = 0.1 (solid: k(eps);
    open symbols, dotted: Richardson), from results/stage4/lattice_validation.csv."""
    R = np.loadtxt(os.path.join(RES, "lattice_validation.csv"), delimiter=",", skiprows=1, ndmin=2)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for kind_id, mk, lab in [(0, "o", "across (aligned)"), (1, "s", "along (axial)"), (2, "^", "tilted 45$^\\circ$")]:
        sub = R[(R[:, 0] == kind_id) & np.isclose(R[:, 1], 0.1)]
        es = np.concatenate([sub[:, 2], 2 * sub[:, 2]])          # fine and coarse widths
        errs = np.abs(np.concatenate([sub[:, 4], sub[:, 5]]))
        order = np.argsort(es)
        line, = ax.loglog(es[order], errs[order], mk + "-", label=lab)
        ax.loglog(sub[:, 2], np.maximum(np.abs(sub[:, 6]), 1e-6), mk + ":", mfc="none", color=line.get_color())
    ee = np.array([0.2, 0.6])
    ax.loglog(ee, 0.6 * ee**2, "k--", lw=0.8, label="slope 2")
    ax.set_xticks([0.2, 0.3, 0.4, 0.6])
    ax.set_xticklabels(["0.2", "0.3", "0.4", "0.6"])
    ax.minorticks_off()
    ax.set_xlabel(r"blob width $\varepsilon/a$")
    ax.set_ylabel(r"relative error in $k$")
    ax.set_title(r"3D lattices, $\alpha=0.1$ (open symbols: Richardson)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_CONV)
    plt.close(fig)


def geometry_figure():
    """Pictures of one layered and one cubic network (the first realization at alpha = 0.1)."""
    import stage4_networks as sn
    fig = plt.figure(figsize=(8, 3.8))
    for i, kind in enumerate(["layered", "cubic"]):
        rng = np.random.default_rng(sn.seed_3d(kind, 0.1, 0))
        net = g3.random_network(0.1, sn.KINDS[kind][0], rng, kind=kind, min_gap=sn.MIN_GAP, balanced=True)
        L = net["L"]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        for f, col in zip(net["fibres"], ["C0", "C1", "C2", "C3", "C4", "C5"]):
            t = g3.unit(f["dirn"])
            # draw the fibre and its periodic copies that cross the cell, as a tube of lines
            for sh in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).reshape(3, -1).T:
                p0 = np.asarray(f["point"]) % L + sh * L
                s = np.linspace(-2 * L, 2 * L, 400)
                pts = p0 + np.outer(s, t)
                inside = np.all((pts >= 0) & (pts <= L), axis=1)
                if inside.sum() > 2:
                    ax.plot(*pts[inside].T, color=col, lw=3.5, solid_capstyle="round")
        for a in "xyz":
            getattr(ax, f"set_{a}lim")(0, L)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"({'ab'[i]}) {kind} network, $\\alpha$ = 0.1, flow along $z$", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(FIG_GEO)
    plt.close(fig)


def main():
    rows, lines, data, header = build()
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figure: permeability of each structure and the orientation rule ----
    alphas = sorted({a for a, _ in data})
    names = ["layered, through the mat", "layered, in the plane", "cubic, any direction"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    al = np.linspace(0.08, 0.17, 50)
    for j, (name, mk) in enumerate(zip(names, ["o", "s", "^"])):
        m = [data[(a, name)][1] for a in alphas]
        ci = [data[(a, name)][2] for a in alphas]
        p = [data[(a, name)][3] for a in alphas]
        h = ax.errorbar(np.array(alphas) * (1 + 0.01 * j), m, yerr=ci, fmt=mk, ms=5, capsize=3, label=name)
        ax.plot(alphas, p, ls="--", lw=1, color=h[0].get_color())
    ax.plot(al, cm.k_jackson_james_3d(al, 1.0), color="0.4", lw=1, label="Jackson-James (isotropic)")
    ax.plot(al, cm.k_kuwabara(al, 1.0), color="0.6", lw=1, ls="-.", label="Kuwabara (2D cell)")
    ax.plot(al, cm.k_davies(al, 1.0), color="C3", lw=1, ls=":", label="Davies (empirical)")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ax.set_yticklabels(["0.5", "0.7", "1", "1.5", "2", "3"])
    ax.minorticks_off()
    ax.set_xlabel("solid fraction $\\alpha$")
    ax.set_ylabel("$k/a^{2}$")
    ax.set_title("(a) random 3D networks (dashed: orientation rule)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    for j, (name, mk) in enumerate(zip(names, ["o", "s", "^"])):
        r = np.array([data[(a, name)][4:7] for a in alphas])
        ax.errorbar(np.array(alphas) * (1 + 0.01 * j), r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]],
                    fmt=mk, ms=5, capsize=3, label=name)
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xlabel("solid fraction $\\alpha$")
    ax.set_ylabel("3D result / orientation rule")
    ax.set_title("(b) test of the orientation rule (bootstrap 95% CI)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG)
    plt.close(fig)
    geometry_figure()
    convergence_figure()


if __name__ == "__main__":
    main()
