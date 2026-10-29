"""
stage6_summary.py
=================
Statistics and figures for Stage 6, from results/stage6/*.csv (written by stage6_design.py).

Quality factor. For a filter of thickness L, P = exp(-4 alpha eta L / (pi d_f)) (Eq. 8.6) and
dp = mu U L / k, so q_F = -ln P / dp = alpha eta k d_f / (pi mu U): the thickness drops out, and at a
fixed design the worst particle size (the MPPS) is simply the size where eta is smallest.

1. Experiment (d): for every design and structure, the MPPS (parabola in ln d_p through the smallest
   eta and its two neighbours, as in Stage 5), eta there, pressure drop of a 0.5 mm filter, P and q_F
   at the MPPS, with the classical formulas of Stage 1; the ratio random / hexagonal; and the
   elasticities  E_X = ln(q_F(X_high) / q_F(X_low)) / ln(X_high / X_low)  for X = d_f, U, alpha and
   the particle density.
2. Experiment (e): q_F at the MPPS for fibre-size spread CV = 0, 0.25, 0.5 (16 cells each), as ratios to CV = 0.
3. Layered filters (Proposition 10): two layers in series, a share w of the pressure drop in layer 1,
   have q(d) = w q_1(d) + (1 - w) q_2(d) at every size d, and worst-case Q(w) = min_d q(d).
4. Sensitivity: blob width, near-wall settings, cell size.

Intervals: 95% bootstrap (N_BOOT draws, fixed seed): random cells resampled with replacement (the
same draw for all designs that share the cells), a single regular cell redrawn from its binomial error.

build() returns everything (the check script calls it); main() writes the files and figures.
Outputs: results/stage6/design_summary.csv, results/stage6/summary.txt,
         paper/figures/stage6_design.pdf, paper/figures/stage6_layers.pdf
Run from the project folder:   python scripts/stage6_summary.py
"""

import csv
import itertools
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import classical_models as cm        # noqa: E402
import stage6_design as d6           # noqa: E402

RES = os.path.join(ROOT, "results", "stage6")
OUT_CSV = os.path.join(RES, "design_summary.csv")
OUT_TXT = os.path.join(RES, "summary.txt")
FIG_D = os.path.join(ROOT, "paper", "figures", "stage6_design.pdf")
FIG_L = os.path.join(ROOT, "paper", "figures", "stage6_layers.pdf")
THICKNESS = 0.5e-3
N_BOOT = 2000
BOOT_SEED = 2026
STRUCTS = ["hexagonal", "random"]
LABEL = {"hexagonal": "hexagonal array", "random": "random, identical fibres"}
COLOR = {"hexagonal": "#2a78d6", "random": "#eb6834", "classical": "#6b6b66"}
MARK = {"hexagonal": "s", "random": "o", "classical": "^"}
# elasticities: (name, design with the low value, design with the high value, low value, high value)
ELASTIC = [("fibre diameter d_f", "d_f=1.5", "d_f=6", 1.5, 6.0),
           ("face velocity U", "U=0.05", "U=0.2", 0.05, 0.2),
           ("solid fraction alpha", "alpha=0.05", "alpha=0.15", 0.05, 0.15),
           ("particle density", "DEHS", "base", d6.RHO_DEHS, cm.RHO_NACL)]
# layer types for Proposition 10: same face velocity and aerosol, so they can be stacked
LAYER_DESIGNS = ["base", "d_f=1.5", "d_f=6", "alpha=0.05", "alpha=0.15"]
W_GRID = np.linspace(0.0, 1.0, 101)


def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run scripts/stage6_design.py first")
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def curve_min(d, y):
    """Smallest value of a positive curve y(d) on a size grid: a parabola in (ln d, ln y) through the
    smallest point and its two neighbours, as for the MPPS in Stage 5. If the smallest point is at an
    end of the grid the end point is returned and edge = True (the true minimum may lie outside)."""
    d, y = np.asarray(d, float), np.asarray(y, float)
    i = int(np.argmin(y))
    if i == 0 or i == len(d) - 1:
        return float(d[i]), float(y[i]), True
    x = np.log(d[i - 1:i + 2])
    c = np.polyfit(x, np.log(y[i - 1:i + 2]), 2)
    if c[0] <= 0:
        return float(d[i]), float(y[i]), False
    xm = -c[1] / (2 * c[0])
    return float(np.exp(xm)), float(np.exp(np.polyval(c, xm))), False


def q_factor(eta, k_over_a2, d_f, U, alpha):
    """q_F = -ln P / dp = alpha eta k d_f / (pi mu U), with k = (k/a^2) (d_f/2)^2 (1/Pa)."""
    return alpha * eta * k_over_a2 * (d_f / 2) ** 2 * 4 / (np.pi * d_f * cm.MU_AIR * U)


def cellset(rows, sizes):
    """Arrays of one set of cells: eta, caught fraction f and particle numbers n (cells x sizes), and k/a^2."""
    seeds = sorted({r["seed"] for r in rows})
    look = {(r["seed"], float(r["d_p_um"])): r for r in rows}
    E = np.array([[float(look[(s, d)]["eta"]) for d in sizes] for s in seeds])
    F = np.array([[float(look[(s, d)]["f_caught"]) for d in sizes] for s in seeds])
    N = np.array([[float(look[(s, d)]["n_particles"]) for d in sizes] for s in seeds])
    K = np.array([float(look[(s, sizes[0])]["k_over_a2"]) for s in seeds])
    return dict(seeds=seeds, E=E, F=F, N=N, K=K)


def draw(cs, rng, idx=None):
    """One bootstrap draw: (mean eta over cells at each size, k_eff). Random cells: resampled by idx
    (the same idx for every design that shares the cells); one cell: binomial redraw of f at every size."""
    E, K = cs["E"], cs["K"]
    if len(K) > 1:
        return E[idx].mean(axis=0), 1.0 / np.mean(1.0 / K[idx])
    f, n = cs["F"][0], cs["N"][0]
    fs = np.minimum(rng.binomial(n.astype(int), f) / n, 1 - 1e-12)
    return E[0] / -np.log(1 - f) * -np.log(1 - fs), K[0]


def point(cs):
    """Point estimate: (mean eta at each size, k_eff = harmonic mean of k: cells are layers in series)."""
    return cs["E"].mean(axis=0), 1.0 / np.mean(1.0 / cs["K"])


def ci(x):
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def classical(design):
    """Classical Stage 1 formulas for one design: MPPS, P there (Hinds), q_F (Hinds and Eq. 8.6), k/a^2."""
    d_f, U, alpha, rho = d6.DESIGNS[design]
    dm, P = cm.most_penetrating_size(d_f, U, alpha, THICKNESS, rho_p=rho)
    k_a2 = cm.kuwabara_factor(alpha) / (4 * alpha)              # Kuwabara: k = a^2 Ku / (4 alpha)
    dp = cm.pressure_drop(k_a2 * (d_f / 2) ** 2, U, THICKNESS)
    q_h = -np.log(P) / dp
    return dict(d=dm * 1e6, P=P, dp=dp, q_hinds=q_h, q=q_h * (1 - alpha), k=k_a2)


def classical_q_curve(design, sizes):
    """Classical q(d) in the convention of Eq. 8.6 on a size grid (for the layered filters)."""
    d_f, U, alpha, rho = d6.DESIGNS[design]
    eta = cm.eta_total(np.asarray(sizes) * 1e-6, d_f, U, alpha, rho)
    return q_factor(eta, cm.kuwabara_factor(alpha) / (4 * alpha), d_f, U, alpha)


def fine(sizes, q, n=400):
    """A curve q(d) on a fine size grid: monotone cubic (PCHIP) interpolation of ln q against ln d,
    which never overshoots the computed points. Returns (fine sizes, q on them)."""
    x = np.log(np.asarray(sizes, float))
    xf = np.linspace(x[0], x[-1], n)
    return np.exp(xf), np.exp(PchipInterpolator(x, np.log(q))(xf))


def stack_Q(q1, q2, sizes):
    """Proposition 10: worst-case q_F of two layers in series for every share w of the pressure drop
    in layer 1 (W_GRID): Q(w) = min_d [w q1(d) + (1 - w) q2(d)]. Both curves are first put on the same
    fine size grid, so Q(w) is exactly a minimum of straight lines in w (a concave function)."""
    _, f1 = fine(sizes, q1)
    _, f2 = fine(sizes, q2)
    return np.array([np.min(w * f1 + (1 - w) * f2) for w in W_GRID])


def build():
    lines = []
    rows = {"hexagonal": load("design_hexagonal.csv"),
            "random": load("design_random1.csv") + load("design_random2.csv")}
    sz = d6.DESIGN_SIZES_UM                     # the size window of each design
    sets = {(st, nm): cellset([r for r in rows[st] if r["design"] == nm], sz[nm])
            for st in STRUCTS for nm in d6.DESIGNS}
    # ---- interception checks and stuck particles ----
    chk = load("interception_checks_design_hexagonal.csv") + load("interception_checks_design_random1.csv") \
        + load("interception_checks_design_random2.csv") + load("interception_checks_poly.csv")
    nf = [int(c["n_fibres"]) for c in chk]
    dev = [abs(float(c["f_tracker"]) - float(c["f_exact"])) * float(c["n_particles"]) / n for c, n in zip(chk, nf)]
    all_rows = rows["hexagonal"] + rows["random"] + load("poly.csv") + load("sensitivity.csv")
    stuck = sum(int(r["n_stuck"]) for r in all_rows)
    tracked = sum(int(r["n_particles"]) for r in all_rows)
    lines.append(f"interception check on all {len(chk)} new cells (10000 particles, r_p = 0.1a): largest difference "
                 f"from the exact answer {max(dev):.2f} particles per fibre; {stuck} of {tracked} particles left in "
                 f"a cell at the time limit")
    # ---- 1. experiment (d): point estimates ----
    table, est = [], {}
    for nm, (d_f, U, alpha, rho) in d6.DESIGNS.items():
        cl = classical(nm)
        for st in STRUCTS:
            em, k = point(sets[(st, nm)])
            dmp, eta_m, edge = curve_min(sz[nm], em)
            dp = cm.pressure_drop(k * (d_f / 2) ** 2, U, THICKNESS)
            q = q_factor(eta_m, k, d_f, U, alpha)
            est[(st, nm)] = q
            table.append(dict(design=nm, structure=st, d_f_um=d_f * 1e6, U=U, alpha=alpha, rho_p=rho, k_over_a2=k,
                              mpps_um=dmp, mpps_at_grid_edge=int(edge), eta_mpps=eta_m, dp_Pa=dp,
                              P_mpps=np.exp(-q * dp), qF=q))
        est[("classical", nm)] = cl["q"]
        table.append(dict(design=nm, structure="classical (Eq. 8.6)", d_f_um=d_f * 1e6, U=U, alpha=alpha, rho_p=rho,
                          k_over_a2=cl["k"], mpps_um=cl["d"], mpps_at_grid_edge=0, eta_mpps=np.nan, dp_Pa=cl["dp"],
                          P_mpps=np.exp(-cl["q"] * cl["dp"]), qF=cl["q"]))
    # ---- bootstrap over cells (paired across designs that share cells) ----
    rng = np.random.default_rng(BOOT_SEED)
    alphas = sorted({v[2] for v in d6.DESIGNS.values()})
    boot = {key: np.empty(N_BOOT) for key in est if key[0] != "classical"}
    boot_m = {key: np.empty(N_BOOT) for key in boot}
    boot_edge = {key: np.zeros(N_BOOT, bool) for key in boot}
    boot_curve = {key: [] for key in boot if key[1] in LAYER_DESIGNS}     # q(d) draws for the layered filters
    for b in range(N_BOOT):
        idx = {a: rng.integers(0, d6.N_RANDOM, d6.N_RANDOM) for a in alphas}
        for (st, nm) in boot:
            d_f, U, alpha, rho = d6.DESIGNS[nm]
            em, k = draw(sets[(st, nm)], rng, idx[alpha])
            dmp, eta_m, edge_b = curve_min(sz[nm], em)
            boot[(st, nm)][b] = q_factor(eta_m, k, d_f, U, alpha)
            boot_m[(st, nm)][b] = dmp
            boot_edge[(st, nm)][b] = edge_b
            if (st, nm) in boot_curve:
                boot_curve[(st, nm)].append(q_factor(em, k, d_f, U, alpha))
    worst_edge = max(boot_edge, key=lambda key: boot_edge[key].mean())
    lines.append(f"bootstrap: largest share of draws whose MPPS falls at an end of the size window "
                 f"{100 * boot_edge[worst_edge].mean():.1f}% ({worst_edge[0]}, {worst_edge[1]}); the hexagonal cell "
                 f"uses the same particle seeds in every design, so its redraws are treated as independent "
                 f"(conservative for the elasticities)")
    for t in table:
        key = (t["structure"], t["design"])
        if key in boot:
            t["qF_lo"], t["qF_hi"] = ci(boot[key])
            t["mpps_lo"], t["mpps_hi"] = ci(boot_m[key])
        else:
            t["qF_lo"] = t["qF_hi"] = t["mpps_lo"] = t["mpps_hi"] = np.nan
    for t in table:
        extra = (f" [{t['qF_lo']:.2e}, {t['qF_hi']:.2e}], MPPS 95% interval [{t['mpps_lo']:.3f}, {t['mpps_hi']:.3f}] um"
                 if np.isfinite(t["qF_lo"]) else "")
        edge = " (smallest eta at the end of the size window)" if t["mpps_at_grid_edge"] else ""
        lines.append(f"design {t['design']}, {t['structure']}: k/a^2 = {t['k_over_a2']:.4f}, pressure drop "
                     f"{t['dp_Pa']:.1f} Pa, MPPS {t['mpps_um']:.3f} um{edge}, penetration {t['P_mpps']:.3f}, "
                     f"q_F {t['qF']:.2e} 1/Pa{extra}")
    ratio = {}
    for nm in d6.DESIGNS:
        r = boot[("random", nm)] / boot[("hexagonal", nm)]
        ratio[nm] = (est[("random", nm)] / est[("hexagonal", nm)],) + ci(r)
        cl_r = est[("hexagonal", nm)] / est[("classical", nm)]
        lines.append(f"design {nm}: q_F random / hexagonal {ratio[nm][0]:.3f} [{ratio[nm][1]:.3f}, {ratio[nm][2]:.3f}]; "
                     f"hexagonal / classical {cl_r:.3f}")
    for st in STRUCTS:
        rm = [next(t["mpps_um"] for t in table if t["design"] == nm and t["structure"] == st) /
              next(t["mpps_um"] for t in table if t["design"] == nm and t["structure"].startswith("classical"))
              for nm in d6.DESIGNS]
        rq = [est[(st, nm)] / est[("classical", nm)] for nm in d6.DESIGNS]
        lines.append(f"{st}: MPPS / classical MPPS {min(rm):.2f}-{max(rm):.2f}, q_F at the MPPS / classical "
                     f"{min(rq):.3f}-{max(rq):.3f} over the {len(rm)} designs")
    elastic = {}
    for name, lo, hi, x_lo, x_hi in ELASTIC:
        out = []
        for st in STRUCTS:
            e = np.log(est[(st, hi)] / est[(st, lo)]) / np.log(x_hi / x_lo)
            eb = np.log(boot[(st, hi)] / boot[(st, lo)]) / np.log(x_hi / x_lo)
            elastic[(st, name)] = (e,) + ci(eb)
            out.append(f"{st} {e:+.2f} [{elastic[(st, name)][1]:+.2f}, {elastic[(st, name)][2]:+.2f}]")
        e_cl = np.log(est[("classical", hi)] / est[("classical", lo)]) / np.log(x_hi / x_lo)
        elastic[("classical", name)] = (e_cl, np.nan, np.nan)
        lines.append(f"elasticity of q_F to {name}: " + ", ".join(out) + f", classical {e_cl:+.2f}")
    half = []
    for st in STRUCTS + ["classical"]:
        e1 = np.log(est[(st, "base")] / est[(st, "alpha=0.05")]) / np.log(2)
        e2 = np.log(est[(st, "alpha=0.15")] / est[(st, "base")]) / np.log(1.5)
        half.append(f"{st} {e1:+.2f} and {e2:+.2f}")
    lines.append("elasticity to solid fraction over 0.05-0.1 and over 0.1-0.15 (secants): " + ", ".join(half))
    k15 = sets[("random", "alpha=0.15")]["K"]
    lines.append(f"random cells at alpha = 0.15: k/a^2 of the six cells " + ", ".join(f"{x:.3f}" for x in k15)
                 + f"; harmonic mean {1 / np.mean(1 / k15):.4f}, arithmetic mean {k15.mean():.4f} (hexagonal "
                 f"{sets[('hexagonal', 'alpha=0.15')]['K'][0]:.4f})")
    # ---- 2. experiment (e): fibre-size spread ----
    prow = load("poly.csv")
    ps = d6.POLY_SIZES_UM
    d_f, U, alpha, rho = d6.DESIGNS["base"]
    pset = {cv: cellset([r for r in prow if float(r["cv"]) == cv], ps) for cv in d6.POLY_CVS}
    poly = {}
    pb = {cv: dict(eta=np.empty(N_BOOT), dp=np.empty(N_BOOT), q=np.empty(N_BOOT)) for cv in d6.POLY_CVS}
    for cv in d6.POLY_CVS:
        em, k = point(pset[cv])
        d_m, eta_m, edge = curve_min(ps, em)
        poly[cv] = dict(d=d_m, eta=eta_m, k=k, q=q_factor(eta_m, k, d_f, U, alpha), edge=edge, n=len(pset[cv]["K"]))
        for b in range(N_BOOT):
            idx = rng.integers(0, poly[cv]["n"], poly[cv]["n"])
            emb, kb = draw(pset[cv], rng, idx)
            _, eb, _ = curve_min(ps, emb)
            pb[cv]["eta"][b], pb[cv]["dp"][b], pb[cv]["q"][b] = eb, 1.0 / kb, q_factor(eb, kb, d_f, U, alpha)
    poly_ratio = {}
    for cv in d6.POLY_CVS[1:]:
        r = {m: (pb[cv][m] / pb[0.0][m]) for m in ("eta", "dp", "q")}
        pt_ = dict(eta=poly[cv]["eta"] / poly[0.0]["eta"], dp=poly[0.0]["k"] / poly[cv]["k"], q=poly[cv]["q"] / poly[0.0]["q"])
        poly_ratio[cv] = {m: (pt_[m],) + ci(r[m]) for m in r}
        lines.append(f"fibre-size spread CV = {cv:g} / identical ({poly[cv]['n']} cells each): eta at the MPPS "
                     f"{pt_['eta']:.3f} [{poly_ratio[cv]['eta'][1]:.3f}, {poly_ratio[cv]['eta'][2]:.3f}], pressure drop "
                     f"{pt_['dp']:.3f} [{poly_ratio[cv]['dp'][1]:.3f}, {poly_ratio[cv]['dp'][2]:.3f}], q_F "
                     f"{pt_['q']:.3f} [{poly_ratio[cv]['q'][1]:.3f}, {poly_ratio[cv]['q'][2]:.3f}]")
    for cv in d6.POLY_CVS:
        lines.append(f"fibre-size spread CV = {cv:g}: k/a^2 = {poly[cv]['k']:.4f}, MPPS {poly[cv]['d']:.3f} um, "
                     f"eta at the MPPS {poly[cv]['eta']:.4f}, "
                     f"q_F {poly[cv]['q']:.2e} 1/Pa" + (" (smallest eta at the end of the size window)" if poly[cv]["edge"] else ""))
    # the 16 new identical-fibre cells against the 6 cells of experiment (d), same sizes
    base_r = sets[("random", "base")]
    j = [sz["base"].index(d) for d in ps]
    e_new, e_old = pset[0.0]["E"].mean(axis=0), base_r["E"][:, j].mean(axis=0)
    lines.append(f"consistency: 16 new identical-fibre cells / the 6 cells of experiment (d), eta at "
                 + ", ".join(f"{d}" for d in ps) + " um: " + ", ".join(f"{x:.3f}" for x in e_new / e_old))
    # ---- 3. layered filters (Proposition 10) ----
    # each layer's q(d) on the sizes common to all layer designs (Proposition 10 compares at equal sizes)
    sizes = sorted(set.intersection(*[set(sz[nm]) for nm in LAYER_DESIGNS]))
    qcurve = {}
    for st in STRUCTS:
        for nm in LAYER_DESIGNS:
            em, k = point(sets[(st, nm)])
            keep = [sz[nm].index(d) for d in sizes]
            d_f, U, alpha, rho = d6.DESIGNS[nm]
            qcurve[(st, nm)] = q_factor(em[keep], k, d_f, U, alpha)
    for nm in LAYER_DESIGNS:
        qcurve[("classical", nm)] = classical_q_curve(nm, sizes)
    at_edge = [f"{a[0]} {a[1]}" for a in qcurve if curve_min(sizes, qcurve[a])[2]]
    lines.append(f"layered filters use the sizes {sizes[0]}-{sizes[-1]} um common to all layer designs; layers whose "
                 f"worst case is at an end of that range: {', '.join(at_edge) if at_edge else 'none'}")
    layer_types = list(qcurve)
    pairs, consistent = [], 0
    for A, B in itertools.combinations(layer_types, 2):
        if (A[0] == "classical") != (B[0] == "classical"):
            continue                       # stack simulated layers together, classical layers together
        Q = stack_Q(qcurve[A], qcurve[B], sizes)
        QA, QB = Q[-1], Q[0]                             # w = 1: layer A alone; w = 0: layer B alone
        best, other = (A, B) if QA >= QB else (B, A)
        _, fb = fine(sizes, qcurve[best])
        _, fo = fine(sizes, qcurve[other])
        # Proposition 10(c): a gain over the better layer needs the other layer to beat it at its MPPS
        cond = fo[np.argmin(fb)] > max(QA, QB)
        gain = Q.max() / max(QA, QB) - 1
        consistent += int((gain > 1e-9) == cond)        # Proposition 10(c): gain if and only if the condition
        pairs.append(dict(A=A, B=B, gain=gain, w_best=W_GRID[np.argmax(Q)] if QA != QB else np.nan, cond=cond))
    pairs.sort(key=lambda p: -p["gain"])
    n_gain = sum(p["gain"] > 1e-3 for p in pairs)
    n_cond = sum(p["cond"] for p in pairs)

    def gain_ci(A, B):
        """95% bootstrap interval of the gain of the best A + B stack over the better of A and B alone."""
        g = []
        for qa, qb in zip(boot_curve[A], boot_curve[B]):
            ka = [sz[A[1]].index(d) for d in sizes]
            kb = [sz[B[1]].index(d) for d in sizes]
            Q = stack_Q(qa[ka], qb[kb], sizes)
            g.append(Q.max() / max(Q[0], Q[-1]) - 1)
        return ci(np.array(g))
    top = pairs[0]
    top_ci = gain_ci(top["A"], top["B"]) if top["A"][0] != "classical" else (np.nan, np.nan)
    lines.append(f"layered filters (Proposition 10): {len(pairs)} two-layer stacks; {n_cond} meet condition (c) and "
                 f"{n_gain} raise the worst-case q_F by more than 0.1% over their better layer; largest gain "
                 f"{100 * top['gain']:.1f}% [{100 * top_ci[0]:.1f}, {100 * top_ci[1]:.1f}] ({top['A'][0]} {top['A'][1]} + "
                 f"{top['B'][0]} {top['B'][1]}); gain if and only if condition (c) in {consistent} of {len(pairs)}")
    fine_coarse = {}
    for st in STRUCTS + ["classical"]:
        Q = stack_Q(qcurve[(st, "d_f=1.5")], qcurve[(st, "d_f=6")], sizes)
        fine_coarse[st] = Q
        _, ff_ = fine(sizes, qcurve[(st, "d_f=1.5")])
        _, fc_ = fine(sizes, qcurve[(st, "d_f=6")])
        g_ci = gain_ci((st, "d_f=1.5"), (st, "d_f=6")) if st != "classical" else None
        extra = (f"; gain {100 * (Q.max() / max(Q[0], Q[-1]) - 1):.1f}% [{100 * g_ci[0]:.1f}, {100 * g_ci[1]:.1f}]"
                 if g_ci else f"; gain {100 * (Q.max() / max(Q[0], Q[-1]) - 1):.1f}%")
        lines.append(f"fine (1.5 um) + coarse (6 um) layers, {st}: worst-case q_F {Q[-1]:.2e} (fine only), "
                     f"{Q[0]:.2e} (coarse only), best stack {Q.max():.2e} at a fine-layer share of the pressure "
                     f"drop {W_GRID[np.argmax(Q)]:.2f}; coarse layer at the fine layer's MPPS {fc_[np.argmin(ff_)]:.2e}"
                     + extra)
    # ---- 4. sensitivity ----
    srow = load("sensitivity.csv")
    sens = {}
    for r in srow:
        sens.setdefault(r["case"], {})[float(r["d_p_um"])] = r
    base_case = sens["eps=0.05"]
    sens_lines = []
    for case in ["eps=0.1", "eps=0.025", "r_out=1.2a", "r_out=1.4a", "modes=32"]:
        rel = [float(sens[case][d]["eta"]) / float(base_case[d]["eta"]) - 1 for d in d6.SENS_SIZES_UM]
        sens_lines.append((case, rel))
        lines.append(f"sensitivity, hexagonal array, {case} instead of the standard settings (same particles): eta "
                     f"changes by {100 * rel[0]:+.1f}% at {d6.SENS_SIZES_UM[0]} um and {100 * rel[1]:+.1f}% at "
                     f"{d6.SENS_SIZES_UM[1]} um")
    f0 = float(base_case[d6.SENS_SIZES_UM[0]]["f_caught"])
    se0 = np.sqrt(f0 * (1 - f0) / float(base_case[d6.SENS_SIZES_UM[0]]["n_particles"])) / f0
    lines.append(f"sensitivity reference: binomial standard error of one hexagonal value at {d6.SENS_SIZES_UM[0]} um "
                 f"{100 * se0:.1f}% (every variant uses the same particle draws)")
    # cell size: 16-fibre cells (experiment e, CV = 0, and experiment d, base) against 32- and 36-fibre cells.
    # With only 4 large cells a bootstrap has too few distinct draws, so the interval of a ratio of means is
    # the Welch t-interval of its logarithm: se^2 = s1^2/(n1 m1^2) + s2^2/(n2 m2^2) (delta method).
    def ratio_t(x, y):
        m1, m2 = x.mean(), y.mean()
        v1, v2 = x.var(ddof=1) / (len(x) * m1 ** 2), y.var(ddof=1) / (len(y) * m2 ** 2)
        dof = (v1 + v2) ** 2 / (v1 ** 2 / (len(x) - 1) + v2 ** 2 / (len(y) - 1))
        h = stats.t.ppf(0.975, dof) * np.sqrt(v1 + v2)
        return m1 / m2, m1 / m2 * np.exp(-h), m1 / m2 * np.exp(h)
    small_k = np.concatenate([pset[0.0]["K"], base_r["K"]])
    cellsize = {}
    for case in ["32 fibres, 2 cells long", "36 fibres, 1.5 x 1.5 cells"]:
        bk = np.array([float(r["k_over_a2"]) for r in srow if r["case"] == case and float(r["d_p_um"]) == d6.CELL_SIZES_UM[0]])
        rk = ratio_t(1 / small_k, 1 / bk)            # resistances average over cells, so compare means of 1/k
        cellsize[case] = dict(k=rk, n=len(bk))
        out = []
        for d in d6.CELL_SIZES_UM:
            small_eta = np.concatenate([pset[0.0]["E"][:, ps.index(d)], base_r["E"][:, sz["base"].index(d)]])
            be = np.array([float(r["eta"]) for r in srow if r["case"] == case and float(r["d_p_um"]) == d])
            cellsize[case][d] = ratio_t(be, small_eta)
            out.append(f"eta at {d} um {cellsize[case][d][0]:.3f} [{cellsize[case][d][1]:.3f}, {cellsize[case][d][2]:.3f}]")
        lines.append(f"cell size, {case} ({len(bk)} cells) / 16-fibre cells ({len(small_k)} cells): " + ", ".join(out)
                     + f", k {rk[0]:.3f} [{rk[1]:.3f}, {rk[2]:.3f}] (Welch t-intervals)")
    return dict(lines=lines, table=table, ratio=ratio, elastic=elastic, poly=poly, poly_ratio=poly_ratio,
                pairs=pairs, fine_coarse=fine_coarse, sens=sens_lines, cellsize=cellsize, est=est, boot=boot,
                qcurve=qcurve)


def figures(B):
    names = list(d6.DESIGNS)
    # ---- design figure: (a) q_F at the MPPS per design, (b) elasticities ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    x = np.arange(len(names))
    for j, st in enumerate(STRUCTS):
        t = [next(r for r in B["table"] if r["design"] == nm and r["structure"] == st) for nm in names]
        q = np.array([r["qF"] for r in t]) * 1e3
        lo = q - np.array([r["qF_lo"] for r in t]) * 1e3
        hi = np.array([r["qF_hi"] for r in t]) * 1e3 - q
        ax[0].errorbar(x + (j - 1) * 0.18, q, yerr=[lo, hi], fmt=MARK[st], ms=7, lw=1.5, capsize=3,
                       color=COLOR[st], label=LABEL[st])
    qc = np.array([B["est"][("classical", nm)] for nm in names]) * 1e3
    ax[0].plot(x + 0.18, qc, MARK["classical"], ms=7, color=COLOR["classical"], label="classical (Eq. 8.6)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([nm.replace("alpha", r"$\alpha$").replace("d_f", "$d_f$") for nm in names], fontsize=8)
    ax[0].set_ylabel(r"$q_F$ at the MPPS (10$^{-3}$ Pa$^{-1}$)")
    ax[0].set_yscale("log")
    ticks = [0.8, 1, 1.5, 2, 3, 4, 6, 8]                       # plain numbers on the log axis
    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels([f"{t:g}" for t in ticks])
    ax[0].minorticks_off()
    ax[0].set_title("(a) quality factor of each design (95% interval)", fontsize=10)
    ax[0].grid(axis="y", color="#e6e6e3", lw=0.6)
    ax[0].legend(fontsize=8, frameon=False)
    xe = np.arange(len(ELASTIC))
    for j, st in enumerate(STRUCTS + ["classical"]):
        e = np.array([B["elastic"][(st, E[0])] for E in ELASTIC])
        yerr = None if st == "classical" else [e[:, 0] - e[:, 1], e[:, 2] - e[:, 0]]
        ax[1].errorbar(xe + (j - 1) * 0.22, e[:, 0], yerr=yerr, fmt=MARK[st], ms=7, lw=1.5, capsize=3,
                       color=COLOR[st], label=LABEL.get(st, "classical (Eq. 8.6)"))
    ax[1].axhline(0, color="#9a9a94", lw=0.8)
    ax[1].set_xticks(xe)
    ax[1].set_xticklabels(["$d_f$", "$U$", r"$\alpha$", r"$\rho_p$"])
    ax[1].set_ylabel(r"elasticity $\Delta\ln q_F / \Delta\ln X$")
    ax[1].set_title("(b) how $q_F$ responds to each design variable", fontsize=10)
    ax[1].grid(axis="y", color="#e6e6e3", lw=0.6)
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_D)
    plt.close(fig)
    # ---- layers figure: (a) fibre-size spread, (b) fine + coarse stacks ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cvs = d6.POLY_CVS[1:]
    for j, (m, lab, col, mk) in enumerate([("eta", r"$\eta$ at the MPPS", "#2a78d6", "s"),
                                           ("dp", "pressure drop", "#eb6834", "o"),
                                           ("q", "$q_F$ at the MPPS", "#1baf7a", "D")]):
        v = np.array([B["poly_ratio"][cv][m] for cv in cvs])
        ax[0].errorbar(np.array(cvs) + (j - 1) * 0.02, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
                       fmt=mk, ms=7, lw=1.5, capsize=3, color=col, label=lab)
    ax[0].axhline(1, color="#9a9a94", lw=0.8)
    ax[0].set_xticks(cvs)
    ax[0].set_xlabel("fibre-size spread CV")
    ax[0].set_ylabel("ratio to identical fibres")
    ax[0].set_title("(a) fibre-size spread (16 cells each, 95% interval)", fontsize=10)
    ax[0].legend(fontsize=8, frameon=False)
    for st in STRUCTS + ["classical"]:
        Q = B["fine_coarse"][st]
        ax[1].plot(W_GRID, Q * 1e3, "-", lw=2, color=COLOR[st], label=LABEL.get(st, "classical (Eq. 8.6)"))
        ax[1].plot(W_GRID[::25], Q[::25] * 1e3, MARK[st], ms=7, color=COLOR[st])
    ax[1].set_xlabel("share of the pressure drop in the fine (1.5 $\\mu$m) layer")
    ax[1].set_ylabel(r"worst-case $q_F$ (10$^{-3}$ Pa$^{-1}$)")
    ax[1].set_title("(b) two layers in series (Proposition 10)", fontsize=10)
    ax[1].grid(color="#e6e6e3", lw=0.6)
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_L)
    plt.close(fig)


def main():
    B = build()
    cols = ["design", "structure", "d_f_um", "U", "alpha", "rho_p", "k_over_a2", "mpps_um", "mpps_lo", "mpps_hi",
            "mpps_at_grid_edge", "eta_mpps", "dp_Pa", "P_mpps", "qF", "qF_lo", "qF_hi"]
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t in B["table"]:
            w.writerow([t[c] for c in cols])
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(B["lines"]) + "\n")
    print("\n".join(B["lines"]))
    figures(B)
    print("wrote", OUT_CSV, OUT_TXT, FIG_D, FIG_L)


if __name__ == "__main__":
    main()
