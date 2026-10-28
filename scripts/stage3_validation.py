"""
stage3_validation.py
====================
Stage 3: validate our 2D regularized Stokeslet (RS) solver.

1. The independent reference solver (spectral boundary integral, reference_bie.py)
   against the published Sangani & Acrivos (1982) square-array drag table.
2. The effective radius of a ring of blobs, against Proposition 6: a_eff = R + eps/(2 sqrt(pi)).
3. Convergence of the RS permeability with and without the offset rule; Richardson extrapolation.
4. Leakage and conditioning against the blob-to-spacing ratio eps/h (Poisson ripple exp(-pi^2 eps^2/h^2)).
5. Near-fibre velocity error (what interception will feel in Stage 5).
6. Random arrays with near-contacts: RS against the reference.
7. Timing: dense solve and GMRES.

Outputs: results/stage3/*.csv, results/stage3/summary.txt, paper/figures/stage3_*.pdf
Run from the project folder:   python scripts/stage3_validation.py      (about 4 minutes)
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import geometry as geo                                                   # noqa: E402
from solver2d import FibreSolver2D, solve_with_rule, richardson_k, OFFSET_FACTOR   # noqa: E402
from reference_bie import ReferenceBIE2D, reference_k                     # noqa: E402

RESULTS = os.path.join(ROOT, "results", "stage3")
FIGURES = os.path.join(ROOT, "paper", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
summary = []
T0 = time.perf_counter()


def note(line):
    summary.append(line)
    print(line, flush=True)


def save(name, header, rows):
    np.savetxt(os.path.join(RESULTS, name), np.array(rows), delimiter=",", header=header,
               comments="", fmt="%.10e")


# ============================================================================
# 1. Reference solver vs Sangani & Acrivos (1982), Table 1 (square arrays)
#    Values F/(mu U) as reproduced in the Basilisk test suite (cylinders.c).
#    k / a^2 = pi / (alpha * F/(mu U))   because k = mu U A / F and A = pi a^2 / alpha.
# ============================================================================
SA_TABLE = [(0.05, 15.56), (0.10, 24.83), (0.20, 51.53), (0.30, 102.90), (0.40, 217.89), (0.50, 532.55)]
rows = []
worst_sa = 0.0
for alpha, F_sa in SA_TABLE:
    lay = geo.square_array(alpha)
    k32, k64, k128 = (reference_k(lay, M) for M in (32, 64, 128))
    F_ref = np.pi / (alpha * k128)
    series = (np.log(1 / alpha) - 1.476 + 2 * alpha - 1.774 * alpha**2 + 4.076 * alpha**3) / (8 * alpha)
    rel = F_ref / F_sa - 1
    worst_sa = max(worst_sa, abs(rel))
    rows.append([alpha, F_sa, F_ref, rel, k128, abs(k64 / k128 - 1), series / k128 - 1])
save("reference_vs_sangani_acrivos.csv",
     "alpha,F_over_muU_SA1982,F_over_muU_reference,rel_diff,k_over_a2_reference,"
     "change_M64_to_M128,series_rel_error", rows)
rows_sa = np.array(rows)
note(f"reference vs Sangani-Acrivos table: largest relative difference = {worst_sa:.1e} (alpha 0.05-0.5)")
note(f"reference: change from M = 64 to 128 points per fibre <= {rows_sa[:, 5].max():.1e}")
note(f"truncated dilute series: error {100 * rows_sa[1, 6]:.2f}% at alpha = 0.1, "
     f"{100 * rows_sa[3, 6]:.0f}% at alpha = 0.3, {100 * rows_sa[5, 6]:.0f}% at alpha = 0.5")

# hexagonal arrays: reference vs the dilute Sangani-Acrivos series (leading terms)
hex_rows = []
for alpha in [1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    k_hex = reference_k(geo.hexagonal_array(alpha), 64)
    ser = (np.log(1 / alpha) - 1.4975 + 2 * alpha - 0.5 * alpha**2) / (8 * alpha)
    hex_rows.append([alpha, k_hex, ser, ser / k_hex - 1, -(8 * alpha * k_hex) - np.log(alpha)])
save("reference_hexagonal.csv", "alpha,k_over_a2_reference,k_over_a2_series,series_rel_error,C", hex_rows)
hex_rows = np.array(hex_rows)
note(f"hexagonal reference: C = {hex_rows[0, 4]:.4f} at alpha = 1e-3 (series 1.4975 - 2 alpha = {1.4975 - 2e-3:.4f}); "
     f"series error {100 * abs(hex_rows[3, 3]):.3f}% at alpha = 0.1")

# ============================================================================
# 2. Effective radius of a ring of blobs (Proposition 6)
# ============================================================================
alpha_d = 0.01
lay_d = geo.square_array(alpha_d)
eff_rows = []
for eps in [0.2, 0.1, 0.05, 0.025]:
    s = FibreSolver2D(lay_d, eps, eps, offset=0.0)          # blobs ON the circle R = 1
    s.solve()
    k_rs = s.permeability()
    a_eff = brentq(lambda a: reference_k(dict(lay_d, radii=np.array([a])), 64) * a**2 - k_rs,
                   0.9, 1.3, xtol=1e-13)
    eff_rows.append([eps, a_eff, (a_eff - 1) / eps])
save("effective_radius.csv", "eps_over_R,a_eff_over_R,shift_over_eps", eff_rows)
eff = np.array(eff_rows)
# straight-line fit shift/eps = beta + gamma eps through the three smallest eps
fit = np.polyfit(eff[1:, 0], eff[1:, 2], 1)
note(f"effective radius: (a_eff - R)/eps -> {fit[1]:.4f} as eps -> 0 (Proposition 6: 1/(2 sqrt(pi)) = "
     f"{OFFSET_FACTOR:.4f}); next term {fit[0]:.2f} eps^2/R")

# ============================================================================
# 3. Convergence of the permeability, with and without the offset rule
# ============================================================================
cases = [("square", 0.1), ("square", 0.3), ("hexagonal", 0.3)]
eps_list = [0.2, 0.1, 0.05, 0.025]
conv_rows = []
conv = {}
for kind, alpha in cases:
    lay = geo.square_array(alpha) if kind == "square" else geo.hexagonal_array(alpha)
    k_ref = reference_k(lay, 128)
    e_plain, e_rule, k_rule = [], [], []
    for eps in eps_list:
        kp = FibreSolver2D(lay, eps, eps, offset=0.0)
        kp.solve()
        kr = solve_with_rule(lay, eps).dimensionless_permeability()
        e_plain.append(kp.dimensionless_permeability() / k_ref - 1)
        e_rule.append(kr / k_ref - 1)
        k_rule.append(kr)
    e_rich = [(4 * k_rule[i + 1] - k_rule[i]) / 3 / k_ref - 1 for i in range(len(eps_list) - 1)]
    conv[(kind, alpha)] = (np.array(e_plain), np.array(e_rule), np.array(e_rich))
    for i, eps in enumerate(eps_list):
        conv_rows.append([0 if kind == "square" else 1, alpha, eps, e_plain[i], e_rule[i],
                          e_rich[i - 1] if i > 0 else np.nan])
save("convergence.csv", "array(0=square;1=hex),alpha,eps_over_a,err_no_offset,err_offset_rule,"
     "err_richardson_eps_and_2eps", conv_rows)
ep, er, eR = conv[("square", 0.3)]
note(f"square alpha=0.3, eps = 0.05a: error {100 * ep[2]:.1f}% without offset, {100 * er[2]:.2f}% with the rule, "
     f"{100 * eR[1]:.3f}% with Richardson (0.05a, 0.1a)")
order_plain = np.log2(ep[2] / ep[3])
order_rule = np.log2(er[2] / er[3])
note(f"observed order (eps 0.05a -> 0.025a): {order_plain:.2f} without offset, {order_rule:.2f} with the rule")
worst_rule_01 = max(abs(conv[c][1][1]) for c in cases)
worst_rich_01 = max(abs(conv[c][2][0]) for c in cases)
note(f"all three regular arrays: |error| <= {100 * worst_rule_01:.1f}% at eps = 0.1a with the rule, "
     f"<= {100 * worst_rich_01:.2f}% with Richardson (0.1a, 0.2a)")

# ============================================================================
# 4. Leakage and conditioning against eps / h (fixed eps = 0.1a, alpha = 0.1 square)
# ============================================================================
lay = geo.square_array(0.1)
k_ref01 = reference_k(lay, 128)
leak_rows = []
for ratio in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    eps = 0.1
    s = FibreSolver2D(lay, eps / ratio, eps, offset=OFFSET_FACTOR * eps)
    s.solve()
    _, qpos, qchord = s.leakage()
    true_ratio = eps / (2 * np.pi * 1.0 / s.n_per[0])        # exact h from the number of points
    leak_rows.append([true_ratio, qpos[0], abs(qchord[0]), s.condition_number(),
                      s.dimensionless_permeability() / k_ref01 - 1, 2 * np.exp(-np.pi**2 * true_ratio**2)])
save("leakage_vs_ratio.csv", "eps_over_h,positive_flux,abs_chord_flux,condition_number,k_rel_error,ripple_2exp",
     leak_rows)
leak = np.array(leak_rows)
i1 = int(np.argmin(np.abs(leak[:, 0] - 1.0)))
note(f"eps/h = {leak[i1, 0]:.2f}: leakage Q+ = {leak[i1, 1]:.1e}, condition number = {leak[i1, 3]:.1e}; "
     f"eps/h = {leak[-1, 0]:.2f}: condition number = {leak[-1, 3]:.1e}")
# Proposition 6(b): leakage falls like exp(-c (eps/h)^2) with c close to pi^2 (Poisson ripple).
# Fit ln Q+ against (eps/h)^2 over 0.5 <= eps/h <= 1.5 (beyond that Q+ reaches round-off).
mfit = (leak[:, 0] > 0.45) & (leak[:, 0] < 1.6)
slope_leak = -np.polyfit(leak[mfit, 0] ** 2, np.log(leak[mfit, 1]), 1)[0]
note(f"leakage decay: Q+ ~ exp(-c (eps/h)^2) with fitted c = {slope_leak:.1f} (ripple theory: pi^2 = {np.pi**2:.2f}); "
     f"chord and positive-part fluxes agree to {np.max(np.abs(leak[:5, 2] / leak[:5, 1] - 1)):.0e} where above round-off")

# ============================================================================
# 5. Near-fibre velocity error: what a particle grazing the fibre feels
# ============================================================================
ref = ReferenceBIE2D(lay, 256)
ref.solve()
c0 = lay["centres"][0]
angles = np.linspace(0, np.pi, 13)[1:-1]
near_rows = []
for eps in [0.1, 0.05, 0.025]:
    s = solve_with_rule(lay, eps)
    for d in [0.1, 0.2, 0.4, 0.8]:
        x = c0 + (1 + d) * np.column_stack([np.cos(angles), np.sin(angles)])
        u_rs, u_ref = s.velocity(x), ref.velocity(x)
        err = np.max(np.linalg.norm(u_rs - u_ref, axis=1) / np.linalg.norm(u_ref, axis=1))
        near_rows.append([eps, d, err])
save("near_field_error.csv", "eps_over_a,distance_from_surface_over_a,max_relative_velocity_error", near_rows)
near = np.array(near_rows)
pick = lambda e, d: near[(near[:, 0] == e) & (near[:, 1] == d), 2][0]
note(f"near-fibre velocity error at distance 0.2a: {100 * pick(0.1, 0.2):.1f}% (eps 0.1a), "
     f"{100 * pick(0.05, 0.2):.1f}% (0.05a), {100 * pick(0.025, 0.2):.2f}% (0.025a)")

# ============================================================================
# 6. Random arrays with near-contacts: RS against the reference
# ============================================================================
rand_rows = []
for seed, alpha, cv in [(1, 0.1, 0.0), (2, 0.3, 0.0), (3, 0.1, 0.5), (4, 0.3, 0.5)]:
    rng = np.random.default_rng(100 + seed)
    lay_r = geo.random_array(alpha, 16, rng, cv=cv, min_gap=0.2)
    k_ref = reference_k(lay_r, 64)
    kR, k1, k2 = richardson_k(lay_r, 0.1)
    rand_rows.append([alpha, cv, geo.min_surface_gap(lay_r), k_ref, k1 / k_ref - 1, kR / k_ref - 1])
save("random_vs_reference.csv", "alpha,cv,min_gap,k_reference,err_rule_eps0.1,err_richardson_0.1_0.2", rand_rows)
rand = np.array(rand_rows)
note(f"random arrays (16 fibres, gaps >= 0.2a): |error| <= {100 * np.abs(rand[:, 4]).max():.1f}% at eps = 0.1a, "
     f"<= {100 * np.abs(rand[:, 5]).max():.2f}% with Richardson (0.1a, 0.2a)")

# ============================================================================
# 7. Timing: dense LU and GMRES
# ============================================================================
time_rows = []
rng = np.random.default_rng(7)
lay_t = geo.random_array(0.2, 16, rng, min_gap=0.2)
for eps in [0.2, 0.1, 0.05]:
    s = FibreSolver2D(lay_t, eps, eps, offset=OFFSET_FACTOR * eps)
    s.assemble()
    s.solve(method="dense")
    t_dense, k_dense = s.timing["solve_s"], s.dimensionless_permeability()
    s.solve(method="gmres", tol=1e-10)
    time_rows.append([s.N, s.timing["assemble_s"], t_dense, s.timing["solve_s"], s.gmres_iterations,
                      abs(s.dimensionless_permeability() / k_dense - 1)])
save("timing.csv", "N_blobs,assemble_s,dense_solve_s,gmres_solve_s,gmres_matvecs,gmres_vs_dense_rel_diff", time_rows)
tim = np.array(time_rows)
note(f"timing (16 fibres): N = {int(tim[-1, 0])} blobs, assembly {tim[-1, 1]:.0f} s, dense solve {tim[-1, 2]:.1f} s, "
     f"GMRES {int(tim[-1, 4])} matrix-vector products; GMRES and dense agree to {tim[:, 5].max():.0e}")

# ============================================================================
# Figures
# ============================================================================
# (a) validation against Sangani & Acrivos, (b) convergence
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
al = np.linspace(0.03, 0.55, 100)
ax.semilogy(rows_sa[:, 0], np.pi / (rows_sa[:, 0] * rows_sa[:, 1]), "ks", ms=7, mfc="none",
            label="Sangani & Acrivos (1982), table")
ax.semilogy(rows_sa[:, 0], rows_sa[:, 4], "C0o", ms=4, label="our reference (boundary integral)")
k_rs03 = [richardson_k(geo.square_array(a_), 0.1)[0] for a_ in rows_sa[:, 0]]
ax.semilogy(rows_sa[:, 0], k_rs03, "C3x", ms=7, label="our RS solver ($\\varepsilon=0.1a$, Richardson)")
ax.semilogy(al, (np.log(1 / al) - 1.476 + 2 * al - 1.774 * al**2 + 4.076 * al**3) / (8 * al), "C2:",
            label="dilute series (Sec. 3)")
ax.semilogy(al, (-0.5 * np.log(al) - 0.75 + al - 0.25 * al**2) / (4 * al), "0.5", ls="--", lw=1,
            label="Kuwabara cell model")
ax.set_xlabel("solid fraction $\\alpha$")
ax.set_ylabel("$k/a^2$")
ax.set_title("(a) square arrays")
ax.legend(frameon=False, fontsize=7.5)
ax = axes[1]
for (kind, alpha), mk in zip(cases, ["o", "s", "^"]):
    ep_, er_, eR_ = conv[(kind, alpha)]
    ax.loglog(eps_list, np.abs(ep_), mk + "-", color="C1", ms=5)
    ax.loglog(eps_list, np.abs(er_), mk + "-", color="C0", ms=5)
    ax.loglog(eps_list[1:], np.abs(eR_), mk + ":", color="C3", ms=5)
ee = np.array(eps_list)
ax.loglog(ee, 1.3 * ee, "k--", lw=0.8)
ax.loglog(ee, 0.7 * ee**2, "k-.", lw=0.8)
ax.text(0.09, 0.15, "slope 1", fontsize=8)
ax.text(0.09, 0.0035, "slope 2", fontsize=8)
ax.plot([], [], "C1-", label="blobs on the surface")
ax.plot([], [], "C0-", label="offset rule $\\varepsilon/(2\\sqrt{\\pi})$")
ax.plot([], [], "C3:", label="rule + Richardson")
ax.plot([], [], "ko", label="square 0.1")
ax.plot([], [], "ks", label="square 0.3")
ax.plot([], [], "k^", label="hexagonal 0.3")
ax.set_xticks(eps_list)
ax.set_xticklabels([f"{e:g}" for e in eps_list])
ax.minorticks_off()
ax.set_xlabel("blob width $\\varepsilon/a$ (spacing $h=\\varepsilon$)")
ax.set_ylabel("relative error in $k$")
ax.set_title("(b) convergence")
ax.legend(frameon=False, fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "stage3_validation.pdf"))
plt.close(fig)

# leakage / conditioning and effective radius
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
ax = axes[0]
ax.semilogy(leak[:, 0], leak[:, 1], "o-", label="leakage $Q_+$")
ax.semilogy(leak[:, 0], leak[:, 2], "s--", label="chord flux $|Q_c|$")
i_ref = int(np.argmin(np.abs(leak[:, 0] - 1.0)))
ax.semilogy(leak[:, 0], leak[i_ref, 1] * np.exp(-np.pi**2 * (leak[:, 0] ** 2 - leak[i_ref, 0] ** 2)), "k:",
            label="$\\propto e^{-\\pi^2\\varepsilon^2/h^2}$ (Prop. 6)")
ax.set_xlabel("$\\varepsilon/h$")
ax.set_ylabel("flux / $(U\\cdot 2R)$")
ax2 = ax.twinx()
ax2.semilogy(leak[:, 0], leak[:, 3], "C3^-", label="condition number")
ax2.set_ylabel("condition number", color="C3")
ax.set_title("(a) leakage closes, conditioning grows")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=7.5, loc="center left")
ax = axes[1]
ax.plot(eff[:, 0], eff[:, 2], "o-", label="measured $(a_{\\rm eff}-R)/\\varepsilon$")
xx = np.linspace(0, 0.21, 20)
ax.plot(xx, fit[1] + fit[0] * xx, "k:", lw=1, label=f"fit {fit[1]:.4f} + {fit[0]:.2f}$\\varepsilon$")
ax.axhline(OFFSET_FACTOR, color="C3", ls="--", label="$1/(2\\sqrt{\\pi})=0.2821$ (Prop. 6)")
ax.set_xlim(0, 0.21)
ax.set_xlabel("$\\varepsilon/R$")
ax.set_ylabel("shift of the effective surface / $\\varepsilon$")
ax.set_title("(b) effective radius of a ring of blobs")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "stage3_leak_radius.pdf"))
plt.close(fig)

# flow through one random polydisperse layout (picture)
rng = np.random.default_rng(104)
lay_p = geo.random_array(0.3, 16, rng, cv=0.5, min_gap=0.2)
s = solve_with_rule(lay_p, 0.1)
Lx, Ly = lay_p["Lx"], lay_p["Ly"]
gx, gy = np.linspace(0, Lx, 120), np.linspace(0, Ly, 120)
GX, GY = np.meshgrid(gx, gy)
pts = np.column_stack([GX.ravel(), GY.ravel()])
U = s.velocity(pts)
speed = np.linalg.norm(U, axis=1).reshape(GX.shape)
inside = np.zeros(len(pts), bool)
for c, a in zip(lay_p["centres"], lay_p["radii"]):
    d = pts - c
    d -= np.array([Lx, Ly]) * np.round(d / [Lx, Ly])
    inside |= np.sum(d**2, axis=1) < a**2
speed.ravel()[inside] = np.nan
fig, ax = plt.subplots(figsize=(5.6, 5.0))
im = ax.pcolormesh(GX, GY, speed, cmap="viridis", shading="auto")
ax.streamplot(GX, GY, U[:, 0].reshape(GX.shape), U[:, 1].reshape(GX.shape), color="w", density=1.4, linewidth=0.5)
for c, a in zip(lay_p["centres"], lay_p["radii"]):
    for ox in (-Lx, 0, Lx):
        for oy in (-Ly, 0, Ly):
            ax.add_patch(Circle((c[0] + ox, c[1] + oy), a, fc="0.85", ec="k", lw=0.5, zorder=3))
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly); ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=ax, label="speed $|\\mathbf{u}|/U$", shrink=0.8)
ax.set_title("Flow through a random polydisperse array ($\\alpha=0.3$, CV 0.5)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "stage3_flow_random.pdf"))
plt.close(fig)

note(f"total run time {time.perf_counter() - T0:.0f} s")
with open(os.path.join(RESULTS, "summary.txt"), "w") as fh:
    fh.write("\n".join(summary) + "\n")
print("Wrote results/stage3 and paper/figures/stage3_*.pdf")
