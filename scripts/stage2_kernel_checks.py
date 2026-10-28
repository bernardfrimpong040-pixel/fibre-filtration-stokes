"""
stage2_kernel_checks.py
=======================
Stage 2: numerical evidence for the theory in main.tex Section 4.

It produces
  results/stage2/summary.txt              every number quoted in main.tex
  results/stage2/far_field.csv            regularized minus (Stokeslet + dipole) vs r/eps
  results/stage2/dilute_limit.csv         one-blob periodic array vs Hasimoto
  paper/figures/stage2_model.pdf          the model: periodic cell, fibres, blobs
  paper/figures/stage2_far_field.pdf      Proposition 2 (far field = Stokeslet + dipole)
  paper/figures/stage2_dilute_leak.pdf    Hasimoto constant + Proposition 1 (zero net flux)

Run from the project folder:   python scripts/stage2_kernel_checks.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import rs_kernels as rk                       # noqa: E402
from periodic import PeriodicStokeslet2D, spectral_only_blocks   # noqa: E402

RESULTS = os.path.join(ROOT, "results", "stage2")
FIGURES = os.path.join(ROOT, "paper", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
summary = []                                  # lines for summary.txt


# ============================================================================
# 1. Free-space kernel solves the forced Stokes equations (finite differences)
# ============================================================================
eps, mu = 0.1, 1.0
f0 = np.array([[0.7, -0.4]])                   # one blob force, centred at the origin
src = np.zeros((1, 2))
h = eps / 200.0                                # finite-difference step

def u_at(p):
    return rk.velocity_2d(np.atleast_2d(p), src, f0, eps, mu)[0]

def p_at(p):
    return rk.pressure_2d(np.atleast_2d(p), src, f0, eps)[0]

worst_div, worst_res, worst_scale = 0.0, 0.0, 0.0
for pt in [(0.03, 0.02), (0.1, -0.05), (0.25, 0.1), (0.5, 0.4)]:
    pt = np.array(pt)
    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    # divergence: du_x/dx + du_y/dy by central differences
    div = (u_at(pt + ex)[0] - u_at(pt - ex)[0] + u_at(pt + ey)[1] - u_at(pt - ey)[1]) / (2 * h)
    # Laplacian of u and gradient of p
    lap = (u_at(pt + ex) + u_at(pt - ex) + u_at(pt + ey) + u_at(pt - ey) - 4 * u_at(pt)) / h**2
    gradp = np.array([p_at(pt + ex) - p_at(pt - ex), p_at(pt + ey) - p_at(pt - ey)]) / (2 * h)
    force = f0[0] * rk.blob_2d(np.linalg.norm(pt), eps)
    residual = mu * lap - gradp + force                    # Eq. (4.1): should be 0
    worst_div = max(worst_div, abs(div))
    worst_res = max(worst_res, np.max(np.abs(residual)))
    worst_scale = max(worst_scale, np.max(np.abs(force)), np.max(np.abs(gradp)))
summary.append(f"max |div u| (finite differences) = {worst_div:.2e}")
summary.append(f"max Stokes residual / max term size = {worst_res / worst_scale:.2e}")

# ============================================================================
# 2. Proposition 2: far field = singular Stokeslet + potential dipole of strength eps^2/2
# ============================================================================
r_over_eps = np.linspace(0.5, 6.0, 56)
A, B = rk.kernel_AB_2d(r_over_eps * eps, eps)
As, Bs = rk.singular_AB_2d(r_over_eps * eps)
r = r_over_eps * eps
# dipole adds +eps^2/(2 r^2) to A and -eps^2/r^4 to B (Eq. 4.14)
A_dip, B_dip = As + eps**2 / (2 * r**2), Bs - eps**2 / r**4
err_stokeslet = np.maximum(np.abs(A - As), np.abs(B - Bs) * r**2)       # compare like with like
err_dipole = np.maximum(np.abs(A - A_dip), np.abs(B - B_dip) * r**2)
np.savetxt(os.path.join(RESULTS, "far_field.csv"),
           np.column_stack([r_over_eps, err_stokeslet, err_dipole]), delimiter=",",
           header="r_over_eps,err_vs_stokeslet,err_vs_stokeslet_plus_dipole", comments="", fmt="%.6e")
i4 = int(np.argmin(np.abs(r_over_eps - 4.0)))
summary.append(f"at r = 4 eps: error vs Stokeslet = {err_stokeslet[i4]:.2e}, "
               f"vs Stokeslet+dipole = {err_dipole[i4]:.2e}")

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.semilogy(r_over_eps, err_stokeslet, "-", lw=2, label="vs. singular Stokeslet")
ax.semilogy(r_over_eps, err_dipole, "--", lw=2, label="vs. Stokeslet + dipole ($\\varepsilon^2/2$)")
ax.semilogy(r_over_eps, np.exp(-r_over_eps**2) / r_over_eps**2, ":", color="0.4",
            label="$e^{-r^2/\\varepsilon^2}\\,\\varepsilon^2/r^2$")
ax.set_xlabel("distance from blob centre $r/\\varepsilon$")
ax.set_ylabel("kernel difference")
ax.set_ylim(1e-17, 1)
ax.legend(frameon=False, fontsize=8)
ax.set_title("Proposition 2: the blob looks like a force plus a dipole")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "stage2_far_field.pdf"))
plt.close(fig)

# ============================================================================
# 3. Ewald periodic kernel: independent of xi, equal to the plain Fourier series
# ============================================================================
rng = np.random.default_rng(2026)
x_eval, x_src = rng.random((8, 2)), rng.random((5, 2))
Lx, Ly, eps_p = 1.0, 1.0, 0.05
b_a = PeriodicStokeslet2D(Lx, Ly, eps_p, xi=Lx / 5).blocks(x_eval, x_src)
b_b = PeriodicStokeslet2D(Lx, Ly, eps_p, xi=Lx / 8).blocks(x_eval, x_src)
b_s = spectral_only_blocks(x_eval, x_src, Lx, Ly, eps_p, n_max=80)
scale = max(np.max(np.abs(b)) for b in b_a)
xi_diff = max(np.max(np.abs(p - q)) for p, q in zip(b_a, b_b)) / scale
spec_diff = max(np.max(np.abs(p - q)) for p, q in zip(b_a, b_s)) / scale
summary.append(f"Ewald: relative change when xi goes from L/5 to L/8 = {xi_diff:.1e}")
summary.append(f"Ewald vs plain Fourier series: relative difference = {spec_diff:.1e}")

# ============================================================================
# 4. Dilute limit: one blob per cell reproduces Hasimoto's constant 1.476
# ============================================================================
GAMMA = rk.EULER_GAMMA
alphas = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
consts = []
for alpha in alphas:
    a = 1.0
    L = np.sqrt(np.pi * a**2 / alpha)                 # square cell with solid fraction alpha
    eps_1 = a * np.exp(GAMMA / 2.0)                   # blob width giving effective radius a (Prop. 3, Eq. 4.15)
    P = PeriodicStokeslet2D(L, L, eps_1)
    Sxx, _, _ = P.blocks(np.zeros((1, 2)), np.zeros((1, 2)))
    # no-slip at the blob centre: U + Sxx f = 0  ->  f = -U / Sxx   (take U = 1, mu = 1)
    f = -1.0 / Sxx[0, 0]
    k = mu * 1.0 * L * L / abs(f)                      # Eq. (4.17): k = mu U A / |F|
    consts.append(-(8 * alpha * k / a**2) - np.log(alpha))   # C in k/a^2 = (ln(1/alpha) - C)/(8 alpha)
consts = np.array(consts)
np.savetxt(os.path.join(RESULTS, "dilute_limit.csv"), np.column_stack([alphas, consts]),
           delimiter=",", header="alpha,C_numerical (Hasimoto: 1.476)", comments="", fmt="%.6e")
summary.append(f"one-blob periodic array: C = {consts[0]:.4f} at alpha = 1e-4 (Hasimoto 1.476)")
# Reference for a real cylinder: the square-array series of Section 3,
# k/a^2 = [ln(1/alpha) - 1.476 + 2 alpha - 1.774 alpha^2 + 4.076 alpha^3]/(8 alpha),
# i.e. C_cyl = 1.476 - 2 alpha + 1.774 alpha^2 - 4.076 alpha^3
C_cyl = 1.476 - 2 * alphas + 1.774 * alphas**2 - 4.076 * alphas**3
k_blob = (np.log(1 / alphas) - consts) / (8 * alphas)
k_cyl = (np.log(1 / alphas) - C_cyl) / (8 * alphas)
slope = (consts[2] - 1.476336) / alphas[2]          # (C - C0)/alpha at alpha = 1e-3
summary.append(f"one-blob periodic array: C = {consts[-1]:.4f} at alpha = 0.1; cylinder series gives {C_cyl[-1]:.4f}; "
               f"k differs by {100 * abs(k_blob[-1] / k_cyl[-1] - 1):.1f}%")
summary.append(f"one-blob periodic array: (C - 1.476336)/alpha = {slope:.2f} at alpha = 1e-3 (cylinder: -2)")

# ============================================================================
# 5. Proposition 1: zero net flux through the fibre surface, yet air can pass through
#    A PREVIEW of the Stage 3 problem: one fixed fibre (a ring of N blobs, radius a)
#    in a periodic square cell with mean flow U = 1 along x. Forces are chosen so the
#    velocity is zero AT the N blob centres. Between the centres air may leak through.
# ============================================================================
a_fib, alpha_demo = 1.0, 0.1
L_demo = np.sqrt(np.pi * a_fib**2 / alpha_demo)
centre = np.array([L_demo / 2, L_demo / 2])


def fixed_ring(n_pts, eps_ring):
    """Solve for blob forces that make u = 0 at the ring points (Eq. 4.20). Returns P, ring, forces."""
    t = 2 * np.pi * np.arange(n_pts) / n_pts
    ring_pts = centre + a_fib * np.column_stack([np.cos(t), np.sin(t)])
    Pk = PeriodicStokeslet2D(L_demo, L_demo, eps_ring)
    Sxx, Sxy, Syy = Pk.blocks(ring_pts, ring_pts)
    matrix = np.block([[Sxx, Sxy], [Sxy, Syy]])
    rhs = -np.concatenate([np.ones(n_pts), np.zeros(n_pts)])     # U + S f = 0  ->  S f = -U
    sol = np.linalg.solve(matrix, rhs)
    return Pk, ring_pts, np.column_stack([sol[:n_pts], sol[n_pts:]])


def leakage(Pk, ring_pts, forces):
    """Net flux and positive-part flux out of the fibre surface, and flux through its
    vertical diameter (chord), all divided by U * 2a (the flow that would hit the fibre)."""
    n_q = 2000
    tq = 2 * np.pi * (np.arange(n_q) + 0.5) / n_q
    normal = np.column_stack([np.cos(tq), np.sin(tq)])
    u_s = Pk.velocity(centre + a_fib * normal, ring_pts, forces, U=(1.0, 0.0))
    un = np.sum(u_s * normal, axis=1)
    ds = 2 * np.pi * a_fib / n_q
    yq = np.linspace(-a_fib, a_fib, 1601)
    u_c = Pk.velocity(centre + np.column_stack([0 * yq, yq]), ring_pts, forces, U=(1.0, 0.0))
    norm = 1.0 * 2 * a_fib
    return (np.sum(un) * ds / norm, np.sum(np.maximum(un, 0)) * ds / norm,
            np.trapezoid(u_c[:, 0], yq) / norm)


leak_rows = []
for n_pts, eps_ring in [(8, 0.05), (32, 0.05), (32, 0.20)]:
    Pk, ring_pts, forces = fixed_ring(n_pts, eps_ring)
    net, pos, chord = leakage(Pk, ring_pts, forces)
    spacing = 2 * np.pi * a_fib / n_pts
    leak_rows.append((n_pts, eps_ring, eps_ring / spacing, net, pos, chord))
    summary.append(f"fixed fibre, N = {n_pts}, eps/spacing = {eps_ring / spacing:.2f}: net flux = {net:.1e}, "
                   f"positive-part flux = {pos:.2e}, chord flux = {chord:.2e} (fractions of U*2a)")
np.savetxt(os.path.join(RESULTS, "leakage_preview.csv"), np.array(leak_rows), delimiter=",",
           header="N,eps,eps_over_spacing,net_flux,positive_flux,chord_flux", comments="", fmt="%.6e")
# keep the leaky case for the picture
Pk, ring, ring_forces = fixed_ring(8, 0.05)
net_flux, _, chord_flux = leakage(Pk, ring, ring_forces)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
ax = axes[0]
ax.semilogx(alphas, consts, "o-", label="one Gaussian blob per cell (ours)")
ax.axhline(1.476, color="k", ls="--", lw=1, label="Hasimoto (1959): 1.476")
ax.semilogx(alphas, C_cyl, "s:", color="C2", ms=4, label="cylinders: square-array series (Sec. 3)")
ax.set_xlabel("solid fraction $\\alpha$")
ax.set_ylabel("$C$ in $k/a^2 = [\\ln(1/\\alpha) - C]/(8\\alpha)$")
ax.set_title("(a) periodic kernel recovers Hasimoto")
ax.legend(frameon=False, fontsize=8)
ax = axes[1]
g1 = np.linspace(-1.8, 1.8, 41)
GX, GY = np.meshgrid(g1, g1)
pts = centre + np.column_stack([GX.ravel(), GY.ravel()])
uu = Pk.velocity(pts, ring, ring_forces, U=(1.0, 0.0))
ax.streamplot(GX, GY, uu[:, 0].reshape(GX.shape), uu[:, 1].reshape(GX.shape),
              density=1.2, color="0.45", linewidth=0.7)
ax.add_patch(Circle((0, 0), a_fib, fill=False, lw=1.5, color="C3"))
ax.plot(ring[:, 0] - centre[0], ring[:, 1] - centre[1], "o", ms=4, color="C3")
ax.plot([0, 0], [-1, 1], "C0-", lw=2)
ax.set_aspect("equal")
ax.set_title(f"(b) 8 tiny blobs: net flux {abs(net_flux):.0e}, chord flux {chord_flux:.2f}", fontsize=9)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "stage2_dilute_leak.pdf"))
plt.close(fig)

# ============================================================================
# 6. Schematic of the model (periodic cell, fibres, blobs, mean flow)
# ============================================================================
fig, ax = plt.subplots(figsize=(5.4, 4.2))
rng = np.random.default_rng(7)
centres = rng.random((5, 2)) * 0.8 + 0.1
radii = rng.uniform(0.05, 0.09, 5)
for ox in (-1, 0, 1):
    for oy in (-1, 0, 1):
        main = (ox == 0 and oy == 0)
        if not main:
            ax.add_patch(Rectangle((ox, oy), 1, 1, fill=False, lw=0.6, ls="--", color="0.6"))
        for c, rad in zip(centres, radii):
            ax.add_patch(Circle((c[0] + ox, c[1] + oy), rad, color="C0" if main else "0.8",
                                alpha=0.9 if main else 0.6))
            if main:
                t = 2 * np.pi * np.arange(16) / 16
                ax.plot(c[0] + rad * np.cos(t), c[1] + rad * np.sin(t), "o", ms=1.8, color="C3")
ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1.8, color="k", zorder=5))   # main cell on top
ax.annotate("", xy=(-0.35, 0.5), xytext=(-0.85, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(-0.95, 0.58, "mean flow $U$", fontsize=9)
ax.text(0.02, 1.04, "computational cell $L_x \\times L_y$", fontsize=9)
ax.text(1.05, 1.04, "periodic copies (grey)", fontsize=8, color="0.4")
ax.text(-0.95, -0.32, "red dots: regularized Stokeslets placed on each fibre surface", fontsize=8, color="C3")
ax.set_xlim(-1.0, 2.0); ax.set_ylim(-0.45, 1.6)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "stage2_model.pdf"))
plt.close(fig)

with open(os.path.join(RESULTS, "summary.txt"), "w") as fh:
    fh.write("\n".join(summary) + "\n")
print("\n".join(summary))
print("Wrote results/stage2 and paper/figures/stage2_*.pdf")
