"""
stage4_validation.py
====================
Stage 4, part 1: validate our 3D regularized Stokeslet (RS) solver.

1. The 3D kernel (rs_kernels3d.py) and its triply periodic Ewald sum (periodic3d.py):
   splitting independence, agreement with the plain Fourier sum, far field, centre values,
   series/closed-form switch, and the forced Stokes equations by finite differences.
2. Lattices with EXACT answers from our 2D reference solvers (reference_bie.py):
     aligned_square  fibres across the flow       (= the 2D problem of Stage 3)
     axial_square    fibres along the flow        (exact: scalar boundary integral, Eq. 7.11)
     tilted          fibres at 45 degrees          (exact: combination of the two, Eq. 7.12)
   at alpha = 0.05-0.3, two blob-width pairs, with and without the pressure balance of
   Proposition 8, and the centreline (one blob per cross-section) model.
   Also: the accuracy of the blob-width pairs on random 2D layouts like those of experiment (b).
3. A crossing network with no exact answer: three perpendicular fibres per cell ("cubic3"):
   convergence in eps, symmetry k_x = k_z, and the centreline model.
   Also: the spacing of the pressure-balance disc points.
4. Timing.

Outputs: results/stage4/*.csv, results/stage4/summary.txt
Run from the project folder:   python scripts/stage4_validation.py      (about 30 minutes)
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import rs_kernels3d as k3                                                # noqa: E402
import geometry3d as g3                                                  # noqa: E402
from periodic3d import PeriodicStokeslet3D, spectral_only_blocks_3d      # noqa: E402
from periodic import PeriodicStokeslet2D                                 # noqa: E402
from solver3d import NetworkSolver3D, CENTRELINE_EPS_FACTOR              # noqa: E402
from reference_bie import reference_k, reference_k_axial                # noqa: E402
import geometry as geo2d                                                 # noqa: E402
from solver2d import richardson_k                                        # noqa: E402

RESULTS = os.path.join(ROOT, "results", "stage4")
os.makedirs(RESULTS, exist_ok=True)
summary = []
timing_rows = []
T0 = time.perf_counter()


def note(line):
    summary.append(line)
    print(line, flush=True)


def save(name, header, rows):
    np.savetxt(os.path.join(RESULTS, name), np.array(rows), delimiter=",", header=header,
               comments="", fmt="%.10e")


asymmetry = []          # |K - K^T| / |K| for every permeability tensor (should be ~ 0)


def solve_k(net, method="surface", eps=0.2, U=(0.0, 0.0, 1.0), balance=True):
    """k / a^2 along U, from the full permeability tensor (Eq. 7.10), and the solver."""
    t0 = time.perf_counter()
    s = NetworkSolver3D(net, method, eps, balance=balance)
    k = s.dimensionless_permeability(U)
    asymmetry.append(np.max(np.abs(s.K - s.K.T)) / np.max(np.abs(s.K)))
    timing_rows.append([s.N, len(s.disk_pts) if s.balance else 0, s.timing["assemble_s"],
                        s.timing["factor_s"], time.perf_counter() - t0])
    return k, s


# ============================================================================
# 1. The 3D kernel
# ============================================================================
rng = np.random.default_rng(2026)
L = 6.0
x = rng.random((60, 3)) * L

# (a) the Ewald sum does not depend on the splitting width xi
blocks = [PeriodicStokeslet3D([L, L, L], 0.3, xi=xi).blocks(x, x) for xi in (L / 4, L / 6, L / 8)]
split_diff = max(np.max(np.abs(b - c)) for B in blocks[1:] for b, c in zip(B, blocks[0]))
scale = max(np.max(np.abs(b)) for b in blocks[0])
note(f"3D Ewald sum: largest change with the splitting width = {split_diff / scale:.1e} (relative)")

# (b) the Ewald sum equals the plain Fourier series (which converges only for a wide blob)
eps_w = 1.5
Bw = PeriodicStokeslet3D([L, L, 1.3 * L], eps_w).blocks(x, x)
Sw = spectral_only_blocks_3d(x, x, [L, L, 1.3 * L], eps_w, n_max=12)
spec_diff = max(np.max(np.abs(b - c)) for b, c in zip(Bw, Sw)) / max(np.max(np.abs(b)) for b in Bw)
note(f"3D Ewald sum vs plain Fourier series (eps = {eps_w}, cell L x L x 1.3L): "
     f"largest relative difference = {spec_diff:.1e}")

# (c) far field: Stokeslet + dipole of strength eps^2/2 (Eq. 7.3)
eps = 0.5
r = np.array([6.0, 8.0, 10.0]) * eps
H1, H2 = k3.kernel_H_3d(r, eps)
far1 = 1 / r + eps**2 / (2 * r**3)
far2 = 1 / r**3 - 1.5 * eps**2 / r**5
far_diff = max(np.max(np.abs(H1 / far1 - 1)), np.max(np.abs(H2 / far2 - 1)))
note(f"3D kernel far field (r >= 6 eps) vs Stokeslet + dipole: largest relative difference = {far_diff:.1e}")

# (d) centre values and the series/closed-form switch at rho = 0.25
H1_0, H2_0 = k3.kernel_H_3d(np.array([0.0]), eps)
centre_err = max(abs(H1_0[0] * eps * np.sqrt(np.pi) * 3 / 8 - 1), abs(H2_0[0] * eps**3 * np.sqrt(np.pi) * 15 / 8 - 1))
# just below the switch the series is used; compare it with the closed form at the SAME point
rho_s = k3.RHO_SERIES * (1 - 1e-9)
rs = np.array([rho_s * eps])
E, G = k3.erf(rho_s), eps * np.exp(-rho_s**2) / np.sqrt(np.pi)
closed = np.array([E / rs + eps**2 * E / (2 * rs**3) - G / rs**2,
                   E / rs**3 - 1.5 * eps**2 * E / rs**5 + 3 * G / rs**4]).ravel()
X = np.exp(-rho_s**2) / np.sqrt(np.pi)
closed_d = np.array([-2 * X / eps**3 - X / (rs**2 * eps) + E / (2 * rs**3),
                     2 * X / (rs**2 * eps**3) + 3 * X / (rs**4 * eps) - 1.5 * E / rs**5]).ravel()
jump = max(np.max(np.abs(np.array(k3.kernel_H_3d(rs, eps)).ravel() / closed - 1)),
           np.max(np.abs(np.array(k3.dH_ds2_3d(rs, eps)).ravel() / closed_d - 1)))
note(f"3D kernel: centre values H1(0) = 8/(3 sqrt(pi) eps), H2(0) = 8/(15 sqrt(pi) eps^3) to {centre_err:.1e}; "
     f"series vs closed form at the switch: {jump:.1e}")

# (e) forced Stokes equations by finite differences: mu lap u - grad p + f phi = 0, div u = 0
f0 = np.array([[0.7, -0.4, 0.3]])
src = np.zeros((1, 3))
h = 1e-3
tests = np.array([[0.3, 0.1, -0.2], [0.6, -0.5, 0.4], [1.1, 0.2, 0.3], [0.05, 0.02, 0.01]])
worst_div = worst_res = 0.0
for p in tests:
    def u_at(q):
        return k3.velocity_3d(np.atleast_2d(q), src, f0, eps)[0]
    lap = np.zeros(3)
    div = 0.0
    gradp = np.zeros(3)
    for c in range(3):
        e = np.zeros(3)
        e[c] = h
        lap += (u_at(p + e) - 2 * u_at(p) + u_at(p - e)) / h**2
        div += (u_at(p + e)[c] - u_at(p - e)[c]) / (2 * h)
        gradp[c] = (k3.pressure_3d(np.atleast_2d(p + e), src, f0, eps)[0]
                    - k3.pressure_3d(np.atleast_2d(p - e), src, f0, eps)[0]) / (2 * h)
    force = f0[0] * k3.blob_3d(np.linalg.norm(p), eps)
    res = lap - gradp + force
    size = max(np.max(np.abs(lap)), np.max(np.abs(gradp)), np.max(np.abs(force)))
    worst_res = max(worst_res, np.max(np.abs(res)) / size)
    worst_div = max(worst_div, abs(div) / np.max(np.abs(u_at(p))))
note(f"3D kernel satisfies the forced Stokes equations: residual / largest term <= {worst_res:.1e}, "
     f"|div u| / |u| <= {worst_div:.1e} (finite differences, h = {h})")

with open(os.path.join(RESULTS, "kernel_checks.txt"), "w") as fh:
    fh.write("\n".join(summary) + "\n")

# ============================================================================
# 2. Lattices with exact answers
# ============================================================================
ALPHAS = [0.05, 0.1, 0.2, 0.3]
PAIRS = [0.2, 0.3]          # fine blob width; the coarse one is twice as wide


def exact_k(kind, alpha):
    """Exact k/a^2 of the lattice from the 2D reference solvers (M = 128 points per fibre)."""
    net = g3.lattice_network(kind, alpha)
    Lc = net["L"]
    if kind == "aligned_square":            # fibres along x, flow along z: 2D transverse problem
        lay = dict(centres=np.array([[Lc / 2, Lc / 2]]), radii=np.array([1.0]), Lx=Lc, Ly=Lc)
        return reference_k(lay, 128)
    if kind == "axial_square":              # fibres along z, flow along z: 2D axial problem
        lay = dict(centres=np.array([[Lc / 2, Lc / 2]]), radii=np.array([1.0]), Lx=Lc, Ly=Lc)
        return reference_k_axial(lay, 128)
    # tilted: fibre along (1,0,1). Its copies form a rectangular array, Lc/sqrt(2) along
    # e = (1,0,-1)/sqrt(2) by Lc along y. The flow direction z = (t - e)/sqrt(2) is half
    # along the fibre and half across it (along e), so (Eq. 7.12)
    #       1/k = (1/2)/k_across + (1/2)/k_along
    lay = dict(centres=np.array([[Lc / np.sqrt(8), Lc / 2]]), radii=np.array([1.0]),
               Lx=Lc / np.sqrt(2), Ly=Lc)
    k_across = reference_k(lay, 128, U=(1.0, 0.0))
    k_along = reference_k_axial(lay, 128)
    return 1.0 / (0.5 / k_across + 0.5 / k_along)


# the axial reference itself, against the dilute series of Drummond & Tahir (1984) for flow along a
# square array:  k_par/a^2 = (ln(1/alpha) - 1.476 + 2 alpha - alpha^2/2) / (4 alpha)
rows = []
for alpha in (0.01, 0.05, 0.1, 0.2, 0.3):
    Lc = np.sqrt(np.pi / alpha)
    lay = dict(centres=np.array([[Lc / 2, Lc / 2]]), radii=np.array([1.0]), Lx=Lc, Ly=Lc)
    k64, k128 = reference_k_axial(lay, 64), reference_k_axial(lay, 128)
    series = (np.log(1 / alpha) - 1.476 + 2 * alpha - alpha**2 / 2) / (4 * alpha)
    rows.append([alpha, k128, series, k128 / series - 1, abs(k64 / k128 - 1)])
save("axial_reference.csv", "alpha,k_par_reference,k_par_series_DT1984,rel_diff,change_M64_to_M128", rows)
A = np.array(rows)
note(f"axial reference vs the Drummond-Tahir series: differences {100 * np.min(np.abs(A[:, 3])):.2f}%-"
     f"{100 * np.max(np.abs(A[:, 3])):.2f}% (alpha 0.01-0.3); change from M = 64 to 128 <= {A[:, 4].max():.1e}")

rows = []
for kind_id, kind in enumerate(["aligned_square", "axial_square", "tilted"]):
    for alpha in ALPHAS:
        net = g3.lattice_network(kind, alpha)
        k_ex = exact_k(kind, alpha)
        kc, _ = solve_k(net, "centreline")
        for e in PAIRS:
            k1, s1 = solve_k(net, "surface", e)
            k2, _ = solve_k(net, "surface", 2 * e)
            kR = (4 * k1 - k2) / 3
            if e == PAIRS[0]:
                # the same without the pressure balance (Proposition 8), also extrapolated
                k_nb1, _ = solve_k(net, "surface", e, balance=False)
                k_nb2, _ = solve_k(net, "surface", 2 * e, balance=False)
                k_nb = (4 * k_nb1 - k_nb2) / 3
                slip = s1.max_slip()
            else:
                k_nb, slip = np.nan, np.nan
            rows.append([kind_id, alpha, e, k_ex, k1 / k_ex - 1, k2 / k_ex - 1, kR / k_ex - 1,
                         k_nb / k_ex - 1, kc / k_ex - 1, slip])
        note(f"{kind:15s} alpha = {alpha:4.2f}: exact k/a^2 = {k_ex:8.4f}; Richardson error "
             f"{100 * rows[-2][6]:+.3f}% (eps 0.2/0.4), {100 * rows[-1][6]:+.3f}% (eps 0.3/0.6); "
             f"without balance {100 * rows[-2][7]:+.2f}%; centreline {100 * rows[-2][8]:+.2f}%")
save("lattice_validation.csv",
     "kind(0=aligned 1=axial 2=tilted),alpha,eps_fine,k_exact,err_eps,err_2eps,err_richardson,"
     "err_richardson_no_balance,err_centreline,max_slip", rows)
R = np.array(rows)
fine = R[R[:, 2] == PAIRS[0]]
coarse = R[R[:, 2] == PAIRS[1]]
upto02 = fine[:, 1] <= 0.2
note(f"lattices: largest Richardson error (eps 0.2/0.4) = {100 * np.max(np.abs(fine[upto02, 6])):.2f}% "
     f"for alpha <= 0.2, {100 * np.max(np.abs(fine[:, 6])):.2f}% up to alpha = 0.3")
note(f"lattices: largest Richardson error (eps 0.3/0.6) = {100 * np.max(np.abs(coarse[coarse[:, 1] <= 0.2, 6])):.2f}% "
     f"for alpha <= 0.2")
ax_tilt = fine[fine[:, 0] > 0]
note(f"pressure balance: without it the axial/tilted Richardson error (eps 0.2/0.4) is up to "
     f"{100 * np.max(np.abs(ax_tilt[ax_tilt[:, 1] <= 0.2, 7])):.1f}% (alpha <= 0.2); with it "
     f"{100 * np.max(np.abs(ax_tilt[ax_tilt[:, 1] <= 0.2, 6])):.2f}%")
note(f"no-slip at the blob centres: largest |u|/U = {np.nanmax(R[:, 9]):.1e}")
note(f"centreline model on lattices: error {100 * np.min(fine[upto02, 8]):+.1f}% to {100 * np.max(fine[upto02, 8]):+.1f}% "
     f"for alpha <= 0.2")

# the flow that remains INSIDE a fibre (Section 7.4, consequence c): axial square array, alpha = 0.1,
# velocity on the fibre axis and the flux inside the fibre, with and without the pressure balance
rows = []
net = g3.lattice_network("axial_square", 0.1)
axis_pt = np.array([[net["L"] / 2, net["L"] / 2, 1.0]])
for e in (0.4, 0.2):
    for bal in (True, False):
        _, s = solve_k(net, "surface", e, balance=bal)        # its last solve is U along z
        rows.append([e, bal, s.velocity(axis_pt)[0, 2], s.inside_flux()[2]])
save("inside_flow.csv", "eps,balance,u_axis_over_U,inside_flux_over_U", rows)
I = np.array(rows, dtype=float)
note(f"flow inside a fibre (axial square, alpha = 0.1), velocity on the axis / U: with balance "
     f"{I[0, 2]:+.3f} (eps 0.4), {I[2, 2]:+.3f} (eps 0.2); without {I[1, 2]:+.3f}, {I[3, 2]:+.3f}; "
     f"inside flux with balance {100 * I[0, 3]:+.2f}% and {100 * I[2, 3]:+.2f}% of U")

# centreline = 2D one-blob model, exactly (Eq. 7.6): aligned square array
alpha = 0.1
net = g3.lattice_network("aligned_square", alpha)
kc3, _ = solve_k(net, "centreline")
P2 = PeriodicStokeslet2D(net["L"], net["L"], CENTRELINE_EPS_FACTOR)
Sxx, _, _ = P2.blocks(np.zeros((1, 2)), np.zeros((1, 2)))
kc2 = Sxx[0, 0] * net["L"] ** 2             # one blob per cell: k = mu A P_xx(0)
note(f"centreline line of 3D blobs vs 2D one-blob model (alpha = 0.1): relative difference "
     f"{abs(kc3 / kc2 - 1):.1e}")

# (the convergence figure is drawn from lattice_validation.csv by scripts/stage4_summary.py)

# ============================================================================
# 2b. Which blob widths can random networks afford? (2D test, cheap)
#     A network with all fibres along x IS a 2D array, so the 2D solver measures the accuracy of a
#     Richardson pair on random 4-fibre layouts with gaps >= 0.6a, like those of experiment (b).
# ============================================================================
rows = []
for alpha in (0.1, 0.15, 0.2):
    rng2 = np.random.default_rng(int(1000 * alpha) + 11)
    for rep in range(12):
        lay = geo2d.random_array(alpha, 4, rng2, cv=0.0, min_gap=0.6)
        k_ex = reference_k(lay, 128)
        errs = [richardson_k(lay, e)[0] / k_ex - 1 for e in (0.2, 0.25, 0.3)]
        rows.append([alpha, geo2d.min_surface_gap(lay)] + errs)
save("pair_accuracy_2d.csv", "alpha,min_gap,err_R_0.2,err_R_0.25,err_R_0.3", rows)
P = np.array(rows)
for alpha in (0.1, 0.15, 0.2):
    sub = P[P[:, 0] == alpha]
    note(f"2D random 4-fibre layouts, alpha = {alpha}: largest Richardson error "
         f"{100 * np.max(np.abs(sub[:, 2])):.2f}% (eps 0.2/0.4), {100 * np.max(np.abs(sub[:, 3])):.2f}% (0.25/0.5), "
         f"{100 * np.max(np.abs(sub[:, 4])):.2f}% (0.3/0.6)")

# ============================================================================
# 3. A crossing network: three perpendicular fibres per cell
# ============================================================================
rows = []
for alpha, widths in [(0.2, [0.4, 0.3, 0.2]), (0.1, [0.6, 0.3])]:
    net = g3.lattice_network("cubic3", alpha)
    ks = {}
    for e in widths:
        ks[e], s = solve_k(net, "surface", e)
    kx = s.dimensionless_permeability((1.0, 0.0, 0.0))
    kc, _ = solve_k(net, "centreline")
    if alpha == 0.2:
        kR = (4 * ks[0.2] - ks[0.4]) / 3
        # three widths: fit k = k0 + c eps^2 to eps = 0.3 and 0.2 as an independent estimate
        k_alt = (ks[0.2] * 0.3**2 - ks[0.3] * 0.2**2) / (0.3**2 - 0.2**2)
    else:
        kR = (4 * ks[0.3] - ks[0.6]) / 3
        k_alt = np.nan
    e_f = min(widths)
    rows.append([alpha, kR, k_alt, ks[e_f], kx, kc])
    note(f"cubic3 alpha = {alpha}: k/a^2 = {kR:.4f} (Richardson)"
         + (f", {k_alt:.4f} (eps^2 fit to 0.3, 0.2): difference {100 * abs(kR / k_alt - 1):.2f}%" if alpha == 0.2 else "")
         + f"; k_x/k_z - 1 = {kx / ks[e_f] - 1:.1e}; centreline {100 * (kc / kR - 1):+.1f}%")
save("cubic3.csv", "alpha,k_richardson,k_eps2_fit,k_z_finest,k_x_finest,k_centreline", rows)

# ============================================================================
# 3b. Spacing of the pressure-balance disc points (default: 2 x the blob spacing)
# ============================================================================
rows = []
for kind, alpha in [("axial_square", 0.1), ("tilted", 0.1), ("tilted", 0.2), ("cubic3", 0.2)]:
    net = g3.lattice_network(kind, alpha)
    kr = []
    for df in (1.0, 2.0):
        ks = []
        for e in (0.2, 0.4):
            s = NetworkSolver3D(net, "surface", e, disk_factor=df)
            ks.append(s.dimensionless_permeability())
        kr.append((4 * ks[0] - ks[1]) / 3)
    k_ex = exact_k(kind, alpha) if kind != "cubic3" else np.nan
    rows.append([alpha, kr[0], kr[1], kr[1] / kr[0] - 1, kr[0] / k_ex - 1, kr[1] / k_ex - 1])
save("disc_spacing.csv", "alpha,k_disc_spacing_1h,k_disc_spacing_2h,rel_change,err_1h,err_2h", rows)
D = np.array(rows)
note(f"disc points 2h instead of h apart: k changes by at most {100 * np.max(np.abs(D[:, 3])):.2f}% "
     f"(axial 0.1, tilted 0.1 and 0.2, cubic3 0.2)")

# ============================================================================
# 4. Timing
# ============================================================================
note(f"permeability tensors are symmetric (as reciprocity requires): largest |K - K^T|/|K| = {max(asymmetry):.1e}")
save("timing.csv", "N_blobs,N_disc_points,assemble_s,factor_s,total_s", timing_rows)
T = np.array(timing_rows)
big = T[np.argmax(T[:, 0])]
note(f"timing: largest system {int(big[0])} blobs ({3 * int(big[0])} unknowns): assembly {big[2]:.0f} s, "
     f"LU factorization {big[3]:.0f} s, whole solve {big[4]:.0f} s")
note(f"total run time {(time.perf_counter() - T0) / 60:.1f} minutes")

with open(os.path.join(RESULTS, "summary.txt"), "w") as fh:
    fh.write("\n".join(summary) + "\n")
