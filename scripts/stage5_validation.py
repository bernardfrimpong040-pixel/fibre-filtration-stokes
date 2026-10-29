"""
stage5_validation.py
====================
Stage 5, part 1: validate the flow fields and the particle tracker used for aerosol capture.

1. The fast flow field (flowfield2d.StreamFunctionField2D) reproduces the RS flow it is built from.
2. Near the fibre the RS flow slips (Section 8.2); the near-wall reconstruction (Proposition 9)
   removes the slip. Both are compared with the EXACT flow from our boundary-integral reference.
3. Pure interception: tracker against the exact stream-function answer, and the small-R law.
4. Straight-line (very heavy) particles: tracker against geometry, particle by particle.
5. Brownian diffusion at high Peclet number: tracker against the Leveque boundary-layer theory
   (Eq. 8.7) built from the computed wall shear.
6. Time-step convergence (with and without the Brownian-bridge correction), and for inertia.
7. Classical comparison: interception on the hexagonal array against Kuwabara's flow (this also
   fixes the efficiency convention, Eq. 8.6), and the high-Peclet diffusion constant on Kuwabara's flow.

Outputs: results/stage5/*.csv, results/stage5/summary.txt, paper/figures/stage5_validation.pdf
Run from the project folder:   python scripts/stage5_validation.py      (about 40 minutes)
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import geometry as geo                                                          # noqa: E402
import stage5_capture as sc                                                     # noqa: E402
import classical_models as cm                                                   # noqa: E402
import particles2d as pt                                                        # noqa: E402
from solver2d import solve_with_rule                                            # noqa: E402
from flowfield2d import StreamFunctionField2D, NearWallField2D, wall_shear, leveque_efficiency  # noqa: E402
from reference_bie import ReferenceBIE2D                                        # noqa: E402

RESULTS = os.path.join(ROOT, "results", "stage5")
FIGURES = os.path.join(ROOT, "paper", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)
summary = []
T0 = time.perf_counter()
EPS = 0.05                   # blob width used for capture (Section 8.2)


def note(line):
    summary.append(line)
    print(line, flush=True)


def save(name, header, rows):
    np.savetxt(os.path.join(RESULTS, name), np.array(rows), delimiter=",", header=header,
               comments="", fmt="%.10e")


def fields(layout, eps=EPS):
    s = solve_with_rule(layout, eps)
    F = StreamFunctionField2D(s, grid_factor=4)
    return s, F, NearWallField2D(F)


sq = geo.square_array(0.1)
hx = geo.hexagonal_array(0.1)
s_sq, F_sq, W_sq = fields(sq)
s_hx, F_hx, W_hx = fields(hx)
rng = np.random.default_rng(2026)

# ============================================================================
# 1. The fast field equals the RS flow
# ============================================================================
lay_r = geo.random_array(0.1, 8, np.random.default_rng(5), min_gap=0.5)
s_r = solve_with_rule(lay_r, EPS)
F_r = StreamFunctionField2D(s_r, grid_factor=4)
worst = 0.0
for s, F, lay in [(s_sq, F_sq, sq), (s_r, F_r, lay_r)]:
    X = rng.random((400, 2)) * [lay["Lx"], lay["Ly"]]
    worst = max(worst, np.max(np.abs(F.velocity(X) - s.velocity(X))))
    i, j = rng.integers(0, F.nx, 60), rng.integers(0, F.ny, 60)
    G = np.column_stack([F.x_grid[i], F.y_grid[j]])
    ug = s.velocity(G)
    spec = max(np.max(np.abs(F.u_grid[i, j] - ug[:, 0])), np.max(np.abs(F.v_grid[i, j] - ug[:, 1])))
note(f"fast field vs direct RS sum: spectral grid values {spec:.1e}, spline anywhere {worst:.1e} (units of U)")

# ============================================================================
# 2. Slip of the RS flow at the fibre, and the near-wall reconstruction
# ============================================================================
rows = []
c = sq["centres"][0]
for eps in (0.1, 0.05):
    s, F, W = fields(sq, eps)
    P = c + np.array([[0.0, 1.0]])                       # top of the fibre, flow along x there
    slip = F.velocity(P)[0, 0]
    rows.append([eps, slip, W.velocity(P)[0, 0]])
    note(f"RS flow at the fibre surface (square, alpha = 0.1, top point): u = {slip:+.4f} U for eps = {eps}a "
         f"(reconstructed: {W.velocity(P)[0, 0]:+.1e})")
save("wall_slip.csv", "eps,u_wall_RS,u_wall_reconstructed", rows)
rows = []
for name, lay, F, W in [("square", sq, F_sq, W_sq), ("hexagonal", hx, F_hx, W_hx)]:
    ref = ReferenceBIE2D(lay, 512)
    ref.solve()
    for d in (0.02, 0.05, 0.1, 0.2, 0.29, 0.5):
        th = rng.random(80) * 2 * np.pi
        m = rng.integers(0, len(lay["radii"]), 80)
        P = lay["centres"][m] + (1 + d) * np.column_stack([np.cos(th), np.sin(th)])
        ur = ref.velocity(P)
        rows.append([len(lay["radii"]), d, np.max(np.abs(W.velocity(P) - ur)), np.max(np.abs(F.velocity(P) - ur))])
save("near_wall_error.csv", "n_fibres(1=square 2=hex),distance,err_reconstructed,err_RS", rows)
R = np.array(rows)
close = R[:, 1] <= 0.1
note(f"velocity error against the exact flow within 0.1a of a fibre (eps = {EPS}a): reconstructed <= "
     f"{R[close, 2].max():.1e} U, RS without reconstruction <= {R[close, 3].max():.1e} U")

# ============================================================================
# 3. Pure interception: tracker vs the exact stream-function answer
# ============================================================================
th_sq, g_sq = wall_shear(W_sq, c, 1.0)
gmax = g_sq.max()
rows = []
for name, lay, W in [("square", sq, W_sq), ("hexagonal", hx, W_hx)]:
    for rp in (0.01, 0.05, 0.2, 0.5):
        ex = pt.interception_exact(W, lay, rp)
        res = pt.track_pass(W, lay, rp, 0.0, 0.0, 0.04, 40000, rng)
        f = res["caught"].mean()
        rows.append([len(lay["radii"]), rp, ex, f, (f - ex) * 40000])
save("interception_check.csv", "n_fibres,r_p,f_exact,f_tracker,difference_in_particles", rows)
R = np.array(rows)
note(f"pure interception, tracker vs exact (40000 particles, 8 cases): largest difference "
     f"{np.max(np.abs(R[:, 4])):.1f} particles (the stratified inlet allows about 1 per interval end)")
small = pt.interception_exact(W_sq, sq, 0.01)
note(f"small-R law (square): caught fraction gamma_max r_p^2 / Ly = {gmax * 0.01**2 / sq['Ly']:.3e} vs exact "
     f"{small:.3e} at r_p = 0.01a; gamma_max = {gmax:.4f} U/a")

# ============================================================================
# 4. Very heavy particles move in straight lines
# ============================================================================
mism = 0
for lay, W in [(sq, W_sq), (hx, W_hx)]:
    res = pt.track_pass(W, lay, 0.2, 0.0, 1e6, 0.02, 4000, rng, v_init="mean")
    mism += int(np.sum(res["caught"] != pt.ballistic_caught(res["start"], lay, 0.2)))
note(f"straight-line particles (St = 1e6): {mism} of 8000 disagree with the geometric prediction")

# ============================================================================
# 5. Brownian diffusion against the Leveque boundary-layer theory
# ============================================================================
rows = []
for Pe, N in [(300, 20000), (1000, 40000), (3000, 100000), (10000, 200000)]:
    D = 2.0 / Pe
    Q = leveque_efficiency(th_sq, g_sq, 1.0, D)
    f_lev = Q / sq["Ly"]
    res = pt.track_pass(W_sq, sq, 0.0, D, 0.0, min(0.04, 0.005 * (D / 4) ** (2 / 3) / D), N, rng)
    f = res["caught"].mean()
    se = np.sqrt(f * (1 - f) / N)
    rows.append([Pe, N, f, se, f_lev, f / f_lev, se / f_lev])
    note(f"Pe = {Pe:5d}: tracker caught {f:.5f} +- {se:.5f}, Leveque {f_lev:.5f}, ratio {f / f_lev:.3f} "
         f"+- {se / f_lev:.3f} ({N} particles)")
save("leveque_check.csv", "Pe,n_particles,f_tracker,se,f_leveque,ratio,ratio_se", rows)
LEV = np.array(rows)

# ============================================================================
# 6. Time-step convergence
# ============================================================================
rows = []
for bridge in (True, False):                            # Brownian, Pe = 100, point particles
    for h in (0.04, 0.02, 0.01, 0.005):
        res = pt.track_pass(W_sq, sq, 0.0, 0.02, 0.0, h, 100000, np.random.default_rng(7), bridge=bridge)
        rows.append([0, h, int(bridge), res["caught"].mean()])
for h in (0.04, 0.02, 0.01):                            # inertia: St = 0.5, r_p = 0.2a, no diffusion
    res = pt.track_pass(W_sq, sq, 0.2, 0.0, 0.5, h, 40000, np.random.default_rng(7))
    rows.append([1, h, 0, res["caught"].mean()])
# the particle model of experiment (c) (stage5_capture.model): the full Langevin equation, h <= St/8.
# (2) 0.25 um WITHOUT inertia (overdamped), h = 0.04, 0.02, 0.01, for comparison; (3) 0.25 um as in
# experiment (c), with its inertia (St = 0.047), h = St/8, St/16, St/32; (4) 0.5 um as in experiment (c)
# (St = 0.15), h = St/4, St/8, St/16; (5) interception with inertia but no diffusion (r_p of 0.25 um,
# St = St_0.25/4, St_0.25, 4 St_0.25, h = St/8) against the exact interception without inertia.
r25, _, _, St25, D25 = sc.particle_parameters(0.25)
r50, _, _, St50, D50 = sc.particle_parameters(0.5)
for case, (rp_, D_, St_, hs) in {2: (r25, D25, 0.0, (0.04, 0.02, 0.01)),
                                 3: (r25, D25, St25, (St25 / 8, St25 / 16, St25 / 32)),
                                 4: (r50, D50, St50, (St50 / 4, St50 / 8, St50 / 16))}.items():
    for h in hs:
        res = pt.track_pass(W_sq, sq, rp_, D_, St_, h, 100000, np.random.default_rng(8))
        rows.append([case, h, 1, res["caught"].mean()])
for St_ in (St25 / 4, St25, 4 * St25):
    res = pt.track_pass(W_sq, sq, r25, 0.0, St_, St_ / 8, 40000, np.random.default_rng(8))
    rows.append([5, St_, 0, res["caught"].mean()])            # column 2 holds St for this case
# (6) case 5 at St = St_0.25 with smaller steps h = St/16, St/32 (is the loss a step error?)
for h in (St25 / 16, St25 / 32):
    res = pt.track_pass(W_sq, sq, r25, 0.0, St25, h, 40000, np.random.default_rng(8))
    rows.append([6, h, 0, res["caught"].mean()])
# (7) the kinetic boundary layer alone: an absorbing wall seen by a Brownian particle with inertia acts
# like a wall moved back by a distance of order lambda = sqrt(D St), the distance over which a particle
# keeps its thermal velocity (about 1.4 lambda, measured in still air in 6b below). So run the
# overdamped tracker at 0.25 um with the capture circle pulled in by 1.03 lambda and 1.5 lambda.
LAM25 = np.sqrt(D25 * St25)
for shift in (1.03 * LAM25, 1.5 * LAM25):
    res = pt.track_pass(W_sq, sq, r25 - shift, D25, 0.0, 0.01, 100000, np.random.default_rng(8))
    rows.append([7, shift, 1, res["caught"].mean()])        # column 2 holds the shift for this case
save("step_convergence.csv", "case(0=Brownian Pe=100,1=inertia St=0.5,2=0.25um overdamped,"
     "3=0.25um with inertia,4=0.5um with inertia,5=interception with inertia: h column holds St,"
     "6=case 5 at St_0.25 with h = St/16 and St/32,7=0.25um overdamped with the wall moved: h column holds the shift)"
     ",h,bridge,f_caught", rows)
S = np.array(rows)
b = S[(S[:, 0] == 0) & (S[:, 2] == 1), 3]
nb = S[(S[:, 0] == 0) & (S[:, 2] == 0), 3]
ine = S[S[:, 0] == 1, 3]
se = np.sqrt(b[-1] * (1 - b[-1]) / 100000)
note(f"Brownian (Pe = 100, 100000 particles, standard error {se:.4f}), h = 0.04, 0.02, 0.01, 0.005: caught "
     f"{b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}, {b[3]:.4f} with the bridge correction; "
     f"{nb[0]:.4f}, {nb[1]:.4f}, {nb[2]:.4f}, {nb[3]:.4f} without")
note(f"inertia (St = 0.5, r_p = 0.2a, 40000 particles), h = 0.04, 0.02, 0.01: caught {ine[0]:.5f}, {ine[1]:.5f}, "
     f"{ine[2]:.5f}")
c2, c3, c4, c5 = (S[S[:, 0] == k, 3] for k in (2, 3, 4, 5))
se2 = np.sqrt(c2[-1] * (1 - c2[-1]) / 100000)
note(f"0.25 um without inertia (overdamped, 100000 particles, standard error {se2:.5f}), h = 0.04, 0.02, 0.01: "
     f"caught {c2[0]:.5f}, {c2[1]:.5f}, {c2[2]:.5f}")
note(f"0.25 um as in experiment (c), with inertia (St = {St25:.4f}), h = St/8, St/16, St/32: caught {c3[0]:.5f}, "
     f"{c3[1]:.5f}, {c3[2]:.5f}; relative to the value without inertia {c3[2] / c2[-1] - 1:+.3f}")
note(f"0.5 um as in experiment (c) (St = {St50:.4f}), h = St/4, St/8, St/16: caught {c4[0]:.5f}, "
     f"{c4[1]:.5f}, {c4[2]:.5f}")
f_int = pt.interception_exact(W_sq, sq, r25)
note(f"interception with inertia, no diffusion (r_p = {r25:.4f}a, 40000 particles), St = {St25 / 4:.4f}, {St25:.4f}, "
     f"{4 * St25:.4f}: caught relative to exact interception without inertia {c5[0] / f_int - 1:+.3f}, "
     f"{c5[1] / f_int - 1:+.3f}, {c5[2] / f_int - 1:+.3f}")
c6, c7 = S[S[:, 0] == 6, 3], S[S[:, 0] == 7, 3]
note(f"interception with inertia at St = {St25:.4f}, h = St/8, St/16, St/32: caught relative to exact "
     f"{c5[1] / f_int - 1:+.3f}, {c6[0] / f_int - 1:+.3f}, {c6[1] / f_int - 1:+.3f}")
se3 = np.sqrt(c3[-1] * (1 - c3[-1]) / 100000)
note(f"kinetic boundary layer at 0.25 um: lambda = sqrt(D St) = {LAM25:.4f}a, delta = (D/4)^(1/3) = "
     f"{(D25 / 4) ** (1 / 3):.3f}a; overdamped with the capture circle pulled in by 1.03 lambda and 1.5 lambda: "
     f"caught {c7[0]:.5f}, {c7[1]:.5f}, relative to the value without inertia {c7[0] / c2[-1] - 1:+.3f}, "
     f"{c7[1] / c2[-1] - 1:+.3f} (standard error of each value {se3:.5f}; with inertia {c3[2] / c2[-1] - 1:+.3f})")

# 6b. The inertial step next to an absorbing wall in still air (no flow): Brownian particles with
# inertia start at distance d from a wall at x = 0. Far from the wall they diffuse as usual, but the
# wall acts like an overdamped absorbing wall moved back by a distance of order lambda (the Milne problem
# of kinetic theory), so the fraction absorbed by time T is about erfc((d + shift) / sqrt(4 D T)).
# We measure that shift.
from scipy.special import erfc, erfcinv                                         # noqa: E402
D_w, St_w, d_w, T_w, N_w = 1.0, 1.0, 20.0, 100.0, 100000                         # lambda = 1 here
rng_w = np.random.default_rng(11)
shifts = []
for h in (St_w / 8, St_w / 32):
    e1 = np.exp(-h / St_w)
    sx2, sxv, sv2 = pt.variances(h, St_w, D_w)
    sx = np.sqrt(sx2)
    a_v = sxv / sx
    b_v = np.sqrt(sv2 - a_v**2)
    X = np.full(N_w, d_w)
    V = rng_w.standard_normal(N_w) * np.sqrt(D_w / St_w)     # thermal velocities
    alive = np.ones(N_w, bool)
    for _ in range(int(round(T_w / h))):
        Z1, Z2 = rng_w.standard_normal(N_w), rng_w.standard_normal(N_w)
        X, V = X + V * St_w * (1 - e1) + sx * Z1, V * e1 + a_v * Z1 + b_v * Z2   # Eq. 8.4 with u = 0
        alive &= X > 0
    f_w = 1 - alive.mean()
    shifts.append(np.sqrt(4 * D_w * T_w) * erfcinv(f_w) - d_w)
note(f"still air next to an absorbing wall ({N_w} particles): absorbed fraction corresponds to a wall moved back by "
     f"{shifts[0]:.2f} lambda (h = St/8) and {shifts[1]:.2f} lambda (h = St/32); "
     f"overdamped (no shift) {erfc(d_w / np.sqrt(4 * D_w * T_w)):.4f}")

# ============================================================================
# 7. Interception on the hexagonal array vs Kuwabara's cell model
# ============================================================================
alpha = 0.1
Ku = cm.kuwabara_factor(alpha)
rows = []
for rp in (0.01, 0.05, 0.1, 0.2):
    r = 1 + rp
    # Kuwabara's stream function (U = 1, a = 1) at r = 1 + r_p, theta = 90 degrees; the fibre catches
    # the flux between the streamlines +psi and -psi, so eta = 2 psi / (U d_f) = psi
    psi_K = (r * (2 * np.log(r) - 1 + alpha) + (1 - alpha / 2) / r - alpha * r**3 / 2) / (2 * Ku)
    eta_K = psi_K
    eta_LL = float(cm.eta_interception(2 * rp, 2.0, alpha))      # Lee & Liu formula used in Stage 1
    f = pt.interception_exact(W_hx, hx, rp)
    eta_hex = pt.efficiency_from_pass(f, hx)                    # flux / (U d_f), Eq. (8.6)
    rows.append([rp, eta_hex, eta_K, eta_LL])
save("kuwabara_interception.csv", "r_p,eta_hex,eta_kuwabara_exact,eta_lee_liu", rows)
K = np.array(rows)
note(f"interception, eta (Eq. 8.6) on the hexagonal array / Kuwabara cell: "
     + ", ".join(f"{r[1] / r[2]:.3f} (r_p = {r[0]:g}a)" for r in K))
# Leveque on Kuwabara's flow: wall shear 2 (1-alpha)/Ku sin(theta) gives eta_D = C Pe^(-2/3), with C to compare
# with the classical Natanson/Stechkina-Fuchs constant 2.9 Ku^(-1/3) (as used for Kuwabara flow)
thK = (np.arange(3600) + 0.5) * 2 * np.pi / 3600
gK = 2 * (1 - alpha) / Ku * np.sin(thK)
C_K = leveque_efficiency(thK, gK, 1.0, 2.0) / 2.0             # D = 2/Pe with Pe = 1; eta = Q / (U d_f)
note(f"Leveque on Kuwabara's flow: eta_D = {C_K:.3f} Pe^(-2/3) = {C_K * Ku**(1 / 3):.3f} Ku^(-1/3) Pe^(-2/3) "
     f"(classical high-Pe constant 2.9 Ku^(-1/3); Lee and Liu's E_D uses {2.6 * ((1 - alpha) / Ku)**(1 / 3):.3f})")

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
R2 = np.loadtxt(os.path.join(RESULTS, "near_wall_error.csv"), delimiter=",", skiprows=1)
for nf, mk, lab in [(1, "o", "square"), (2, "s", "hexagonal")]:
    sub = R2[R2[:, 0] == nf]
    ax[0].semilogy(sub[:, 1], sub[:, 3], mk + "--", mfc="none", label=f"RS flow, {lab}")
    ax[0].semilogy(sub[:, 1], sub[:, 2], mk + "-", label=f"with near-wall reconstruction, {lab}")
ax[0].axvline(0.3, color="0.6", lw=0.8)
ax[0].set_xlabel("distance from the fibre surface / $a$")
ax[0].set_ylabel("largest velocity error / $U$")
ax[0].set_title("(a) flow next to a fibre ($\\varepsilon = 0.05a$)", fontsize=9)
ax[0].legend(fontsize=7)
ax[1].errorbar(LEV[:, 0], LEV[:, 5], yerr=1.96 * LEV[:, 6], fmt="o", capsize=3)
ax[1].axhline(1.0, color="k", lw=0.6)
ax[1].set_xscale("log")
ax[1].set_xlabel("Péclet number $Pe = d_f U/D$")
ax[1].set_ylabel("tracker / Lévêque theory")
ax[1].set_title("(b) Brownian capture at high $Pe$ (95% CI)", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "stage5_validation.pdf"))
plt.close(fig)

note(f"total run time {(time.perf_counter() - T0) / 60:.1f} minutes")
with open(os.path.join(RESULTS, "summary.txt"), "w") as fh:
    fh.write("\n".join(summary) + "\n")
