"""
check_stage5.py
===============
Every Stage 5 check must PASS, the Stage 1-4 checks must still pass,
and every check must FAIL on a deliberately broken input (one fault at a time).

Run from the project folder (after scripts/stage5_validation.py, scripts/stage5_capture.py and
scripts/stage5_summary.py):
    python checks/check_stage5.py          (about 8 minutes)
Exit code 0 = all good; 1 = something failed.
"""

import csv
import glob
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
import geometry as geo                                                          # noqa: E402
import particles2d as pt                                                        # noqa: E402
import flowfield2d as ff                                                        # noqa: E402
from solver2d import solve_with_rule                                            # noqa: E402
from reference_bie import ReferenceBIE2D                                        # noqa: E402
import check_stage4                                                             # noqa: E402

TEX = os.path.join(ROOT, "paper", "main.tex")
RES = os.path.join(ROOT, "results", "stage5")
SUMMARY = os.path.join(RES, "summary.txt")
CAP_SUMMARY = os.path.join(RES, "capture_summary.txt")

_cache = {}


def square_fields():
    """RS solution, fast field and near-wall field of the square array at alpha = 0.1 (cached)."""
    if "sq" not in _cache:
        lay = geo.square_array(0.1)
        s = solve_with_rule(lay, 0.05)
        F = ff.StreamFunctionField2D(s, grid_factor=4)
        _cache["sq"] = (lay, s, F, ff.NearWallField2D(F))
    return _cache["sq"]


# ============================================================================
# The checks. Each returns (passed: bool, message: str).
# ============================================================================

def check_fast_field(field_class=ff.StreamFunctionField2D):
    """Eq. (8.2): the FFT stream-function field reproduces the RS flow: on the grid to round-off,
    and anywhere through the spline to 1e-4 U."""
    lay, s, _, _ = square_fields()
    F = field_class(s, grid_factor=4)
    rng = np.random.default_rng(1)
    i, j = rng.integers(0, F.nx, 40), rng.integers(0, F.ny, 40)
    ug = s.velocity(np.column_stack([F.x_grid[i], F.y_grid[j]]))
    grid = max(np.max(np.abs(F.u_grid[i, j] - ug[:, 0])), np.max(np.abs(F.v_grid[i, j] - ug[:, 1])))
    X = rng.random((200, 2)) * [lay["Lx"], lay["Ly"]]
    anywhere = np.max(np.abs(F.velocity(X) - s.velocity(X)))
    return grid < 1e-12 and anywhere < 1e-4, f"grid {grid:.1e}, anywhere {anywhere:.1e}"


def check_divergence_free(velocity=None):
    """The interpolated flow is exactly divergence-free (derivatives of one spline): central
    differences of the velocity give div u below 1e-6 at random points."""
    lay, _, F, _ = square_fields()
    vel = velocity or F.velocity
    rng = np.random.default_rng(2)
    X = rng.random((200, 2)) * [lay["Lx"], lay["Ly"]]
    h = 1e-4
    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    div = (vel(X + ex)[:, 0] - vel(X - ex)[:, 0] + vel(X + ey)[:, 1] - vel(X - ey)[:, 1]) / (2 * h)
    return np.max(np.abs(div)) < 1e-6, f"largest |div u| = {np.max(np.abs(div)):.1e}"


def check_near_wall(n_modes=64):
    """Proposition 9: the reconstructed flow is at rest on the fibre, continuous at r_out, and within
    1e-3 U of the exact (boundary-integral) flow 0.05a from the surface (jump at r_out below 1e-5 U), where the RS flow itself is
    off by several times more."""
    lay, s, F, _ = square_fields()
    W = ff.NearWallField2D(F, n_modes=n_modes)
    c = lay["centres"][0]
    th = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    ring = np.column_stack([np.cos(th), np.sin(th)])
    wall = np.max(np.abs(W.velocity(c + ring)))
    jump = np.max(np.abs(W.velocity(c + (1.3 - 1e-9) * ring) - F.velocity(c + (1.3 + 1e-9) * ring)))
    ref = ReferenceBIE2D(lay, 512)
    ref.solve()
    P = c + 1.05 * ring
    err = np.max(np.abs(W.velocity(P) - ref.velocity(P)))
    err_rs = np.max(np.abs(F.velocity(P) - ref.velocity(P)))
    ok = wall < 1e-10 and jump < 1e-5 and err < 1e-3 and err_rs > 3 * err
    return ok, f"wall {wall:.1e}, jump at r_out {jump:.1e}, error {err:.1e} (RS alone {err_rs:.1e})"


def check_interception(r_p_tracker=0.05):
    """Pure interception: the tracker (20000 particles) agrees with the exact stream-function answer
    for r_p = 0.05a to within 3 particles."""
    lay, _, _, W = square_fields()
    ex = pt.interception_exact(W, lay, 0.05)
    res = pt.track_pass(W, lay, r_p_tracker, 0.0, 0.0, 0.04, 20000, np.random.default_rng(3))
    diff = abs(res["caught"].mean() - ex) * 20000
    return diff <= 3, f"difference {diff:.1f} particles"


def check_ballistic(v_init="mean"):
    """Very heavy particles (St = 1e6) launched along x move in straight lines: every one is caught
    exactly when the geometry says so."""
    lay, _, _, W = square_fields()
    res = pt.track_pass(W, lay, 0.2, 0.0, 1e6, 0.02, 2000, np.random.default_rng(4), v_init=v_init)
    bad = int(np.sum(res["caught"] != pt.ballistic_caught(res["start"], lay, 0.2)))
    return bad == 0, f"{bad} of 2000 disagree"


def check_variances(var=pt.variances):
    """The exact Ornstein-Uhlenbeck step (Eq. 8.4): 20 steps of h must have the same position and
    velocity variance and covariance as ONE step of 20 h (composition), within 2% (200000 samples),
    and a very long step must give the thermal velocity variance D/St and the diffusive 2 D h."""
    St, D, h, n = 0.3, 0.01, 0.05, 200000
    rng = np.random.default_rng(5)
    sx2, sxv, sv2 = var(h, St, D)
    sx = np.sqrt(sx2)
    av = sxv / sx
    bv = np.sqrt(sv2 - av**2)
    e = np.exp(-h / St)
    X = np.zeros(n)
    V = np.zeros(n)
    for _ in range(20):
        z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
        X, V = X + V * St * (1 - e) + sx * z1, V * e + av * z1 + bv * z2
    tx2, txv, tv2 = var(20 * h, St, D)                          # the reference: one big step
    rel = [abs(np.var(X) / tx2 - 1), abs(np.mean(X * V) / txv - 1), abs(np.var(V) / tv2 - 1)]
    # and the physical limits for a long step: thermal velocity variance D/St, diffusion 2 D h
    big = var(1e4 * St, St, D)
    lim = max(abs(big[2] * St / D - 1), abs(big[0] / (2 * D * 1e4 * St) - 1))
    return max(rel) < 0.02 and lim < 1e-3, f"composition {max(rel):.3f}, long-step limits {lim:.1e}"


def check_bridge(fn=pt.bridge_probability):
    """Eq. (8.5) against a direct simulation: Brownian paths (variance s2 per step, 400 sub-steps)
    from distance d0 that END near d1 touched the wall with frequency exp(-2 d0 d1 / s2)."""
    rng = np.random.default_rng(6)
    d0, d1, s2, n, m = 0.6, 0.6, 1.0, 400000, 400
    X = np.full(n, d0)
    touched = np.zeros(n, bool)
    for _ in range(m):
        X = X + np.sqrt(s2 / m) * rng.standard_normal(n)
        touched |= X <= 0
    end = np.abs(X - d1) < 0.05
    sim = touched[end].mean()
    # the discrete walk misses some touches; correct to continuous time with the classical
    # Siegmund shift of the wall by 0.5826 sqrt(s2/m)
    beta = 0.5826 * np.sqrt(s2 / m)
    theory = fn(d0 + beta, d1 + beta, s2)
    return abs(sim / theory - 1) < 0.05, f"simulated {sim:.4f}, formula {theory:.4f}"


def check_leveque_formula(fn=ff.leveque_efficiency):
    """Eq. (8.7) against a hand calculation: wall shear gamma = g0 |sin theta| on a unit circle gives
    I = sqrt(g0) * 2.39628 per side, and with Leveque's tabulated constant 0.5384 for the local flux
    q = 0.5384 D (gamma / (D x))^(1/3) (flat wall, constant shear), Q = 2 x (3/2) 0.5384 D^(2/3) I^(2/3)."""
    th = (np.arange(3600) + 0.5) * 2 * np.pi / 3600
    g0, D = 4.0, 1e-3
    gam = g0 * np.sin(th)                    # positive on top, negative below: two segments
    I = np.sqrt(g0) * 2.3962804694711844
    Q_hand = 2 * 1.5 * 0.5384 * D ** (2 / 3) * I ** (2 / 3)
    Q = fn(th, gam, 1.0, D)
    return abs(Q / Q_hand - 1) < 2e-4, f"Q = {Q:.6e}, hand {Q_hand:.6e}"


def check_efficiency_conversion(fn=pt.efficiency_from_pass):
    """Eq. (8.6) has the convention of the Kuwabara-based formulas (caught flux / (U d_f)): on the
    hexagonal array (alpha = 0.1, which Kuwabara's cell describes well) the eta of pure interception
    for r_p = 0.05a is within 3% of Kuwabara's exact value; and P_cell^(L/Lx) = exp(-4 alpha eta L/(pi d_f))."""
    from classical_models import kuwabara_factor
    lay = geo.hexagonal_array(0.1)
    s = solve_with_rule(lay, 0.05)
    W = ff.NearWallField2D(ff.StreamFunctionField2D(s, grid_factor=4))
    f = pt.interception_exact(W, lay, 0.05)
    eta = fn(f, lay)
    r, al, Ku = 1.05, 0.1, kuwabara_factor(0.1)
    eta_K = (r * (2 * np.log(r) - 1 + al) + (1 - al / 2) / r - al * r**3 / 2) / (2 * Ku)
    lhs = (1 - f) ** (37.0 / lay["Lx"])
    rhs = np.exp(-4 * 0.1 * eta * 37.0 / (np.pi * 2.0))
    ok = abs(eta / eta_K - 1) < 0.03 and abs(lhs / rhs - 1) < 1e-12
    return ok, f"eta / Kuwabara = {eta / eta_K:.4f}; stacking {lhs:.6f} vs {rhs:.6f}"


def check_validation_results(path=RES):
    """stage5_validation.py results within tolerance: interception within 3 particles in all cases,
    straight-line particles all correct, near-wall error <= 1e-3 U within 0.1a, the weighted mean of
    the Leveque ratios for Pe >= 1000 within 3 standard errors of 1, with the bridge correction the
    Brownian caught fraction changes by < 3% from h = 0.01 to 0.005, the inertial one by < 1%, and the
    production step h = St/8 within 3% of the smallest step at 0.25 um (St/32) and 0.5 um (St/16); interception on the hexagonal array
    within 2% of Kuwabara's, and the small-R law within 1.5%."""
    I = np.loadtxt(os.path.join(path, "interception_check.csv"), delimiter=",", skiprows=1)
    N = np.loadtxt(os.path.join(path, "near_wall_error.csv"), delimiter=",", skiprows=1)
    L = np.loadtxt(os.path.join(path, "leveque_check.csv"), delimiter=",", skiprows=1)
    S = np.loadtxt(os.path.join(path, "step_convergence.csv"), delimiter=",", skiprows=1)
    s = open(os.path.join(path, "summary.txt"), encoding="utf-8").read()
    ballistic = int(re.search(r"\(St = 1e6\): (\d+) of", s).group(1))
    K = np.loadtxt(os.path.join(path, "kuwabara_interception.csv"), delimiter=",", skiprows=1)
    hi = L[L[:, 0] >= 1000]
    w = 1 / hi[:, 6] ** 2                                         # weighted mean of the ratios
    mean, se = np.sum(w * hi[:, 5]) / np.sum(w), 1 / np.sqrt(np.sum(w))
    lev_ok = abs(mean - 1) <= 3 * se
    b = S[(S[:, 0] == 0) & (S[:, 2] == 1), 3]
    ine = S[S[:, 0] == 1, 3]
    c3, c4 = S[S[:, 0] == 3, 3], S[S[:, 0] == 4, 3]
    step_b, step_i = abs(b[-2] / b[-1] - 1), abs(ine[0] / ine[-1] - 1)
    # the production step h = St/8 against the smallest step: 0.25 um (St/32) and 0.5 um (St/16)
    step_a = max(abs(c3[0] / c3[-1] - 1), abs(c4[1] / c4[-1] - 1))
    kuw = np.max(np.abs(K[:, 1] / K[:, 2] - 1))
    small = re.search(r"r_p\^2 / Ly = (\S+) vs exact (\S+) at", s)
    small_ok = abs(float(small.group(1)) / float(small.group(2)) - 1) < 0.015
    ok = (np.max(np.abs(I[:, 4])) <= 3 and ballistic == 0 and N[N[:, 1] <= 0.1, 2].max() <= 1e-3 and lev_ok
          and step_b < 0.03 and step_i < 0.01 and step_a < 0.03 and kuw < 0.02 and small_ok)
    return ok, (f"interception {np.max(np.abs(I[:, 4])):.1f} particles, ballistic {ballistic}, near-wall "
                f"{N[N[:, 1] <= 0.1, 2].max():.1e}, Leveque mean ratio {mean:.3f} +- {se:.3f}, step changes "
                f"{step_b:.3f} / {step_i:.3f} / {step_a:.3f}, Kuwabara {kuw:.3f}, small-R ok = {small_ok}")


def load_capture():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, "capture_*.csv"))):
        if not p.endswith("capture_summary.csv"):
            with open(p, newline="") as fh:
                rows += list(csv.DictReader(fh))
    return rows


def check_capture_results(rows=None):
    """Experiment (c) complete: (1 + 1 + 6 + 6) cells x 11 sizes = 154 rows, at most 1 particle in 100000
    left in a cell at the time limit (counted as not caught), 0 < eta, and the tracker passed its interception check on every cell:
    deviation from the exact answer below 3.5 standard deviations of the stratified inlet's
    resolution."""
    rows = load_capture() if rows is None else rows
    chk = []
    for p in sorted(glob.glob(os.path.join(RES, "interception_checks_*.csv"))):
        with open(p, newline="") as fh:
            chk += list(csv.DictReader(fh))
    keys = {(r["structure"], r["seed"], r["d_p_um"]) for r in rows}
    nf = {(r["structure"], r["seed"]): int(r["n_fibres"]) for r in rows}
    # stratified inlet: each end of a fibre's band of caught streamlines falls inside one particle's
    # flux band, which is caught with probability = the covered part q of that band; for q uniform
    # this has variance 1/6, so the count has standard deviation sqrt(2 n_fibres / 6)
    z = [abs(float(c["f_tracker"]) - float(c["f_exact"])) * float(c["n_particles"])
         / np.sqrt(2 * nf[(c["structure"], c["seed"])] / 6) for c in chk]
    stuck = sum(int(r["n_stuck"]) for r in rows)
    tracked = sum(int(r["n_particles"]) for r in rows)
    ok = (len(rows) == 154 and len(keys) == 154 and len(chk) == 14 and stuck <= 1e-5 * tracked and max(z) < 3.5
          and all(float(r["eta"]) > 0 for r in rows))
    return ok, f"{len(rows)} rows, {len(chk)} cell checks (largest z-score {max(z):.2f}), {stuck} of {tracked} particles stuck"


def tex_sci(x):
    m, e = f"{float(x):.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def expected_table_rows():
    """Rows of Table 8 (capture summary at the MPPS) exactly as main.tex must print them."""
    rows = []
    txt = open(CAP_SUMMARY, encoding="utf-8").read()
    for g in re.findall(r"^(.*?) \(\d+ cells?\): k/a\^2 = ([\d.]+), pressure drop ([\d.]+) Pa; MPPS ([\d.]+) um, "
                        r"eta there ([\d.]+), penetration ([\d.]+), q_F at the MPPS (\S+) 1/Pa", txt, re.M):
        rows.append(f"{g[0]} & {g[1]} & {g[2]} & {g[3]} & {g[4]} & {g[5]} & ${tex_sci(g[6])}$ \\\\")
    g = re.search(r"classical \(Stage 1, Kuwabara-based formulas\): MPPS ([\d.]+) um, penetration ([\d.]+), "
                  r"pressure drop ([\d.]+) Pa, q_F (\S+) 1/Pa", txt)
    rows.append(f"classical (Stage 1) & -- & {g.group(3)} & {g.group(1)} & -- & {g.group(2)} & "
                f"${tex_sci(g.group(4))}$ \\\\")
    # the same classical formulas, converted with the convention of Eq. 8.6 (no 1/(1 - alpha) factor)
    g = re.search(r"classical formulas with the convention of Eq. 8.6: MPPS ([\d.]+) um, penetration ([\d.]+), "
                  r"pressure drop ([\d.]+) Pa, q_F (\S+) 1/Pa", txt)
    rows.append(f"classical, Eq.~\\eqref{{eq:etapass}} & -- & {g.group(3)} & {g.group(1)} & -- & {g.group(2)} & "
                f"${tex_sci(g.group(4))}$ \\\\")
    return rows


def expected_inline_numbers():
    """Numbers quoted in the running text of Section 8, read from the summary files."""
    s = open(SUMMARY, encoding="utf-8").read()
    c = open(CAP_SUMMARY, encoding="utf-8").read()
    want = [tex_sci(re.search(r"spectral grid values (\S+), spline anywhere (\S+) ", s).group(1)),
            tex_sci(re.search(r"spectral grid values (\S+), spline anywhere (\S+) ", s).group(2))]
    want += re.findall(r"u = \+?([-\d.]+) U for eps", s)
    m = re.search(r"reconstructed <= (\S+) U, RS without reconstruction <= (\S+) U", s)
    want += [tex_sci(m.group(1)), tex_sci(m.group(2))]
    want += re.findall(r"largest difference ([\d.]+) particles \(the stratified", s)
    want += re.findall(r"gamma_max = ([\d.]+) U/a", s)
    for g in re.findall(r"Pe = +(\d+): tracker caught [\d.]+ \+- [\d.]+, Leveque [\d.]+, ratio ([\d.]+) \+- ([\d.]+)", s):
        want += [g[1], g[2]]
    # step convergence of the production model, and the effect of inertia (validation section)
    m = re.search(r"with inertia \(St = [\d.]+\), h = St/8, St/16, St/32: caught ([\d.]+), ([\d.]+), ([\d.]+); "
                  r"relative to the value without inertia ([-+\d.]+)", s)
    want += [m.group(1), m.group(2), m.group(3), f"{abs(float(m.group(4))) * 100:.0f}"]
    want += list(re.search(r"\(St = [\d.]+\), h = St/4, St/8, St/16: caught ([\d.]+), ([\d.]+), ([\d.]+)", s).groups())
    want += list(re.search(r"without inertia \(overdamped, .*h = 0.04, 0.02, 0.01: caught [\d.]+, [\d.]+, ([\d.]+)", s).groups())
    m = re.search(r"interception with inertia, no diffusion \(r_p = ([\d.]+)a, .*St = ([\d.]+), ([\d.]+), ([\d.]+): caught "
                  r"relative to exact interception without inertia ([-+\d.]+), ([-+\d.]+), ([-+\d.]+)", s)
    want += [m.group(1), m.group(2), m.group(3), m.group(4)] + [f"{abs(float(x)) * 100:.1f}" for x in m.groups()[4:]]
    m = re.search(r"interception with inertia at St = [\d.]+, h = St/8, St/16, St/32: caught relative to exact "
                  r"([-+\d.]+), ([-+\d.]+), ([-+\d.]+)", s)
    want += [f"{abs(float(x)) * 100:.1f}" for x in m.groups()]
    m = re.search(r"lambda = sqrt\(D St\) = ([\d.]+)a, delta = \(D/4\)\^\(1/3\) = ([\d.]+)a; .*relative to the value "
                  r"without inertia ([-+\d.]+), ([-+\d.]+) \(standard error of each value ([\d.]+);", s)
    want += [m.group(1), m.group(2), f"{abs(float(m.group(3))) * 100:.1f}", f"{abs(float(m.group(4))) * 100:.1f}"]
    m = re.search(r"wall moved back by ([\d.]+) lambda \(h = St/8\) and ([\d.]+) lambda \(h = St/32\)", s)
    want += [m.group(1), m.group(2)]
    ci = r"([\d.]+) \[([\d.]+), ([\d.]+)\]"
    for name in ("random / hexagonal", "random CV = 0.5 / random identical"):
        want += re.findall(re.escape(name) + r": eta at the MPPS " + ci + r", pressure drop " + ci +
                           r", q_F at the MPPS " + ci, c)[0]
    want += re.findall(r"largest difference from the exact answer ([\d.]+) particles per fibre", c)
    # mechanisms on the hexagonal array: each alone, the extra from acting together, classical terms
    mech = re.findall(r"hexagonal, d_p = [\d.]+ um: diffusion alone ([\d.]+), interception alone ([\d.]+), "
                      r"sum ([\d.]+), both without inertia ([\d.]+), extra from acting together ([-\d.]+) \+- ([\d.]+) "
                      r"\(2 s.e.\), with inertia ([\d.]+) \(change ([-+\d.]+)\); classical E_D ([\d.]+), E_R ([\d.]+), "
                      r"E_DR ([\d.]+), total ([\d.]+)", c)
    if len(mech) != 3:                                  # a format change must not silently skip Table 9
        return ["mechanism lines not found in capture_summary.txt"]
    for m in mech:                                      # every entry of Table 9 except the sum
        want += [m[0], m[1], m[3], m[4], m[5], m[6], m[8], m[9], m[10], m[11]]
    for g in re.findall(r": MPPS 95% interval \[([\d.]+), ([\d.]+)\] um", c):
        want += list(g)
    want += re.findall(r"smallest clearance from a capture circle ([\d.]+) a_mean", c)
    m = re.search(r"hexagonal, d_p = 0.25 um: diffusion alone [\d.]+, interception alone ([\d.]+), .*E_R ([\d.]+)", c)
    want += [m.group(1), m.group(2)]
    with open(os.path.join(RES, "capture_summary.csv"), newline="") as fh:
        tab = {(r["structure"], float(r["d_p_um"])): r for r in csv.DictReader(fh)}
    want += [f"{float(tab[('hexagonal', 0.25)]['eta']):.4f}", f"{float(tab[('hexagonal', 0.25)]['eta_classical']):.4f}",
             f"{float(tab[('hexagonal', 1.0)]['eta']):.3f}", f"{float(tab[('square', 1.0)]['eta']):.3f}"]
    return want


def check_numbers_in_text(tex_path=TEX):
    """Every row of Table 8 appears in Section 8 of main.tex as rebuilt from the results, and the numbers
    quoted in the running text of Section 8 appear there as whole tokens."""
    full = open(tex_path, encoding="utf-8").read()
    start = full.find("\\section{Particle transport and capture}")
    end = full.find("\\section{Methods: use of AI tools}")
    tex = full[start:end] if 0 <= start < end else ""
    flat = re.sub(r"\s+", " ", tex)
    rows = expected_table_rows()
    missing_rows = [r for r in rows if re.sub(r"\s+", " ", r) not in flat]

    def present(w):
        return re.search(r"(?<![\d.])" + re.escape(w) + r"(?![\d])", tex) is not None
    nums = expected_inline_numbers()
    missing = [w for w in nums if not present(w)]
    ok = not missing_rows and not missing and len(rows) == 6
    return ok, (f"{len(rows)} table rows and {len(nums)} inline numbers match main.tex" if ok
                else f"missing rows: {missing_rows[:2]}; missing numbers: {missing}")


def check_summary_current(summary_path=SUMMARY):
    """summary.txt is current: recompute the RS slip at the fibre surface for eps = 0.05a."""
    s = open(summary_path, encoding="utf-8").read()
    quoted = float(re.search(r"u = \+?([-\d.]+) U for eps = 0.05a", s).group(1))
    lay, _, F, _ = square_fields()
    slip = F.velocity(lay["centres"][0] + np.array([[0.0, 1.0]]))[0, 0]
    return abs(slip - quoted) < 6e-5, f"recomputed {slip:+.4f} (summary {quoted:+.4f})"


def check_capture_summary_current(path=CAP_SUMMARY):
    """capture_summary.txt is exactly what scripts/stage5_summary.py computes from the CSV files now."""
    import stage5_summary as summ
    lines = summ.build()[1]
    stored = open(path, encoding="utf-8").read().splitlines()
    return stored == lines, "matches a fresh computation" if stored == lines else "out of date"


def check_earlier_stages(checks=None):
    """Stage 1-4 checks still pass."""
    checks = checks if checks is not None else check_stage4.CHECKS
    failed = [c.__name__ for c in checks if not c()[0]]
    return not failed, "all Stage 1-4 checks pass" if not failed else f"failures: {failed}"


CHECKS = [check_fast_field, check_divergence_free, check_near_wall, check_interception, check_ballistic,
          check_variances, check_bridge, check_leveque_formula, check_efficiency_conversion, check_validation_results,
          check_capture_results, check_numbers_in_text, check_summary_current, check_capture_summary_current,
          check_earlier_stages]


# ============================================================================
# Broken inputs, one fault each
# ============================================================================

def broken_inputs(tmpdir):
    class WrongSignField(ff.StreamFunctionField2D):          # sign error in Eq. (8.2)
        def __init__(self, solver, grid_factor=4):
            super().__init__(solver, grid_factor)
            self.coef = -self.coef
            self.u_grid = 2 * self.U - self.u_grid
            self.v_grid = -self.v_grid

    _, _, F, _ = square_fields()

    def separate_velocity(X):                                  # u and v not from one stream function
        u = F.velocity(X)
        return u * np.array([1.0, 1.01])

    def variances_no_covariance(h, St, D):                     # forgets the X-V correlation
        sx2, sxv, sv2 = pt.variances(h, St, D)
        return sx2, 1e-12, sv2

    def leveque_no_three_halves(th, g, a, D):
        return ff.leveque_efficiency(th, g, a, D) / 1.5

    def efficiency_interstitial(f, lay):                       # Hinds' (1 - alpha) convention instead
        return pt.efficiency_from_pass(f, lay) * (1 - 0.1)

    bad_dir = os.path.join(tmpdir, "val")
    os.makedirs(bad_dir)
    for name in ["interception_check.csv", "near_wall_error.csv", "leveque_check.csv", "step_convergence.csv",
                 "kuwabara_interception.csv", "summary.txt"]:
        txt = open(os.path.join(RES, name), encoding="utf-8").read()
        if name == "leveque_check.csv":
            lines = txt.splitlines()
            parts = lines[-1].split(",")
            parts[5] = "1.5"
            lines[-1] = ",".join(parts)
            txt = "\n".join(lines) + "\n"
        open(os.path.join(bad_dir, name), "w", encoding="utf-8").write(txt)
    rows = load_capture()[:-1]                                  # one size missing
    bad_summary = os.path.join(tmpdir, "summary_bad.txt")
    s = open(SUMMARY, encoding="utf-8").read()
    q = re.search(r"u = \+?([-\d.]+) U for eps = 0.05a", s).group(1)
    open(bad_summary, "w", encoding="utf-8").write(s.replace(f"{q} U for eps = 0.05a", "0.0999 U for eps = 0.05a"))
    bad_cap = os.path.join(tmpdir, "cap_bad.txt")
    open(bad_cap, "w", encoding="utf-8").write(open(CAP_SUMMARY, encoding="utf-8").read().replace("MPPS", "MPPS ", 1))
    bad_tex = os.path.join(tmpdir, "main_bad.tex")
    tex = open(TEX, encoding="utf-8").read()
    num = expected_inline_numbers()[-1]
    i0 = tex.find("\\section{Particle transport and capture}")
    i1 = tex.find("\\section{Methods: use of AI tools}")
    # replace EVERY occurrence of that number in Section 8, so the stale value is the only one left
    open(bad_tex, "w", encoding="utf-8").write(tex[:i0] + tex[i0:i1].replace(num, "9.999") + tex[i1:])

    return [
        ("sign error in the stream-function formula", lambda: check_fast_field(WrongSignField)),
        ("velocity components not from one stream function", lambda: check_divergence_free(separate_velocity)),
        ("near-wall reconstruction with only 4 modes", lambda: check_near_wall(n_modes=4)),
        ("tracker ignoring the particle radius", lambda: check_interception(r_p_tracker=0.0)),
        ("heavy particles launched with the air velocity", lambda: check_ballistic(v_init="fluid")),
        ("Ornstein-Uhlenbeck step without the X-V correlation", lambda: check_variances(variances_no_covariance)),
        ("Leveque formula without the factor 3/2", lambda: check_leveque_formula(leveque_no_three_halves)),
        ("efficiency in the interstitial (1 - alpha) convention", lambda: check_efficiency_conversion(efficiency_interstitial)),
        ("bridge formula without its factor 2", lambda: check_bridge(lambda d0, d1, s2: np.exp(-d0 * d1 / s2))),
        ("a Leveque ratio of 1.5 in the results", lambda: check_validation_results(bad_dir)),
        ("a capture result missing", lambda: check_capture_results(rows)),
        ("paper quoting a stale number", lambda: check_numbers_in_text(bad_tex)),
        ("summary.txt quoting a wrong slip", lambda: check_summary_current(bad_summary)),
        ("capture_summary.txt edited by hand", lambda: check_capture_summary_current(bad_cap)),
        ("an earlier check failing", lambda: check_earlier_stages([lambda: (False, "")])),
    ]


def main():
    import tempfile
    print("=" * 72)
    print("PART A: Stage 5 checks (all must PASS)")
    print("=" * 72)
    all_ok = True
    for chk in CHECKS:
        ok, msg = chk()
        all_ok &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {chk.__name__}: {msg}", flush=True)
    print("=" * 72)
    print("PART B: broken inputs, one fault each (every check must FAIL on these)")
    print("=" * 72)
    with tempfile.TemporaryDirectory() as tmp:
        for name, run in broken_inputs(tmp):
            ok, _ = run()
            all_ok &= (not ok)
            print(f"[{'PASS' if not ok else 'FAIL'}] check correctly rejects: {name}", flush=True)
    print("=" * 72)
    print("ALL STAGE 5 CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
