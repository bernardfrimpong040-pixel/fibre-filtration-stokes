"""
check_stage2.py
===============
Every Stage 2 check must PASS, the Stage 1 checks must still pass, and every
check must FAIL on a deliberately broken input (one fault at a time).

Run from the project folder:   python checks/check_stage2.py
Exit code 0 = all good; 1 = something failed.
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import rs_kernels as rk                                            # noqa: E402
from periodic import PeriodicStokeslet2D, spectral_only_blocks    # noqa: E402
import check_stage1                                                # noqa: E402

TEX = os.path.join(ROOT, "paper", "main.tex")
SUMMARY = os.path.join(ROOT, "results", "stage2", "summary.txt")
GAMMA = rk.EULER_GAMMA


# ============================================================================
# Helpers: build velocity functions from a kernel so broken kernels can be fed in
# ============================================================================

def velocity_from_AB(AB_func, eps, f=np.array([0.7, -0.4])):
    """Velocity at a point p of one blob at the origin, using scalar functions A, B."""
    def u(p):
        r = np.linalg.norm(p)
        A, B = AB_func(np.array([r]), eps)
        return (A[0] * f + B[0] * np.dot(f, p) * p) / (4 * np.pi)
    return u


def fd_divergence_and_residual(AB_func, eps=0.1, h=None):
    """Largest |div u| and largest relative Stokes residual at four test points."""
    h = h or eps / 200.0
    f = np.array([0.7, -0.4])
    u = velocity_from_AB(AB_func, eps, f)

    def p(x):
        return rk.pressure_2d(np.atleast_2d(x), np.zeros((1, 2)), f[None, :], eps)[0]

    worst_div, worst_res = 0.0, 0.0
    ex, ey = np.array([h, 0.0]), np.array([0.0, h])
    for pt in [(0.03, 0.02), (0.1, -0.05), (0.25, 0.1), (0.5, 0.4)]:
        x = np.array(pt)
        div = (u(x + ex)[0] - u(x - ex)[0] + u(x + ey)[1] - u(x - ey)[1]) / (2 * h)
        lap = (u(x + ex) + u(x - ex) + u(x + ey) + u(x - ey) - 4 * u(x)) / h**2
        gp = np.array([p(x + ex) - p(x - ex), p(x + ey) - p(x - ey)]) / (2 * h)
        force = f * rk.blob_2d(np.linalg.norm(x), eps)
        scale = max(np.max(np.abs(force)), np.max(np.abs(gp)))
        worst_div = max(worst_div, abs(div) / scale)
        worst_res = max(worst_res, np.max(np.abs(lap - gp + force)) / scale)
    return worst_div, worst_res


# ============================================================================
# The checks. Each returns (passed: bool, message: str).
# ============================================================================

def check_divergence(AB_func=rk.kernel_AB_2d):
    """Eq. (4.9) must be divergence-free (finite differences, relative to term size)."""
    d, _ = fd_divergence_and_residual(AB_func)
    return d < 1e-6, f"max relative |div u| = {d:.1e} (need < 1e-6)"


def check_stokes_residual(AB_func=rk.kernel_AB_2d):
    """Eq. (4.9)-(4.10) must satisfy the forced Stokes equations (4.1)."""
    _, res = fd_divergence_and_residual(AB_func)
    return res < 1e-4, f"max relative Stokes residual = {res:.1e} (need < 1e-4)"


def check_centre_values(AB_func=rk.kernel_AB_2d, eps=0.3):
    """Eq. (4.11): A(0) = -ln eps + gamma/2, B(0) = 1/(2 eps^2), and continuity just off centre."""
    A0, B0 = AB_func(np.array([0.0]), eps)
    A1, B1 = AB_func(np.array([1e-3 * eps]), eps)
    ok = (abs(A0[0] - (-np.log(eps) + GAMMA / 2)) < 1e-10 and abs(B0[0] * 2 * eps**2 - 1) < 1e-10
          and abs(A1[0] - A0[0]) < 1e-5 and abs(B1[0] / B0[0] - 1) < 1e-5)
    return ok, f"A(0) = {A0[0]:.6f} (expect {-np.log(eps) + GAMMA/2:.6f}), B(0) eps^2 = {B0[0]*eps**2:.6f} (expect 0.5)"


def check_singular_limit(AB_func=rk.kernel_AB_2d):
    """Eq. (4.12): as eps -> 0 the kernel becomes the ordinary Stokeslet."""
    r = np.array([0.5, 1.0, 3.0])
    A, B = AB_func(r, 1e-4)
    As, Bs = rk.singular_AB_2d(r)
    err = max(np.max(np.abs(A - As)), np.max(np.abs((B - Bs) * r**2)))
    return err < 1e-7, f"max difference from singular Stokeslet at eps = 1e-4: {err:.1e}"


def check_dipole_bound(AB_func=rk.kernel_AB_2d, eps=0.1):
    """Proposition 2, Eq. (4.14): kernel minus (Stokeslet + dipole) is below e^{-X}/X."""
    # stop at r = 4.5 eps, where the bound (1.6e-10) is still far above round-off
    r = eps * np.linspace(1.0, 4.5, 60)
    X = (r / eps) ** 2
    A, B = AB_func(r, eps)
    As, Bs = rk.singular_AB_2d(r)
    dA = np.abs(A - (As + eps**2 / (2 * r**2)))
    dB = np.abs((B - (Bs - eps**2 / r**4)) * r**2)
    ratio = np.max(np.maximum(dA, dB) / (np.exp(-X) / X))
    return ratio <= 1.0 + 1e-6, f"max (remainder / bound e^-X/X) = {ratio:.3f} (need <= 1)"


def check_ewald_xi(builder=PeriodicStokeslet2D):
    """Proposition 5: the periodic kernel does not depend on the splitting width xi."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for Lx, Ly in [(1.3, 0.8), (5.0, 1.0)]:                 # nearly square and elongated cells
        xe, xs = rng.random((7, 2)) * [Lx, Ly], rng.random((5, 2)) * [Lx, Ly]
        b1 = builder(Lx, Ly, 0.04, xi=min(Lx, Ly) / 5).blocks(xe, xs)
        b2 = builder(Lx, Ly, 0.04, xi=min(Lx, Ly) / 9).blocks(xe, xs)
        scale = max(np.max(np.abs(b)) for b in b1)
        worst = max(worst, max(np.max(np.abs(p - q)) for p, q in zip(b1, b2)) / scale)
    return worst < 1e-14, f"relative change between xi = L/5 and L/9: {worst:.1e} (need < 1e-14)"


def check_ewald_vs_fourier(builder=PeriodicStokeslet2D):
    """Proposition 5: Ewald equals the plain Fourier series (4.18), an independent calculation."""
    rng = np.random.default_rng(12)
    worst = 0.0
    for Lx, Ly in [(1.0, 1.0), (5.0, 1.0)]:
        xe, xs = rng.random((6, 2)) * [Lx, Ly], rng.random((4, 2)) * [Lx, Ly]
        b1 = builder(Lx, Ly, 0.06).blocks(xe, xs)
        b2 = spectral_only_blocks(xe, xs, Lx, Ly, 0.06, n_max=70)
        scale = max(np.max(np.abs(b)) for b in b1)
        worst = max(worst, max(np.max(np.abs(p - q)) for p, q in zip(b1, b2)) / scale)
    return worst < 1e-14, f"relative difference Ewald vs Fourier series: {worst:.1e} (need < 1e-14)"


def check_periodicity(builder=PeriodicStokeslet2D):
    """Shifting the evaluation point by one cell length must not change the kernel."""
    P = builder(1.0, 0.7, 0.05)
    xs = np.array([[0.3, 0.2]])
    x = np.array([[0.55, 0.41]])
    b0 = P.blocks(x, xs)
    b1 = P.blocks(x + [1.0, 0.0], xs)
    b2 = P.blocks(x + [0.0, 0.7], xs)
    diff = max(np.max(np.abs(p - q)) for p, q in zip(b0, b1)) + max(np.max(np.abs(p - q)) for p, q in zip(b0, b2))
    return diff < 1e-13, f"change after shifting by Lx or Ly: {diff:.1e}"


def check_zero_mean(builder=PeriodicStokeslet2D):
    """Zero k-mode removed: the kernel averages to zero over the cell (Prop. 4 and 5)."""
    P = builder(1.0, 1.0, 0.1)
    n = 64
    g = (np.arange(n) + 0.5) / n
    GX, GY = np.meshgrid(g, g)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    Sxx, Sxy, Syy = P.blocks(pts, np.array([[0.5, 0.5]]))
    m = max(abs(Sxx.mean()), abs(Sxy.mean()), abs(Syy.mean())) / np.max(np.abs(Sxx))
    return m < 1e-10, f"cell average / max = {m:.1e}"


def hasimoto_constant(alpha, builder=PeriodicStokeslet2D):
    """C in k/a^2 = (ln(1/alpha) - C)/(8 alpha) from one blob per cell with eps = a e^{gamma/2}."""
    a = 1.0
    L = np.sqrt(np.pi * a**2 / alpha)
    P = builder(L, L, a * np.exp(GAMMA / 2))
    Sxx, _, _ = P.blocks(np.zeros((1, 2)), np.zeros((1, 2)))
    k = L * L * abs(Sxx[0, 0])              # k = mu U A / |f| with f = -U / Sxx, mu = U = 1
    return -(8 * alpha * k / a**2) - np.log(alpha)


def check_hasimoto(builder=PeriodicStokeslet2D):
    """Proposition 3: one blob per cell recovers Hasimoto's constant 1.476 as alpha -> 0."""
    C = hasimoto_constant(1e-4, builder)
    return abs(C - 1.476) < 2e-3, f"C(alpha = 1e-4) = {C:.4f} (Hasimoto 1.476)"


def ring_leakage(n_pts, eps_ring, builder=PeriodicStokeslet2D):
    """Net, positive-part and chord flux (Eq. 4.13) for one fixed fibre at alpha = 0.1."""
    a, alpha = 1.0, 0.1
    L = np.sqrt(np.pi * a**2 / alpha)
    c = np.array([L / 2, L / 2])
    t = 2 * np.pi * np.arange(n_pts) / n_pts
    ring = c + a * np.column_stack([np.cos(t), np.sin(t)])
    P = builder(L, L, eps_ring)
    Sxx, Sxy, Syy = P.blocks(ring, ring)
    sol = np.linalg.solve(np.block([[Sxx, Sxy], [Sxy, Syy]]),
                          -np.concatenate([np.ones(n_pts), np.zeros(n_pts)]))
    F = np.column_stack([sol[:n_pts], sol[n_pts:]])
    nq = 2000
    tq = 2 * np.pi * (np.arange(nq) + 0.5) / nq
    nrm = np.column_stack([np.cos(tq), np.sin(tq)])
    un = np.sum(P.velocity(c + a * nrm, ring, F, U=(1.0, 0.0)) * nrm, axis=1)
    ds = 2 * np.pi * a / nq
    yq = np.linspace(-a, a, 1601)
    uc = P.velocity(c + np.column_stack([0 * yq, yq]), ring, F, U=(1.0, 0.0))
    return np.sum(un) * ds / (2 * a), np.sum(np.maximum(un, 0)) * ds / (2 * a), np.trapezoid(uc[:, 0], yq) / (2 * a)


def flux_random_sources(AB_func=rk.kernel_AB_2d):
    """Net flux of the free-space velocity of 5 random blobs (random forces, no symmetry)
    through the circle of radius 1 around (1.5, 1.5); some blobs lie inside it."""
    rng = np.random.default_rng(5)
    src = rng.random((5, 2)) * 3.0
    forces = rng.normal(size=(5, 2))
    eps = 0.2
    nq = 4000
    t = 2 * np.pi * (np.arange(nq) + 0.5) / nq
    nrm = np.column_stack([np.cos(t), np.sin(t)])
    pts = np.array([1.5, 1.5]) + nrm
    dx = pts[:, None, :] - src[None, :, :]                       # (Q, 5, 2)
    r = np.linalg.norm(dx, axis=2)
    A, B = AB_func(r, eps)
    fdotx = np.sum(forces[None] * dx, axis=2)
    u = (A[..., None] * forces[None] + (B * fdotx)[..., None] * dx).sum(axis=1) / (4 * np.pi)
    return np.sum(np.sum(u * nrm, axis=1)) * (2 * np.pi / nq)


def check_zero_net_flux(AB_func=rk.kernel_AB_2d):
    """Proposition 1: zero net flux through a closed curve for ASYMMETRIC sources (so symmetry
    cannot hide an error), while a leaky fibre still passes air through it."""
    net = flux_random_sources(AB_func)
    _, pos, chord = ring_leakage(8, 0.05)
    ok = abs(net) < 1e-12 and pos > 0.1 and chord > 0.1
    return ok, f"net flux (random sources) = {net:.1e}; leaky 8-blob fibre: positive-part = {pos:.3f}, chord = {chord:.3f}"


def check_leak_closes(leak=None):
    """With eps comparable to the spacing the fibre is closed to better than 1e-5."""
    net, pos, chord = leak if leak is not None else ring_leakage(32, 0.2)
    ok = abs(net) < 1e-12 and pos < 1e-5 and abs(chord) < 1e-5
    return ok, f"N = 32, eps/h = 1.02: positive-part = {pos:.1e}, chord = {chord:.1e}"


def tex_sci(x, digits=1):
    """Format like the paper: 6.8\\times10^{-16}."""
    mant, ex = f"{x:.{digits}e}".split("e")
    return f"{mant}\\times10^{{{int(ex)}}}"


def stage2_numbers():
    """Recompute every Stage 2 number quoted EXACTLY in main.tex, formatted as in the paper."""
    eps = 0.1
    r = 4.0 * eps
    A, B = rk.kernel_AB_2d(np.array([r]), eps)
    As, Bs = rk.singular_AB_2d(np.array([r]))
    e_st = max(abs(A[0] - As[0]), abs((B[0] - Bs[0]) * r**2))
    e_dp = max(abs(A[0] - As[0] - eps**2 / (2 * r**2)), abs((B[0] - Bs[0] + eps**2 / r**4) * r**2))
    leak_8, leak_mid, leak_good = ring_leakage(8, 0.05), ring_leakage(32, 0.05), ring_leakage(32, 0.2)
    C0, C1 = hasimoto_constant(1e-4), hasimoto_constant(0.1)
    C_cyl = 1.476 - 2 * 0.1 + 1.774 * 0.1**2 - 4.076 * 0.1**3
    k_ratio = (np.log(10) - C1) / (np.log(10) - C_cyl)
    slope = (hasimoto_constant(1e-3) - 1.476336) / 1e-3
    return {
        "C small alpha": f"{C0:.4f}",
        "C alpha 0.1": f"{C1:.4f}",
        "C cylinder 0.1": f"{C_cyl:.4f}",
        "k difference": f"{100 * abs(k_ratio - 1):.1f}\\%",
        "slope": f"${slope:.2f}$",
        "far vs Stokeslet": tex_sci(e_st, 2),
        "far vs dipole": tex_sci(e_dp, 1),
        "leak 8 chord": f"{leak_8[2]:.3f}",
        "leak 8 percent": f"{100 * leak_8[2]:.0f}\\%",
        "leak 32 coarse": tex_sci(leak_mid[1], 2),
        "leak 32 good pos": tex_sci(leak_good[1], 2),
        "leak 32 good chord": tex_sci(leak_good[2], 2),
    }


def stage2_bounds():
    """Quantities the paper states as BOUNDS (they vary in the last digits between machines):
    (value, bound, text that must appear in main.tex)."""
    div, res = fd_divergence_and_residual(rk.kernel_AB_2d)
    nets = [abs(ring_leakage(*c)[0]) for c in [(8, 0.05), (32, 0.05), (32, 0.2)]]
    return {
        "div": (div, 5e-7, "$5\\times10^{-7}$"),
        "Stokes residual": (res, 1e-5, "below $10^{-5}$ of the largest term"),
        "Ewald xi": (float(check_ewald_xi()[1].split(": ")[1].split(" ")[0]), 1e-14, "less than $10^{-14}$"),
        "net flux table": (max(nets), 1e-15, "$<10^{-15}$"),
    }


def check_numbers_in_text(numbers=None, bounds=None, tex_path=TEX):
    """Every exactly quoted number must appear in main.tex as formatted, and every quantity
    quoted as a bound must satisfy it and the bound must appear in the text."""
    numbers = numbers or stage2_numbers()
    bounds = bounds or stage2_bounds()
    tex = open(tex_path, encoding="utf-8").read()
    problems = [f"{k}={v} not in text" for k, v in numbers.items() if v not in tex]
    for name, (value, bound, text) in bounds.items():
        if not value < bound:
            problems.append(f"{name}: {value:.1e} not below {bound:.0e}")
        if text not in tex:
            problems.append(f"{name}: bound text not in main.tex")
    msg = f"{len(numbers)} exact numbers and {len(bounds)} bounds consistent with main.tex"
    return not problems, msg if not problems else "; ".join(problems[:4])


def check_summary_file(expected=None, summary_path=SUMMARY):
    """results/stage2/summary.txt must contain the values a fresh recomputation gives,
    written in the same format the Stage 2 script uses."""
    if expected is None:
        C_cyl = 1.476 - 2 * 0.1 + 1.774 * 0.1**2 - 4.076 * 0.1**3
        _, _, chord8 = ring_leakage(8, 0.05)
        _, pos_good, _ = ring_leakage(32, 0.2)
        expected = [f"C = {hasimoto_constant(1e-4):.4f}", f"C = {hasimoto_constant(0.1):.4f}",
                    f"gives {C_cyl:.4f}", f"chord flux = {chord8:.2e}", f"positive-part flux = {pos_good:.2e}"]
    text = open(summary_path, encoding="utf-8").read()
    missing = [w for w in expected if w not in text]
    return not missing, "summary.txt matches recomputation" if not missing else f"summary.txt missing {missing}"


def check_stage1_still_passes(checks=None):
    """Earlier work must not break: all Stage 1 checks still pass."""
    checks = checks if checks is not None else check_stage1.CHECKS
    failed = [c.__name__ for c in checks if not c()[0]]
    return not failed, "all Stage 1 checks pass" if not failed else f"Stage 1 failures: {failed}"


CHECKS = [check_divergence, check_stokes_residual, check_centre_values, check_singular_limit,
          check_dipole_bound, check_ewald_xi, check_ewald_vs_fourier, check_periodicity,
          check_zero_mean, check_hasimoto, check_zero_net_flux, check_leak_closes,
          check_numbers_in_text, check_summary_file, check_stage1_still_passes]


# ============================================================================
# Broken inputs, one fault each
# ============================================================================

def broken_inputs():
    def AB_wrong_B(r, eps):                 # B with the wrong factor: not divergence-free
        A, B = rk.kernel_AB_2d(r, eps)
        return A, 1.1 * B

    def AB_wrong_A(r, eps):                 # A changed near the blob (adds 0.5 e^{-X}): breaks the Stokes equations
        A, B = rk.kernel_AB_2d(r, eps)
        X = (np.asarray(r) / eps) ** 2
        return A + 0.5 * np.exp(-X), B

    def AB_wrong_centre(r, eps):            # A shifted by gamma/2 (wrong centre value)
        A, B = rk.kernel_AB_2d(r, eps)
        return A - GAMMA / 2, B

    def AB_B_times_1p3(r, eps):             # B scaled by 1.3: not divergence-free
        A, B = rk.kernel_AB_2d(r, eps)
        return A, 1.3 * B

    def AB_no_dipole(r, eps):               # far field without the eps^2 terms
        As, Bs = rk.singular_AB_2d(np.maximum(r, 1e-12))
        return As, Bs

    class NoShortRange(PeriodicStokeslet2D):     # Ewald with the short-range part dropped
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.n_images = -1                     # range(1, 0) is empty: no copies summed

    class FewImages(PeriodicStokeslet2D):          # only the central copy summed (n_images = 0)
        def __init__(self, Lx, Ly, eps, **kw):
            super().__init__(Lx, Ly, eps, **kw)
            self.n_images = 0                          # set after the guard, to simulate the bug

    class KeepZeroModeMean(PeriodicStokeslet2D):   # adds a constant: non-zero cell mean
        def blocks(self, xe, xs):
            Sxx, Sxy, Syy = super().blocks(xe, xs)
            return Sxx + 0.01, Sxy, Syy + 0.01

    class NotPeriodic(PeriodicStokeslet2D):        # adds a term that grows with x (not periodic)
        def blocks(self, xe, xs):
            Sxx, Sxy, Syy = super().blocks(xe, xs)
            return Sxx + 1e-3 * xe[:, None, 0], Sxy, Syy

    numbers = stage2_numbers()
    return [
        ("B scaled by 1.1 (not divergence-free)", lambda: check_divergence(AB_wrong_B)),
        ("A changed near the blob (not Stokes)", lambda: check_stokes_residual(AB_wrong_A)),
        ("A shifted by gamma/2 (wrong centre)", lambda: check_centre_values(AB_wrong_centre)),
        ("kernel stuck at large eps", lambda: check_singular_limit(lambda r, e: rk.kernel_AB_2d(r, 0.05))),
        ("far field without the dipole", lambda: check_dipole_bound(AB_no_dipole)),
        ("Ewald without the short-range sum", lambda: check_ewald_xi(NoShortRange)),
        ("Ewald with too few copies", lambda: check_ewald_vs_fourier(FewImages)),
        ("kernel not periodic", lambda: check_periodicity(NotPeriodic)),
        ("kernel with non-zero cell mean", lambda: check_zero_mean(KeepZeroModeMean)),
        ("Hasimoto test with a non-zero mean added", lambda: check_hasimoto(KeepZeroModeMean)),
        ("net flux with a non-divergence-free kernel (B x 1.3)", lambda: check_zero_net_flux(AB_B_times_1p3)),
        ("fibre that still leaks (N = 32, eps/h = 0.25)", lambda: check_leak_closes(ring_leakage(32, 0.05))),
        ("paper quoting a wrong Hasimoto constant", lambda: check_numbers_in_text(dict(numbers, **{"C small alpha": "1.4999"}))),
        ("a quantity above its stated bound", lambda: check_numbers_in_text(numbers, dict(stage2_bounds(), div=(1e-6, 5e-7, "$5\\times10^{-7}$")))),
        ("summary.txt out of date", lambda: check_summary_file(["C = 1.2000"])),
        ("a Stage 1 check failing", lambda: check_stage1_still_passes([lambda: (False, "")])),
    ]


def main():
    print("=" * 72)
    print("PART A: Stage 2 checks (all must PASS)")
    print("=" * 72)
    all_ok = True
    for chk in CHECKS:
        ok, msg = chk()
        all_ok &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {chk.__name__}: {msg}")
    print("=" * 72)
    print("PART B: broken inputs, one fault each (every check must FAIL on these)")
    print("=" * 72)
    for name, run in broken_inputs():
        ok, _ = run()
        all_ok &= (not ok)
        print(f"[{'PASS' if not ok else 'FAIL'}] check correctly rejects: {name}")
    print("=" * 72)
    print("ALL STAGE 2 CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
