"""
check_stage4.py
===============
Every Stage 4 check must PASS, the Stage 1-3 checks must still pass,
and every check must FAIL on a deliberately broken input (one fault at a time).

Run from the project folder (after scripts/stage4_validation.py, scripts/stage4_networks.py
and scripts/stage4_summary.py):
    python checks/check_stage4.py          (about 3-5 minutes)
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
sys.path.insert(0, HERE)
import rs_kernels3d as k3                                                  # noqa: E402
import geometry3d as g3                                                    # noqa: E402
import periodic3d as p3                                                    # noqa: E402
from periodic import PeriodicStokeslet2D                                   # noqa: E402
from solver3d import NetworkSolver3D, CENTRELINE_EPS_FACTOR, OFFSET_FACTOR  # noqa: E402
from reference_bie import reference_k, reference_k_axial                  # noqa: E402
import check_stage3                                                        # noqa: E402

TEX = os.path.join(ROOT, "paper", "main.tex")
RES = os.path.join(ROOT, "results", "stage4")
SUMMARY = os.path.join(RES, "summary.txt")
NET_SUMMARY = os.path.join(RES, "networks_summary.txt")
LATTICE_CSV = os.path.join(RES, "lattice_validation.csv")
CUBIC_CSV = os.path.join(RES, "cubic3.csv")


# ============================================================================
# The checks. Each returns (passed: bool, message: str).
# ============================================================================

def closed_form_H(r, eps):
    """Eq. (7.2) written out again, independently of rs_kernels3d.py (valid away from r = 0)."""
    from scipy.special import erf
    rho = r / eps
    E, G = erf(rho), eps * np.exp(-rho**2) / np.sqrt(np.pi)
    return (E / r + eps**2 * E / (2 * r**3) - G / r**2,
            E / r**3 - 1.5 * eps**2 * E / r**5 + 3 * G / r**4)


def check_kernel3d(kernel=k3.kernel_H_3d):
    """The 3D kernel: far field = Stokeslet + dipole eps^2/2 (Eq. 7.3); centre values (Eq. 7.4);
    the Taylor series used for rho < 0.25 agrees with the closed form (7.2) just below the switch;
    and the velocity it gives satisfies the forced Stokes equations (finite differences)."""
    eps = 0.5
    rs = np.array([0.15, 0.2, 0.249]) * eps
    ser = kernel(rs, eps)
    clo = closed_form_H(rs, eps)
    series_err = max(np.max(np.abs(ser[0] / clo[0] - 1)), np.max(np.abs(ser[1] / clo[1] - 1)))
    r = np.array([6.0, 9.0]) * eps
    H1, H2 = kernel(r, eps)
    far = max(np.max(np.abs(H1 / (1 / r + eps**2 / (2 * r**3)) - 1)),
              np.max(np.abs(H2 / (1 / r**3 - 1.5 * eps**2 / r**5) - 1)))
    c1, c2 = kernel(np.array([0.0]), eps)
    centre = max(abs(c1[0] / (8 / (3 * np.sqrt(np.pi) * eps)) - 1), abs(c2[0] / (8 / (15 * np.sqrt(np.pi) * eps**3)) - 1))
    # Stokes residual at one point, by finite differences, with the kernel under test
    f = np.array([0.7, -0.4, 0.3])
    p = np.array([0.4, 0.2, -0.3])
    h = 1e-3

    def u(q):
        rr = np.linalg.norm(q)
        a1, a2 = kernel(np.array([rr]), eps)
        return (a1[0] * f + a2[0] * (f @ q) * q) / (8 * np.pi)

    def pr(q):
        return k3.pressure_3d(np.atleast_2d(q), np.zeros((1, 3)), f[None, :], eps)[0]
    lap, gp = np.zeros(3), np.zeros(3)
    for c in range(3):
        e = np.zeros(3)
        e[c] = h
        lap += (u(p + e) - 2 * u(p) + u(p - e)) / h**2
        gp[c] = (pr(p + e) - pr(p - e)) / (2 * h)
    force = f * k3.blob_3d(np.linalg.norm(p), eps)
    res = np.max(np.abs(lap - gp + force)) / np.max(np.abs(lap))
    ok = far < 1e-12 and centre < 1e-12 and series_err < 1e-10 and res < 1e-4
    return ok, f"far field {far:.1e}, centre {centre:.1e}, series vs closed form {series_err:.1e}, Stokes residual {res:.1e}"


def check_ewald3d(cls=p3.PeriodicStokeslet3D):
    """The triply periodic sum does not depend on the splitting width, and equals the plain Fourier
    series for a wide blob."""
    L = 5.0
    x = np.random.default_rng(3).random((25, 3)) * L
    B1 = cls([L, L, L], 0.3, xi=L / 4).blocks(x, x)
    B2 = cls([L, L, L], 0.3, xi=L / 7).blocks(x, x)
    split = max(np.max(np.abs(a - b)) for a, b in zip(B1, B2)) / max(np.max(np.abs(a)) for a in B1)
    Bw = cls([L, L, L], 1.5).blocks(x, x)
    Sw = p3.spectral_only_blocks_3d(x, x, [L, L, L], 1.5, n_max=12)
    spec = max(np.max(np.abs(a - b)) for a, b in zip(Bw, Sw)) / max(np.max(np.abs(a)) for a in Bw)
    return split < 1e-12 and spec < 1e-10, f"xi-dependence {split:.1e}, vs Fourier series {spec:.1e}"


def check_axis_distance(dist=g3.axis_distance):
    """Distance between periodic fibre axes, against a brute-force search over sampled points of the
    lines and their copies (includes a pair whose cross product has gcd 2)."""
    L = 4.0
    pairs = [((1, 0, 0), (0, 1, 0)), ((1, 1, 0), (1, -1, 0)), ((1, 0, 1), (0, 1, 0)), ((1, 0, 0), (1, 0, 0)),
             ((1, 1, 1), (1, -1, 0))]
    rng = np.random.default_rng(5)
    worst = 0.0
    for d1, d2 in [p for p in pairs for _ in range(4)]:       # four random placements of each pair
        f1 = dict(point=tuple(rng.random(3) * L), dirn=d1, radius=1.0)
        f2 = dict(point=tuple(rng.random(3) * L), dirn=d2, radius=1.0)
        s = np.linspace(0, 1, 801)[:, None]
        P1 = np.asarray(f1["point"]) + s * np.linalg.norm(d1) * L * g3.unit(d1)
        P2 = np.asarray(f2["point"]) + s * np.linalg.norm(d2) * L * g3.unit(d2)
        best = np.inf
        for sh in np.array(np.meshgrid(*[np.arange(-2, 3)] * 3)).reshape(3, -1).T:
            D = P1[:, None, :] - (P2[None, :, :] + sh * L)
            best = min(best, np.sqrt(np.min(np.sum(D**2, axis=2))))
        worst = max(worst, abs(dist(f1, f2, L) - best))
    # sampling at 1/800 of the fibre length limits the brute force to about 1e-4 L
    return worst < 2e-3, f"largest difference from brute force {worst:.1e} (cell side 4)"


def check_networks(make=g3.random_network):
    """Random networks: exact solid fraction, every gap >= min_gap, exact orientation tensor when
    balanced, reproducible from the seed; the tilted lattice has the rectangular cross-section."""
    problems = []
    for kind, n, T in [("layered", 4, np.diag([0.5, 0.5, 0.0])), ("cubic", 3, np.eye(3) / 3)]:
        n1 = make(0.15, n, np.random.default_rng(8), kind=kind, min_gap=0.6, balanced=True)
        n2 = make(0.15, n, np.random.default_rng(8), kind=kind, min_gap=0.6, balanced=True)
        if abs(g3.solid_fraction(n1) - 0.15) > 1e-12:
            problems.append(f"{kind} alpha")
        if g3.min_surface_gap(n1) < 0.6 - 1e-9:
            problems.append(f"{kind} gap")
        if np.max(np.abs(g3.orientation_tensor(n1) - T)) > 1e-12:
            problems.append(f"{kind} <nn>")
        counts = [sum(tuple(f["dirn"]) == d for f in n1["fibres"]) for d in g3.DIRECTION_SETS[kind]]
        if len(set(counts)) != 1:
            problems.append(f"{kind} direction counts {counts}")
        if not np.allclose([f["point"] for f in n1["fibres"]], [f["point"] for f in n2["fibres"]]):
            problems.append(f"{kind} not reproducible")
    t = g3.lattice_network("tilted", 0.1)
    f = t["fibres"][0]
    if abs(g3.axis_distance(f, f, t["L"]) - t["L"] / np.sqrt(2)) > 1e-12 or abs(g3.solid_fraction(t) - 0.1) > 1e-12:
        problems.append("tilted lattice")
    return not problems, "solid fraction, gaps, <n n>, seeds, tilted lattice all correct" if not problems else str(problems)


def check_disk_points(points=g3.disk_points):
    """The pressure-balance discs: every point inside its disc, weights adding up to the disc volume."""
    net = g3.lattice_network("cubic3", 0.1)
    shift = OFFSET_FACTOR * 0.3
    P, w, ids = points(net, 0.6, shift)
    L = net["L"]
    vol = sum(np.pi * (1 - shift) ** 2 * np.linalg.norm(f["dirn"]) * L for f in net["fibres"])
    inside = True
    for m, f in enumerate(net["fibres"]):
        d = P[ids == m] - np.asarray(f["point"])
        d -= L * np.round(d / L)
        t = g3.unit(f["dirn"])
        rad = np.linalg.norm(d - np.outer(d @ t, t), axis=1)
        inside &= bool(np.all(rad < 1 - shift))
    ok = inside and abs(w.sum() / vol - 1) < 1e-12
    return ok, f"points inside, total weight / disc volume - 1 = {w.sum() / vol - 1:.1e}"


class NoInsideFluxSolver(NetworkSolver3D):
    """Broken-input helper: counts the flow inside the fibres as Darcy flux."""
    def inside_flux(self, *args, **kwargs):
        return np.zeros(3)


def axial_error(balance=True, solver=NetworkSolver3D, alpha=0.1):
    """Richardson (eps 0.2 / 0.4) error of the surface method for flow ALONG a square array of fibres."""
    net = g3.lattice_network("axial_square", alpha)
    L = net["L"]
    lay = dict(centres=np.array([[L / 2, L / 2]]), radii=np.array([1.0]), Lx=L, Ly=L)
    ks = [solver(net, "surface", e, balance=balance).dimensionless_permeability() for e in (0.2, 0.4)]
    return (4 * ks[0] - ks[1]) / 3 / reference_k_axial(lay, 128) - 1


def check_pressure_balance(balance=True, solver=NetworkSolver3D):
    """Proposition 8 and Eq. (7.10): with the pressure balance and the fluid-only flux, flow along
    the fibres is within 0.1% of the exact answer (without either, the error is 0.6-0.8%)."""
    err = axial_error(balance, solver)
    return abs(err) < 1e-3, f"axial square array, alpha = 0.1: error {100 * err:+.3f}%"


def check_transverse_and_centreline(eps_factor=CENTRELINE_EPS_FACTOR):
    """Fibres across the flow (aligned square array): the 3D centreline model equals the 2D one-blob
    model (Eq. 7.6) to 1e-6, and the 3D surface method is within 0.2% of the exact 2D answer."""
    alpha = 0.1
    net = g3.lattice_network("aligned_square", alpha)
    L = net["L"]
    s = NetworkSolver3D(net, "centreline")
    s.eps = eps_factor                                           # the width under test
    s.kernel = p3.PeriodicStokeslet3D([L, L, L], s.eps)
    kc3 = s.dimensionless_permeability()
    S, _, _ = PeriodicStokeslet2D(L, L, CENTRELINE_EPS_FACTOR).blocks(np.zeros((1, 2)), np.zeros((1, 2)))
    kc2 = S[0, 0] * L**2
    ks = [NetworkSolver3D(net, "surface", e).dimensionless_permeability() for e in (0.2, 0.4)]
    lay = dict(centres=np.array([[L / 2, L / 2]]), radii=np.array([1.0]), Lx=L, Ly=L)
    err = (4 * ks[0] - ks[1]) / 3 / reference_k(lay, 128) - 1
    ok = abs(kc3 / kc2 - 1) < 1e-6 and abs(err) < 2e-3
    return ok, f"centreline vs 2D one-blob {abs(kc3 / kc2 - 1):.1e}; surface error {100 * err:+.3f}%"


def tilted_exact(k_across, k_along):
    """Eq. (7.12): half of the flow is across the fibre, half along it; resistances add."""
    return 1.0 / (0.5 / k_across + 0.5 / k_along)


def check_tilted(exact=tilted_exact):
    """Fibres at 45 degrees (mixed axial and transverse flow): surface method within 0.5% of Eq. (7.12)."""
    alpha = 0.1
    net = g3.lattice_network("tilted", alpha)
    L = net["L"]
    lay = dict(centres=np.array([[L / np.sqrt(8), L / 2]]), radii=np.array([1.0]), Lx=L / np.sqrt(2), Ly=L)
    k_ex = exact(reference_k(lay, 128, U=(1.0, 0.0)), reference_k_axial(lay, 128))
    ks = [NetworkSolver3D(net, "surface", e).dimensionless_permeability() for e in (0.3, 0.6)]
    err = (4 * ks[0] - ks[1]) / 3 / k_ex - 1
    return abs(err) < 5e-3, f"tilted array, alpha = 0.1: error {100 * err:+.3f}%"


def check_validation_results(lattice_path=LATTICE_CSV, cubic_path=CUBIC_CSV):
    """stage4_validation.py results are complete and within tolerance: 24 lattice rows; Richardson
    error <= 0.5% (eps 0.2/0.4) and <= 1% (eps 0.3/0.6) for alpha <= 0.2; no-slip to round-off;
    cubic3: k_x = k_z and two extrapolations agree to 0.1%."""
    R = np.loadtxt(lattice_path, delimiter=",", skiprows=1, ndmin=2)
    C = np.loadtxt(cubic_path, delimiter=",", skiprows=1, ndmin=2)
    low = R[:, 1] <= 0.2
    e1 = np.max(np.abs(R[low & (R[:, 2] == 0.2), 6]))
    e2 = np.max(np.abs(R[low & (R[:, 2] == 0.3), 6]))
    slip = np.nanmax(R[:, 9])
    sym = np.max(np.abs(C[:, 4] / C[:, 3] - 1))
    fit = np.nanmax(np.abs(C[:, 1] / C[:, 2] - 1))
    ok = len(R) == 24 and e1 < 5e-3 and e2 < 1e-2 and slip < 1e-10 and sym < 1e-6 and fit < 1e-3
    return ok, (f"{len(R)} rows; errors {100 * e1:.2f}% and {100 * e2:.2f}%; slip {slip:.1e}; "
                f"cubic3 symmetry {sym:.1e}, extrapolations agree to {100 * fit:.3f}%")


def load_networks(pattern="networks_alpha*.csv"):
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, pattern))):
        with open(p, newline="") as fh:
            rows += list(csv.DictReader(fh))
    return rows


def check_network_results(rows=None):
    """Experiment (b) is complete (2 solid fractions x (12 + 20 + 20) networks), every network keeps
    its gaps, the permeability tensors are symmetric, and the aligned networks agree with the exact
    2D answer: each within 2.5% (the worst case of the 2D accuracy test for these blob widths was
    2.1%) and on average within 0.5%."""
    rows = load_networks() if rows is None else rows
    keys = {(r["kind"], r["alpha"], r["seed"]) for r in rows}
    al = [r for r in rows if r["kind"] == "aligned"]
    errs = np.array([abs(float(r["kz_R"]) / float(r["kz_exact"]) - 1) for r in al])
    gaps = min(float(r["min_gap"]) for r in rows)
    asym = max(float(r["K_asymmetry"]) for r in rows)
    ok = (len(rows) == 104 and len(keys) == 104 and len(al) == 24 and errs.max() < 0.025 and errs.mean() < 0.005
          and gaps >= 0.6 - 1e-9 and asym < 1e-3)
    return ok, (f"{len(rows)} networks, smallest gap {gaps:.2f}, K symmetric to {asym:.1e}, aligned-network error "
                f"largest {100 * errs.max():.2f}%, mean {100 * errs.mean():.2f}%")


def tex_sci(x):
    m, e = f"{float(x):.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def expected_table_rows():
    """Rows of Tables 6 and 7 exactly as main.tex must print them, rebuilt from the results."""
    rows = []
    names = {0: "across", 1: "along", 2: "tilted"}
    R = np.loadtxt(LATTICE_CSV, delimiter=",", skiprows=1, ndmin=2)
    for r in R[R[:, 2] == 0.2]:
        coarse = R[(R[:, 0] == r[0]) & (R[:, 1] == r[1]) & (R[:, 2] == 0.3)][0]
        rows.append(f"{names[int(r[0])]} & {r[1]:.2f} & {r[3]:.4f} & {100 * r[4]:+.2f} & {100 * r[6]:+.3f} & "
                    f"{100 * coarse[6]:+.3f} & {100 * r[7]:+.2f} & {100 * r[8]:+.2f} \\\\")
    with open(os.path.join(RES, "networks_summary.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(f"{float(r['alpha']):.2f} & {r['structure']} & {int(r['n_networks'])} & "
                        f"{float(r['k_mean']):.3f} $\\pm$ {float(r['k_ci95']):.3f} & {float(r['rule_4']):.3f} & "
                        f"{float(r['ratio_4']):.3f} [{float(r['ratio_4_lo']):.3f}, {float(r['ratio_4_hi']):.3f}] & "
                        f"{100 * float(r['centreline_err_mean']):+.1f} \\\\")
    return rows


def expected_inline_numbers():
    """Numbers quoted in the running text of Section 7, read from the summary files."""
    s = open(SUMMARY, encoding="utf-8").read()
    n = open(NET_SUMMARY, encoding="utf-8").read()
    want = [tex_sci(re.search(r"splitting width = (\S+) ", s).group(1)),
            tex_sci(re.search(r"Fourier series \(eps = 1.5, cell L x L x 1.3L\): largest relative difference = (\S+)", s).group(1)),
            tex_sci(re.search(r"residual / largest term <= (\S+),", s).group(1)),
            tex_sci(re.search(r"largest \|u\|/U = (\S+)", s).group(1)),
            tex_sci(re.search(r"one-blob model \(alpha = 0.1\): relative difference (\S+)", s).group(1))]
    want += re.findall(r"\(eps 0.2/0.4\) = ([\d.]+)% for alpha <= 0.2, ([\d.]+)% up to", s)[0]
    want += re.findall(r"\(eps 0.3/0.6\) = ([\d.]+)% for alpha", s)
    want += re.findall(r"is up to ([\d.]+)% \(alpha <= 0.2\); with it ([\d.]+)%", s)[0]
    for g in re.findall(r"cubic3 alpha = ([\d.]+): k/a\^2 = ([\d.]+) \(Richardson\)", s):
        want.append(g[1])
    want += re.findall(r"difference ([\d.]+)%; k_x", s)
    want += re.findall(r"([\d.]+) \(eps\^2 fit", s)
    want += re.findall(r"cubic3 .*centreline ([+-][\d.]+)%", s)                # both solid fractions
    want.append(tex_sci(re.search(r"series vs closed form at the switch: (\S+)", s).group(1)))
    want += re.findall(r"Drummond-Tahir series: differences ([\d.]+)%-([\d.]+)%", s)[0]
    want.append(tex_sci(re.search(r"change from M = 64 to 128 <= (\S+)", s).group(1)))
    want += re.findall(r"centreline model on lattices: error ([+-][\d.]+)% to ([+-][\d.]+)%", s)[0]
    want += re.findall(r"with balance ([+-][\d.]+) \(eps 0.4\), ([+-][\d.]+) \(eps 0.2\); without ([+-][\d.]+), "
                       r"([+-][\d.]+); inside flux with balance ([+-][\d.]+)% and ([+-][\d.]+)%", s)[0]
    for g in re.findall(r"largest Richardson error ([\d.]+)% \(eps 0.2/0.4\), ([\d.]+)% \(0.25/0.5\), "
                        r"([\d.]+)% \(0.3/0.6\)", s):
        want += list(g)
    want += re.findall(r"k changes by at most ([\d.]+)%", s)
    want.append(tex_sci(re.search(r"largest \|K - K\^T\|/\|K\| = (\S+)", s).group(1)))
    want += re.findall(r"largest system (\d+) blobs \((\d+) unknowns\): assembly (\d+) s, "
                       r"LU factorization (\d+) s, whole solve (\d+) s", s)[0]
    for pattern in [r"largest error ([\d.]+)%, mean ([\d.]+)%",
                    r"exact 2D k\): k_perp = ([\d.]+) \+- ([\d.]+); ratio to the 2D layouts ([\d.]+) \[([\d.]+), ([\d.]+)\]",
                    r"aligned 3D networks: k_z = ([\d.]+) \+- ([\d.]+)",
                    r"anisotropy k_in-plane / k_through = ([\d.]+) \+- ([\d.]+); rule predicts ([\d.]+)",
                    r"cubic random / JJ = ([\d.]+)\)", r"layered through / Kuwabara = ([\d.]+)\)",
                    r"Davies ([\d.]+)", r"random/lattice ([\d.]+)\)"]:
        for g in re.findall(pattern, n):                    # every solid fraction
            want += list(g) if isinstance(g, tuple) else [g]
    want.append(tex_sci(re.search(r"symmetric to (\S+) \(largest", n).group(1)))
    r16 = [float(x) for x in re.findall(r"rule \(16-fibre 2D\) [\d.]+: ratio ([\d.]+)", n)]
    want += [f"{min(r16):.3f}", f"{max(r16):.3f}"]
    worst = max(float(x) for x in re.findall(r"centreline error mean [+-][\d.]+%, worst ([\d.]+)%", n))
    want.append(f"{worst:.1f}")
    return want


def check_numbers_in_text(tex_path=TEX):
    """Every row of Tables 6-7 appears in main.tex exactly as rebuilt from the results, and the numbers
    quoted in the running text of Section 7 appear as whole tokens."""
    full = open(tex_path, encoding="utf-8").read()
    # only Section 7 counts: a number elsewhere in the paper must not satisfy the check by chance
    start = full.find("\\section{The 3D solver and the permeability of fibre networks}")
    end = full.find("\\section{", start + 10)                 # the next section
    tex = full[start:end] if 0 <= start < end else ""
    flat = re.sub(r"\s+", " ", tex)
    rows = expected_table_rows()
    missing_rows = [row for row in rows if re.sub(r"\s+", " ", row) not in flat]

    def present(w):
        return re.search(r"(?<![\d.])" + re.escape(w) + r"(?![\d])", tex) is not None
    nums = expected_inline_numbers()
    missing = [w for w in nums if not present(w)]
    ok = not missing_rows and not missing and len(rows) == 12 + 6
    msg = f"{len(rows)} table rows and {len(nums)} inline numbers match main.tex"
    return ok, msg if ok else f"missing rows: {missing_rows[:2]}; missing numbers: {missing}"


def check_summary_current(summary_path=SUMMARY):
    """summary.txt is current: recompute one of its lattice lines (flow along the fibres, alpha = 0.1)."""
    s = open(summary_path, encoding="utf-8").read()
    quoted = float(re.search(r"axial_square    alpha = 0.10: .*?Richardson error ([+-][\d.]+)% \(eps 0.2/0.4\)", s).group(1))
    err = axial_error()
    ok = abs(100 * err - quoted) < 6e-4
    return ok, f"recomputed {100 * err:+.3f}% (summary: {quoted:+.3f}%)"


def check_networks_summary_current(summary_path=NET_SUMMARY):
    """networks_summary.txt is exactly what scripts/stage4_summary.py computes from the CSV files now."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import stage4_summary as summ
    lines = summ.build()[1]
    stored = open(summary_path, encoding="utf-8").read().splitlines()
    same = stored == lines
    return same, "networks_summary.txt matches a fresh computation" if same else "networks_summary.txt is out of date"


def check_earlier_stages(checks=None):
    """Stage 1, 2 and 3 checks still pass."""
    checks = checks if checks is not None else check_stage3.CHECKS
    failed = [c.__name__ for c in checks if not c()[0]]
    return not failed, "all Stage 1-3 checks pass" if not failed else f"failures: {failed}"


CHECKS = [check_kernel3d, check_ewald3d, check_axis_distance, check_networks, check_disk_points,
          check_pressure_balance, check_transverse_and_centreline, check_tilted, check_validation_results,
          check_network_results, check_numbers_in_text, check_summary_current, check_networks_summary_current,
          check_earlier_stages]


# ============================================================================
# Broken inputs, one fault each
# ============================================================================

def broken_inputs(tmpdir):
    def kernel_wrong_dipole(r, eps):                     # dipole with the wrong sign in H1
        H1, H2 = k3.kernel_H_3d(r, eps)
        r = np.asarray(r, dtype=float)
        rr = np.where(r > 0, r, 1.0)                     # avoids 0/0 at the centre (no change there)
        return H1 - np.where(r > 0, eps**2 / rr**3, 0.0), H2

    class EwaldNoHasimoto(p3.PeriodicStokeslet3D):       # Fourier part missing the (1 + xi^2 k^2/4) factor
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            fac = 1.0 / (1.0 + self.xi**2 * self.k2 / 4.0)
            self.tensor_weights = [w * fac for w in self.tensor_weights]

    def distance_no_gcd(f1, f2, L):                      # forgets the gcd in the lattice spacing
        d1, d2 = np.array(f1["dirn"]), np.array(f2["dirn"])
        m = np.cross(d1, d2)
        if np.any(m != 0):
            spacing = L / np.linalg.norm(m)
            c = (np.asarray(f2["point"]) - np.asarray(f1["point"])) @ m / np.linalg.norm(m)
            return abs(c - spacing * np.round(c / spacing))
        return g3.axis_distance(f1, f2, L)

    def network_unbalanced(alpha, n, rng, kind, min_gap, balanced):   # ignores balanced=True
        return g3.random_network(alpha, n, rng, kind=kind, min_gap=min_gap, balanced=False)

    def disk_wrong_weights(net, spacing, shift):         # weights 10% too large in total
        P, w, ids = g3.disk_points(net, spacing, shift)
        return P, w * 1.1, ids

    def kernel_wrong_series(r, eps):                     # second Taylor coefficient of H1 10% off
        H1, H2 = k3.kernel_H_3d(r, eps)
        rho = np.asarray(r) / eps
        small = rho < k3.RHO_SERIES
        H1 = H1 + small * 0.1 * k3.C_H1[1] * rho**2 / (np.sqrt(np.pi) * eps)
        return H1, H2

    rows = load_networks()
    bad_rows = [dict(r) for r in rows]
    for r in bad_rows:
        if r["kind"] == "aligned":
            r["kz_R"] = str(float(r["kz_exact"]) * 1.03)
            break
    bad_lat = os.path.join(tmpdir, "lattice_bad.csv")
    txt = open(LATTICE_CSV).read().splitlines()
    parts = txt[2].split(",")
    parts[6] = "2.0e-02"
    txt[2] = ",".join(parts)
    open(bad_lat, "w").write("\n".join(txt) + "\n")
    bad_summary = os.path.join(tmpdir, "summary_bad.txt")
    s = open(SUMMARY, encoding="utf-8").read()
    q = re.search(r"(axial_square    alpha = 0.10: .*?Richardson error )([+-][\d.]+)%", s)
    open(bad_summary, "w", encoding="utf-8").write(s.replace(q.group(0), q.group(1) + "+0.500%"))
    bad_net = os.path.join(tmpdir, "networks_summary_bad.txt")
    nt = open(NET_SUMMARY, encoding="utf-8").read()
    open(bad_net, "w", encoding="utf-8").write(nt.replace("(4-fibre 2D)", "(4-fibre 2D) ", 1))
    bad_tex = os.path.join(tmpdir, "main_bad.tex")
    tex = open(TEX, encoding="utf-8").read()
    num = re.search(r"\(eps 0.2/0.4\) = ([\d.]+)% for alpha <= 0.2", s).group(1)
    open(bad_tex, "w", encoding="utf-8").write(tex.replace(num + "\\%", "9.99\\%"))

    return [
        ("3D kernel with the wrong dipole sign", lambda: check_kernel3d(kernel_wrong_dipole)),
        ("3D kernel with one Taylor coefficient 10% off", lambda: check_kernel3d(kernel_wrong_series)),
        ("Ewald sum without Hasimoto's factor", lambda: check_ewald3d(EwaldNoHasimoto)),
        ("axis distance without the gcd", lambda: check_axis_distance(distance_no_gcd)),
        ("networks that ignore balanced=True", lambda: check_networks(network_unbalanced)),
        ("disc weights adding up to 10% too much", lambda: check_disk_points(disk_wrong_weights)),
        ("no pressure balance", lambda: check_pressure_balance(balance=False)),
        ("flux inside the fibres counted as Darcy flux", lambda: check_pressure_balance(solver=NoInsideFluxSolver)),
        ("centreline blob width a instead of a exp(gamma/2)", lambda: check_transverse_and_centreline(1.0)),
        ("tilted answer from averaged permeabilities instead of resistances",
         lambda: check_tilted(lambda ka, kp: 0.5 * (ka + kp))),
        ("a lattice result with a 2% error", lambda: check_validation_results(bad_lat)),
        ("an aligned network 3% off the exact answer", lambda: check_network_results(bad_rows)),
        ("paper quoting a stale number", lambda: check_numbers_in_text(bad_tex)),
        ("summary.txt quoting a wrong error", lambda: check_summary_current(bad_summary)),
        ("networks_summary.txt edited by hand", lambda: check_networks_summary_current(bad_net)),
        ("an earlier check failing", lambda: check_earlier_stages([lambda: (False, "")])),
    ]


def main():
    import tempfile
    print("=" * 72)
    print("PART A: Stage 4 checks (all must PASS)")
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
    print("ALL STAGE 4 CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
