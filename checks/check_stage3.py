"""
check_stage3.py
===============
Every Stage 3 check must PASS, the Stage 1 and Stage 2 checks must still pass,
and every check must FAIL on a deliberately broken input (one fault at a time).

Run from the project folder (after scripts/stage3_validation.py,
scripts/stage3_random_arrays.py and scripts/stage3_random_summary.py):
    python checks/check_stage3.py          (about 2 minutes)
Exit code 0 = all good; 1 = something failed.
"""

import csv
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import geometry as geo                                                          # noqa: E402
from solver2d import FibreSolver2D, solve_with_rule, richardson_k, OFFSET_FACTOR   # noqa: E402
from reference_bie import ReferenceBIE2D, reference_k, log_weights              # noqa: E402
import check_stage1                                                              # noqa: E402
import check_stage2                                                              # noqa: E402

TEX = os.path.join(ROOT, "paper", "main.tex")
RES = os.path.join(ROOT, "results", "stage3")
SUMMARY = os.path.join(RES, "summary.txt")
RANDOM_CSV = os.path.join(RES, "random_arrays.csv")
RANDOM_SUMMARY = os.path.join(RES, "random_summary.txt")
CV_TEST = os.path.join(RES, "cv_test.txt")
SA_TABLE = [(0.05, 15.56), (0.10, 24.83), (0.20, 51.53), (0.30, 102.90), (0.40, 217.89), (0.50, 532.55)]


# ============================================================================
# The checks. Each returns (passed: bool, message: str).
# ============================================================================

def check_layouts(square=geo.square_array, hexa=geo.hexagonal_array, rand=geo.random_array):
    """Regular arrays have exactly the requested solid fraction; random arrays too, with every
    gap at least min_gap, and the same seed gives the same layout."""
    problems = []
    for alpha in [0.05, 0.3]:
        for f in (square, hexa):
            if abs(geo.solid_fraction(f(alpha)) - alpha) > 1e-12:
                problems.append(f"{f.__name__} alpha")
    l1 = rand(0.3, 16, np.random.default_rng(9), cv=0.5, min_gap=0.2)
    l2 = rand(0.3, 16, np.random.default_rng(9), cv=0.5, min_gap=0.2)
    if abs(geo.solid_fraction(l1) - 0.3) > 1e-12:
        problems.append("random alpha")
    if geo.min_surface_gap(l1) < 0.2 - 1e-12:
        problems.append(f"gap {geo.min_surface_gap(l1):.3f} < 0.2")
    if not np.allclose(l1["centres"], l2["centres"]):
        problems.append("not reproducible")
    # hexagonal: every fibre's nearest neighbour is at the lattice spacing
    h = hexa(0.2)
    c, L = h["centres"], np.array([h["Lx"], h["Ly"]])
    d = c[1] - c[0]
    d -= L * np.round(d / L)
    if abs(np.linalg.norm(d) - h["Lx"]) > 1e-12:
        problems.append("hexagonal spacing")
    return not problems, "solid fractions exact, gaps respected, seeds reproducible" if not problems else str(problems)


def check_log_quadrature(weights=log_weights):
    """Eq. (5.3): the weights integrate -ln|2 sin(t/2)| cos(n t) exactly: answer pi/n for n >= 1,
    for every mode the M points can represent, including the last one n = M/2."""
    M = 32
    W = weights(M)
    theta = 2 * np.pi * np.arange(M) / M
    errs = [abs(W[0] @ np.cos(n * theta) - np.pi / n) for n in range(1, M // 2 + 1)]
    errs.append(abs(W[0] @ np.ones(M)))                 # n = 0: integral of -ln|2 sin| is 0
    return max(errs) < 1e-12, f"largest error over modes 0..{M//2}: {max(errs):.1e}"


def check_reference_vs_table(kfun=reference_k):
    """The reference solver reproduces Sangani & Acrivos (1982) Table 1 to 3e-4."""
    worst = max(abs(np.pi / (al * kfun(geo.square_array(al), 32)) / F - 1) for al, F in SA_TABLE)
    return worst < 3e-4, f"largest relative difference from the table: {worst:.1e}"


def check_reference_spectral(kfun=reference_k):
    """Spectral convergence on a random polydisperse layout: 64 and 96 points per fibre agree
    to 1e-10 (32 points give about 1e-6, 48 about 1e-8: the error falls exponentially)."""
    lay = geo.random_array(0.2, 6, np.random.default_rng(4), cv=0.3, min_gap=0.3)
    k64, k96 = kfun(lay, 64), kfun(lay, 96)
    return abs(k64 / k96 - 1) < 1e-10, f"|k(64)/k(96) - 1| = {abs(k64 / k96 - 1):.1e}"


def check_reference_dilute(kfun=reference_k):
    """Hasimoto: C = 1.476336 - 2 alpha + ... ; at alpha = 1e-4 the reference gives 1.476136 to 1e-5."""
    alpha = 1e-4
    C = -(8 * alpha * kfun(geo.square_array(alpha), 32)) - np.log(alpha)
    return abs(C - (1.476336 - 2 * alpha)) < 1e-5, f"C = {C:.6f} (expect {1.476336 - 2 * alpha:.6f})"


def rs_error(alpha, eps, offset_factor=OFFSET_FACTOR, ratio=1.0):
    lay = geo.square_array(alpha)
    s = FibreSolver2D(lay, eps / ratio, eps, offset=offset_factor * eps)
    s.solve()
    return s.dimensionless_permeability() / reference_k(lay, 64) - 1


def check_offset_rule(offset_factor=OFFSET_FACTOR):
    """Proposition 6: with the offset eps/(2 sqrt(pi)) the error at eps = 0.1a, alpha = 0.1 is below 1%
    (without it, 5%). The test is numerical: a wrong offset fails because k comes out wrong."""
    e = rs_error(0.1, 0.1, offset_factor)
    return abs(e) < 0.01, f"offset factor {offset_factor:.6f}, error at eps = 0.1a: {100 * e:.2f}%"


def check_second_order(offset_factor=OFFSET_FACTOR):
    """With the rule the error is second order in eps (observed order 1.8-2.2 between 0.1a and 0.05a)."""
    e1, e2 = rs_error(0.3, 0.1, offset_factor), rs_error(0.3, 0.05, offset_factor)
    order = np.log2(e1 / e2)
    return 1.8 < order < 2.2, f"observed order {order:.2f}"


def check_richardson(kfun=None):
    """Richardson (Eq. 5.7) brings the square alpha = 0.3 error below 0.2% at eps = 0.1a."""
    lay = geo.square_array(0.3)
    kR = (kfun or (lambda l, e: richardson_k(l, e)[0]))(lay, 0.1)
    e = kR / reference_k(lay, 64) - 1
    return abs(e) < 2e-3, f"Richardson error {100 * e:.3f}%"


def check_leakage(ratio=1.0):
    """At eps = h the fibre is closed: positive-part and chord leakage below 1e-5, net flux 0,
    and chord equals positive-part flux (mass conservation)."""
    lay = geo.square_array(0.1)
    s = FibreSolver2D(lay, 0.1 / ratio, 0.1, offset=OFFSET_FACTOR * 0.1)
    s.solve()
    net, pos, chord = s.leakage()
    ok = abs(net[0]) < 1e-12 and pos[0] < 1e-5 and abs(chord[0] / pos[0] - 1) < 0.05
    return ok, f"net {net[0]:.0e}, Q+ {pos[0]:.1e}, chord {chord[0]:.1e}"


def check_gmres_matches_dense(method="gmres"):
    """GMRES and the dense solve give the same permeability."""
    lay = geo.random_array(0.2, 4, np.random.default_rng(2), min_gap=0.3)
    s = FibreSolver2D(lay, 0.15, 0.15, offset=OFFSET_FACTOR * 0.15)
    s.solve(method="dense")
    k0 = s.dimensionless_permeability()
    s.solve(method=method, tol=1e-11)
    k1 = s.dimensionless_permeability()
    return abs(k1 / k0 - 1) < 1e-8, f"|k_gmres/k_dense - 1| = {abs(k1 / k0 - 1):.1e}"


def check_random_results(path=RANDOM_CSV):
    """Experiment (a) is complete (4 solid fractions x 3 spreads x 40 layouts = 480 realizations)
    and the RS results agree with the reference: every one within 1.5%, on average within 0.2%."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    errs = np.array([abs(float(r["rs_rel_error"])) for r in rows])
    keys = {(r["alpha"], r["cv"], r["seed"]) for r in rows}
    ok = len(rows) == 480 and len(keys) == 480 and errs.max() < 0.015 and errs.mean() < 0.002
    return ok, f"{len(rows)} realizations, largest RS error {100 * errs.max():.2f}%"


def tex_sci(x):
    """1.8e-04 -> 1.8\\times10^{-4} (how the paper writes small numbers)."""
    m, e = f"{float(x):.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def expected_table_rows():
    """The rows of Tables 3, 4 and 5 exactly as main.tex must print them, rebuilt from the results."""
    rows = []
    with open(os.path.join(RES, "reference_vs_sangani_acrivos.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(f"{float(r['alpha']):.2f} & {float(r['F_over_muU_SA1982']):.2f} & "
                        f"{float(r['F_over_muU_reference']):.3f} & ${tex_sci(r['rel_diff'])}$ & "
                        f"{float(r['k_over_a2_reference']):.5f} \\\\")
    txt = open(RANDOM_SUMMARY, encoding="utf-8").read()
    for g in re.findall(r"alpha = ([\d.]+): random monodisperse k/a\^2 = ([\d.]+) \+- ([\d.]+) .*median ([\d.]+), "
                        r"sd/mean ([\d.]+); square ([\d.]+), hexagonal ([\d.]+), Kuwabara ([\d.]+), Davies ([\d.]+); "
                        r"random/square = ([\d.]+) \[([\d.]+), ([\d.]+)\]", txt):
        rows.append(f"{float(g[0]):.2f} & {g[1]} $\\pm$ {g[2]} & {g[3]} & {g[4]} & {g[5]} & {g[6]} & {g[7]} & {g[8]} & "
                    f"{g[9]} [{g[10]}, {g[11]}] \\\\")
    for g in re.findall(r"alpha = ([\d.]+), CV = ([\d.]+): k_poly/k_mono \(mean radius\) = ([\d.]+) \[([\d.]+), ([\d.]+)\], "
                        r"Prop. 7 predicts ([\d.]+) -> \w+; \(r.m.s. radius\) = ([\d.]+) \[([\d.]+), ([\d.]+)\]; "
                        r"H = ([\d.]+), first-order correction predicts ([\d.]+)", txt):
        rows.append(f"{float(g[0]):.2f} & {float(g[1]):g} & {g[2]} [{g[3]}, {g[4]}] & {g[5]} & {g[6]} [{g[7]}, {g[8]}] & "
                    f"{g[9]} & {g[10]} \\\\")
    return rows


def expected_inline_numbers():
    """Numbers quoted in the running text of Sections 5-6, read from the summary files."""
    s = open(SUMMARY, encoding="utf-8").read()
    r = open(RANDOM_SUMMARY, encoding="utf-8").read()
    c = open(CV_TEST, encoding="utf-8").read()
    want = [tex_sci(re.search(r"largest relative difference = (\S+) ", s).group(1))]
    want += re.findall(r"error ([\d.]+)% at alpha = 0.1, ([\d.]+)% at alpha = 0.3", s)[0]
    want.append(re.search(r"hexagonal reference: C = ([\d.]+)", s).group(1))
    want += re.findall(r"-> ([\d.]+) as eps.*next term ([\d.]+) eps", s)[0]
    want += re.findall(r"error (-[\d.]+)% without offset, (-[\d.]+)% with the rule, (-[\d.]+)% with Richardson", s)[0]
    want += re.findall(r"([\d.]+) without offset, ([\d.]+) with the rule", s)[0]
    want += re.findall(r"<= ([\d.]+)% at eps = 0.1a with the rule, <= ([\d.]+)% with Richardson", s)[0]
    m = re.search(r"leakage Q\+ = (\S+), condition number = (\S+); eps/h = [\d.]+: condition number = (\S+)", s)
    want += [tex_sci(m.group(1)), tex_sci(m.group(2)), tex_sci(m.group(3))]
    want.append(re.search(r"fitted c = ([\d.]+)", s).group(1))
    want += re.findall(r"0.2a: ([\d.]+)% \(eps 0.1a\), ([\d.]+)% \(0.05a\), ([\d.]+)% \(0.025a\)", s)[0]
    want += re.findall(r"<= ([\d.]+)% at eps = 0.1a, <= ([\d.]+)% with Richardson", s)[0]
    want += re.findall(r"realizations = ([\d.]+)%; mean ([\d.]+)%, 95th percentile ([\d.]+)%", r)[0]
    want += re.findall(r"monodisperse ([\d.]+)-([\d.]+), polydisperse ([\d.]+)-([\d.]+)", r)[0]
    m = re.search(r"inside the 95% interval in (\d+) of (\d+) cases", r)
    want.append(f"{m.group(1)} of {m.group(2)}")
    want += re.findall(r"first 10 layouts per case: k_poly/k_mono \(mean radius\) ranges ([\d.]+)-([\d.]+)", c)[0]
    want += re.findall(r"first 40 layouts: CV = 0 gives k/a\^2 = ([\d.]+) \+- ([\d.]+), CV = 0.01 gives ([\d.]+) \+- ([\d.]+)", c)[0]
    return want


def check_numbers_in_text(tex_path=TEX):
    """Every table row of Tables 3-5 appears in main.tex exactly as rebuilt from the results, and every
    number quoted in the running text of Sections 5-6 appears as a whole token."""
    tex = open(tex_path, encoding="utf-8").read()
    flat = re.sub(r"\s+", " ", tex)
    rows = expected_table_rows()
    missing_rows = [row for row in rows if re.sub(r"\s+", " ", row) not in flat]

    def present(w):
        # match the number as a whole token: "1.01" must not be found inside "1.013"
        return re.search(r"(?<![\d.])" + re.escape(w) + r"(?![\d])", tex) is not None
    nums = expected_inline_numbers()
    missing = [w for w in nums if not present(w)]
    ok = not missing_rows and not missing and len(rows) == 6 + 4 + 8
    msg = f"{len(rows)} table rows and {len(nums)} inline numbers match main.tex"
    return ok, msg if ok else f"missing rows: {missing_rows[:2]}; missing numbers: {missing}"


def check_summary_current(summary_path=SUMMARY):
    """summary.txt is current: recompute two of its numbers (alpha = 0.3 square, eps = 0.05a)."""
    s = open(summary_path, encoding="utf-8").read()
    quoted = [float(x) for x in re.findall(r"error (-[\d.]+)% without offset, (-[\d.]+)% with the rule", s)[0]]
    e_plain = rs_error(0.3, 0.05, 0.0)
    e_rule = rs_error(0.3, 0.05)
    ok = abs(100 * e_plain - quoted[0]) < 0.06 and abs(100 * e_rule - quoted[1]) < 0.006
    return ok, f"recomputed {100 * e_plain:.2f}% and {100 * e_rule:.3f}% (summary: {quoted[0]}%, {quoted[1]}%)"


def check_random_summary_current(summary_path=RANDOM_SUMMARY):
    """random_summary.txt is exactly what scripts/stage3_random_summary.py computes from the CSV now."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import stage3_random_summary as summ
    lines = summ.build()[1]
    stored = open(summary_path, encoding="utf-8").read().splitlines()
    same = stored == lines
    return same, "random_summary.txt matches a fresh computation" if same else "random_summary.txt is out of date"


def check_earlier_stages(checks=None):
    """Stage 1 and Stage 2 checks still pass."""
    checks = checks if checks is not None else (check_stage1.CHECKS + check_stage2.CHECKS[:-1])
    failed = [c.__name__ for c in checks if not c()[0]]
    return not failed, "all Stage 1 and Stage 2 checks pass" if not failed else f"failures: {failed}"


CHECKS = [check_layouts, check_log_quadrature, check_reference_vs_table, check_reference_spectral,
          check_reference_dilute, check_offset_rule, check_second_order, check_richardson, check_leakage,
          check_gmres_matches_dense, check_random_results, check_numbers_in_text, check_summary_current,
          check_random_summary_current, check_earlier_stages]


# ============================================================================
# Broken inputs, one fault each
# ============================================================================

def broken_inputs(tmpdir):
    def square_wrong(alpha, a=1.0):                     # cell side from diameter instead of radius
        L = np.sqrt(np.pi * (2 * a) ** 2 / alpha)
        return dict(centres=np.array([[L / 2, L / 2]]), radii=np.array([a]), Lx=L, Ly=L)

    def weights_no_last_term(M):                         # drops the n = M/2 term of Eq. (5.3)
        t = 2 * np.pi * (np.arange(M)[:, None] - np.arange(M)[None, :]) / M
        m = np.arange(1, M // 2)
        return 2 * np.pi / M * (np.cos(t[..., None] * m) @ (1.0 / m))

    def reference_k_no_correction(layout, M):            # drops the smooth (Fourier) part of the periodic kernel
        r = ReferenceBIE2D(layout, M)
        r.periodic.weight = r.periodic.weight * 0.0       # removes the Fourier part of the periodic kernel
        r.solve()
        return r.dimensionless_permeability()

    def reference_k_low_res(layout, M):                  # the 64-point solve secretly uses 8 points
        return reference_k(layout, 8 if M == 64 else M)

    def reference_k_hexagonal(layout, M):                # wrong geometry: hexagonal instead of square
        alpha = geo.solid_fraction(layout)
        return reference_k(geo.hexagonal_array(alpha), M)

    # random results file with one bad realization
    with open(RANDOM_CSV, newline="") as fh:
        rows = list(csv.DictReader(fh))
    bad_csv = os.path.join(tmpdir, "random_bad.csv")
    rows[0] = dict(rows[0], rs_rel_error="0.05")
    with open(bad_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    bad_summary = os.path.join(tmpdir, "summary_bad.txt")
    txt = open(SUMMARY, encoding="utf-8").read()
    quoted = re.findall(r"error (-[\d.]+)% without offset", txt)[0]
    open(bad_summary, "w", encoding="utf-8").write(txt.replace(quoted + "% without offset", "-9.9% without offset"))
    bad_rsummary = os.path.join(tmpdir, "random_summary_bad.txt")
    rtxt = open(RANDOM_SUMMARY, encoding="utf-8").read()
    open(bad_rsummary, "w", encoding="utf-8").write(rtxt.replace("n = 40", "n = 41", 1))
    bad_tex = os.path.join(tmpdir, "main_bad.tex")
    tex = open(TEX, encoding="utf-8").read()
    rnum = re.search(r"realizations = ([\d.]+)%", open(RANDOM_SUMMARY).read()).group(1)
    open(bad_tex, "w", encoding="utf-8").write(tex.replace(rnum + "\\%", "9.99\\%"))

    return [
        ("square cell sized with the diameter", lambda: check_layouts(square=square_wrong)),
        ("log weights missing the last term", lambda: check_log_quadrature(weights_no_last_term)),
        ("reference without the Fourier part of the periodic kernel",
         lambda: check_reference_vs_table(reference_k_no_correction)),
        ("reference whose 64-point solve uses 8 points", lambda: check_reference_spectral(reference_k_low_res)),
        ("dilute test run on the wrong (hexagonal) array", lambda: check_reference_dilute(reference_k_hexagonal)),
        ("no offset (blobs on the surface)", lambda: check_offset_rule(0.0)),
        ("offset 1/sqrt(pi) instead of 1/(2 sqrt(pi))", lambda: check_second_order(1 / np.sqrt(np.pi))),
        ("Richardson with the wrong weights (k(eps) only)",
         lambda: check_richardson(lambda l, e: solve_with_rule(l, e).dimensionless_permeability())),
        ("blobs too sparse (eps/h = 0.5)", lambda: check_leakage(0.5)),
        ("an iterative solve stopped early",
         lambda: check_gmres_matches_dense_stopped()),
        ("random results with a 5% error", lambda: check_random_results(bad_csv)),
        ("paper quoting a stale number", lambda: check_numbers_in_text(bad_tex)),
        ("summary.txt quoting a wrong error", lambda: check_summary_current(bad_summary)),
        ("random_summary.txt edited by hand", lambda: check_random_summary_current(bad_rsummary)),
        ("an earlier check failing", lambda: check_earlier_stages([lambda: (False, "")])),
    ]


def check_gmres_matches_dense_stopped():
    """Broken input helper: GMRES with a loose tolerance must NOT match to 1e-8."""
    lay = geo.random_array(0.2, 4, np.random.default_rng(2), min_gap=0.3)
    s = FibreSolver2D(lay, 0.15, 0.15, offset=OFFSET_FACTOR * 0.15)
    s.solve(method="dense")
    k0 = s.dimensionless_permeability()
    s.solve(method="gmres", tol=1e-2)
    k1 = s.dimensionless_permeability()
    return abs(k1 / k0 - 1) < 1e-8, ""


def main():
    import tempfile
    print("=" * 72)
    print("PART A: Stage 3 checks (all must PASS)")
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
    print("ALL STAGE 3 CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
