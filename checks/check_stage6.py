"""
check_stage6.py
===============
Every Stage 6 check must PASS, the Stage 1-5 checks must still pass,
and every check must FAIL on a deliberately broken input (one fault at a time).

Run from the project folder (after scripts/stage6_design.py and scripts/stage6_summary.py):
    python checks/check_stage6.py          (about 10 minutes)
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
import classical_models as cm                                                   # noqa: E402
import stage5_capture as sc                                                     # noqa: E402
import stage6_design as d6                                                      # noqa: E402
import stage6_summary as s6                                                     # noqa: E402
import check_stage5                                                             # noqa: E402

TEX = os.path.join(ROOT, "paper", "main.tex")
RES = os.path.join(ROOT, "results", "stage6")
SUMMARY = os.path.join(RES, "summary.txt")
RES5 = os.path.join(ROOT, "results", "stage5")


# ============================================================================
# The checks. Each returns (passed: bool, message: str).
# ============================================================================

def check_classical_cap(eta_total=cm.eta_total):
    """Each classical term is capped at 1 before Hinds' product rule: for 1.5 um fibres the penetration
    of large particles stays (essentially) zero, the total stays in [0, 1], and the Stage 1 numbers
    (MPPS 0.207 um, P = 0.357 for the 3 um filter) are unchanged."""
    d = np.logspace(-8, np.log10(2e-6), 400)
    E = eta_total(d, 1.5e-6, 0.1, 0.1)
    P = np.exp(-4 * 0.1 * E * 0.5e-3 / (np.pi * 0.9 * 1.5e-6))
    big = P[d > 0.8e-6].max()
    i = np.argmax(cm.penetration(d, 3e-6, 0.1, 0.1, 0.5e-3))
    dm, Pm = cm.most_penetrating_size(3e-6, 0.1, 0.1, 0.5e-3)
    ok = np.all((E >= 0) & (E <= 1)) and big < 1e-6 and abs(dm * 1e6 - 0.207) < 0.001 and abs(Pm - 0.357) < 0.001
    return ok, f"largest penetration above 0.8 um (1.5 um fibres) {big:.1e}; Stage 1 MPPS {dm * 1e6:.3f} um, P {Pm:.3f}"


def check_particle_scaling(params=sc.particle_parameters):
    """The dimensionless particle parameters follow their definitions when the design changes:
    defaults = Stage 5; r_p = d_p/d_f; Pe = d_f U / D_p; St = 2 Stk proportional to rho_p U / d_f; D = 2/Pe."""
    base = params(0.25)
    ok = np.allclose(base, (0.25e-6 / 3e-6, float(cm.peclet(0.25e-6, 3e-6, 0.1)),
                            float(cm.stokes_number(0.25e-6, 3e-6, 0.1)),
                            2 * float(cm.stokes_number(0.25e-6, 3e-6, 0.1)), 2 / float(cm.peclet(0.25e-6, 3e-6, 0.1))))
    r_p, Pe, Stk, St, D = params(0.25, 1.5e-6, 0.2, 912.0)
    ok &= np.isclose(r_p, 2 * base[0]) and np.isclose(Pe, base[1]) and np.isclose(D, 2 / Pe)
    ok &= np.isclose(St, base[3] * (912.0 / cm.RHO_NACL) * 2 * 2)          # rho ratio, x2 for U, x2 for 1/d_f
    return bool(ok), f"Stage 5 defaults and the scaling with d_f, U and rho_p hold (St = {St:.4f})"


def check_q_formula(q_factor=s6.q_factor):
    """q_F = alpha eta k d_f / (pi mu U) equals -ln P / dp computed the long way, for random designs."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(50):
        eta, k_a2, d_f, U, alpha, L = (rng.uniform(0.01, 0.5), rng.uniform(0.5, 5), rng.uniform(1e-6, 1e-5),
                                       rng.uniform(0.02, 0.3), rng.uniform(0.03, 0.2), rng.uniform(1e-4, 1e-3))
        P = np.exp(-4 * alpha * eta * L / (np.pi * d_f))                      # Eq. (8.6)
        dp = cm.MU_AIR * U * L / (k_a2 * (d_f / 2) ** 2)                      # Darcy
        worst = max(worst, abs(q_factor(eta, k_a2, d_f, U, alpha) / (-np.log(P) / dp) - 1))
    return worst < 1e-12, f"largest relative difference {worst:.1e}"


def check_curve_min(curve_min=s6.curve_min):
    """The MPPS rule is exact for a parabola in (ln d, ln y), and reports a minimum at the grid end."""
    d = np.array(d6.SIZES_UM)
    x0 = np.log(0.25)
    y = np.exp(0.4 * (np.log(d) - x0) ** 2 - 3.0)
    dm, ym, edge = curve_min(d, y)
    _, _, edge2 = curve_min(d, d)                                             # increasing: minimum at the start
    ok = abs(dm - 0.25) < 1e-9 and abs(ym - np.exp(-3.0)) < 1e-12 and not edge and edge2
    return ok, f"vertex {dm:.6f} um (exact 0.25), value {ym:.6e} (exact {np.exp(-3.0):.6e}), end detected = {edge2}"


def check_proposition10(stack_Q=s6.stack_Q):
    """Proposition 10 on 300 random pairs of layer curves: (a) the stack's q at every size equals
    -ln(P1 P2) / (dp1 + dp2) for independent layers; (b) Q(w) is concave and Q(w) >= w Q1 + (1-w) Q2;
    (c) the stack beats its better layer only if the other layer beats it at that layer's MPPS."""
    rng = np.random.default_rng(10)
    d = np.array(d6.SIZES_UM)
    bad_a = bad_b = bad_c = 0
    for _ in range(300):
        q1 = np.exp(rng.uniform(0.3, 1.5) * (np.log(d) - rng.uniform(-2.2, -0.5)) ** 2 + rng.normal(0, 0.3))
        q2 = np.exp(rng.uniform(0.3, 1.5) * (np.log(d) - rng.uniform(-2.2, -0.5)) ** 2 + rng.normal(0, 0.3))
        dp1, dp2 = rng.uniform(0.1, 1.0, 2) * 3 / max(q1.max(), q2.max())   # penetrations of order 0.05-1
        w = dp1 / (dp1 + dp2)
        P1, P2 = np.exp(-q1 * dp1), np.exp(-q2 * dp2)                         # independent layers multiply
        bad_a += not np.allclose(-np.log(P1 * P2) / (dp1 + dp2), w * q1 + (1 - w) * q2, rtol=1e-10)
        Q = stack_Q(q1, q2, d)
        own1, own2 = s6.fine(d, q1)[1].min(), s6.fine(d, q2)[1].min()        # each layer's own worst case
        bad_b += (np.diff(Q, 2).max() > 1e-12 * Q.max()) or not np.isclose(Q[-1], own1) \
            or not np.isclose(Q[0], own2) or np.any(Q < s6.W_GRID * own1 + (1 - s6.W_GRID) * own2 - 1e-12)
        best_is_1 = Q[-1] >= Q[0]
        _, fb = s6.fine(d, q1 if best_is_1 else q2)
        _, fo = s6.fine(d, q2 if best_is_1 else q1)
        cond = fo[np.argmin(fb)] > max(Q[0], Q[-1])
        bad_c += (Q.max() > max(Q[0], Q[-1]) * (1 + 1e-12)) and not cond
    return bad_a + bad_b + bad_c == 0, f"violations: (a) {bad_a}, (b) {bad_b}, (c) {bad_c} of 300"


def read(name, path=RES):
    with open(os.path.join(path, name), newline="") as fh:
        return list(csv.DictReader(fh))


def check_results_complete(path=RES):
    """All runs present: hexagonal, every design on its size window (stage6_design.DESIGN_SIZES_UM); random,
    the same for 6 cells; poly 3 x 16 cells x 5 sizes; sensitivity 6 x 2 + 2 x 4 x 2; every new cell passed its interception check (within 3.5 standard
    deviations of the stratified inlet's resolution, as in Stage 5); at most 1 particle in 100000 left in a
    cell at the time limit; every eta > 0."""
    hx, po, se = (read(f, path) for f in ("design_hexagonal.csv", "poly.csv", "sensitivity.csv"))
    rd = read("design_random1.csv", path) + read("design_random2.csv", path)
    n_win = sum(len(v) for v in d6.DESIGN_SIZES_UM.values())
    in_win = all(float(r["d_p_um"]) in d6.DESIGN_SIZES_UM[r["design"]] for r in hx + rd)
    n_ok = (in_win and len({(r["design"], r["alpha"], r["d_p_um"]) for r in hx}) == n_win == len(hx)
            and len({(r["design"], r["seed"], r["d_p_um"]) for r in rd}) == 6 * n_win == len(rd)
            and len({(r["seed"], r["d_p_um"]) for r in po}) == 3 * d6.N_POLY * len(d6.POLY_SIZES_UM) == len(po)
            and len({(r["case"], r["seed"], r["d_p_um"]) for r in se})
            == 6 * len(d6.SENS_SIZES_UM) + 2 * d6.N_BIG * len(d6.CELL_SIZES_UM) == len(se))
    chk = read("interception_checks_design_hexagonal.csv", path) + read("interception_checks_design_random1.csv", path) \
        + read("interception_checks_design_random2.csv", path) + read("interception_checks_poly.csv", path)
    z = [abs(float(c["f_tracker"]) - float(c["f_exact"])) * float(c["n_particles"]) / np.sqrt(2 * int(c["n_fibres"]) / 6)
         for c in chk]
    rows = hx + rd + po + se
    stuck = sum(int(r["n_stuck"]) for r in rows)
    tracked = sum(int(r["n_particles"]) for r in rows)
    ok = n_ok and len(chk) == 3 + 18 + 48 and max(z) < 3.5 and stuck <= 1e-5 * tracked and all(
        float(r["eta"]) > 0 for r in rows)
    return ok, (f"{len(hx)} + {len(rd)} + {len(po)} + {len(se)} rows, {len(chk)} cell checks (largest z-score "
                f"{max(z):.2f}), {stuck} of {tracked} particles stuck")


def check_same_cells_as_stage5(path=RES):
    """Experiment (d) at alpha = 0.1 uses the cells of Stage 5: the same exact permeability for each seed."""
    old = {(r["structure"], r["seed"]): float(r["k_over_a2"]) for r in read("capture_random.csv", RES5)
           + read("capture_hexagonal.csv", RES5)}
    rows = [r for r in read("design_random1.csv", path) + read("design_random2.csv", path)
            + read("design_hexagonal.csv", path) if float(r["alpha"]) == 0.1]
    cells = {(r["structure"], r["seed"]) for r in rows}
    worst = max(abs(float(r["k_over_a2"]) / old[(r["structure"], r["seed"])] - 1) for r in rows)   # every row
    return len(cells) == 7 and worst < 1e-9, f"{len(cells)} cells, largest difference in k {worst:.1e}"


def check_summary_current(summary_path=SUMMARY):
    """summary.txt is exactly what scripts/stage6_summary.py computes from the CSV files now."""
    lines = s6.build()["lines"]
    old = open(summary_path, encoding="utf-8").read().splitlines()
    diff = [a for a, b in zip(lines, old) if a != b] + (["(length)"] if len(lines) != len(old) else [])
    return not diff, "matches a fresh computation" if not diff else f"differs: {diff[0][:80]}"


# ---- what Section 9 of main.tex must print, rebuilt from summary.txt ----
LABEL_TEX = {"base": "Stage 1 filter", "d_f=1.5": r"$d_f = 1.5\,\mu$m", "d_f=6": r"$d_f = 6\,\mu$m",
             "U=0.05": r"$U = 0.05$\,m/s", "U=0.2": r"$U = 0.2$\,m/s", "alpha=0.05": r"$\alpha = 0.05$",
             "alpha=0.15": r"$\alpha = 0.15$", "DEHS": "DEHS aerosol"}
ELASTIC_TEX = {"fibre diameter d_f": r"fibre diameter $d_f$", "face velocity U": r"face velocity $U$",
               "solid fraction alpha": r"solid fraction $\alpha$", "particle density": r"particle density $\rho_p$"}
SENS_TEX = {"eps=0.1": r"blob width $\varepsilon = 0.1\bar a$", "eps=0.025": r"blob width $\varepsilon = 0.025\bar a$",
            "r_out=1.2a": r"near-wall annulus $r_{\rm out} = 1.2a$", "r_out=1.4a": r"near-wall annulus $r_{\rm out} = 1.4a$",
            "modes=32": "32 Fourier modes instead of 64"}


def milli(x):
    """q_F in units of 1e-3 1/Pa with three significant figures, from the summary's '2.46e-03' style."""
    m, e = x.split("e")
    return m if int(e) == -3 else ("0." + m.replace(".", "") if int(e) == -4 else f"{float(x) * 1e3:.3g}")


def expected_table_rows(summary_path=SUMMARY):
    """Rows of Tables 10-12 (design, elasticities, numerical sensitivity) exactly as main.tex must print them."""
    s = open(summary_path, encoding="utf-8").read()
    design, elastic, sens = [], [], []
    for nm, lab in LABEL_TEX.items():
        e = re.escape(nm)
        g = {st: re.search(rf"design {e}, {st}: .*?MPPS ([\d.]+) um.*?q_F (\S+) 1/Pa", s)
             for st in ("hexagonal", "random", r"classical \(Eq\. 8\.6\)")}
        r = re.search(rf"design {e}: q_F random / hexagonal ([\d.]+) \[([\d.]+), ([\d.]+)\]", s)
        h, rd, cl = (g[k] for k in g)
        design.append(f"{lab} & {h.group(1)} & {rd.group(1)} & {cl.group(1)} & {milli(h.group(2))} & "
                      f"{milli(rd.group(2))} & {milli(cl.group(2))} & {r.group(1)} [{r.group(2)}, {r.group(3)}] \\\\")
    for nm, lab in ELASTIC_TEX.items():
        m = re.search(rf"elasticity of q_F to {re.escape(nm)}: hexagonal (\S+) \[(\S+), (\S+)\], random (\S+) \[(\S+), "
                      rf"(\S+)\], classical (\S+)", s)
        v = m.groups()
        elastic.append(f"{lab} & ${v[0]}$ $[{v[1]}, {v[2]}]$ & ${v[3]}$ $[{v[4]}, {v[5]}]$ & ${v[6]}$ \\\\")
    for nm, lab in SENS_TEX.items():
        m = re.search(rf"sensitivity, hexagonal array, {re.escape(nm)} instead .*?changes by (\S+)% at \S+ um and (\S+)% at", s)
        sens.append(f"{lab} & ${m.group(1)}\\%$ & ${m.group(2)}\\%$ \\\\")
    return design, elastic, sens


def expected_inline_numbers(summary_path=SUMMARY):
    """Numbers quoted in the running text of Section 9, read from summary.txt."""
    s = open(summary_path, encoding="utf-8").read()
    want = re.findall(r"largest difference from the exact answer ([\d.]+) particles per fibre", s)
    m = re.search(r"hexagonal: MPPS / classical MPPS ([\d.]+)-([\d.]+), q_F at the MPPS / classical ([\d.]+)-([\d.]+)", s)
    want += [m.group(1), m.group(2), m.group(3), m.group(4)]
    r = sorted(float(x) for x in re.findall(r"q_F random / hexagonal ([\d.]+) \[([\d.]+)", s) for x in [x[0]]
               if float(x) > 1)
    want += [f"{r[0]:.3f}", f"{r[-1]:.3f}"]
    want += list(re.search(r"design alpha=0\.15: q_F random / hexagonal ([\d.]+) \[([\d.]+), ([\d.]+)\]", s).groups())
    want += list(re.search(r"design base: q_F random / hexagonal ([\d.]+) \[([\d.]+), ([\d.]+)\]", s).groups())
    want += re.findall(r"design alpha=0\.15, (?:hexagonal|random): k/a\^2 = ([\d.]+)", s)
    want += re.findall(r"design (?:base|DEHS), hexagonal: .*?MPPS ([\d.]+) um", s)
    m = re.search(r"layered filters \(Proposition 10\): (\d+) two-layer stacks; (\d+) meet condition \(c\) and (\d+) raise "
                  r".*?largest gain ([\d.]+)% \[([-\d.]+), ([-\d.]+)\] .*?condition \(c\) in (\d+) of (\d+)", s)
    want += list(m.groups())
    for st in ("hexagonal", "random", "classical"):
        m = re.search(rf"fine \(1\.5 um\) \+ coarse \(6 um\) layers, {st}: worst-case q_F (\S+) \(fine only\), \S+ "
                      rf"\(coarse only\), best stack (\S+) at a fine-layer share of the pressure drop ([\d.]+); coarse layer "
                      rf"at the fine layer's MPPS (\S+); gain ([-\d.]+)%( \[([-\d.]+), ([-\d.]+)\])?", s)
        want += [m.group(1).replace("e-03", ""), m.group(2).replace("e-03", ""), m.group(4).replace("e-03", ""), m.group(5)]
        if st != "classical":
            want += [m.group(3), m.group(7), m.group(8)]
    want += re.findall(r"binomial standard error of one hexagonal value at \S+ um ([\d.]+)%", s)
    for case in ("32 fibres, 2 cells long", "36 fibres, 1.5 x 1.5 cells"):
        m = re.search(rf"cell size, {re.escape(case)} .*?eta at 0\.22 um ([\d.]+) \[([\d.]+), ([\d.]+)\], eta at 0\.483 um "
                      rf"([\d.]+) \[([\d.]+), ([\d.]+)\], k ([\d.]+) \[([\d.]+), ([\d.]+)\]", s)
        want += list(m.groups())
    m = re.search(r"random cells at alpha = 0\.15: k/a\^2 of the six cells ([\d., ]+); harmonic mean ([\d.]+), arithmetic "
                  r"mean ([\d.]+) \(hexagonal ([\d.]+)\)", s)
    k6 = sorted(float(x) for x in m.group(1).split(","))
    want += [f"{k6[0]:.3f}", f"{k6[-1]:.3f}", m.group(2), m.group(3), m.group(4)]
    m = re.search(r"over 0\.05-0\.1 and over 0\.1-0\.15 \(secants\): hexagonal (\S+) and (\S+), random (\S+) and (\S+), "
                  r"classical (\S+) and (\S+)", s)
    want += [x.rstrip(",") for x in m.groups()]
    want += re.findall(r"largest share of draws whose MPPS falls at an end of the size window ([\d.]+)%", s)
    for cv in ("0.25", "0.5"):
        m = re.search(rf"fibre-size spread CV = {cv} / identical .*?eta at the MPPS (\S+) \[(\S+), (\S+)\], pressure drop "
                      rf"(\S+) \[(\S+), (\S+)\], q_F (\S+) \[(\S+), (\S+)\]", s)
        want += [x.rstrip(",") for x in m.groups()]
    want += re.findall(r"fibre-size spread CV = [\d.]+: k/a\^2 = ([\d.]+)", s)
    return want


def check_numbers_in_text(tex_path=TEX, summary_path=SUMMARY):
    """Every row of Tables 10-12 appears in Section 9 of main.tex as rebuilt from summary.txt, and the numbers
    quoted in the running text of Section 9 appear there as whole tokens."""
    full = open(tex_path, encoding="utf-8").read()
    start = full.find("\\section{Design study and sensitivity}")
    end = full.find("\\section{Methods: use of AI tools}")
    tex = full[start:end] if 0 <= start < end else ""
    flat = re.sub(r"\s+", " ", tex)
    rows = [r for group in expected_table_rows(summary_path) for r in group]
    missing_rows = [r for r in rows if re.sub(r"\s+", " ", r) not in flat]

    def present(w):
        return re.search(r"(?<![\d.])" + re.escape(w) + r"(?![\d])", tex) is not None
    nums = expected_inline_numbers(summary_path)
    missing = [w for w in nums if not present(w)]
    ok = bool(tex) and not missing_rows and not missing and len(rows) == 8 + 4 + 5
    return ok, (f"{len(rows)} table rows and {len(nums)} inline numbers match main.tex" if ok
                else f"missing rows: {missing_rows[:2]}; missing numbers: {missing[:8]}")


def check_earlier_stages(checks=None):
    """Stage 1-5 checks still pass."""
    checks = checks if checks is not None else check_stage5.CHECKS
    failed = [c.__name__ for c in checks if not c()[0]]
    return not failed, "all Stage 1-5 checks pass" if not failed else f"failures: {failed}"


CHECKS = [check_classical_cap, check_particle_scaling, check_q_formula, check_curve_min, check_proposition10,
          check_results_complete, check_same_cells_as_stage5, check_numbers_in_text, check_summary_current,
          check_earlier_stages]


# ============================================================================
# Broken inputs, one fault each
# ============================================================================

def broken_inputs(tmpdir):
    def eta_total_uncapped(d_p, d_f, U0, alpha):                 # the Stage 1 bug: terms not capped at 1
        E = 1.0 - (1.0 - cm.eta_interception(d_p, d_f, alpha)) * (1.0 - cm.eta_diffusion(d_p, d_f, U0, alpha)) \
            * (1.0 - cm.eta_diffusion_interception(d_p, d_f, U0, alpha)) * (1.0 - cm.eta_impaction(d_p, d_f, U0, alpha))
        return np.clip(E, 0.0, 1.0)

    def params_wrong_st(d, d_f=sc.D_F, U=sc.U0, rho_p=cm.RHO_NACL):  # St growing with d_f instead of 1/d_f
        r_p, Pe, Stk, St, D = sc.particle_parameters(d, d_f, U, rho_p)
        return r_p, Pe, Stk, St * (d_f / sc.D_F) ** 2, D

    def q_without_alpha(eta, k, d_f, U, alpha):
        return s6.q_factor(eta, k, d_f, U, alpha) / alpha

    def curve_min_wrong(d, y):                                    # vertex with the wrong sign
        dm, ym, e = s6.curve_min(d, y)
        return float(np.exp(2 * np.log(d[int(np.argmin(y))]) - np.log(dm))), ym, e

    def stack_unweighted(q1, q2, sizes):                         # ignores the pressure-drop shares
        _, f1 = s6.fine(sizes, q1)
        _, f2 = s6.fine(sizes, q2)
        return np.array([np.min(0.5 * (f1 + f2)) + 0 * w for w in s6.W_GRID])

    bad_dir = os.path.join(tmpdir, "res")
    os.makedirs(bad_dir)
    for name in ["design_hexagonal.csv", "design_random1.csv", "design_random2.csv", "poly.csv", "sensitivity.csv",
                 "interception_checks_design_hexagonal.csv", "interception_checks_design_random1.csv",
                 "interception_checks_design_random2.csv", "interception_checks_poly.csv"]:
        lines = open(os.path.join(RES, name), encoding="utf-8").read().splitlines()
        if name == "poly.csv":
            lines = lines[:-1]                                   # one run missing
        if name == "design_random1.csv":                         # a Stage 5 cell with a different permeability
            i = next(j for j, ln in enumerate(lines) if j > 0 and ln.split(",")[3] == "0.1")
            parts = lines[i].split(",")
            parts[20] = str(float(parts[20]) * 1.01)
            lines[i] = ",".join(parts)
        open(os.path.join(bad_dir, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    bad_tex = os.path.join(tmpdir, "main_bad.tex")
    tex = open(TEX, encoding="utf-8").read()
    i0, i1 = tex.find("\\section{Design study and sensitivity}"), tex.find("\\section{Methods: use of AI tools}")
    num = expected_inline_numbers()[-1]                          # replace every occurrence in Section 9
    open(bad_tex, "w", encoding="utf-8").write(tex[:i0] + tex[i0:i1].replace(num, "9.999") + tex[i1:])
    bad_summary = os.path.join(tmpdir, "summary.txt")
    txt = open(SUMMARY, encoding="utf-8").read()
    open(bad_summary, "w", encoding="utf-8").write(txt.replace("q_F", "q_F ", 1))
    return [
        ("classical terms not capped at 1 (the Stage 1 bug)", lambda: check_classical_cap(eta_total_uncapped)),
        ("Stokes number growing with the fibre diameter", lambda: check_particle_scaling(params_wrong_st)),
        ("quality factor without the solid fraction", lambda: check_q_formula(q_without_alpha)),
        ("MPPS parabola with the wrong vertex", lambda: check_curve_min(curve_min_wrong)),
        ("layered stack ignoring the pressure-drop shares", lambda: check_proposition10(stack_unweighted)),
        ("a polydispersity run missing", lambda: check_results_complete(bad_dir)),
        ("a cell not the same as in Stage 5", lambda: check_same_cells_as_stage5(bad_dir)),
        ("paper quoting a stale number", lambda: check_numbers_in_text(bad_tex)),
        ("summary.txt edited by hand", lambda: check_summary_current(bad_summary)),
        ("an earlier check failing", lambda: check_earlier_stages([lambda: (False, "")])),
    ]


def main():
    import tempfile
    print("=" * 72)
    print("PART A: Stage 6 checks (all must PASS)")
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
    print("ALL STAGE 6 CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
