"""
check_stage1.py
===============
Every check must PASS before Stage 1 is delivered.

The second half of this file proves each check CAN FAIL: we feed it a
deliberately broken input (one fault at a time) and make sure it says "fail".
A check that can never fail would be useless.

Run from the project folder:   python checks/check_stage1.py
Exit code 0 = all good; 1 = something failed.
"""

import csv
import os
import re
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import classical_models as cm  # noqa: E402

TABLE = os.path.join(ROOT, "literature", "literature_table.csv")
BIB = os.path.join(ROOT, "paper", "refs.bib")
TEX = os.path.join(ROOT, "paper", "main.tex")
README = os.path.join(ROOT, "README.md")
SUMMARY = os.path.join(ROOT, "results", "stage1", "summary.txt")

REQUIRED_COLUMNS = ["key", "authors", "year", "title", "venue", "volume_pages",
                    "doi_or_url", "theme", "role_in_paper", "verification"]
ALLOWED_VERIFICATION = {"verified-web", "partly-verified-web", "from-memory"}

# \cite, \citep, \citet, \citep*, \citealt, \nocite, with an optional [..] argument
CITE_RE = re.compile(r"\\(?:no)?cite[a-z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")


# ============================================================================
# The checks. Each returns (passed: bool, message: str).
# They take their inputs as arguments so we can also feed them broken inputs.
# ============================================================================

def check_cunningham(cc_func=cm.cunningham):
    """Hinds (1999) tabulates Cc = 2.85 at 0.1 um and 1.15 at 1 um. His Eq. 3.20,
    which we code, gives about 3% more at 0.1 um, so we allow 3.5% there."""
    c01, c1 = float(cc_func(0.1e-6)), float(cc_func(1e-6))
    ok = abs(c01 - 2.85) / 2.85 < 0.035 and abs(c1 - 1.15) / 1.15 < 0.02
    return ok, f"Cc(0.1um) = {c01:.3f} (table 2.85), Cc(1um) = {c1:.3f} (table 1.15)"


def check_diffusion(d_func=cm.diffusion_coefficient):
    """Hinds tabulates D = 6.75e-10 m^2/s at 0.1 um. D is proportional to Cc, so the
    same ~3% offset carries over; allow 4%."""
    D = float(d_func(0.1e-6))
    return abs(D - 6.75e-10) / 6.75e-10 < 0.04, f"D(0.1um) = {D:.3e} m^2/s (table 6.75e-10)"


def check_dilute_limit(k_cell=cm.k_kuwabara, k_exact=cm.k_drummond_tahir_square):
    """As alpha -> 0, the Kuwabara cell model and the exact square-array result agree."""
    alpha = 1e-4
    ratio = float(k_cell(alpha, 1.0) / k_exact(alpha, 1.0))
    return abs(ratio - 1.0) < 0.01, f"k_Kuwabara / k_square at alpha=1e-4: {ratio:.4f} (expect 1 within 1%)"


def check_square_at_high_alpha(k_square=cm.k_drummond_tahir_square):
    """At alpha = 0.3 the square-array series and the Kuwabara cell model should still
    agree within 15% (cell models work well for ORDERED arrays). Dropping the
    alpha^3 term makes the series ~35% too small, which this catches."""
    ratio = float(k_square(0.3, 1.0) / cm.k_kuwabara(0.3, 1.0))
    return abs(ratio - 1.0) < 0.15, f"k_square / k_Kuwabara at alpha=0.3: {ratio:.3f} (expect within 15% of 1)"


def check_davies_absolute(k_func=cm.k_davies):
    """Radius/diameter guard: with a = 1 (so d_f = 2), Davies gives
    k = 4 / (64 alpha^1.5 (1+56 alpha^3)) = 1 / (16 alpha^1.5 (1+56 alpha^3))."""
    alpha = 0.1
    expect = 1.0 / (16.0 * alpha**1.5 * (1.0 + 56.0 * alpha**3))
    got = float(k_func(alpha, 1.0))
    return abs(got / expect - 1.0) < 1e-12, f"k_Davies(0.1, a=1) = {got:.6f} (expect {expect:.6f})"


def check_monotone(funcs=None):
    """Every permeability formula must DECREASE as more fibre is packed in."""
    funcs = funcs or [cm.k_kuwabara, cm.k_happel, cm.k_drummond_tahir_square,
                      cm.k_drummond_tahir_hex, cm.k_davies, cm.k_jackson_james_3d]
    alpha = np.linspace(0.05, 0.30, 50)
    bad = [f.__name__ for f in funcs if not np.all(np.diff(f(alpha, 1.0)) < 0)]
    return len(bad) == 0, "all decreasing in alpha" if not bad else f"not decreasing: {bad}"


def check_happel_above_kuwabara(k_hap=cm.k_happel, k_kuw=cm.k_kuwabara, alpha=0.1):
    """Known ordering: Happel's free-surface cell gives a larger permeability than Kuwabara's."""
    kh, kk = float(k_hap(alpha, 1.0)), float(k_kuw(alpha, 1.0))
    return kh > kk, f"k_Happel/a^2 = {kh:.3f} > k_Kuwabara/a^2 = {kk:.3f}"


def check_mpps(pen_func=cm.penetration):
    """For an uncharged microfibre layer the most penetrating size should be 0.05-0.5 um,
    and penetration must lie strictly between 0 and 1 for every size."""
    d = np.logspace(-8, np.log10(2e-6), 1500)
    P = pen_func(d, 3e-6, 0.10, 0.10, 0.5e-3)
    d_star = d[int(np.argmax(P))]
    interior = 0 < np.argmax(P) < len(d) - 1
    ok = bool(interior and 0.05e-6 <= d_star <= 0.5e-6 and np.all((P > 0) & (P < 1)))
    return ok, f"MPPS = {d_star*1e6:.3f} um, interior minimum of efficiency: {interior}"


def baseline_numbers():
    """Recompute the Stage 1 baseline (same inputs as the figure script)."""
    a_f, alpha0, L, U0 = 1.5e-6, 0.10, 0.5e-3, 0.10
    k = cm.k_kuwabara(alpha0, a_f)
    dP = cm.pressure_drop(k, U0, L)
    d_mpps, P_max = cm.most_penetrating_size(2 * a_f, U0, alpha0, L)
    return dict(k=k, dP=dP, d_mpps=d_mpps, P=P_max, qF=cm.quality_factor(P_max, dP))


def check_regression(numbers=None, summary_path=SUMMARY):
    """Tight regression: the numbers in results/stage1/summary.txt must equal a fresh
    recomputation to 0.5%. This catches any silent change in the formulas."""
    numbers = numbers or baseline_numbers()
    text = open(summary_path, encoding="utf-8").read()
    stored = dict(
        k=float(re.search(r"k = ([\d.e+-]+) m\^2", text).group(1)),
        dP=float(re.search(r"pressure drop = ([\d.]+) Pa", text).group(1)),
        d_mpps=float(re.search(r"MPPS = ([\d.]+) um", text).group(1)) * 1e-6,
        P=float(re.search(r"penetration at MPPS = ([\d.]+)", text).group(1)),
        qF=float(re.search(r"quality factor at MPPS = ([\d.e+-]+)", text).group(1)),
    )
    bad = [name for name in stored if abs(stored[name] / numbers[name] - 1) > 0.005]
    return not bad, "summary.txt matches recomputation" if not bad else f"mismatch in {bad}"


def check_numbers_in_text(numbers=None, paths=(TEX, README)):
    """The rounded baseline numbers quoted in main.tex and README.md must match the code."""
    n = numbers or baseline_numbers()
    wanted = [f"{n['k']:.2e}".split("e")[0],          # e.g. 2.81
              f"{n['dP']:.0f}",                        # e.g. 323
              f"{n['d_mpps']*1e6:.2f}",                # e.g. 0.21
              f"{n['P']:.2f}",                         # e.g. 0.36
              f"{n['qF']:.1e}".split("e")[0]]          # e.g. 3.2
    missing = []
    for path in paths:
        text = open(path, encoding="utf-8").read()
        missing += [f"{os.path.basename(path)}:{w}" for w in wanted if w not in text]
    return not missing, f"quoted numbers {wanted} found" if not missing else f"not found: {missing}"


def read_table(table_path):
    with open(table_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return (reader.fieldnames or []), list(reader)


def check_table(table_path=TABLE):
    """Literature table: required columns present, no empty cells, unique keys,
    allowed verification labels."""
    cols, rows = read_table(table_path)
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        problems.append(f"missing columns {missing}")
    keys = [r.get("key", "") for r in rows]
    if len(keys) != len(set(keys)):
        problems.append("duplicate keys")
    for r in rows:
        for c in REQUIRED_COLUMNS:
            if c in r and not (r[c] or "").strip():
                problems.append(f"{r.get('key')}: empty '{c}'")
        if r.get("verification") not in ALLOWED_VERIFICATION:
            problems.append(f"{r.get('key')}: bad verification label '{r.get('verification')}'")
    n_confirm = sum(r.get("doi_or_url") == "TO-CONFIRM" for r in rows)
    msg = f"{len(rows)} references; {n_confirm} DOI(s) still TO-CONFIRM (allowed until Stage 7)"
    return (not problems), (msg if not problems else "; ".join(problems[:5]))


def check_cross_references(table_path=TABLE, bib_path=BIB, tex_path=TEX):
    """Every table key is in refs.bib, every bib key is cited in main.tex,
    and every citation in main.tex exists in refs.bib."""
    _, rows = read_table(table_path)
    table_keys = {r["key"] for r in rows}
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", open(bib_path, encoding="utf-8").read()))
    tex = open(tex_path, encoding="utf-8").read()
    cited = set()
    for group in CITE_RE.findall(tex):
        cited.update(k.strip() for k in group.split(",") if k.strip())
    problems = []
    if table_keys != bib_keys:
        problems.append(f"table/bib mismatch: {sorted(table_keys ^ bib_keys)[:5]}")
    if bib_keys - cited:
        problems.append(f"never cited: {sorted(bib_keys - cited)[:5]}")
    if cited - bib_keys:
        problems.append(f"cited but missing from bib: {sorted(cited - bib_keys)[:5]}")
    return (not problems), (f"{len(cited)} keys cited, all consistent" if not problems else "; ".join(problems))


def check_ai_statement(tex_path=TEX, table_path=TABLE):
    """main.tex must contain the AI-use statement naming tool and maker, and must NOT
    claim that all references were checked while some are still unconfirmed."""
    tex = open(tex_path, encoding="utf-8").read()
    _, rows = read_table(table_path)
    unconfirmed = sum(r["doi_or_url"] == "TO-CONFIRM" or r["verification"] != "verified-web" for r in rows)
    has_statement = ("use of AI tools" in tex) and ("Claude" in tex) and ("Anthropic" in tex)
    overclaims = unconfirmed > 0 and "All references were checked" in tex
    ok = has_statement and not overclaims
    if not has_statement:
        return False, "AI-use statement missing"
    return ok, ("AI-use statement present and consistent with the reference table"
                if ok else f"statement claims all references checked but {unconfirmed} are not")


def check_figures_float_H(tex_path=TEX):
    """House style: float package loaded and every figure/table placed with [H]."""
    tex = open(tex_path, encoding="utf-8").read()
    envs = re.findall(r"\\begin\{(figure\*?|table\*?)\}(\[[^\]]*\])?", tex)
    ok = "\\usepackage{float}" in tex and len(envs) > 0 and all(opt == "[H]" for _, opt in envs)
    return ok, f"{len(envs)} floats, all [H]" if ok else "float package missing or a float without [H]"


CHECKS = [check_cunningham, check_diffusion, check_dilute_limit, check_square_at_high_alpha,
          check_davies_absolute, check_monotone, check_happel_above_kuwabara, check_mpps,
          check_regression, check_numbers_in_text, check_table, check_cross_references,
          check_ai_statement, check_figures_float_H]


# ============================================================================
# "Can the checks fail?" -- one deliberate fault per broken input
# ============================================================================

def write_table(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def broken_inputs(tmpdir):
    """Return (name, callable) pairs that MUST come back as failures."""
    cols, rows = read_table(TABLE)
    real_tex = open(TEX, encoding="utf-8").read()

    def table_with(change):
        """Copy of the real table with exactly one fault applied."""
        new_cols, new_rows = list(cols), [dict(r) for r in rows]
        new_cols, new_rows = change(new_cols, new_rows)
        path = os.path.join(tmpdir, f"table_{len(os.listdir(tmpdir))}.csv")
        write_table(path, new_cols, new_rows)
        return path

    def tex_with(text):
        path = os.path.join(tmpdir, f"tex_{len(os.listdir(tmpdir))}.tex")
        open(path, "w", encoding="utf-8").write(text)
        return path

    empty_cell = table_with(lambda c, r: (c, [dict(r[0], doi_or_url="")] + r[1:]))
    bad_label = table_with(lambda c, r: (c, [dict(r[0], verification="trust-me")] + r[1:]))
    dup_key = table_with(lambda c, r: (c, r + [dict(r[0])]))
    no_column = table_with(lambda c, r: ([x for x in c if x != "theme"], r))
    extra_key = table_with(lambda c, r: (c, r + [dict(r[0], key="ExtraKey2099")]))
    unconfirmed = table_with(lambda c, r: (c, [dict(r[0], doi_or_url="TO-CONFIRM")] + r[1:]))

    fake_cite = tex_with(real_tex.replace("\\end{document}", "\\cite{NotARealPaper2099}\n\\end{document}"))
    no_ai = tex_with(real_tex.replace("use of AI tools", "use of tools").replace("Anthropic", "a company"))
    overclaim = tex_with(real_tex.replace("\\end{document}", "All references were checked.\n\\end{document}"))
    not_H = tex_with(real_tex.replace("\\begin{figure}[H]", "\\begin{figure}[t]", 1))

    good = baseline_numbers()
    shifted = dict(good, P=good["P"] * 1.02)          # a 2% change in penetration

    return [
        ("Cunningham with sign error", lambda: check_cunningham(lambda d: 1 - (0.066e-6 / d) * 2.5)),
        ("diffusion 10% too high", lambda: check_diffusion(lambda d: 1.10 * cm.diffusion_coefficient(d))),
        ("dilute limit vs Happel (wrong model)", lambda: check_dilute_limit(cm.k_happel)),
        ("square array without the alpha^3 term",
         lambda: check_square_at_high_alpha(lambda al, a: a**2 * (np.log(1 / al) - 1.476 + 2 * al
                                                                    - 1.774 * al**2) / (8 * al))),
        ("Davies with radius used as diameter",
         lambda: check_davies_absolute(lambda al, a: a**2 / (64 * al**1.5 * (1 + 56 * al**3)))),
        ("increasing permeability", lambda: check_monotone([lambda al, a: al * a**2])),
        ("Happel and Kuwabara swapped", lambda: check_happel_above_kuwabara(cm.k_kuwabara, cm.k_happel)),
        ("penetration without diffusion (no MPPS)",
         lambda: check_mpps(lambda d, df, U, al, L: np.exp(-4 * al * cm.eta_interception(d, df, al) * L
                                                          / (np.pi * (1 - al) * df)))),
        ("summary.txt 2% off the code", lambda: check_regression(shifted)),
        ("text quoting an old penetration value", lambda: check_numbers_in_text(dict(good, P=0.40))),
        ("table: one empty cell", lambda: check_table(empty_cell)),
        ("table: bad verification label", lambda: check_table(bad_label)),
        ("table: duplicate key", lambda: check_table(dup_key)),
        ("table: missing column", lambda: check_table(no_column)),
        ("table has a key the bib lacks", lambda: check_cross_references(extra_key, BIB, TEX)),
        ("tex cites a paper not in the bib", lambda: check_cross_references(TABLE, BIB, fake_cite)),
        ("tex without AI statement", lambda: check_ai_statement(no_ai, TABLE)),
        ("AI statement over-claims while a DOI is unconfirmed", lambda: check_ai_statement(overclaim, unconfirmed)),
        ("one figure placed with [t] instead of [H]", lambda: check_figures_float_H(not_H)),
    ]


def main():
    print("=" * 72)
    print("PART A: the real checks (all must PASS)")
    print("=" * 72)
    all_ok = True
    for chk in CHECKS:
        ok, msg = chk()
        all_ok &= bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {chk.__name__}: {msg}")

    print("=" * 72)
    print("PART B: broken inputs, one fault each (every check must FAIL on these)")
    print("=" * 72)
    with tempfile.TemporaryDirectory() as tmpdir:
        for name, run in broken_inputs(tmpdir):
            ok, _ = run()
            caught = not ok
            all_ok &= caught
            print(f"[{'PASS' if caught else 'FAIL'}] check correctly rejects: {name}")

    print("=" * 72)
    print("ALL STAGE 1 CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
