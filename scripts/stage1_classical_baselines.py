"""
stage1_classical_baselines.py
=============================
Stage 1: tabulate and plot the classical formulas the paper will test.

Outputs
-------
results/stage1/permeability_baselines.csv   dimensionless permeability k/a^2 vs alpha
results/stage1/efficiency_baseline.csv      single-fibre efficiencies and penetration vs particle size
paper/figures/stage1_permeability.pdf       Figure 1 of the Stage 1 write-up
paper/figures/stage1_efficiency.pdf         Figure 2 of the Stage 1 write-up

Run from the project folder:   python scripts/stage1_classical_baselines.py
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")            # draw to files only; no window needed
import matplotlib.pyplot as plt

# Let Python find our own module in the src/ folder
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import classical_models as cm  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "stage1")
FIGURES = os.path.join(ROOT, "paper", "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Permeability baselines over the solid fractions in the study (0.05 - 0.3)
# ----------------------------------------------------------------------------
alpha = np.linspace(0.02, 0.30, 57)   # stop at 0.30: Davies and the series are not valid beyond
a = 1.0   # we report k / a^2, so the radius value does not matter here
models = {
    "Kuwabara (cell model)": cm.k_kuwabara,
    "Happel (cell model)": cm.k_happel,
    "Square array (Sangani-Acrivos / Drummond-Tahir)": cm.k_drummond_tahir_square,
    "Hexagonal array (dilute series)": cm.k_drummond_tahir_hex,
    "Davies (empirical, real filters)": cm.k_davies,
    "Jackson-James (3D random)": cm.k_jackson_james_3d,
}
table = np.column_stack([alpha] + [f(alpha, a) / a**2 for f in models.values()])
header = "alpha," + ",".join(m.replace(",", "") for m in models)
np.savetxt(os.path.join(RESULTS, "permeability_baselines.csv"), table,
           delimiter=",", header=header, comments="", fmt="%.6e")

fig, ax = plt.subplots(figsize=(6.0, 4.2))
styles = ["-", "--", "-.", ":", "-", "--"]
for (name, f), ls in zip(models.items(), styles):
    ax.semilogy(alpha, f(alpha, a) / a**2, ls, lw=1.8, label=name)
ax.axvspan(0.05, 0.30, color="0.92", zorder=0, label="range studied")
ax.set_xlabel(r"solid volume fraction $\alpha$")
ax.set_ylabel(r"dimensionless permeability $k/a^2$")
ax.set_title("Classical permeability formulas (to be tested)")
ax.legend(fontsize=7.5, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "stage1_permeability.pdf"))
plt.close(fig)

# ----------------------------------------------------------------------------
# 2. Capture efficiency baseline for one typical uncharged microfibre layer
# ----------------------------------------------------------------------------
a_f, alpha0, L, U0 = 1.5e-6, 0.10, 0.5e-3, 0.10   # radius, solid fraction, thickness, face velocity
d_f = 2 * a_f
d_p = np.logspace(np.log10(0.01e-6), np.log10(2e-6), 400)
E_R = cm.eta_interception(d_p, d_f, alpha0)
E_D = cm.eta_diffusion(d_p, d_f, U0, alpha0)
E_I = cm.eta_impaction(d_p, d_f, U0, alpha0)
E_DR = cm.eta_diffusion_interception(d_p, d_f, U0, alpha0)
E = cm.eta_total(d_p, d_f, U0, alpha0)
P = cm.penetration(d_p, d_f, U0, alpha0, L)
np.savetxt(os.path.join(RESULTS, "efficiency_baseline.csv"),
           np.column_stack([d_p, E_R, E_D, E_DR, E_I, E, P]), delimiter=",",
           header="d_p_m,E_interception,E_diffusion,E_diff_interception,E_impaction,E_total,penetration",
           comments="", fmt="%.6e")

d_mpps, P_max = cm.most_penetrating_size(d_f, U0, alpha0, L)
k = cm.k_kuwabara(alpha0, a_f)
dP = cm.pressure_drop(k, U0, L)

fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.loglog(d_p * 1e6, E_D, "--", label="diffusion")
ax.loglog(d_p * 1e6, E_R, "-.", label="interception")
ax.loglog(d_p * 1e6, E_DR, linestyle=(0, (5, 1, 1, 1)), label="diffusion-interception")
ax.loglog(d_p * 1e6, np.maximum(E_I, 1e-6), ":", label="impaction")
ax.loglog(d_p * 1e6, E, "k-", lw=2, label="total")
ax.axvline(d_mpps * 1e6, color="0.5", lw=1)
ax.text(d_mpps * 1e6 * 1.08, 2e-3, f"MPPS = {d_mpps*1e6:.2f} $\\mu$m", fontsize=8)
ax.set_ylim(1e-4, 2)
ax.set_xlabel(r"particle diameter $d_p$ ($\mu$m)")
ax.set_ylabel(r"single-fibre efficiency $E$")
ax.set_title(r"Classical single-fibre efficiency, $d_f=3\,\mu$m, $\alpha=0.1$, $U_0=0.1$ m/s")
ax.legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, "stage1_efficiency.pdf"))
plt.close(fig)

# A small summary file that main.tex numbers are copied from
with open(os.path.join(RESULTS, "summary.txt"), "w") as fh:
    fh.write(f"fibre diameter d_f = {d_f*1e6:.1f} um, alpha = {alpha0}, L = {L*1e3} mm, U0 = {U0} m/s\n")
    fh.write(f"Kuwabara permeability k = {k:.3e} m^2, pressure drop = {dP:.1f} Pa\n")
    fh.write(f"MPPS = {d_mpps*1e6:.3f} um, penetration at MPPS = {P_max:.3f}\n")
    fh.write(f"quality factor at MPPS = {cm.quality_factor(P_max, dP):.4e} 1/Pa\n")

print(open(os.path.join(RESULTS, "summary.txt")).read())
print("Wrote CSVs to results/stage1 and figures to paper/figures")
