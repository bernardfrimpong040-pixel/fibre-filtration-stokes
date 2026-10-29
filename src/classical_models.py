"""
classical_models.py
===================
The "classical" formulas that fibrous-filter design still relies on.

These are the BASELINES our regularized Stokeslet simulations must be compared
against in later stages. Nothing here is new science: every formula comes from a
textbook or paper listed in literature/literature_table.csv.

All quantities are in SI units (metres, seconds, pascals, kilograms).

Notation used throughout the project
------------------------------------
alpha : solid volume fraction (fraction of the filter volume that is fibre), 0 < alpha < 1
a     : fibre RADIUS  (m)
d_f   : fibre DIAMETER (m) = 2a
d_p   : particle diameter (m)
U0    : face velocity = volumetric flow rate / filter face area (m/s)
mu    : air viscosity (Pa s)
k     : Darcy permeability (m^2), defined by  pressure gradient = mu * U0 / k
"""

import numpy as np

# ----------------------------------------------------------------------------
# Physical constants for air at about 20 degC, 1 atm (Hinds 1999, Appendix)
# ----------------------------------------------------------------------------
MU_AIR = 1.81e-5          # dynamic viscosity of air, Pa s
MEAN_FREE_PATH = 0.066e-6 # mean free path of air molecules, m
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant, J/K
T_ROOM = 293.15           # temperature, K
RHO_NACL = 2165.0         # density of NaCl test aerosol particles, kg/m^3


# ============================================================================
# PART 1: PERMEABILITY (how easily air flows through the fibres)
# ============================================================================
# All cell models give the drag per unit length on one fibre as
#       F = 4 * pi * mu * U0 / H(alpha)
# where H is a "hydrodynamic factor" (Kuwabara or Happel).
#
# Turning drag into permeability (this is the step-by-step bridge):
#   * number of fibres per unit cross-section area  n = alpha / (pi a^2)
#   * force per unit volume on the fluid           = n * F
#   * Darcy's law: pressure gradient = mu U0 / k must balance that force
#   =>  mu U0 / k = (alpha / (pi a^2)) * (4 pi mu U0 / H)
#   =>  k = a^2 * H / (4 alpha)
# ----------------------------------------------------------------------------

def kuwabara_factor(alpha):
    """Kuwabara (1959) hydrodynamic factor Ku(alpha)."""
    alpha = np.asarray(alpha, dtype=float)
    return -0.5 * np.log(alpha) - 0.75 + alpha - 0.25 * alpha**2


def happel_factor(alpha):
    """Happel (1959) hydrodynamic factor Ha(alpha)."""
    alpha = np.asarray(alpha, dtype=float)
    return -0.5 * np.log(alpha) - 0.5 * (1.0 - alpha**2) / (1.0 + alpha**2)


def k_kuwabara(alpha, a):
    """Permeability from the Kuwabara cell model: k = a^2 Ku / (4 alpha)."""
    return a**2 * kuwabara_factor(alpha) / (4.0 * alpha)


def k_happel(alpha, a):
    """Permeability from the Happel cell model: k = a^2 Ha / (4 alpha)."""
    return a**2 * happel_factor(alpha) / (4.0 * alpha)


def k_drummond_tahir_square(alpha, a):
    """Square array of parallel cylinders, flow perpendicular to the fibres
    (Sangani & Acrivos 1982 series, as used by Drummond & Tahir 1984):
    k/a^2 = [ln(1/alpha) - 1.476 + 2 alpha - 1.774 alpha^2 + 4.076 alpha^3] / (8 alpha).
    The alpha^3 term matters: without it k is ~40% too small at alpha = 0.3."""
    alpha = np.asarray(alpha, dtype=float)
    return a**2 * (np.log(1.0 / alpha) - 1.476 + 2.0 * alpha - 1.774 * alpha**2
                   + 4.076 * alpha**3) / (8.0 * alpha)


def k_drummond_tahir_hex(alpha, a):
    """Hexagonal array, flow perpendicular to the fibres: leading terms of the dilute series,
    k/a^2 = [ln(1/alpha) - 1.4975 + 2 alpha - alpha^2/2] / (8 alpha).
    The constant 1.4975 is confirmed by our exact reference solver (main.tex Sec. 5.2) and
    agrees with Drummond & Tahir's 1.497. (An earlier version used 1.490, which is wrong.)
    Truncated series: treat values above alpha ~ 0.3 with care."""
    alpha = np.asarray(alpha, dtype=float)
    return a**2 * (np.log(1.0 / alpha) - 1.4975 + 2.0 * alpha - 0.5 * alpha**2) / (8.0 * alpha)


def k_davies(alpha, a):
    """Davies (1952/1973) empirical fit for real filters:
    pressure gradient = 64 mu U0 alpha^1.5 (1 + 56 alpha^3) / d_f^2
    =>  k = d_f^2 / (64 alpha^1.5 (1 + 56 alpha^3)),  with d_f = 2a."""
    alpha = np.asarray(alpha, dtype=float)
    d_f = 2.0 * a
    return d_f**2 / (64.0 * alpha**1.5 * (1.0 + 56.0 * alpha**3))


def k_jackson_james_3d(alpha, a):
    """Jackson & James (1986), 3D random (isotropic) fibre network:
    k/a^2 = 3 (-ln alpha - 0.931) / (20 alpha)."""
    alpha = np.asarray(alpha, dtype=float)
    return a**2 * 3.0 * (-np.log(alpha) - 0.931) / (20.0 * alpha)


def pressure_drop(k, U0, thickness, mu=MU_AIR):
    """Darcy's law across a filter of given thickness: dP = mu U0 L / k (Pa)."""
    return mu * U0 * thickness / k


# ============================================================================
# PART 2: SINGLE-FIBRE CAPTURE EFFICIENCY (how well each fibre catches particles)
# ============================================================================

def cunningham(d_p, lam=MEAN_FREE_PATH):
    """Cunningham slip correction Cc (Hinds 1999, Eq. 3.20):
    small particles 'slip' between air molecules, so they feel less drag."""
    d_p = np.asarray(d_p, dtype=float)
    return 1.0 + (lam / d_p) * (2.34 + 1.05 * np.exp(-0.39 * d_p / lam))


def diffusion_coefficient(d_p, mu=MU_AIR, T=T_ROOM):
    """Stokes-Einstein diffusion coefficient of a particle: D = k_B T Cc / (3 pi mu d_p)."""
    return K_BOLTZMANN * T * cunningham(d_p) / (3.0 * np.pi * mu * d_p)


def peclet(d_p, d_f, U0):
    """Peclet number Pe = d_f U0 / D (carried by the flow vs. spread by diffusion)."""
    return d_f * U0 / diffusion_coefficient(d_p)


def stokes_number(d_p, d_f, U0, rho_p=RHO_NACL, mu=MU_AIR):
    """Stokes number as defined by Hinds (1999): Stk = tau U0 / d_f, where the particle
    relaxation time is tau = rho_p d_p^2 Cc / (18 mu). So Stk = rho_p d_p^2 Cc U0 / (18 mu d_f)."""
    return rho_p * d_p**2 * cunningham(d_p) * U0 / (18.0 * mu * d_f)


def eta_interception(d_p, d_f, alpha):
    """Interception, Lee & Liu (1982): E_R = (1-alpha)/Ku * R^2/(1+R), R = d_p/d_f."""
    R = d_p / d_f
    return (1.0 - alpha) / kuwabara_factor(alpha) * R**2 / (1.0 + R)


def eta_diffusion(d_p, d_f, U0, alpha):
    """Diffusion, Lee & Liu (1982): E_D = 2.6 ((1-alpha)/Ku)^(1/3) Pe^(-2/3)."""
    Pe = peclet(d_p, d_f, U0)
    return 2.6 * ((1.0 - alpha) / kuwabara_factor(alpha))**(1.0 / 3.0) * Pe**(-2.0 / 3.0)


def eta_impaction(d_p, d_f, U0, alpha, rho_p=RHO_NACL):
    """Inertial impaction (Hinds 1999): E_I = Stk J / (2 Ku^2), with
    J = (29.6 - 28 alpha^0.62) R^2 - 27.5 R^2.8 for R < 0.4 and J = 2 for R >= 0.4."""
    R = d_p / d_f
    J_fit = (29.6 - 28.0 * alpha**0.62) * R**2 - 27.5 * R**2.8
    J = np.where(R < 0.4, J_fit, 2.0)
    return stokes_number(d_p, d_f, U0, rho_p) * J / (2.0 * kuwabara_factor(alpha)**2)


def eta_diffusion_interception(d_p, d_f, U0, alpha):
    """Extra capture when diffusion and interception act together
    (Hinds 1999, Eq. 9.24): E_DR = 1.24 R^(2/3) / sqrt(Ku Pe)."""
    R = d_p / d_f
    return 1.24 * R**(2.0 / 3.0) / np.sqrt(kuwabara_factor(alpha) * peclet(d_p, d_f, U0))


def eta_total(d_p, d_f, U0, alpha, rho_p=RHO_NACL):
    """Total single-fibre efficiency (Hinds 1999 rule):
    E = 1 - (1-E_R)(1-E_D)(1-E_DR)(1-E_I), capped at 1
    (a fibre cannot catch more than everything in its path)."""
    # each term is a fraction of the particles in the fibre's path, so it is capped at 1 before the
    # terms are combined (otherwise a term above 1, e.g. impaction of large particles on thin fibres,
    # would make its factor (1 - E) negative and flip the sign of the product)
    def cap(E):
        return np.clip(E, 0.0, 1.0)
    E = 1.0 - (1.0 - cap(eta_interception(d_p, d_f, alpha))) \
            * (1.0 - cap(eta_diffusion(d_p, d_f, U0, alpha))) \
            * (1.0 - cap(eta_diffusion_interception(d_p, d_f, U0, alpha))) \
            * (1.0 - cap(eta_impaction(d_p, d_f, U0, alpha, rho_p)))
    return np.clip(E, 0.0, 1.0)


def penetration(d_p, d_f, U0, alpha, thickness, rho_p=RHO_NACL):
    """Fraction of particles that pass through the whole filter (Hinds Eq. 9.12):
    P = exp( -4 alpha E L / (pi (1-alpha) d_f) )."""
    E = eta_total(d_p, d_f, U0, alpha, rho_p)
    return np.exp(-4.0 * alpha * E * thickness / (np.pi * (1.0 - alpha) * d_f))


def quality_factor(P, dP):
    """Filter quality factor q_F = -ln(P) / dP  (1/Pa). Higher is better:
    more particles caught per pascal of breathing resistance."""
    return -np.log(P) / dP


def most_penetrating_size(d_f, U0, alpha, thickness, d_min=0.01e-6, d_max=2e-6, n=2000, rho_p=RHO_NACL):
    """Scan particle sizes on a log grid and return (d_mpps, P_max)."""
    d_grid = np.logspace(np.log10(d_min), np.log10(d_max), n)
    P = penetration(d_grid, d_f, U0, alpha, thickness, rho_p)
    i = int(np.argmax(P))
    return d_grid[i], P[i]


if __name__ == "__main__":
    # Quick demonstration with a typical mechanical (uncharged) microfibre layer.
    a, alpha, L, U0 = 1.5e-6, 0.10, 0.5e-3, 0.10
    print("Classical baselines for a = 1.5 um, alpha = 0.10, L = 0.5 mm, U0 = 0.10 m/s")
    for name, f in [("Kuwabara", k_kuwabara), ("Happel", k_happel),
                    ("Drummond-Tahir square", k_drummond_tahir_square),
                    ("Davies", k_davies), ("Jackson-James 3D", k_jackson_james_3d)]:
        k = f(alpha, a)
        print(f"  {name:22s} k = {k:.3e} m^2   dP = {pressure_drop(k, U0, L):7.1f} Pa")
    d_mpps, P_max = most_penetrating_size(2 * a, U0, alpha, L)
    print(f"  Most penetrating particle size = {d_mpps*1e6:.3f} um, penetration there = {P_max:.3f}")
