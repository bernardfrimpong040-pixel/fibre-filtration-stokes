"""
rs_kernels3d.py
===============
Our own THREE-dimensional regularized Stokeslet with a Gaussian blob
(main.tex Section 7.1; derived and checked symbolically with sympy).

Blob:      phi(r) = exp(-r^2/eps^2) / (pi^{3/2} eps^3)          (integrates to 1 over space)
Velocity:  8 pi mu u = H1(r) f + H2(r) (f . x) x
with rho = r / eps and
    H1 = erf(rho)/r + eps^2 erf(rho)/(2 r^3) - eps exp(-rho^2)/(sqrt(pi) r^2)
    H2 = erf(rho)/r^3 - 3 eps^2 erf(rho)/(2 r^5) + 3 eps exp(-rho^2)/(sqrt(pi) r^4)
Far away (rho >> 1): H1 -> 1/r + eps^2/(2 r^3), H2 -> 1/r^3 - 3 eps^2/(2 r^5):
the ordinary Stokeslet plus a potential dipole of strength eps^2/2 (as in 2D).
At the centre: H1(0) = 8/(3 sqrt(pi) eps), H2(0) = 8/(15 sqrt(pi) eps^3) (finite).

Close to the centre the closed form loses digits by cancellation, so for rho < 0.25
we use the exact Taylor series (coefficients from sympy, scripts in main.tex appendix).
"""

import numpy as np
from scipy.special import erf

SQRT_PI = np.sqrt(np.pi)
RHO_SERIES = 0.25          # below this, use the Taylor series

# Taylor coefficients (times sqrt(pi)) of H1 eps, H2 eps^3 in powers rho^0, rho^2, ..., rho^14
C_H1 = np.array([2.6666666666666667, -1.0666666666666667, 0.34285714285714286, -0.084656084656084656,
                 0.016835016835016835, -0.0027972027972027972, 0.00039886039886039886, -4.9797696856520386e-5])
C_H2 = np.array([0.53333333333333333, -0.22857142857142857, 0.063492063492063492, -0.013468013468013468,
                 0.0023310023310023310, -0.00034188034188034188, 4.3572984749455338e-5, -4.9142464003145118e-6])
# and of the derivatives with respect to the squared width: dH1/d(s^2) s^3, dH2/d(s^2) s^5
C_D1 = np.array([-1.3333333333333333, 1.6000000000000000, -0.85714285714285714, 0.29629629629629630,
                 -0.075757575757575758, 0.015384615384615385, -0.0025925925925925926, 0.00037348272642390289])
C_D2 = np.array([-0.80000000000000000, 0.57142857142857143, -0.22222222222222222, 0.060606060606060606,
                 -0.012820512820512821, 0.0022222222222222222, -0.00032679738562091503, 4.1771094402673350e-5])


def _series(coeffs, rho2):
    """sum_k coeffs[k] rho^(2k) / sqrt(pi), by Horner's rule."""
    out = np.zeros_like(rho2)
    for c in coeffs[::-1]:
        out = out * rho2 + c
    return out / SQRT_PI


def kernel_H_3d(r, eps):
    """H1(r), H2(r) of the Gaussian regularized Stokeslet (velocity times 8 pi mu)."""
    r = np.asarray(r, dtype=float)
    rho = r / eps
    H1 = np.empty_like(r)
    H2 = np.empty_like(r)
    small = rho < RHO_SERIES
    big = ~small
    rs, rb = rho[small], r[big]
    H1[small] = _series(C_H1, rs**2) / eps
    H2[small] = _series(C_H2, rs**2) / eps**3
    E = erf(rho[big])
    G = eps * np.exp(-rho[big] ** 2) / SQRT_PI
    H1[big] = E / rb + eps**2 * E / (2 * rb**3) - G / rb**2
    H2[big] = E / rb**3 - 1.5 * eps**2 * E / rb**5 + 3 * G / rb**4
    return H1, H2


def dH_ds2_3d(r, s):
    """Derivatives dH1/d(s^2), dH2/d(s^2) of the kernel of width s (used by the Ewald split)."""
    r = np.asarray(r, dtype=float)
    rho = r / s
    D1 = np.empty_like(r)
    D2 = np.empty_like(r)
    small = rho < RHO_SERIES
    big = ~small
    rs, rb = rho[small], r[big]
    D1[small] = _series(C_D1, rs**2) / s**3
    D2[small] = _series(C_D2, rs**2) / s**5
    E = erf(rho[big])
    X = np.exp(-rho[big] ** 2) / SQRT_PI
    D1[big] = -2 * X / s**3 - X / (rb**2 * s) + E / (2 * rb**3)
    D2[big] = 2 * X / (rb**2 * s**3) + 3 * X / (rb**4 * s) - 1.5 * E / rb**5
    return D1, D2


def singular_H_3d(r):
    """The ordinary 3D Stokeslet in the same form: H1 = 1/r, H2 = 1/r^3."""
    r = np.asarray(r, dtype=float)
    return 1.0 / r, 1.0 / r**3


def blob_3d(r, eps):
    return np.exp(-(np.asarray(r) / eps) ** 2) / (np.pi**1.5 * eps**3)


def velocity_3d(x_eval, x_src, forces, eps, mu=1.0):
    """Free-space velocity (M, 3) at x_eval from regularized point forces (N, 3) at x_src."""
    d = x_eval[:, None, :] - x_src[None, :, :]
    r = np.sqrt(np.sum(d**2, axis=2))
    H1, H2 = kernel_H_3d(r, eps)
    fdotd = np.einsum("mnk,nk->mn", d, forces)
    u = H1 @ forces + np.einsum("mn,mnk->mk", H2 * fdotd, d)
    return u / (8 * np.pi * mu)


def pressure_3d(x_eval, x_src, forces, eps):
    """Pressure p = f . grad G with G = -erf(r/eps)/(4 pi r):
    p = (f . x) [erf(rho) - (2/sqrt(pi)) rho exp(-rho^2)] / (4 pi r^3)."""
    d = x_eval[:, None, :] - x_src[None, :, :]
    r = np.sqrt(np.sum(d**2, axis=2))
    rho = r / eps
    # [erf(rho) - 2 rho exp(-rho^2)/sqrt(pi)] / r^3 loses digits by cancellation near 0, so for
    # rho < 0.1 we use its series  (2/(sqrt(pi) eps^3)) (2/3 - 2 rho^2/5 + rho^4/7 - rho^6/27 + rho^8/132)
    small = rho < 0.1
    rho2 = rho**2
    series = 2 / (SQRT_PI * eps**3) * (2 / 3 - 2 * rho2 / 5 + rho2**2 / 7 - rho2**3 / 27 + rho2**4 / 132)
    rs = np.where(small, 1.0, r)                       # avoids 0/0 in the branch not used
    closed = (erf(rho) - 2 / SQRT_PI * rho * np.exp(-rho2)) / rs**3
    ratio = np.where(small, series, closed)
    fdotd = np.einsum("mnk,nk->mn", d, forces)
    return np.sum(fdotd * ratio, axis=1) / (4 * np.pi)
