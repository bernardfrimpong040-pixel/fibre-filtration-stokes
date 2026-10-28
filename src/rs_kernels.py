"""
rs_kernels.py
=============
Our own two-dimensional REGULARIZED STOKESLET with a Gaussian blob.

What this is (plain language)
-----------------------------
A Stokeslet is the flow made by pushing on the fluid at one point. A point push
gives an infinite velocity right at that point. The "regularized" Stokeslet
spreads the push over a small Gaussian blob of width eps, so the velocity is
finite everywhere. Far from the blob it is the ordinary Stokeslet.

Equations (numbers refer to main.tex, Section 4)
-----------------------------------------------
Blob (Eq. 4.2):      phi(r) = exp(-r^2/eps^2) / (pi eps^2)
Velocity (Eq. 4.9):  4 pi mu u = A(r) f + B(r) (f . x) x
Pressure (Eq. 4.10): p = (f . x) (1 - exp(-X)) / (2 pi r^2)
with X = r^2/eps^2 and
    A(r) = 1/2 [ -ln r^2 - E1(X) - 1 + g(X) ],     g(X) = (1 - exp(-X)) / X
    B(r) = [ 1 - g(X) ] / r^2
E1 is the exponential integral (scipy.special.exp1).

Here x is the vector FROM the blob centre TO the point where we evaluate.
"""

import numpy as np
from scipy.special import exp1

EULER_GAMMA = 0.5772156649015329


def _g(X):
    """g(X) = (1 - e^{-X}) / X, computed safely for small X (g -> 1 as X -> 0)."""
    X = np.asarray(X, dtype=float)
    out = np.ones_like(X)
    big = X > 1e-8
    out[big] = -np.expm1(-X[big]) / X[big]      # expm1 avoids round-off when X is small
    out[~big] = 1.0 - X[~big] / 2.0             # two-term Taylor series
    return out


def kernel_AB_2d(r, eps):
    """Return the scalar functions A(r) and B(r) of Eq. (4.9) for blob width eps.

    Both are finite at r = 0:  A(0) = 1/2 (-ln eps^2 + gamma),  B(0) = 1/(2 eps^2).
    """
    r = np.asarray(r, dtype=float)
    X = (r / eps) ** 2
    A = np.empty_like(r)
    B = np.empty_like(r)

    small = X < 1e-8                       # essentially at the blob centre
    big = ~small

    # ---- away from the centre: use the closed form directly ----
    Xb, rb = X[big], r[big]
    gb = _g(Xb)
    A[big] = 0.5 * (-np.log(rb**2) - exp1(Xb) - 1.0 + gb)
    # 1 - g(X) loses digits for small X; use its series there
    one_minus_g = np.where(Xb > 1e-3, 1.0 - gb,
                           Xb / 2 - Xb**2 / 6 + Xb**3 / 24 - Xb**4 / 120 + Xb**5 / 720)
    B[big] = one_minus_g / rb**2

    # ---- at the centre: Taylor series (ln r^2 + E1(X) -> ln eps^2 - gamma + X) ----
    Xs = X[small]
    A[small] = 0.5 * (-(np.log(eps**2) - EULER_GAMMA + Xs) - 1.0 + (1.0 - Xs / 2.0))
    B[small] = (0.5 - Xs / 6.0) / eps**2
    return A, B


def singular_AB_2d(r):
    """The ordinary (singular) 2D Stokeslet in the same form: A = -ln r - 1/2, B = 1/r^2.
    This is the eps -> 0 limit of kernel_AB_2d."""
    r = np.asarray(r, dtype=float)
    return -np.log(r) - 0.5, 1.0 / r**2


def stokeslet_blocks_2d(x_eval, x_src, eps, mu=1.0):
    """Matrix blocks so that, for forces f_j at source points x_j,

        u_x(x_i) = sum_j ( Sxx[i,j] f_x[j] + Sxy[i,j] f_y[j] )
        u_y(x_i) = sum_j ( Syx[i,j] f_x[j] + Syy[i,j] f_y[j] )

    x_eval : (M, 2) points where velocity is wanted
    x_src  : (N, 2) blob centres
    Returns Sxx, Sxy, Syy  (Syx = Sxy because the kernel is symmetric).
    """
    dx = x_eval[:, None, 0] - x_src[None, :, 0]     # (M, N) x-separations
    dy = x_eval[:, None, 1] - x_src[None, :, 1]     # (M, N) y-separations
    r = np.sqrt(dx**2 + dy**2)
    A, B = kernel_AB_2d(r, eps)
    c = 1.0 / (4.0 * np.pi * mu)
    Sxx = c * (A + B * dx * dx)
    Sxy = c * (B * dx * dy)
    Syy = c * (A + B * dy * dy)
    return Sxx, Sxy, Syy


def velocity_2d(x_eval, x_src, forces, eps, mu=1.0):
    """Velocity (M, 2) at x_eval from regularized point forces (N, 2) at x_src (free space)."""
    Sxx, Sxy, Syy = stokeslet_blocks_2d(x_eval, x_src, eps, mu)
    ux = Sxx @ forces[:, 0] + Sxy @ forces[:, 1]
    uy = Sxy @ forces[:, 0] + Syy @ forces[:, 1]
    return np.column_stack([ux, uy])


def pressure_2d(x_eval, x_src, forces, eps):
    """Pressure (M,) from Eq. (4.10): p = sum_j (f_j . x) (1 - e^{-X}) / (2 pi r^2)."""
    dx = x_eval[:, None, 0] - x_src[None, :, 0]
    dy = x_eval[:, None, 1] - x_src[None, :, 1]
    r2 = dx**2 + dy**2
    X = r2 / eps**2
    # (1 - e^{-X}) / r^2 = g(X) / eps^2, which is finite at r = 0
    ratio = _g(X) / eps**2
    fdotx = forces[None, :, 0] * dx + forces[None, :, 1] * dy
    return np.sum(fdotx * ratio, axis=1) / (2.0 * np.pi)


def blob_2d(r, eps):
    """The Gaussian blob phi(r) of Eq. (4.2); integrates to 1 over the plane."""
    return np.exp(-(np.asarray(r) / eps) ** 2) / (np.pi * eps**2)
