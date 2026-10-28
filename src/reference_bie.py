"""
reference_bie.py
================
An INDEPENDENT, spectrally accurate reference solver for Stokes flow through
periodic arrays of circular fibres. It uses the ordinary (singular) Stokeslet,
no blobs, so it does not share the regularized method's errors. We use it only
to measure how accurate the regularized Stokeslet solver is (main.tex Sec. 5.2).

Method (single-layer boundary integral, main.tex Eq. 5.1)
---------------------------------------------------------
On each fibre surface there is an unknown force density f(theta) (force per
unit length of surface). The velocity anywhere is

    u(x) = U + integral over all fibre surfaces of  P0(x - y) f(y) ds(y),

with P0 the singular periodic Stokeslet. We require u = 0 at M equally spaced
points on every circle. The difficulty is that P0 has a ln r singularity when
x and y are on the same circle. We split, for points on the same circle:

    4 pi mu P0 = [ -ln r ] I  +  [ -1/2 I + r r^T / r^2 ]  +  4 pi mu C(x - y)

  * -ln r on a circle is handled EXACTLY with the Fourier series
        -ln|x - y| = -ln a + sum_{n>=1} cos(n (theta - phi)) / n       (Eq. 5.2)
    applied to the trigonometric interpolant of f (spectral accuracy);
  * r r^T / r^2 is smooth on a circle (its limit at r -> 0 is the tangent
    direction t t^T), so the trapezoid rule is spectrally accurate;
  * C = P0 - (free-space Stokeslet) is the effect of all periodic copies; it is
    smooth near r = 0 and is taken from our Ewald code with a vanishing blob.
For points on DIFFERENT circles the kernel is smooth and the trapezoid rule is used.

Like every single-layer Stokes formulation, the density f = (normal vector) on a
circle produces zero velocity, so the matrix has one null vector per fibre; it
carries no net force, and we solve by least squares, which picks the solution
without that component. The total force is unaffected.
"""

import numpy as np

from periodic import PeriodicStokeslet2D
from rs_kernels import stokeslet_blocks_2d

TINY_EPS_FACTOR = 1e-9      # "vanishing" blob width used only to evaluate C = P0 - K0


def log_weights(M):
    """R(t) = (2 pi / M) [ sum_{m=1}^{M/2-1} cos(m t)/m + cos(M t / 2) / M ]  (Eq. 5.3)
    as an M x M matrix with t = theta_i - theta_j: integral of -ln|2 sin(t/2)| f dphi."""
    assert M % 2 == 0, "use an even number of points per circle"
    t = 2 * np.pi * (np.arange(M)[:, None] - np.arange(M)[None, :]) / M
    m = np.arange(1, M // 2)
    R = np.cos(t[..., None] * m) @ (1.0 / m)
    R = R + np.cos(M * t / 2) / M
    return 2 * np.pi / M * R


class ReferenceBIE2D:
    """Spectral boundary-integral reference for circular fibres in a periodic cell."""

    def __init__(self, layout, points_per_fibre=128, mu=1.0):
        self.layout, self.mu = layout, mu
        self.M = points_per_fibre
        c, a = layout["centres"], layout["radii"]
        self.n_fib = len(a)
        t = 2 * np.pi * np.arange(self.M) / self.M
        self.theta = t
        self.pts = np.vstack([c[m] + a[m] * np.column_stack([np.cos(t), np.sin(t)]) for m in range(self.n_fib)])
        self.ds = np.concatenate([np.full(self.M, 2 * np.pi * a[m] / self.M) for m in range(self.n_fib)])
        eps0 = TINY_EPS_FACTOR * min(layout["Lx"], layout["Ly"])
        self.periodic = PeriodicStokeslet2D(layout["Lx"], layout["Ly"], eps0, mu=mu)
        self.eps0 = eps0

    def assemble(self):
        N = len(self.pts)
        M, mu = self.M, self.mu
        c4 = 1.0 / (4 * np.pi * mu)
        # Start from the periodic kernel (with a vanishing blob) for ALL pairs, times ds.
        Pxx, Pxy, Pyy = self.periodic.blocks(self.pts, self.pts)
        Mxx, Mxy, Myy = Pxx * self.ds, Pxy * self.ds, Pyy * self.ds
        logW = log_weights(M)
        a_all = self.layout["radii"]
        for m in range(self.n_fib):
            sl = slice(m * M, (m + 1) * M)
            p = self.pts[sl]
            a = a_all[m]
            ds = 2 * np.pi * a / M
            # remove the free-space part of the same-circle block (it is singular) ...
            Kxx, Kxy, Kyy = stokeslet_blocks_2d(p, p, self.eps0, mu)
            Mxx[sl, sl] -= Kxx * ds
            Mxy[sl, sl] -= Kxy * ds
            Myy[sl, sl] -= Kyy * ds
            # ... and add it back with exact singular quadrature:
            # (a) -ln r part (Eqs. 5.2-5.3): -ln a * ds + a * R(theta_i - theta_j)
            L = c4 * (-np.log(a) * ds + a * logW)
            # (b) -1/2 I + r r^T / r^2 by the trapezoid rule, diagonal limit = t t^T
            dx = p[:, None, 0] - p[None, :, 0]
            dy = p[:, None, 1] - p[None, :, 1]
            r2 = dx**2 + dy**2
            np.fill_diagonal(r2, 1.0)
            Rxx, Rxy, Ryy = dx * dx / r2, dx * dy / r2, dy * dy / r2
            tx, ty = -np.sin(self.theta), np.cos(self.theta)          # unit tangent
            idx = np.arange(M)
            Rxx[idx, idx], Rxy[idx, idx], Ryy[idx, idx] = tx * tx, tx * ty, ty * ty
            Mxx[sl, sl] += L + c4 * ds * (Rxx - 0.5)
            Mxy[sl, sl] += c4 * ds * Rxy
            Myy[sl, sl] += L + c4 * ds * (Ryy - 0.5)
        self.matrix = np.block([[Mxx, Mxy], [Mxy, Myy]])
        return self.matrix

    def solve(self, U=(1.0, 0.0)):
        self.assemble()
        N = len(self.pts)
        rhs = -np.concatenate([np.full(N, U[0]), np.full(N, U[1])])
        sol, *_ = np.linalg.lstsq(self.matrix, rhs, rcond=1e-13)
        self.U = np.array(U, dtype=float)
        self.density = np.column_stack([sol[:N], sol[N:]])
        self.residual = np.linalg.norm(self.matrix @ sol - rhs) / np.linalg.norm(rhs)
        return self.density

    def total_force(self):
        return (self.density * self.ds[:, None]).sum(axis=0)

    def velocity(self, x):
        """Velocity at points x OFF the surfaces, by the trapezoid rule. Accurate when x is
        at least a few point-spacings 2 pi a / M away from every fibre surface."""
        x = np.atleast_2d(x)
        Pxx, Pxy, Pyy = self.periodic.blocks(x, self.pts)
        fx, fy = self.density[:, 0] * self.ds, self.density[:, 1] * self.ds
        return np.column_stack([self.U[0] + Pxx @ fx + Pxy @ fy, self.U[1] + Pxy @ fx + Pyy @ fy])

    def permeability(self):
        U = self.U
        Uhat = U / np.linalg.norm(U)
        area = self.layout["Lx"] * self.layout["Ly"]
        return self.mu * np.linalg.norm(U) * area / abs(self.total_force() @ Uhat)

    def dimensionless_permeability(self, a_ref=None):
        a_ref = a_ref or np.mean(self.layout["radii"])
        return self.permeability() / a_ref**2


def reference_k(layout, points_per_fibre=128, U=(1.0, 0.0)):
    """k / a_mean^2 from the reference solver."""
    r = ReferenceBIE2D(layout, points_per_fibre)
    r.solve(U)
    return r.dimensionless_permeability()


class ReferenceAxialBIE2D:
    """Exact reference for flow ALONG parallel fibres (axial flow, main.tex Eq. 7.11).

    The axial velocity w(x, y) obeys the scalar problem  mu lap w = dp/dz  with w = 0 on the
    fibres. Its free-space Green's function is -(1/(2 pi mu)) ln r, and the periodic one is the
    TRACE of our 2D periodic Stokes kernel (tr(I - k k / k^2) = 1 in 2D), so we reuse the same
    Ewald code and the same exact logarithmic quadrature as ReferenceBIE2D:
        w(x) = W + sum_m integral G_P(x - y) q(y) ds(y),      w = 0 at M points per circle,
    and the axial permeability is k_par = mu W A / |sum q ds|."""

    def __init__(self, layout, points_per_fibre=128, mu=1.0):
        self.layout, self.mu = layout, mu
        self.M = points_per_fibre
        c, a = layout["centres"], layout["radii"]
        self.n_fib = len(a)
        t = 2 * np.pi * np.arange(self.M) / self.M
        self.pts = np.vstack([c[m] + a[m] * np.column_stack([np.cos(t), np.sin(t)]) for m in range(self.n_fib)])
        self.ds = np.concatenate([np.full(self.M, 2 * np.pi * a[m] / self.M) for m in range(self.n_fib)])
        eps0 = TINY_EPS_FACTOR * min(layout["Lx"], layout["Ly"])
        self.periodic = PeriodicStokeslet2D(layout["Lx"], layout["Ly"], eps0, mu=mu)
        self.eps0 = eps0

    def solve(self, W=1.0):
        M = self.M
        Pxx, _, Pyy = self.periodic.blocks(self.pts, self.pts)
        A = (Pxx + Pyy) * self.ds                      # trace = periodic Laplace Green's function / mu
        logW = log_weights(M)
        for m in range(self.n_fib):
            sl = slice(m * M, (m + 1) * M)
            p = self.pts[sl]
            a = self.layout["radii"][m]
            ds = 2 * np.pi * a / M
            Kxx, _, Kyy = stokeslet_blocks_2d(p, p, self.eps0, self.mu)
            A[sl, sl] -= (Kxx + Kyy) * ds                 # remove the singular free-space part ...
            A[sl, sl] += (-np.log(a) * ds + a * logW) / (2 * np.pi * self.mu)   # ... and add it exactly
        sol, *_ = np.linalg.lstsq(A, -np.full(len(self.pts), W), rcond=1e-13)
        self.W, self.q = W, sol
        return sol

    def permeability(self):
        """Darcy uses the SUPERFICIAL velocity: the flux through the fluid only, divided by the
        cell area. Inside each fibre this single-layer solution is not at rest: there
        mu lap w = Q/A (Q = total force per length), a uniform pipe-flow driving, so with w = 0 on
        the circle w_in = (Q/(4 mu A)) (r^2 - a^2), whose flux is -pi a^4 Q / (8 mu A).
        We subtract it (main.tex Eq. 7.10)."""
        area = self.layout["Lx"] * self.layout["Ly"]
        Q = np.sum(self.q * self.ds)
        interior_flux = sum(-np.pi * a**4 * Q / (8 * self.mu * area) for a in self.layout["radii"])
        W_sup = self.W - interior_flux / area
        return self.mu * abs(W_sup) * area / abs(Q)

    def dimensionless_permeability(self):
        return self.permeability() / np.mean(self.layout["radii"]) ** 2


def reference_k_axial(layout, points_per_fibre=128):
    r = ReferenceAxialBIE2D(layout, points_per_fibre)
    r.solve()
    return r.dimensionless_permeability()
