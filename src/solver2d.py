"""
solver2d.py
===========
Our 2D regularized Stokeslet solver for flow through a periodic array of fibres.

The problem (main.tex, Eq. 4.20): air with mean velocity U flows through one
periodic cell containing fibres. Each fibre carries regularized Stokeslets.
We choose their forces so the air is AT REST at every blob centre:

        U + sum_j P(x_i - x_j) f_j = 0      for every blob i.

That is a dense linear system "M f = -U" of size 2N x 2N. Once the forces are
known, the permeability comes from the total force (Proposition 4, Eq. 4.17):

        k = mu U A / |sum_j f_x,j|

The blob width eps and where the blobs sit (the "offset" inside the fibre)
are set by the rule derived in main.tex Section 5.3 (Proposition 6, Eq. 5.6):

        offset = eps / (2 sqrt(pi)),   eps = spacing.

The helper solve_with_rule() applies it, and richardson_k() combines two blob
widths to cancel the leading O(eps^2) error.
"""

import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from periodic import PeriodicStokeslet2D
from geometry import ring_points


class FibreSolver2D:
    """Set up and solve the regularized Stokeslet problem for one layout.

    layout  : dict from geometry.py (centres, radii, Lx, Ly)
    spacing : arc length h between neighbouring blobs on a fibre
    eps     : blob width
    offset  : blobs sit on circles of radius a - offset (0 = on the fibre surface)
    """

    def __init__(self, layout, spacing, eps, offset=0.0, mu=1.0):
        self.layout, self.spacing, self.eps, self.offset, self.mu = layout, spacing, eps, offset, mu
        self.points, self.fibre_id, self.n_per, self.weights = ring_points(layout, spacing, offset)
        self.N = len(self.points)
        self.kernel = PeriodicStokeslet2D(layout["Lx"], layout["Ly"], eps, mu=mu)
        self.matrix = None
        self.forces = None
        self.timing = {}

    # ------------------------------------------------------------------
    # 1. Assemble the 2N x 2N matrix [[Sxx, Sxy], [Sxy, Syy]]
    # ------------------------------------------------------------------
    def assemble(self):
        t0 = time.perf_counter()
        Sxx, Sxy, Syy = self.kernel.blocks(self.points, self.points)
        self.matrix = np.block([[Sxx, Sxy], [Sxy, Syy]])
        self.timing["assemble_s"] = time.perf_counter() - t0
        return self.matrix

    # ------------------------------------------------------------------
    # 2. Solve M f = -U  (dense LU, or GMRES as an iterative alternative)
    # ------------------------------------------------------------------
    def solve(self, U=(1.0, 0.0), method="dense", tol=1e-12):
        if self.matrix is None:
            self.assemble()
        rhs = -np.concatenate([np.full(self.N, U[0]), np.full(self.N, U[1])])
        t0 = time.perf_counter()
        if method == "dense":
            sol = np.linalg.solve(self.matrix, rhs)
            self.gmres_iterations = None
        elif method == "gmres":
            # GMRES only needs matrix-vector products; with a fast summation method
            # those could be done without storing the matrix (future work for 3D).
            count = [0]

            def matvec(v):
                count[0] += 1
                return self.matrix @ v
            op = LinearOperator(self.matrix.shape, matvec=matvec)
            sol, info = gmres(op, rhs, rtol=tol, atol=0.0, restart=200, maxiter=50)
            if info != 0:
                raise RuntimeError(f"GMRES did not converge (info = {info})")
            self.gmres_iterations = count[0]
        else:
            raise ValueError("method must be 'dense' or 'gmres'")
        self.timing["solve_s"] = time.perf_counter() - t0
        self.U = np.array(U, dtype=float)
        self.forces = np.column_stack([sol[: self.N], sol[self.N:]])
        self.residual = np.linalg.norm(self.matrix @ sol - rhs) / np.linalg.norm(rhs)
        return self.forces

    # ------------------------------------------------------------------
    # 3. Outputs
    # ------------------------------------------------------------------
    def total_force(self):
        """Sum of all blob forces = force the fibres exert on the air (per unit length)."""
        return self.forces.sum(axis=0)

    def permeability(self):
        """Eq. (4.17): k = mu |U| A / |F . U_hat| (Darcy permeability along U)."""
        U = self.U
        Uhat = U / np.linalg.norm(U)
        area = self.layout["Lx"] * self.layout["Ly"]
        return self.mu * np.linalg.norm(U) * area / abs(self.total_force() @ Uhat)

    def dimensionless_permeability(self, a_ref=None):
        """k / a^2 using the mean fibre radius (or a_ref)."""
        a_ref = a_ref or np.mean(self.layout["radii"])
        return self.permeability() / a_ref**2

    def velocity(self, x):
        """Air velocity at points x (K, 2)."""
        return self.kernel.velocity(np.atleast_2d(x), self.points, self.forces, U=self.U)

    def leakage(self, n_quad=800):
        """Leakage measures of Eq. (4.13), as fractions of |U| * 2R, one value per fibre:
        (net flux, positive-part flux, chord flux). They are measured on the circle where
        the blobs sit, radius R = a - offset, because that is where the method makes the
        fibre impermeable. (Just outside R, inside the thin blob layer, air still moves
        along the fibre; that is the slip that the offset rule accounts for, not leakage.)
        The chord is the diameter perpendicular to U."""
        U = self.U
        Uhat = U / np.linalg.norm(U)
        perp = np.array([-Uhat[1], Uhat[0]])
        tq = 2 * np.pi * (np.arange(n_quad) + 0.5) / n_quad
        nrm = np.column_stack([np.cos(tq), np.sin(tq)])
        # Gauss-Legendre nodes for the chord: spectrally accurate for the smooth velocity,
        # whereas the trapezoid rule converges only like 1/n^2 on this open interval
        yq, wq = np.polynomial.legendre.leggauss(n_quad)
        out = []
        for c, a in zip(self.layout["centres"], self.layout["radii"]):
            R = a - self.offset
            un = np.sum(self.velocity(c + R * nrm) * nrm, axis=1)
            ds = 2 * np.pi * R / n_quad
            uc = self.velocity(c + R * yq[:, None] * perp) @ Uhat
            norm = np.linalg.norm(U) * 2 * R
            out.append((np.sum(un) * ds / norm, np.sum(np.maximum(un, 0)) * ds / norm,
                        R * np.sum(wq * uc) / norm))
        out = np.array(out)
        return out[:, 0], out[:, 1], out[:, 2]

    def blob_circle_speed(self, n_quad=400):
        """Largest air speed on the blob circles (radius a - offset), as a fraction of |U|.
        Zero for perfect no slip; measures how well the blobs hold the air still BETWEEN
        their centres."""
        tq = 2 * np.pi * (np.arange(n_quad) + 0.5) / n_quad
        nrm = np.column_stack([np.cos(tq), np.sin(tq)])
        worst = 0.0
        for c, a in zip(self.layout["centres"], self.layout["radii"]):
            sp = np.linalg.norm(self.velocity(c + (a - self.offset) * nrm), axis=1)
            worst = max(worst, sp.max())
        return worst / np.linalg.norm(self.U)

    def condition_number(self):
        return np.linalg.cond(self.matrix)


OFFSET_FACTOR = 1.0 / (2.0 * np.sqrt(np.pi))     # Proposition 6: a_eff = R + eps / (2 sqrt(pi))


def solve_layout(layout, spacing, eps, offset=0.0, U=(1.0, 0.0), method="dense"):
    """Convenience: build, solve, and return the solver."""
    s = FibreSolver2D(layout, spacing, eps, offset)
    s.solve(U=U, method=method)
    return s


def solve_with_rule(layout, eps, U=(1.0, 0.0), method="dense"):
    """Solve with the Section 5 rule: spacing h = eps, blobs pulled in by eps/(2 sqrt(pi))."""
    return solve_layout(layout, spacing=eps, eps=eps, offset=OFFSET_FACTOR * eps, U=U, method=method)


def richardson_k(layout, eps, U=(1.0, 0.0)):
    """k / a_mean^2 with the leading O(eps^2) error removed (Eq. 5.7):
        k_R = (4 k(eps) - k(2 eps)) / 3.
    Returns (k_R, k(eps), k(2 eps))."""
    k1 = solve_with_rule(layout, eps, U).dimensionless_permeability()
    k2 = solve_with_rule(layout, 2 * eps, U).dimensionless_permeability()
    return (4 * k1 - k2) / 3, k1, k2
