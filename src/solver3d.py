"""
solver3d.py
===========
Our 3D regularized Stokeslet solver for flow through a periodic network of straight fibres
(main.tex Section 7). Two ways to represent a fibre:

  "surface"    : blobs on rings around the fibre, width eps = spacing, pulled in by eps/(2 sqrt(pi))
                 (the Stage 3 rule, Proposition 6; the planar-sheet argument is the same in 3D);
                 Richardson extrapolation with eps and 2 eps.
                 PLUS a known body force inside each fibre (Proposition 8, "pressure balance").
                 In the regularized model the inside of a fibre is fluid. The mean pressure
                 gradient G pushes that fluid along the fibre axis (pipe flow inside the fibre),
                 which periodic surface forces cannot stop (Proposition 8 i). A uniform body force
                 b = G spread over the disc inside each ring removes that driving for any flow
                 direction (Proposition 8 ii). The body force is not an unknown:
                 b = (sum of blob forces) / (V - V_disc)  (Eq. 7.8), a rank-3 term in the matrix.
                 A smaller inside flow, of order eps, remains (the blobs' own forces spread inwards);
                 it is removed from the Darcy flux by the superficial-velocity correction.
  "centreline" : blobs along the fibre axis with width eps_c = a exp(gamma/2) = 1.3346 a.
                 Integrated along a straight line, a 3D Gaussian blob is exactly the 2D Gaussian
                 blob (Eq. 7.6), so in cross-section this is the one-blob-per-fibre model of
                 Proposition 3. Cheap (no ring of points), but it cannot respond to straining flow
                 from close neighbours; its error is measured against the surface method.

Permeability (Eq. 7.10). Darcy's law in 3D is a matrix relation:
        U_sup = -(1/mu) K G ,
with G the mean pressure gradient (= total force / V, Proposition 4) and U_sup the superficial
velocity: the flux through the FLUID only, i.e. the mean velocity minus the flux inside the
fibres. U_sup is a vector; for a tilted fibre the flux inside it runs along the fibre, not along U.
We impose U along x, y and z in turn (one LU factorization, three cheap back-substitutions),
collect the three U_sup and G vectors as the columns of two 3 x 3 matrices, and get
        K = -mu [U_sup] [G]^(-1).
The permeability "along direction c" that we report is 1 / (K^(-1))_cc: the mean flow is along c and
we measure the pressure drop along c, as in a filter test rig.
"""

import time

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from periodic3d import PeriodicStokeslet3D
from geometry3d import surface_points, centreline_points, disk_points, frame

EULER_GAMMA = 0.5772156649015329
OFFSET_FACTOR = 1.0 / (2.0 * np.sqrt(np.pi))
CENTRELINE_EPS_FACTOR = np.exp(EULER_GAMMA / 2.0)      # eps_c / a
DIRECTIONS = np.eye(3)                                  # unit flow along x, y, z


class NetworkSolver3D:
    def __init__(self, network, method="surface", eps=0.2, spacing=None, mu=1.0, balance=True, disk_factor=2.0):
        self.network, self.method, self.mu = network, method, mu
        L = network["L"]
        a_mean = np.mean([f["radius"] for f in network["fibres"]])
        if method == "surface":
            self.eps = eps
            self.spacing = spacing or eps
            self.points, self.ids = surface_points(network, self.spacing, OFFSET_FACTOR * eps)
            # Pressure balance inside the fibres (Section 7.4, Proposition 8): a known, uniform body
            # force b per unit volume over the disc inside each ring of blobs
            # (radius a - eps/(2 sqrt(pi))), with b = G, the mean pressure gradient.
            # The disc points are disk_factor x spacing apart: they carry a known, smooth force,
            # so a coarser grid is enough (the change in k is measured in main.tex Section 7.5).
            self.balance = balance
            if balance:
                self.disk_pts, self.disk_w, _ = disk_points(network, disk_factor * self.spacing, OFFSET_FACTOR * eps)
                self.disc_volume = np.sum(self.disk_w)
        elif method == "centreline":
            self.balance = False
            radii = {f["radius"] for f in network["fibres"]}
            if len(radii) > 1:
                raise ValueError("the centreline model uses one blob width: identical fibres only")
            self.eps = CENTRELINE_EPS_FACTOR * a_mean
            self.spacing = spacing or self.eps
            self.points, self.ids = centreline_points(network, self.spacing)
        else:
            raise ValueError("method must be 'surface' or 'centreline'")
        self.N = len(self.points)
        self.kernel = PeriodicStokeslet3D([L, L, L], self.eps, mu=mu)
        self.a_mean = a_mean
        self.timing = {}
        self.lu = None
        self.K = None

    def factorize(self):
        """Build the matrix and its LU factorization once; every flow direction reuses it."""
        t0 = time.perf_counter()
        M = self.kernel.matrix(self.points)
        N = self.N
        V = self.network["L"] ** 3
        if self.balance:
            # velocity at every blob centre from a unit body force (x, y, z) spread over the
            # fibre discs:  C[:, c] = sum_q w_q P(x_i - y_q) e_c
            B = self.kernel.blocks(self.points, self.disk_pts)        # six (N, Nq) blocks
            w = self.disk_w
            bxx, bxy, bxz, byy, byz, bzz = [blk @ w for blk in B]
            C = np.zeros((3 * N, 3))
            C[:N, 0], C[N:2 * N, 0], C[2 * N:, 0] = bxx, bxy, bxz
            C[:N, 1], C[N:2 * N, 1], C[2 * N:, 1] = bxy, byy, byz
            C[:N, 2], C[N:2 * N, 2], C[2 * N:, 2] = bxz, byz, bzz
            # b = (sum of blob forces) / (V - V_disc)   (Eq. 7.8). Component c of the sum of blob
            # forces is the sum of the unknowns in block c, so Eq. (7.9) adds C[:, c] / (V - V_disc)
            # to every column of block c (done in place, without building a second 3N x 3N matrix)
            for c in range(3):
                M[:, c * N:(c + 1) * N] += C[:, c:c + 1] / (V - self.disc_volume)
        self.timing["assemble_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        self.lu = lu_factor(M, overwrite_a=True, check_finite=False)
        self.timing["factor_s"] = time.perf_counter() - t0
        del M

    def solve(self, U=(0.0, 0.0, 1.0)):
        """Blob forces for mean velocity U (the air is still at every blob centre)."""
        if self.lu is None:
            self.factorize()
        N = self.N
        V = self.network["L"] ** 3
        rhs = -np.concatenate([np.full(N, U[0]), np.full(N, U[1]), np.full(N, U[2])])
        sol = lu_solve(self.lu, rhs, check_finite=False)
        self.U = np.array(U, dtype=float)
        self.forces = sol.reshape(3, N).T
        if self.balance:
            self.b = self.forces.sum(axis=0) / (V - self.disc_volume)
        return self.forces

    def total_force(self):
        """Total force on the fluid per cell = G V, G = mean pressure gradient (Proposition 4)."""
        F = self.forces.sum(axis=0)
        if self.balance:
            F = F + self.b * self.disc_volume
        return F

    def inside_flux(self, n_sections=4, n_r=5, n_theta=12):
        """(1/V) x volume integral of the velocity VECTOR over the TRUE fibre volumes (radius a).
        In the regularized model the inside of a fibre is fluid, and some flow remains there
        (Section 7.4); it is not part of the Darcy flux. Quadrature: Gauss-Legendre in r^2 (so
        dA = pi d(r^2)), trapezoid in angle, n_sections cross-sections along each fibre (midpoint
        rule). For the centreline model there is no wall, so nothing is subtracted (as in 2D)."""
        if self.method != "surface":
            return np.zeros(3)
        L = self.network["L"]
        xg, wg = np.polynomial.legendre.leggauss(n_r)
        total = np.zeros(3)
        for f in self.network["fibres"]:
            a = f["radius"]
            t, e1, e2 = frame(f["dirn"])
            length = np.linalg.norm(f["dirn"]) * L
            r = a * np.sqrt((xg + 1) / 2)                     # nodes in r^2 on (0, a^2)
            wr = wg / 2 * np.pi * a**2 / n_theta              # area weight per node
            th = 2 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
            disk = (r[:, None, None] * (np.cos(th)[None, :, None] * e1 + np.sin(th)[None, :, None] * e2)).reshape(-1, 3)
            wts = np.repeat(wr, n_theta)
            for q in range(n_sections):
                centre = np.asarray(f["point"]) + ((q + 0.5) * length / n_sections) * t
                u = self.velocity(centre + disk)
                total += wts @ u * length / n_sections
        return total / L**3

    def superficial_velocity(self):
        """U_sup = U - (flux inside the fibres), a vector (Eq. 7.10)."""
        return self.U - self.inside_flux()

    def permeability_tensor(self):
        """K (3 x 3) from three flow directions: K = -mu [U_sup] [G]^(-1)  (Eq. 7.10)."""
        if self.K is None:
            V = self.network["L"] ** 3
            Usup, G = [], []
            for U in DIRECTIONS:
                self.solve(U)
                Usup.append(self.superficial_velocity())
                G.append(self.total_force() / V)
            Usup, G = np.column_stack(Usup), np.column_stack(G)
            self.K = -self.mu * Usup @ np.linalg.inv(G)
        return self.K

    def permeability(self, direction=(0.0, 0.0, 1.0)):
        """Permeability along a direction n: 1 / (n . K^(-1) n). Mean flow along n, pressure drop
        measured along n."""
        n = np.asarray(direction, dtype=float)
        n = n / np.linalg.norm(n)
        return 1.0 / (n @ np.linalg.inv(self.permeability_tensor()) @ n)

    def dimensionless_permeability(self, direction=(0.0, 0.0, 1.0)):
        return self.permeability(direction) / self.a_mean**2

    def max_slip(self):
        """Largest |u| at the blob centres divided by |U|, for the last solve (round-off level)."""
        u = self.velocity(self.points)
        return np.max(np.linalg.norm(u, axis=1)) / np.linalg.norm(self.U)

    def velocity(self, x):
        """Velocity at points x (M, 3) for the last solve: mean flow + blobs + pressure balance."""
        x = np.atleast_2d(x)
        u = self.kernel.velocity(x, self.points, self.forces, U=self.U)
        if self.balance:
            u += self.kernel.velocity(x, self.disk_pts, np.outer(self.disk_w, self.b))
        return u


def k_network(network, method="surface", eps=0.2, direction=(0.0, 0.0, 1.0), richardson=True):
    """k / a_mean^2 along `direction`. For the surface method with richardson=True, returns
    (4 k(eps) - k(2 eps)) / 3 together with k(eps) and k(2 eps)."""
    s1 = NetworkSolver3D(network, method, eps)
    k1 = s1.dimensionless_permeability(direction)
    if method != "surface" or not richardson:
        return k1, k1, None, s1
    s2 = NetworkSolver3D(network, method, 2 * eps)
    k2 = s2.dimensionless_permeability(direction)
    return (4 * k1 - k2) / 3, k1, k2, s1
