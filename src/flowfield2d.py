"""
flowfield2d.py
==============
A fast, divergence-free copy of our 2D regularized Stokeslet (RS) flow, for particle tracking
(main.tex Section 8.1), and the near-wall reconstruction (Section 8.2, Proposition 9).

Why: tracking thousands of particles needs millions of velocity evaluations. Evaluating the
periodic RS sum (periodic.py) at every particle and every step would take days.

How (exact, not an approximation of the RS model):
  1. The RS flow is the periodic Stokes flow driven by the force density
         F(x) = sum_j f_j phi_eps(x - x_j),    phi_eps = exp(-r^2/eps^2) / (pi eps^2).
     It is smooth (Gaussians), so it is captured exactly by a Fourier series on a grid fine
     enough for the blobs (grid step h = eps/grid_factor; with grid_factor = 4, as used here, the
     aliasing error ~ exp(-pi^2 eps^2/h^2) = exp(-16 pi^2) < 1e-60).
  2. We sample F on the grid ("spreading": each Gaussian touches only nearby nodes) and solve
     Stokes' equations in Fourier space. In 2D the velocity has a stream function psi,
         u = d psi/dy ,   v = -d psi/dx ,
     and Stokes' equations give (Eq. 8.2), for every wavevector k != 0,
         psi_hat(k) = -i (k_y F_x_hat - k_x F_y_hat) / (mu |k|^4).
     The k = 0 mode is left out (Proposition 4), and the mean flow U = (U, 0) adds psi = U y.
  3. A periodic bicubic B-spline of the periodic part of psi gives the velocity anywhere as
     derivatives of ONE smooth function, so the interpolated flow is exactly divergence-free:
     particles cannot drift across streamlines because of interpolation error.

Only mean flow along x is supported (U = (U, 0)); that is how every layout is used here.
"""

import numpy as np
from scipy.ndimage import spline_filter


class StreamFunctionField2D:
    """Stream function and velocity of the RS solution of one layout.

    solver : a solved FibreSolver2D (its blob points, forces, eps, U and layout are used)
    grid_factor : grid step = eps / grid_factor
    """

    def __init__(self, solver, grid_factor=6):
        lay = solver.layout
        self.layout = lay
        self.Lx, self.Ly = lay["Lx"], lay["Ly"]
        self.U = float(solver.U[0])
        if abs(solver.U[1]) > 0:
            raise ValueError("StreamFunctionField2D assumes mean flow along x")
        self.mu = solver.mu
        eps = solver.eps
        # grid with a whole number of steps in each direction
        self.nx = int(np.ceil(self.Lx / (eps / grid_factor)))
        self.ny = int(np.ceil(self.Ly / (eps / grid_factor)))
        self.hx, self.hy = self.Lx / self.nx, self.Ly / self.ny
        x = np.arange(self.nx) * self.hx
        y = np.arange(self.ny) * self.hy

        # ---- 1. spread the blob forces onto the grid: F(x) = sum_j f_j phi_eps(x - x_j) ----
        Fx = np.zeros((self.nx, self.ny))
        Fy = np.zeros((self.nx, self.ny))
        reach = 6.0 * eps                                  # exp(-36): negligible beyond
        mx, my = int(np.ceil(reach / self.hx)), int(np.ceil(reach / self.hy))
        offs_x, offs_y = np.arange(-mx, mx + 1), np.arange(-my, my + 1)
        for (px, py), (fx, fy) in zip(solver.points, solver.forces):
            ix0, iy0 = int(np.round(px / self.hx)), int(np.round(py / self.hy))
            ix, iy = ix0 + offs_x, iy0 + offs_y
            dx = ix * self.hx - px
            dy = iy * self.hy - py
            g = np.exp(-(dx[:, None] ** 2 + dy[None, :] ** 2) / eps**2) / (np.pi * eps**2)
            sx, sy = ix % self.nx, iy % self.ny                 # periodic wrap
            Fx[np.ix_(sx, sy)] += fx * g
            Fy[np.ix_(sx, sy)] += fy * g

        # ---- 2. Stokes in Fourier space: the periodic stream function ----
        kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.hx)
        ky = 2 * np.pi * np.fft.fftfreq(self.ny, d=self.hy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k2 = KX**2 + KY**2
        k2[0, 0] = 1.0                                          # avoid 0/0; the mode is zeroed below
        psi_hat = -1j * (KY * np.fft.fft2(Fx) - KX * np.fft.fft2(Fy)) / (self.mu * k2**2)
        psi_hat[0, 0] = 0.0
        self.psi_periodic = np.real(np.fft.ifft2(psi_hat))      # psi = U y + psi_periodic
        # the same field's velocity on the grid, spectrally (used only for checks)
        self.u_grid = self.U + np.real(np.fft.ifft2(1j * KY * psi_hat))
        self.v_grid = np.real(np.fft.ifft2(-1j * KX * psi_hat))

        # ---- 3. periodic bicubic B-spline of the periodic part of psi ----
        # coefficients c_ij such that psi'(x, y) = sum c_ij B(x/hx - i) B(y/hy - j), with B the cubic
        # B-spline; the spline interpolates psi' at every grid node and is periodic
        self.coef = spline_filter(self.psi_periodic, order=3, mode="grid-wrap")
        self.x_grid, self.y_grid = x, y

    @staticmethod
    def _weights(s):
        """Cubic B-spline weights (and their derivatives) of the 4 nodes i0-1 .. i0+2 around the
        point s (in grid units), i0 = floor(s), t = s - i0."""
        i0 = np.floor(s).astype(int)
        t = s - i0
        w = np.stack([(1 - t) ** 3 / 6, (3 * t**3 - 6 * t**2 + 4) / 6,
                      (-3 * t**3 + 3 * t**2 + 3 * t + 1) / 6, t**3 / 6], axis=1)
        dw = np.stack([-(1 - t) ** 2 / 2, 1.5 * t**2 - 2 * t, -1.5 * t**2 + t + 0.5, t**2 / 2], axis=1)
        return i0, w, dw

    def _eval(self, X, which):
        """psi' and/or its derivatives at points X, from the spline coefficients (periodic)."""
        X = np.atleast_2d(X)
        ix, wx, dwx = self._weights(X[:, 0] / self.hx)
        iy, wy, dwy = self._weights(X[:, 1] / self.hy)
        I = (ix[:, None] + np.arange(-1, 3)[None, :]) % self.nx            # (m, 4)
        J = (iy[:, None] + np.arange(-1, 3)[None, :]) % self.ny
        C = self.coef[I[:, :, None], J[:, None, :]]                          # (m, 4, 4)
        out = []
        for name in which:
            if name == "psi":
                out.append(np.einsum("mi,mij,mj->m", wx, C, wy))
            elif name == "dx":
                out.append(np.einsum("mi,mij,mj->m", dwx, C, wy) / self.hx)
            elif name == "dy":
                out.append(np.einsum("mi,mij,mj->m", wx, C, dwy) / self.hy)
        return out

    def psi(self, X):
        """Stream function at points X (N, 2), with psi = U y + periodic part (y NOT wrapped in
        the U y term, so psi increases by U Ly from one cell to the next)."""
        X = np.atleast_2d(X)
        return self.U * X[:, 1] + self._eval(X, ["psi"])[0]

    def velocity(self, X):
        """Velocity (N, 2) at points X: u = U + d psi'/dy, v = -d psi'/dx (derivatives of one
        spline, so the interpolated flow is exactly divergence-free)."""
        px, py = self._eval(X, ["dx", "dy"])
        return np.column_stack([self.U + py, -px])


def wall_shear(field, centre, a, n_theta=720, deltas=(0.01, 0.02, 0.03)):
    """Wall shear rate gamma(theta) on the circle of radius a around `centre`, from the tangential
    velocity at small distances delta outside it, fitted as u_t = gamma delta + beta delta^2
    (least squares). theta is measured from the +x axis; u_t is the anticlockwise component.
    Returns theta, gamma."""
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    er = np.column_stack([np.cos(th), np.sin(th)])
    et = np.column_stack([-np.sin(th), np.cos(th)])
    D = np.array(deltas)
    ut = np.array([np.sum(field.velocity(centre + (a + d) * er) * et, axis=1) for d in D])   # (nd, n_theta)
    A = np.column_stack([D, D**2])
    coef, *_ = np.linalg.lstsq(A, ut, rcond=None)
    return th, coef[0]


def leveque_efficiency(theta, gamma, a, D):
    """High-Peclet (Leveque) diffusion flux to an absorbing circle of radius a with wall shear
    rate gamma(theta), main.tex Eq. (8.7), Section 8.5. The boundary layer starts at each stagnation point
    where the surface flow diverges and runs to the next one; on each such segment with
        I = integral of sqrt(|gamma|) ds   (s = arc length),
    the flux per unit upstream concentration is
        Q = (3/2) D^(2/3) I^(2/3) / (Gamma(4/3) 9^(1/3)).
    Returns the total Q of the fibre (sum over segments), per unit upstream concentration."""
    from scipy.special import gamma as Gamma
    n = len(theta)
    ds = 2 * np.pi * a / n
    sg = np.sign(gamma)
    # cut the circle at the sign changes of gamma: each run of constant sign is one segment
    cuts = np.nonzero(sg != np.roll(sg, 1))[0]
    if len(cuts) == 0:
        raise ValueError("wall shear never changes sign: no stagnation point")
    Q = 0.0
    for c0, c1 in zip(cuts, np.roll(cuts, -1)):
        idx = np.arange(c0, c1 if c1 > c0 else c1 + n) % n
        I = np.sum(np.sqrt(np.abs(gamma[idx]))) * ds
        Q += 1.5 * D ** (2 / 3) * I ** (2 / 3) / (Gamma(4 / 3) * 9 ** (1 / 3))
    return Q


# ----------------------------------------------------------------------------
# Near-wall reconstruction (main.tex Section 8.2, Proposition 9)
# ----------------------------------------------------------------------------
# Within about eps of the blobs the RS flow is not the flow past a no-slip cylinder: it slips
# along the fibre (u_t at r = a is about +0.06 U for eps = 0.1a, first order in eps). Interception
# and diffusion happen exactly there, so near each fibre we replace the RS flow by the EXACT
# Stokes flow in the annulus a <= r <= r_out that
#     (i)  is at rest on the fibre (psi = const, d psi/dr = 0 at r = a), and
#     (ii) has the same psi and d psi/dr as the RS flow on the outer circle r = r_out,
# where the RS flow is accurate (several blob widths away). In polar coordinates around the fibre,
#     psi = psi_0(r) + 2 Re sum_{n>=1} psi_n(r) e^{i n theta},
# and every Stokes stream function is a sum of the separable biharmonic solutions
#     n = 0 : 1, ln r, r^2                  (r^2 ln r is excluded: it needs a multivalued pressure)
#     n = 1 : r, 1/r, r^3, r ln r
#     n >= 2: r^n, r^-n, r^(n+2), r^(2-n).
# Four conditions per mode (three for n = 0, whose wall value psi_0(a) is free) fix the
# coefficients. The annulus problem with velocity given on both circles has a unique solution,
# so if the outer data were exact this would be the exact flow.

def _basis(n, r):
    """Radial functions of mode n and their r-derivatives at radii r: arrays (nb, len(r))."""
    r = np.asarray(r, dtype=float)
    if n == 0:
        f = [np.ones_like(r), np.log(r), r**2]
        df = [np.zeros_like(r), 1 / r, 2 * r]
    elif n == 1:
        f = [r, 1 / r, r**3, r * np.log(r)]
        df = [np.ones_like(r), -1 / r**2, 3 * r**2, np.log(r) + 1]
    else:
        f = [r**n, r**-n, r**(n + 2), r**(2 - n)]
        df = [n * r**(n - 1), -n * r**(-n - 1), (n + 2) * r**(n + 1), (2 - n) * r**(1 - n)]
    return np.array(f), np.array(df)


class NearWallField2D:
    """The flow of a StreamFunctionField2D with the near-wall reconstruction around every fibre.

    r_out_gap : r_out - a (default 0.3 a); must be smaller than the gap to the nearest neighbour
    n_modes   : Fourier modes kept in theta
    Outside every annulus the velocity is that of the RS field; inside the annulus of the nearest
    fibre it is the reconstructed exact Stokes flow (continuous at r_out up to the truncation).
    Neighbouring annuli may overlap (gaps >= 0.5a, annuli 0.3a wide); a point in two of them uses the
    nearest fibre's, and the outer data of an annulus may then be taken only ~0.2a from another fibre,
    where the RS flow is less accurate (Section 8.2)."""

    def __init__(self, field, r_out_gap=0.3, n_modes=64):
        self.field = field
        self.layout = field.layout
        self.Lx, self.Ly, self.U = field.Lx, field.Ly, field.U
        c, a = self.layout["centres"], self.layout["radii"]
        L = np.array([self.Lx, self.Ly])
        # the annulus must not contain any other fibre
        for m in range(len(a)):
            d = c - c[m]
            d -= L * np.round(d / L)
            dist = np.sqrt(np.sum(d**2, axis=1)) - a - a[m]
            dist[m] = np.inf
            # the fibre's own periodic copies (small cells)
            for sh in ([L[0], 0], [0, L[1]]):
                dist = np.append(dist, np.linalg.norm(sh) - 2 * a[m])
            if r_out_gap >= dist.min():
                raise ValueError(f"r_out_gap {r_out_gap} >= gap {dist.min():.3f} next to fibre {m}")
        self.r_out = a + r_out_gap
        self.n_modes = n_modes
        nth = 4 * n_modes
        th = 2 * np.pi * np.arange(nth) / nth
        er = np.column_stack([np.cos(th), np.sin(th)])
        et = np.column_stack([-np.sin(th), np.cos(th)])
        self.coef = []
        for m in range(len(a)):
            P = c[m] + self.r_out[m] * er
            psi = field.psi(P)
            # d psi/dr = -u_theta (u_theta = anticlockwise velocity)
            dpsi = -np.sum(field.velocity(P) * et, axis=1)
            # psi(P) uses the unwrapped y in its U y term, so on this small circle it is a smooth,
            # periodic function of theta
            ph, dph = np.fft.rfft(psi) / nth, np.fft.rfft(dpsi) / nth
            Cm = []
            for n in range(n_modes + 1):
                f_a, df_a = _basis(n, [a[m]])
                f_o, df_o = _basis(n, [self.r_out[m]])
                if n == 0:
                    A = np.array([df_a[:, 0], f_o[:, 0], df_o[:, 0]])
                    rhs = np.array([0.0, ph[0].real, dph[0].real])
                    Cm.append(np.linalg.solve(A, rhs).astype(complex))
                else:
                    A = np.array([f_a[:, 0], df_a[:, 0], f_o[:, 0], df_o[:, 0]])
                    rhs = np.array([0.0, 0.0, ph[n], dph[n]])
                    Cm.append(np.linalg.solve(A, rhs))
            self.coef.append(Cm)
        # the same coefficients as arrays, for fast evaluation of many points at once
        self.C0 = np.array([Cm[0].real for Cm in self.coef])
        self.C1 = np.array([Cm[1] for Cm in self.coef])
        self.Cn = np.array([np.array(Cm[2:]) for Cm in self.coef])
        self.n_list = np.arange(2, n_modes + 1)

    def _local(self, j, rel):
        """psi and velocity from the annulus solutions at points rel (relative to the centre of
        fibre j, one fibre index per point). All modes at once (arrays of shape points x modes)."""
        r = np.sqrt(np.sum(rel**2, axis=1))
        th = np.arctan2(rel[:, 1], rel[:, 0])
        C0, C1, Cn = self.C0[j], self.C1[j], self.Cn[j]          # (m,3), (m,4), (m,N-1,4)
        # mode 0
        psi = C0[:, 0] + C0[:, 1] * np.log(r) + C0[:, 2] * r**2
        dpsi_r = C0[:, 1] / r + 2 * C0[:, 2] * r
        # mode 1
        e1 = np.exp(1j * th)
        p1 = C1[:, 0] * r + C1[:, 1] / r + C1[:, 2] * r**3 + C1[:, 3] * r * np.log(r)
        d1 = C1[:, 0] - C1[:, 1] / r**2 + 3 * C1[:, 2] * r**2 + C1[:, 3] * (np.log(r) + 1)
        psi = psi + 2 * np.real(p1 * e1)
        dpsi_r = dpsi_r + 2 * np.real(d1 * e1)
        dpsi_t = 2 * np.real(1j * p1 * e1)
        # modes n >= 2: r^n, r^-n, r^(n+2), r^(2-n)
        n = self.n_list[None, :]
        R = r[:, None]
        Rn, Rm = R**n, R**(-n)
        f = np.stack([Rn, Rm, Rn * R**2, Rm * R**2], axis=2)                  # (m, N-1, 4)
        df = np.stack([n * Rn / R, -n * Rm / R, (n + 2) * Rn * R, (2 - n) * Rm * R], axis=2)
        pn = np.sum(Cn * f, axis=2)
        dn = np.sum(Cn * df, axis=2)
        en = np.exp(1j * n * th[:, None])
        psi = psi + 2 * np.real(np.sum(pn * en, axis=1))
        dpsi_r = dpsi_r + 2 * np.real(np.sum(dn * en, axis=1))
        dpsi_t = dpsi_t + 2 * np.real(np.sum(1j * n * pn * en, axis=1))
        ur, ut = dpsi_t / r, -dpsi_r
        cs, sn = rel[:, 0] / r, rel[:, 1] / r
        return psi, np.column_stack([ur * cs - ut * sn, ur * sn + ut * cs])

    def _nearest(self, X):
        c = self.layout["centres"]
        L = np.array([self.Lx, self.Ly])
        d = X[:, None, :] - c[None, :, :]
        d -= L * np.round(d / L)
        dist = np.sqrt(np.sum(d**2, axis=2))
        j = np.argmin(dist - self.r_out[None, :], axis=1)
        rel = d[np.arange(len(X)), j]
        inside = dist[np.arange(len(X)), j] < self.r_out[j]
        return j, rel, inside

    def velocity(self, X):
        X = np.atleast_2d(X)
        u = self.field.velocity(X)
        j, rel, inside = self._nearest(X)
        if inside.any():
            u[inside] = self._local(j[inside], rel[inside])[1]
        return u

    def psi(self, X):
        """Stream function; inside an annulus the reconstructed one. (Its value is continuous at
        r_out up to truncation; the U y offset of the periodic copy is restored.)"""
        X = np.atleast_2d(X)
        p = self.field.psi(X)
        j, rel, inside = self._nearest(X)
        c = self.layout["centres"]
        if inside.any():
            jj = j[inside]
            # the annulus solution was fitted with psi of the copy centred at c[j]; points near
            # another periodic copy differ by a multiple of U Ly
            shift = np.round((X[inside, 1] - rel[inside, 1] - c[jj, 1]) / self.Ly) * self.U * self.Ly
            p[inside] = self._local(jj, rel[inside])[0] + shift
        return p
