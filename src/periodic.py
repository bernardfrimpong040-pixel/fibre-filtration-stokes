"""
periodic.py
===========
Our own doubly periodic 2D regularized Stokeslet, computed by Ewald summation.

Why periodic (plain language)
-----------------------------
A filter is wide: the pattern of fibres repeats sideways and (statistically) in
depth. We model one rectangular cell of size Lx by Ly and assume it repeats
forever in both directions. Every blob therefore has infinitely many copies
("images"). Adding up the flow of all the copies directly does not work in 2D
(the sum does not converge), so we use Ewald's trick (main.tex, Section 4.3, Proposition 5):

    periodic kernel = (short-range part summed over a few nearby copies)
                    + (smooth part summed as a Fourier series)

The split uses Hasimoto's factor H(k) = (1 + xi^2 k^2 / 4) exp(-xi^2 k^2 / 4).
With it, the short-range part dies off like exp(-r^2/s^2) with s^2 = eps^2 + xi^2,
so only the 5 x 5 block of nearest copies is needed. The answer does not depend
on the splitting width xi; the check script tests exactly that.

The zero Fourier mode (k = 0) is left out. Physically this means: the net push
of all the fibres on the air in one cell is balanced by the mean pressure
gradient (main.tex, Proposition 4, Eq. 4.16), and the velocity we compute is measured
relative to the mean velocity U, which is added separately.
"""

import numpy as np

from rs_kernels import kernel_AB_2d, _g


def _dAB_ds2(r, s):
    """Derivatives dA/d(s^2) and dB/d(s^2) of the Gaussian kernel with width s (Eq. 4.19):
        dA/d(s^2) = (g(X) - 2 e^{-X}) / (2 s^2)
        dB/d(s^2) = (e^{-X} - g(X)) / (s^2 r^2)     with X = r^2 / s^2.
    """
    r = np.asarray(r, dtype=float)
    X = (r / s) ** 2
    g = _g(X)
    eX = np.exp(-X)
    dA = (g - 2.0 * eX) / (2.0 * s**2)
    dB = np.empty_like(r)
    small = X < 1e-3
    # (e^{-X} - g)/X = -1/2 + X/3 - X^2/8 + X^3/30 - X^4/144 + ...  (series avoids 0/0 at r = 0)
    Xs = X[small]
    dB[small] = (-0.5 + Xs / 3 - Xs**2 / 8 + Xs**3 / 30 - Xs**4 / 144) / s**4
    big = ~small
    dB[big] = (eX[big] - g[big]) / (s**2 * r[big] ** 2)
    return dA, dB


def real_space_AB(r, eps, xi):
    """Short-range scalars A_R, B_R (Eq. 4.19):  K_eps - K_s + xi^2 dK_s/d(s^2).
    The slowly decaying parts cancel exactly; what is left decays like exp(-r^2/s^2)."""
    s = np.sqrt(eps**2 + xi**2)
    A_eps, B_eps = kernel_AB_2d(r, eps)
    A_s, B_s = kernel_AB_2d(r, s)
    dA, dB = _dAB_ds2(r, s)
    return A_eps - A_s + xi**2 * dA, B_eps - B_s + xi**2 * dB


class PeriodicStokeslet2D:
    """Doubly periodic regularized Stokeslet in a Lx-by-Ly cell.

    Usage:
        P = PeriodicStokeslet2D(Lx, Ly, eps)
        Sxx, Sxy, Syy = P.blocks(x_eval, x_src)   # velocity per unit force, zero cell mean
    """

    def __init__(self, Lx, Ly, eps, mu=1.0, xi=None, n_images=2, tol=36.0):
        self.Lx, self.Ly, self.eps, self.mu = float(Lx), float(Ly), float(eps), float(mu)
        self.area = self.Lx * self.Ly
        # splitting width: small enough that the short-range part is negligible
        # beyond n_images copies (exp(-tol) ~ 2e-16 at distance n_images * L)
        self.xi = xi if xi is not None else min(Lx, Ly) / 6.0
        self.s = np.sqrt(self.eps**2 + self.xi**2)
        self.n_images = n_images
        # Guard: the copies we leave out sit at least n_images * min(Lx, Ly) away, where the
        # short-range part is about exp(-(distance/s)^2). If eps is not well below xi this is
        # not negligible, so we refuse rather than return an inaccurate kernel.
        if (self.n_images * min(Lx, Ly)) ** 2 / self.s**2 < tol:
            raise ValueError("blob width eps too large for this cell: increase n_images or reduce eps")

        # Fourier wavevectors k = 2 pi (m/Lx, n/Ly), keeping only those where the
        # smooth factor exp(-s^2 k^2/4) (1 + xi^2 k^2/4) is not negligible
        k_max = 2.0 * np.sqrt(tol + 5.0) / self.s
        mx = int(np.ceil(k_max * self.Lx / (2 * np.pi)))
        my = int(np.ceil(k_max * self.Ly / (2 * np.pi)))
        m, n = np.meshgrid(np.arange(-mx, mx + 1), np.arange(-my, my + 1), indexing="ij")
        kx = 2 * np.pi * m.ravel() / self.Lx
        ky = 2 * np.pi * n.ravel() / self.Ly
        k2 = kx**2 + ky**2
        keep = (k2 > 0) & (k2 <= k_max**2)            # k = 0 mode left out on purpose
        self.kx, self.ky, k2 = kx[keep], ky[keep], k2[keep]
        # Fourier weight (Proposition 5, smooth part of Eq. 4.18): (1 + xi^2 k^2/4) exp(-s^2 k^2/4) / (mu k^2 area)
        self.weight = (1.0 + self.xi**2 * k2 / 4.0) * np.exp(-self.s**2 * k2 / 4.0) / (self.mu * k2 * self.area)
        self.k2 = k2

    def blocks(self, x_eval, x_src):
        """Return Sxx, Sxy, Syy (each M x N) of the periodic kernel (zero cell mean)."""
        dx0 = x_eval[:, None, 0] - x_src[None, :, 0]
        dy0 = x_eval[:, None, 1] - x_src[None, :, 1]
        # wrap separations into the central cell so the nearest copies are the closest
        dx0 = dx0 - self.Lx * np.round(dx0 / self.Lx)
        dy0 = dy0 - self.Ly * np.round(dy0 / self.Ly)

        Sxx = np.zeros_like(dx0)
        Sxy = np.zeros_like(dx0)
        Syy = np.zeros_like(dx0)
        c = 1.0 / (4.0 * np.pi * self.mu)

        # ---- 1. short-range part: sum over nearby copies ----
        for i in range(-self.n_images, self.n_images + 1):
            for j in range(-self.n_images, self.n_images + 1):
                dx = dx0 + i * self.Lx
                dy = dy0 + j * self.Ly
                r = np.sqrt(dx**2 + dy**2)
                A, B = real_space_AB(r, self.eps, self.xi)
                Sxx += c * (A + B * dx * dx)
                Sxy += c * (B * dx * dy)
                Syy += c * (A + B * dy * dy)

        # ---- 2. smooth part: Fourier series ----
        # cos(k.(x_i - x_j)) = cos(k.x_i) cos(k.x_j) + sin(k.x_i) sin(k.x_j), so each tensor entry
        # is two matrix products (M x K)(K x N). This needs memory M*K + K*N, not M*N*K.
        ce = np.cos(x_eval[:, 0:1] * self.kx + x_eval[:, 1:2] * self.ky)      # (M, K)
        se = np.sin(x_eval[:, 0:1] * self.kx + x_eval[:, 1:2] * self.ky)
        cs = np.cos(x_src[:, 0:1] * self.kx + x_src[:, 1:2] * self.ky)        # (N, K)
        ss = np.sin(x_src[:, 0:1] * self.kx + x_src[:, 1:2] * self.ky)
        # (I - k k^T / k^2) gives the three tensor entries
        for S, tensor in ((Sxx, 1.0 - self.kx**2 / self.k2),
                          (Sxy, -self.kx * self.ky / self.k2),
                          (Syy, 1.0 - self.ky**2 / self.k2)):
            w = self.weight * tensor
            S += (ce * w) @ cs.T + (se * w) @ ss.T
        return Sxx, Sxy, Syy

    def velocity(self, x_eval, x_src, forces, U=(0.0, 0.0)):
        """u(x) = U + sum_j P(x - x_j) f_j  (Eq. 4.20). U is the cell-averaged velocity."""
        Sxx, Sxy, Syy = self.blocks(x_eval, x_src)
        ux = U[0] + Sxx @ forces[:, 0] + Sxy @ forces[:, 1]
        uy = U[1] + Sxy @ forces[:, 0] + Syy @ forces[:, 1]
        return np.column_stack([ux, uy])


def spectral_only_blocks(x_eval, x_src, Lx, Ly, eps, mu=1.0, n_max=60):
    """Independent check: the same periodic kernel as a plain Fourier series
        P(x) = (1/(mu area)) sum_{k != 0} exp(-eps^2 k^2/4) (I - k k / k^2) cos(k . x) / k^2
    This converges only because the blob smooths the kernel; it is slow but simple."""
    # n_max modes along the SHORTER side; proportionally more along the longer side,
    # so both directions reach the same largest wavenumber
    nx = int(np.ceil(n_max * Lx / min(Lx, Ly)))
    ny = int(np.ceil(n_max * Ly / min(Lx, Ly)))
    m, n = np.meshgrid(np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), indexing="ij")
    kx = 2 * np.pi * m.ravel() / Lx
    ky = 2 * np.pi * n.ravel() / Ly
    k2 = kx**2 + ky**2
    keep = k2 > 0
    kx, ky, k2 = kx[keep], ky[keep], k2[keep]
    w = np.exp(-eps**2 * k2 / 4.0) / (mu * k2 * Lx * Ly)
    dx = x_eval[:, None, 0] - x_src[None, :, 0]
    dy = x_eval[:, None, 1] - x_src[None, :, 1]
    cosw = np.cos(dx[..., None] * kx + dy[..., None] * ky) * w
    return (np.sum(cosw * (1 - kx**2 / k2), -1),
            np.sum(cosw * (-kx * ky / k2), -1),
            np.sum(cosw * (1 - ky**2 / k2), -1))
