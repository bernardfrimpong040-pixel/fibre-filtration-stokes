"""
periodic3d.py
=============
Our triply periodic 3D regularized Stokeslet, by Ewald summation (main.tex Section 7.2).

Same idea as the 2D version (periodic.py):
    periodic kernel = short-range part (nearby copies, real space)
                    + smooth part (Fourier series over k = 2 pi (l/Lx, m/Ly, n/Lz), k != 0)
With Hasimoto's split factor (1 + xi^2 k^2/4) exp(-xi^2 k^2/4) and s^2 = eps^2 + xi^2:
    short-range:  K_eps - K_s + xi^2 dK_s/d(s^2)          (decays like exp(-r^2/s^2))
    Fourier:      (1/(mu V)) sum (1 + xi^2 k^2/4) exp(-s^2 k^2/4) (I - k k / k^2) cos(k . x) / k^2
The k = 0 term is left out: the net force on the fluid in a cell is balanced by the mean
pressure gradient (Proposition 4 holds unchanged in 3D), and U is the cell-averaged velocity.

The matrix has 3 x 3 blocks; we return the six independent components xx, xy, xz, yy, yz, zz.
Rows are processed in chunks so memory stays bounded.
"""

import numpy as np

from rs_kernels3d import kernel_H_3d, dH_ds2_3d

COMPONENTS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def real_space_H(r, eps, xi):
    """Short-range scalars H1_R, H2_R:  K_eps - K_s + xi^2 dK_s/d(s^2)."""
    s = np.sqrt(eps**2 + xi**2)
    a1, a2 = kernel_H_3d(r, eps)
    b1, b2 = kernel_H_3d(r, s)
    d1, d2 = dH_ds2_3d(r, s)
    return a1 - b1 + xi**2 * d1, a2 - b2 + xi**2 * d2


class PeriodicStokeslet3D:
    """Triply periodic regularized Stokeslet in an Lx x Ly x Lz cell."""

    def __init__(self, L, eps, mu=1.0, xi=None, n_images=None, tol=36.0):
        self.L = np.array(L, dtype=float)
        self.eps, self.mu = float(eps), float(mu)
        self.volume = float(np.prod(self.L))
        Lmin = self.L.min()
        self.xi = xi if xi is not None else Lmin / 6.0
        self.s = np.sqrt(self.eps**2 + self.xi**2)
        # copies needed: after wrapping, left-out copies are >= (n + 1/2) Lmin away
        if n_images is None:
            n_images = max(1, int(np.ceil(np.sqrt(tol) * self.s / Lmin - 0.5)))
        self.n_images = n_images
        self.r_cut = np.sqrt(tol) * self.s
        if ((n_images + 0.5) * Lmin) ** 2 / self.s**2 < tol:
            raise ValueError("blob too wide for this cell: increase n_images or reduce eps")
        # Fourier wavevectors: keep one of each pair (k, -k); the kernel is even, so the pair
        # contributes twice the cosine term. k = 0 is left out.
        # Fourier terms fall like exp(-s^2 k^2 / 4); stopping at k_max leaves terms below
        # exp(-(tol + 5)) ~ 1e-18 (for tol = 36), so the Fourier tail is negligible
        k_max = 2.0 * np.sqrt(tol + 5.0) / self.s
        m = [int(np.ceil(k_max * Li / (2 * np.pi))) for Li in self.L]
        g = np.meshgrid(*[np.arange(-mi, mi + 1) for mi in m], indexing="ij")
        n = np.column_stack([gi.ravel() for gi in g])
        # half space: first non-zero index positive
        first = np.where(n[:, 0] != 0, n[:, 0], np.where(n[:, 1] != 0, n[:, 1], n[:, 2]))
        n = n[first > 0]
        k = 2 * np.pi * n / self.L
        k2 = np.sum(k**2, axis=1)
        keep = k2 <= k_max**2
        self.k, self.k2 = k[keep], k2[keep]
        self.weight = 2.0 * (1.0 + self.xi**2 * self.k2 / 4.0) * np.exp(-self.s**2 * self.k2 / 4.0) \
            / (self.mu * self.k2 * self.volume)
        # (I - k k / k^2) for each of the six components, folded into the weights
        self.tensor_weights = [self.weight * ((i == j) - self.k[:, i] * self.k[:, j] / self.k2)
                               for i, j in COMPONENTS]
        offs = np.arange(-n_images, n_images + 1)
        self.images = np.array(np.meshgrid(offs, offs, offs, indexing="ij")).reshape(3, -1).T * self.L

    def blocks(self, x_eval, x_src, chunk=400, full=None):
        """The six kernel components, each an (M, N) array (velocity per unit force).
        If `full` (a 3M x 3N array) is given, the components are written straight into its
        3 x 3 sub-blocks instead (saves memory: no separate copies of the six blocks)."""
        M, N = len(x_eval), len(x_src)
        out = [np.zeros((M, N)) for _ in COMPONENTS] if full is None else None
        c = 1.0 / (8.0 * np.pi * self.mu)
        # Fourier factors for the sources (computed once)
        ps = x_src @ self.k.T
        cs, ss = np.cos(ps), np.sin(ps)
        for i0 in range(0, M, chunk):
            xe = x_eval[i0:i0 + chunk]
            d0 = xe[:, None, :] - x_src[None, :, :]
            d0 -= self.L * np.round(d0 / self.L)              # wrap into the central cell
            blk = [np.zeros(d0.shape[:2]) for _ in COMPONENTS]
            # ---- short-range part over nearby copies ----
            for img in self.images:
                d = d0 + img
                r = np.sqrt(np.sum(d**2, axis=2))
                # only pairs closer than the cut-off matter (beyond it the short-range part is
                # below exp(-tol)); evaluating just those saves most of the work
                near = r < self.r_cut
                if not np.any(near):
                    continue
                H1, H2 = real_space_H(r[near], self.eps, self.xi)
                dn = d[near]
                for b, (i, j) in zip(blk, COMPONENTS):
                    b[near] += c * (H1 * (i == j) + H2 * dn[:, i] * dn[:, j])
            # ---- smooth part: Fourier series ----
            pe = xe @ self.k.T
            ce, se = np.cos(pe), np.sin(pe)
            for b, w in zip(blk, self.tensor_weights):
                b += (ce * w) @ cs.T + (se * w) @ ss.T
            if full is None:
                for o, b in zip(out, blk):
                    o[i0:i0 + chunk] = b
            else:
                rows = slice(i0, i0 + len(xe))
                for b, (i, j) in zip(blk, COMPONENTS):
                    full[i * M + rows.start:i * M + rows.stop, j * N:(j + 1) * N] = b
                    if i != j:                                   # the matrix is symmetric in (i, j)
                        full[j * M + rows.start:j * M + rows.stop, i * N:(i + 1) * N] = b
        return out

    def matrix(self, x):
        """The full 3N x 3N matrix for points x (N, 3): u_i = sum_j M_ij f_j,
        ordered [all x-components, all y-components, all z-components]."""
        N = len(x)
        full = np.empty((3 * N, 3 * N))
        self.blocks(x, x, full=full)
        return full

    def velocity(self, x_eval, x_src, forces, U=(0.0, 0.0, 0.0)):
        bxx, bxy, bxz, byy, byz, bzz = self.blocks(np.atleast_2d(x_eval), x_src)
        fx, fy, fz = forces[:, 0], forces[:, 1], forces[:, 2]
        return np.column_stack([U[0] + bxx @ fx + bxy @ fy + bxz @ fz,
                                U[1] + bxy @ fx + byy @ fy + byz @ fz,
                                U[2] + bxz @ fx + byz @ fy + bzz @ fz])


def spectral_only_blocks_3d(x_eval, x_src, L, eps, mu=1.0, n_max=12):
    """Independent check: the plain Fourier series with the blob factor exp(-eps^2 k^2/4).
    Converges only because the blob smooths the kernel (slowly for small eps)."""
    L = np.array(L, dtype=float)
    g = np.meshgrid(*[np.arange(-n_max, n_max + 1)] * 3, indexing="ij")
    n = np.column_stack([gi.ravel() for gi in g])
    n = n[np.any(n != 0, axis=1)]
    k = 2 * np.pi * n / L
    k2 = np.sum(k**2, axis=1)
    w = np.exp(-eps**2 * k2 / 4) / (mu * k2 * np.prod(L))
    d = x_eval[:, None, :] - x_src[None, :, :]
    cosw = np.cos(d @ k.T) * w
    return [np.sum(cosw * ((i == j) - k[:, i] * k[:, j] / k2), axis=-1) for i, j in COMPONENTS]
