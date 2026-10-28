"""
geometry.py
===========
Fibre arrangements in one periodic 2D cell, and the points placed on each fibre.

A "layout" is a small dictionary:
    centres : (M, 2) array of fibre centres
    radii   : (M,)   array of fibre radii
    Lx, Ly  : cell size
All lengths are in units of the mean fibre radius unless stated otherwise.

The solid volume fraction of a layout is   alpha = sum(pi a_m^2) / (Lx Ly).
"""

import numpy as np


# ----------------------------------------------------------------------------
# Regular arrays (used for validation against exact results)
# ----------------------------------------------------------------------------

def square_array(alpha, a=1.0):
    """One fibre of radius a in a square cell chosen so the solid fraction is alpha."""
    L = np.sqrt(np.pi * a**2 / alpha)
    return dict(centres=np.array([[L / 2, L / 2]]), radii=np.array([a]), Lx=L, Ly=L)


def hexagonal_array(alpha, a=1.0):
    """Hexagonal (triangular) array as a rectangular cell L by sqrt(3) L holding two fibres.
    Each fibre has six neighbours at distance L. Solid fraction: 2 pi a^2 / (sqrt(3) L^2)."""
    L = np.sqrt(2 * np.pi * a**2 / (np.sqrt(3) * alpha))
    Ly = np.sqrt(3) * L
    centres = np.array([[L / 4, Ly / 4], [3 * L / 4, 3 * Ly / 4]])
    return dict(centres=centres, radii=np.array([a, a]), Lx=L, Ly=Ly)


# ----------------------------------------------------------------------------
# Random arrays (the real question)
# ----------------------------------------------------------------------------

def draw_radii(n_fibres, cv, rng, a_mean=1.0):
    """Fibre radii with mean a_mean and coefficient of variation cv (std/mean).
    cv = 0 gives identical fibres. Otherwise radii are lognormal, the usual
    model for melt-blown and electrospun fibre diameters."""
    if cv == 0:
        return np.full(n_fibres, a_mean)
    sigma2 = np.log(1 + cv**2)                       # lognormal parameters giving this mean and cv
    mu = np.log(a_mean) - sigma2 / 2
    return rng.lognormal(mu, np.sqrt(sigma2), n_fibres)


def random_array(alpha, n_fibres, rng, cv=0.0, min_gap=0.2, aspect=1.0, max_tries=200000):
    """Random non-overlapping fibres in a periodic cell (random sequential addition).

    alpha    : target solid fraction (met exactly by choosing the cell size)
    n_fibres : number of fibres in the cell
    cv       : spread of radii (0 = monodisperse)
    min_gap  : smallest allowed surface-to-surface gap, in units of the mean radius.
               Real fibres touch; we keep a small gap so that the numerical
               resolution near every gap can be controlled (Section 5).
    aspect   : Lx / Ly
    Fibres are placed largest first, which makes dense packings easier to reach.
    """
    radii = np.sort(draw_radii(n_fibres, cv, rng))[::-1]
    area = np.sum(np.pi * radii**2) / alpha
    Ly = np.sqrt(area / aspect)
    Lx = aspect * Ly
    centres = np.zeros((n_fibres, 2))
    placed = 0
    tries = 0
    while placed < n_fibres:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("could not place all fibres: lower alpha or min_gap")
        trial = rng.random(2) * [Lx, Ly]
        if placed > 0:
            d = trial - centres[:placed]
            d -= np.array([Lx, Ly]) * np.round(d / [Lx, Ly])      # nearest periodic copy
            dist = np.sqrt(np.sum(d**2, axis=1))
            if np.any(dist < radii[:placed] + radii[placed] + min_gap):
                continue
        centres[placed] = trial
        placed += 1
    return dict(centres=centres, radii=radii, Lx=Lx, Ly=Ly)


def solid_fraction(layout):
    return np.sum(np.pi * layout["radii"] ** 2) / (layout["Lx"] * layout["Ly"])


def min_surface_gap(layout):
    """Smallest surface-to-surface distance between any two fibres (periodic)."""
    c, r = layout["centres"], layout["radii"]
    L = np.array([layout["Lx"], layout["Ly"]])
    best = np.inf
    for i in range(len(r)):
        for j in range(len(r)):
            d = c[j] - c[i]
            d -= L * np.round(d / L)
            if i == j:
                continue
            best = min(best, np.linalg.norm(d) - r[i] - r[j])
    # a fibre also faces its own periodic copies one cell away
    best = min(best, np.min(L) - 2 * np.max(r))
    return best


# ----------------------------------------------------------------------------
# Points on each fibre
# ----------------------------------------------------------------------------

def ring_points(layout, spacing, offset=0.0):
    """Place points on every fibre, evenly spaced along its circumference.

    spacing : target arc length h between neighbouring points
    offset  : the points sit on a circle of radius (a_m - offset); offset > 0 pulls
              them inside the fibre (used by the regularized Stokeslet method, Sec. 5)
    Returns
        points   : (N, 2) all points
        fibre_id : (N,)   which fibre each point belongs to
        n_per    : (M,)   number of points on each fibre
        weights  : (N,)   arc length each point represents on the TRUE surface (2 pi a / n)
    """
    pts, ids, n_per, wts = [], [], [], []
    for m, (c, a) in enumerate(zip(layout["centres"], layout["radii"])):
        n = max(8, int(np.ceil(2 * np.pi * a / spacing)))
        t = 2 * np.pi * np.arange(n) / n
        R = a - offset
        pts.append(c + R * np.column_stack([np.cos(t), np.sin(t)]))
        ids.append(np.full(n, m))
        n_per.append(n)
        wts.append(np.full(n, 2 * np.pi * a / n))
    return np.vstack(pts), np.concatenate(ids), np.array(n_per), np.concatenate(wts)
