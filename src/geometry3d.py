"""
geometry3d.py
=============
Straight, infinitely long, PERIODIC fibres in a cubic cell of side L (main.tex Section 7.3).

Why lattice directions
----------------------
A straight fibre in a periodic box joins up with itself only if its direction is a
lattice vector n = (n1, n2, n3) (small integers). Its length inside the cell is then
|n| L. We use these direction sets:

  "aligned"   : all fibres along x               (transverse to the flow along z: the 2D case)
  "axial"     : all fibres along z               (parallel to the flow)
  "planar"    : (1,0,0) (0,1,0) (1,1,0) (1,-1,0)  fibres lie in planes normal to the flow, like the
                layers of a nonwoven; their orientation tensor <n n> is (xx + yy)/2
  "layered"   : (1,0,0) (0,1,0) only: crossed layers, the simplest model of a nonwoven mat
  "cubic"     : (1,0,0) (0,1,0) (0,0,1): <n n> = I/3, the cheapest isotropic orientation tensor
  "isotropic" : the 13 directions of a cube (3 edges, 6 face diagonals, 4 body diagonals);
                <n n> = I/3 exactly

Directions are drawn with probability proportional to 1/|n|, so that every direction gets
the same expected fibre LENGTH (volume), which is what makes <n n> come out as stated.

A fibre is a dict(point, dirn (integer lattice vector), radius). A network is
dict(L, fibres).  Solid fraction  alpha = sum(pi a^2 |n| L) / L^3  (fibres do not overlap).
"""

import numpy as np

DIRECTION_SETS = {
    "aligned": [(1, 0, 0)],
    "axial": [(0, 0, 1)],
    "planar": [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, -1, 0)],
    "layered": [(1, 0, 0), (0, 1, 0)],
    "cubic": [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    "isotropic": [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                  (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1),
                  (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1)],
}


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def orientation_tensor(network):
    """Volume-weighted orientation tensor <n n> = sum(a^2 l n n) / sum(a^2 l)."""
    T = np.zeros((3, 3))
    w = 0.0
    for f in network["fibres"]:
        d = unit(f["dirn"])
        wt = f["radius"] ** 2 * np.linalg.norm(f["dirn"])
        T += wt * np.outer(d, d)
        w += wt
    return T / w


def solid_fraction(network):
    L = network["L"]
    return sum(np.pi * f["radius"] ** 2 * np.linalg.norm(f["dirn"]) * L for f in network["fibres"]) / L**3


# ----------------------------------------------------------------------------
# Distances between periodic fibres
# ----------------------------------------------------------------------------

def axis_distance(f1, f2, L):
    """Smallest distance between the axis of f1 and the axis of f2 or of any periodic copy.
    (If f1 is f2, the distance to its own copies that are not on the same line.)"""
    d1, d2 = np.array(f1["dirn"]), np.array(f2["dirn"])
    m = np.cross(d1, d2)
    dp = np.asarray(f2["point"]) - np.asarray(f1["point"])
    if np.any(m != 0):
        # Non-parallel: distance to the copy shifted by L*(i,j,k) is |(dp + L ijk) . m| / |m|.
        # The values L*(ijk . m) are all multiples of L*g, g = gcd of the components of m.
        g = np.gcd.reduce(np.abs(m[m != 0]))
        spacing = L * g / np.linalg.norm(m)
        c = dp @ m / np.linalg.norm(m)
        return abs(c - spacing * np.round(c / spacing))
    # Parallel: distance in the plane normal to the common direction, over nearby copies
    dhat = unit(d1)
    best = np.inf
    rng = range(-3, 4)
    for i in rng:
        for j in rng:
            for k in rng:
                v = dp + L * np.array([i, j, k])
                perp = v - (v @ dhat) * dhat
                dist = np.linalg.norm(perp)
                # skip copies on the SAME line (zero distance). Two different fibres exactly on one
                # line would also be skipped, but random placement makes that impossible in practice.
                if dist > 1e-9 * L:
                    best = min(best, dist)
    return best


def min_surface_gap(network):
    """Smallest surface-to-surface gap between any two fibres (including a fibre and its copies)."""
    fs, L = network["fibres"], network["L"]
    best = np.inf
    for i in range(len(fs)):
        for j in range(i, len(fs)):
            best = min(best, axis_distance(fs[i], fs[j], L) - fs[i]["radius"] - fs[j]["radius"])
    return best


# ----------------------------------------------------------------------------
# Random networks
# ----------------------------------------------------------------------------

def random_network(alpha, n_fibres, rng, kind="isotropic", cv=0.0, min_gap=0.4, max_tries=50000,
                   balanced=False):
    """Random periodic network of n_fibres straight fibres (mean radius 1) at solid fraction alpha.

    1. draw directions from DIRECTION_SETS[kind] with probability ~ 1/|n| and radii (lognormal, cv);
       with balanced=True every direction of the set is used equally often instead (n_fibres must
       then be a multiple of the number of directions), so <n n> is exact in every realization;
    2. choose the cell size L so the solid fraction is exactly alpha;
    3. place the fibres one by one at random positions, rejecting any that comes closer than
       min_gap (surface to surface) to a fibre already placed or to its own copies.
    """
    dirs = DIRECTION_SETS[kind]
    p = np.array([1.0 / np.linalg.norm(d) for d in dirs])
    p /= p.sum()
    if balanced:
        if n_fibres % len(dirs):
            raise ValueError("balanced=True needs n_fibres to be a multiple of the number of directions")
        chosen = [dirs[i % len(dirs)] for i in range(n_fibres)]
    else:
        chosen = [dirs[i] for i in rng.choice(len(dirs), n_fibres, p=p)]
    if cv == 0:
        radii = np.ones(n_fibres)
    else:
        s2 = np.log(1 + cv**2)
        radii = rng.lognormal(-s2 / 2, np.sqrt(s2), n_fibres)
    order = np.argsort(-radii)                       # largest first
    radii = radii[order]
    chosen = [chosen[i] for i in order]
    L = np.sqrt(sum(np.pi * a**2 * np.linalg.norm(d) for a, d in zip(radii, chosen)) / alpha)
    net = dict(L=L, fibres=[])
    tries = 0
    for a, d in zip(radii, chosen):
        while True:
            tries += 1
            if tries > max_tries:
                raise RuntimeError("could not place all fibres: lower alpha or min_gap")
            f = dict(point=rng.random(3) * L, dirn=tuple(d), radius=a)
            if axis_distance(f, f, L) < 2 * a + min_gap:
                continue
            if all(axis_distance(f, g, L) >= a + g["radius"] + min_gap for g in net["fibres"]):
                net["fibres"].append(f)
                break
    return net


def lattice_network(kind, alpha, L=None):
    """Simple deterministic networks for validation.
    "aligned_square": one fibre along x per cell (square array in the y-z plane),
    "axial_square":   one fibre along z per cell,
    "cubic3":         three mutually perpendicular fibres per cell (a 3D 'jungle gym', non-touching),
    "tilted":         one fibre along (1, 0, 1) per cell, i.e. at 45 degrees to the flow along z.
                      Its copies form a RECTANGULAR array, L/sqrt(2) by L, in the plane normal to
                      the fibre, so its permeability is known exactly from two 2D problems
                      (Eq. 7.12): this tests mixed axial + transverse flow."""
    if kind == "aligned_square":
        L = np.sqrt(np.pi / alpha)
        return dict(L=L, fibres=[dict(point=(0.0, L / 2, L / 2), dirn=(1, 0, 0), radius=1.0)])
    if kind == "axial_square":
        L = np.sqrt(np.pi / alpha)
        return dict(L=L, fibres=[dict(point=(L / 2, L / 2, 0.0), dirn=(0, 0, 1), radius=1.0)])
    if kind == "cubic3":
        L = np.sqrt(3 * np.pi / alpha)
        return dict(L=L, fibres=[dict(point=(0.0, 0.0, L / 2), dirn=(1, 0, 0), radius=1.0),
                                 dict(point=(L / 2, 0.0, 0.0), dirn=(0, 1, 0), radius=1.0),
                                 dict(point=(0.0, L / 2, 0.0), dirn=(0, 0, 1), radius=1.0)])
    if kind == "tilted":
        L = np.sqrt(np.sqrt(2) * np.pi / alpha)
        return dict(L=L, fibres=[dict(point=(0.0, L / 2, 0.0), dirn=(1, 0, 1), radius=1.0)])
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Points on (or along) the fibres
# ----------------------------------------------------------------------------

def frame(dirn):
    """Unit axis t and two unit vectors e1, e2 normal to it."""
    t = unit(dirn)
    helper = np.array([1.0, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = unit(np.cross(t, helper))
    e2 = np.cross(t, e1)
    return t, e1, e2


def surface_points(network, spacing, offset=0.0):
    """Points on circles of radius (a - offset) around each fibre axis, about `spacing` apart
    around the circle and along the axis (exact periodic closure along the fibre).
    Returns points (N, 3) and the fibre index of each point."""
    pts, ids = [], []
    L = network["L"]
    for m, f in enumerate(network["fibres"]):
        a = f["radius"]
        length = np.linalg.norm(f["dirn"]) * L
        n_ring = max(8, int(np.ceil(2 * np.pi * a / spacing)))
        n_ax = max(1, int(np.ceil(length / spacing)))
        t, e1, e2 = frame(f["dirn"])
        theta = 2 * np.pi * np.arange(n_ring) / n_ring
        R = a - offset
        circle = R * (np.outer(np.cos(theta), e1) + np.outer(np.sin(theta), e2))
        for q in range(n_ax):
            # rotate every other ring by half a step: a staggered (near-hexagonal) pattern
            shift = (q % 2) * np.pi / n_ring
            c2 = R * (np.outer(np.cos(theta + shift), e1) + np.outer(np.sin(theta + shift), e2))
            pts.append(np.asarray(f["point"]) + (q * length / n_ax) * t + c2)
            ids.append(np.full(n_ring, m))
    P = np.vstack(pts)
    return P - L * np.floor(P / L), np.concatenate(ids)


def centreline_points(network, spacing):
    """Points along each fibre axis, about `spacing` apart (exact periodic closure)."""
    pts, ids = [], []
    L = network["L"]
    for m, f in enumerate(network["fibres"]):
        length = np.linalg.norm(f["dirn"]) * L
        n_ax = max(1, int(np.ceil(length / spacing)))
        t = unit(f["dirn"])
        s = np.arange(n_ax) * length / n_ax
        pts.append(np.asarray(f["point"]) + np.outer(s, t))
        ids.append(np.full(n_ax, m))
    P = np.vstack(pts)
    return P - L * np.floor(P / L), np.concatenate(ids)


def disk_points(network, spacing, radius_shift=0.0):
    """Points filling each fibre's cross-section (radius a - radius_shift), repeated along the
    axis every `spacing`, with area weights. Used for the pressure-balance body force inside
    the fibres (main.tex Section 7.4). Polar layout: rings at r_j = (j + 1/2) h, h ~ spacing,
    each with about 2 pi r_j / h points; weights add up exactly to pi R^2 per cross-section.
    Returns points (N, 3), the weight of each point (area x axial step) and the fibre index."""
    pts, wts, ids = [], [], []
    L = network["L"]
    for m, f in enumerate(network["fibres"]):
        R = f["radius"] - radius_shift
        length = np.linalg.norm(f["dirn"]) * L
        n_ax = max(1, int(np.ceil(length / spacing)))
        n_r = max(1, int(round(R / spacing)))
        h = R / n_r
        t, e1, e2 = frame(f["dirn"])
        disk, dw = [], []
        for j in range(n_r):
            r = (j + 0.5) * h
            n_th = max(1, int(round(2 * np.pi * r / h)))
            th = 2 * np.pi * (np.arange(n_th) + 0.5 * j) / n_th
            disk.append(r * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2)))
            dw.append(np.full(n_th, np.pi * ((r + h / 2) ** 2 - (r - h / 2) ** 2) / n_th))
        disk, dw = np.vstack(disk), np.concatenate(dw)
        for q in range(n_ax):
            pts.append(np.asarray(f["point"]) + (q * length / n_ax) * t + disk)
            wts.append(dw * length / n_ax)
            ids.append(np.full(len(dw), m))
    P = np.vstack(pts)
    return P - L * np.floor(P / L), np.concatenate(wts), np.concatenate(ids)
