"""
particles2d.py
==============
Our particle tracker for 2D fibre arrays (main.tex Section 8.3).

Units: lengths in mean fibre radii a, velocities in the mean (superficial) air speed U, time in a/U.
A particle of radius r_p (in units of a) moves by the Langevin equation (Eq. 4.21) written as
        dX = V dt ,      dV = (u(X) - V)/St dt + (sqrt(2 D)/St) dW ,
with  St = tau U / a  (tau = particle relaxation time; St = 2 Stk in Hinds' notation)
and   D  = D_p / (a U)  (Brownian diffusion; D = 2 / Pe with Pe = d_f U / D_p).

Time stepping (Eq. 8.4). Over one step h we freeze the air velocity u = u(X_n). The equation is then
linear (an Ornstein-Uhlenbeck process) and can be integrated EXACTLY (Ermak & Buckholz 1980):
    V_{n+1} = u + (V_n - u) e^{-h/St}                    + velocity noise
    X_{n+1} = X_n + u h + (V_n - u) St (1 - e^{-h/St})   + position noise
with Gaussian noises of known variances and covariance (variances() below). For a frozen u this is
exact and stable for any h/St (the error comes only from the change of u over the step, first
order in h); it is used when St >= h. For lighter particles (St < h, which includes St = 0) the
particle velocity relaxes within one step, so inertia cannot be resolved anyway, and we use the
overdamped limit dX = u dt + sqrt(2 D) dW with the stochastic Heun (predictor-corrector) method:
    X* = X_n + u(X_n) h + sqrt(2 D h) Z ,   X_{n+1} = X_n + [u(X_n) + u(X*)] h / 2 + sqrt(2 D h) Z,
which is second-order accurate for the flow part (Euler-Maruyama, Eq. 4.22, is first order).

Capture (Eq. 8.5): when the particle centre comes within a_m + r_p of a fibre axis. A Brownian
path can touch the capture circle and come back out within one step; the probability of that,
given the start and end distances d0, d1 > 0 from the circle, is exp(-2 d0 d1 / s2) for a flat
wall, with s2 the variance of the position step per component (the Brownian-bridge formula;
s2 = 2 D h in the overdamped regime, s2 = sx2 of variances() in the inertial one). We draw that
chance every step whenever D > 0. For heavy particles s2 is tiny and the correction vanishes, as it
should: their paths are smooth on the scale of one step.

One pass (Section 8.4): particles enter on a vertical line x = x0 chosen as far as possible from the
capture circles (see inlet_line for lines that must cut one),
spread uniformly in stream function (i.e. in proportion to the flux), and are followed until they
are caught or reach x0 + Lx (one cell further downstream).
"""

import numpy as np


def inlet_line(layout, r_p, n_test=400):
    """x0 of the vertical line that stays farthest from every capture circle (centres wrapped),
    and its clearance. In a dense random cell every line may cut some capture circle (negative
    clearance); that is allowed: particles that start inside a capture circle are caught at once,
    which is correct because, the flow being periodic, the part of a fibre upstream of x0 is met
    again (as its periodic copy) before the pass ends at x0 + Lx."""
    Lx, Ly = layout["Lx"], layout["Ly"]
    xs = (np.arange(n_test) + 0.5) * Lx / n_test
    best, x_best = -np.inf, None
    for x0 in xs:
        dx = np.abs(((layout["centres"][:, 0] - x0) + Lx / 2) % Lx - Lx / 2)
        clearance = np.min(dx - layout["radii"] - r_p)
        if clearance > best:
            best, x_best = clearance, x0
    return x_best, best


def inlet_positions(field, x0, n, rng, n_line=4000):
    """n starting points on x = x0, uniform in stream function over one period (flux-weighted),
    stratified: one point in each of n equal flux bands, at a random place inside the band."""
    Ly = field.Ly
    y = np.linspace(0.0, Ly, n_line + 1)
    psi = field.psi(np.column_stack([np.full_like(y, x0), y]))
    psi = np.maximum.accumulate(psi)                 # guard against tiny negative u (not expected)
    targets = psi[0] + (np.arange(n) + rng.random(n)) / n * (psi[-1] - psi[0])
    ys = np.interp(targets, psi, y)
    return np.column_stack([np.full(n, x0), ys])


def variances(h, St, D):
    """Variances of the position noise (sx2) and velocity noise (sv2) and their covariance (sxv),
    per component, over one step h of the Ornstein-Uhlenbeck process (Ermak & Buckholz 1980).
    With g = h/St:
        sx2 = 2 D St [g - 3/2 + 2 e^{-g} - e^{-2g}/2]   (series g^3/3 - g^4/4 + 7 g^5/60 for small g)
        sxv = D (1 - e^{-g})^2
        sv2 = (D/St) (1 - e^{-2g})."""
    g = h / St
    if g < 1e-2:
        bracket = g**3 / 3 - g**4 / 4 + 7 * g**5 / 60
    else:
        bracket = g - 1.5 + 2 * np.exp(-g) - 0.5 * np.exp(-2 * g)
    sx2 = 2 * D * St * bracket
    sxv = D * (1 - np.exp(-g)) ** 2
    sv2 = D / St * (1 - np.exp(-2 * g))
    return sx2, sxv, sv2


def bridge_probability(d0, d1, s2):
    """Chance that a Brownian path with position-step variance s2 (per component) touched a flat
    wall during a step that starts and ends at distances d0, d1 >= 0 from it:
    exp(-2 d0 d1 / s2) (the Brownian-bridge result; Mannella 1999)."""
    return np.exp(-2.0 * d0 * d1 / s2)


def gap_to_capture(X, layout, r_p):
    """Distance from each particle centre to the nearest capture circle (negative = inside),
    and the index of that fibre. Periodic in both directions."""
    c, a = layout["centres"], layout["radii"]
    L = np.array([layout["Lx"], layout["Ly"]])
    d = X[:, None, :] - c[None, :, :]
    d -= L * np.round(d / L)
    gap = np.sqrt(np.sum(d**2, axis=2)) - (a + r_p)[None, :]
    j = np.argmin(gap, axis=1)
    return gap[np.arange(len(X)), j], j


def track_pass(field, layout, r_p, D, St, h, n, rng, x0=None, max_passes_time=30.0, bridge=True,
               v_init="fluid"):
    """Follow n particles through one cell. Returns a dict with
        caught (bool, n), fibre (index of the catching fibre, -1 if not caught),
        stuck (bool, n: still in the cell after max_passes_time x Lx/U, counted as not caught),
        steps (number of time steps taken).
    v_init = "fluid": particles start with the air velocity (the usual case);
             "mean" : with the mean velocity (U, 0) (used for the straight-line test)."""
    if x0 is None:
        x0, clearance = inlet_line(layout, r_p)
        if clearance + r_p <= 0:
            # the line would pass through a fibre itself; the stream function there is not the
            # flux of the air (the fibre is solid), so this case is refused (it never occurs here)
            raise ValueError("every vertical line cuts a fibre: inlet not supported")
    X = inlet_positions(field, x0, n, rng)
    V = field.velocity(X) if v_init == "fluid" else np.tile([field.U, 0.0], (n, 1))
    X0 = X.copy()
    gap0, j0 = gap_to_capture(X, layout, r_p)
    start_in = gap0 <= 0                      # started inside a capture circle (see inlet_line)
    caught = np.zeros(n, bool)
    fibre = np.full(n, -1)
    active = np.ones(n, bool)
    caught[start_in], fibre[start_in], active[start_in] = True, j0[start_in], False
    inertial = St > 0 and St >= h            # below this, the overdamped limit is used
    if inertial:
        e1 = np.exp(-h / St)
        sx2, sxv, sv2 = variances(h, St, D) if D > 0 else (0.0, 0.0, 0.0)
        sx = np.sqrt(sx2)
        a_v = sxv / sx if sx > 0 else 0.0
        b_v = np.sqrt(max(sv2 - a_v**2, 0.0))
        s2 = sx2                               # position-step variance, for the bridge correction
    else:
        s2 = 2 * D * h
    t_max = max_passes_time * layout["Lx"]
    steps = 0
    t = 0.0
    while active.any() and t < t_max:
        idx = np.nonzero(active)[0]
        Xa, Va = X[idx], V[idx]
        u = field.velocity(Xa)
        m = len(idx)
        if inertial:
            Z1, Z2 = rng.standard_normal((m, 2)), rng.standard_normal((m, 2))
            Xn = Xa + u * h + (Va - u) * St * (1 - e1) + sx * Z1
            Vn = u + (Va - u) * e1 + a_v * Z1 + b_v * Z2
        else:
            # overdamped: stochastic Heun (predictor-corrector) step; the same noise in both stages
            noise = np.sqrt(2 * D * h) * rng.standard_normal((m, 2)) if D > 0 else 0.0
            Xp = Xa + u * h + noise
            Vn = 0.5 * (u + field.velocity(Xp))
            Xn = Xa + Vn * h + noise
        g1, j1 = gap_to_capture(Xn, layout, r_p)
        hit = g1 <= 0
        if bridge and D > 0 and s2 > 0:
            # Brownian bridge: chance that the path touched the circle during the step
            p = bridge_probability(np.maximum(gap0[idx], 0.0), np.maximum(g1, 0.0), s2)
            hit |= rng.random(m) < p
        X[idx], V[idx] = Xn, Vn
        gap0[idx] = g1
        ci = idx[hit]
        caught[ci] = True
        fibre[ci] = j1[hit]
        active[ci] = False
        out = idx[(~hit) & (Xn[:, 0] >= x0 + layout["Lx"])]
        active[out] = False
        t += h
        steps += 1
    return dict(caught=caught, fibre=fibre, stuck=active.copy(), steps=steps, x0=x0, start=X0)


# ----------------------------------------------------------------------------
# From the captured fraction of one pass to the single-fibre efficiency
# ----------------------------------------------------------------------------

def efficiency_from_pass(f_caught, layout):
    """Single-fibre efficiency eta from the fraction f caught in one pass (Eq. 8.6).
    eta = flux caught by one fibre / (U d_f), U = face (superficial) speed, d_f = 2 a_mean: the
    convention of the Kuwabara-based formulas (checked in main.tex Section 8.5). A cell of length Lx
    holds fibre length 4 alpha Lx / (pi d_f^2) per unit area, so P_cell = 1 - f = exp(-4 alpha eta
    Lx / (pi d_f)), i.e.  eta = -ln(1 - f) pi d_f / (4 alpha Lx), and a filter of thickness L has
    P = exp(-4 alpha eta L / (pi d_f))."""
    a = layout["radii"]
    alpha = np.sum(np.pi * a**2) / (layout["Lx"] * layout["Ly"])
    d_f = 2 * np.mean(a)
    return -np.log(1.0 - f_caught) * np.pi * d_f / (4 * alpha * layout["Lx"])


def interception_exact(field, layout, r_p, n_circle=4000):
    """Pure interception (no diffusion, no inertia) in one pass, EXACTLY from the stream function:
    a particle on the streamline psi is caught if that streamline comes within r_p of a fibre.
    The streamlines that touch capture circle m are those with psi between the smallest and
    largest psi on that circle. The caught fraction is the length of the union of these
    intervals (taken modulo the flux U Ly through one cell) divided by U Ly."""
    period = field.U * layout["Ly"]
    th = 2 * np.pi * np.arange(n_circle) / n_circle
    intervals = []
    for c, a in zip(layout["centres"], layout["radii"]):
        P = c + (a + r_p) * np.column_stack([np.cos(th), np.sin(th)])
        ps = field.psi(P)
        lo, hi = ps.min(), ps.max()
        lo_w = lo % period
        intervals.append((lo_w, lo_w + (hi - lo)))
    # union on a circle of circumference `period`: unroll intervals that wrap past the end
    segs = []
    for lo, hi in intervals:
        if hi <= period:
            segs.append((lo, hi))
        else:
            segs.append((lo, period))
            segs.append((0.0, hi - period))
    segs.sort()
    total, cur_lo, cur_hi = 0.0, None, None
    for lo, hi in segs:
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    total += cur_hi - cur_lo
    return total / period


def ballistic_caught(start, layout, r_p):
    """Straight-line prediction, particle by particle: a particle moving along x from height y is
    caught by fibre m if |y - y_m| <= a_m + r_p (periodic in y) and it reaches that fibre within one
    cell (always true for a full pass). Returns a boolean array."""
    Ly = layout["Ly"]
    dy = start[:, 1][:, None] - layout["centres"][None, :, 1]
    dy -= Ly * np.round(dy / Ly)
    return np.any(np.abs(dy) <= (layout["radii"] + r_p)[None, :], axis=1)
