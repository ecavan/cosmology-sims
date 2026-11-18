"""Flipped-RVM fit: screen a (z1, z2 − z1) grid in ν at the ΛCDM best fit, then polish the best points in all six."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .inference import START, STEP, log_like
from .models import PARAMS, PRIORS

Z1_GRID = tuple(np.round(np.arange(0.2, 3.01, 0.2), 2))
DZ_GRID = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0)


def chi2(model, p, d):
    L = log_like(np.asarray(p, float), model, d)
    return -2 * L if np.isfinite(L) else 1e30


def nm(model, d, x0, maxfev=6000):
    """Nelder–Mead over all parameters with an initial simplex of one STEP per axis."""
    x0 = np.asarray(x0, float)
    scale = np.array([STEP[n] for n in PARAMS[model]])
    simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[k] * scale[k] for k in range(len(x0))])
    res = minimize(lambda q: chi2(model, q, d), x0, method="Nelder-Mead",
                   options={"initial_simplex": simplex, "xatol": 1e-8, "fatol": 1e-4, "maxfev": maxfev,
                            "adaptive": True})
    return res.x, res.fun


def fit_simple(model, d, start=None):
    """Nelder–Mead with one restart (it can stall)."""
    x0 = np.array(start if start is not None else START[model], float)
    x, c = nm(model, d, x0)
    x, c2 = nm(model, d, x)
    return x, min(c, c2)


def fit_flipped(d, x_lcdm, z1s=Z1_GRID, dzs=DZ_GRID, n_polish=4):
    """Returns (best x, χ², screen [(z1, Δz, χ², ν)])."""
    h, wb, wbc = x_lcdm
    lo, hi = PRIORS["nu_f"]
    screen = []
    for z1 in z1s:
        for dz in dzs:
            if z1 + dz > 10.0:
                continue
            f = lambda nu: chi2("flipped", [h, wb, wbc, nu, z1, z1 + dz], d)
            r = minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4, "maxiter": 60})
            screen.append((float(z1), float(dz), float(r.fun), float(r.x)))
    screen.sort(key=lambda q: q[2])
    best = (None, np.inf)
    for z1, dz, _, nu in screen[:n_polish]:
        x, c = nm("flipped", d, [h, wb, wbc, nu, z1, z1 + dz])
        x, c = nm("flipped", d, x)
        if c < best[1]:
            best = (x, c)
    return best[0], best[1], screen
