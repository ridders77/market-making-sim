"""Mid-price process and per-path random numbers.

Model
-----
The mid-price follows arithmetic Brownian motion (ABM)

    dS_t = sigma dW_t,

so over a step of length dt the increment is exactly Normal(0, sigma^2 dt):

    S_{i+1} = S_i + sigma * sqrt(dt) * Z_i,   Z_i ~ N(0, 1) i.i.d.

(For ABM this Euler scheme is exact, not an approximation.) ABM rather than
geometric BM because the A-S formulas are derived under ABM and, over a short
horizon, the difference is negligible. Prices can in principle go negative,
which is irrelevant at these volatilities (S0 = 100, sigma*sqrt(T) = 2).

Seeding and pairing
-------------------
Each simulated path i has its own integer seed. All the randomness a path
needs (price shocks Z, and the uniforms that decide whether the bid and the
ask are hit in each step) is drawn from that seed *before* any strategy runs.
So two strategies run on the same seeds see the same prices and the same
"order flow" uniforms: the comparison is paired (common random numbers), and
any single path can be reproduced on its own from its seed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def draw_shocks(seeds: Iterable[int], n_steps: int) -> dict[str, np.ndarray]:
    """Draw all random numbers for each path, one generator per seed.

    Returns a dict of arrays with shape (n_paths, n_steps):
    ``z`` (standard normals for the price), ``u_bid`` and ``u_ask`` (uniforms on [0, 1)
    that decide whether the bid / ask are hit in each step).
    """
    seeds = list(seeds)
    z = np.empty((len(seeds), n_steps))
    u_bid = np.empty_like(z)
    u_ask = np.empty_like(z)
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        z[i] = rng.standard_normal(n_steps)
        u_bid[i] = rng.random(n_steps)
        u_ask[i] = rng.random(n_steps)
    return {"z": z, "u_bid": u_bid, "u_ask": u_ask}


def mid_price_paths(s0: float, sigma: float, dt: float, z: np.ndarray) -> np.ndarray:
    """Build ABM paths S (shape (n_paths, n_steps + 1), S[:, 0] = s0) from shocks z."""
    increments = sigma * np.sqrt(dt) * z
    S = np.empty((z.shape[0], z.shape[1] + 1))
    S[:, 0] = s0
    S[:, 1:] = s0 + np.cumsum(increments, axis=1)
    return S
