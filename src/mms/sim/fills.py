"""Order arrivals: when does a resting quote get filled?

Model (Avellaneda & Stoikov 2008, section 2.3)
----------------------------------------------
Market orders that would hit our quote arrive as a Poisson process whose
intensity depends on how far the quote is from the mid, delta = |quote - S|:

    lambda(delta) = A * exp(-k * delta).

Intuition: A is the arrival rate of orders at the mid itself; every extra unit
of distance cuts the rate by a factor exp(-k). (In the paper this comes from
power-law order sizes plus a log price-impact; here we take it as given and
calibrate A and k to data in stage 2.)

Over a short step dt, the number of arrivals is Poisson(lambda dt), so

    P(at least one arrival) = 1 - exp(-lambda dt)  ~  lambda dt  (for small lambda dt).

Two rules, at most one unit filled per side per step:

* ``rule="linear"`` (stage 1, the paper's rule): p = lambda dt. Only a
  probability if lambda dt < 1, so ``check_dt`` asserts A dt < 1.
* ``rule="exact"`` (stage 2 onwards): p = 1 - exp(-lambda dt), always in [0, 1).
  "At most one fill per step" still undercounts when lambda dt is not small,
  so ``check_dt`` asserts A dt < 0.1 (then P(2+ arrivals) / P(1+) < 5%).

A quote *through* the mid (delta < 0, e.g. an A-S ask below the mid when very
long) gives lambda * dt > 1. Then the quote fills for certain; we clip the
probability to 1 and *count* how often this happens so it is reported,
not hidden (see DECISIONS.md).
"""

from __future__ import annotations

import numpy as np


FILL_RULES = ("linear", "exact")
MAX_RATE_DT = {"linear": 1.0, "exact": 0.1}


def fill_probability(delta: np.ndarray | float, A: float, k: float, dt: float,
                     rule: str = "linear") -> np.ndarray:
    """P(fill in one step) for a quote at distance delta.

    linear: lambda(delta) * dt = A exp(-k delta) dt (unclipped, may exceed 1 through the mid);
    exact:  1 - exp(-lambda(delta) dt).
    """
    lam_dt = A * np.exp(-k * np.asarray(delta, dtype=float)) * dt
    if rule == "linear":
        return lam_dt
    if rule == "exact":
        return -np.expm1(-lam_dt)
    raise ValueError(f"unknown fill rule {rule!r}; use one of {FILL_RULES}")


def check_dt(A: float, dt: float, rule: str = "linear") -> None:
    """Raise if A*dt (the rate at the mid times the step) is too large for the chosen rule."""
    limit = MAX_RATE_DT[rule]
    if A * dt >= limit:
        raise ValueError(f"A*dt = {A * dt:.3f} >= {limit} ({rule} rule): dt too large")


def bernoulli_fills(p: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, int]:
    """Decide fills: a quote fills iff u < p, with p clipped to [0, 1].

    Returns (filled boolean array, number of probabilities that exceeded 1).
    """
    p = np.asarray(p, dtype=float)
    n_clipped = int(np.count_nonzero(p > 1.0))
    filled = u < np.clip(p, 0.0, 1.0)
    return filled, n_clipped
