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

Over a short step dt, P(at least one arrival) = 1 - exp(-lambda dt) ~ lambda dt.
Following the paper we use lambda * dt directly and fill at most one unit per
side per step. That only makes sense if lambda * dt is a probability, i.e.
well below 1, so ``check_dt`` asserts A * dt < 1 (the largest value for any
quote at or outside the mid).

A quote *through* the mid (delta < 0, e.g. an A-S ask below the mid when very
long) gives lambda * dt > 1. Then the quote fills for certain; we clip the
probability to 1 and *count* how often this happens so it is reported,
not hidden (see DECISIONS.md).
"""

from __future__ import annotations

import numpy as np


def fill_probability(delta: np.ndarray | float, A: float, k: float, dt: float) -> np.ndarray:
    """P(fill in one step) = lambda(delta) * dt = A * exp(-k * delta) * dt (unclipped)."""
    return A * np.exp(-k * np.asarray(delta, dtype=float)) * dt


def check_dt(A: float, dt: float) -> None:
    """Raise if a quote at the mid would have fill 'probability' A*dt >= 1."""
    if A * dt >= 1.0:
        raise ValueError(f"A*dt = {A * dt:.3f} >= 1: dt too large for the Bernoulli fill approximation")


def bernoulli_fills(p: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, int]:
    """Decide fills: a quote fills iff u < p, with p clipped to [0, 1].

    Returns (filled boolean array, number of probabilities that exceeded 1).
    """
    p = np.asarray(p, dtype=float)
    n_clipped = int(np.count_nonzero(p > 1.0))
    filled = u < np.clip(p, 0.0, 1.0)
    return filled, n_clipped
