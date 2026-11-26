r"""Flow-aware Avellaneda-Stoikov: fee-aware A-S that prices adverse selection and leans on the order-flow signal.

Two ingredients, each a small change to the A-S quotes.

1. Reservation price with drift (lean ahead of the flow)
--------------------------------------------------------
A-S's reservation price is the price at which holding one more unit leaves our certainty
equivalent unchanged. With CARA utility and terminal wealth X + q S_T, the certainty equivalent is
"mean minus gamma/2 times variance". If the mid has a predictable drift, the mean of S_T is no
longer S but S + E[S_T - S_t | x_t], so (repeating the stage 1 derivation)

    r = S + E[S_T - S_t | x_t] - q gamma sigma^2 (T - t).

For a constant drift mu this is r = S + mu (T - t) - q gamma sigma^2 (T - t). Our signal does not
last: it is an OU process with time constant tau_x and drift kappa x, so its expected effect over
the remaining time tau = T - t is

    D(x, t) = E[S_T - S_t | x_t] = kappa x int_0^tau e^{-s/tau_x} ds = kappa tau_x (1 - e^{-tau/tau_x}) x
            = mu_hat * tau_eff,   mu_hat = kappa x,  tau_eff = tau_x (1 - e^{-tau/tau_x}) (~ tau_x, a few seconds).

Shifting r by D moves *both* quotes by D. Measured from the mid, the side that the flow is about
to pick off (the ask when buyers are pushing, x > 0) is widened by D and the other side is
tightened by D: we sell less eagerly into a rising market and buy more eagerly. That *is* the
"widen the side about to be picked off" rule in this model; no separate term is needed, and adding
one would count the same expected move twice.

2. Markout allowance (price the expected adverse move after a fill)
--------------------------------------------------------------------
After our quote at distance delta is filled, the mid moves against us by J(delta) on average (the
measured markout curve). A fill is therefore worth delta - c S - J(delta) rather than delta - c S.
For a constant J this is exactly the fee argument again (FeeAwareAvellanedaStoikov) with c S
replaced by c S + J: every half-spread grows by J. The measured J rises with delta (only big
sweeps reach far quotes, and they move the price more), so we take the shift of the risk-neutral
optimum: delta*_rn = argmax_delta e^{-k delta} (delta - c S - J(delta)), and

    allowance = delta*_rn - (1/k + c S)    (= J when J is flat),

and add it to both A-S half-spreads. Adding J at the quoting distance instead (delta = h + J(delta))
over-widens when J rises with delta (see the hand-worked test).

Quotes:  bid = r - (h_AS(t) + c S + allowance),  ask = r + (h_AS(t) + c S + allowance), r with D.
Post-only: a maker order cannot cross the book, so each quote is kept at least half a tick
(min_half) from the mid.

``use_signal=False`` gives the "markout-aware" ablation (same quotes with D = 0): the difference
between the two isolates what reading the flow is worth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from mms.strategies.avellaneda_stoikov import FeeAwareAvellanedaStoikov


def markout_allowance(k: float, fee_per_fill: float, jump_delta, jump_mean, delta_max: float = 50.0,
                      delta_min: float = 0.0) -> float:
    """Shift of the risk-neutral optimal distance caused by the markout curve J (section 2).

    Maximises f(delta) = e^{-k delta} (delta - fee_per_fill - J(delta)) on a 0.001 grid over [delta_min, delta_max]
    (delta_min: the range where lambda and J are calibrated; below it the optimum would be an extrapolation),
    refines around the best grid point, and returns delta* - (1/k + fee_per_fill)."""
    J = lambda d: np.interp(d, jump_delta, jump_mean)
    f = lambda d: np.exp(-k * d) * (d - fee_per_fill - J(d))
    grid = np.arange(delta_min, delta_max, 0.001)
    i = int(np.argmax(f(grid)))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    best = minimize_scalar(lambda d: -f(d), bounds=(lo, hi), method="bounded", options={"xatol": 1e-9})
    d_star = float(best.x) if -best.fun >= f(grid[i]) else float(grid[i])
    if d_star >= delta_max - 0.01:
        raise ValueError("no interior optimum: the markout curve leaves no profitable distance")
    return d_star - (1.0 / k + fee_per_fill)


@dataclass(frozen=True)
class FlowAwareAvellanedaStoikov(FeeAwareAvellanedaStoikov):
    kappa: float = 0.0                 # drift per unit signal (USDT/s), calibrated
    tau_x: float = 1.0                 # signal time constant (s), calibrated
    jump_delta: tuple = (0.0,)         # markout curve J (USDT), calibrated
    jump_mean: tuple = (0.0,)
    s_ref: float = 1.0                 # price level used for the fee term in the allowance
    use_signal: bool = True
    min_half: float = 0.05             # post-only: at least half a tick from the mid
    delta_min: float = 0.1             # the allowance is optimised over delta >= this (calibrated range)
    name: str = "flow_aware_as"

    def __post_init__(self) -> None:
        super().__post_init__()
        a = markout_allowance(self.k, self.fee_bps * 1e-4 * self.s_ref, self.jump_delta, self.jump_mean,
                               delta_min=self.delta_min)
        object.__setattr__(self, "allowance", a)

    def drift_shift(self, x, t: float):
        """D = kappa tau_x (1 - exp(-(T - t)/tau_x)) x: expected move of the mid until T from the current signal."""
        if not self.use_signal:
            return np.zeros_like(np.asarray(x, float))
        return self.kappa * self.tau_x * (1.0 - np.exp(-(self.T - t) / self.tau_x)) * np.asarray(x, float)

    def act(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        S, q, x, t = np.asarray(obs["mid"], float), obs["inventory"], obs["signal"], obs["t"]
        bid, ask = self.quotes(t, S, q)                       # fee-aware A-S
        D = self.drift_shift(x, t)
        bid = np.minimum(bid - self.allowance + D, S - self.min_half)
        ask = np.maximum(ask + self.allowance + D, S + self.min_half)
        return bid, ask
