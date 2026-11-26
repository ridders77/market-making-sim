r"""Avellaneda-Stoikov (2008) optimal quotes, in the paper's approximate closed form.

Setting
-------
Mid S follows dS = sigma dW. We hold q units and cash X; wealth is X + q S.
We maximise expected CARA (exponential) utility of terminal wealth,
E[-exp(-gamma (X_T + q_T S_T))], where gamma > 0 is risk aversion.

1. Reservation (indifference) price
-----------------------------------
Suppose we could not trade at all until T. Holding q units, terminal wealth is
X + q S_T with S_T ~ N(S_t, sigma^2 (T - t)). For CARA utility and normal
wealth, the certainty equivalent is "mean minus gamma/2 times variance":

    CE(q) = X + q S - (gamma/2) q^2 sigma^2 (T - t).

The reservation *bid* is the price r_b at which we are indifferent to buying
one more unit: CE(q+1) with cash X - r_b equals CE(q). Solving,

    r_b = S - (2q + 1)/2 * gamma sigma^2 (T - t),

and symmetrically r_a = S - (2q - 1)/2 * gamma sigma^2 (T - t). Their average is

    r = S - q gamma sigma^2 (T - t).

Intuition: if we are long (q > 0) the inventory is a risk, so we value the
asset *below* the mid and centre our quotes there: we sell more eagerly and buy
less eagerly, which pushes inventory back towards zero. The adjustment grows
with risk aversion gamma, variance sigma^2 and the time left (T - t), and
vanishes at T.

2. Optimal spread
-----------------
With fills arriving at rate A exp(-k delta), the HJB equation for the value
function gives (after the paper's approximation for small q) a total spread

    delta_a + delta_b = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k).

Two pieces:
* gamma sigma^2 (T - t): an inventory-risk premium, larger when holding risk
  for longer is costlier; zero at T.
* (2/gamma) ln(1 + gamma/k): the "monopoly" liquidity term. It trades off a
  wider spread (more profit per fill) against fewer fills (rate falls like
  exp(-k delta)). As gamma -> 0 it tends to 2/k, the spread a risk-neutral
  maker facing exponential intensity would choose (maximise delta e^{-k delta}
  per side: delta = 1/k).

3. Quotes
---------
    bid = r - spread/2,   ask = r + spread/2.

Note the spread does not depend on q: inventory only *shifts* the quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AvellanedaStoikov:
    gamma: float  # risk aversion
    sigma: float  # mid-price volatility (price units per sqrt(time unit))
    k: float      # fill-intensity decay
    T: float      # horizon
    name: str = "avellaneda_stoikov"

    def __post_init__(self) -> None:
        if self.gamma <= 0 or self.k <= 0 or self.sigma < 0 or self.T <= 0:
            raise ValueError("need gamma > 0, k > 0, sigma >= 0, T > 0")

    def reservation_price(self, S, q, t: float):
        """r = S - q * gamma * sigma^2 * (T - t)."""
        return np.asarray(S, dtype=float) - np.asarray(q) * self.gamma * self.sigma**2 * (self.T - t)

    def spread(self, t: float) -> float:
        """Total spread gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k)."""
        inventory_risk = self.gamma * self.sigma**2 * (self.T - t)
        liquidity = (2.0 / self.gamma) * np.log1p(self.gamma / self.k)
        return float(inventory_risk + liquidity)

    def quotes(self, t: float, S: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.reservation_price(S, q, t)
        half = self.spread(t) / 2
        return r - half, r + half
