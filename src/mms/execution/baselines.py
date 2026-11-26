"""Baseline execution strategies, run through the same environment and episodes as the learned policies.

* immediate taker:   cross at the first decision (``FixedAction(CROSS)``)
* TWAP taker slices: an equal share of the lots as a taker order at every decision (``TWAP``)
* rest at touch:     the touch action at every decision, clean-up at the deadline (``FixedAction(TOUCH)``)
* best fixed action: whichever of the four fixed actions has the lowest mean cost on the training days
"""

from __future__ import annotations

from mms.execution.env import EPS


class FixedAction:
    def __init__(self, action: int):
        self.action = int(action)

    def __call__(self, obs: dict) -> int:
        return self.action


class TWAP:
    """Taker slices: after decision t (0-based) round(L (t + 1) / N) of the L lots have been sent in total."""

    def __init__(self, qty: float, n_steps: int, lot: float = 0.01):
        self.lots = int(round(qty / lot))
        self.n, self.lot = n_steps, lot

    def slice_lots(self, step: int) -> int:
        return round(self.lots * (step + 1) / self.n) - round(self.lots * step / self.n)

    def act(self, obs: dict) -> tuple[None, float, bool]:
        x = self.slice_lots(obs["step"]) * self.lot
        return None, min(x, obs["remaining"]) if x > EPS else 0.0, False
