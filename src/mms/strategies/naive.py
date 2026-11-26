"""Naive benchmark: symmetric quotes around the mid with a fixed spread.

    bid = S - spread/2,   ask = S + spread/2

It ignores inventory entirely, so inventory follows a random walk driven by
which side happens to get hit, and the P&L picks up the full price risk of that
inventory. In the A-S paper this is the "symmetric strategy", run with the same
(average) spread as A-S so that only the *centre* of the quotes differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NaiveSymmetric:
    spread: float
    name: str = "symmetric"

    def __post_init__(self) -> None:
        if self.spread < 0:
            raise ValueError("spread must be non-negative")

    def quotes(self, t: float, S: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S = np.asarray(S, dtype=float)
        half = self.spread / 2
        return S - half, S + half
