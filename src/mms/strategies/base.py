"""Common interface for quoting strategies."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Strategy(Protocol):
    """A strategy maps (time, mid, inventory) to (bid, ask), vectorised over paths."""

    name: str

    def quotes(self, t: float, S: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (bid, ask) arrays with the same shape as S."""
        ...
