"""Stochastic DAM bid curves from price-scenario LP solutions + clearing by step lookup."""
from __future__ import annotations

from typing import List

import numpy as np


def build_bid_curve(prices: np.ndarray, quantities: np.ndarray, cap: float = 10.0) -> tuple:
    """Monotone non-increasing stepwise buy curve for one hour.

    prices, quantities: scenario price points (any order) and the LP-optimal purchase at each.
    Returns (price_steps ascending, quantity_steps non-increasing) with prices capped at ``cap``.
    Isotonic (pool-adjacent-violators) regression enforces monotonicity.
    """
    order = np.argsort(prices)
    p = np.minimum(np.asarray(prices, float)[order], cap)
    q = np.asarray(quantities, float)[order]
    q = _pava_decreasing(q)
    return p, np.maximum(q, 0.0)


def _pava_decreasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a non-increasing fit (equal weights)."""
    vals: List[float] = []
    wts: List[int] = []
    for v in y:
        vals.append(float(v))
        wts.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            w = wts[-2] + wts[-1]
            m = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:] = [m]
            wts[-2:] = [w]
    return np.repeat(vals, wts)


def clear_bid(price_steps: np.ndarray, qty_steps: np.ndarray, realised: float) -> float:
    """Cleared quantity at the realised price: buy Q_k at the first step whose price >= realised.

    Above the highest step the last (smallest) quantity is bid at the cap, so it is always cleared.
    """
    idx = np.searchsorted(price_steps, realised, side="left")
    if idx >= len(price_steps):
        return float(qty_steps[-1])
    return float(qty_steps[idx])


def is_monotone_nonincreasing(q: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.all(np.diff(q) <= tol))
