"""Dynamic pricing engine with hard regulatory/commercial guardrails.

p_t = p_ref,d * (1 + m_t),  |m_t| <= 0.20 (fleet segments <= 0.10),
p_ref,d = previous day's realised average tariff; daily-average change <= +-5%.
m_t = m_shift + m_cost + m_scarcity - m_green (+ a small anchor back to list price).
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def compose_markup(defer: np.ndarray, recov: np.ndarray, flex: np.ndarray, backlog: np.ndarray, px_fc: np.ndarray,
                   scarcity: np.ndarray, green: np.ndarray, pc: dict) -> np.ndarray:
    """Raw (unclipped) markup from the LP plan and forecasts for one day (24 h)."""
    band = pc["band"]
    m_shift = band * np.where(defer > 1e-6, defer / np.maximum(flex, 1e-6), 0.0)
    prev_back = np.concatenate([[0.0], backlog[:-1]])
    m_shift -= band * np.where(recov > 1e-6, np.clip(recov / np.maximum(prev_back, 1e-6), 0, 1), 0.0)
    m_cost = pc["beta"] * np.clip((px_fc - px_fc.mean()) / max(px_fc.mean(), 1e-6), -0.4, 0.4)
    m_scar = pc["scarcity"] * scarcity.astype(float)
    m_green = pc["green"] * green.astype(float)
    m = m_shift + m_cost + m_scar - m_green
    return m - m.mean()  # revenue-neutral shape: pricing shifts load, it does not raise the tariff level


def apply_guardrails(m_raw: np.ndarray, p_ref: float, p_list: float, band: float, guard: float) -> np.ndarray:
    """Clip to the band, anchor towards list price, enforce the daily-average guardrail. Returns prices."""
    anchor = float(np.clip(-(p_ref / p_list - 1.0), -guard, guard))
    m = np.clip(m_raw + anchor, -band, band)
    mean_m = m.mean()
    if abs(mean_m) > guard:  # daily-average change limit
        m = m - (mean_m - np.sign(mean_m) * guard)
        m = np.clip(m, -band, band)
    return p_ref * (1.0 + m)


def check_price_band(prices: np.ndarray, p_ref: float, band: float, tol: float = 1e-9) -> bool:
    return bool(np.all(np.abs(prices / p_ref - 1.0) <= band + tol))


def segment_prices(m_raw: np.ndarray, p_ref: Dict[str, float], p_list: Dict[str, float], cfg_c: dict,
                   fleet_segments=("ride", "logi")) -> Dict[str, np.ndarray]:
    out = {}
    for s in p_ref:
        band = cfg_c["fleet_band"] if s in fleet_segments else cfg_c["price_band"]
        mr = m_raw * (0.5 if s in fleet_segments else 1.0)
        out[s] = apply_guardrails(mr, p_ref[s], p_list[s], band, cfg_c["daily_guardrail"])
    return out
