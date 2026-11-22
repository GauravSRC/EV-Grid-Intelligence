"""LFP degradation: throughput (cycle) cost + temperature/SoC-dependent calendar fade."""
from __future__ import annotations

import numpy as np


def throughput_cost(dg: dict, kind: str) -> float:
    """INR per kWh discharged: replacement x (1 - residual) / (EFC to 80% x 0.8 usable DoD)."""
    rep = dg["pack_replacement_inr_kwh"] if kind == "pack" else dg["bess_replacement_inr_kwh"]
    efc = dg["pack_efc_life"] if kind == "pack" else dg["bess_efc_life"]
    return rep * (1 - dg["residual_frac"]) / (efc * 0.8)


def calendar_rate_h(temp: np.ndarray, soc_frac: np.ndarray, dg: dict) -> np.ndarray:
    """Capacity fraction lost per hour (calendar ageing)."""
    base = dg["calendar_fade_yr"] / 8760.0
    tf = 2.0 ** ((np.asarray(temp) - 25.0) / dg["calendar_temp_doubling_c"])
    sf = 1.0 + (dg["calendar_high_soc_mult"] - 1.0) * np.clip((np.asarray(soc_frac) - 0.5) / 0.5, -1, 1)
    return base * tf * sf


def holding_cost_inr_kwh_h(temp: np.ndarray, dg: dict, kind: str) -> np.ndarray:
    """Linear SoC-holding cost used in the LP: marginal calendar fade of storing 1 kWh for 1 h (INR)."""
    rep = dg["pack_replacement_inr_kwh"] if kind == "pack" else dg["bess_replacement_inr_kwh"]
    base = dg["calendar_fade_yr"] / 8760.0 * 2.0 ** ((np.asarray(temp) - 25.0) / dg["calendar_temp_doubling_c"])
    marginal = base * (dg["calendar_high_soc_mult"] - 1.0) / 0.5  # d(fade)/d(SoC)
    return marginal * rep * (1 - dg["residual_frac"]) / 0.2  # 20% fade = end of life


def soh_update(soh: float, throughput_kwh: float, cap_kwh: float, cal_frac: float, dg: dict, kind: str) -> float:
    efc = dg["pack_efc_life"] if kind == "pack" else dg["bess_efc_life"]
    cyc = throughput_kwh / max(cap_kwh, 1e-9) / efc * 0.2
    return soh - cyc - cal_frac
