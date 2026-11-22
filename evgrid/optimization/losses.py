"""Nonlinear loss models used by the simulator ("plant"); the LP uses linearised versions ("model").

All energies in MWh per hour (= MW average), temperatures in degC.
"""
from __future__ import annotations

import numpy as np


def charger_loss(p_out: np.ndarray, p_rated: np.ndarray, on_frac: np.ndarray, lc: dict) -> np.ndarray:
    """DC charger losses per hub: n_on*P0 + k1*P + k2*P^2/P_rated (standby + conversion)."""
    p_rated = np.maximum(p_rated, 1e-9)
    return lc["charger_p0_frac"] * on_frac * p_rated + lc["charger_k1"] * p_out + lc["charger_k2"] * p_out ** 2 / p_rated


def charger_eta_linear(lc: dict, load_frac: float = 0.35) -> float:
    """Linearised average charger efficiency for the LP (at a typical load fraction)."""
    loss = lc["charger_k1"] + lc["charger_k2"] * load_frac
    return 1.0 / (1.0 + loss)


def bess_eta(c_rate: float, st: dict) -> tuple:
    """One-way charge / discharge efficiencies falling with C-rate."""
    d = st["bess_eta_c_slope"] * c_rate
    return st["bess_eta_ch"] - d, st["bess_eta_dis"] - d


def bess_aux(cap_mwh: float, temp: float, st: dict) -> float:
    """BESS auxiliaries/HVAC per hour (MWh), rising with ambient temperature."""
    frac = st["bess_aux_frac_day"] + st["bess_aux_temp_slope"] * (temp - 30.0)
    return cap_mwh * max(frac, 0.001) / 24.0


def pack_eta(c_rate: np.ndarray, st: dict) -> np.ndarray:
    """Pack charging efficiency: 0.93 @0.5C -> 0.90 @1.5C (linear)."""
    lo, hi = st["pack_eta_low_c"], st["pack_eta_high_c"]
    return np.clip(lo + (hi - lo) * (np.asarray(c_rate) - 0.5) / 1.0, hi - 0.01, lo + 0.01)


def pack_idle_losses(inv_mwh: np.ndarray, n_packs: float, temp: float, st: dict) -> tuple:
    """(self-discharge from inventory, BMS standby, cabinet cooling) in MWh for one hour."""
    sd = np.asarray(inv_mwh) * st["pack_self_discharge_month"] / 730.0
    bms = n_packs * st["pack_bms_standby_w"] / 1e6
    cool = st["pack_cooling_kw_per_mwh_degc"] * np.sum(inv_mwh) * max(temp - 25.0, 0.0) / 1000.0
    return sd, bms, cool


def transformer_loss(p_load: float, rating: float, gc: dict) -> tuple:
    """(no-load, load) transformer/distribution losses in MWh for one hour."""
    nl = gc["transformer_noload_frac"] * rating
    ll = gc["transformer_load_loss_frac"] * rating * (p_load / max(rating, 1e-9)) ** 2
    return nl, ll
