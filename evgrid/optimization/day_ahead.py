"""Day-ahead stochastic scheduling & bidding LP (per cluster, 24 h + 12 h look-ahead, receding).

Units: energy MWh per hour, prices INR/kWh -> objective in INR thousand.
See docs/model_formulation.md for the full algebraic statement.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from evgrid.optimization.lp_builder import LPBuilder, LPResult


@dataclass
class DAInputs:
    dc_dem: np.ndarray        # forecast reference DC demand (MWh/h)
    dc_cap: np.ndarray        # spatially servable DC energy at forecast demand, incl. steering (MWh/h)
    dc_free: np.ndarray       # free hub capacity reachable by shifted fleet demand (MWh/h)
    dc_flex: np.ndarray       # shiftable DC demand (MWh/h)
    dc_price: np.ndarray      # volume-weighted DC tariff (INR/kWh)
    sw_dem: np.ndarray        # swap energy demand (MWh/h)
    sw_price: np.ndarray      # swap revenue (INR/kWh delivered)
    re: np.ndarray            # RE forecast (MWh/h)
    dam_px: np.ndarray        # DAM price scenario (INR/kWh)
    discom_px: np.ndarray     # DISCOM ToD energy charge (INR/kWh)
    sell_px: np.ndarray       # surplus RE value (max of G-DAM net, banking value)
    aux: np.ndarray           # fixed auxiliary load (MWh/h)
    temp: np.ndarray          # forecast temperature (degC)
    grid_mw: float
    bess_mwh: float
    bess_p: float
    soc0: float
    pack_q: float
    pack_rack: float
    inv0: float
    params: Dict[str, float]


def solve_day_ahead(inp: DAInputs) -> LPResult:
    p = inp.params
    T = len(inp.dc_dem)
    lp = LPBuilder()
    eta_c, eta_d, eta_chg, eta_p = p["eta_c"], p["eta_d"], p["eta_chg"], p["eta_p"]
    res = p["reserve"]
    landed = inp.dam_px + p["dam_fee"] + p["oa"]
    voll = p["voll"]
    soc_lo = (p["soc_min"] + res) * inp.bess_mwh
    soc_hi = p["soc_max"] * inp.bess_mwh

    dam = lp.var("dam", T, 0, inp.grid_mw, cost=landed)
    dis = lp.var("dis", T, 0, inp.grid_mw, cost=inp.discom_px)
    re_use = lp.var("re_use", T, 0)
    re_sell = lp.var("re_sell", T, 0, cost=-inp.sell_px)
    re_spill = lp.var("re_spill", T, 0)
    bch = lp.var("bch", T, 0, inp.bess_p)
    bdis = lp.var("bdis", T, 0, inp.bess_p, cost=p["deg_bess"])
    soc = lp.var("soc", T, soc_lo, soc_hi, cost=p["hold_bess"])
    pch = lp.var("pch", T, 0, inp.pack_rack, cost=p["deg_pack"] * eta_p)
    inv = lp.var("inv", T, res * inp.pack_q, inp.pack_q, cost=p["hold_pack"])
    dsrv = lp.var("dsrv", T, 0, inp.dc_cap + inp.dc_free, cost=-inp.dc_price)
    dun = lp.var("dun", T, 0, cost=voll)
    ssrv = lp.var("ssrv", T, 0, cost=-inp.sw_price)
    sun = lp.var("sun", T, 0, cost=voll)
    defer = lp.var("defer", T, 0, inp.dc_flex, cost=p["shift_cost"])
    recov = lp.var("recov", T, 0, inp.dc_free)
    back = lp.var("back", T, 0, p["backlog_max"] * max(inp.dc_flex.mean(), 1e-6))

    # AC energy balance
    lp.eq([(dam, 1), (dis, 1), (re_use, 1), (bdis, eta_d), (bch, -1), (dsrv, -1.0 / eta_chg), (pch, -1)], inp.aux)
    # RE split
    lp.eq([(re_use, 1), (re_sell, 1), (re_spill, 1)], inp.re)
    # sanctioned grid connection
    lp.le([(dam, 1), (dis, 1)], inp.grid_mw)
    # BESS dynamics: soc_t = (1-sd) soc_{t-1} + eta_c bch_t - bdis_t
    sd = p["bess_sd_h"]
    lp.eq([(soc[:1], 1), (bch[:1], -eta_c), (bdis[:1], 1)], (1 - sd) * inp.soc0)
    lp.eq([(soc[1:], 1), (soc[:-1], -(1 - sd)), (bch[1:], -eta_c), (bdis[1:], 1)], 0.0)
    # Pack inventory dynamics (packs swapped out leave the charged inventory)
    sdp = p["pack_sd_h"]
    lp.eq([(inv[:1], 1), (pch[:1], -eta_p), (ssrv[:1], 1)], (1 - sdp) * inp.inv0)
    lp.eq([(inv[1:], 1), (inv[:-1], -(1 - sdp)), (pch[1:], -eta_p), (ssrv[1:], 1)], 0.0)
    # swaps served from start-of-hour inventory above the reserve floor
    lp.le([(ssrv[1:], 1), (inv[:-1], -1)], -res * inp.pack_q)
    lp.le([(ssrv[:1], 1)], inp.inv0 - res * inp.pack_q)
    # demand: swaps
    lp.eq([(ssrv, 1), (sun, 1)], inp.sw_dem)
    # demand: DC with virtual-battery load shifting
    lp.eq([(dsrv, 1), (dun, 1), (defer, 1), (recov, -1)], inp.dc_dem)
    lp.le([(dsrv, 1), (recov, -1)], inp.dc_cap)
    lp.eq([(back[:1], 1), (defer[:1], -1), (recov[:1], 1)], 0.0)
    lp.eq([(back[1:], 1), (back[:-1], -1), (defer[1:], -1), (recov[1:], 1)], 0.0)
    # zero backlog at the end of the operating day (hour 23) and horizon end
    end = [min(23, T - 1), T - 1]
    lp.eq([(back[end], 1)], 0.0)
    # terminal value of stored energy (end of horizon)
    tv = p["terminal_value"]
    lp.add_cost(soc[-1:], -tv)
    lp.add_cost(inv[-1:], -tv * 1.2)
    return lp.solve()
