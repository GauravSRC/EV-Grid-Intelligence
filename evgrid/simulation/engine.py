"""Rolling digital-twin simulator (one cluster at a time; clusters decouple given the monthly pack plan).

Day loop:  D-1 forecasts -> day-ahead LP (AI) / fixed rules (baseline) -> bid curves & DAM clearing ->
prices -> hourly real-time dispatch with ACTUAL demand/RE/prices -> nonlinear physics -> records.
Baseline and AI share identical data, random draws (common random numbers) and physics; only the
decisions differ.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from evgrid.data.synth import archetype_profile, cluster_capacities
from evgrid.forecasting.models import price_quantile
from evgrid.optimization import losses as L
from evgrid.optimization.bidding import build_bid_curve, clear_bid
from evgrid.optimization.day_ahead import DAInputs, solve_day_ahead
from evgrid.optimization.degradation import calendar_rate_h, holding_cost_inr_kwh_h, throughput_cost
from evgrid.optimization.pricing import compose_markup, segment_prices
from evgrid.simulation.policies import Policy

log = logging.getLogger(__name__)
SEGS = ["retail", "ride", "logi"]
SWC = ["s2w", "s3w"]
REC_FIELDS = [
    "ref_retail", "ref_ride", "ref_logi", "dem_retail", "dem_ride", "dem_logi", "del_retail", "del_ride", "del_logi",
    "fc_dc", "fc_sw", "ref_sw", "dem_sw", "del_sw", "ref_sw_n", "dem_sw_n", "del_sw_n", "unserved_dc", "lost_dc",
    "re_total", "re_used", "re_sold", "re_bank", "re_spill", "dam_bought", "dam_resold", "rtm", "discom",
    "bess_ch_ac", "bess_dis_ac", "soc", "inv", "d_soc", "d_inv", "pack_ac", "pack_stored",
    "loss_charger", "loss_charger_standby", "loss_bess", "loss_pack_chg", "loss_pack_idle", "loss_dist",
    "revenue", "rev_dc", "rev_sw", "cost_energy", "cost_dam", "cost_rtm", "cost_discom", "cost_re", "credit_sales",
    "opex_var", "deg_cost", "grid_import", "grid_limit", "price_dev_max", "fleet_dev_max", "guard_dev",
    "inv_reserve_slack", "soc_reserve_slack", "dc_commit_frac", "priority_dem", "priority_srv", "swaps_n_served",
    "bess_throughput", "pack_throughput", "cal_fade_bess", "cal_fade_pack", "balance_err", "temp", "dam_price",
]


def hub_share_matrix(hubs: pd.DataFrame, times: pd.DatetimeIndex, kind: str) -> np.ndarray:
    """Expected hub demand shares (T x H) = weight x archetype profile, normalised per hour."""
    hrs = times.hour.values
    wk = times.dayofweek.values >= 5
    w = hubs["dc_demand_weight" if kind == "dc" else "swap_demand_weight"].values
    prof = np.column_stack([archetype_profile(a, hrs, wk) for a in hubs["archetype"].values])
    m = prof * w[None, :]
    return m / m.sum(axis=1, keepdims=True)


class ClusterSim:
    """Holds static data for one cluster and runs a policy over a list of day segments."""

    def __init__(self, cfg: dict, cl: str, hourly: pd.DataFrame, fc: pd.DataFrame, hubs: pd.DataFrame,
                 re_mw: pd.Series, pack_plan: Optional[Dict[str, float]] = None, scenario: Optional[dict] = None):
        self.cfg, self.cl = cfg, cl
        sc = scenario or {}
        self.sc = sc
        caps = cluster_capacities(cfg).loc[cl]
        self.dc_mw, self.swap_mw = float(caps["dc_mw"]), float(caps["swap_mw"])
        self.bess_mwh = float(caps["bess_mwh"]) * sc.get("bess_mult", 1.0)
        self.pack_q0 = float(caps["pack_mwh"])
        self.grid_mw = float(caps["grid_mw"]) * (sc.get("grid_mult", {}).get(cl, 1.0) if isinstance(sc.get("grid_mult"), dict) else 1.0)
        self.grid_derate = sc.get("grid_derate", {})
        if sc.get("grid_derate_cluster") == cl:
            self.grid_mw *= 1 - sc.get("grid_derate_frac", 0.2)
        self.pack_q0 *= sc.get("pack_mult", 1.0)
        self.total_mw = float(caps["total_mw"])
        rm = sc.get("re_mult", 1.0)
        rm = rm.get(cl, 1.0) if isinstance(rm, dict) else rm
        self.solar_mw, self.wind_mw = float(re_mw["solar_mw"]) * rm, float(re_mw["wind_mw"]) * rm
        h = hourly[hourly["cluster"] == cl].sort_values("time").reset_index(drop=True)
        f = fc[fc["cluster"] == cl].sort_values("time").reset_index(drop=True)
        h = h[h["time"] >= f["time"].min()].reset_index(drop=True)
        assert len(h) == len(f), (len(h), len(f))
        self.times = pd.DatetimeIndex(h["time"])
        ds = cfg["demand"]["demand_scale"] * sc.get("demand_mult", 1.0)
        self.a: Dict[str, np.ndarray] = {}
        a = self.a
        for s in SEGS:
            a[f"ref_{s}"] = h[f"dc_{s}_kwh"].values / 1000.0 * ds
            a[f"fc_{s}"] = f[f"{s}_p50"].values / 1000.0 * ds
            a[f"fc90_{s}"] = f[f"{s}_p90"].values / 1000.0 * ds
            a[f"nv_{s}"] = f[f"{s}_naive"].values / 1000.0 * ds
        for s in SWC:
            a[f"ref_{s}"] = h[f"swaps_{s}"].values * ds
            a[f"fc_{s}"] = f[f"{s}_p50"].values * ds
            a[f"fc90_{s}"] = f[f"{s}_p90"].values * ds
            a[f"nv_{s}"] = f[f"{s}_naive"].values * ds
        for k in ["solar_cf", "wind_cf", "dam_price", "rtm_price", "gdam_price", "temp", "dc_avail"]:
            a[k] = h[k].values.astype(float).copy()
        for s in ["solar", "wind", "dam", "spread"]:
            for q in ["p10", "p50", "p90", "naive"]:
                a[f"{s}_{q}"] = f[f"{s}_{q}"].values.astype(float).copy()
        a["temp_fc"] = h["temp_fc"].values.astype(float)
        self._apply_scenario(sc)
        hb = hubs[hubs["cluster"] == cl].reset_index(drop=True)
        self.H = len(hb)
        g = cfg["network"].get("hub_misalloc_gamma", 1.0)  # >1 concentrates assets -> more misallocation
        k = hb["dc_cap_share"].values ** g
        self.kappa = k / k.sum()
        k = hb["pack_share"].values ** g
        self.kpack0 = k / k.sum()
        self.share_dc = hub_share_matrix(hb, self.times, "dc")
        self.share_sw = hub_share_matrix(hb, self.times, "swap")
        self.w_swap = hb["swap_demand_weight"].values
        # common random numbers (identical across policies)
        rng = np.random.default_rng(cfg["meta"]["seed"] + 7 * (cfg["clusters"].index(cl) + 1))
        T = len(self.times)
        self.noise_dc = rng.lognormal(0, 0.18, (T, self.H))
        self.noise_sw = rng.lognormal(0, 0.18, (T, self.H))
        sd = cfg["customers"]["response_noise_sd"]
        self.noise_resp = np.clip(rng.normal(1, sd, (T, 4)), 0, 2)
        self.pack_plan = pack_plan or {}
        self.day_index = pd.factorize(self.times.normalize())[0]

    # ------------------------------------------------------------------
    def _apply_scenario(self, sc: dict) -> None:
        a = self.a
        n = len(self.times)
        win = np.ones(n, bool)
        if "window" in sc:
            w0, w1 = pd.Timestamp(sc["window"][0]), pd.Timestamp(sc["window"][1])
            win = (self.times >= w0) & (self.times < w1)
        if sc.get("temp_add"):
            a["temp"][win] += sc["temp_add"]
            a["temp_fc"][win] += sc["temp_add"] * 0.8
        if sc.get("demand_add"):
            for s in SEGS + SWC:
                a[f"ref_{s}"][win] *= 1 + sc["demand_add"]
                a[f"fc_{s}"][win] *= 1 + sc["demand_add"] * 0.5
                a[f"fc90_{s}"][win] *= 1 + sc["demand_add"] * 0.5
        if sc.get("solar_mult") is not None:
            a["solar_cf"][win] *= sc["solar_mult"]
            a["solar_p50"][win] *= 1 - (1 - sc["solar_mult"]) * 0.5
        if sc.get("wind_mult") is not None:
            a["wind_cf"][win] *= sc["wind_mult"]
            a["wind_p50"][win] *= 1 - (1 - sc["wind_mult"]) * 0.5
        if sc.get("price_mult"):
            for k in ["dam_price", "rtm_price", "gdam_price"]:
                a[k] = np.where(win, np.minimum(a[k] * sc["price_mult"], 10.0), a[k])
        if sc.get("cap_hours_mult"):
            hr = self.times.hour.values
            eve = win & ((hr >= 18) | (hr <= 1))
            hot = eve & (a["dam_price"] >= np.quantile(a["dam_price"][eve], 1 - min(0.05 * sc["cap_hours_mult"], 0.6)))
            a["dam_price"][hot] = 10.0
            a["rtm_price"][hot] = 10.0
        if sc.get("fc_error_mult"):
            k = sc["fc_error_mult"]
            for s in SEGS + SWC:
                a[f"fc_{s}"] = np.clip(a[f"ref_{s}"] + k * (a[f"fc_{s}"] - a[f"ref_{s}"]), 0, None)
                a[f"fc90_{s}"] = a[f"fc_{s}"] + k * np.abs(a[f"fc90_{s}"] - a[f"fc_{s}"])
            for s, act in [("solar", "solar_cf"), ("wind", "wind_cf"), ("dam", "dam_price")]:
                a[f"{s}_p50"] = a[act] + k * (a[f"{s}_p50"] - a[act])
                a[f"{s}_p10"] = a[f"{s}_p50"] - k * np.abs(a[f"{s}_p50"] - a[f"{s}_p10"])
                a[f"{s}_p90"] = a[f"{s}_p50"] + k * np.abs(a[f"{s}_p90"] - a[f"{s}_p50"])
        if sc.get("price_paths_seed") is not None:  # Monte Carlo price path (daily level shocks)
            rng = np.random.default_rng(sc["price_paths_seed"])
            nd = int(np.ceil(n / 24)) + 1
            z = np.zeros(nd)
            e = rng.normal(0, 0.18, nd)
            for i in range(1, nd):
                z[i] = 0.85 * z[i - 1] + e[i]
            mult = np.exp(z[np.arange(n) // 24] + rng.normal(0, 0.08, n))
            for k in ["dam_price", "rtm_price", "gdam_price"]:
                a[k] = np.clip(a[k] * mult, 0.5, 10.0)
        if sc.get("tod_mult"):
            self.cfg = dict(self.cfg)
            self.cfg["grid"] = dict(self.cfg["grid"], tod_mult=sc["tod_mult"])

    # ------------------------------------------------------------------
    def tod(self, hours: np.ndarray) -> np.ndarray:
        tm = self.cfg["grid"]["tod_mult"]
        base = self.cfg["grid"]["discom_energy"][self.cl]
        m = np.where((hours >= 9) & (hours < 16), tm["solar"], np.where((hours >= 18) & (hours < 23), tm["peak"], tm["normal"]))
        return base * m

    def run(self, policy: Policy, days: List[pd.Timestamp], init_state: Optional[dict] = None) -> pd.DataFrame:
        """Simulate the given days (contiguous segments are handled by passing consecutive days)."""
        cfg, a = self.cfg, self.a
        cu, st, gc, dg, oc = cfg["customers"], cfg["storage"], cfg["grid"], cfg["degradation"], cfg["optimization"]
        res = cfg["reserve"]["frac"]
        prio = cfg["reserve"]["priority_demand_frac"]
        p_list = {**{s: cu["dc_price"][s] for s in SEGS}, "swap": 1.0}
        p_ref = dict(p_list)
        eta_b_c, eta_b_d = L.bess_eta(st["bess_c_rate"], st)
        Pb = self.bess_mwh * st["bess_c_rate"]
        soc_lo = (st["bess_soc_min"] + res) * self.bess_mwh
        soc_hi = st["bess_soc_max"] * self.bess_mwh
        soc_lo_d = soc_lo * 1.003 + 1e-6  # dispatch floor incl. self-discharge buffer (reserve never breached passively)
        soc = 0.5 * self.bess_mwh
        sd_h = st["bess_self_discharge_day"] / 24
        acc = {s: (cu["steering_ai"][s] if policy.steering else cu["steering_baseline"]) for s in SEGS + ["swap"]}
        leak = cu["steering_leak"]
        el = cu["elasticity"]
        ss, win_h = cu["shift_share"], cu["shift_window_h"]
        kwh = cu["swap_kwh"]
        eta_chg_lin = L.charger_eta_linear(cfg["losses"])
        deg_b, deg_p = throughput_cost(dg, "bess"), throughput_cost(dg, "pack")
        tr_rating = gc["transformer_rating_frac"] * self.total_mw
        n_packs_per_mwh = 1000.0 / st["pack_avg_kwh"]
        landed_add = gc["dam_fee"] + gc["oa_charges"]
        bank_val = (1 - gc["bank_charge"]) * gc["discom_energy"][self.cl] * gc["bank_value_frac_of_discount_discom"]
        # pack fleet in this cluster (monthly redeployment plan if enabled)
        kpack = self.kpack0.copy()
        Q = self.pack_q0
        inv_h = 0.9 * Q * kpack
        wait_dc = 0.0
        backlog = np.zeros(3)
        cur_month = None
        recs: Dict[str, List[float]] = {k: [] for k in REC_FIELDS}
        rec_time: List[pd.Timestamp] = []
        bank_month = 0.0
        discom_month = 0.0
        prev_day = None
        for day in days:
            idx0 = int(np.searchsorted(self.times.values, np.datetime64(day)))
            if idx0 + 24 > len(self.times):
                break
            if prev_day is None or (day - prev_day).days != 1:  # new segment: reset state
                soc, wait_dc, backlog = 0.5 * self.bess_mwh, 0.0, np.zeros(3)
                p_ref = dict(p_list)
            prev_day = day
            month = day.strftime("%Y-%m")
            if month != cur_month:
                cur_month = month
                bank_month = discom_month = 0.0
                if policy.redeploy and month in self.pack_plan:
                    newQ = self.pack_plan[month]
                    frac_inv = inv_h.sum() / max(Q, 1e-9)
                    Q = newQ
                    kpack = 0.5 * self.kpack0 + 0.5 * self.w_swap
                    inv_h = frac_inv * Q * kpack
                elif not policy.redeploy and Q != self.pack_q0:
                    Q, kpack = self.pack_q0, self.kpack0.copy()
            Qh = Q * kpack
            rack_h = np.minimum(st["pack_max_c"] * Qh, self.swap_mw * self.kpack0)  # racks are fixed; only packs move
            sl = slice(idx0, min(idx0 + oc["horizon_h"], len(self.times)))
            hrs = self.times[sl].hour.values
            Th = len(hrs)
            tod_px = self.tod(hrs)
            grid_lim = self.grid_mw * self.grid_derate.get(day.strftime("%Y-%m-%d"), 1.0)
            # ---------------- forecasts available at D-1 10:00 -------------------------
            src = "fc" if policy.ml_forecast else "nv"
            dfc = {s: a[f"{src}_{s}"][sl] for s in SEGS}
            swfc_n = {s: a[f"{src}_{s}"][sl] for s in SWC}
            dc90 = sum(a[f"fc90_{s}"][sl] if policy.ml_forecast else a[f"nv_{s}"][sl] * 1.25 for s in SEGS)
            if policy.ml_forecast:
                sol_fc, wnd_fc = a["solar_p50"][sl], a["wind_p50"][sl]
                px10, px50, px90 = a["dam_p10"][sl], a["dam_p50"][sl], a["dam_p90"][sl]
            else:
                sol_fc, wnd_fc = a["solar_naive"][sl], a["wind_naive"][sl]
                px50 = a["dam_naive"][sl]
                px10, px90 = px50 * 0.7, np.minimum(px50 * 1.35, 10.0)
            re_fc = self.solar_mw * np.clip(sol_fc, 0, 1) + self.wind_mw * np.clip(wnd_fc, 0, 1)
            dc_fc = sum(dfc.values())
            sw_fc = sum(swfc_n[s] * kwh[s] for s in SWC) / 1000.0
            # spatial servability at forecast demand (hub forecasts)
            sh = self.share_dc[sl]
            cap_h = 0.95 * self.dc_mw * self.kappa[None, :] * np.mean(a["dc_avail"][max(0, idx0 - 168):idx0 or 1])
            Dh = dc_fc[:, None] * sh
            S0 = np.minimum(Dh, cap_h).sum(1)
            E = np.clip(Dh - cap_h, 0, None).sum(1)
            F = np.clip(cap_h - Dh, 0, None).sum(1)
            rho_fc = sum(acc[s] * dfc[s] for s in SEGS) / np.maximum(dc_fc, 1e-9)
            rec_fc = np.minimum(rho_fc * E * (1 - leak), F)
            dc_cap = S0 + rec_fc
            dc_free = (F - rec_fc) * 0.5
            flex = sum(ss[s] * dfc[s] for s in SEGS) if policy.pricing else np.zeros(Th)
            p_dc = sum(dfc[s] * p_ref[s] for s in SEGS) / np.maximum(dc_fc, 1e-9)
            sw_px = (sum(swfc_n[s] * cu["swap_price"][s] for s in SWC) / np.maximum(sum(swfc_n[s] * kwh[s] for s in SWC), 1e-9)) * p_ref["swap"]
            temp_fc = a["temp_fc"][sl]
            # module staging plan (loss-aware ops)
            on_plan = np.clip(cfg["losses"]["ai_module_margin"] * (dc90[:, None] * sh) / np.maximum(cap_h / 0.95, 1e-9),
                              cfg["losses"]["ai_min_on_frac"], 1.0) if policy.loss_aware else np.ones((Th, self.H))
            aux_fc = (L.bess_aux(self.bess_mwh, float(temp_fc.mean()), st) + Q * n_packs_per_mwh * st["pack_bms_standby_w"] / 1e6
                      + gc["transformer_noload_frac"] * tr_rating
                      + cfg["losses"]["charger_p0_frac"] * (on_plan * (cap_h / 0.95)).sum(1))
            gdam_fc = px50 * cfg["prices"]["gdam_ratio_mean"]
            sell_px = np.maximum(gdam_fc - gc["gdam_fee"], bank_val) if policy.loss_aware else np.zeros(Th)
            inv_now = float(inv_h.sum())
            plan = None
            dam_q = np.zeros(24)
            if policy.use_lp:
                params = dict(eta_c=eta_b_c, eta_d=eta_b_d, eta_chg=eta_chg_lin,
                              eta_p=st["pack_eta_low_c"] if policy.storage_opt else st["pack_eta_high_c"],
                              reserve=res, dam_fee=gc["dam_fee"], oa=gc["oa_charges"], voll=oc["voll"],
                              soc_min=st["bess_soc_min"], soc_max=st["bess_soc_max"],
                              deg_bess=deg_b, deg_pack=deg_p,
                              hold_bess=holding_cost_inr_kwh_h(temp_fc, dg, "bess") if policy.storage_opt else 0.0,
                              hold_pack=holding_cost_inr_kwh_h(temp_fc, dg, "pack") if policy.storage_opt else 0.0,
                              bess_sd_h=st["bess_self_discharge_day"] / 24, pack_sd_h=st["pack_self_discharge_month"] / 730,
                              shift_cost=oc["shift_incentive"], backlog_max=4.0,
                              terminal_value=oc["terminal_value_frac"] * float(np.mean(np.minimum(px50 + landed_add, tod_px))))
                bess_cap_lp = self.bess_mwh if policy.storage_opt else 1e-3
                base_inp = dict(dc_dem=dc_fc, dc_cap=dc_cap, dc_free=dc_free if policy.pricing else np.zeros(Th), dc_flex=flex,
                                dc_price=p_dc, sw_dem=sw_fc, sw_price=sw_px, re=re_fc, discom_px=tod_px, sell_px=sell_px,
                                aux=aux_fc, temp=temp_fc, grid_mw=grid_lim, bess_mwh=bess_cap_lp,
                                bess_p=bess_cap_lp * st["bess_c_rate"], soc0=min(max(soc, soc_lo), soc_hi) if policy.storage_opt else 0.0,
                                pack_q=Q, pack_rack=float(rack_h.sum()) if not policy.storage_opt else float(np.minimum(rack_h, st["pack_max_c"] * Qh).sum()),
                                inv0=inv_now, params=params)
                plan = solve_day_ahead(DAInputs(dam_px=px50, **base_inp))
                if plan.status != 0:
                    log.warning("LP status %s on %s %s: %s", plan.status, self.cl, day.date(), plan.message)
                if policy.stochastic_bid and plan.status == 0:
                    qs, pxs = [], []
                    for q in oc["price_scenarios"]:
                        pq = np.clip(price_quantile(px10, px50, px90, q), 0.5, 10.0)
                        r = plan if q == 0.5 else solve_day_ahead(DAInputs(dam_px=pq, **base_inp))
                        qs.append(r["dam"][:24] if r.status == 0 else plan["dam"][:24])
                        pxs.append(pq[:24])
                    qs, pxs = np.array(qs), np.array(pxs)
                    for t in range(24):
                        ps, qq = build_bid_curve(pxs[:, t], qs[:, t], gc["iex_cap"])
                        dam_q[t] = clear_bid(ps, qq, a["dam_price"][idx0 + t])
                elif plan.status == 0:
                    dam_q[:] = plan["dam"][:24]
            else:
                # status-quo rule: buy forecast net need on DAM (at cap) when DAM landed < DISCOM ToD
                need = (np.minimum(dc_fc, dc_cap)[:24] / eta_chg_lin + sw_fc[:24] / st["pack_eta_high_c"] + aux_fc[:24]
                        - re_fc[:24])
                hr24 = hrs[:24]
                need = need + np.where(hr24 < 6, Pb, 0.0) - np.where((hr24 >= 18) & (hr24 < 22), Pb * eta_b_d, 0.0)
                cheaper = (px50[:24] + landed_add) < tod_px[:24]
                dam_q = np.clip(np.where(cheaper, need, 0.0), 0, grid_lim)
            # ---------------- pricing for day D -----------------------------------------
            hr24 = hrs[:24]
            if policy.pricing:
                if plan is not None and plan.status == 0:
                    dfr, rcv, bk = plan["defer"][:24], plan["recov"][:24], plan["back"][:24]
                else:
                    dfr = rcv = bk = np.zeros(24)
                scar = dc90[:24] > 0.95 * dc_cap[:24] / 0.95
                green = (re_fc[:24] > (dc_fc[:24] + sw_fc[:24])) | ((hr24 >= 10) & (hr24 < 16))
                m_raw = compose_markup(dfr, rcv, np.maximum(flex[:24], 1e-9), bk, px50[:24],
                                       scar, green, dict(band=cu["price_band"], beta=oc["pricing_beta"],
                                                         scarcity=oc["scarcity_markup"], green=oc["green_discount"]))
                prices = segment_prices(m_raw, {s: p_ref[s] for s in SEGS}, {s: p_list[s] for s in SEGS}, cu)
                m_sw = oc["scarcity_markup"] * scar - 0.5 * oc["green_discount"] * green
                m_sw = np.clip(m_sw - m_sw.mean(), -0.2, 0.2)  # revenue-neutral shape (no level ratchet)
                prices["swap"] = segment_prices(m_sw, {"swap": p_ref["swap"]}, {"swap": 1.0}, cu, fleet_segments=())["swap"]
            else:
                bt = cu["baseline_tod"]
                mt = np.where((hr24 >= 9) & (hr24 < 16), bt["solar"], np.where((hr24 >= 18) & (hr24 < 23), bt["peak"], bt["normal"]))
                prices = {s: p_list[s] * (1 + mt) for s in SEGS}
                prices["swap"] = np.ones(24) * p_list["swap"]
            level = {s: prices[s].mean() / p_list[s] - 1 for s in SEGS + ["swap"]}
            # ---------------- hourly real-time operation --------------------------------
            day_rev_vol = {s: [0.0, 0.0] for s in SEGS + ["swap"]}
            soc_plan = plan["soc"][:24] if (plan is not None and plan.status == 0 and policy.storage_opt) else None
            inv_plan = plan["inv"][:24] if (plan is not None and plan.status == 0 and policy.storage_opt) else None
            for t in range(24):
                i = idx0 + t
                hr = int(hr24[t])
                temp = a["temp"][i]
                nr = self.noise_resp[i]
                # --- demand response (own-price level + intra-day shifting) ---
                dev = {s: (prices[s][t] - prices[s].mean()) / p_list[s] for s in SEGS}
                dem = np.zeros(3)
                for j, s in enumerate(SEGS):
                    d = a[f"ref_{s}"][i] * max(1 + el[s] * level[s] * nr[j], 0.0)
                    defer = ss[s] * d * np.clip(dev[s] / cu["price_band"], 0, 1) * nr[j]
                    defer = min(defer, d)
                    rel = backlog[j] * np.clip(-dev[s] / (0.5 * cu["price_band"]), 0, 1)
                    rel = max(rel, backlog[j] / (win_h[s] + 1))
                    if t == 23:
                        rel = backlog[j] + defer
                    rel = min(rel, backlog[j] + defer)
                    backlog[j] += defer - rel
                    dem[j] = d - defer + rel
                dem_tot = dem.sum() + wait_dc
                seg_share = (dem + wait_dc * dem / max(dem.sum(), 1e-9)) / max(dem_tot, 1e-9)
                # --- hub-level DC service with steering (closed form) ---
                shr = self.share_dc[i] * self.noise_dc[i]
                shr = shr / shr.sum()
                Dh = dem_tot * shr
                caph = self.dc_mw * self.kappa * a["dc_avail"][i]
                pr_h = prio * Dh
                pr_srv_h = np.minimum(pr_h, res * caph)
                nd_h = Dh - pr_h
                ncap = (1 - res) * caph
                s0_h = np.minimum(nd_h, ncap)
                exc = np.clip(nd_h - ncap, 0, None)
                free = np.clip(ncap - nd_h, 0, None)
                rho = sum(acc[s] * dem[j] for j, s in enumerate(SEGS)) / max(dem.sum(), 1e-9)
                recv = min(rho * exc.sum() * (1 - leak), free.sum())
                srv_h = s0_h + pr_srv_h + (free / max(free.sum(), 1e-9)) * recv
                dc_srv = float(srv_h.sum())
                # --- swaps: vans, hub service with steering ---
                sw_n = np.array([a[f"ref_{s}"][i] * max(1 + el["swap"] * level["swap"] * nr[3], 0) for s in SWC])
                sw_e = float((sw_n * np.array([kwh[s] for s in SWC])).sum()) / 1000.0
                ssh = self.share_sw[i] * self.noise_sw[i]
                ssh = ssh / ssh.sum()
                SWh = sw_e * ssh
                if policy.redeploy:  # intra-city van rebalancing toward forecast need
                    look = slice(i, min(i + 3, len(self.times)))
                    need_h = (sum(a[f"{src}_{s}"][look].sum() * kwh[s] for s in SWC) / 1000.0) * self.share_sw[i] * 1.2 + res * Qh
                    surplus = np.clip(inv_h - need_h, 0, None)
                    deficit = np.clip(np.minimum(need_h, Qh) - inv_h, 0, None)
                    mv = min(surplus.sum(), deficit.sum(), st["van_rebalance_frac_h"] * Q)
                    if mv > 0:
                        inv_h = inv_h - surplus / surplus.sum() * mv + deficit / deficit.sum() * mv
                av_h = np.clip(inv_h - res * Qh, 0, None)
                pr_sw = np.minimum(prio * SWh, np.minimum(res * Qh, inv_h - av_h))
                s0 = np.minimum(SWh - prio * SWh, av_h)
                exs = np.clip(SWh - prio * SWh - av_h, 0, None)
                frs = np.clip(av_h - s0, 0, None)
                rsw = min(acc["swap"] * exs.sum() * (1 - leak), frs.sum())
                take_h = s0 + pr_sw + (frs / max(frs.sum(), 1e-9)) * rsw
                inv_h = inv_h - take_h
                sw_srv = float(take_h.sum())
                # --- pack charging ---
                depl = np.clip(Qh - inv_h, 0, None)
                if policy.storage_opt:
                    look = slice(i + 1, min(i + 1 + oc["pack_lookahead_h"], len(self.times)))
                    fut = (sum(a[f"{src}_{s}"][look].sum() * kwh[s] for s in SWC) / 1000.0) * self.share_sw[min(i + 1, len(self.times) - 1)]
                    tgt = fut * (1 + oc["pack_safety"]) + res * Qh * 1.5
                    if inv_plan is not None:
                        tgt = np.maximum(tgt, inv_plan[t] * kpack * 0.8)
                    cheap = (a["dam_price"][i] <= np.quantile(a["dam_price"][idx0:idx0 + 24], 0.35)) or (re_fc[t] > dc_fc[t] + sw_fc[t])
                    if cheap:
                        tgt = np.maximum(tgt, 0.85 * Qh)
                    gap = np.clip(tgt - inv_h, 0, depl)
                    c_rate = np.clip(gap / np.maximum(Qh, 1e-9), 0, st["pack_max_c"])
                    stored = np.minimum(gap, np.minimum(c_rate, st["pack_max_c"]) * Qh)
                    eta_p = L.pack_eta(np.maximum(c_rate, st["pack_ai_c"]), st)
                else:
                    stored = np.minimum(depl, rack_h * st["pack_eta_high_c"])
                    eta_p = np.full(self.H, st["pack_eta_high_c"])
                pack_ac = stored / eta_p
                # --- standby / idle / aux loads ---
                cap_rated = self.dc_mw * self.kappa
                if policy.loss_aware:
                    on = np.maximum(on_plan[t], np.clip(srv_h / np.maximum(cap_rated, 1e-9) * 1.1, 0, 1))
                else:
                    on = np.ones(self.H)
                sd_p, bms, cool = L.pack_idle_losses(inv_h, Q * n_packs_per_mwh, temp, st)
                b_aux = L.bess_aux(self.bess_mwh, temp, st)
                tr_nl = gc["transformer_noload_frac"] * tr_rating
                # --- BESS decision ---
                bch = bdis = 0.0
                if self.bess_mwh > 0:
                    if policy.storage_opt and soc_plan is not None:
                        delta = soc_plan[t] - soc
                        if delta > 0.02 * self.bess_mwh:
                            bch = min(delta / eta_b_c, Pb, (soc_hi - soc) / eta_b_c)
                        elif delta < -0.02 * self.bess_mwh:
                            bdis = min(-delta, Pb, max(soc * (1 - sd_h) - soc_lo_d, 0))
                    elif not policy.storage_opt:
                        if hr < 6:
                            bch = max(min(Pb, (soc_hi - soc) / eta_b_c), 0)
                        elif 18 <= hr < 22:
                            bdis = max(min(Pb, soc * (1 - sd_h) - soc_lo_d), 0)
                    # BMS floor maintenance: trickle-charge so passive self-discharge never breaches the reserve
                    floor_gap = soc_lo_d - (soc * (1 - sd_h) - bdis)
                    if floor_gap > 0:
                        bch = max(bch, floor_gap / eta_b_c)
                # --- supply/demand balance on the AC bus ---
                re_t = self.solar_mw * a["solar_cf"][i] + self.wind_mw * a["wind_cf"][i]
                q_dam = float(dam_q[t])
                head = max(grid_lim - q_dam, 0.0)

                def ac_load(dc_s, pack):
                    p_c = L.charger_loss(dc_s * (srv_h / max(dc_srv, 1e-9)) if dc_srv > 0 else np.zeros(self.H), cap_rated, on, cfg["losses"])
                    return dc_s + p_c.sum() + pack + bms + cool + b_aux + tr_nl, p_c

                load, pc_h = ac_load(dc_srv, pack_ac.sum())
                tr_ll = L.transformer_loss(load, tr_rating, gc)[1]
                load += tr_ll
                net = load + bch - eta_b_d * bdis - re_t - q_dam
                rtm_b = disc = 0.0
                curtail_dc = 0.0
                if net > 0:
                    if net > head:  # grid limit binds: cancel charging, use BESS, curtail packs then DC
                        short = net - head
                        cut = min(max(bch - max(soc_lo_d - soc * (1 - sd_h), 0) / eta_b_c, 0.0), short)
                        bch -= cut
                        short -= cut
                        if short > 0 and (policy.storage_opt or 18 <= hr < 22):
                            extra = min(short / eta_b_d, Pb - bdis, max(soc * (1 - sd_h) - soc_lo_d - bdis, 0))
                            bdis += extra
                            short -= extra * eta_b_d
                        if short > 0:
                            red = min(short, pack_ac.sum())
                            if pack_ac.sum() > 0:
                                f_ = 1 - red / pack_ac.sum()
                                stored, pack_ac = stored * f_, pack_ac * f_
                            short -= red
                        if short > 0:
                            curtail_dc = min(short / 1.06, dc_srv)
                            srv_h = srv_h * (1 - curtail_dc / max(dc_srv, 1e-9))
                            dc_srv -= curtail_dc
                        load, pc_h = ac_load(dc_srv, pack_ac.sum())
                        tr_ll = L.transformer_loss(load, tr_rating, gc)[1]
                        load += tr_ll
                        net = load + bch - eta_b_d * bdis - re_t - q_dam
                    for _ in range(3):  # residual grid-limit violation -> curtail more DC
                        if net <= head + 1e-9 or dc_srv <= 0:
                            break
                        more = min((net - head) * 1.02, dc_srv)
                        srv_h = srv_h * (1 - more / max(dc_srv, 1e-9))
                        dc_srv -= more
                        curtail_dc += more
                        load, pc_h = ac_load(dc_srv, pack_ac.sum())
                        tr_ll = L.transformer_loss(load, tr_rating, gc)[1]
                        load += tr_ll
                        net = load + bch - eta_b_d * bdis - re_t - q_dam
                    imp = min(max(net, 0.0), head)
                    if policy.storage_opt and imp > 0 and soc * (1 - sd_h) - soc_lo_d - bdis > 0:
                        rtm_land = a["rtm_price"][i] + landed_add
                        cost_now = min(rtm_land, tod_px[t])
                        if cost_now > 1.25 * params["terminal_value"] / max(oc["terminal_value_frac"], 1e-9) and bdis < Pb:
                            extra = min(imp / eta_b_d, Pb - bdis, max(soc * (1 - sd_h) - soc_lo_d - bdis, 0)) * 0.5
                            bdis += extra
                            imp -= extra * eta_b_d
                    if policy.ml_forecast or policy.stochastic_bid or policy.storage_opt:
                        if a["rtm_price"][i] + landed_add < tod_px[t]:
                            rtm_b = imp
                        else:
                            disc = imp
                    else:
                        disc = imp
                    net = 0.0
                surplus = max(-net, 0.0)
                re_sold = re_bank = re_spill = dam_resold = 0.0
                if surplus > 0:
                    cancel = min(bdis * eta_b_d, surplus)  # never export battery energy
                    bdis -= cancel / eta_b_d
                    surplus -= cancel
                    if policy.storage_opt and self.bess_mwh > 0:
                        room = max(min(Pb - bch, (soc_hi - soc) / eta_b_c - bch), 0.0)
                        add = min(room, surplus)
                        bch += add
                        surplus -= add
                    dam_resold = min(surplus, q_dam)
                    surplus -= dam_resold
                    re_x = min(surplus, re_t)
                    if policy.loss_aware and re_x > 0:
                        cap_bank = max(0.3 * discom_month - bank_month, 0.0)
                        if bank_val > a["gdam_price"][i] - gc["gdam_fee"]:
                            re_bank = min(re_x, cap_bank)
                        re_sold = re_x - re_bank
                        bank_month += re_bank
                    else:
                        re_spill = re_x
                discom_month += disc
                # --- state updates & physics ---
                sd_b = soc * st["bess_self_discharge_day"] / 24
                soc_new = soc - sd_b + eta_b_c * bch - bdis
                soc_new = min(max(soc_new, 0.0), self.bess_mwh)
                d_soc = soc_new - soc
                soc = soc_new
                inv_before = inv_h.sum() + take_h.sum()
                inv_h = np.minimum(inv_h + stored - sd_p, Qh)
                d_inv = inv_h.sum() - inv_before
                re_used = re_t - re_sold - re_bank - re_spill
                dam_used = q_dam - dam_resold
                grid_imp = dam_used + rtm_b + disc
                # losses (MWh)
                l_chg_sb = float((cfg["losses"]["charger_p0_frac"] * on * cap_rated).sum())
                l_chg = float(pc_h.sum())
                l_bess = (1 - eta_b_c) * bch + (1 - eta_b_d) * bdis + b_aux + sd_b
                l_pack_chg = float((pack_ac - stored).sum())
                l_idle = float(sd_p.sum()) + bms + cool
                l_dist = tr_nl + tr_ll
                # unserved DC -> 30% waits one hour
                unserved = max(dem_tot - dc_srv, 0.0)
                wait_dc = cu["dc_wait_frac"] * unserved
                lost = unserved - wait_dc
                # delivered by segment (pro-rata)
                del_s = dc_srv * seg_share
                sw_share_n = sw_n / max(sw_n.sum(), 1e-9)
                sw_e_unit = sw_e / max(sw_n.sum(), 1e-9)
                n_served = sw_srv / max(sw_e_unit, 1e-12) if sw_e > 0 else 0.0
                # money (INR)
                rev_dc = float(sum(del_s[j] * prices[s][t] for j, s in enumerate(SEGS))) * 1000.0
                rev_sw = float(sum(n_served * sw_share_n[k] * cu["swap_price"][s] * prices["swap"][t] for k, s in enumerate(SWC)))
                revenue = rev_dc + rev_sw
                c_dam = q_dam * (a["dam_price"][i] + landed_add) * 1000
                c_rtm = rtm_b * (a["rtm_price"][i] + landed_add) * 1000
                c_dis = disc * tod_px[t] * 1000
                c_re = re_t * gc["oa_charges"] * 0 + re_t * cfg["re"]["landed_cost"] * 1000
                credit = (dam_resold * (a["rtm_price"][i] - gc["dam_fee"]) + re_sold * (a["gdam_price"][i] - gc["gdam_fee"])
                          + re_bank * bank_val) * 1000
                cost_energy = c_dam + c_rtm + c_dis + c_re - credit
                opex_var = (cfg["opex"]["payments_frac"] + cfg["opex"]["site_host_frac"]) * revenue + cfg["opex"]["swap_partner_inr"] * n_served
                cf_b = float(calendar_rate_h(temp, soc / max(self.bess_mwh, 1e-9), dg)) if self.bess_mwh > 0 else 0.0
                cf_p = float(calendar_rate_h(temp, inv_h.sum() / max(Q, 1e-9), dg))
                deg = (bdis * deg_b + sw_srv * deg_p) * 1000 + cf_b * self.bess_mwh * 1000 * dg["bess_replacement_inr_kwh"] * (1 - dg["residual_frac"]) / 0.2 \
                    + cf_p * Q * 1000 * dg["pack_replacement_inr_kwh"] * (1 - dg["residual_frac"]) / 0.2
                # energy balance closure
                inputs = re_t + q_dam + rtm_b + disc
                outputs = (dc_srv + sw_srv + l_chg + l_bess + l_pack_chg + l_idle + l_dist + d_soc + d_inv
                           + re_sold + re_bank + re_spill + dam_resold)
                # compliance metrics
                pdev = max(abs(prices[s][t] / p_ref[s] - 1) for s in ["retail"])
                fdev = max(abs(prices[s][t] / p_ref[s] - 1) for s in ["ride", "logi"])
                vals = dict(
                    ref_retail=a["ref_retail"][i], ref_ride=a["ref_ride"][i], ref_logi=a["ref_logi"][i],
                    dem_retail=dem[0], dem_ride=dem[1], dem_logi=dem[2], del_retail=del_s[0], del_ride=del_s[1], del_logi=del_s[2],
                    fc_dc=dc_fc[t], fc_sw=sw_fc[t], ref_sw=sum(a[f"ref_{s}"][i] * kwh[s] for s in SWC) / 1000.0, dem_sw=sw_e,
                    del_sw=sw_srv, ref_sw_n=sum(a[f"ref_{s}"][i] for s in SWC), dem_sw_n=float(sw_n.sum()), del_sw_n=n_served,
                    unserved_dc=unserved, lost_dc=lost, re_total=re_t, re_used=re_used, re_sold=re_sold, re_bank=re_bank,
                    re_spill=re_spill, dam_bought=q_dam, dam_resold=dam_resold, rtm=rtm_b, discom=disc,
                    bess_ch_ac=bch, bess_dis_ac=eta_b_d * bdis, soc=soc, inv=float(inv_h.sum()), d_soc=d_soc, d_inv=d_inv,
                    pack_ac=float(pack_ac.sum()), pack_stored=float(stored.sum()),
                    loss_charger=l_chg - l_chg_sb, loss_charger_standby=l_chg_sb, loss_bess=l_bess, loss_pack_chg=l_pack_chg,
                    loss_pack_idle=l_idle, loss_dist=l_dist, revenue=revenue, rev_dc=rev_dc, rev_sw=rev_sw,
                    cost_energy=cost_energy, cost_dam=c_dam, cost_rtm=c_rtm, cost_discom=c_dis, cost_re=c_re, credit_sales=credit,
                    opex_var=opex_var, deg_cost=deg, grid_import=grid_imp, grid_limit=grid_lim, price_dev_max=pdev,
                    fleet_dev_max=fdev, guard_dev=abs(prices["retail"].mean() / p_ref["retail"] - 1),
                    inv_reserve_slack=float(inv_h.sum() - res * Q), soc_reserve_slack=soc - soc_lo if self.bess_mwh > 0 else 0.0,
                    dc_commit_frac=float(((srv_h - pr_srv_h) / np.maximum(caph, 1e-9)).max()), priority_dem=float(pr_h.sum()),
                    priority_srv=float(pr_srv_h.sum()), swaps_n_served=n_served, bess_throughput=bdis, pack_throughput=sw_srv,
                    cal_fade_bess=cf_b, cal_fade_pack=cf_p, balance_err=inputs - outputs, temp=temp, dam_price=a["dam_price"][i],
                )
                for k in REC_FIELDS:
                    recs[k].append(float(vals[k]))
                rec_time.append(self.times[i])
                for j, s in enumerate(SEGS):
                    day_rev_vol[s][0] += del_s[j] * prices[s][t]
                    day_rev_vol[s][1] += del_s[j]
                day_rev_vol["swap"][0] += sw_srv * prices["swap"][t]
                day_rev_vol["swap"][1] += sw_srv
            # previous day's realised average tariff -> tomorrow's reference
            for s in SEGS + ["swap"]:
                v = day_rev_vol[s]
                p_ref[s] = v[0] / v[1] if v[1] > 1e-9 else prices[s].mean()
        out = pd.DataFrame(recs)
        out.insert(0, "time", rec_time)
        out.insert(1, "cluster", self.cl)
        return out
