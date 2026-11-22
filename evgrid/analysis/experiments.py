"""Ablations, stress scenarios, Monte Carlo EBITDA-at-risk, tornado sensitivity, Pareto, scalability, economics.

All runs use the frozen calibrated config, identical data and common random numbers.  Ablations/scenarios run
on representative weeks (annualised with season weights); Monte Carlo/sensitivity/Pareto on one week per season.
"""
from __future__ import annotations

import logging
import time
import tracemalloc
from typing import Dict, List

import numpy as np
import pandas as pd

from evgrid.config import CONFIG_DIR, RESULTS_DIR, deep_merge, load_config, read_yaml, save_json
from evgrid.simulation.kpis import fixed_opex_alloc, kpi_table, kpis
from evgrid.simulation.policies import AI, BASELINE, LEVER_LABELS, ablations
from evgrid.simulation.runner import REP_WEEKS, SEASON_DAYS, Runner, rep_week_days

log = logging.getLogger(__name__)
KEYS = ["R", "R_dc", "R_swap", "loss_pct", "ebitda_margin", "ebitda_cr", "revenue_cr", "energy_cost_cr", "delivered_gwh",
        "swap_stockout_pct", "dc_lost_session_pct", "re_utilisation_pct", "avg_procurement_inr_kwh", "ebitda_after_deg_margin"]


def one_week_per_season():
    days, w = [], {}
    for season, starts in REP_WEEKS.items():
        for d in pd.date_range(starts[0], periods=7, freq="D"):
            days.append(d)
            w[d] = SEASON_DAYS[season] / 7.0
    return days, w


def _summ(tab: Dict[str, dict]) -> Dict[str, dict]:
    return {c: {k: float(v[k]) for k in KEYS} for c, v in tab.items() if c != "monthly"}


class Experiments:
    def __init__(self, mode: str = "quick"):
        from evgrid.pipeline import ai_extra_opex, pack_plan

        self.mode = mode
        self.cfg = load_config()
        self.plan, self.rcost, _ = pack_plan(self.cfg)
        self.extra = ai_extra_opex(self.cfg, self.rcost)
        self.fixed = fixed_opex_alloc(self.cfg)
        self.runner = Runner(self.cfg)
        self.days8, self.w8 = rep_week_days()
        self.days4, self.w4 = one_week_per_season()

    def close(self):
        self.runner.close()

    def run(self, policy, scenario=None, cfg=None, short=False):
        cfg = cfg or self.cfg
        days, w = (self.days4, self.w4) if short else (self.days8, self.w8)
        df = self.runner.run(policy, days, scenario=scenario, cfg=cfg, pack_plan=self.plan, weights=w)
        extra = self.extra if policy.use_lp else None
        return kpi_table(df, cfg, fixed_opex_alloc(cfg), extra)

    # ------------------------------------------------------------------
    def ablations(self) -> dict:
        out = {"AI": _summ(self.run(AI)), "Baseline": _summ(self.run(BASELINE))}
        for lev, pol in ablations().items():
            out[lev] = _summ(self.run(pol))
            log.info("ablation %-15s R %.4f loss %.4f EBITDA %.4f", lev, out[lev]["overall"]["R"], out[lev]["overall"]["loss_pct"],
                     out[lev]["overall"]["ebitda_margin"])
        contrib = {}
        for lev in ablations():
            contrib[lev] = {k: out["AI"]["overall"][k] - out[lev]["overall"][k] for k in ("R", "loss_pct", "ebitda_margin")}
        return {"runs": out, "lever_contribution": contrib, "labels": LEVER_LABELS}

    def scenarios(self) -> dict:
        scs = read_yaml(CONFIG_DIR / "scenarios.yaml")
        out = {}
        for name, sc in scs.items():
            sc = {k: v for k, v in sc.items() if k != "label"}
            b = _summ(self.run(BASELINE, scenario=sc))
            a = _summ(self.run(AI, scenario=sc))
            out[name] = {"label": scs[name]["label"], "Baseline": b, "AI": a}
            log.info("scenario %-20s AI R %.4f loss %.4f EBITDA %.4f | min cluster EBITDA %.4f", name, a["overall"]["R"],
                     a["overall"]["loss_pct"], a["overall"]["ebitda_margin"], min(a[c]["ebitda_margin"] for c in self.cfg["clusters"]))
        return out

    def monte_carlo(self, n_paths: int) -> dict:
        res = {"AI": [], "Baseline": []}
        for i in range(n_paths):
            sc = {"price_paths_seed": 1000 + i}
            for pol in (AI, BASELINE):
                t = self.run(pol, scenario=sc, short=True)
                res[pol.name].append({c: t[c]["ebitda_margin"] for c in ["overall"] + self.cfg["clusters"]})
        summ = {}
        for p, rows in res.items():
            df = pd.DataFrame(rows)
            summ[p] = {c: {"mean": float(df[c].mean()), "p5": float(df[c].quantile(0.05)), "p95": float(df[c].quantile(0.95))} for c in df}
        return {"n_paths": n_paths, "summary": summ, "paths": res}

    def sensitivity(self) -> dict:
        base = _summ(self.run(AI, short=True))["overall"]
        cu = self.cfg["customers"]
        cases = {
            "DAM/RTM price +-30%": ({"price_mult": 0.7}, None, {"price_mult": 1.3}, None),
            "Elasticity +-50%": (None, {"customers": {"elasticity": {k: v * 0.5 for k, v in cu["elasticity"].items()}}},
                                 None, {"customers": {"elasticity": {k: v * 1.5 for k, v in cu["elasticity"].items()}}}),
            "Steering acceptance x0.5 / x1.5": (None, {"customers": {"steering_ai": {k: v * 0.5 for k, v in cu["steering_ai"].items()}}},
                                                None, {"customers": {"steering_ai": {k: min(v * 1.5, 0.95) for k, v in cu["steering_ai"].items()}}}),
            "BESS -50% / +50%": ({"bess_mult": 0.5}, None, {"bess_mult": 1.5}, None),
            "VoLL INR 10 / 80": (None, {"optimization": {"voll": 10.0}}, None, {"optimization": {"voll": 80.0}}),
            "Swap price -20% / +20%": (None, {"customers": {"swap_price": {k: v * 0.8 for k, v in cu["swap_price"].items()}}},
                                       None, {"customers": {"swap_price": {k: v * 1.2 for k, v in cu["swap_price"].items()}}}),
            "RE contracts -30% / +30%": ({"re_mult": 0.7}, None, {"re_mult": 1.3}, None),
        }
        out = {"base": base, "cases": {}}
        for name, (sl, cl_, sh, ch) in cases.items():
            lo = _summ(self.run(AI, scenario=sl, cfg=deep_merge(self.cfg, cl_) if cl_ else None, short=True))["overall"]
            hi = _summ(self.run(AI, scenario=sh, cfg=deep_merge(self.cfg, ch) if ch else None, short=True))["overall"]
            out["cases"][name] = {"low": lo, "high": hi}
            log.info("sensitivity %-32s EBITDA %.4f / %.4f  R %.4f / %.4f", name, lo["ebitda_margin"], hi["ebitda_margin"], lo["R"], hi["R"])
        return out

    def pareto(self, vols: List[float]) -> dict:
        pts = []
        for v in vols:
            k = _summ(self.run(AI, cfg=deep_merge(self.cfg, {"optimization": {"voll": v}}), short=True))["overall"]
            pts.append({"voll": v, **k})
        return {"points": pts}

    def levers_for_gaps(self) -> dict:
        """Quantify what closes target gaps (reliability; EBITDA in surplus clusters)."""
        cu = self.cfg["customers"]
        out = {}
        for mult in (1.25, 1.5):
            c = deep_merge(self.cfg, {"customers": {"steering_ai": {k: min(v * mult, 0.95) for k, v in cu["steering_ai"].items()}}})
            out[f"steering_x{mult}"] = _summ(self.run(AI, cfg=c))
        out["bess_plus_100mwh"] = _summ(self.run(AI, scenario={"bess_mult": 400 / 300}))
        # annual RE-contract re-allocation: move half of HYD/CHE excess contracts to DEL/MUM
        out["re_reallocation"] = _summ(self.run(AI, scenario={"re_mult": {"HYD": 0.6, "CHE": 0.6, "DEL": 1.45, "MUM": 1.4}}))
        out["fleet_contract_share"] = _summ(self.run(AI, cfg=deep_merge(self.cfg, {"customers": {
            "shift_share": {"retail": 0.10, "ride": 0.35, "logi": 0.55, "swap": 0.15}}})))
        # structural: re-site 3 pp of installed capacity each from HYD/CHE to MUM/DEL (assets + their fixed cost move)
        out["asset_resiting"] = _summ(self.run(AI, cfg=deep_merge(self.cfg, {"network": {"capacity_share": {
            "DEL": 0.29, "MUM": 0.27, "BLR": 0.20, "HYD": 0.12, "CHE": 0.12}}})))
        return out


# ----------------------------------------------------------------------------
def scalability(cfg: dict) -> dict:
    """LP compute time & memory vs nodes, hubs and horizon; fitted scaling exponents."""
    from evgrid.optimization.day_ahead import DAInputs, solve_day_ahead

    rng = np.random.default_rng(0)

    def inputs(T: int, dt: float = 1.0) -> DAInputs:
        h = np.arange(T) * dt % 24
        dem = (40 + 25 * np.sin(2 * np.pi * (h - 14) / 24)) * dt
        return DAInputs(dc_dem=dem, dc_cap=dem * 0.9, dc_free=dem * 0.2, dc_flex=dem * 0.25, dc_price=np.full(T, 18.0),
                        sw_dem=dem * 0.3, sw_price=np.full(T, 40.0), re=np.clip(30 * np.sin(np.pi * (h - 6) / 12), 0, None) * dt,
                        dam_px=4 + 3 * rng.random(T), discom_px=np.full(T, 6.0), sell_px=np.full(T, 3.5), aux=np.full(T, 1.0 * dt),
                        temp=np.full(T, 30.0), grid_mw=120.0 * dt, bess_mwh=78.0, bess_p=39.0 * dt, soc0=39.0, pack_q=70.0,
                        pack_rack=100.0 * dt, inv0=60.0,
                        params=dict(eta_c=0.935, eta_d=0.935, eta_chg=0.96, eta_p=0.93, reserve=0.05, dam_fee=0.02, oa=1.4, voll=35,
                                    soc_min=0.1, soc_max=0.95, deg_bess=1.1, deg_pack=1.9, hold_bess=0.05, hold_pack=0.08,
                                    bess_sd_h=0.001 / 24 * dt, pack_sd_h=0.025 / 730 * dt, shift_cost=0.3, backlog_max=4 / dt,
                                    terminal_value=5.0))

    def time_lp(T: int, dt: float = 1.0, reps: int = 5) -> float:
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            r = solve_day_ahead(inputs(T, dt))
            ts.append(time.perf_counter() - t0)
            assert r.status == 0
        return float(np.median(ts))

    horizon = {}
    for T in (24, 36, 72, 144, 288):
        horizon[T] = time_lp(T, 24 / 24 if T <= 72 else 0.25)
    xs = np.log(list(horizon)), np.log(list(horizon.values()))
    exp_T = float(np.polyfit(xs[0], xs[1], 1)[0])
    t36 = horizon[36]
    nodes = {}
    for n in (5, 10, 25, 50, 100):
        t0 = time.perf_counter()
        for _ in range(n):
            solve_day_ahead(inputs(36))
        nodes[n] = time.perf_counter() - t0
    exp_n = float(np.polyfit(np.log(list(nodes)), np.log(list(nodes.values())), 1)[0])
    # hub-level closed form cost vs hubs (per cluster-hour)
    hubs = {}
    for H in (12, 50, 100, 200):
        D = rng.random((8760, H))
        C = rng.random((8760, H))
        t0 = time.perf_counter()
        s0 = np.minimum(D, C).sum(1)
        e = np.clip(D - C, 0, None).sum(1)
        f = np.clip(C - D, 0, None).sum(1)
        _ = s0 + np.minimum(0.4 * e, f)
        hubs[H] = (time.perf_counter() - t0) / 8760
    tracemalloc.start()
    solve_day_ahead(inputs(144, 0.25))
    mem96 = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    tracemalloc.start()
    solve_day_ahead(inputs(36))
    mem36 = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    per_day_100 = nodes[100] * 5  # 5 price scenarios per node, sequential single core
    return {
        "lp_time_vs_horizon_s": {str(k): v for k, v in horizon.items()}, "horizon_exponent": exp_T,
        "lp_time_vs_nodes_s_sequential": {str(k): v for k, v in nodes.items()}, "nodes_exponent": exp_n,
        "hub_closed_form_s_per_cluster_hour": {str(k): v for k, v in hubs.items()},
        "lp_36h_s": t36, "lp_mem_mb_36h": mem36, "lp_mem_mb_96blocks_plus_lookahead": mem96,
        "day_ahead_100_nodes_5_scenarios_s_single_core": per_day_100,
        "iex_dam_window_s": 2 * 3600, "rtm_cadence_s": 1800,
        "headroom_vs_dam_window_x": 2 * 3600 / max(per_day_100, 1e-9),
    }


def economics(main: dict, gaps: dict, scen: dict, cfg: dict) -> dict:
    ec = cfg["economics"]
    b, a = main["Baseline"]["overall"], main["AI"]["overall"]
    d_ebitda = a["ebitda_cr"] - b["ebitda_cr"]  # AI EBITDA already net of programme run cost & redeployment
    wacc, yrs = ec["wacc"], ec["years"]
    flows = [-ec["ai_build_cr"]] + [d_ebitda] * yrs
    npv = float(sum(f / (1 + wacc) ** t for t, f in enumerate(flows)))
    payback_months = float(ec["ai_build_cr"] / max(d_ebitda, 1e-9) * 12)
    growth_flows = [-ec["ai_build_cr"]] + [d_ebitda * (1.2 ** (t - 1)) for t in range(1, yrs + 1)]
    npv_g = float(sum(f / (1 + wacc) ** t for t, f in enumerate(growth_flows)))
    # demand growth / 2x network: clusters decouple -> benefits & compute scale ~linearly
    dg = scen.get("demand_growth", {})
    d_growth = (dg["AI"]["overall"]["ebitda_cr"] - dg["Baseline"]["overall"]["ebitda_cr"]) if dg else None
    # marginal +100 MWh BESS
    bp = gaps.get("bess_plus_100mwh", {}).get("overall")
    ref = gaps.get("_ai_rep", {}).get("overall")
    out = {"delta_ebitda_cr_yr": d_ebitda, "npv_5yr_cr": npv, "npv_5yr_with_20pct_growth_cr": npv_g, "payback_months": payback_months,
           "ai_build_cr": ec["ai_build_cr"], "ai_run_cr_yr": ec["ai_run_cr_yr"], "wacc": wacc,
           "delta_ebitda_demand_growth_cr_yr": d_growth, "delta_ebitda_2x_network_cr_yr": 2 * d_ebitda}
    if bp and ref:
        dv = bp["ebitda_cr"] - ref["ebitda_cr"]
        capex = 100 * 1000 * ec["bess_capex_inr_kwh"] / 1e7
        crf = wacc / (1 - (1 + wacc) ** -10)
        lo, hi = ec["bess_auction_lakh_mw_month"]
        out["bess_100mwh"] = {"delta_ebitda_cr_yr": dv, "delta_R_pp": 100 * (bp["R"] - ref["R"]), "capex_cr": capex,
                              "annualised_capex_cr_10yr": capex * crf,
                              "auction_benchmark_cr_yr": [lo * 1e5 * 50 * 12 / 1e7, hi * 1e5 * 50 * 12 / 1e7]}
    return out


def run_all_experiments(mode: str = "quick") -> dict:
    ex = Experiments(mode)
    out = {}
    try:
        t0 = time.time()
        out["ablations"] = ex.ablations()
        save_json(out["ablations"], RESULTS_DIR / "ablations.json")
        out["scenarios"] = ex.scenarios()
        save_json(out["scenarios"], RESULTS_DIR / "scenarios.json")
        gaps = ex.levers_for_gaps()
        gaps["_ai_rep"] = out["ablations"]["runs"]["AI"]
        save_json(gaps, RESULTS_DIR / "gap_levers.json")
        out["pareto"] = ex.pareto([10.0, 35.0, 80.0] if mode == "quick" else [5.0, 15.0, 35.0, 60.0, 100.0])
        save_json(out["pareto"], RESULTS_DIR / "pareto.json")
        out["sensitivity"] = ex.sensitivity()
        save_json(out["sensitivity"], RESULTS_DIR / "sensitivity.json")
        out["monte_carlo"] = ex.monte_carlo(10 if mode == "quick" else 20)
        save_json(out["monte_carlo"], RESULTS_DIR / "monte_carlo.json")
        log.info("experiments done in %.0f s", time.time() - t0)
    finally:
        ex.close()
    return out
