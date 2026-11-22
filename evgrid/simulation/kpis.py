"""KPI computation from hourly simulation records (overall, per cluster, per month).

Records may carry an annualisation weight column ``w`` (representative-week runs); full-year runs use w=1.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from evgrid.data.synth import cluster_capacities

SEGS = ["retail", "ride", "logi"]
LOSS_COLS = ["loss_charger", "loss_charger_standby", "loss_bess", "loss_pack_chg", "loss_pack_idle", "loss_dist"]


def reliability(df: pd.DataFrame) -> Dict[str, float]:
    """R = credited delivered / reference demand; credit capped at reference per segment-day."""
    d = df.assign(day=pd.to_datetime(df["time"]).dt.normalize())
    w = d.get("w", pd.Series(1.0, index=d.index))
    d = d.assign(**{c: d[c] * w for c in [f"ref_{s}" for s in SEGS] + [f"del_{s}" for s in SEGS] + ["ref_sw", "del_sw"]})
    g = d.groupby(["cluster", "day"])[[f"ref_{s}" for s in SEGS] + [f"del_{s}" for s in SEGS] + ["ref_sw", "del_sw"]].sum()
    credit = sum(np.minimum(g[f"del_{s}"], g[f"ref_{s}"]) for s in SEGS) + np.minimum(g["del_sw"], g["ref_sw"])
    ref = sum(g[f"ref_{s}"] for s in SEGS) + g["ref_sw"]
    dc_credit = sum(np.minimum(g[f"del_{s}"], g[f"ref_{s}"]) for s in SEGS)
    dc_ref = sum(g[f"ref_{s}"] for s in SEGS)
    return dict(R=float(credit.sum() / ref.sum()), R_dc=float(dc_credit.sum() / dc_ref.sum()),
                R_swap=float(np.minimum(g["del_sw"], g["ref_sw"]).sum() / g["ref_sw"].sum()))


def kpis(df: pd.DataFrame, cfg: dict, fixed_cr: Optional[Dict[str, float]] = None, extra_opex_cr: Optional[Dict[str, float]] = None,
         year_frac: Optional[float] = None) -> Dict[str, float]:
    """KPIs for a record subset. ``fixed_cr`` = annual fixed opex by cluster (INR crore)."""
    w = df["w"] if "w" in df.columns else pd.Series(1.0, index=df.index)
    S = lambda c: float((df[c] * w).sum())  # noqa: E731
    out = reliability(df)
    dc_del = sum(S(f"del_{s}") for s in SEGS)
    dc_ref = sum(S(f"ref_{s}") for s in SEGS)
    dc_dem = sum(S(f"dem_{s}") for s in SEGS)
    delivered = dc_del + S("del_sw")
    out["R_literal"] = delivered / max(S("fc_dc") + S("fc_sw"), 1e-9)
    out["dc_lost_session_pct"] = S("lost_dc") / max(dc_dem, 1e-9)
    out["swap_stockout_pct"] = 1 - S("del_sw_n") / max(S("dem_sw_n"), 1e-9)
    out["induced_minus_suppressed_mwh"] = (dc_dem - dc_ref) + (S("dem_sw") - S("ref_sw"))
    meter = S("re_used") + S("dam_bought") - S("dam_resold") + S("rtm") + S("discom")
    loss = sum(S(c) for c in LOSS_COLS)
    out["loss_pct"] = loss / max(meter, 1e-9)
    for c in LOSS_COLS:
        out[f"{c}_pct"] = S(c) / max(meter, 1e-9)
    out["delivered_gwh"] = delivered / 1000
    out["dc_delivered_gwh"] = dc_del / 1000
    out["swap_delivered_gwh"] = S("del_sw") / 1000
    out["reference_gwh"] = (dc_ref + S("ref_sw")) / 1000
    out["meter_input_gwh"] = meter / 1000
    out["re_total_gwh"] = S("re_total") / 1000
    out["re_spill_gwh"] = S("re_spill") / 1000
    out["re_sold_banked_gwh"] = (S("re_sold") + S("re_bank")) / 1000
    out["re_utilisation_pct"] = S("re_used") / max(S("re_total"), 1e-9)
    out["re_monetised_pct"] = (S("re_used") + S("re_sold") + S("re_bank")) / max(S("re_total"), 1e-9)
    out["effective_re_share"] = S("re_used") / max(meter, 1e-9)
    out["avg_procurement_inr_kwh"] = S("cost_energy") / max(meter * 1000, 1e-9)
    out["peak_grid_import_mw"] = float(df.groupby("time")["grid_import"].sum().max()) if len(df) else 0.0
    rev, ce, ov = S("revenue") / 1e7, S("cost_energy") / 1e7, S("opex_var") / 1e7
    if year_frac is None:
        year_frac = float(w.sum()) / (8760.0 * max(df["cluster"].nunique(), 1))
    fixed = sum((fixed_cr or {}).get(c, 0.0) for c in df["cluster"].unique()) * year_frac
    extra = sum((extra_opex_cr or {}).get(c, 0.0) for c in df["cluster"].unique()) * year_frac
    deg = S("deg_cost") / 1e7
    out.update(revenue_cr=rev, energy_cost_cr=ce, opex_var_cr=ov + extra, opex_fixed_cr=fixed, degradation_cr=deg)
    out["ebitda_cr"] = rev - ce - ov - extra - fixed
    out["ebitda_margin"] = out["ebitda_cr"] / max(rev, 1e-9)
    out["ebitda_after_deg_cr"] = out["ebitda_cr"] - deg
    out["ebitda_after_deg_margin"] = out["ebitda_after_deg_cr"] / max(rev, 1e-9)
    # storage
    caps = cluster_capacities(cfg)
    cls = list(df["cluster"].unique())
    bess = float(caps.loc[cls, "bess_mwh"].sum())
    packs = float(caps.loc[cls, "pack_mwh"].sum())
    days = float(w.sum()) / 24.0 / max(len(cls), 1)
    out["bess_efc"] = S("bess_throughput") / max(bess, 1e-9)
    out["bess_efc_per_year"] = out["bess_efc"] / max(year_frac, 1e-9)
    out["pack_turns_per_day"] = S("del_sw") / max(packs, 1e-9) / max(days, 1e-9)
    efc_b, efc_p = cfg["degradation"]["bess_efc_life"], cfg["degradation"]["pack_efc_life"]
    out["bess_soh_change_pp_yr"] = -100 * (out["bess_efc_per_year"] / efc_b * 0.2
                                           + float((df["cal_fade_bess"] * w).sum()) / max(len(cls), 1) / max(year_frac, 1e-9))
    out["pack_soh_change_pp_yr"] = -100 * (S("del_sw") / max(packs, 1e-9) / max(year_frac, 1e-9) / efc_p * 0.2
                                           + float((df["cal_fade_pack"] * w).sum()) / max(len(cls), 1) / max(year_frac, 1e-9))
    # compliance (hard constraints)
    tol = 1e-6
    out["compliance_price_band"] = float((df["price_dev_max"] <= cfg["customers"]["price_band"] + tol).mean())
    out["compliance_fleet_band"] = float((df["fleet_dev_max"] <= cfg["customers"]["price_band"] + tol).mean())
    out["compliance_daily_guardrail"] = float((df["guard_dev"] <= cfg["customers"]["daily_guardrail"] + 1e-3).mean())
    out["compliance_grid_limit"] = float((df["grid_import"] <= df["grid_limit"] + tol).mean())
    out["compliance_bess_reserve"] = float((df["soc_reserve_slack"] >= -tol).mean())
    out["compliance_charger_reserve"] = float((df["dc_commit_frac"] <= 1 - cfg["reserve"]["frac"] + tol).mean())
    out["compliance_pack_reserve"] = float((df["inv_reserve_slack"] >= -cfg["reserve"]["priority_demand_frac"] * df["dem_sw"] - tol).mean())
    out["max_energy_balance_error_mwh"] = float(df["balance_err"].abs().max())
    return out


def kpi_table(df: pd.DataFrame, cfg: dict, fixed_cr: Dict[str, float], extra_opex_cr: Optional[Dict[str, float]] = None,
              monthly: bool = False) -> Dict[str, dict]:
    res = {"overall": kpis(df, cfg, fixed_cr, extra_opex_cr)}
    for cl, g in df.groupby("cluster", sort=False):
        res[cl] = kpis(g, cfg, fixed_cr, extra_opex_cr)
    if monthly:
        m = pd.to_datetime(df["time"]).dt.strftime("%Y-%m")
        res["monthly"] = {}
        for (mo, cl), g in df.groupby([m, "cluster"]):
            k = kpis(g, cfg, fixed_cr, extra_opex_cr, year_frac=len(g) / 8760.0)
            res["monthly"].setdefault(mo, {})[cl] = {x: k[x] for x in ["R", "loss_pct", "ebitda_margin"]}
    return res


def fixed_opex_alloc(cfg: dict) -> Dict[str, float]:
    tot = cfg["opex"]["fixed_opex_cr"]
    sh = cfg["network"]["capacity_share"]
    return {c: tot * sh[c] for c in cfg["clusters"]}
