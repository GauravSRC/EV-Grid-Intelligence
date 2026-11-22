"""Strategic (monthly) layer: inter-cluster swap-pack redeployment transport LP.

min  sum_ij c_ij x_ij + pi * sum_j short_j
s.t. short_j >= Qtarget_j - (Q_j + sum_i x_ij - sum_k x_jk)
     sum_k x_jk <= max(Q_j - (1+reserve) Qtarget_j, 0)        (source keeps its own need + reserve)
     x, short >= 0
Targets use only information available at the start of the month (last year's same month x growth).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from evgrid.optimization.lp_builder import LPBuilder

# INR per pack moved (case brief: CHE->BLR ~40, HYD->BLR ~55, DEL<->MUM ~130); others assumed by distance
PACK_MOVE_INR = {
    ("CHE", "BLR"): 40, ("HYD", "BLR"): 55, ("DEL", "MUM"): 130, ("HYD", "CHE"): 60, ("HYD", "MUM"): 80,
    ("CHE", "MUM"): 110, ("BLR", "MUM"): 90, ("HYD", "DEL"): 120, ("CHE", "DEL"): 150, ("BLR", "DEL"): 150,
}


def move_cost(i: str, j: str) -> float:
    return float(PACK_MOVE_INR.get((i, j), PACK_MOVE_INR.get((j, i), 150)))


def redeploy_lp(Q: Dict[str, float], target: Dict[str, float], packs_per_mwh: float, reserve: float,
                shortage_inr_per_mwh: float) -> Tuple[Dict[str, float], float, Dict[Tuple[str, str], float]]:
    cls = list(Q)
    pairs = [(i, j) for i in cls for j in cls if i != j]
    lp = LPBuilder()
    cost = np.array([move_cost(i, j) * packs_per_mwh for i, j in pairs])
    x = lp.var("x", len(pairs), 0, cost=cost)
    sh = lp.var("short", len(cls), 0, cost=shortage_inr_per_mwh)
    for k, j in enumerate(cls):
        inn = [x[n] for n, (a, b) in enumerate(pairs) if b == j]
        out = [x[n] for n, (a, b) in enumerate(pairs) if a == j]
        terms = [(np.array([sh[k]]), 1.0)] + [(np.array([v]), 1.0) for v in inn] + [(np.array([v]), -1.0) for v in out]
        lp.le([(np.array([sh[k]]), -1.0)] + [(np.array([v]), -1.0) for v in inn] + [(np.array([v]), 1.0) for v in out],
              -(target[j] - Q[j]))
        lp.le([(np.array([v]), 1.0) for v in out], max(Q[j] - (1 + reserve) * target[j], 0.0))
        del terms
    r = lp.solve()
    flows = {p: float(v) for p, v in zip(pairs, r["x"]) if v > 1e-6}
    newQ = dict(Q)
    for (i, j), v in flows.items():
        newQ[i] -= v
        newQ[j] += v
    tcost = float(sum(move_cost(i, j) * packs_per_mwh * v for (i, j), v in flows.items()))
    return newQ, tcost, flows


def monthly_pack_plan(cfg: dict, hourly: pd.DataFrame, caps: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], list]:
    """Plan pack MWh per cluster for every FY26 month. Returns (plan[cluster][YYYY-MM], cost_cr by cluster, log)."""
    kwh = cfg["customers"]["swap_kwh"]
    h = hourly.assign(e=(hourly["swaps_s2w"] * kwh["s2w"] + hourly["swaps_s3w"] * kwh["s3w"]) / 1000.0 * cfg["demand"]["demand_scale"])
    h = h.assign(m=pd.to_datetime(h["time"]).dt.to_period("M"))
    hist = h.groupby(["cluster", "m"])["e"].sum()
    days = h.groupby(["cluster", "m"])["time"].count() / 24
    growth = 1 + cfg["demand"]["growth_yoy"]
    target_turns = 2.6  # turns/day at which stock-outs stay low under hub misallocation (assumption)
    Q = {c: float(caps.loc[c, "pack_mwh"]) for c in cfg["clusters"]}
    plan: Dict[str, Dict[str, float]] = {c: {} for c in cfg["clusters"]}
    cost = {c: 0.0 for c in cfg["clusters"]}
    logs = []
    for m in pd.period_range(cfg["meta"]["sim_start"], cfg["meta"]["sim_end"], freq="M"):
        prev = m - 12
        tgt = {c: float(hist.get((c, prev), 0.0) / max(days.get((c, prev), 30), 1) * growth / target_turns) for c in Q}
        # scale targets to the fleet (cannot create packs)
        s = sum(Q.values()) / max(sum(tgt.values()), 1e-9)
        tgt = {c: v * min(s, 1.0) if s < 1 else v for c, v in tgt.items()}
        newQ, tc, flows = redeploy_lp(Q, tgt, 1000.0 / cfg["storage"]["pack_avg_kwh"], cfg["reserve"]["frac"], 150000.0)
        for (i, j), v in flows.items():
            cost[j] += move_cost(i, j) * 1000.0 / cfg["storage"]["pack_avg_kwh"] * v / 1e7
        Q = newQ
        for c in Q:
            plan[c][str(m)] = Q[c]
        logs.append(dict(month=str(m), moved_mwh=float(sum(flows.values())), cost_inr=tc, **{f"Q_{c}": Q[c] for c in Q}))
    return plan, cost, logs
