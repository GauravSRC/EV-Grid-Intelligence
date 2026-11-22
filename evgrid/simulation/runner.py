"""Multi-cluster runner (parallel over clusters with a persistent process pool) + representative weeks."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evgrid.simulation.policies import Policy

log = logging.getLogger(__name__)
_G: dict = {}

# 8 representative weeks (2 per season) with season weights for annualisation
REP_WEEKS = {
    "summer": ["2025-04-14", "2025-05-19"],
    "monsoon": ["2025-07-07", "2025-08-18"],
    "post-monsoon": ["2025-10-13", "2025-11-17"],
    "winter": ["2026-01-12", "2026-02-16"],
}
SEASON_DAYS = {"summer": 91, "monsoon": 92, "post-monsoon": 91, "winter": 91}


def full_year_days(cfg: dict) -> List[pd.Timestamp]:
    return list(pd.date_range(cfg["meta"]["sim_start"], cfg["meta"]["sim_end"], freq="D"))


def rep_week_days() -> Tuple[List[pd.Timestamp], Dict[pd.Timestamp, float]]:
    """Days of the 8 representative weeks and per-day annualisation weights."""
    days, w = [], {}
    for season, starts in REP_WEEKS.items():
        for s in starts:
            for d in pd.date_range(s, periods=7, freq="D"):
                days.append(d)
                w[d] = SEASON_DAYS[season] / (7.0 * len(starts))
    return days, w


def _init(cfg_base: dict) -> None:
    from evgrid.data.io import load_or_generate, re_contract_mw
    from evgrid.pipeline import load_forecasts

    data = load_or_generate(cfg_base)
    _G["hourly"] = data["hourly"]
    _G["hubs"] = data["hubs"]
    _G["fc"] = load_forecasts()
    _G["re"] = re_contract_mw(cfg_base, data["hourly"])


def _task(args) -> pd.DataFrame:
    from evgrid.simulation.engine import ClusterSim

    cfg, cl, policy, days, scenario, pack_plan = args
    if not _G:
        _init(cfg)
    hourly = _G["hourly"]
    sim = ClusterSim(cfg, cl, hourly[hourly["fy"].isin(["FY25", "FY26"])], _G["fc"], _G["hubs"], _G["re"].loc[cl],
                     pack_plan=pack_plan.get(cl) if pack_plan else None, scenario=scenario)
    return sim.run(policy, days)


class Runner:
    """Keeps a process pool alive across many policy/scenario runs."""

    def __init__(self, cfg: dict, workers: Optional[int] = None, parallel: bool = True):
        self.cfg = cfg
        self.parallel = parallel
        self.workers = workers or min(5, os.cpu_count() or 1)
        self.pool = ProcessPoolExecutor(max_workers=self.workers, initializer=_init, initargs=(cfg,)) if parallel else None
        if not parallel:
            _init(cfg)

    def run(self, policy: Policy, days: List[pd.Timestamp], scenario: Optional[dict] = None, cfg: Optional[dict] = None,
            pack_plan: Optional[dict] = None, weights: Optional[Dict[pd.Timestamp, float]] = None,
            clusters: Optional[List[str]] = None) -> pd.DataFrame:
        cfg = cfg or self.cfg
        cls = clusters or cfg["clusters"]
        args = [(cfg, cl, policy, days, scenario, pack_plan) for cl in cls]
        if self.parallel:
            outs = list(self.pool.map(_task, args))
        else:
            outs = [_task(a) for a in args]
        df = pd.concat(outs, ignore_index=True)
        if weights:
            dd = pd.to_datetime(df["time"]).dt.normalize()
            df["w"] = dd.map(weights).astype(float).values
        else:
            df["w"] = 1.0
        return df

    def close(self) -> None:
        if self.pool:
            self.pool.shutdown()


def season_of(ts: pd.Timestamp) -> str:
    m = ts.month
    return "summer" if m in (4, 5, 6) else "monsoon" if m in (7, 8, 9) else "post-monsoon" if m in (10, 11, 12) else "winter"


def np_safe(x):
    return float(np.asarray(x))
