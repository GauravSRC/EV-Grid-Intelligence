"""Pipeline stages orchestrated by run_all.py (each writes results/*.json)."""
from __future__ import annotations

import logging

from evgrid.config import RESULTS_DIR, load_config, save_json

log = logging.getLogger(__name__)


def stage_data(force: bool = False) -> None:
    from evgrid.analysis.data_figures import data_overview, data_stats
    from evgrid.data.io import load_or_generate, re_contract_mw

    cfg = load_config()
    data = load_or_generate(cfg, force=force)
    stats = data_stats(data["hourly"])
    stats["re_contracts"] = re_contract_mw(cfg, data["hourly"]).round(2).to_dict(orient="index")
    save_json(stats, RESULTS_DIR / "data_stats.json")
    data_overview(data["hourly"])
    log.info("data stage: %s", {k: stats[k] for k in ("dam_mean_by_fy", "fy26_reference_demand_twh")})


def load_data():
    from evgrid.data.io import load_or_generate

    return load_or_generate(load_config())


def forecast_path():
    from evgrid.config import CACHE_DIR

    return CACHE_DIR / "forecasts_fy26.parquet"


def stage_forecast(mode: str = "quick", force: bool = False) -> None:
    from evgrid.forecasting.figures import forecast_figure
    from evgrid.forecasting.models import run_forecasting

    cfg = load_config()
    data = load_data()
    p = forecast_path()
    if p.exists() and not force and (RESULTS_DIR / "forecast_metrics.json").exists():
        log.info("forecasts cached")
        return
    fc, metrics = run_forecasting(data["hourly"], cfg, quick=(mode == "quick"))
    p.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(p, index=False)
    save_json(metrics, RESULTS_DIR / "forecast_metrics.json")
    forecast_figure(fc, metrics)


def load_forecasts():
    import pandas as pd

    return pd.read_parquet(forecast_path())
