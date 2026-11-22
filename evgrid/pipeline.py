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


def ai_extra_opex(cfg: dict, redeploy_cost: dict) -> dict:
    """AI-only opex: programme run cost (allocated by capacity) + pack redeployment transport."""
    sh = cfg["network"]["capacity_share"]
    return {c: cfg["economics"]["ai_run_cr_yr"] * sh[c] + redeploy_cost.get(c, 0.0) for c in cfg["clusters"]}


def stage_calibrate(mode: str = "quick", force: bool = False) -> None:
    from evgrid.config import CONFIG_DIR
    from evgrid.simulation.calibrate import calibrate
    from evgrid.simulation.runner import Runner

    if (CONFIG_DIR / "calibrated.yaml").exists() and not force:
        log.info("calibrated.yaml exists and is frozen - skipping (use --force to recalibrate)")
        return
    r = Runner(load_config(calibrated=False))
    try:
        doc = calibrate(r)
    finally:
        r.close()
    save_json(doc, RESULTS_DIR / "calibration.json")


def pack_plan(cfg: dict):
    from evgrid.data.synth import cluster_capacities
    from evgrid.optimization.strategic import monthly_pack_plan

    data = load_data()
    return monthly_pack_plan(cfg, data["hourly"], cluster_capacities(cfg))


def stage_simulate(mode: str = "quick") -> None:
    """Baseline vs AI on identical data/seeds (full year in --full, representative weeks in --quick)."""
    from evgrid.config import CACHE_DIR
    from evgrid.simulation.kpis import fixed_opex_alloc, kpi_table
    from evgrid.simulation.policies import AI, BASELINE
    from evgrid.simulation.runner import Runner, full_year_days, rep_week_days

    cfg = load_config()
    plan, rcost, plan_log = pack_plan(cfg)
    if mode == "full":
        days, w = full_year_days(cfg), None
    else:
        days, w = rep_week_days()
    r = Runner(cfg)
    out = {"mode": mode, "period": "FY2025-26 full year" if mode == "full" else "8 representative weeks, season-weighted"}
    try:
        for pol in (BASELINE, AI):
            df = r.run(pol, days, pack_plan=plan, weights=w)
            extra = ai_extra_opex(cfg, rcost) if pol.name == "AI" else None
            out[pol.name] = kpi_table(df, cfg, fixed_opex_alloc(cfg), extra, monthly=(mode == "full"))
            df.to_parquet(CACHE_DIR / f"sim_{pol.name}_{mode}.parquet", index=False)
            log.info("%s: R %.4f loss %.4f EBITDA %.4f", pol.name, out[pol.name]["overall"]["R"],
                     out[pol.name]["overall"]["loss_pct"], out[pol.name]["overall"]["ebitda_margin"])
    finally:
        r.close()
    out["targets"] = {"R": 0.897, "loss_pct": 0.08, "ebitda_margin": 0.12}
    out["redeployment"] = {"cost_cr": rcost, "log": plan_log}
    save_json(out, RESULTS_DIR / f"kpis_main_{mode}.json")


def stage_analysis(mode: str = "quick") -> None:
    from evgrid.analysis.experiments import economics, run_all_experiments, scalability
    from evgrid.analysis.figures import make_all_figures
    from evgrid.config import load_json

    out = run_all_experiments(mode)
    sc = scalability(load_config())
    save_json(sc, RESULTS_DIR / "scalability.json")
    main_p = RESULTS_DIR / "kpis_main_full.json"
    main = load_json(main_p if main_p.exists() else RESULTS_DIR / "kpis_main_quick.json")
    gaps = load_json(RESULTS_DIR / "gap_levers.json")
    eco = economics(main, gaps, out["scenarios"], load_config())
    save_json(eco, RESULTS_DIR / "economics.json")
    make_all_figures()
