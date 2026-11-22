"""Pooled HistGradientBoosting forecasters with split-conformal P10/P90 by hour band.

Protocol: fit on years 1-3 -> conformal residual quantiles on year 4 (by series x 4-hour band) ->
refit P50 on years 1-4 -> evaluate on year 5 (FY26, the simulated year).  Baseline forecaster:
seasonal naive (same hour last week, lag 168 h).
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from evgrid.forecasting.features import TARGETS, build_long, feature_columns

log = logging.getLogger(__name__)
Q_LO, Q_HI = 0.10, 0.90
Z90 = 1.2815515655446004


def _model(quick: bool) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=150 if quick else 300, learning_rate=0.1, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=0)


def wape(y: np.ndarray, f: np.ndarray) -> float:
    return float(np.sum(np.abs(y - f)) / max(np.sum(np.abs(y)), 1e-9))


def pinball(y: np.ndarray, f: np.ndarray, q: float) -> float:
    d = y - f
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def fit_target(hourly: pd.DataFrame, target: str, cal_start: str, test_start: str, quick: bool) -> Tuple[pd.DataFrame, dict]:
    """Fit one pooled target. Returns (test-year forecasts frame, metrics)."""
    spec = TARGETS[target]
    rel = spec["relative"]
    df = build_long(hourly, target)
    feats = feature_columns(target)
    df = df.dropna(subset=["lag168", "level"]).reset_index(drop=True)
    ty = df["y"] / df["level"].clip(lower=1e-6) if rel else df["y"]
    t = df["time"]
    tr1 = t < pd.Timestamp(cal_start)
    cal = (t >= pd.Timestamp(cal_start)) & (t < pd.Timestamp(test_start))
    tr2 = t < pd.Timestamp(test_start)
    te = t >= pd.Timestamp(test_start)
    if quick:  # subsample training rows for speed (keeps all calibration/test rows)
        rng = np.random.default_rng(0)
        keep = rng.random(len(df)) < 0.5
        tr1 = tr1 & keep
        tr2 = tr2 & (keep | cal)
    X = df[feats].values.astype(float)
    cat_idx = [feats.index("cluster_code"), feats.index("series_code")]
    m1 = _model(quick).set_params(categorical_features=cat_idx).fit(X[tr1], ty[tr1])
    pred_cal = m1.predict(X[cal])
    band = (df["hour"] // 4).values
    key = df["series"].astype(str).values + "_" + df["cluster"].astype(str).values + "_" + band.astype(str)
    if rel:
        res = ty[cal].values / np.clip(pred_cal, 1e-6, None)  # multiplicative residuals
    else:
        res = ty[cal].values - pred_cal
    rq = pd.DataFrame(dict(k=key[cal], r=res)).groupby("k")["r"].quantile([Q_LO, Q_HI]).unstack()
    m2 = _model(quick).set_params(categorical_features=cat_idx).fit(X[tr2], ty[tr2])
    p50 = m2.predict(X[te])
    lo = rq.reindex(key[te])[Q_LO].fillna(rq[Q_LO].median()).values
    hi = rq.reindex(key[te])[Q_HI].fillna(rq[Q_HI].median()).values
    if rel:
        lvl = df.loc[te, "level"].values
        f50 = np.clip(p50, 0, None) * lvl
        f10 = np.clip(p50 * lo, 0, None) * lvl
        f90 = np.clip(p50 * hi, 0, None) * lvl
        naive = df.loc[te, "lag168"].values * lvl
    else:
        f50, f10, f90 = p50, p50 + lo, p50 + hi
        naive = df.loc[te, "lag168"].values
    if target in ("solar", "wind"):
        f10, f50, f90 = (np.clip(a, 0, 1) for a in (f10, f50, f90))
        night = df.loc[te, "ghi_fc"].values <= 1.0
        if target == "solar":
            f10[night] = f50[night] = f90[night] = 0.0
    if target == "dam":
        f10, f50, f90 = (np.clip(a, 0.5, 10.0) for a in (f10, f50, f90))
    f10 = np.minimum(f10, f50)
    f90 = np.maximum(f90, f50)
    y = df.loc[te, "y"].values
    out = pd.DataFrame(dict(time=df.loc[te, "time"].values, cluster=df.loc[te, "cluster"].values,
                            series=df.loc[te, "series"].values, y=y, p10=f10, p50=f50, p90=f90, naive=naive))
    metrics = dict(
        wape_ml=wape(y, f50), wape_naive=wape(y, naive),
        pinball_p10=pinball(y, f10, 0.1), pinball_p50=pinball(y, f50, 0.5), pinball_p90=pinball(y, f90, 0.9),
        pinball_naive_p50=pinball(y, naive, 0.5),
        coverage_p10_p90=float(np.mean((y >= f10) & (y <= f90))), n_test=int(len(y)),
    )
    log.info("forecast %-6s WAPE ML %.3f vs naive %.3f | coverage %.2f", target, metrics["wape_ml"], metrics["wape_naive"],
             metrics["coverage_p10_p90"])
    return out, metrics


def run_forecasting(hourly: pd.DataFrame, cfg: dict, quick: bool = True) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    """Fit all targets; return wide FY26 forecast table (time, cluster, <var>_p10/p50/p90/naive) + metrics."""
    test_start = cfg["meta"]["sim_start"]
    cal_start = "2024-04-01"
    wide = None
    metrics: Dict[str, dict] = {}
    for target in TARGETS:
        out, met = fit_target(hourly, target, cal_start, test_start, quick)
        metrics[target] = met
        for s, g in out.groupby("series"):
            g = g.rename(columns={c: f"{s}_{c}" for c in ["y", "p10", "p50", "p90", "naive"]}).drop(columns="series")
            wide = g if wide is None else wide.merge(g, on=["time", "cluster"], how="outer")
    wide = wide.sort_values(["cluster", "time"]).reset_index(drop=True)
    return wide, metrics


def price_quantile(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray, q: float) -> np.ndarray:
    """Interpolate an intermediate quantile from (P10, P50, P90) with a split-normal assumption."""
    from scipy.stats import norm

    z = norm.ppf(q)
    sd_lo = (p50 - p10) / Z90
    sd_hi = (p90 - p50) / Z90
    return np.where(z < 0, p50 + z * sd_lo, p50 + z * sd_hi)
