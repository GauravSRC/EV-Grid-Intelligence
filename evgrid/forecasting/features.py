"""Leakage-safe feature engineering for day-ahead (D-1 10:00) forecasts.

Rules enforced here (and unit-tested in tests/test_leakage.py):
- weather enters ONLY through day-ahead forecast columns (``*_fc``);
- target lags are >= 48 h (``LAGS_H``) and rolling statistics end at D-2 23:00;
- calendar/festival/trend features are known in advance.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from evgrid.data.synth import FESTIVALS

WEATHER_FC = ["temp_fc", "cloud_fc", "rain_fc", "ghi_fc", "wind_fc"]
WEATHER_ACTUAL = ["temp", "rh", "cloud", "rain", "ghi", "wind"]
LAGS_H = [48, 168]
ROLL_END_LAG_DAYS = 2  # rolling level uses days D-8 .. D-2
CALENDAR = ["hour", "dow", "month", "doy_sin", "doy_cos", "weekend", "fest_near", "fest_day", "trend"]

# target -> (source column(s), relative (ratio-to-level) model?)
TARGETS: Dict[str, dict] = {
    "dc": dict(cols={"retail": "dc_retail_kwh", "ride": "dc_ride_kwh", "logi": "dc_logi_kwh"}, relative=True),
    "swaps": dict(cols={"s2w": "swaps_s2w", "s3w": "swaps_s3w"}, relative=True),
    "solar": dict(cols={"solar": "solar_cf"}, relative=False),
    "wind": dict(cols={"wind": "wind_cf"}, relative=False),
    "dam": dict(cols={"dam": "dam_price"}, relative=False),
    "spread": dict(cols={"spread": "_spread"}, relative=False),
}


def calendar_features(time: pd.Series) -> pd.DataFrame:
    t = pd.DatetimeIndex(time)
    doy = t.dayofyear.values
    days = t.normalize()
    fest_dates = pd.DatetimeIndex([pd.Timestamp(d) for v in FESTIVALS.values() for d in v])
    dd = np.array([np.min(np.abs((fest_dates - d).days)) for d in days.unique()])
    near = pd.Series(dd, index=days.unique())
    dist = near.reindex(days).values
    return pd.DataFrame(dict(
        hour=t.hour.values, dow=t.dayofweek.values, month=t.month.values,
        doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25),
        weekend=(t.dayofweek.values >= 5).astype(int), fest_near=(dist <= 14).astype(int) * (14 - np.minimum(dist, 14)),
        fest_day=(dist == 0).astype(int), trend=(t - pd.Timestamp("2021-04-01")).days.values / 365.25,
    ))


def series_lag_features(y: pd.Series) -> pd.DataFrame:
    """Lag features for one contiguous hourly series (index = hourly DatetimeIndex)."""
    out = pd.DataFrame(index=y.index)
    for L in LAGS_H:
        out[f"lag{L}"] = y.shift(L).values
    daily = y.resample("D").mean()
    lvl = daily.rolling(7, min_periods=3).mean().shift(ROLL_END_LAG_DAYS)  # mean over D-8..D-2
    out["level"] = lvl.reindex(y.index.normalize()).values
    std = daily.rolling(28, min_periods=7).std().shift(ROLL_END_LAG_DAYS)
    out["level_sd"] = std.reindex(y.index.normalize()).values
    return out


def build_long(hourly: pd.DataFrame, target: str) -> pd.DataFrame:
    """Long frame (time, cluster, series, y, features) for a pooled model."""
    spec = TARGETS[target]
    frames: List[pd.DataFrame] = []
    cal = None
    for cl, g in hourly.groupby("cluster", sort=False):
        g = g.sort_values("time")
        if cal is None:
            cal = calendar_features(g["time"]).reset_index(drop=True)
        for sname, col in spec["cols"].items():
            if col == "_spread":
                yv = g["rtm_price"].values - g["dam_price"].values
            else:
                yv = g[col].values
            y = pd.Series(yv, index=pd.DatetimeIndex(g["time"]))
            lf = series_lag_features(y).reset_index(drop=True)
            f = pd.concat([cal, lf], axis=1)
            for w in WEATHER_FC:
                f[w] = g[w].values
            f["time"] = g["time"].values
            f["cluster"] = cl
            f["series"] = sname
            f["y"] = yv
            frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["cluster_code"] = df["cluster"].astype("category").cat.codes
    df["series_code"] = df["series"].astype("category").cat.codes
    if spec["relative"]:
        lvl = df["level"].clip(lower=1e-6)
        df["lag48"] = df["lag48"] / lvl
        df["lag168"] = df["lag168"] / lvl
        df["level_sd"] = df["level_sd"] / lvl
    return df


def feature_columns(target: str) -> List[str]:
    cols = CALENDAR + WEATHER_FC + [f"lag{L}" for L in LAGS_H] + ["level", "level_sd", "cluster_code", "series_code"]
    return cols
