"""Data schema + adapter for plugging real operator data into the pipeline.

The simulator and forecasting stack only consume the columns below.  To use real
data, map your sources onto this schema with :func:`adapt_real_data` (or write an
equivalent ETL) and save to ``data/generated/hourly.parquet``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HOURLY_SCHEMA: Dict[str, str] = {
    "time": "datetime64[ns], hour-beginning IST",
    "cluster": "str, one of DEL/MUM/BLR/HYD/CHE",
    "fy": "str, fiscal year label FYyy (Apr-Mar)",
    "temp": "degC, actual 2-m air temperature",
    "rh": "%, relative humidity",
    "cloud": "0-1, total cloud fraction",
    "rain": "mm/h, precipitation",
    "ghi": "W/m2, global horizontal irradiance",
    "wind": "m/s, hub-height wind speed at contracted wind site",
    "temp_fc": "degC, day-ahead (D-1 10:00) forecast of temp",
    "cloud_fc": "0-1, day-ahead forecast of cloud",
    "rain_fc": "mm/h, day-ahead forecast of rain",
    "ghi_fc": "W/m2, day-ahead forecast of GHI",
    "wind_fc": "m/s, day-ahead forecast of wind",
    "solar_cf": "0-1, metered solar capacity factor of contracted plants",
    "wind_cf": "0-1, metered wind capacity factor of contracted plants",
    "dam_price": "INR/kWh, IEX DAM area clearing price (hourly avg of 15-min blocks)",
    "rtm_price": "INR/kWh, IEX RTM price (hourly avg of half-hourly auctions)",
    "gdam_price": "INR/kWh, IEX G-DAM price",
    "dc_retail_kwh": "kWh, DC fast-charging demand requested by retail drivers (sessions incl. lost)",
    "dc_ride_kwh": "kWh, DC demand of ride-hailing fleets",
    "dc_logi_kwh": "kWh, DC demand of logistics fleets",
    "swaps_s2w": "count, 2-wheeler swap requests (served + stock-out)",
    "swaps_s3w": "count, 3-wheeler swap requests",
    "dc_avail": "0-1, share of DC charger capacity available (OCPP status)",
}

HUB_SCHEMA: Dict[str, str] = {
    "cluster": "str", "hub": "str", "archetype": "CBD/RES/ITP/HWY",
    "dc_demand_weight": "share of cluster DC demand (base)", "swap_demand_weight": "share of cluster swap demand",
    "dc_cap_share": "share of cluster DC capacity", "pack_share": "share of cluster pack inventory",
}


def validate(df: pd.DataFrame, schema: Optional[Dict[str, str]] = None) -> List[str]:
    """Return a list of problems (empty if the frame satisfies the schema)."""
    schema = schema or HOURLY_SCHEMA
    issues = [f"missing column {c}" for c in schema if c not in df.columns]
    if "time" in df.columns and "cluster" in df.columns:
        if df.duplicated(["time", "cluster"]).any():
            issues.append("duplicate (time, cluster) rows")
    for c in ("dam_price", "rtm_price"):
        if c in df.columns and (df[c] > 10.0 + 1e-6).any():
            issues.append(f"{c} above IEX cap 10 INR/kWh")
    return issues


def adapt_real_data(sessions: pd.DataFrame, swaps: pd.DataFrame, weather: pd.DataFrame, prices: pd.DataFrame,
                    re_meter: pd.DataFrame, availability: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Adapter from typical operator extracts to the hourly schema.

    Parameters (minimal expected columns)
    - sessions: [start_time, cluster, segment in {retail, ride, logi}, kwh_requested]
    - swaps: [time, cluster, vehicle in {s2w, s3w}]   (one row per swap request incl. stock-outs)
    - weather: [time, cluster, temp, rh, cloud, rain, ghi, wind, *_fc]
    - prices: [time, region, dam_price, rtm_price, gdam_price] (15-min/30-min blocks are averaged to hourly)
    - re_meter: [time, cluster, solar_cf, wind_cf]
    """
    s = sessions.assign(time=pd.to_datetime(sessions["start_time"]).dt.floor("h"))
    dc = s.pivot_table(index=["time", "cluster"], columns="segment", values="kwh_requested", aggfunc="sum").fillna(0.0)
    dc.columns = [f"dc_{c}_kwh" for c in dc.columns]
    sw = swaps.assign(time=pd.to_datetime(swaps["time"]).dt.floor("h"), n=1.0)
    sw = sw.pivot_table(index=["time", "cluster"], columns="vehicle", values="n", aggfunc="sum").fillna(0.0)
    sw.columns = [f"swaps_{c}" for c in sw.columns]
    px = prices.assign(time=pd.to_datetime(prices["time"]).dt.floor("h")).groupby(["time", "region"]).mean(numeric_only=True)
    out = dc.join(sw, how="outer").reset_index()
    out = out.merge(weather, on=["time", "cluster"], how="left").merge(re_meter, on=["time", "cluster"], how="left")
    region_map = {"DEL": "N", "MUM": "W", "BLR": "S", "HYD": "S", "CHE": "S"}
    out["region"] = out["cluster"].map(region_map)
    out = out.merge(px.reset_index(), on=["time", "region"], how="left").drop(columns="region")
    out["dc_avail"] = 1.0 if availability is None else out.merge(availability, on=["time", "cluster"], how="left")["dc_avail"].values
    t = out["time"]
    out["fy"] = ["FY" + str(y + (m >= 4))[-2:] for y, m in zip(t.dt.year, t.dt.month)]
    return out.fillna(0.0)


def demand_columns() -> List[str]:
    return ["dc_retail_kwh", "dc_ride_kwh", "dc_logi_kwh", "swaps_s2w", "swaps_s3w"]


def summary_stats(df: pd.DataFrame) -> Dict[str, float]:
    return {c: float(np.nanmean(df[c])) for c in df.columns if df[c].dtype.kind in "fi"}
