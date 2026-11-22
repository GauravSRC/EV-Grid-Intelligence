"""Calibrated synthetic "digital twin" dataset (company data was not provided).

Generates 5 years of hourly IST data (FY22-FY26) for the 5 metro clusters:
weather actuals + day-ahead weather forecasts, solar/wind capacity factors,
regional IEX DAM / RTM / G-DAM prices, DC charging demand by segment, swap
counts by vehicle class, charger availability, plus static hub tables and
monthly battery state-of-health records.

Everything is driven by ``config/base.yaml`` and a fixed seed.  The pipeline is
data-agnostic: real data in the schema of :mod:`evgrid.data.schema` can replace
this generator.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MONTHS = np.arange(12)
# ---------------------------------------------------------------------------
# Climatology (Jan..Dec).  Sources: IMD climatological normals (rounded).
# ---------------------------------------------------------------------------
CLIMATE = {
    "DEL": dict(lat=28.6, lon=77.2,
                t=[14, 17, 23, 29, 33, 33, 31, 30, 29, 26, 20, 15],
                tamp=[6.5, 7, 7.5, 8, 7.5, 6, 4, 3.5, 4.5, 7, 7.5, 7],
                cloud=[.35, .30, .25, .20, .25, .45, .70, .70, .50, .20, .20, .32],
                wind=[4.6, 5.0, 5.5, 6.0, 7.0, 7.5, 7.0, 6.5, 5.5, 4.6, 4.4, 4.5],
                rain_mult=1.0, fog=True),
    "MUM": dict(lat=19.1, lon=72.9,
                t=[24, 25, 27, 29, 30, 29, 28, 27, 28, 29, 28, 26],
                tamp=[4.5, 4.5, 4, 3.5, 3, 2.5, 2, 2, 2.5, 3.5, 4.5, 4.5],
                cloud=[.15, .15, .15, .15, .30, .75, .90, .88, .70, .35, .20, .15],
                wind=[5.0, 5.0, 5.0, 5.5, 7.0, 8.5, 9.0, 8.5, 7.0, 5.0, 4.8, 5.0],
                rain_mult=2.2, fog=False),
    "BLR": dict(lat=12.97, lon=77.6,
                t=[21, 23, 26, 28, 27, 24, 23, 23, 24, 23, 22, 21],
                tamp=[5.5, 6, 6, 5.5, 5, 3.5, 3, 3, 3.5, 4, 4.5, 5],
                cloud=[.20, .15, .20, .35, .45, .65, .75, .75, .65, .55, .45, .30],
                wind=[5.0, 4.8, 4.8, 5.0, 7.0, 9.0, 9.5, 9.0, 7.5, 5.0, 5.0, 5.2],
                rain_mult=1.0, fog=False),
    "HYD": dict(lat=17.4, lon=78.5,
                t=[22, 25, 28, 32, 33, 29, 27, 26, 26, 25, 23, 21],
                tamp=[6, 6.5, 7, 6.5, 6, 4.5, 3.5, 3.5, 4, 5, 5.5, 6],
                cloud=[.15, .15, .20, .25, .30, .60, .75, .75, .65, .40, .25, .20],
                wind=[5.0, 5.0, 5.0, 5.5, 7.0, 8.5, 9.0, 8.5, 7.0, 5.0, 5.0, 5.0],
                rain_mult=1.0, fog=False),
    "CHE": dict(lat=13.08, lon=80.27,
                t=[25, 26, 28, 31, 33, 32, 31, 30, 30, 28, 26, 25],
                tamp=[4, 4.5, 4.5, 4, 4.5, 4.5, 4, 4, 4, 3.5, 3, 3.5],
                cloud=[.35, .25, .20, .20, .30, .45, .50, .50, .50, .70, .80, .65],
                wind=[5.5, 5.0, 4.8, 5.0, 7.5, 9.5, 10.0, 9.5, 8.0, 5.5, 5.5, 6.0],
                rain_mult=1.3, fog=False),
}

# Festival calendar (dates, effect type).  Sources: national holiday calendars.
FESTIVALS = {
    "diwali": ["2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20"],
    "holi": ["2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04"],
    "ganesh": ["2021-09-10", "2022-08-31", "2023-09-19", "2024-09-07", "2025-08-27"],
    "pongal": ["2022-01-14", "2023-01-15", "2024-01-15", "2025-01-14", "2026-01-15"],
    "dussehra": ["2021-10-15", "2022-10-05", "2023-10-24", "2024-10-12", "2025-10-02"],
}

SEGMENTS = ["retail", "ride", "logi"]
SWAPS = ["s2w", "s3w"]


def _gauss_bump(h: np.ndarray, centre: float, width: float) -> np.ndarray:
    d = np.minimum(np.abs(h - centre), 24 - np.abs(h - centre))
    return np.exp(-0.5 * (d / width) ** 2)


def diurnal_profile(kind: str, hours: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Hour-of-day demand shapes (mean ~1) by segment."""
    h = hours.astype(float)
    if kind == "retail":
        p = 0.25 + 0.9 * _gauss_bump(h, 9, 1.6) + 1.5 * _gauss_bump(h, 19.5, 2.2) + 0.6 * _gauss_bump(h, 14, 2.5)
        p = p * np.where(weekend, 1.10, 1.0)
    elif kind == "ride":
        p = 0.35 + 1.1 * _gauss_bump(h, 7, 1.5) + 0.9 * _gauss_bump(h, 15, 1.8) + 1.3 * _gauss_bump(h, 23.5, 2.0)
        p = p * np.where(weekend, 0.95, 1.0)
    elif kind == "logi":
        p = 0.30 + 1.6 * _gauss_bump(h, 2, 2.6) + 0.8 * _gauss_bump(h, 14, 1.8)
        p = p * np.where(weekend, 0.80, 1.0)
    elif kind == "s2w":  # delivery riders: lunch & dinner peaks
        p = 0.15 + 0.8 * _gauss_bump(h, 9, 1.5) + 1.5 * _gauss_bump(h, 13, 1.5) + 1.8 * _gauss_bump(h, 20, 1.8)
        p = p * np.where(weekend, 1.10, 1.0)
    elif kind == "s3w":  # e-rickshaw morning/evening
        p = 0.15 + 1.6 * _gauss_bump(h, 7.5, 1.6) + 0.6 * _gauss_bump(h, 13, 2) + 1.5 * _gauss_bump(h, 18.5, 1.6)
    else:
        raise ValueError(kind)
    return p


def archetype_profile(arch: str, hours: np.ndarray, weekend: np.ndarray) -> np.ndarray:
    """Relative demand intensity of a hub archetype by hour (for hub demand shares)."""
    h = hours.astype(float)
    if arch == "CBD":
        p = 0.3 + 1.4 * _gauss_bump(h, 13, 4.0)
        return p * np.where(weekend, 0.7, 1.0)
    if arch == "RES":
        return 0.5 + 1.2 * _gauss_bump(h, 21, 3.0) + 0.5 * _gauss_bump(h, 7, 2.0)
    if arch == "ITP":
        p = 0.25 + 1.5 * _gauss_bump(h, 11, 3.0) + 0.6 * _gauss_bump(h, 18, 2.0)
        return p * np.where(weekend, 0.45, 1.0)
    if arch == "HWY":
        return 0.6 + 1.2 * _gauss_bump(h, 2, 3.5)
    raise ValueError(arch)


def _ar1(rng: np.random.Generator, n: int, phi: float, sd: float) -> np.ndarray:
    """Stationary AR(1) with marginal sd ``sd``."""
    from scipy.signal import lfilter

    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    e[0] = rng.normal(0, sd)
    return lfilter([1.0], [1.0, -phi], e)


def _interp_monthly(values, doy: np.ndarray) -> np.ndarray:
    """Smooth monthly normals -> daily values via periodic interpolation (mid-month knots)."""
    knots = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=float)
    v = np.asarray(values, dtype=float)
    kx = np.concatenate([knots - 365, knots, knots + 365])
    kv = np.concatenate([v, v, v])
    return np.interp(doy, kx, kv)


def fiscal_year(ts: pd.DatetimeIndex) -> np.ndarray:
    y = ts.year + (ts.month >= 4).astype(int)
    return np.array([f"FY{str(v)[-2:]}" for v in y])


def solar_geometry(ts: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Sine of solar elevation at the middle of each IST hour."""
    doy = ts.dayofyear.values
    decl = np.deg2rad(23.45) * np.sin(2 * np.pi * (284 + doy) / 365)
    solar_time = ts.hour.values + 0.5 + (lon - 82.5) / 15.0
    omega = np.deg2rad(15 * (solar_time - 12))
    phi = np.deg2rad(lat)
    s = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(omega)
    return np.clip(s, 0, None)


def wind_power_curve(v: np.ndarray, cut_in: float = 3.0, rated: float = 12.0, cut_out: float = 25.0) -> np.ndarray:
    out = np.clip((v ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3), 0, 1)
    out[(v < cut_in) | (v >= cut_out)] = 0.0
    return out


def _weather(ts: pd.DatetimeIndex, cl: str, rng: np.random.Generator) -> pd.DataFrame:
    c = CLIMATE[cl]
    n = len(ts)
    days = ts.normalize()
    day_idx = pd.factorize(days)[0]
    nd = day_idx.max() + 1
    doy = ts.dayofyear.values.astype(float)
    hour = ts.hour.values
    month = ts.month.values

    # --- temperature -------------------------------------------------------
    tmean = _interp_monthly(c["t"], doy)
    tamp = _interp_monthly(c["tamp"], doy)
    t_anom_d = _ar1(rng, nd, 0.75, 1.6)
    heat = np.zeros(nd)
    day_dates = pd.DatetimeIndex(days.unique())
    hw_prone = 1.0 if cl in ("DEL", "HYD") else 0.5
    for yr in range(ts.year.min(), ts.year.max() + 1):
        for _ in range(rng.poisson(2.5 * hw_prone)):
            start = pd.Timestamp(f"{yr}-04-01") + pd.Timedelta(days=int(rng.integers(0, 85)))
            dur = int(rng.integers(3, 9))
            mask = (day_dates >= start) & (day_dates < start + pd.Timedelta(days=dur))
            heat[mask] += rng.uniform(3.5, 6.0)
    t_anom = (t_anom_d + heat)[day_idx] + _ar1(rng, n, 0.9, 0.6)
    temp = tmean + tamp * np.sin(2 * np.pi * (hour - 9) / 24) + t_anom

    # --- cloud (logit AR) -----------------------------------------------------
    cl_norm = np.clip(_interp_monthly(c["cloud"], doy), 0.03, 0.97)
    logit = np.log(cl_norm / (1 - cl_norm))
    lg_d = _ar1(rng, nd, 0.6, 0.7)
    brk = np.zeros(nd)
    for yr in range(ts.year.min(), ts.year.max() + 1):  # monsoon breaks
        for _ in range(rng.integers(1, 3)):
            start = pd.Timestamp(f"{yr}-07-01") + pd.Timedelta(days=int(rng.integers(0, 50)))
            dur = int(rng.integers(6, 13))
            mask = (day_dates >= start) & (day_dates < start + pd.Timedelta(days=dur))
            brk[mask] -= 1.6
    is_monsoon = np.isin(month, [6, 7, 8, 9]) if cl != "CHE" else np.isin(month, [10, 11, 12])
    lg = logit + lg_d[day_idx] + brk[day_idx] * is_monsoon + _ar1(rng, n, 0.85, 0.45)
    cloud = 1 / (1 + np.exp(-lg))
    temp = temp - 3.0 * (cloud - cl_norm) * (np.sin(2 * np.pi * (hour - 9) / 24) > 0)

    # --- rain ---------------------------------------------------------------
    p_rain = np.clip((cloud - 0.55) * 1.2, 0, 0.9)
    rain = (rng.random(n) < p_rain) * rng.exponential(2.0 * c["rain_mult"], n)

    # --- RH -----------------------------------------------------------------
    rh = np.clip(40 + 50 * cloud - 1.2 * (temp - 25) + rng.normal(0, 5, n) + 8 * (rain > 0), 10, 100)

    # --- irradiance -----------------------------------------------------------
    s = solar_geometry(ts, c["lat"], c["lon"])
    ghi_cs = 1050 * s ** 1.2
    fog = np.zeros(n)
    if c["fog"]:
        fog_day = rng.random(nd) < 0.35
        fog = (fog_day[day_idx] & np.isin(month, [12, 1]) & (hour >= 6) & (hour <= 11)).astype(float)
    ghi = ghi_cs * (1 - 0.75 * cloud ** 3.4) * (1 - 0.55 * fog)

    # --- wind (Gaussian copula -> Weibull k=2) --------------------------------
    wc = _interp_monthly(c["wind"], doy)
    z = 0.40 * _ar1(rng, nd, 0.5, 1.0)[day_idx] + 0.9165 * _ar1(rng, n, 0.85, 1.0)
    from scipy.stats import norm

    u = norm.cdf(z)
    diurnal_w = 1 + 0.12 * np.sin(2 * np.pi * (hour - 11) / 24)
    wind = wc * diurnal_w * np.sqrt(-np.log(np.clip(1 - u, 1e-9, 1)))  # Weibull k=2 with scale wc

    # --- day-ahead forecasts (lead-time noise; models only see these) ---------
    temp_fc = temp + _ar1(rng, n, 0.9, 1.2)
    cloud_fc = np.clip(cloud + _ar1(rng, n, 0.9, 0.12), 0, 1)
    ghi_fc = ghi_cs * (1 - 0.75 * cloud_fc ** 3.4) * (1 - 0.45 * fog)
    wind_fc = np.clip(wind + _ar1(rng, n, 0.9, 1.3), 0, None)
    rain_fc = np.clip(rain * rng.lognormal(0, 0.6, n) + (rng.random(n) < 0.05) * (cloud_fc > 0.6) * 1.0, 0, None)
    return pd.DataFrame(dict(temp=temp, rh=rh, cloud=cloud, rain=rain, ghi=ghi, wind=wind,
                             temp_fc=temp_fc, cloud_fc=cloud_fc, rain_fc=rain_fc, ghi_fc=ghi_fc,
                             wind_fc=wind_fc, ghi_cs=ghi_cs))


def solar_cf_from(ghi: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """PV AC capacity factor (unscaled): irradiance x temperature derate x PR, DC/AC clipping."""
    t_cell = temp + 0.03 * ghi
    p = 1.3 * (ghi / 1000.0) * (1 - 0.004 * (t_cell - 25)) * 0.80
    return np.clip(p, 0, 1)


def _prices(ts: pd.DatetimeIndex, wx: Dict[str, pd.DataFrame], cfg: dict, rng: np.random.Generator) -> Dict[str, pd.DataFrame]:
    """Regional DAM (duck curve, seasonality, temperature & RE dependence, regime jumps, cap), RTM, G-DAM."""
    n = len(ts)
    hour = ts.hour.values.astype(float)
    month = ts.month.values
    cap = cfg["grid"]["iex_cap"]
    days = ts.normalize()
    day_idx = pd.factorize(days)[0]
    nd = day_idx.max() + 1
    t_anom = np.mean([w["temp"].values - pd.Series(w["temp"].values).rolling(24 * 30, center=True, min_periods=1).mean().values
                      for w in wx.values()], axis=0)
    sol = np.mean([w["ghi"].values for w in wx.values()], axis=0) / 1000.0
    season_solar_depth = _interp_monthly([.50, .58, .62, .60, .52, .35, .25, .25, .35, .50, .52, .48], ts.dayofyear.values)
    season_level = _interp_monthly([0.95, 0.92, 1.00, 1.18, 1.22, 1.08, 0.92, 0.88, 0.95, 1.05, 0.95, 0.95], ts.dayofyear.values)
    eve = 0.75 * _gauss_bump(hour, 20, 2.2)
    night = 0.45 * ((hour <= 5) | (hour >= 23))
    summer = np.isin(month, [4, 5, 6])
    shape = 1 + eve + night * (0.6 + 0.8 * summer) - season_solar_depth * np.clip(sol / 0.85, 0, 1.2)
    # scarcity regime: hot days push evening/night to the cap
    hot_day = pd.Series(t_anom).groupby(day_idx).transform("mean").values
    regime = (rng.random(nd) < 0.04)[day_idx] | (hot_day > 2.0)
    shape = shape * (1 + 0.035 * np.clip(t_anom, -5, 8))
    shape = shape * np.where(regime & ((hour >= 18) | (hour <= 5)), 1.9, 1.0)
    noise = np.exp(_ar1(rng, n, 0.9, 0.12) + _ar1(rng, nd, 0.8, 0.10)[day_idx])
    base = shape * season_level * noise
    fy = fiscal_year(ts)
    dam = np.empty(n)
    for f, target in cfg["prices"]["fy_dam_mean"].items():
        m = fy == f
        if not m.any():
            continue
        x = base[m]
        scale = target / x.mean()
        for _ in range(8):  # rescale after capping/flooring
            y = np.clip(x * scale, 0.8, cap)
            scale *= target / y.mean()
        dam[m] = np.clip(x * scale, 0.8, cap)
    out = {}
    for cl in cfg["clusters"]:
        reg = cfg["price_region"][cl]
        r = dam * (1 + cfg["prices"]["region_offset"][reg]) * np.exp(rng.normal(0, 0.03, n))
        r = np.clip(r, 0.5, cap)
        spread = (0.25 + 1.1 * _gauss_bump(hour, 20, 1.8)) * rng.gamma(1.5, 0.7, n) - 0.4 + _ar1(rng, n, 0.7, 0.55)
        spread = np.clip(spread, -2.5, cfg["prices"]["rtm_evening_spread_max"])
        rtm = np.clip(r + spread, 0.5, cap)
        gdam = np.clip(r * rng.uniform(0.72, 0.95, n) * (1 - 0.1 * (sol > 0.3)), 0.5, cap)
        out[cl] = pd.DataFrame(dict(dam_price=r, rtm_price=rtm, gdam_price=gdam))
    return out


def _festival_factors(ts: pd.DatetimeIndex, cl: str) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (retail_factor, fleet_factor) multiplicative festival effects."""
    d = ts.normalize()
    retail = np.ones(len(ts))
    fleet = np.ones(len(ts))
    for date in FESTIVALS["diwali"]:
        t0 = pd.Timestamp(date)
        pre = (d >= t0 - pd.Timedelta(days=14)) & (d < t0)
        retail[pre] *= 1.12
        fleet[pre] *= 1.20
        day = (d >= t0) & (d <= t0 + pd.Timedelta(days=1))
        retail[day] *= 0.85
        fleet[day] *= 0.65
    for date in FESTIVALS["holi"]:
        day = d == pd.Timestamp(date)
        retail[day] *= 0.70
        fleet[day] *= 0.60
    for date in FESTIVALS["dussehra"]:
        day = d == pd.Timestamp(date)
        retail[day] *= 1.10
        fleet[day] *= 0.85
    if cl == "MUM":
        for date in FESTIVALS["ganesh"]:
            t0 = pd.Timestamp(date)
            win = (d >= t0) & (d <= t0 + pd.Timedelta(days=10))
            retail[win] *= 1.15
            fleet[win] *= 1.08
    if cl == "CHE":
        for date in FESTIVALS["pongal"]:
            t0 = pd.Timestamp(date)
            win = (d >= t0 - pd.Timedelta(days=1)) & (d <= t0 + pd.Timedelta(days=2))
            retail[win] *= 0.85
            fleet[win] *= 0.70
    return retail, fleet


def build_hubs(cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Static hub table: archetype, demand weights and (misaligned) capacity / pack shares."""
    rows = []
    net = cfg["network"]
    for cl in cfg["clusters"]:
        archs = [a for a, k in net["hub_archetypes"].items() for _ in range(k)]
        H = len(archs)
        w_dem = rng.dirichlet(np.full(H, net["hub_demand_dirichlet_alpha"]))
        w_swap = rng.dirichlet(np.full(H, net["hub_demand_dirichlet_alpha"]))
        k_cap = rng.dirichlet(np.full(H, net["hub_capacity_dirichlet_alpha"]))
        k_pack = rng.dirichlet(np.full(H, net["hub_capacity_dirichlet_alpha"]))
        for h in range(H):
            rows.append(dict(cluster=cl, hub=f"{cl}-H{h:02d}", archetype=archs[h], dc_demand_weight=w_dem[h],
                             swap_demand_weight=w_swap[h], dc_cap_share=k_cap[h], pack_share=k_pack[h]))
    return pd.DataFrame(rows)


def cluster_capacities(cfg: dict) -> pd.DataFrame:
    """Derived installed capacities per cluster (MW / MWh)."""
    net, st = cfg["network"], cfg["storage"]
    rows = []
    for cl in cfg["clusters"]:
        tot = net["total_capacity_mw"] * net["capacity_share"][cl]
        swap_mw = tot * net["swap_share"][cl]
        rows.append(dict(cluster=cl, total_mw=tot, dc_mw=tot - swap_mw, swap_mw=swap_mw,
                         bess_mwh=st["bess_mwh_total"] * st["bess_share"][cl],
                         pack_mwh=st["pack_mwh_total"] * st["pack_share"][cl],
                         grid_mw=tot * net["sanctioned_ratio"][cl] * net["sanctioned_scale"]))
    return pd.DataFrame(rows).set_index("cluster")


def generate(cfg: dict, seed: int = 42) -> Dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset. Returns dict of DataFrames."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(cfg["meta"]["start"], cfg["meta"]["end"], freq="h")
    n = len(ts)
    log.info("Generating %d hours x %d clusters", n, len(cfg["clusters"]))
    hour = ts.hour.values
    weekend = ts.dayofweek.values >= 5
    fy = fiscal_year(ts)
    # years relative to FY26 mid-point (growth ~20%/yr)
    t_years = (ts - pd.Timestamp("2025-10-01")).days.values / 365.25
    growth = (1 + cfg["demand"]["growth_yoy"]) ** t_years
    caps = cluster_capacities(cfg)

    wx = {cl: _weather(ts, cl, rng) for cl in cfg["clusters"]}
    prices = _prices(ts, wx, cfg, rng)

    # national wind speed multiplier so that the average wind CUF hits the target
    frames = []
    for cl in cfg["clusters"]:
        w = wx[cl]
        sol_raw = solar_cf_from(w["ghi"].values, w["temp"].values)
        sol = np.clip(sol_raw * cfg["re"]["solar_cuf"] / sol_raw.mean(), 0, 1)
        lo, hi = 0.5, 2.0
        for _ in range(18):
            mid = 0.5 * (lo + hi)
            cf = wind_power_curve(w["wind"].values * mid).mean()
            lo, hi = (mid, hi) if cf < cfg["re"]["wind_cuf"] else (lo, mid)
        wmult = 0.5 * (lo + hi)
        wcf = wind_power_curve(w["wind"].values * wmult)

        # ---------------- demand ------------------------------------------------
        retail_f, fleet_f = _festival_factors(ts, cl)
        temp_eff = 1 + cfg["demand"]["temp_slope"] * np.clip(w["temp"].values - cfg["demand"]["temp_threshold_c"], 0, None)
        heavy_rain = w["rain"].values > 4
        month = ts.month.values
        seas = _interp_monthly([1.0, 1.0, 1.02, 1.03, 1.02, 0.98, 0.96, 0.96, 0.99, 1.02, 1.03, 1.0], ts.dayofyear.values)
        dc_mw = caps.loc[cl, "dc_mw"]
        ref_util = cfg["demand"]["dc_ref_util"][cl] * cfg["demand"]["demand_scale"]
        mix = cfg["customers"]["dc_segment_mix"]
        cols = {}
        for seg in SEGMENTS:
            prof = diurnal_profile(seg, hour, weekend)
            prof = prof / prof.mean()
            rain_eff = np.where(heavy_rain, 0.88 if seg == "retail" else 1.06, 1.0)
            fest = retail_f if seg == "retail" else fleet_f
            noise = np.exp(_ar1(rng, n, cfg["demand"]["noise_phi"], cfg["demand"]["noise_sd"]))
            mwh = dc_mw * ref_util * mix[seg] * prof * seas * growth * temp_eff * rain_eff * fest * noise
            cols[f"dc_{seg}_kwh"] = mwh * 1000.0
        pack_mwh = caps.loc[cl, "pack_mwh"]
        turns = cfg["demand"]["swap_ref_turns"][cl] * cfg["demand"]["demand_scale"]
        cm, kwh = cfg["customers"]["swap_count_mix"], cfg["customers"]["swap_kwh"]
        avg_kwh = sum(cm[s] * kwh[s] for s in SWAPS)
        swaps_day = pack_mwh * 1000 * turns / avg_kwh  # FY26 average swaps/day
        for s in SWAPS:
            prof = diurnal_profile(s, hour, weekend)
            prof = prof / prof.mean()
            rain_eff = np.where(heavy_rain, 0.85, 1.0)
            noise = np.exp(_ar1(rng, n, 0.5, 0.06))
            lam = swaps_day / 24 * cm[s] * prof * seas * growth * temp_eff ** 0.5 * rain_eff * fleet_f * noise
            cols[f"swaps_{s}"] = rng.poisson(np.clip(lam, 0, None)).astype(float)
        avail = np.clip(cfg["network"]["charger_availability_mean"] + _ar1(rng, n, 0.95, 0.012), 0.85, 1.0)
        df = pd.DataFrame(dict(time=ts, cluster=cl, fy=fy, **{k: w[k].values for k in
                               ["temp", "rh", "cloud", "rain", "ghi", "wind", "temp_fc", "cloud_fc", "rain_fc", "ghi_fc", "wind_fc"]},
                               solar_cf=sol, wind_cf=wcf, wind_mult=wmult, **{k: prices[cl][k].values for k in prices[cl]},
                               **cols, dc_avail=avail))
        frames.append(df)
    hourly = pd.concat(frames, ignore_index=True)
    hubs = build_hubs(cfg, rng)
    # monthly SoH records
    months = pd.date_range(ts[0], ts[-1], freq="MS")
    soh_rows = []
    for cl in cfg["clusters"]:
        bess0 = rng.uniform(0.97, 0.99)
        for i, m in enumerate(months):
            soh_rows.append(dict(month=m, cluster=cl, bess_soh=max(0.90, bess0 - 0.0013 * i + rng.normal(0, 0.002)),
                                 pack_soh_mean=float(np.clip(rng.normal(0.90, 0.05 / np.sqrt(50)), 0.8, 1.0)),
                                 pack_soh_sd=0.05))
    soh = pd.DataFrame(soh_rows)
    return dict(hourly=hourly, hubs=hubs, soh=soh)
