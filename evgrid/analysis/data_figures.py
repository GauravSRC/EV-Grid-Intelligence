"""Sanity figures for the synthetic digital-twin dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evgrid.analysis.style import CLUSTER_COLORS, apply_style, plt, save


def data_overview(hourly: pd.DataFrame) -> str:
    """4-panel overview: DAM duck curve, RE seasonality, demand diurnal shapes, temperature climatology."""
    apply_style()
    h = hourly[hourly["fy"] == "FY26"].copy()
    h["hour"] = h["time"].dt.hour
    h["month"] = h["time"].dt.month
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    seasons = {"Summer (Apr-Jun)": [4, 5, 6], "Monsoon (Jul-Sep)": [7, 8, 9], "Winter (Oct-Mar)": [10, 11, 12, 1, 2, 3]}
    for name, ms in seasons.items():
        s = h[(h["cluster"] == "DEL") & h["month"].isin(ms)].groupby("hour")["dam_price"].mean()
        ax[0, 0].plot(s.index, s.values, label=name)
    ax[0, 0].axhline(10, ls="--", lw=1, color="#e34948")
    ax[0, 0].text(0.3, 9.6, "IEX cap INR 10/kWh", fontsize=8, color="#52514e")
    ax[0, 0].set(title="IEX DAM duck curve, N region FY26", xlabel="Hour (IST)", ylabel="INR/kWh")
    ax[0, 0].legend(fontsize=8)
    order = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]
    lab = ["A", "M", "J", "J", "A", "S", "O", "N", "D", "J", "F", "M"]
    for cl in ["DEL", "CHE"]:
        m = h[h["cluster"] == cl].groupby("month")[["solar_cf", "wind_cf"]].mean().loc[order]
        ax[0, 1].plot(range(12), m["solar_cf"].values, color=CLUSTER_COLORS[cl], label=f"{cl} solar")
        ax[0, 1].plot(range(12), m["wind_cf"].values, color=CLUSTER_COLORS[cl], ls="--", label=f"{cl} wind")
    ax[0, 1].set_xticks(range(12), lab)
    ax[0, 1].set(title="RE capacity factor by month (FY26)", ylabel="CUF")
    ax[0, 1].legend(fontsize=8, ncol=2)
    m = h[h["cluster"] == "BLR"]
    for col, name in [("dc_retail_kwh", "DC retail"), ("dc_ride_kwh", "DC ride-hailing"), ("dc_logi_kwh", "DC logistics")]:
        s = m.groupby("hour")[col].mean() / 1000
        ax[1, 0].plot(s.index, s.values, label=name)
    s = m.groupby("hour").apply(lambda x: (x["swaps_s2w"] * 1.7 + x["swaps_s3w"] * 3.8).mean() / 1000)
    ax[1, 0].plot(s.index, s.values, label="Swaps (energy)")
    ax[1, 0].set(title="Bengaluru demand by hour (FY26 mean)", xlabel="Hour (IST)", ylabel="MWh/h")
    ax[1, 0].legend(fontsize=8)
    for cl, c in CLUSTER_COLORS.items():
        t = h[h["cluster"] == cl].groupby("month")["temp"].max().loc[order]
        ax[1, 1].plot(range(12), t.values, color=c, label=cl)
    ax[1, 1].set_xticks(range(12), lab)
    ax[1, 1].set(title="Monthly max temperature (FY26)", ylabel="degC")
    ax[1, 1].legend(fontsize=8, ncol=3)
    fig.suptitle("Synthetic digital twin - sanity checks", fontweight="bold")
    fig.tight_layout()
    return str(save(fig, "01_data_overview.png"))


def data_stats(hourly: pd.DataFrame) -> dict:
    out = {}
    out["dam_mean_by_fy"] = hourly.groupby("fy")["dam_price"].mean().round(3).to_dict()
    f = hourly[hourly["fy"] == "FY26"]
    out["fy26_dam_cap_hours_share"] = float((f["dam_price"] >= 9.99).mean())
    out["fy26_solar_cuf"] = f.groupby("cluster")["solar_cf"].mean().round(3).to_dict()
    out["fy26_wind_cuf"] = f.groupby("cluster")["wind_cf"].mean().round(3).to_dict()
    dc = f[["dc_retail_kwh", "dc_ride_kwh", "dc_logi_kwh"]].sum().sum() / 1e9
    sw = (f["swaps_s2w"] * 1.7 + f["swaps_s3w"] * 3.8).sum() / 1e9
    out["fy26_reference_demand_twh"] = {"dc": round(float(dc), 4), "swap": round(float(sw), 4)}
    out["fy26_swaps_million"] = round(float((f["swaps_s2w"] + f["swaps_s3w"]).sum() / 1e6), 2)
    out["max_temp_by_cluster"] = f.groupby("cluster")["temp"].max().round(1).to_dict()
    out["solar_hour_dam_mean_surplus_months"] = float(f[f["time"].dt.hour.between(10, 14) & f["time"].dt.month.isin([2, 3, 4, 10, 11])]["dam_price"].mean())
    return out
