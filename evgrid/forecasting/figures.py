"""Forecast evaluation figure."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evgrid.analysis.style import PALETTE, apply_style, plt, save


def forecast_figure(fc: pd.DataFrame, metrics: dict) -> str:
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[1, 1.4, 1.4]))
    names = list(metrics)
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, [metrics[n]["wape_naive"] for n in names], 0.38, color="#8a8984", label="Seasonal naive")
    ax[0].bar(x + 0.2, [metrics[n]["wape_ml"] for n in names], 0.38, color=PALETTE[0], label="ML (HGB)")
    ax[0].set_xticks(x, names)
    ax[0].set(title="Test-year WAPE (FY26)", ylabel="WAPE")
    ax[0].legend(fontsize=8)
    wk = fc[(fc["cluster"] == "BLR") & (fc["time"] >= "2025-05-05") & (fc["time"] < "2025-05-12")]
    y = wk[["retail_y", "ride_y", "logi_y"]].sum(axis=1) / 1000
    p50 = wk[["retail_p50", "ride_p50", "logi_p50"]].sum(axis=1) / 1000
    lo = wk[["retail_p10", "ride_p10", "logi_p10"]].sum(axis=1) / 1000
    hi = wk[["retail_p90", "ride_p90", "logi_p90"]].sum(axis=1) / 1000
    ax[1].fill_between(wk["time"], lo, hi, color=PALETTE[0], alpha=0.18, lw=0, label="P10-P90")
    ax[1].plot(wk["time"], y, color="#0b0b0b", lw=1.2, label="Actual")
    ax[1].plot(wk["time"], p50, color=PALETTE[0], lw=1.6, label="P50")
    ax[1].set(title="Bengaluru DC demand, D-1 forecast (May-25 week)", ylabel="MWh/h")
    ax[1].legend(fontsize=8, ncol=3)
    ax[1].tick_params(axis="x", rotation=30)
    wk = fc[(fc["cluster"] == "DEL") & (fc["time"] >= "2025-05-05") & (fc["time"] < "2025-05-12")]
    ax[2].fill_between(wk["time"], wk["dam_p10"], wk["dam_p90"], color=PALETTE[1], alpha=0.18, lw=0, label="P10-P90")
    ax[2].plot(wk["time"], wk["dam_y"], color="#0b0b0b", lw=1.2, label="Actual")
    ax[2].plot(wk["time"], wk["dam_p50"], color=PALETTE[1], lw=1.6, label="P50")
    ax[2].set(title="IEX DAM price (N), D-1 forecast", ylabel="INR/kWh")
    ax[2].legend(fontsize=8, ncol=3)
    ax[2].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return str(save(fig, "02_forecasting.png"))
