"""Dataset caching (parquet if pyarrow present else csv.gz), sample export, RE contract sizing."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from evgrid.config import DATA_DIR, GEN_DIR
from evgrid.data import synth

log = logging.getLogger(__name__)

try:  # optional dependency
    import pyarrow  # noqa: F401

    _EXT = ".parquet"
except Exception:  # pragma: no cover
    _EXT = ".csv.gz"


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if _EXT == ".parquet":
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    else:
        df.to_csv(Path(str(path) + ".csv.gz"), index=False)


def _read(path: Path) -> pd.DataFrame:
    if _EXT == ".parquet":
        return pd.read_parquet(path.with_suffix(".parquet"))
    df = pd.read_csv(Path(str(path) + ".csv.gz"))
    for c in ("time", "month"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def _exists(path: Path) -> bool:
    return path.with_suffix(".parquet").exists() if _EXT == ".parquet" else Path(str(path) + ".csv.gz").exists()


def load_or_generate(cfg: dict, force: bool = False) -> Dict[str, pd.DataFrame]:
    """Load the cached synthetic dataset or generate it (seeded)."""
    names = ["hourly", "hubs", "soh"]
    if not force and all(_exists(GEN_DIR / n) for n in names):
        return {n: _read(GEN_DIR / n) for n in names}
    data = synth.generate(cfg, seed=cfg["meta"]["seed"])
    for n in names:
        _write(data[n], GEN_DIR / n)
    export_sample(data)
    return data


def export_sample(data: Dict[str, pd.DataFrame], days: int = 14) -> None:
    """Commit-able 14-day sample (FY26 start) for reviewers."""
    h = data["hourly"]
    t0 = pd.Timestamp("2025-04-01")
    s = h[(h["time"] >= t0) & (h["time"] < t0 + pd.Timedelta(days=days))]
    out = DATA_DIR / "sample"
    out.mkdir(parents=True, exist_ok=True)
    s.round(4).to_csv(out / "hourly_sample_14d.csv", index=False)
    data["hubs"].round(5).to_csv(out / "hubs.csv", index=False)
    data["soh"].round(4).to_csv(out / "soh_monthly.csv", index=False)


def re_contract_mw(cfg: dict, hourly: pd.DataFrame) -> pd.DataFrame:
    """Size solar/wind open-access contracts (MW) per cluster so RE ~ share_target of company procurement.

    Procurement is approximated by FY26 reference demand / 0.90 (meter vs delivered); the split across
    clusters follows ``re.cluster_mult`` (HYD/CHE over-contracted by design).
    """
    fy26 = hourly[hourly["fy"] == "FY26"]
    kwh = synth_avg_swap_kwh(cfg)
    rows = []
    for cl in cfg["clusters"]:
        d = fy26[fy26["cluster"] == cl]
        dem = (d[["dc_retail_kwh", "dc_ride_kwh", "dc_logi_kwh"]].sum().sum()
               + (d["swaps_s2w"] * cfg["customers"]["swap_kwh"]["s2w"] + d["swaps_s3w"] * cfg["customers"]["swap_kwh"]["s3w"]).sum())
        rows.append(dict(cluster=cl, ref_mwh=dem / 1000.0 / 0.90 * 0.80, sol=d["solar_cf"].mean(), wnd=d["wind_cf"].mean()))
    df = pd.DataFrame(rows).set_index("cluster")
    mult = pd.Series(cfg["re"]["cluster_mult"])
    total_re = cfg["re"]["share_target"] * df["ref_mwh"].sum()
    w = df["ref_mwh"] * mult
    re_mwh = total_re * w / w.sum()
    sf = cfg["re"]["solar_frac"]
    df["solar_mw"] = re_mwh * sf / (8760 * df["sol"])
    df["wind_mw"] = re_mwh * (1 - sf) / (8760 * df["wnd"])
    df["re_mwh_yr"] = re_mwh
    return df[["solar_mw", "wind_mw", "re_mwh_yr", "ref_mwh"]]


def synth_avg_swap_kwh(cfg: dict) -> float:
    cm, kwh = cfg["customers"]["swap_count_mix"], cfg["customers"]["swap_kwh"]
    return float(sum(cm[s] * kwh[s] for s in cm))
