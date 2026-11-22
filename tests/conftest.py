import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sim_records():
    """Two simulated days (DEL) for Baseline and AI on the cached dataset; skipped if data is not generated."""
    import pandas as pd

    from evgrid.config import CACHE_DIR, GEN_DIR, load_config

    if not (CACHE_DIR / "forecasts_fy26.parquet").exists() or not any(GEN_DIR.glob("hourly.*")):
        pytest.skip("run `python run_all.py --stage data` and `--stage forecast` first")
    from evgrid.simulation.policies import AI, BASELINE
    from evgrid.simulation.runner import Runner

    cfg = load_config()
    r = Runner(cfg, parallel=False)
    days = list(pd.date_range("2025-05-19", periods=2))
    return {p.name: r.run(p, days, clusters=["DEL", "MUM"]) for p in (BASELINE, AI)}, cfg
