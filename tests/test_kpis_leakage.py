import numpy as np
import pandas as pd

from evgrid.forecasting import features as F
from evgrid.simulation.kpis import reliability


def test_reliability_caps_credit_at_reference():
    t = pd.date_range("2025-04-01", periods=2, freq="h")
    df = pd.DataFrame(dict(time=t, cluster="DEL", ref_retail=[10, 10], del_retail=[15, 2], ref_ride=[0, 0], del_ride=[0, 0],
                           ref_logi=[0, 0], del_logi=[0, 0], ref_sw=[5, 5], del_sw=[4, 4]))
    # day credit retail = min(17, 20) = 17 ; swaps min(8, 10) = 8 ; R = 25 / 30
    assert abs(reliability(df)["R"] - 25 / 30) < 1e-12
    df.loc[0, "del_retail"] = 40  # induced demand cannot inflate R above reference
    assert reliability(df)["R"] <= 1.0


def test_features_use_only_forecast_weather_and_old_lags():
    cols = F.feature_columns("dc")
    for w in F.WEATHER_ACTUAL:
        assert w not in cols, f"actual weather column {w} leaks into features"
    assert all(c in cols for c in F.WEATHER_FC)
    assert min(F.LAGS_H) >= 48 and F.ROLL_END_LAG_DAYS >= 2


def test_lag_features_shift_correctly():
    idx = pd.date_range("2024-01-01", periods=24 * 20, freq="h")
    y = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    f = F.series_lag_features(y)
    assert np.allclose(f["lag48"].values[48:], y.values[:-48])
    # rolling level for day D only uses data up to the end of D-2
    d = 10
    day_rows = slice(24 * d, 24 * (d + 1))
    max_used = y.values[: 24 * (d - 1)].max()
    lvl = f["level"].values[day_rows][0]
    assert lvl <= max_used
