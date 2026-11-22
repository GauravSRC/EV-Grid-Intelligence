import numpy as np

from evgrid.optimization.bidding import build_bid_curve, clear_bid, is_monotone_nonincreasing
from evgrid.optimization.day_ahead import DAInputs, solve_day_ahead


def _toy(T=36, grid=20.0):
    h = np.arange(T) % 24
    dem = 30 + 20 * np.sin(2 * np.pi * (h - 14) / 24)
    return DAInputs(dc_dem=dem, dc_cap=dem, dc_free=np.zeros(T), dc_flex=0.2 * dem, dc_price=np.full(T, 18.0),
                    sw_dem=np.full(T, 5.0), sw_price=np.full(T, 40.0), re=np.full(T, 2.0), dam_px=np.full(T, 4.0),
                    discom_px=np.full(T, 6.0), sell_px=np.zeros(T), aux=np.full(T, 0.5), temp=np.full(T, 30.0),
                    grid_mw=grid, bess_mwh=40.0, bess_p=20.0, soc0=20.0, pack_q=30.0, pack_rack=40.0, inv0=20.0,
                    params=dict(eta_c=0.935, eta_d=0.935, eta_chg=0.96, eta_p=0.93, reserve=0.05, dam_fee=0.02, oa=1.4,
                                voll=35, soc_min=0.1, soc_max=0.95, deg_bess=1.1, deg_pack=1.9, hold_bess=0.05,
                                hold_pack=0.08, bess_sd_h=0.0, pack_sd_h=0.0, shift_cost=0.3, backlog_max=4.0,
                                terminal_value=5.0))


def test_toy_lp_respects_grid_limit_and_reserves():
    inp = _toy(grid=20.0)
    r = solve_day_ahead(inp)
    assert r.status == 0
    assert np.all(r["dam"] + r["dis"] <= 20.0 + 1e-6)
    assert np.all(r["soc"] >= (0.10 + 0.05) * 40.0 - 1e-6)
    assert np.all(r["inv"] >= 0.05 * 30.0 - 1e-6)
    assert abs(r["back"][23]) < 1e-6  # zero backlog at day end
    assert r["dun"].sum() > 0  # the binding grid limit forces some unserved demand


def test_bid_curve_monotone_and_clearing():
    rng = np.random.default_rng(1)
    for _ in range(200):
        p = rng.uniform(1, 12, 5)
        q = rng.uniform(0, 50, 5)
        ps, qs = build_bid_curve(p, q, cap=10.0)
        assert np.all(np.diff(ps) >= 0) and ps.max() <= 10.0
        assert is_monotone_nonincreasing(qs)
        lo, hi = clear_bid(ps, qs, 0.5), clear_bid(ps, qs, 9.99)
        assert lo >= hi


def test_reserve_floors_in_simulation(sim_records):
    recs, cfg = sim_records
    res = cfg["reserve"]["frac"]
    for df in recs.values():
        assert (df["soc_reserve_slack"] >= -1e-6).all()
        assert (df["dc_commit_frac"] <= 1 - res + 1e-9).all()
        assert (df["inv_reserve_slack"] >= -cfg["reserve"]["priority_demand_frac"] * df["dem_sw"] - 1e-6).all()
        assert (df["grid_import"] <= df["grid_limit"] + 1e-6).all()


def test_hourly_energy_balance_closes(sim_records):
    recs, _ = sim_records
    for df in recs.values():
        inputs = df["re_total"] + df["dam_bought"] + df["rtm"] + df["discom"]
        losses = df[["loss_charger", "loss_charger_standby", "loss_bess", "loss_pack_chg", "loss_pack_idle", "loss_dist"]].sum(axis=1)
        delivered = df[["del_retail", "del_ride", "del_logi", "del_sw"]].sum(axis=1)
        out = delivered + losses + df["d_soc"] + df["d_inv"] + df["re_sold"] + df["re_bank"] + df["re_spill"] + df["dam_resold"]
        assert np.allclose(inputs, out, atol=1e-6)
