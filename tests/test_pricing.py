import numpy as np

from evgrid.optimization.pricing import apply_guardrails, check_price_band, compose_markup, segment_prices


def test_price_band_and_guardrail_random():
    rng = np.random.default_rng(0)
    for _ in range(500):
        m_raw = rng.normal(0, 0.3, 24)
        p_ref = rng.uniform(14, 24)
        p = apply_guardrails(m_raw, p_ref, 20.0, 0.20, 0.05)
        assert check_price_band(p, p_ref, 0.20)
        assert abs(p.mean() / p_ref - 1) <= 0.05 + 1e-9


def test_fleet_band_tighter():
    cfg = {"price_band": 0.2, "fleet_band": 0.1, "daily_guardrail": 0.05}
    m = np.linspace(-0.5, 0.5, 24)
    out = segment_prices(m, {"retail": 20.0, "ride": 17.5}, {"retail": 20.0, "ride": 17.5}, cfg)
    assert check_price_band(out["retail"], 20.0, 0.20)
    assert check_price_band(out["ride"], 17.5, 0.10)


def test_markup_is_revenue_neutral_shape():
    z = np.zeros(24)
    m = compose_markup(z, z, np.ones(24), z, np.linspace(3, 9, 24), np.arange(24) > 18, np.arange(24) < 6,
                       dict(band=0.2, beta=0.4, scarcity=0.1, green=0.06))
    assert abs(m.mean()) < 1e-12


def test_simulated_prices_within_band(sim_records):
    recs, cfg = sim_records
    for df in recs.values():
        assert (df["price_dev_max"] <= cfg["customers"]["price_band"] + 1e-9).all()
        assert (df["guard_dev"] <= cfg["customers"]["daily_guardrail"] + 1e-3).all()
