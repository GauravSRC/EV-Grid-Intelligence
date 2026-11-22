# Data

**All data in this repository is synthetic.** The operator's 5-year hourly history was not provided, so
`evgrid/data/synth.py` builds a calibrated *digital twin*: weather (actuals + day-ahead forecasts),
solar/wind capacity factors, regional IEX DAM/RTM/G-DAM prices, DC charging demand by customer segment,
swap requests by vehicle class, charger availability, hub tables and monthly battery state-of-health.
It is seeded (`meta.seed` in `config/base.yaml`) and fully reproducible.

- `data/generated/` (git-ignored) - full 5-year dataset, created by `python run_all.py --stage data`
  (parquet if `pyarrow` is installed, else `.csv.gz`). ~219k rows (43,824 h x 5 clusters).
- `data/sample/` - committed 14-day extract (1-14 Apr 2025) + hub table + SoH records.

## Plugging in real data

The pipeline is data-agnostic. Map operator extracts to the hourly schema below with
`evgrid.data.schema.adapt_real_data(...)` (sessions, swap logs, weather, IEX prices, RE meter data),
validate with `evgrid.data.schema.validate(df)`, and write the frame to `data/generated/hourly.parquet`
(+ `hubs.parquet`, `soh.parquet`). Every downstream stage (forecasting, optimisation, simulation, report)
then runs unchanged. Demand must be *requested* demand (served + lost sessions / stock-outs), which is
what the D-1 forecast targets.

## Hourly schema (`hourly`)

| column | unit / meaning |
|---|---|
| time | hour-beginning timestamp, IST |
| cluster | DEL, MUM, BLR, HYD, CHE |
| fy | fiscal year label (Apr-Mar), e.g. FY26 |
| temp, rh, cloud, rain, ghi, wind | actual weather: degC, %, 0-1, mm/h, W/m2, m/s (wind at contracted wind site hub height) |
| temp_fc, cloud_fc, rain_fc, ghi_fc, wind_fc | **day-ahead forecasts** issued D-1 10:00 (the only weather the models may use) |
| solar_cf, wind_cf | metered capacity factors of contracted RE (0-1) |
| dam_price, rtm_price, gdam_price | IEX prices, INR/kWh (hourly average of 15/30-min blocks; DAM/RTM capped at 10) |
| dc_retail_kwh, dc_ride_kwh, dc_logi_kwh | requested DC energy by segment, kWh |
| swaps_s2w, swaps_s3w | swap requests (count) for 2W (~1.7 kWh) and 3W (~3.8 kWh) packs |
| dc_avail | available share of DC charger capacity |

## Hub table (`hubs`)

12 hubs per cluster with archetypes CBD / RES (residential) / ITP (IT park) / HWY (highway-logistics);
`dc_demand_weight`, `swap_demand_weight` (Dirichlet) set where demand arises, `dc_cap_share`, `pack_share`
(independent Dirichlet) set where assets sit - the deliberate misalignment that creates simultaneous
queues and idle assets. Time-varying hub demand shares = weight x archetype hourly profile.

## SoH table (`soh`)

Monthly BESS state of health per cluster (90-98%) and pack-fleet SoH (mean ~0.90, sd 0.05).

## Generator design

- Weather: IMD-like monthly normals per city, diurnal cycle, AR(1) anomalies, heatwaves (Apr-Jun), monsoon
  breaks, Delhi winter fog; Mumbai heavy Jun-Sep monsoon, Chennai NE monsoon Oct-Dec.
- Forecasts = actual + AR(1) lead-time noise (temp sd 1.2 degC, cloud 0.12, wind 1.3 m/s).
- Solar = clear-sky geometry x Kasten-Czeplak cloud clearness x temperature derate (scaled to CUF 20%);
  wind = Weibull(k=2) via Gaussian copula + power curve (cut-in 3, rated 12, cut-out 25 m/s), CUF ~31%.
- DAM: duck curve (solar-hour trough, evening peak), seasonality, temperature and solar dependence, scarcity
  regimes (hot days push evening/night to the INR 10 cap), FY means re-scaled to IEX anchors (FY24 5.24, FY25 4.47).
  RTM = DAM + evening-skewed spread (<= 3.4); G-DAM = 0.72-0.95 x DAM.
- Demand: segment diurnal/weekly profiles, festivals (Diwali, Holi, Dussehra, Ganesh Chaturthi in MMR,
  Pongal in Chennai), ~20%/yr growth, +1.5%/degC above 32 degC, rain effects, AR(1) log-noise; swaps Poisson.
