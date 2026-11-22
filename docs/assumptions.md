# Assumptions and parameters

All parameters live in `config/base.yaml` (unit and source in the trailing comment); this table is generated from it by `evgrid/reporting/build_report.py`. Parameters marked CALIBRATED are overwritten by the frozen `config/calibrated.yaml`.

## Frozen baseline calibration (structural parameters only)

- `network.hub_misalloc_gamma` = **3.0312**
- `losses.charger_p0_frac` = **0.005**
- `losses.charger_k1` = **0.02375**
- `opex.fixed_opex_cr` = **1090.0**

## Verified market anchors

| Anchor | Source | Link |
|---|---|---|
| IEX DAM average price FY24 INR 5.24/kWh, FY25 INR 4.47/kWh; I-DAM cap INR 10/kWh; 15-min blocks | IEX Annual Report 2024-25 | <https://www.iexindia.com/investor-relations> |
| Solar-hour DAM ~INR 2.5/kWh in surplus months; evening/night at the INR 10 cap in summer (e.g. 19-Apr-2025) | IEX market snapshot | <https://www.iexindia.com/market-data/day-ahead-market/market-snapshot> |
| RTM: 48 half-hourly auctions/day, delivery 1 h after gate closure | IEX Real-Time Market | <https://www.iexindia.com/products/real-time-market> |
| RTM-DAM evening spread up to ~INR 3.4/kWh; G-DAM below DAM (Apr-2025: 4.07 vs 5.20) | IIT-Kanpur EAL Power Chronicle; Mercom India | <https://www.mercomindia.com> |
| EV tariff <= ACoS until 31-Mar-2028; 0.7x ACoS solar hours, 1.3x otherwise | MoP Guidelines for EV Charging Infrastructure 2024 | <https://powermin.gov.in> |
| DISCOM EV energy charges: Delhi ~4.5; Mumbai 7.4-9.5; Telangana 5-7; Tamil Nadu 6-9 ToD | DERC / MERC / TGERC / TNERC tariff orders | <https://cercind.gov.in> |
| Public DC fast charging INR 15-25/kWh; utilisation 1-5% nationally, breakeven ~30-35% | Industry reports (assumption for fleet-anchored network: 10-15%) | <https://www.niti.gov.in> |
| Swapping ~2 kWh packs, ~INR 50/swap; BaaS INR 999/25 kWh, 2,499/75 kWh | Battery Smart public plans | <https://www.batterysmart.in> |
| Standalone BESS auctions INR 1.5-3.6 lakh/MW/month; AC-AC RTE >= 85% | SECI / NTPC / NHPC BESS tenders 2025-26 | <https://www.seci.co.in> |
| Battery pack prices 2025: avg $108/kWh, LFP $81/kWh, stationary $70/kWh | BloombergNEF Battery Price Survey | <https://about.bnef.com> |
| DC charger efficiency ~95% peak, >92% above ~50 kW, lower at partial load | INL / ORNL; TU Dortmund | <https://www.osti.gov> |
| Banking with in-kind charge (8%) for green energy open access | Electricity (Promoting RE through Green Energy Open Access) Rules 2022 | <https://powermin.gov.in> |

## Full parameter table

| Parameter | Value | Unit | Source / note |
|---|---|---|---|
| `meta.seed` | 42 | - | reproducibility |
| `meta.start` | "2021-04-01 00:00" | IST | FY22 start |
| `meta.end` | "2026-03-31 23:00" | IST | FY26 end |
| `meta.sim_start` | "2025-04-01" | IST | simulated fiscal year (FY26 = test year 5) |
| `meta.sim_end` | "2026-03-31" | IST | assumption |
| `meta.train_end` | "2025-03-31 23:00" | IST | forecasting train years 1-4 |
| `clusters` | [DEL, MUM, BLR, HYD, CHE] | - | case study: 5 metro clusters |
| `cluster_names` | {DEL: "Delhi NCR", MUM: "Mumbai MMR", BLR: "Bengaluru", HYD: "Hyderabad", CHE: "Chennai"} |  | assumption |
| `price_region` | {DEL: N, MUM: W, BLR: S, HYD: S, CHE: S} | - | IEX bid areas (simplified to regions) |
| `network.total_capacity_mw` | 1200 | MW | case study |
| `network.dc_share_overall` | 0.60 | - | case study (60% DC / 40% swap) |
| `network.capacity_share` | {DEL: 0.26, MUM: 0.24, BLR: 0.20, HYD: 0.15, CHE: 0.15} | - | assumption |
| `network.swap_share` | {DEL: 0.46, MUM: 0.30, BLR: 0.48, HYD: 0.38, CHE: 0.38} | - | assumption (DEL/BLR swap-heavy, MUM DC-heavy; overall 40%) |
| `network.hubs_per_cluster` | 12 | - | assumption |
| `network.hub_archetypes` | {CBD: 3, RES: 4, ITP: 3, HWY: 2} | hubs | assumption (CBD, residential, IT-park, highway/logistics) |
| `network.hub_capacity_dirichlet_alpha` | 4.0 | - | assumption: misalignment of capacity vs demand shares |
| `network.hub_demand_dirichlet_alpha` | 2.5 | - | assumption |
| `network.hub_misalloc_gamma` | 1.0 | - | CALIBRATED: exponent concentrating hub capacity/pack shares (1 = as drawn) |
| `network.charger_availability_mean` | 0.97 | - | assumption (OCPP uptime ~97%) |
| `network.sanctioned_ratio` | {DEL: 0.40, MUM: 0.36, BLR: 0.42, HYD: 0.50, CHE: 0.50} | - | assumption within 0.25-0.6 of nameplate |
| `network.sanctioned_scale` | 1.0 | - | CALIBRATED multiplier |
| `storage.bess_mwh_total` | 300 | MWh | case study 600 MWh split (assumption 50/50) |
| `storage.bess_c_rate` | 0.5 | 1/h | assumption (2-h LFP) |
| `storage.bess_share` | {DEL: 0.26, MUM: 0.24, BLR: 0.20, HYD: 0.15, CHE: 0.15} | - | pro-rata capacity |
| `storage.bess_eta_ch` | 0.95 | - | at low C-rate; AC-AC RTE 85-88% (SECI/NTPC BESS tenders: >=85%) |
| `storage.bess_eta_dis` | 0.95 | - | assumption |
| `storage.bess_eta_c_slope` | 0.03 | -/C | efficiency drop per C (assumption) |
| `storage.bess_aux_frac_day` | 0.005 | 1/day of capacity at 30C | 0.3-0.8%/day assumption |
| `storage.bess_aux_temp_slope` | 0.0002 | 1/day/degC | assumption |
| `storage.bess_self_discharge_day` | 0.001 | 1/day | LFP datasheets |
| `storage.bess_soc_min` | 0.10 | - | SoC lower bound |
| `storage.bess_soc_max` | 0.95 | - | assumption |
| `storage.pack_mwh_total` | 300 | MWh | case study (swap-pack inventory at stations) |
| `storage.pack_share` | {DEL: 0.24, MUM: 0.14, BLR: 0.18, HYD: 0.22, CHE: 0.22} | - | assumption: excess packs in HYD/CHE |
| `storage.pack_avg_kwh` | 2.5 | kWh/pack | 2W ~2 kWh, 3W ~4.5 kWh (Battery Smart / Sun Mobility) |
| `storage.pack_max_c` | 1.5 | 1/h | racks 480 MW ~ 1.5C |
| `storage.pack_ai_c` | 0.5 | 1/h | AI just-in-time C-rate |
| `storage.pack_eta_low_c` | 0.93 | - | at 0.5C (assumption) |
| `storage.pack_eta_high_c` | 0.90 | - | at 1.5C (assumption) |
| `storage.pack_self_discharge_month` | 0.025 | 1/month | 2-3%/month (LFP incl. BMS) |
| `storage.pack_bms_standby_w` | 1.0 | W/pack | assumption |
| `storage.pack_cooling_kw_per_mwh_degc` | 0.6 | kW per MWh charged per degC above 25C | assumption (cabinet cooling) |
| `storage.van_rebalance_frac_h` | 0.015 | 1/h of cluster inventory | AI intra-city van capacity (~6 vans x 100 packs/h per cluster; assumption) |
| `reserve.frac` | 0.05 | - | case study: 5% of capacity & charged inventory uncommitted |
| `reserve.priority_demand_frac` | 0.02 | - | priority fleets/emergency ~2% (assumption) |
| `grid.dam_fee` | 0.02 | INR/kWh | IEX transaction fee |
| `grid.oa_charges` | 1.40 | INR/kWh | assumption within 1.0-1.8 (CSS, wheeling, losses) |
| `grid.iex_cap` | 10.0 | INR/kWh | CERC/IEX DAM price cap |
| `grid.discom_energy` | {DEL: 4.5, MUM: 8.5, BLR: 5.5, HYD: 6.0, CHE: 7.0} | INR/kWh | DERC/MERC/TGERC/TNERC EV tariffs; BLR assumption |
| `grid.tod_mult` | {solar: 0.8, normal: 1.0, peak: 1.2} | - | base case (09-16 / other / 18-23); MoP 0.7/1.3 as scenario |
| `grid.bank_charge` | 0.08 | - | in-kind banking charge (Green Energy OA Rules 2022) |
| `grid.bank_value_frac_of_discount_discom` | 1.0 | - | banked kWh offsets off-peak DISCOM energy (assumption) |
| `grid.gdam_fee` | 0.04 | INR/kWh | fees on G-DAM sales (assumption) |
| `grid.rtm_premium_ai_exec` | 0.0 | INR/kWh | no extra slippage |
| `grid.transformer_noload_frac` | 0.0025 | - of rating | IEC 60076 typical |
| `grid.transformer_load_loss_frac` | 0.010 | - at rated load | typical |
| `grid.transformer_rating_frac` | 0.60 | - of nameplate (diversity) | assumption |
| `re.share_target` | 0.35 | - | case study: ~35% of energy from solar+wind OA |
| `re.cluster_mult` | {DEL: 0.65, MUM: 0.70, BLR: 1.00, HYD: 1.70, CHE: 1.75} | - | assumption: HYD/CHE over-contracted |
| `re.solar_frac` | 0.55 | - | of RE energy from solar (rest wind) |
| `re.landed_cost` | 4.75 | INR/kWh | 4.5-5.0 incl. OA (assumption, CERC/SECI tariffs + OA) |
| `re.solar_cuf` | 0.20 | - | 19-21% (MNRE) |
| `re.wind_cuf` | 0.31 | - | 28-35% (NIWE) |
| `prices.fy_dam_mean` | {FY22: 4.40, FY23: 5.90, FY24: 5.24, FY25: 4.47, FY26: 4.65} | INR/kWh | IEX AR FY24-25; FY22/23 IEX; FY26 assumption |
| `prices.region_offset` | {N: -0.02, W: 0.0, S: 0.04} | - | relative regional spread (assumption) |
| `prices.gdam_ratio_mean` | 0.82 | - | G-DAM vs DAM (Apr-25: 4.07 vs 5.20) |
| `prices.rtm_evening_spread_max` | 3.4 | INR/kWh | EAL Power Chronicle |
| `customers.dc_price` | {retail: 20.0, ride: 17.5, logi: 16.5} | INR/kWh ex-GST | public DC Rs15-25/kWh |
| `customers.dc_segment_mix` | {retail: 0.35, ride: 0.40, logi: 0.25} | - | assumption |
| `customers.swap_price` | {s2w: 70.0, s3w: 150.0} | INR/swap ex-GST | assumption (BaaS Rs33-40/kWh) |
| `customers.swap_kwh` | {s2w: 1.7, s3w: 3.8} | kWh/swap delivered | assumption |
| `customers.swap_count_mix` | {s2w: 0.78, s3w: 0.22} | - | assumption |
| `customers.elasticity` | {retail: -0.3, ride: -0.5, logi: -0.6, swap: -0.2} | - | assumption |
| `customers.shift_share` | {retail: 0.05, ride: 0.25, logi: 0.40, swap: 0.10} | - | assumption |
| `customers.shift_window_h` | {retail: 2, ride: 3, logi: 6, swap: 1} | h | assumption |
| `customers.steering_ai` | {retail: 0.20, ride: 0.50, logi: 0.70, swap: 0.40} | - | app/fleet-API acceptance (assumption) |
| `customers.steering_baseline` | 0.05 | - | assumption |
| `customers.steering_leak` | 0.10 | - | share of steered demand lost en route (assumption) |
| `customers.dc_wait_frac` | 0.30 | - | unserved DC demand that waits 1 h |
| `customers.response_noise_sd` | 0.25 | - | multiplicative noise on price response |
| `customers.price_band` | 0.20 | - | case study: +-20% of previous-day average tariff |
| `customers.daily_guardrail` | 0.05 | - | assumption: daily-average change <= 5% |
| `customers.fleet_band` | 0.10 | - | fleet contracts +-10% |
| `customers.baseline_tod` | {solar: -0.05, normal: 0.0, peak: 0.05} | - | static ToD tariff of status quo |
| `demand.dc_ref_util` | {DEL: 0.205, MUM: 0.215, BLR: 0.200, HYD: 0.095, CHE: 0.095} | - | assumption (fleet-anchored network) |
| `demand.swap_ref_turns` | {DEL: 3.3, MUM: 2.9, BLR: 3.6, HYD: 1.35, CHE: 1.35} | turns/day of station packs | assumption |
| `demand.demand_scale` | 1.0 | - | CALIBRATED multiplier on reference demand |
| `demand.growth_yoy` | 0.20 | 1/yr | case study context (~20%/yr) |
| `demand.temp_threshold_c` | 32.0 | degC | assumption |
| `demand.temp_slope` | 0.015 | 1/degC | +1.5%/degC above 32C |
| `demand.noise_sd` | 0.08 | - | AR(1) log-noise |
| `demand.noise_phi` | 0.6 | - | assumption |
| `losses.charger_p0_frac` | 0.006 | - of module rating per energised module | CALIBRATED (0.5-1%) |
| `losses.charger_k1` | 0.040 | - proportional loss | INL/ORNL: ~95% peak |
| `losses.charger_k2` | 0.020 | - quadratic loss at rated | assumption |
| `losses.charger_module_kw` | 30 | kW | module size (assumption) |
| `losses.ai_module_margin` | 1.25 | - | AI keeps forecast need x margin modules energised |
| `losses.ai_min_on_frac` | 0.15 | - | AI minimum energised share (warm standby) |
| `losses.distribution_extra_frac` | 0.0 | - | other distribution losses |
| `degradation.pack_efc_life` | 4500 | cycles to 80% | 4000-6000 LFP |
| `degradation.bess_efc_life` | 6000 | cycles to 80% | assumption |
| `degradation.pack_replacement_inr_kwh` | 9000 | INR/kWh | BNEF 2025 + integration |
| `degradation.bess_replacement_inr_kwh` | 7000 | INR/kWh | assumption |
| `degradation.residual_frac` | 0.25 | - | assumption |
| `degradation.calendar_fade_yr` | 0.02 | 1/yr at 25C & 50% SoC | assumption |
| `degradation.calendar_temp_doubling_c` | 10.0 | degC | assumption |
| `degradation.calendar_high_soc_mult` | 1.5 | - near 100% SoC | assumption |
| `opex.payments_frac` | 0.025 | - of revenue | payment gateway/platform |
| `opex.site_host_frac` | 0.09 | - of revenue | 8-10% site-host share |
| `opex.swap_partner_inr` | 10.0 | INR/swap | assumption |
| `opex.pack_logistics_inr_per_pack_day` | 0.0 | INR | included in fixed |
| `opex.fixed_opex_cr` | 1000.0 | INR crore/yr company | CALIBRATED -> baseline EBITDA ~12.5% |
| `opex.fixed_alloc_capacity_weight` | 1.0 | - | allocation by installed capacity |
| `optimization.horizon_h` | 36 | h | 24 h + 12 h look-ahead |
| `optimization.voll` | 35.0 | INR/kWh | value of lost load (swept for Pareto) |
| `optimization.terminal_value_frac` | 0.85 | - of mean landed price for stored energy | assumption |
| `optimization.shift_incentive` | 0.3 | INR/kWh | cost of shifting (discount given) |
| `optimization.price_scenarios` | [0.1, 0.3, 0.5, 0.7, 0.9] | quantiles for stochastic bid curves | assumption |
| `optimization.pricing_beta` | 0.4 | - | wholesale pass-through |
| `optimization.scarcity_markup` | 0.10 | - | when P90 load > 95% deliverable |
| `optimization.green_discount` | 0.06 | - | in RE-surplus / solar hours |
| `optimization.pack_safety` | 0.25 | - | JIT pack target safety on top of forecast |
| `optimization.pack_lookahead_h` | 4 | h | assumption |
| `economics.ai_build_cr` | 30.0 | INR crore | case brief |
| `economics.ai_run_cr_yr` | 18.0 | INR crore/yr | case brief |
| `economics.wacc` | 0.12 | - | case brief |
| `economics.years` | 5 | yr | assumption |
| `economics.bess_capex_inr_kwh` | 16000 | INR/kWh installed | BNEF stationary $70/kWh pack + BoS (assumption) |
| `economics.bess_auction_lakh_mw_month` | [1.5, 3.6] | INR lakh/MW/month | 2025-26 Indian standalone BESS auctions |

## Modelling assumptions

- Company data were not provided: a calibrated synthetic digital twin replaces the 5-year history (see data/README.md).
- The operator is a price-taker (~1% of DAM volume); a +-30% price sensitivity bounds price-impact effects.
- 15-minute IEX blocks are aggregated to hourly decisions, as stated in the case.
- Operational hours are 24x7; reference demand = demand at the reference (list) tariff, identical for both policies.
- Fixed opex is allocated to clusters by installed capacity; the AI programme run cost (INR 18 crore/yr) and pack-redeployment transport are charged to the AI policy's EBITDA.
- Pack redeployment moves packs (not racks); racks and chargers stay where they are unless the re-siting lever is applied.
- Monthly pack targets use last year's same-month swap volume x 20% growth (no look-ahead).
