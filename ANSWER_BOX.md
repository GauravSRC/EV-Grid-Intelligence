EV GRID INTELLIGENCE - AI DECISION FRAMEWORK

Approach
Company data was not provided, so I built a calibrated synthetic digital twin: five years of hourly weather (actuals and day-ahead forecasts), IEX prices, solar and wind output, segment-level charging demand, swap requests and battery state of health for the five clusters. I calibrated the status-quo policy to today's position using structural parameters only (hub asset misallocation and charger conversion loss, with fixed opex solved in closed form) and froze that calibration. The baseline reproduces reliability 77.9%, losses 10.1% and EBITDA 12.1%, with MUM, HYD, CHE below 12%. Both policies then run on identical data, random draws and physics; only the decisions differ.

Architecture
- Data layer: weather forecasts, IEX prices, session and swap logs, RE meters, SoH and OCPP telemetry, with a real-data adapter.
- Forecasting: pooled gradient-boosted models with conformal P10/P50/P90 for demand by cluster and segment, swaps, solar, wind, DAM price and the RTM-DAM spread, using only information available at D-1 10:00. On the test year, WAPE for DC demand is 0.077 vs 0.100 for a seasonal-naive forecast, and 0.161 vs 0.193 for the DAM price.
- Strategic layer (monthly): a transport LP redeploys swap packs from surplus to deficit clusters. Tactical layer (day-ahead): a stochastic LP per cluster over 36 hours for five price scenarios gives isotonic DAM bid curves, BESS and pack plans and a load-shifting plan. Operational layer (hourly): a dispatcher using actuals and an RTM nowcast, just-in-time pack charging, hub-level steering through the app and fleet APIs, and charger module sleep.
- Pricing engine: tariff = previous-day average x (1 + markup). The markup combines load shifting, wholesale pass-through, scarcity and green-hour terms. It is clipped to plus or minus 20% (10% for fleet contracts), with a 5% cap on the daily-average change.
- Execution: trader-approved IEX bids, OCPP 2.0.1 charger control, BMS, fleet API and app, with drift monitoring, retraining and a rule-based fail-safe.

Quantitative model
The LP maximises revenue plus surplus-RE sales, minus DAM, DISCOM and degradation costs and the value of lost load. It is subject to the hourly AC energy balance, the sanctioned grid limit, BESS and pack-inventory dynamics with self-discharge, 5% reserve floors on charger capacity, charged packs and BESS energy, a closed-form hub servability limit with steering, and a virtual-battery backlog for shifted demand. In the model, weather drives solar through irradiance and temperature, wind through a power curve, demand through heat (+1.5% per degree C above 32) and rain, and prices through temperature and solar. The simulator uses nonlinear losses (charger standby and conversion, C-rate dependent pack and BESS efficiency, idle-pack cooling, transformer losses) and LFP ageing (cycle cost about INR 1.9/kWh for packs, 1.1 for the BESS, plus temperature-driven calendar fade).

Key assumptions
- DAM FY26 mean INR 4.74/kWh (FY24 5.24 and FY25 4.47 anchors), price cap INR 10, open-access charges INR 1.4/kWh.
- DISCOM EV tariffs: Delhi 4.5 to Mumbai 8.5 INR/kWh, with ToD multipliers 0.8/1.0/1.2. RE landed cost INR 4.75/kWh take-or-pay; solar CUF 20% and wind CUF 31%.
- DC prices INR 20/17.5/16.5 per kWh (retail/ride-hailing/logistics); swaps INR 70 (2W) and 150 (3W).
- Price elasticities -0.3 to -0.6; steering acceptance 20-70% with AI vs 5% today.
- BESS 300 MWh at 0.5C with round-trip efficiency about 87%; 300 MWh of packs. Fixed opex INR 1,090 crore/yr. The AI run cost of INR 18 crore/yr is charged to AI EBITDA.

Results (full year FY2025-26)
- Reliability 77.9% to 88.9% against a 89.7% target (missed by 0.8 pp). Swap stock-outs fall from 17.1% to 0.01% and DC lost sessions from 23.5% to 13.7%. Raising steering acceptance by 25% gives 90.7%, which closes the gap.
- Losses 10.09% to 7.14% against an 8% target (met).
- EBITDA 12.1% to 19.5% (INR 294 crore to INR 550 crore). By cluster: DEL 28.3%, MUM 18.2%, BLR 26.2%, HYD 2.5%, CHE 0.6%. Below 12%: HYD, CHE. These clusters are over-built for their demand, which operations alone cannot fix; re-siting 3 pp of capacity each to Mumbai and Delhi lifts them to 14.1% and 11.4%.
- Every hard constraint held in 100% of hours: price band, guardrail, reserves and grid limits.
- Ten stress tests (heatwave, weak monsoon, wind lull, +40% prices, +20% demand, doubled forecast errors, ageing, grid derate, MoP ToD, combined): AI beats baseline on both reliability and EBITDA in 10/10. The weakest is 'Price shock (+40% DAM/RTM, capped)' at 17.8% company EBITDA. Monte Carlo P5 company margin: 18.3%.

Scalability and economics
Clusters decouple, so compute grows linearly with clusters. One day-ahead LP solves in about 11 ms, and 100 clusters x 5 scenarios take 5 s on one core, far inside the 2-hour DAM window and the 30-minute RTM cadence. Build costs INR 30 crore; running costs INR 18 crore/yr. Incremental EBITDA is INR 255 crore/yr, giving a payback of 1.4 months and a 5-year NPV at 12% of INR 891 crore. An extra 100 MWh of BESS adds only INR 7.3 crore/yr, so contracting storage beats owning it.

Roadmap
- Quarter 1: pilot in Mumbai, with 4 weeks of shadow bidding and then live trading with trader approval, module staging and two fleet partners.
- Quarter 2: roll out to all five clusters with pack redeployment, dynamic pricing and an RE sale and banking desk. Prepare a business case for re-siting Hyderabad and Chennai capacity and re-allocate RE contracts at renewal.

Repository: https://github.com/GauravSRC/EV-Grid-Intelligence
