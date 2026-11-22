"""Policy definitions: Baseline (status quo), AI (full framework) and single-lever ablations."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict

LEVERS = ["ml_forecast", "stochastic_bid", "storage_opt", "pricing", "steering", "redeploy", "loss_aware"]
LEVER_LABELS = {
    "ml_forecast": "ML forecasting", "stochastic_bid": "Stochastic bidding", "storage_opt": "Storage optimisation",
    "pricing": "Dynamic pricing & shifting", "steering": "App/fleet steering", "redeploy": "Pack redeployment & vans",
    "loss_aware": "Loss-aware ops",
}


@dataclass(frozen=True)
class Policy:
    name: str
    ml_forecast: bool = False      # HGB P10/P50/P90 vs seasonal naive
    stochastic_bid: bool = False   # 5-scenario bid curves vs single P50 bid
    storage_opt: bool = False      # LP BESS schedule + JIT pack charging vs fixed schedule / always-full packs
    pricing: bool = False          # dynamic pricing & load shifting vs static ToD tariff
    steering: bool = False         # app/fleet-API steering acceptance (AI) vs 5%
    redeploy: bool = False         # monthly inter-cluster pack redeployment + intra-city vans
    loss_aware: bool = False       # module staging / sleep + surplus RE sale/banking
    use_lp: bool = True            # day-ahead LP (False only for the status-quo baseline)


BASELINE = Policy("Baseline", use_lp=False)
AI = Policy("AI", **{k: True for k in LEVERS})


def ablations() -> Dict[str, Policy]:
    return {lev: replace(AI, name=f"AI without {LEVER_LABELS[lev]}", **{lev: False}) for lev in LEVERS}
