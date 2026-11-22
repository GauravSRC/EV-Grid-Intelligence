"""Result figures (all numbers read from results/*.json and cached simulation records)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from evgrid.analysis.style import CLUSTER_COLORS, PALETTE, POLICY_COLORS, apply_style, plt, save
from evgrid.config import CACHE_DIR, RESULTS_DIR, load_json

log = logging.getLogger(__name__)
CL = ["DEL", "MUM", "BLR", "HYD", "CHE"]


def _main():
    p = RESULTS_DIR / "kpis_main_full.json"
    return load_json(p if p.exists() else RESULTS_DIR / "kpis_main_quick.json")


def architecture() -> str:
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc="#ffffff", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=9, fontweight="bold", color="#0b0b0b")
        ax.text(x + w / 2, y + h - 0.6, body, ha="center", va="top", fontsize=7.4, color="#52514e", linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#8a8984", lw=1.2))

    box(0.1, 3.3, 2.1, 2.5, "1. Data layer", "weather + D-1 forecasts\nIEX DAM/RTM/G-DAM\nsessions, swap logs\nRE meters, BESS/pack SoH\nhub telemetry (OCPP)", PALETTE[0])
    box(2.6, 3.3, 2.1, 2.5, "2. Forecasting", "HGB P10/P50/P90\ndemand x segment, swaps\nsolar, wind, DAM, RTM spread\nconformal intervals\nfeatures at D-1 10:00", PALETTE[1])
    box(5.1, 4.25, 2.5, 1.55, "3a. Strategic (monthly)", "pack redeployment LP\nRE-contract re-allocation", PALETTE[2])
    box(5.1, 2.45, 2.5, 1.65, "3b. Tactical (day-ahead)", "stochastic LP per cluster\n36 h, 5 price scenarios\nisotonic DAM bid curves", PALETTE[2])
    box(5.1, 0.55, 2.5, 1.75, "3c. Operational (hourly)", "RT dispatcher, RTM nowcast\nBESS tracking, JIT packs\nhub steering, module staging", PALETTE[2])
    box(8.0, 3.3, 1.8, 2.5, "4. Pricing", "p = p_ref(1+m)\n|m|<=20%, fleets 10%\ndaily avg <=5%\nshift + cost +\nscarcity - green", PALETTE[3])
    box(10.1, 3.3, 1.8, 2.5, "5. Execution", "IEX bids (trader\napproval)\nOCPP 2.0.1 chargers\nBMS / racks\nfleet API & app", PALETTE[4])
    box(8.0, 0.55, 3.9, 2.2, "6. Monitoring & MLOps", "KPI + constraint monitors (band, reserve, grid limit)\ndrift detection -> weekly retrain, champion/challenger\nhuman-in-the-loop trader approval\nrule-based fail-safe on LP or forecast faults", PALETTE[6])
    arrow(2.2, 4.55, 2.6, 4.55)
    arrow(4.7, 4.9, 5.1, 5.0)
    arrow(4.7, 4.2, 5.1, 3.3)
    arrow(4.7, 3.6, 5.1, 1.4)
    arrow(7.6, 3.3, 8.0, 4.3)
    arrow(9.8, 4.55, 10.1, 4.55)
    arrow(6.35, 2.45, 6.35, 2.3)
    arrow(11.0, 3.3, 11.0, 2.75)
    arrow(8.0, 1.6, 7.6, 1.4)
    ax.text(6.0, 0.12, "Clusters decouple given the monthly pack plan -> one small LP per cluster per day, solved in parallel", ha="center", fontsize=8, color="#52514e")
    return str(save(fig, "03_architecture.png"))


def kpi_comparison() -> str:
    apply_style()
    m = _main()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    groups = ["overall"] + CL
    x = np.arange(len(groups))
    for i, (key, tgt, title, fmt) in enumerate([("R", 0.897, "Service reliability", 100), ("loss_pct", 0.08, "Storage & conversion losses", 100),
                                                ("ebitda_margin", 0.12, "EBITDA margin", 100)]):
        b = [m["Baseline"][g][key] * fmt for g in groups]
        a = [m["AI"][g][key] * fmt for g in groups]
        ax[i].bar(x - 0.2, b, 0.38, color=POLICY_COLORS["Baseline"], label="Baseline")
        ax[i].bar(x + 0.2, a, 0.38, color=POLICY_COLORS["AI"], label="AI framework")
        ax[i].axhline(tgt * fmt, color=POLICY_COLORS["Target"], ls="--", lw=1.2, label=f"Target {tgt * fmt:.1f}%")
        ax[i].set_xticks(x, ["Total"] + CL)
        ax[i].set_title(title)
        ax[i].set_ylabel("%")
        ax[i].legend(fontsize=7.5, loc="lower left" if key != "loss_pct" else "upper right")
    fig.suptitle(f"Baseline vs AI ({m['period']})", fontweight="bold")
    fig.tight_layout()
    return str(save(fig, "04_kpi_comparison.png"))


def monthly_heatmaps() -> str:
    apply_style()
    m = _main()
    if "monthly" not in m["AI"]:
        return ""
    months = sorted(m["AI"]["monthly"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    for i, (pol, key, title, cmap) in enumerate([("Baseline", "R", "Baseline reliability %", "Blues"), ("AI", "R", "AI reliability %", "Blues"),
                                                 ("AI", "ebitda_margin", "AI EBITDA margin %", "Blues")]):
        z = np.array([[m[pol]["monthly"][mo][c][key] * 100 for mo in months] for c in CL])
        im = ax[i].imshow(z, aspect="auto", cmap=cmap, vmin=(60 if key == "R" else -10), vmax=(100 if key == "R" else 40))
        ax[i].set_yticks(range(5), CL)
        ax[i].set_xticks(range(len(months)), [mo[2:] for mo in months], rotation=90, fontsize=7)
        ax[i].set_title(title)
        ax[i].grid(False)
        for r in range(z.shape[0]):
            for c in range(z.shape[1]):
                ax[i].text(c, r, f"{z[r, c]:.0f}", ha="center", va="center", fontsize=6, color="#0b0b0b" if z[r, c] < (85 if key == "R" else 20) else "white")
        fig.colorbar(im, ax=ax[i], fraction=0.03)
    fig.tight_layout()
    return str(save(fig, "05_monthly_heatmaps.png"))


def waterfalls() -> str:
    apply_style()
    ab = load_json(RESULTS_DIR / "ablations.json")
    m = _main()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    # (a) reliability uplift by lever (ablation: AI minus AI-without-lever), rescaled to the AI-Baseline gap
    runs = ab["runs"]
    gap = runs["AI"]["overall"]["R"] - runs["Baseline"]["overall"]["R"]
    contrib = {ab["labels"][k]: v["R"] for k, v in ab["lever_contribution"].items()}
    tot = sum(max(v, 0) for v in contrib.values())
    items = sorted(contrib.items(), key=lambda kv: -kv[1])
    base = runs["Baseline"]["overall"]["R"] * 100
    cur = base
    ax[0].bar(0, base, color=POLICY_COLORS["Baseline"])
    for i, (k, v) in enumerate(items, 1):
        h = (max(v, 0) / tot * gap * 100) if tot > 0 else 0
        ax[0].bar(i, h, bottom=cur, color=PALETTE[0])
        cur += h
    ax[0].bar(len(items) + 1, runs["AI"]["overall"]["R"] * 100, color=PALETTE[0])
    ax[0].axhline(89.7, color=POLICY_COLORS["Target"], ls="--", lw=1)
    ax[0].set_xticks(range(len(items) + 2), ["Baseline"] + [k for k, _ in items] + ["AI"], rotation=40, ha="right", fontsize=7.5)
    ax[0].set_ylim(base - 5, max(cur, 90) + 2)
    ax[0].set(title="Reliability uplift by lever (ablations, %)", ylabel="R %")
    # (b) loss by component baseline vs AI
    comps = [("loss_charger_standby_pct", "Charger standby"), ("loss_charger_pct", "Charger conversion"), ("loss_pack_chg_pct", "Pack charging"),
             ("loss_pack_idle_pct", "Idle packs"), ("loss_bess_pct", "BESS"), ("loss_dist_pct", "Distribution")]
    bottom = np.zeros(2)
    for j, (k, lab) in enumerate(comps):
        vals = np.array([m["Baseline"]["overall"][k], m["AI"]["overall"][k]]) * 100
        ax[1].bar([0, 1], vals, bottom=bottom, color=PALETTE[j], label=lab, edgecolor="#fcfcfb", lw=1.5)
        bottom += vals
    ax[1].axhline(8.0, color=POLICY_COLORS["Target"], ls="--", lw=1)
    for i, v in enumerate(bottom):
        ax[1].text(i, v + 0.15, f"{v:.2f}%", ha="center", fontsize=8)
    ax[1].set_xticks([0, 1], ["Baseline", "AI"])
    ax[1].set(title="Losses by component (% of meter input)", ylabel="%")
    ax[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax[1].set_xlim(-0.6, 1.6)
    # (c) EBITDA bridge
    b, a = m["Baseline"]["overall"], m["AI"]["overall"]
    steps = [("Revenue", a["revenue_cr"] - b["revenue_cr"]), ("Energy cost", -(a["energy_cost_cr"] - b["energy_cost_cr"])),
             ("Var. opex + AI run", -(a["opex_var_cr"] - b["opex_var_cr"]))]
    cur = b["ebitda_cr"]
    ax[2].bar(0, cur, color=POLICY_COLORS["Baseline"])
    for i, (k, v) in enumerate(steps, 1):
        ax[2].bar(i, v, bottom=cur if v >= 0 else cur + v, color=PALETTE[2] if v >= 0 else PALETTE[7])
        ax[2].text(i, max(cur, cur + v) + 15, f"{v:+.0f}", ha="center", fontsize=8)
        cur += v
    ax[2].bar(len(steps) + 1, a["ebitda_cr"], color=PALETTE[0])
    ax[2].set_xticks(range(len(steps) + 2), ["Baseline"] + [k for k, _ in steps] + ["AI"], rotation=25, ha="right", fontsize=8)
    ax[2].set(title="EBITDA bridge (INR crore / yr)", ylabel="INR crore")
    fig.tight_layout()
    return str(save(fig, "06_waterfalls.png"))


def scenarios_fig() -> str:
    apply_style()
    sc = load_json(RESULTS_DIR / "scenarios.json")
    names = list(sc)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))
    y = np.arange(len(names))
    for i, (k, t, title) in enumerate([("R", 0.897, "Reliability %"), ("loss_pct", 0.08, "Losses %"), ("ebitda_margin", 0.12, "EBITDA margin %")]):
        ax[i].barh(y + 0.2, [sc[n]["Baseline"]["overall"][k] * 100 for n in names], 0.38, color=POLICY_COLORS["Baseline"], label="Baseline")
        ax[i].barh(y - 0.2, [sc[n]["AI"]["overall"][k] * 100 for n in names], 0.38, color=POLICY_COLORS["AI"], label="AI")
        ax[i].axvline(t * 100, color=POLICY_COLORS["Target"], ls="--", lw=1)
        ax[i].set_yticks(y, [n.replace("_", " ") for n in names] if i == 0 else [""] * len(names))
        ax[i].set_title(title)
        ax[i].invert_yaxis()
    ax[0].legend(fontsize=7.5)
    fig.suptitle("Stress scenarios (8 representative weeks, annualised)", fontweight="bold")
    fig.tight_layout()
    return str(save(fig, "07_scenarios.png"))


def tornado() -> str:
    apply_style()
    s = load_json(RESULTS_DIR / "sensitivity.json")
    base = s["base"]["ebitda_margin"] * 100
    cases = sorted(s["cases"].items(), key=lambda kv: -abs(kv[1]["high"]["ebitda_margin"] - kv[1]["low"]["ebitda_margin"]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.0))
    for j, (key, b0, title) in enumerate([("ebitda_margin", base, "AI EBITDA margin % (tornado)"), ("R", s["base"]["R"] * 100, "AI reliability % (tornado)")]):
        for i, (n, v) in enumerate(cases):
            lo, hi = v["low"][key] * 100 - b0, v["high"][key] * 100 - b0
            ax[j].barh(i, lo, left=b0, color=PALETTE[7], height=0.6)
            ax[j].barh(i, hi, left=b0, color=PALETTE[2], height=0.6)
        ax[j].axvline(b0, color="#0b0b0b", lw=1)
        ax[j].set_yticks(range(len(cases)), [n for n, _ in cases] if j == 0 else [""] * len(cases), fontsize=8)
        ax[j].invert_yaxis()
        ax[j].set_title(title)
    ax[0].text(0.99, 0.02, "red = low case, green = high case", transform=ax[0].transAxes, ha="right", fontsize=7.5, color="#52514e")
    fig.tight_layout()
    return str(save(fig, "08_tornado.png"))


def pareto_mc() -> str:
    apply_style()
    p = load_json(RESULTS_DIR / "pareto.json")["points"]
    mc = load_json(RESULTS_DIR / "monte_carlo.json")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
    ax[0].plot([q["R"] * 100 for q in p], [q["ebitda_margin"] * 100 for q in p], marker="o", color=PALETTE[0])
    for q in p:
        ax[0].annotate(f"VoLL {q['voll']:.0f}", (q["R"] * 100, q["ebitda_margin"] * 100), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    ax[0].set(title="Reliability-EBITDA frontier (VoLL sweep)", xlabel="Reliability %", ylabel="EBITDA margin %")
    groups = ["overall"] + CL
    x = np.arange(len(groups))
    for k, (pol, off) in enumerate([("Baseline", -0.2), ("AI", 0.2)]):
        s = mc["summary"][pol]
        mean = np.array([s[g]["mean"] for g in groups]) * 100
        p5 = np.array([s[g]["p5"] for g in groups]) * 100
        p95 = np.array([s[g]["p95"] for g in groups]) * 100
        ax[1].errorbar(x + off, mean, yerr=[mean - p5, p95 - mean], fmt="o", color=POLICY_COLORS[pol], capsize=3, label=f"{pol} mean (P5-P95)")
    ax[1].axhline(12, color=POLICY_COLORS["Target"], ls="--", lw=1)
    ax[1].set_xticks(x, ["Total"] + CL)
    ax[1].set(title=f"EBITDA-at-risk, {mc['n_paths']} Monte Carlo price paths", ylabel="EBITDA margin %")
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    return str(save(fig, "09_pareto_montecarlo.png"))


def scalability_fig() -> str:
    apply_style()
    s = load_json(RESULTS_DIR / "scalability.json")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    h = s["lp_time_vs_horizon_s"]
    ax[0].loglog([int(k) for k in h], [v * 1000 for v in h.values()], marker="o", color=PALETTE[0])
    ax[0].set(title=f"LP solve time vs horizon steps (exponent {s['horizon_exponent']:.2f})", xlabel="time steps", ylabel="ms")
    n = s["lp_time_vs_nodes_s_sequential"]
    ax[1].loglog([int(k) for k in n], list(n.values()), marker="o", color=PALETTE[1], label="1 scenario, single core")
    ax[1].loglog([int(k) for k in n], [v * 5 for v in n.values()], marker="s", color=PALETTE[2], label="5 price scenarios, single core")
    ax[1].axhline(1800, color=POLICY_COLORS["Target"], ls="--", lw=1)
    ax[1].text(5, 1900, "RTM cadence 30 min", fontsize=7.5, color="#52514e")
    ax[1].set(title=f"Day-ahead compute vs clusters (exponent {s['nodes_exponent']:.2f})", xlabel="clusters (nodes)", ylabel="seconds")
    ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    return str(save(fig, "10_scalability.png"))


def dispatch_example() -> str:
    apply_style()
    p = CACHE_DIR / "sim_AI_full.parquet"
    if not p.exists():
        return ""
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for i, pol in enumerate(["Baseline", "AI"]):
        d = pd.read_parquet(CACHE_DIR / f"sim_{pol}_full.parquet")
        d = d[(d["cluster"] == "BLR") & (pd.to_datetime(d["time"]) >= "2025-05-20") & (pd.to_datetime(d["time"]) < "2025-05-22")]
        t = pd.to_datetime(d["time"])
        re_used = d["re_used"].values
        dam = (d["dam_bought"] - d["dam_resold"]).values
        oth = (d["rtm"] + d["discom"]).values
        bd = d["bess_dis_ac"].values
        ax[i].stackplot(t, re_used, dam, oth, bd, labels=["RE used", "DAM (net)", "RTM + DISCOM", "BESS discharge"],
                        colors=[PALETTE[2], PALETTE[0], PALETTE[3], PALETTE[6]], alpha=0.9)
        ax[i].plot(t, d["grid_limit"], color=POLICY_COLORS["Target"], ls="--", lw=1, label="Sanctioned limit")
        ax[i].plot(t, -d["bess_ch_ac"], color=PALETTE[6], lw=1, label="BESS charge (neg.)")
        ax[i].set_title(f"{pol}: Bengaluru supply stack, 20-21 May 2025")
        ax[i].tick_params(axis="x", rotation=30)
    ax[0].set_ylabel("MWh/h")
    ax[1].legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    return str(save(fig, "11_dispatch_example.png"))


def make_all_figures() -> list:
    out = []
    for f in (architecture, kpi_comparison, monthly_heatmaps, waterfalls, scenarios_fig, tornado, pareto_mc, scalability_fig, dispatch_example):
        try:
            out.append(f())
        except Exception as e:  # keep the report build robust
            log.warning("figure %s failed: %s", f.__name__, e)
    return out
