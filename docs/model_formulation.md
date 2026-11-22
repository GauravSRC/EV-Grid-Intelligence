# Model formulation

This document states the quantitative model behind the framework: the planning optimisation the AI policy
solves (the **model**) and the nonlinear physics the digital twin simulates (the **plant**). Code references are
given for each block.

## 1. Sets and indices

| Symbol | Meaning |
|---|---|
| $c \in \mathcal{C}$ | clusters {DEL, MUM, BLR, HYD, CHE} (decoupled given the monthly pack plan) |
| $h \in \mathcal{H}_c$ | hubs in cluster $c$ (12; archetypes CBD / residential / IT park / highway) |
| $t \in \{1..T\}$ | hourly steps, $T = 36$ (24 h operating day + 12 h look-ahead, receding) |
| $s \in \{retail, ride, logi\}$ | DC customer segments; $k \in \{2W, 3W\}$ swap classes |
| $\omega \in \{P10,P30,P50,P70,P90\}$ | DAM price scenarios |

## 2. Exogenous relationships (data layer, `evgrid/data/synth.py`)

- **Weather to RE supply.** Solar: $CF^{pv}_t = \min\{1,\; 1.3\,\tfrac{GHI^{cs}_t(1-0.75\,cloud_t^{3.4})}{1000}\,(1-0.004(T^{cell}_t-25))\,PR\}$ with clear-sky GHI from solar geometry and $T^{cell}=T+0.03\,GHI$.
  Wind: Weibull($k$=2) speed via a Gaussian copula with daily + hourly AR(1), power curve $CF^{w}=\mathrm{clip}\big(\tfrac{v^3-3^3}{12^3-3^3},0,1\big)$, zero above 25 m/s.
- **Weather to demand.** $D_{s,t} = \bar D_s\, g^{yr}_t\, \pi_s(hour, dow)\, \sigma_t\, F^{fest}_t\, (1+0.015\,(T_t-32)^+)\, \rho^{rain}_{s,t}\, e^{\epsilon_t}$, $\epsilon$ AR(1) ($\phi$=0.6, sd 0.08), growth 20%/yr.
- **Price formation.** $\lambda^{DAM}_t = \min\{10,\; L_{FY}\, S_t\, [1 + a^{eve}_t + a^{night}_t - d^{solar}_m\,\widetilde{GHI}_t]\,(1+0.035\,\Delta T_t)\,(1.9 \text{ in scarcity regimes})\,e^{\eta_t}\}$ re-scaled to IEX FY means; $\lambda^{RTM} = \lambda^{DAM} +$ evening-skewed spread; $\lambda^{G\text{-}DAM} = (0.72\text{-}0.95)\lambda^{DAM}$.
- **Hub spatial structure.** Hub demand shares $s_{h,t} \propto w_h\,\pi_{arch(h)}(t)$; capacity and pack shares $\kappa_h$ drawn independently (Dirichlet, concentrated by $\gamma$) - the source of simultaneous queues and idle assets.

## 3. Day-ahead stochastic scheduling LP (`evgrid/optimization/day_ahead.py`)

Solved per cluster and day, for each price scenario $\omega$ (5 LPs, ~600 variables each, HiGHS).

**Decision variables (per $t$, MWh):** DAM buy $x^{dam}_t$, DISCOM draw $x^{dis}_t$, RE used / sold-or-banked / spilled $r^{u}_t, r^{s}_t, r^{sp}_t$,
BESS charge/discharge/energy $b^{+}_t, b^{-}_t, e_t$, pack charging $q_t$ and charged-pack inventory $I_t$, DC served/unserved $y_t, u_t$,
swaps served/unserved $z_t, v_t$, deferred/recovered flexible demand $f^{-}_t, f^{+}_t$ and backlog $B_t$.

**Objective (maximise margin = minimise):**

$$
\min \sum_t \Big[(\lambda^{\omega}_t + \phi + OA)x^{dam}_t + \tau^{ToD}_t x^{dis}_t - \pi^{sell}_t r^{s}_t - p^{dc}_t y_t - p^{sw}_t z_t
 + VoLL\,(u_t + v_t) + c^{deg}_{B} b^{-}_t + c^{deg}_{P}\eta_P q_t + h^{B}(T_t)e_t + h^{P}(T_t) I_t + c^{shift} f^{-}_t \Big]
 - V^{term}(e_T + 1.2\,I_T)
$$

**Constraints:**

1. AC energy balance: $x^{dam}_t + x^{dis}_t + r^{u}_t + \eta_d b^{-}_t = b^{+}_t + y_t/\eta_{chg} + q_t + a_t$ (linearised charger efficiency; $a_t$ forecast auxiliaries incl. staged-module standby).
2. RE split: $r^{u}_t + r^{s}_t + r^{sp}_t = RE^{fc}_t$.
3. Sanctioned grid connection: $x^{dam}_t + x^{dis}_t \le G_c$.
4. BESS: $e_t = (1-\delta_B)e_{t-1} + \eta_c b^{+}_t - b^{-}_t$; $(SoC_{min} + 5\%)E \le e_t \le SoC_{max}E$; $b^{\pm}_t \le 0.5C \cdot E$.
5. Packs: $I_t = (1-\delta_P)I_{t-1} + \eta_P q_t - z_t$; $0.05\,Q \le I_t \le Q$; $q_t \le$ rack power; $z_t \le I_{t-1} - 0.05Q$ (swaps come from start-of-hour inventory above the reserve).
6. Swap demand: $z_t + v_t = SW^{fc}_t$.
7. DC demand with virtual-battery load shifting: $y_t + u_t + f^{-}_t - f^{+}_t = D^{fc}_t$; $y_t - f^{+}_t \le \Phi_t$ where
   $\Phi_t = \sum_h \min(D^{fc}_t s_{h,t}, 0.95\,Cap_h) + \min\{\rho\,E_t(1-\ell), F_t\}$ is the spatially servable volume (closed-form hub model with steering acceptance $\rho$);
   $f^{-}_t \le \sigma\,D^{fc}_t$, $f^{+}_t \le F^{free}_t$; $B_t = B_{t-1} + f^{-}_t - f^{+}_t$, $B_{23} = B_T = 0$.
8. Non-negativity and bounds.

Degradation enters linearly: throughput cost $c^{deg} = \frac{C^{rep}(1-\text{residual})}{EFC_{80}\cdot 0.8}$ (packs ~INR 1.9/kWh, BESS ~INR 1.1/kWh) and a temperature-scaled SoC-holding cost
$h(T) = \frac{0.02}{8760}\,2^{(T-25)/10}\cdot\frac{(1.5-1)}{0.5}\cdot\frac{C^{rep}(1-\text{residual})}{0.2}$ (calendar fade x2 per +10 degC, x1.5 near full SoC).

## 4. Bid curves and clearing (`evgrid/optimization/bidding.py`)

For each hour, the five scenario solutions $(\lambda^{\omega}_t, x^{dam,\omega}_t)$ are sorted by price and projected onto a
non-increasing step function by pool-adjacent-violators (isotonic) regression; prices are capped at INR 10/kWh. The realised
DAM price clears the first step whose price is at or above it; above the top step the P90 quantity is bid at the cap.

## 5. Pricing engine (`evgrid/optimization/pricing.py`)

$p_{s,t} = p^{ref}_{s,d}(1+m_{s,t})$, $p^{ref}_{s,d}$ = previous day's realised average tariff. $m_t = m^{shift}_t + m^{cost}_t + m^{scar}_t - m^{green}_t$,
de-meaned (shape only, no level ratchet) plus an anchor towards list price; then $|m| \le 20\%$ (fleet contracts $\le 10\%$), daily average
change $\le 5\%$ (iterated shift-and-clip). $m^{shift}$ inverts the behavioural response so the LP's defer/recover plan is induced.

## 6. Real-time dispatch and physics (`evgrid/simulation/engine.py`, `losses.py`, `degradation.py`)

Merit order each hour with actuals: RE, cleared DAM, BESS (track plan SoC +-2%, deeper only to avoid unserved or when RTM/DISCOM is expensive),
RTM or DISCOM within the grid limit, then curtail pack charging, then DC (unserved; 30% waits one hour). Surplus: cancel discharge, charge
BESS, resell DAM on RTM, sell RE on G-DAM or bank (8% in-kind, <= 30% of monthly DISCOM draw), else spill.

Hub service (closed form): $S_0 = \sum_h \min(D_h, Cap_h)$, $E = \sum_h (D_h - Cap_h)^+$, $F = \sum_h (Cap_h - D_h)^+$, recovered $= \min(\rho E(1-\ell), F)$; same for swaps on
charged inventory, plus intra-city vans (AI). Behaviour: own-price elasticity on the daily tariff level, shiftable share responding to intra-day deviations (noise $N(1,0.25)$).

Losses (plant, nonlinear): charger $P^{loss} = n_{on}P_0 + k_1P + k_2P^2/P_{rated}$ per hub; BESS $\eta(C)$, auxiliaries rising with temperature, self-discharge;
pack charging $\eta$ 0.93 at 0.5C to 0.90 at 1.5C; idle packs (self-discharge 2.5%/month, BMS 1 W/pack, cooling proportional to charged MWh x (T-25)+);
distribution = transformer no-load 0.25% of rating + load loss 1% x (P/P_rated)^2.

## 7. Strategic layer (`evgrid/optimization/strategic.py`)

Monthly transport LP: $\min \sum_{ij} c_{ij}x_{ij} + \pi\sum_j short_j$ s.t. $short_j \ge Q^{tgt}_j - (Q_j + \sum_i x_{ij} - \sum_k x_{jk})$,
$\sum_k x_{jk} \le (Q_j - 1.05\,Q^{tgt}_j)^+$, with $Q^{tgt}$ from last year's same-month swap volume x growth / 2.6 turns/day and move costs INR 40-150/pack.

## 8. KPI definitions (`evgrid/simulation/kpis.py`)

- Reliability $R = \sum_{c,d}\big[\sum_s \min(del_{s,d}, ref_{s,d}) + \min(del^{sw}_d, ref^{sw}_d)\big] / \sum ref$ (reference = demand at reference tariff; same denominator for both policies; induced demand cannot inflate R).
- Loss % = (charger conversion + standby + BESS + pack charging + idle packs + distribution) / meter input (DISCOM + DAM net + RTM + RE consumed).
- EBITDA margin = (revenue - energy cost - variable opex - fixed opex) / revenue (ex-GST); degradation reported separately.
