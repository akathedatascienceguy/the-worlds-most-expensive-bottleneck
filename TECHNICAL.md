# Technical Document: Risk-Aware Routing on a Dynamic Oil Network

**Project:** The World's Most Expensive Bottleneck  
**Stack:** Python 3.x · NetworkX · Plotly · Streamlit · NumPy · Pandas  
**Entry point:** `app.py` (self-contained; all logic is inline)

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Graph Construction](#2-graph-construction)
3. [Risk Modelling](#3-risk-modelling)
4. [Routing Algorithm](#4-routing-algorithm)
5. [Reinforcement Learning Agent](#5-reinforcement-learning-agent)
6. [Stress Testing & Monte Carlo](#6-stress-testing--monte-carlo)
7. [Economic Cascade Model](#7-economic-cascade-model)
8. [Visualisation Layer](#8-visualisation-layer)
9. [Application Architecture](#9-application-architecture)
10. [Data Flow End-to-End](#10-data-flow-end-to-end)
11. [Limitations & Extensions](#11-limitations--extensions)
12. [Concept Deep Dives](#12-concept-deep-dives)

---

## 1. Problem Formulation

### 1.1 Motivation

The Strait of Hormuz carries approximately 18–21 million barrels of oil per day — roughly 20% of global petroleum liquids. From a network-theory perspective this is a **single point of failure** with:

- **High betweenness centrality**: it lies on the shortest path between most Gulf producers and most global consumers
- **High flow dependency**: ~80% of shortest producer→consumer paths pass through it
- **Catastrophic failure impact**: bypass alternatives have lower capacity and significantly longer transit times

The conventional logistics objective is:

```
min Σ c(e)    for e in path(source, target)
```

This optimises for the normal case and is blind to tail risk. The goal of this POC is to replace it with:

```
min Σ [c(e) + α·t(e) + λ·r(e, t)]    for e in path(source, target)
```

Where `r(e, t)` is a time-varying, stochastically evolving risk score per edge.

### 1.2 Formal Definition

Define a directed graph:

```
G = (V, E)
```

**Node set V** — four categories:
| Category | Examples |
|----------|---------|
| Producers | Saudi Arabia, UAE, Iraq, Kuwait, Qatar |
| Chokepoints | Hormuz, Suez Canal, Strait of Malacca, Bab-el-Mandeb, Cape of Good Hope |
| Bypass/Transit Hubs | Yanbu, Fujairah, Indian Ocean Hub, Red Sea |
| Consumers | India, China, Japan, Europe, USA |

**Edge set E** — each edge `e = (u, v)` carries five attributes:

| Attribute | Symbol | Unit | Description |
|-----------|--------|------|-------------|
| `cost` | c(e) | USD index | Normalised shipping + operational cost |
| `time` | t(e) | days | Transit time in days |
| `capacity` | k(e) | MBD | Million barrels per day throughput ceiling |
| `base_risk` | μ(e) | [0,1] | Long-run equilibrium risk level |
| `risk` | r(e,t) | [0,1] | Current time-varying risk estimate |
| `hormuz_dependent` | — | bool | Whether disrupting Hormuz affects this edge |

**Routing objective:**

```
w(e, t) = c(e) + α·t(e) + λ·r(e, t)

path* = argmin Σ w(e, t)    over all paths from source to target
```

Parameters:
- `α` — time aversion weight (default 0.5)
- `λ` — risk aversion weight (default 10.0); higher values push routing away from risky edges

---

## 2. Graph Construction

### 2.1 Node Layout

Nodes are positioned using real-world latitude/longitude coordinates for geographical accuracy in the Plotly geo visualisation. Each node carries `{type, lat, lon}`.

```
build_oil_network() → nx.DiGraph
```

**Node types and their display properties:**

| Type | Colour | Marker Size | Role |
|------|--------|-------------|------|
| `producer` | #2ECC71 (green) | 14px | Oil-exporting nations |
| `chokepoint` | #E74C3C (red) | 18px | Strategically critical straits/canals |
| `bypass_hub` | #F39C12 (orange) | 10px | Alternative routing terminals |
| `hub` | #3498DB (blue) | 10px | Open-ocean routing waypoints |
| `consumer` | #9B59B6 (purple) | 14px | Importing nations/regions |

### 2.2 Edge Topology

The graph has **25 directed edges**. Key structural properties:

**Hormuz funnel:** All five Gulf producers have a direct edge into the `Hormuz` node. `Hormuz` has a single outbound edge to `Indian Ocean Hub` with capacity 20.0 MBD — the bottleneck edge. All six of these edges are flagged `hormuz_dependent=True`.

**Bypass routes:**
- `Saudi Arabia → Yanbu` (Saudi East-West/Petroline, **7.0 MBD** post-March 2026 upgrade — Fortune/Argus Media/CNBC, March 28 2026) — two structurally distinct onward legs:
  - **Northbound:** `Yanbu → Suez Canal` directly (~1,000nm, ~3 days, risk 0.10). Yanbu sits at 24°N, Suez at 30°N — ships sail straight up the Red Sea without touching Bab-el-Mandeb. This is the cleanest bypass in the network: avoids both Hormuz and Houthi-exposed southern corridor.
  - **Southbound:** `Yanbu → Red Sea → Bab-el-Mandeb → Indian Ocean Hub` (~7.5 days total). Used for Asia-bound cargo. Ships exit through the southern gate of the Red Sea into the Indian Ocean.
- `UAE → Fujairah` (ADCO/Habshan pipeline, 1.5 MBD, ADNOC) → Indian Ocean Hub directly

**Critical connectivity note:** The bypass graph is fully closed for all producer-consumer pairs. For Europe and USA, the Yanbu northbound leg bypasses both Hormuz and Bab-el-Mandeb. For Asian consumers, `Bab-el-Mandeb → Indian Ocean Hub` is the enabling edge — without it, no Hormuz-free path to India, China, or Japan exists.

**Consumer-facing legs:**
- `Indian Ocean Hub → India` (short haul, ~2 days)
- `Indian Ocean Hub → Strait of Malacca → China / Japan` (East Asia, ~20 + 5/8 days)
- `Indian Ocean Hub → Cape of Good Hope → Europe / USA / India / Strait of Malacca` (ultra-long bypass, lowest risk)
- `Suez Canal → Europe / USA` (reachable directly from Yanbu northbound, or via Bab-el-Mandeb from Indian Ocean)

**Complete edge list by category:**

| Category | Edges | Count |
|----------|-------|-------|
| Producers → Hormuz | SA, UAE, Iraq, Kuwait, Qatar | 5 |
| Bypass pipelines | SA→Yanbu, UAE→Fujairah | 2 |
| Hormuz outbound | Hormuz→IOH, Fujairah→IOH | 2 |
| Yanbu northbound | Yanbu→Suez | 1 |
| Red Sea southbound corridor | Yanbu→RS, RS→Bab, Bab→Suez, Bab→IOH | 4 |
| IOH → consumers/hubs | IOH→India, IOH→Malacca, IOH→Cape | 3 |
| Malacca → East Asia | Malacca→China, Malacca→Japan | 2 |
| Suez → West | Suez→Europe, Suez→USA | 2 |
| Cape → consumers | Cape→Europe, Cape→USA, Cape→India, Cape→Malacca | 4 |
| **Total** | | **25** |

### 2.3 Capacity Constraints

Capacity is stored per edge but is not enforced as a hard constraint in the current routing implementation — it is surfaced as a metric (bottleneck capacity along the chosen path). Full capacity-constrained routing would require min-cost max-flow, noted in §10.

---

## 3. Risk Modelling

### 3.1 Risk as a Stochastic Process

Risk is not a static scalar — it evolves continuously in response to geopolitical events, market signals, and random shocks. The chosen model is the **Ornstein-Uhlenbeck (OU) process**, a mean-reverting stochastic differential equation:

```
dR(t) = θ(μ - R(t))dt + σ dW(t)
```

| Parameter | Symbol | Value | Meaning |
|-----------|--------|-------|---------|
| Mean reversion speed | θ | 0.3 | How fast risk returns to baseline |
| Long-run mean | μ | `base_risk` per edge | Equilibrium risk level |
| Volatility | σ | `volatility × 0.12` | Scales random shock magnitude |
| Wiener increment | dW | N(0,1) | Random shock draw |

**Why OU?** Unlike a random walk, OU is mean-reverting — risk does not drift to 0 or 1 permanently absent new shocks. This mirrors real geopolitical risk: crises occur, escalate, and (usually) de-escalate back toward a baseline.

**Discrete-time implementation:**

```python
def simulate_step(G, volatility=0.3):
    for u, v in G.edges():
        mu      = G[u][v]["base_risk"]
        current = G[u][v]["risk"]
        theta   = 0.3
        sigma   = volatility * 0.12
        shock   = np.random.normal(0, 1)
        new_r   = current + theta * (mu - current) + sigma * shock
        G[u][v]["risk"] = float(np.clip(new_r, 0.0, 1.0))
```

Each call to `simulate_step` advances the simulation by one tick. The `volatility` parameter (0–1) is controlled via the sidebar slider and scales σ.

### 3.2 Hormuz Crisis Shock

A separate function applies a sudden, large risk spike to all Hormuz-dependent edges:

```python
def apply_hormuz_crisis(G, severity=0.88):
    for u, v in G.edges():
        if G[u][v].get("hormuz_dependent", False):
            G[u][v]["risk"] = float(np.clip(
                severity + random.uniform(-0.05, 0.05), 0, 1
            ))
```

`severity` is sampled uniformly in `[0.45, 0.95]` during Monte Carlo runs, and set to 0.90 for the deterministic crisis button. The ±0.05 jitter prevents unrealistic uniformity across edges.

### 3.3 Baseline Risk Values (Real Data Calibration)

`base_risk` is calibrated to **Lloyd's / S&P war-risk insurance premium bands** — the market's real-time aggregation of geopolitical risk into a price. Full sourcing in `DATA_SOURCES.md §5`.

| Route Segment | Base Risk | Insur. Premium Band | Source |
|---------------|-----------|---------------------|--------|
| Producers → Hormuz | 0.28 | ~0.25–0.50% hull | S&P Global / Lloyd's 2024 baseline |
| Hormuz → Indian Ocean | 0.28 | ~0.25% hull | Same Hormuz-dependent exposure |
| Saudi East-West Pipeline (Yanbu) | 0.08 | ~0.05–0.10% hull | Onshore, below Gulf tension zone |
| UAE Fujairah pipeline | 0.09 | ~0.05–0.10% hull | Onshore UAE, low war-risk |
| Red Sea / Bab-el-Mandeb | 0.35 | ~0.70% hull (2024) | AGBI / Maplecroft; down from 2.0% peak |
| Suez Canal | 0.18 | ~0.10–0.20% hull | IEA / EIA Suez analysis |
| Strait of Malacca | 0.05 | ~0.05% hull | Lloyd's JWC removed from high-risk May 2024 |
| Indian Ocean (open water) | 0.07 | ~0.05% hull | Standard open-ocean baseline |
| Cape of Good Hope | 0.02 | ~0.01% hull | Remote, stable maritime zone |

**Crisis severity benchmarks** (used in Monte Carlo / stress test):

| Scenario | Severity | Real Equivalent |
|----------|----------|-----------------|
| Moderate tension | 0.45 | Hormuz 2019 tanker attacks (~0.50% hull) |
| Serious escalation | 0.70 | Estimated major-incident level |
| Near-closure | 0.90 | Lloyd's March 2026 Gulf escalation (~1.0% hull) |
| Active conflict | 1.00 | 1980s Tanker War peak (~5% hull, ~WS 1000) |

### 3.4 Risk Signal Sources (Production Extension)

In a production system, `r(e, t)` would be estimated from:

| Signal Type | Source | Model |
|-------------|--------|-------|
| News sentiment | NLP on headlines/diplomatic cables | BERT-based classifier → sentiment score |
| Oil price volatility | OHLCV data | GARCH / realised variance |
| Insurance premiums | War-risk P&I club rates | Normalised delta vs rolling mean |
| AIS vessel data | MarineTraffic / exact Earth | Anomaly detection on ship density |
| Satellite imagery | SAR / optical | Congestion + military vessel classification |

All signals feed a **Bayesian update** on the prior risk distribution per edge, producing a calibrated posterior probability.

---

## 4. Routing Algorithm

### 4.1 Risk-Aware Dijkstra

Standard Dijkstra finds the minimum-weight path in a non-negative weighted graph. Risk-aware Dijkstra redefines the weight function to incorporate time and risk alongside cost:

```python
def edge_weight(G, u, v, alpha=0.5, lam=10.0):
    e = G[u][v]
    return e["cost"] + alpha * e["time"] + lam * e["risk"]

def risk_dijkstra(G, source, target, alpha=0.5, lam=10.0):
    pq = [(0.0, source, [source])]   # (cumulative_weight, node, path)
    visited = set()
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == target:
            return round(cost, 2), path
        for nb in G.neighbors(node):
            if nb not in visited:
                w = edge_weight(G, node, nb, alpha, lam)
                heapq.heappush(pq, (cost + w, nb, path + [nb]))
    return float("inf"), []
```

**Complexity:** O((V + E) log V) — identical to standard Dijkstra since the modification is only to the weight function.

**Path reconstruction:** The path is tracked inline in the heap tuple. This trades memory for simplicity; a production implementation would use a predecessor map.

### 4.2 The λ Threshold Effect

The most important emergent behaviour is the **λ switchover**. As λ increases from 0:

1. At low λ, the Hormuz route dominates (cheap, fast, but risky)
2. At a critical λ*, the risk penalty on Hormuz edges outweighs the cost advantage of bypass routes
3. Above λ*, the algorithm switches to bypass routing (more expensive, longer, safer)

This threshold is not set manually — it emerges from the graph structure and current risk values. The Route Finder tab sweeps λ across [0, 50] to reveal this Pareto frontier explicitly.

**Implication:** The cost difference between the Hormuz path and the optimal bypass path at λ* represents the **market price of resilience** — what it would cost to route safely under current risk conditions.

### 4.3 Path Statistics

For every computed path, the following statistics are derived:

```python
def path_stats(G, path):
    pairs = [(path[i], path[i+1]) for i in range(len(path)-1)]
    return {
        "Cost":             sum(G[u][v]["cost"] for u,v in pairs),
        "Transit (days)":   sum(G[u][v]["time"] for u,v in pairs),
        "Max Risk":         max(G[u][v]["risk"] for u,v in pairs),
        "Avg Risk":         mean([G[u][v]["risk"] for u,v in pairs]),
        "Hormuz Exposed":   any(G[u][v]["hormuz_dependent"] for u,v in pairs),
        "Bottleneck (MBD)": min(G[u][v]["capacity"] for u,v in pairs),
    }
```

`Bottleneck (MBD)` is the minimum-capacity edge on the path — the true throughput ceiling regardless of production or demand.

---

## 5. Reinforcement Learning Agent

### 5.1 Motivation

Dijkstra solves a static snapshot of the graph. In practice:
- Risk values change continuously
- Re-running Dijkstra on every tick is reactive, not adaptive
- The algorithm has no memory of what worked under similar conditions in the past

A Q-learning agent addresses this by learning a **policy** — a mapping from (state, action) → expected cumulative reward — that generalises across graph states.

### 5.2 MDP Formulation

The routing problem is cast as a Markov Decision Process:

| Component | Definition |
|-----------|-----------|
| **State** s | (current_node, discretised_risk_vector) |
| **Action** a | Choose next node to move to (from neighbours) |
| **Reward** R | -(cost + 40·risk + 2·time) + 100·[reached target] |
| **Transition** | Deterministic: moving to action node |
| **Discount** γ | 0.9 |

**State representation:**

```python
def _state(self, G, node):
    risks = tuple(
        round(G[u][v]["risk"] * 4) / 4      # discretise to 0.25 buckets
        for u, v in sorted(G.edges())
    )
    return (node, risks)
```

Risk is discretised to 5 levels (0, 0.25, 0.5, 0.75, 1.0) to keep the state space tractable. With 25 edges and 5 risk levels, the theoretical state space is 19 × 5²⁵ — in practice, only a tiny fraction of states is visited during training. This is the fundamental limitation of tabular Q-learning on this problem (see §11.1).

**Reward shaping:** The +100 terminal reward for reaching the target is critical — without it, the agent learns to wander (accumulating small penalties is preferable to the large cost of any single high-risk edge).

**Risk coefficient rationale:** The risk coefficient of 40 is calibrated so that at crisis severity (risk ≈ 0.9), the two Hormuz-dependent edges to the Indian Ocean Hub accumulate enough negative reward (~−78) to make the longer bypass route (~−68 to reach the same hub) genuinely preferable. At base risk (0.28), Hormuz remains cheaper, so the agent correctly uses it under normal conditions and reroutes only under genuine crisis. A lower coefficient (e.g. 10) fails because the bypass path's extra hops accumulate more step penalty than the risk savings provide.

**Dead-end penalty:** If the agent navigates into a node with no valid onward path to the target (e.g. Suez Canal when the destination is Japan — Suez connects only to Europe and USA), the episode terminates and a −100 penalty is applied to the move that caused the dead end. This prevents the agent from repeatedly routing into structural dead ends for specific source-target pairs.

### 5.3 Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ · max_a' Q(s',a') − Q(s,a)]
```

Implementation:

```python
def _update(self, G, s_node, action, reward, next_node):
    state      = self._state(G, s_node)
    next_state = self._state(G, next_node)
    next_nbrs  = list(G.neighbors(next_node))
    best_next  = max(
        (self.Q[(next_state, a)] for a in next_nbrs), default=0.0
    )
    self.Q[(state, action)] += self.alpha * (
        reward + self.gamma * best_next - self.Q[(state, action)]
    )
```

**Hyperparameters:**

| Parameter | Value | Role |
|-----------|-------|------|
| Learning rate α | 0.15 | Step size for Q updates |
| Discount γ | 0.9 | Future reward weighting |
| Initial ε | 0.5 | Exploration probability at start |
| ε decay | ×0.994 per episode | Shifts from exploration to exploitation |
| Min ε | 0.05 | Retains residual exploration |

### 5.4 Training Loop

```python
def train(self, G, source, target, episodes=300):
    rewards = []
    for _ in range(episodes):
        self.epsilon = max(0.05, self.epsilon * 0.994)
        _, r = self._episode(G, source, target)
        rewards.append(r)
    return rewards
```

Each episode:
1. Start at `source`
2. At each step: ε-greedy action selection (random with prob ε, greedy Q-value otherwise)
3. Move to selected neighbour, compute reward, update Q-table
4. If the chosen node has no valid onward neighbours (dead end), apply −100 penalty to the move that caused it, then terminate
5. Terminate at `target` or after `max_steps=30`

**Visited set:** Nodes already visited in the current episode are excluded from action selection — this prevents cycles and is biologically reasonable (a tanker doesn't revisit a port it just left).

**Dead-end detection:** Non-target consumer nodes are excluded from the action set. If this leaves no valid neighbours (e.g. reaching Suez Canal when the target is Japan — Suez only connects to Europe and USA), the episode terminates. The −100 penalty is written back to the Q-entry for the *previous* (node, action) pair, not the current dead-end node, so the agent learns to avoid the move that caused the trap.

### 5.5 Greedy Policy Extraction

After training, the greedy policy is extracted by running one episode with ε=0:

```python
def greedy_path(self, G, source, target):
    old, self.epsilon = self.epsilon, 0.0
    path, _ = self._episode(G, source, target)
    self.epsilon = old
    return path
```

This path is compared directly against the Dijkstra output in the UI.

### 5.6 Expected Behaviour

- **Early episodes:** High ε causes random walks; most episodes fail to reach the target; rewards are very negative
- **Mid training:** Agent learns which nodes are dead ends; starts finding target-reaching paths
- **Late episodes:** ε has decayed; agent consistently follows learned high-value paths; reward curve plateaus
- **Convergence:** Q-values stabilise when the TD error (term in brackets in the update rule) consistently approaches zero

---

## 6. Stress Testing & Monte Carlo

### 6.1 Deterministic Crisis Scenario

The "Hormuz Crisis" button:
1. Applies `apply_hormuz_crisis(G, severity=0.90)` to the live graph in session state
2. Re-runs `risk_dijkstra` with current α and λ
3. Logs the result to `risk_history` for display in the Risk Simulator tab
4. Sets `crisis_active=True` flag in session state

The Stress Test tab runs a **fresh, isolated scenario** (creates two separate graph instances — `G_normal` and `G_crisis`) to produce clean before/after comparisons independent of the live simulation state.

### 6.2 Monte Carlo Simulation

500 independent disruption scenarios are run inline in the Stress Test tab:

```python
for _ in range(500):
    G_mc = build_oil_network()               # fresh graph each iteration
    sev  = random.uniform(0.45, 0.95)        # random severity
    apply_hormuz_crisis(G_mc, severity=sev)
    _, p = risk_dijkstra(G_mc, source, target, alpha, lam)
    s    = path_stats(G_mc, p)
    mc.append({
        "severity":     sev,
        "actual_cost":  s["Cost"],
        "transit_days": s["Transit (days)"],
        "hormuz_free":  not s["Hormuz Exposed"],
    })
```

**Key design decisions:**
- Each iteration creates a **fresh** `build_oil_network()` — this ensures independence between scenarios
- `severity` is drawn from U(0.45, 0.95) — covering the range from moderate tension to near-complete closure
- The ±0.05 jitter in `apply_hormuz_crisis` adds within-scenario randomness

**Output metrics:**
1. **Cost distribution histogram** — shows spread of actual (not weighted) route costs across scenarios
2. **Rerouting gauge** — % of scenarios where the optimal path is Hormuz-free
3. **Summary statistics** — average cost increase, worst-case transit time, high-severity rerouting rate

---

## 7. Economic Cascade Model

### 7.1 Motivation

Routing cost is an operational abstraction. The question a policymaker, central banker, or food security minister needs answered is different: *what happens to the real economy* when Hormuz is disrupted? This model translates a disruption (characterised by severity and duration) into a quantified macro-economic impact chain.

### 7.2 Oil Price Scenario (`oil_price_scenario`)

**Inputs:** `hormuz_risk` (severity), `duration_days`, `base_price` (Brent crude USD/bbl)

**Model:**

```
supply_disrupted = hormuz_risk × 0.20     # Hormuz ≈ 20% of global supply (EIA 2024)

pct_change = net_disruption × dur_mult × 100 + panic × 100
```

**Duration multiplier** (calibrated to Abqaiq 2019 and Gulf War 1990):

| Duration | Multiplier | Rationale |
|----------|-----------|-----------|
| ≤ 7 days | 3.5× | Markets expect short disruption; SPR can absorb |
| 8–30 days | 5.5× | Strategic reserve deployment underway |
| 31–90 days | 8.0× | Rerouting via Cape established; lingering premium |
| > 90 days | 12.0× | Demand destruction + structural rebalancing |

**Offsets:**
- OPEC spare capacity: up to 35% of disrupted supply, capped at 7% of global supply (IEA Oil Market Report 2024 — ~3.5–4.5 MBD available)
- IEA strategic reserve: 4 MBD release capacity for ~30 days (IEA Emergency Response Manual)
- Panic premium: `max(0, (severity − 0.5) × 0.40 × 100%)` applied above severity=0.5

**Freight premium model:**

```
cape_extra_cost = 14 days × $45,000/day bunker
freight_premium = (cape_extra_cost / cargo_value) × 100 + hormuz_risk × 250
```

Calibrated to: 2019 Hormuz attacks (TD3C WS 60→300, +400%), Houthi period (+150%).

**Sources:** Hamilton (2009) NBER; Kilian (2008) J. Econ. Literature; IEA Emergency Response Manual; EIA "How much of the retail price of gasoline is tax?"

### 7.3 Inflation Cascade (`inflation_cascade`)

Computes headline CPI, core CPI, food price, GDP, and rate hike impacts per region.

**Regional parameters (5 regions):**

| Region | Oil Import Dep. | CPI Pass-Through | GDP Elasticity | Food Dep. |
|--------|----------------|-----------------|----------------|-----------|
| East Asia (Japan/Korea/China) | 85% | 0.18 | −0.040 | 0.55 |
| India | 85% | 0.16 | −0.050 | 0.48 |
| Europe | 55% | 0.13 | −0.028 | 0.30 |
| USA | 15% | 0.08 | −0.015 | 0.20 |
| Developing Markets | 80% | 0.22 | −0.060 | 0.65 |

**Sources:** IMF WP/17/53 (Gelos & Ustyugova 2017) — CPI pass-through; IEA Energy Security 2023; World Bank Commodity Markets Outlook 2022.

**Per-region computation:**

```
eff_oil_chg  = oil_pct_change × import_dep
headline_cpi = eff_oil_chg × cpi_pt
core_cpi     = headline_cpi × 0.38          # second-round effects, 2–4 month lag

food_chg     = energy_to_food               # 15% of oil chg (direct energy in production)
             + fertilizer_cost              # gas-oil correlation (0.60) × food share (0.18)
             + freight_to_food             # freight × food import dep
             + panic_food                  # hoarding premium above 25% oil price rise

direct_gdp   = eff_oil_chg × gdp_elast
monetary_drag = −0.15 × max(0, headline_cpi − 0.5)   # CB rate hike effect
total_gdp    = direct_gdp + monetary_drag

rate_hike    = max(0, 1.5 × headline_cpi × 0.1)      # Taylor rule approximation
```

**Food price validation:** 2022 Russia sanctions — oil +60% → FAO food index +34% (elasticity ≈ 0.57). Model reproduces this at `oil_pct_change=60`.

### 7.4 Economic Time Series (`economic_time_series`)

Simulates day-by-day indicator evolution over 180 days using a 5-phase model:

| Phase | Days | Oil Price Factor | Description |
|-------|------|-----------------|-------------|
| Shock | 0–3 | 0 → 1.0 (√ ramp) | Rapid price spike |
| Peak | 4–7 | 1.0 | Plateau; reserves announced |
| Reserve Deployment | 8–30 | 1.0 → 0.70 | SPR + OPEC moderation |
| Cape Rerouting | 31–90 | 0.70 → 0.50 | New shipping equilibrium |
| New Equilibrium | 91–180 | 0.50 → 0.40 | Demand destruction |
| Recovery (post-disruption) | d > duration | Exponential decay | Snap-back + overshoot correction |

**Lagged indicators:**
- CPI: 30-day lag on oil price (supply chain pipeline pass-through)
- Food prices: 45-day lag + freight component
- Petrol at pump: 7-day lag (weekly repricing cycle)

### 7.5 Monte Carlo Economic Distribution (`monte_carlo_economic`)

Runs 500 random scenarios sampling:
- `severity ~ U(0.30, 0.95)`
- `duration` from `{3, 7, 14, 30, 60, 90, 180}` days

Reports distribution of oil price change, global CPI, global food price, and global GDP impact. The 95th percentile of each distribution is the tail-risk figure planners should design for.

### 7.6 Historical Calibration

Seven real events are used to validate model coefficients:

| Event | Duration | Oil Δ% | CPI Peak | Food Δ% | GDP Impact |
|-------|----------|--------|---------|---------|-----------|
| 1973 Arab Embargo | 150d | +400% | +11.0% | +30% | −2.5% |
| 1979 Iranian Revolution | 365d | +150% | +13.5% | +20% | −3.5% |
| 1990 Gulf War | 180d | +100% | +6.2% | +15% | −1.5% |
| 2005 Hurricane Katrina | 30d | +25% | +3.4% | +5% | −0.3% |
| 2019 Abqaiq Attack | 14d | +15% | +0.2% | +2% | −0.1% |
| 2022 Russia Sanctions | 365d | +60% | +9.1% | +34% | −1.0% |
| 2023–24 Houthi/Red Sea | 365d | +8% | +0.3% | +8% | −0.3% |

Sources: Hamilton (1983, 2009); Kilian (2008 AER); IMF WEO Oct 2022; EIA/IEA 2024.

### 7.7 UI Components (Economic Cascade Tab)

| Component | Description |
|-----------|-------------|
| KPI metrics row (6 cols) | Brent crude spot, petrol price, freight premium, global CPI, food prices, GDP |
| 6-month time series chart | Oil price, CPI, food prices, freight rates with phase annotations and disruption-end marker |
| Regional bar charts (3 cols) | Headline CPI / food prices / GDP per region |
| Full regional table | Import dependency, effective oil Δ, all indicators per region |
| Sankey transmission chain | Hormuz → Oil Supply → Energy/Freight → Manufacturing/Food → CPI → Rate Hikes → GDP |
| Historical comparison table + scatter | Current scenario vs 7 real events; linear trend line |
| Monte Carlo histograms (3 cols) | Oil price / CPI / food price distributions with mean and current-scenario markers |

---

## 8. Visualisation Layer

### 7.1 Network Map (Plotly Scattergeo)

The network is drawn on a Plotly `Scattergeo` figure — a world map projection with overlay traces.

**Edge rendering:** One trace per edge (25 traces total). Each trace is a 3-point line `[lon_u, lon_v, None]` / `[lat_u, lat_v, None]` — the `None` breaks the line so edges don't connect to each other.

```python
fig.add_trace(go.Scattergeo(
    lon=[u_node["lon"], v_node["lon"], None],
    lat=[u_node["lat"], v_node["lat"], None],
    mode="lines",
    line=dict(
        width = 5 if is_path_edge else max(1.0, capacity / 5),
        color = "#FFD700" if is_path_edge else _risk_rgb(risk),
    ),
    opacity = 1.0 if is_path_edge else 0.60,
))
```

**Risk-to-colour mapping:**

```python
def _risk_rgb(risk):
    r = int(255 * risk)        # 0 → 0, 1 → 255
    g = int(255 * (1 - risk))  # 0 → 255, 1 → 0
    return f"rgb({r},{g},50)"  # green → yellow → red gradient
```

**Optimal path highlighting:** Path edges are drawn in gold (`#FFD700`) at width 5. Path nodes get a gold border (line width 3) to visually distinguish them from non-path nodes.

**Map style:**
- Land: `#1a1a2e` (dark navy)
- Ocean: `#0d1117` (near-black)
- Coastlines: `#2d3748` (subtle grey)
- Projection: Natural Earth

### 7.2 Risk Evolution Chart

A filled area chart (`go.Scatter` with `fill="tozeroy"`) shows Hormuz-route risk over simulation ticks. A dashed horizontal reference line marks the baseline risk (0.25). Crisis events appear as vertical dotted lines wherever `hormuz_risk > 0.70`.

### 7.3 Cost vs Risk Pareto Chart

A dual-axis chart sweeps λ from 0→50 in 40 steps, computing the optimal path at each point and recording actual cost and max risk. This directly visualises the Pareto frontier between cost efficiency and risk exposure.

### 7.4 Monte Carlo Histogram + Gauge

- `px.histogram` on the `actual_cost` column — shows the distribution of route costs across 500 scenarios
- `go.Indicator` gauge — shows what percentage of scenarios successfully rerouted away from Hormuz

### 7.5 RL Training Curve

Episode rewards are plotted raw (thin, transparent) and with a rolling mean (thick, orange). The window size is `max(10, n_episodes // 20)`. The upward trend in smoothed rewards confirms the agent is learning.

---

## 9. Application Architecture

### 9.1 Single-File Design

All logic — graph construction, risk engine, routing, RL agent, visualisation, and UI — lives in `app.py`. This is intentional for a POC: no import machinery, no packaging, no relative path issues.

### 9.2 Streamlit Session State

Streamlit reruns the entire script on every user interaction. Mutable state that must persist across reruns is stored in `st.session_state`:

| Key | Type | Description |
|-----|------|-------------|
| `G` | `nx.DiGraph` | Live graph with current risk values |
| `t` | `int` | Current simulation tick |
| `risk_hist` | `list[dict]` | History of risk/cost/path per tick |
| `rl_agent` | `QLearningAgent \| None` | Trained agent instance |
| `rl_rewards` | `list[float]` | Episode reward history |
| `crisis` | `bool` | Whether crisis is currently active |

Initialisation guard:

```python
if "G" not in st.session_state:
    st.session_state.G          = build_oil_network()
    st.session_state.t          = 0
    st.session_state.risk_hist  = []
    st.session_state.rl_agent   = None
    st.session_state.rl_rewards = []
    st.session_state.crisis     = False
```

### 9.3 Sidebar Controls and Side Effects

| Control | Effect |
|---------|--------|
| Source / Target | Changes routing endpoints; re-computes path on next render |
| α slider | Changes time weight in `edge_weight`; re-computes path |
| λ slider | Changes risk weight in `edge_weight`; triggers route switchover |
| Volatility slider | Changes σ in `simulate_step`; affects next step |
| ▶ Step | Calls `simulate_step`, advances t, appends to `risk_hist` |
| ⏩ ×10 | Calls `simulate_step` 10 times in a loop |
| 🔥 Hormuz Crisis | Calls `apply_hormuz_crisis(G, 0.90)`, sets `crisis=True` |
| 🔄 Reset | Deletes all session state keys, calls `st.rerun()` |

### 9.4 Tab Structure

Each tab is a logical section of the interactive blog:

```
tab1: Network Map        → draw_network() + path_stats()
tab2: Route Finder       → risk_dijkstra() sweep over λ + Pareto chart
tab3: Risk Simulator     → risk_hist time series charts
tab4: Stress Test        → isolated G_normal/G_crisis + Monte Carlo loop
tab5: RL Agent           → QLearningAgent.train() + greedy_path()
tab6: Economic Cascade   → oil_price_scenario() + inflation_cascade() + economic_time_series()
                           + monte_carlo_economic() + Sankey + historical comparison
tab7: How It Works       → first-principles concept explainers
```

Tabs 4, 5, and 6 create their own isolated graph instances or compute independently to avoid contaminating the live simulation graph `st.session_state.G`.

---

## 10. Data Flow End-to-End

```
User interaction (sidebar / button)
        │
        ▼
Session state mutation
  ├── simulate_step(G, volatility)
  │     └── OU process updates G[u][v]["risk"] for all edges
  ├── apply_hormuz_crisis(G, severity)
  │     └── Overrides risk on hormuz_dependent edges
  └── reset → rebuild G from build_oil_network()
        │
        ▼
Routing computation
  risk_dijkstra(G, source, target, α, λ)
    ├── Reads current G[u][v]["risk"] for all traversed edges
    ├── Computes edge_weight = cost + α·time + λ·risk
    └── Returns (weighted_cost, path_list)
        │
        ▼
Statistics extraction
  path_stats(G, path)
    └── Returns {Cost, Transit, MaxRisk, AvgRisk, HormuzExposed, Bottleneck}
        │
        ▼
History logging (Risk Simulator tab)
  risk_hist.append({t, hormuz_risk, path_cost, path_string})
        │
        ▼
Visualisation
  draw_network(G, path)
    ├── 25 edge traces (risk-coloured, width ∝ capacity)
    ├── Path edges drawn in gold at width 5
    └── Node markers coloured by type, gold border if in path
        │
        ▼
Streamlit render → Browser

─── Parallel: Monte Carlo (Tab 4) ───────────────────────────
500 × [
  G_mc = build_oil_network()
  apply_hormuz_crisis(G_mc, U(0.45, 0.95))
  risk_dijkstra(G_mc, source, target, α, λ)
  path_stats(G_mc, path)
] → DataFrame → histogram + gauge

─── Parallel: RL Training (Tab 5) ───────────────────────────
QLearningAgent.train(G, source, target, n_episodes)
  └── n_episodes × [
        ε-greedy episode on live G
        Q-table update via Bellman equation
      ]
  → rewards list → training curve chart
  → greedy_path(G, source, target) → compare vs Dijkstra

─── Parallel: Economic Cascade (Tab 6) ──────────────────────
oil_price_scenario(severity, duration, base_price)
  └── supply disruption + duration multiplier + offsets → oil Δ%, freight premium

inflation_cascade(oil_pct_change, freight_pct, duration)
  └── per region: headline CPI, core CPI, food Δ, GDP impact, rate hike

economic_time_series(severity, duration, base_price, n_days=180)
  └── day-by-day: oil price, CPI (30d lag), food (45d lag), petrol (7d lag), freight
  → multi-trace Plotly chart with phase annotations

monte_carlo_economic(n=500, base_price)
  └── 500 × [random severity, random duration → oil_price_scenario + inflation_cascade]
  → DataFrames → 3× histogram with mean + current-scenario markers

→ Sankey diagram (transmission chain)
→ Historical comparison table + scatter
```

---

## 11. Limitations & Extensions

### 11.1 Current Limitations (v1)

| Limitation | Description | v2 Status |
|------------|-------------|-----------|
| Capacity not enforced | Edge capacity is surfaced as a metric but not a hard constraint. Flow can exceed `k(e)`. | Still open |
| Static graph topology | Edges don't appear/disappear. In reality, canal closures and pipeline shutdowns remove edges entirely. | Still open |
| Simulated risk only | Risk is generated by OU process, not real signals. | **LSTM in v2** trains on correlated synthetic signals (oil vol, insurance, sentiment) |
| Single-commodity flow | Model routes one flow at a time. Real logistics involves simultaneous multi-commodity flows. | Still open |
| Q-table scalability | Tabular Q-learning doesn't scale — 19 × 5²⁵ theoretical state space, almost entirely unvisited. | **DQN in v2** generalises via neural interpolation |
| Economic model uses simulated risk | `oil_price_scenario` uses the slider severity, not live graph risk — cascade is a scenario tool, not a live feed. | Accepted trade-off for legibility |

### 11.2 Direct Extensions

**Capacity-constrained routing (Min-Cost Max-Flow):**

Replace Dijkstra with successive shortest path algorithm operating on residual capacity:
```
min Σ c(e)·f(e)    subject to:
  Σ f(e) ≤ k(e)   for all e
  flow conservation at all non-source/sink nodes
```

**Multi-objective Pareto optimisation:**

Instead of scalarising cost/time/risk into a single weight, compute the full Pareto frontier across two or three objectives using NSGA-II or epsilon-constraint method.

**Deep Q-Network (DQN):** *(implemented in v2)*

v2 replaces the tabular Q-function with a 3-layer neural network (41→256→128→19 dims) with experience replay, frozen target network, Huber loss, and action masking. See `v2/TECHNICAL_V2.md §4` for the full implementation.

**LSTM Risk Predictor:** *(implemented in v2)*

v2 trains a 2-layer LSTM on correlated synthetic risk signals (risk, oil volatility, insurance premium, sentiment) to predict next-step edge risks across all edges simultaneously. See `v2/TECHNICAL_V2.md §3`.

**Economic cascade model:** *(implemented in v1 and v2)*

v1 includes a full macro transmission model: oil price shock → freight premium → regional CPI → food prices → GDP contraction, calibrated to 7 historical disruptions with IMF WP/17/53 pass-through coefficients. See §7 above for full specification.

**Live data integration:**

```
News API → sentiment classifier → r_news(e, t)
Shipping API (AIS) → anomaly score → r_ais(e, t)
Options market IV → volatility surface → r_mkt(e, t)

r(e, t) = Bayesian combination of [r_news, r_ais, r_mkt]
```

**Bayesian risk updating:**

```
P(disruption | signals) ∝ P(signals | disruption) × P(disruption)
```

Update prior per edge using conjugate Beta-Binomial model as signals arrive.

**Survival analysis:**

Model "time until disruption" as a survival function with hazard rate λ(t) driven by risk signals. This gives a disruption probability over a planning horizon rather than an instantaneous risk score.

---

---

## 12. Concept Deep Dives

This section explains the core mathematical ideas from first principles — intended for readers who want to understand *why* each model is built the way it is, not just what it does.

### 12.1 Graph Theory & Network Representation

**What is a graph?**

A graph G = (V, E) is a set of nodes V connected by edges E. In this model:

```
Nodes V:   anything oil passes THROUGH
           (countries, straits, pipeline terminals, ocean waypoints)

Edges E:   directed connections between nodes
           each edge e carries: cost · time · capacity · risk(t)
```

**Why directed?** Oil flows one way. A VLCC leaving Ras Tanura bound for Yokohama doesn't return through Hormuz on the same trip. The graph is directed because the logistics are asymmetric — different routes have different costs and risks depending on direction.

**Betweenness Centrality — why Hormuz is a structural vulnerability:**

Betweenness centrality measures how often node v appears on shortest paths between all other pairs:

```
C_B(v) = Σ_{s≠v≠t} σ_st(v) / σ_st
```

where σ_st = number of shortest paths from s to t, and σ_st(v) = those passing through v.

A node with C_B → 1.0 is on *almost every* shortest path in the network. Remove it and the network fragments. Hormuz has the highest betweenness centrality in this graph — ~80% of producer-to-consumer shortest paths pass through it. No engineering fix (adding capacity, raising pipeline throughput) reduces this without building structurally parallel paths that bypass the node entirely.

---

### 12.2 Risk-Aware Dijkstra

**Standard Dijkstra:** finds minimum-weight path in O((V+E) log V) using a min-heap priority queue. Always expands the cheapest reachable node next. Proven optimal for non-negative edge weights.

**Our modification:** change the weight function only. Everything else is identical.

```
Standard:       w(e) = cost(e)
Risk-aware:     w(e,t) = cost(e) + α·time(e) + λ·risk(e,t)
```

This is still a non-negative-weight shortest-path problem, so Dijkstra's optimality guarantee holds.

**The λ switchover mechanism:**

At low λ: Hormuz path cost < bypass path cost → Hormuz wins  
At λ = λ*: both paths have equal weighted cost → indifferent  
At λ > λ*: bypass weighted cost < Hormuz weighted cost → bypass wins  

λ* is not set manually. It emerges from the current risk values on every edge. The Route Finder tab sweeps λ ∈ [0, 50] in 40 steps to reveal this frontier explicitly.

**λ* as "the price of resilience":** the actual cost difference between the Hormuz path and the bypass path at λ* is what the market would need to offer to make safe routing economically rational under current risk conditions. Historically, shippers pay this only when forced.

---

### 12.3 Ornstein-Uhlenbeck Risk Simulation

**Why risk must be modelled as a stochastic process:**

Risk is not observable — it is inferred from signals (insurance premiums, news sentiment, AIS anomalies). It is not static — it evolves in response to geopolitical events. And it is mean-reverting — crises escalate and (usually) de-escalate back toward some baseline level.

**The OU process:**

```
dR(t) = θ(μ - R(t))dt + σ dW(t)
```

| Term | Role |
|------|------|
| θ(μ - R(t))dt | Deterministic drift: pulls R toward μ proportional to distance |
| σ dW(t) | Stochastic term: random shock, scaled by volatility σ |

**Mean reversion intuition:** when R > μ (elevated risk), the drift term is negative — risk is pulled down. When R < μ (suppressed risk), drift is positive — risk is pulled up. The strength of the pull scales with θ. At θ=0 the process is a pure random walk.

**Discrete-time Euler approximation (one tick):**

```
R(t+1) = R(t) + θ(μ - R(t)) + σ · ε
where ε ~ N(0,1)
```

**Calibration:** μ = base_risk per edge, derived from Lloyd's/S&P war-risk insurance premium bands. σ = volatility × 0.12, where volatility is user-controlled (0 to 1). At volatility=0.30 (default), σ ≈ 0.036 per tick — consistent with real daily premium fluctuations during periods of moderate tension.

**Crisis shock:** `apply_hormuz_crisis(G, severity)` bypasses the OU process and directly sets risk on all hormuz_dependent edges to `severity ± U(-0.05, 0.05)`. This models a sudden geopolitical event (e.g. an attack on tankers) that instantly reprices war-risk premiums.

---

### 12.4 Q-Learning: From Table to Policy

**The limitation of Dijkstra:** it solves the current graph state optimally, but has no memory. Every time the graph changes (risk evolves), Dijkstra recomputes from scratch. It cannot generalise across states.

**The MDP formulation:**

```
State  s:  (current_node, hormuz_risk_bucket)
           19 possible nodes × 5 risk buckets = 95 reachable states

Action a:  choose next node from unvisited neighbours
           (non-target consumer nodes excluded to prevent dead-ends)

Reward R:  -(cost + 40·risk + 2·time)   per edge traversed
           + 100.0                        if action == target
           - 100.0                        if action leads to a dead end (no onward path to target)

Discount γ = 0.9 (future rewards worth 90% of immediate rewards)
```

**The Q-function:** Q(s, a) estimates the *total expected future reward* from taking action a in state s, then following the optimal policy thereafter.

**Bellman optimality equation:**

```
Q*(s,a) = R(s,a) + γ · max_{a'} Q*(s', a')
```

The optimal Q-function satisfies this self-consistency condition. Q-learning converges to Q* by iteratively applying:

```
Q(s,a) ← Q(s,a) + α [r + γ·max_{a'} Q(s',a') − Q(s,a)]
```

The bracket `[r + γ·max Q(s',a') − Q(s,a)]` is the **TD error** (temporal difference) — the "surprise" between what we expected and what we actually got. We nudge Q toward the target by fraction α.

**Why training uses varied risk regimes:** training on a single risk level means the Q-table only has meaningful entries for one Hormuz risk bucket. The agent then gives identical paths regardless of current risk. Training cycles through four regimes (normal, elevated, crisis, recovery) so all 5 × 19 = 95 states accumulate real Q-values, and the greedy policy genuinely differs between normal and crisis conditions.

**ε-greedy decay:**

```
ε(ep) = max(0.05, 0.5 × 0.994^ep)

Episode  0:   ε = 0.50  (50% random exploration)
Episode 100:  ε = 0.30
Episode 300:  ε = 0.17
Episode 600:  ε = 0.05  (5% residual exploration — never fully greedy)
```

---

*End of v1 Technical Document*  
*See also: `v2/TECHNICAL_V2.md` (LSTM + DQN upgrades), `DATA_SOURCES.md` (real data citations)*
