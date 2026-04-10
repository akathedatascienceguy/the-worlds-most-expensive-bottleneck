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
7. [Visualisation Layer](#7-visualisation-layer)
8. [Application Architecture](#8-application-architecture)
9. [Data Flow End-to-End](#9-data-flow-end-to-end)
10. [Limitations & Extensions](#10-limitations--extensions)

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

The graph has 22 directed edges. Key structural properties:

**Hormuz funnel:** All five Gulf producers have a direct edge into the `Hormuz` node. `Hormuz` has a single outbound edge to `Indian Ocean Hub` with capacity 18.0 MBD — the bottleneck edge. All six of these edges are flagged `hormuz_dependent=True`.

**Bypass routes:**
- `Saudi Arabia → Yanbu` (East-West Pipeline, capacity 2.0 MBD) → Red Sea → Suez/Bab-el-Mandeb
- `UAE → Fujairah` (Abu Dhabi Crude Oil Pipeline, capacity 1.5 MBD) → Indian Ocean Hub

**Consumer-facing legs:**
- `Indian Ocean Hub → India` (short haul)
- `Indian Ocean Hub → Strait of Malacca → China / Japan` (East Asia)
- `Suez Canal / Bab-el-Mandeb → Europe / USA` (West)
- `Cape of Good Hope → Europe / USA` (long-haul bypass, lowest risk, highest time)

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
| **Reward** R | -(cost + 10·risk + 2·time) + 50·[reached target] |
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

Risk is discretised to 5 levels (0, 0.25, 0.5, 0.75, 1.0) to keep the state space tractable. With 22 edges and 5 risk levels, the theoretical state space is 19 × 5²² — in practice, only a small subset of states is visited during training.

**Reward shaping:** The +50 terminal reward for reaching the target is critical — without it, the agent learns to wander (accumulating small penalties is preferable to the large cost of any single high-risk edge).

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
4. Terminate at `target` or after `max_steps=30`

**Visited set:** Nodes already visited in the current episode are excluded from action selection — this prevents cycles and is biologically reasonable (a tanker doesn't revisit a port it just left).

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

## 7. Visualisation Layer

### 7.1 Network Map (Plotly Scattergeo)

The network is drawn on a Plotly `Scattergeo` figure — a world map projection with overlay traces.

**Edge rendering:** One trace per edge (22 traces total). Each trace is a 3-point line `[lon_u, lon_v, None]` / `[lat_u, lat_v, None]` — the `None` breaks the line so edges don't connect to each other.

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

## 8. Application Architecture

### 8.1 Single-File Design

All logic — graph construction, risk engine, routing, RL agent, visualisation, and UI — lives in `app.py`. This is intentional for a POC: no import machinery, no packaging, no relative path issues.

### 8.2 Streamlit Session State

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

### 8.3 Sidebar Controls and Side Effects

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

### 8.4 Tab Structure

Each tab is a logical section of the interactive blog:

```
tab1: Network Map      → draw_network() + path_stats()
tab2: Route Finder     → risk_dijkstra() sweep over λ + Pareto chart
tab3: Risk Simulator   → risk_hist time series charts
tab4: Stress Test      → isolated G_normal/G_crisis + Monte Carlo loop
tab5: RL Agent         → QLearningAgent.train() + greedy_path()
```

Tabs 4 and 5 create their own isolated graph instances to avoid contaminating the live simulation graph `st.session_state.G`.

---

## 9. Data Flow End-to-End

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
    ├── 22 edge traces (risk-coloured, width ∝ capacity)
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
```

---

## 10. Limitations & Extensions

### 10.1 Current Limitations

| Limitation | Description |
|------------|-------------|
| Capacity not enforced | Edge capacity is surfaced as a metric but not a hard constraint. Flow can exceed `k(e)`. |
| Static graph topology | Edges don't appear/disappear. In reality, canal closures and pipeline shutdowns remove edges entirely. |
| Simulated risk only | Risk is generated by OU process, not real signals. Production system would ingest news/AIS/market data. |
| Single-commodity flow | Model routes one flow at a time. Real logistics involves simultaneous multi-commodity flows. |
| Q-table scalability | Tabular Q-learning doesn't scale to large graphs. DQN (neural network approximator) needed for real deployments. |
| Undirected bypass routes | Some real bypass pipelines are bidirectional; current model only has directed edges. |

### 10.2 Direct Extensions

**Capacity-constrained routing (Min-Cost Max-Flow):**

Replace Dijkstra with successive shortest path algorithm operating on residual capacity:
```
min Σ c(e)·f(e)    subject to:
  Σ f(e) ≤ k(e)   for all e
  flow conservation at all non-source/sink nodes
```

**Multi-objective Pareto optimisation:**

Instead of scalarising cost/time/risk into a single weight, compute the full Pareto frontier across two or three objectives using NSGA-II or epsilon-constraint method.

**Deep Q-Network (DQN):**

Replace the tabular Q-function with a neural network:
```
Input:  graph embedding (node features + edge risk vector)
Output: Q-value for each possible next node
```
GNN-based state encoders (Graph Attention Networks) are a natural fit.

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

*End of Technical Document*
