# We Built a Simulation That Shows How the World's Oil Supply Could Collapse — Here's What the Math Says

*Graph theory + geopolitics + a suspicious amount of Python. Buckle up.*

---

## The Bottleneck That Runs the World

Here's a fun fact that should terrify anyone who's ever filled a tank:

**One in every five barrels of oil on Earth passes through a 33km-wide gap between Oman and Iran.**

That's the Strait of Hormuz. And from a systems design perspective — the kind where you're supposed to avoid single points of failure — this is roughly equivalent to routing 20% of global internet traffic through a single server in a geopolitically unstable neighbourhood.

```
                    ╔═══════════════════════════════════════╗
                    ║   GLOBAL OIL FLOW (daily, MBD)        ║
                    ╠═══════════════════════════════════════╣
                    ║  🛢️ Hormuz       ████████████████  20  ║
                    ║  🚢 Malacca      █████████████     16  ║
                    ║  ⚓ Suez          ██████             7  ║
                    ║  🌊 Bab-el-Mandeb ██████             8  ║
                    ║  🔧 Yanbu bypass  █████              7  ║
                    ╚═══════════════════════════════════════╝
```

The alternatives exist. They're just expensive, slow, and massively under-capacity. The world didn't build redundancy because redundancy costs money. And nothing bad ever happens, right?

Right.

---

## Stripping Away the Politics (and Adding Math)

Let's describe this system the way an engineer would — without the geopolitics, the cable news, or the hot takes.

Oil flows from **sources** to **destinations** through a **network of routes**. Each route has cost, time, capacity, and risk.

That's not international relations. That's a **graph**.

```
G = (V, E)

V (nodes):    Saudi Arabia, UAE, Iraq, Kuwait, Qatar      ← producers
              Hormuz, Suez Canal, Bab-el-Mandeb,          ← chokepoints
              Strait of Malacca, Cape of Good Hope
              Yanbu, Fujairah, Indian Ocean Hub, Red Sea  ← bypass hubs
              India, China, Japan, Europe, USA            ← consumers

E (edges):    24 directed connections, each labelled:
              cost | transit_days | capacity_mbd | risk(t)
```

The conventional logistics objective is:

```
min Σ cost(e)     ← "find the cheapest route"
```

The problem: this is **blind to tail risk**. It optimises perfectly for the 99% of time nothing goes wrong, and catastrophically for the 1% when it does.

The risk-aware objective:

```
min Σ [ cost(e) + α·time(e) + λ·risk(e,t) ]
```

`λ` is your **risk aversion parameter**. The higher it is, the more expensive you're willing to make your route in order to avoid a dangerous edge. Crank it high enough and the algorithm abandons Hormuz entirely.

That cost premium at switchover = **the market price of resilience**. Nobody pays it voluntarily. Everybody wishes they had when things go sideways.

---

## The Graph: What We Actually Built

The graph has **24 directed edges** across five categories of nodes.

**Node types and their roles:**

| Type | Examples | Role |
|------|----------|------|
| Producer | Saudi Arabia, UAE, Iraq, Kuwait, Qatar | Oil-exporting nations |
| Chokepoint | Hormuz, Suez Canal, Strait of Malacca, Bab-el-Mandeb, Cape of Good Hope | Strategically critical straits/canals |
| Bypass/Transit Hub | Yanbu, Fujairah, Indian Ocean Hub, Red Sea | Alternative routing terminals |
| Consumer | India, China, Japan, Europe, USA | Importing nations/regions |

**The Hormuz funnel:** All five Gulf producers have a direct edge into the `Hormuz` node. Hormuz has a single outbound edge at 20 MBD — the bottleneck edge. All six of these edges are flagged `hormuz_dependent=True`.

**Bypass routes:**
- `Saudi Arabia → Yanbu` (Saudi East-West/Petroline, **7.0 MBD** post-March 2026 upgrade) → Red Sea → Bab-el-Mandeb → Indian Ocean Hub
- `UAE → Fujairah` (ADCO/Habshan pipeline, 1.5 MBD) → Indian Ocean Hub directly

**A critical connectivity bug we found and fixed:** The bypass graph wasn't fully connected. A risk-averse router forced to avoid Hormuz had no complete path to Asian consumers — not because the routes don't exist geographically, but because three edges were missing from the graph.

We added them:

```
Bab-el-Mandeb ──────────────────► Indian Ocean Hub  (southbound Red Sea exit)
Cape of Good Hope ──────────────► India              (round-Cape long bypass)
Cape of Good Hope ──────────────► Strait of Malacca  (round-Cape to East Asia)
```

Now Saudi Arabia can reach China without Hormuz:

```
Saudi Arabia → Yanbu → Red Sea → Bab-el-Mandeb → Indian Ocean Hub → Malacca → China
```

At base risk: Hormuz still wins (cheaper). During a crisis: bypass wins by ~7 risk-adjusted weight units. The switchover is automatic — no manual override needed.

**Complete edge breakdown:**

| Category | Edges | Count |
|----------|-------|-------|
| Producers → Hormuz | SA, UAE, Iraq, Kuwait, Qatar | 5 |
| Bypass pipelines | SA→Yanbu, UAE→Fujairah | 2 |
| Hormuz outbound | Hormuz→IOH, Fujairah→IOH | 2 |
| Red Sea corridor | Yanbu→RS, RS→Bab, Bab→Suez, Bab→IOH | 4 |
| IOH → consumers/hubs | IOH→India, IOH→Malacca, IOH→Cape | 3 |
| Malacca → East Asia | Malacca→China, Malacca→Japan | 2 |
| Suez → West | Suez→Europe, Suez→USA | 2 |
| Cape → consumers | Cape→Europe, Cape→USA, Cape→India, Cape→Malacca | 4 |
| **Total** | | **24** |

---

## The Graph Has a Structural Flaw: Betweenness Centrality

In network theory, **betweenness centrality** measures how often a node appears on shortest paths between all other node pairs:

```
C_B(v) = Σ σ_st(v) / σ_st    for all s≠v≠t
```

where σ_st = number of shortest paths from s to t, and σ_st(v) = those passing through v.

Hormuz has the **highest betweenness centrality in the network**. Remove it, and ~80% of shortest producer-to-consumer paths break. No other node comes close.

A node with C_B → 1.0 is on almost every shortest path in the network. No engineering fix — adding capacity, raising pipeline throughput — reduces this without building structurally parallel paths that bypass the node entirely. This means even a dramatically risk-averse router gets forced through Hormuz unless bypass paths are **complete**. The fix isn't to increase λ — it's to close the graph topology.

---

## How Risk Evolves: The Rubber Band Model

Risk isn't static. A strait that's perfectly safe today can be under missile fire tomorrow. We model this with the **Ornstein-Uhlenbeck process** — a mean-reverting stochastic differential equation that behaves exactly like a rubber band:

```
dR(t) = θ(μ - R(t))dt + σdW(t)

         ↑─────────────────┘   └──────────┐
    "pull toward baseline"    "random shock"
```

| Parameter | Symbol | Value | Meaning |
|-----------|--------|-------|---------|
| Mean reversion speed | θ | 0.3 | How fast risk returns to baseline |
| Long-run mean | μ | `base_risk` per edge | Equilibrium risk level |
| Volatility | σ | `volatility × 0.12` | Scales random shock magnitude |
| Wiener increment | dW | N(0,1) | Random shock draw |

**Why not a random walk?** Because a random walk **drifts**. Risk would eventually hit 0 or 1 and stay there forever. Real geopolitical risk mean-reverts — crises happen, escalate, and (usually) de-escalate back toward some grim baseline.

The 2023 Houthi crisis sent Bab-el-Mandeb war-risk premiums to **2,700% of baseline**. They partially unwound within 12 months. Rubber band.

```
Risk level
  1.0 │         🔥 CRISIS
  0.9 │        ╱╲
  0.8 │       ╱  ╲
  0.7 │      ╱    ╲
  0.6 │     ╱      ╲
  0.5 │    ╱        ╲_____
  0.4 │   ╱               ╲___
  0.3 │──╱────────────────────────  ← baseline (μ)
  0.2 │
      └─────────────────────────── time
            ↑ shock       ↑ mean reversion
```

**Discrete-time implementation (one tick):**

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

**Baseline risk values (calibrated to Lloyd's / S&P war-risk insurance premiums):**

| Route Segment | Base Risk | Insurance Premium Band |
|---------------|-----------|------------------------|
| Producers → Hormuz | 0.28 | ~0.25–0.50% hull |
| Hormuz → Indian Ocean | 0.28 | ~0.25% hull |
| Saudi East-West Pipeline (Yanbu) | 0.08 | ~0.05–0.10% hull |
| UAE Fujairah pipeline | 0.09 | ~0.05–0.10% hull |
| Red Sea / Bab-el-Mandeb | 0.35 | ~0.70% hull (2024) |
| Suez Canal | 0.18 | ~0.10–0.20% hull |
| Strait of Malacca | 0.05 | ~0.05% hull |
| Indian Ocean (open water) | 0.07 | ~0.05% hull |
| Cape of Good Hope | 0.02 | ~0.01% hull |

**Crisis shock:** A sudden, large risk spike applied to all Hormuz-dependent edges — modelling a geopolitical event that instantly reprices war-risk premiums:

```python
def apply_hormuz_crisis(G, severity=0.88):
    for u, v in G.edges():
        if G[u][v].get("hormuz_dependent", False):
            G[u][v]["risk"] = float(np.clip(
                severity + random.uniform(-0.05, 0.05), 0, 1
            ))
```

---

## The Routing Algorithm: Risk-Aware Dijkstra

Standard Dijkstra is a GPS. It finds the minimum-weight path in O((V+E) log V) using a priority queue — always expanding the cheapest reachable node next.

We keep the algorithm identical. We only change the **weight function**:

```python
def edge_weight(G, u, v, alpha=0.5, lam=10.0):
    e = G[u][v]
    return e["cost"] + alpha * e["time"] + lam * e["risk"]

def risk_dijkstra(G, source, target, alpha=0.5, lam=10.0):
    pq = [(0.0, source, [source])]
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

The result: risk becomes **another form of cost**. A risky edge is expensive. At high λ, the Hormuz route becomes so "expensive" (in risk-adjusted terms) that the algorithm prefers the longer, safer bypass.

**The λ switchover mechanism:**

- At low λ: Hormuz path cost < bypass path cost → Hormuz wins
- At λ = λ*: both paths have equal weighted cost → indifferent
- At λ > λ*: bypass weighted cost < Hormuz weighted cost → bypass wins

λ* is not set manually — it emerges from the current risk values on every edge. The Route Finder tab sweeps λ ∈ [0, 50] in 40 steps to reveal this Pareto frontier explicitly.

**The λ Pareto frontier:**

```
Route Cost                    Risk
    ▲                           ▲
    │  ●──────────────────────● │   λ = 0  (cost-only)
    │          ●──────────●      │
    │                  ●         │   λ = 10 (balanced)
    │                       ●    │   λ = 30 (risk-averse)
    └──────────────────────────► └─►
              λ →                     λ →
                  ↑
              SWITCHOVER POINT
         (Hormuz → Bypass flip)
```

The jump in cost at the switchover = exactly what you'd need to pay to make safe routing economically rational under normal conditions. Most shippers don't pay it. Most of the time, they're right.

**Path statistics computed for every route:**

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

## When Optimisation Isn't Enough: Q-Learning

Dijkstra answers: *"What's the best route right now?"*

It doesn't remember what worked last week. It doesn't learn. It recomputes from scratch every time the graph changes. In a world where risk evolves continuously, that's reactive rather than adaptive.

**Q-Learning** solves a different problem: learn a *policy* — a mapping from situations to actions — that generalises across graph states.

The framework: a Markov Decision Process.

| Component | Definition |
|-----------|-----------|
| **State** s | (current_node, discretised_risk_vector) |
| **Action** a | Choose next node to move to (from neighbours) |
| **Reward** R | -(cost + 10·risk + 2·time) + 50·[reached target] |
| **Discount** γ | 0.9 |

```
Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
                        └────────────────────────────┘
                              Bellman target: what Q should be
                        └──────────────────────────────────────┘
                                    TD error: the surprise
```

After training, the agent has a **policy table**: for every (node, risk_level) pair, it knows the best next move — *without running Dijkstra*. That's a lookup, not a search.

**Q-learning update implementation:**

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

**Exploration vs exploitation:**

```
ε = 1.0 (start)  ──────────decay 0.994/episode──────────►  ε = 0.05 (converged)
[100% random]                                              [95% greedy]
     ↑                                                           ↑
 "Try everything"                                         "Exploit learned policy"
```

**Expected training behaviour:**
- Early episodes: high ε causes random walks; most episodes fail to reach the target
- Mid training: agent learns which nodes are dead ends; starts finding target-reaching paths
- Late episodes: ε has decayed; agent consistently follows learned high-value paths; reward curve plateaus

**The tabular Q-learning ceiling:** With 24 edges and 5 risk levels, the theoretical state space is 19 × 5²⁴ — in practice, only a tiny fraction of states is visited during training. Any unseen state gets Q-value = 0 by default, meaning the agent falls back to random selection exactly when reliable decisions matter most. This is the fundamental limitation DQN (v2) addresses.

---

## The Stress Test: 500 Monte Carlo Scenarios

Here's the counterintuitive conclusion that took a simulation to make obvious:

> **The best decision is often not finding a better route — it's committing less to any single route.**

When you run 500 Monte Carlo disruptions of varying severity across the network, the system always adapts. But it adapts *at a cost premium*. The more tightly optimised for the normal case, the more you pay in the tail.

**Monte Carlo implementation:**

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

```
Disruption severity vs rerouting cost premium:

Cost
premium   ╭──────────────────────────────────────────╮
  +50%    │                                    ●●●●● │
  +40%    │                              ●●●●●       │
  +30%    │                        ●●●●●             │
  +20%    │                  ●●●●●                   │
  +10%    │           ●●●●●                          │
   +0%    │●●●●●                                     │
          ╰──────────────────────────────────────────╯
          0.3    0.5    0.6    0.75   0.85   0.95
                       Disruption severity
```

At severity 0.75+, almost 100% of simulated routes abandon Hormuz entirely. But the bypass costs 30–50% more. **That premium is the price of resilience the market refuses to pay upfront.**

**Crisis severity benchmarks:**

| Scenario | Severity | Real Equivalent |
|----------|----------|-----------------|
| Moderate tension | 0.45 | Hormuz 2019 tanker attacks |
| Serious escalation | 0.70 | Estimated major-incident level |
| Near-closure | 0.90 | Lloyd's March 2026 Gulf escalation |
| Active conflict | 1.00 | 1980s Tanker War peak |

---

## Application Architecture

### Single-File Design

All logic — graph construction, risk engine, routing, RL agent, visualisation, and UI — lives in `app.py`. This is intentional for a POC: no import machinery, no packaging, no relative path issues.

### Streamlit Session State

Streamlit reruns the entire script on every user interaction. Mutable state that must persist across reruns is stored in `st.session_state`:

| Key | Type | Description |
|-----|------|-------------|
| `G` | `nx.DiGraph` | Live graph with current risk values |
| `t` | `int` | Current simulation tick |
| `risk_hist` | `list[dict]` | History of risk/cost/path per tick |
| `rl_agent` | `QLearningAgent` | Trained agent instance |
| `rl_rewards` | `list[float]` | Episode reward history |
| `crisis` | `bool` | Whether crisis is currently active |

### The Five Tabs

| Tab | What It Does |
|-----|-------------|
| 🗺️ Network Map | 24-edge graph on a world map; risk-coloured edges (green→red), gold optimal path |
| 🛣️ Route Finder | Sweep λ and watch the Hormuz→bypass switchover happen live |
| 📡 Risk Simulator | Step through OU risk evolution; watch paths adapt in real time |
| 🔥 Stress Test | 500 Monte Carlo disruption scenarios; cost distribution histogram; rerouting gauge |
| 🤖 RL Agent | Train Q-learning; compare learned policy vs Dijkstra under normal and crisis conditions |

**Risk-to-colour mapping:**

```python
def _risk_rgb(risk):
    r = int(255 * risk)        # 0 → 0, 1 → 255
    g = int(255 * (1 - risk))  # 0 → 255, 1 → 0
    return f"rgb({r},{g},50)"  # green → yellow → red gradient
```

---

## v2: Teaching the System to *Predict* Risk

v1 had two fundamental limitations.

**Limitation 1 — Simulated risk, not predicted risk.** The OU process generates plausible-looking risk trajectories but does not learn from signal data. It cannot distinguish between a random volatility spike and a genuine escalation event.

**Limitation 2 — Tabular Q-learning does not generalise.** The Q-table stores Q-values only for (state, action) pairs explicitly visited during training. In a 24-edge network with 5 risk buckets per edge, the theoretical state space is 19 × 5²⁴ ≈ 60 trillion states. The agent sees a negligible fraction of this during training.

| Problem | v1 Approach | v2 Approach |
|---------|------------|------------|
| Risk prediction | OU formula (no learning) | LSTM trained on structured synthetic data |
| Q-function | Dict lookup (no generalisation) | Neural network (continuous state, interpolates) |
| Training stability | None | Experience replay + target network |
| State space | Discrete, exponential | Continuous, 43-dimensional |
| Unseen states | Q = 0 (random fallback) | Forward pass through network (learned estimate) |

### The LSTM Risk Predictor

The LSTM takes a 10-step window of four signals per edge and predicts next-step risk across all 24 edges simultaneously:

```
Signals fed to LSTM (per edge, per timestep):
┌─────────────────────────────────────────────────────┐
│  risk             — current edge risk level         │
│  oil_vol          — oil price volatility (concurrent)│
│  insurance_premium— war-risk premium (7-step lag)   │
│  sentiment        — news sentiment (leading signal)  │
└─────────────────────────────────────────────────────┘
         └─── 10 steps of history ───► LSTM ──► r̂(t+1)
```

The lag/lead structure is deliberate. Insurance premiums lag risk by ~7 steps (markets are slow to reprice). Sentiment leads risk by ~3–5 steps (analysts see tension before tankers reroute). The LSTM must learn to use sentiment as an early warning signal and not be misled by the lagging insurance data.

**Architecture (~251K parameters):**

```
Input:  (B, 10, 24×4=96)
        ↓
LSTM Layer 1:  hidden=128, dropout=0.2
LSTM Layer 2:  hidden=128
        ↓ take last hidden state
Linear(128→64) → ReLU → Dropout(0.1)
        ↓
Linear(64→24) → Sigmoid
        ↓
Output: (B, 24)    predicted r(e, t+1) ∈ (0, 1) for all edges
```

**Training convergence:**

| Metric | Initial | After 20ep | After 120ep |
|--------|---------|------------|-------------|
| Train MSE | ~0.040 | ~0.012 | ~0.003 |
| Val MSE   | ~0.045 | ~0.015 | ~0.005 |

When the LSTM sees sentiment drop *before* insurance reprices, it predicts rising Hormuz risk **7 steps ahead**. The router then pre-commits to bypass routes before the market has even priced in the tension.

### v2: Deep Q-Network — When the Table Gets Too Big

DQN replaces the lookup table with a neural network that maps a *continuous* state vector to Q-values for all actions — generalising to states it has never seen before.

**State representation (43 dimensions):**

```
[ one-hot node encoding (19) | LSTM-predicted edge risks (24) ]
```

**Network architecture (~46K parameters):**

```
Input (43,)
    │
Linear(43 → 256) + LayerNorm + ReLU + Dropout(0.1)
    │
Linear(256 → 128) + ReLU
    │
Linear(128 → 19)
    │
Output (19,) — Q-value per node (masked to valid neighbours before argmax)
```

**Two stabilisation tricks that make this trainable:**

**Experience replay** — store 10,000 past transitions, sample 64 randomly per gradient step. Breaks temporal correlation between consecutive training samples. Prevents the network from forgetting rare but important events.

**Target network** — a frozen copy of the policy network, synced every 100 gradient steps. Without it, Q-targets chase a moving bullseye. With it, targets are stationary over 100 steps, providing a stable learning signal.

```python
# Bellman update
q_pred = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
with torch.no_grad():
    q_next = target_net(next_states).max(1)[0]
    q_tgt  = rewards + GAMMA * q_next * (1 - dones)
loss = smooth_l1(q_pred, q_tgt)

# Target sync every 100 steps
if steps % 100 == 0:
    target_net.load_state_dict(policy_net.state_dict())
```

**Huber loss over MSE:** Early in training, TD errors are large and noisy. MSE amplifies them. Huber loss behaves like MSE for small errors and like MAE for large ones — bounding the gradient magnitude and keeping training stable.

**Reward function:**

```
R(edge e = (u,v)) = -(cost(e) + 12·risk(e) + 2·time(e))
R(reaching target) += +100
```

The +100 target bonus is critical. Without it, the agent learns that staying put maximises reward — a pathological fixed point.

### v2: The Economic Cascade

Knowing the routing cost of a Hormuz closure is only half the picture. The other half: what happens to the global economy?

```
Hormuz closure
    ↓  (immediate)      Oil price spike
    ↓  (days 0–7)       Freight premium — rerouting adds 10–34 days
    ↓  (days 7–30)      Energy & manufacturing costs rise
    ↓  (days 30–45)     Headline CPI — regional pass-through (IMF calibrated)
    ↓  (days 45–60)     Food prices — fertilizer, freight, panic buying
    ↓  (days 60–90)     Central bank rate hikes
    ↓  (days 90–180)    GDP contraction
```

The damage is not uniform. East Asia (85% oil import dependent) gets hit 4× harder than the USA (15% dependent). 500 Monte Carlo scenarios quantify the tail risk — the 95th percentile outcomes planners should design for.

---

## Limitations and Extensions

### Current Limitations

| Limitation | Description | v2 Status |
|------------|-------------|-----------|
| Capacity not enforced | Edge capacity is surfaced as a metric but not a hard constraint | Still open |
| Static graph topology | Edges don't appear/disappear; canal closures not modelled | Still open |
| Simulated risk only | Risk generated by OU, not real signals | **LSTM in v2** trains on correlated synthetic signals |
| Single-commodity flow | Model routes one flow at a time | Still open |
| Q-table scalability | Tabular Q-learning doesn't scale — 19 × 5²⁴ state space | **DQN in v2** generalises via neural interpolation |
| No economic impact | Routing cost only, not macro consequences | **Economic Cascade in v2** |

### Planned Extensions

**Capacity-constrained routing (Min-Cost Max-Flow):** Replace Dijkstra with successive shortest path on residual capacity — properly enforcing the 20 MBD Hormuz ceiling as a hard constraint.

**Live data integration:**
```
News API → sentiment classifier → r_news(e, t)
Shipping API (AIS) → anomaly score → r_ais(e, t)
Options market IV → volatility surface → r_mkt(e, t)
r(e, t) = Bayesian combination of [r_news, r_ais, r_mkt]
```

**Bayesian risk updating:** Update prior per edge using conjugate Beta-Binomial model as signals arrive — giving a calibrated posterior probability rather than a point estimate.

**Survival analysis:** Model "time until disruption" as a survival function with hazard rate λ(t) driven by risk signals. This gives a disruption probability over a planning horizon rather than an instantaneous risk score.

---

## The Honest Conclusion

The Strait of Hormuz will remain a chokepoint. That's a physical fact — 33km of navigable water between two coastlines.

But whether it's a *vulnerability* is a design choice. One the global energy system keeps making, quietly, every day.

The engineering lesson generalises to every system you'll ever build:

```
╔══════════════════════════════════════════════════════╗
║                                                      ║
║   Any system optimised entirely for the normal       ║
║   case is fragile by construction.                   ║
║                                                      ║
║   Redundancy > Efficiency.                           ║
║   Optionality > Optimisation.                        ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
```

The math doesn't lie. The market just refuses to listen — until it has to.

---

## Run It Yourself

```bash
# v1: Graph theory + Q-Learning (no ML dependencies)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

# v2: + LSTM risk prediction + DQN + Economic Cascade (PyTorch)
pip install -r v2/requirements_v2.txt
streamlit run v2/app_v2.py
```

---

*v1 stack: NetworkX · Plotly · Streamlit · Q-Learning*
*v2 stack: above + PyTorch · LSTM · DQN · Economic Cascade*
*All numeric parameters sourced from EIA, IEA, Lloyd's, S&P Global, CEIC — see `DATA_SOURCES.md`*
