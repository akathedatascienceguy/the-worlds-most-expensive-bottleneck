# Graph Theory Meets Geopolitics: Solving the Hormuz Problem

*How to model the world's most critical shipping lane as a dynamic graph — and what the math tells us about resilience, risk, and the cost of optionality.*

---

## The Bottleneck No Engineer Would Design

Roughly one in every five barrels of oil passes through a 33-kilometre-wide strait between Oman and Iran. The Strait of Hormuz. From a systems design perspective, this is roughly equivalent to routing 20% of global internet traffic through a single server — and hoping it doesn't crash.

It's not that alternatives don't exist. Pipelines bypass Hormuz. Tankers can round the Cape of Good Hope. The Saudi East-West pipeline terminates at Yanbu on the Red Sea. But these alternatives are underutilised, under-capacity, and significantly more expensive.

The world didn't optimise for resilience. It optimised for cost. And in a world where nothing goes wrong, that's a perfectly rational choice.

The problem is that things go wrong.

---

## Stripping Away the Politics

Let's describe the system the way an engineer would:

- Oil flows from **sources** (Gulf producers) to **destinations** (refineries and consumers)
- Through a **network of routes** (shipping lanes, pipelines)
- Constrained by **capacity**, **transit time**, and **cost**

That's not geopolitics. That's a **graph**.

Formally, define:

```
G = (V, E)
```

Where:
- **V** = ports, refineries, chokepoints, transit hubs
- **E** = shipping lanes and pipelines, each with attributes:
  - `c(e)` — cost
  - `t(e)` — transit time in days
  - `k(e)` — capacity in million barrels per day
  - `r(e, t)` — **time-dependent risk** ← the missing variable in most models

The conventional approach optimises for:

```
min Σ c(e)     for e in path
```

The risk-aware approach optimises for:

```
min Σ [c(e) + α·t(e) + λ·r(e, t)]     for e in path
```

Where λ is your **risk aversion parameter** — how much you're willing to pay in extra cost to route around danger.

This is no longer the shortest path problem. It's risk-aware routing on a dynamic graph.

---

## The Strait of Hormuz in Graph Terms

In network theory, Hormuz has several notable properties:

**High betweenness centrality.** Remove Hormuz from the graph and the number of shortest paths between Gulf producers and East Asian consumers drops dramatically. No other single node has this property.

**High flow dependency.** Roughly 20 million barrels per day transit through it (EIA 2024). The next largest single-point flow dependency — the Strait of Malacca — handles 16.6 MBD, but receives oil *from* Hormuz, so it is downstream of the bottleneck, not parallel to it.

**Catastrophic failure impact.** A single node failure doesn't just increase route cost. It can make the destination *unreachable* within time constraints — unless bypass alternatives exist with sufficient capacity. A critical design question: is the bypass graph *connected* for every producer-consumer pair?

### The Connectivity Bug

A subtle problem: an incomplete bypass graph means that even with high risk aversion (λ → ∞), the router is forced through Hormuz because no alternative complete path exists. In the original model, Saudi Arabia had *no Hormuz-free path to India, China, or Japan* — the Yanbu bypass led only to Suez Canal, which connects to Europe and USA but not Asia.

The fix required adding three missing edges:

```
Bab-el-Mandeb → Indian Ocean Hub   (ships exit Red Sea southward into the Indian Ocean)
Cape of Good Hope → India           (round-Cape long bypass, ~16 days from Cape)
Cape of Good Hope → Strait of Malacca  (Cape to East Asia, ~18 days from Cape)
```

With these edges, the bypass chain closes completely:

```
Saudi Arabia → Yanbu → Red Sea → Bab-el-Mandeb → Indian Ocean Hub → India
                                                                   → Malacca → China / Japan
```

At default risk (λ=10), Hormuz still wins — it is cheaper and faster. After a crisis (Hormuz risk → 0.90), the Yanbu bypass becomes cheaper by ~7 weight units and the router switches automatically. **This is the intended behaviour**: Hormuz dominates under normal conditions, bypass routes activate under stress.

---

## Where Does Risk Come From?

This is where data science stops being decorative and starts doing actual work.

Risk `r(e, t)` is not directly observed. It's inferred from signals.

### Textual Signals

News headlines, diplomatic statements, shipping advisories, and social media encode conflict escalation before it shows up in market prices. A transformer-based NLP model can estimate a sentiment score — from -1 (peaceful) to +1 (conflictual) — that modulates the prior risk level.

### Time-Series Market Signals

Oil price volatility, tanker day rates, and war-risk insurance premiums are real-time market aggregations of dispersed information. A sharp spike in insurance premiums for Hormuz-transiting vessels is a Bayesian update: the market knows something.

The baseline risk values in this model are calibrated directly to **Lloyd's / S&P war-risk insurance premium bands**:

| Route | Base Risk | Insur. Premium Band |
|-------|-----------|---------------------|
| Producers → Hormuz | 0.28 | ~0.25–0.50% hull (S&P Global 2024) |
| Bab-el-Mandeb (Houthi-era) | 0.35 | ~0.70% hull (AGBI/Maplecroft 2024) |
| Strait of Malacca | 0.05 | ~0.05% hull (Lloyd's JWC delisted May 2024) |
| Cape of Good Hope | 0.02 | ~0.01% hull (open ocean baseline) |

### Spatial Signals

AIS tracking data gives vessel positions in near-real-time. Unusual clustering, vessels deviating from standard lanes, or suspicious transponder gaps are anomaly signals. Satellite imagery adds another layer.

---

## The Graph Is Alive

In textbooks, graphs are static. In reality, the oil network is dynamic:

- Edges appear and disappear (canal closures, pipeline shutdowns)
- Weights fluctuate (insurance premiums change daily)
- Capacity constraints tighten (congestion cascades)

### v1: Ornstein-Uhlenbeck Risk Simulation

The OU process is a natural model for risk evolution — it's mean-reverting (risk returns to baseline absent new shocks) and stochastic (exogenous shocks move it away):

```
dR(t) = θ(μ - R(t))dt + σ dW(t)
```

Where:
- `θ` — mean reversion speed (0.3)
- `μ` — long-run baseline risk per edge
- `σ` — volatility (user-controlled, scales with geopolitical tension)
- `W(t)` — Wiener process (random shocks)

### v2: LSTM Risk Prediction

The OU process generates risk internally from a formula. It captures mean reversion but cannot learn from multi-signal patterns. v2 replaces it with a trained **2-layer LSTM**:

```
Input:  10-step window of (risk, oil_vol, insurance, sentiment) per edge
Output: predicted r̂(e, t+1) for all edges simultaneously
```

The LSTM is trained on 2,000 synthetic timesteps of OU-generated risk with three correlated signals:
- `oil_vol` — concurrent with risk (same geopolitical event drives both)
- `insurance` — 7-step lag (insurance repricing follows incidents)
- `sentiment` — leading indicator (tension visible in news before markets reprice)

The LSTM learns these lag structures from data. This gives it a genuine predictive advantage over the reactive OU model: when sentiment spikes, the LSTM predicts rising risk and the router pre-empts the spike before insurance catches up.

---

## Risk-Aware Dijkstra

Standard Dijkstra finds the minimum-cost path. Risk-aware Dijkstra redefines the edge weight:

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
            return cost, path
        for nb in G.neighbors(node):
            if nb not in visited:
                w = edge_weight(G, node, nb, alpha, lam)
                heapq.heappush(pq, (cost + w, nb, path + [nb]))
    return float("inf"), []
```

The crucial insight: as λ increases, the algorithm switches from Hormuz-dependent routing to bypass routes. This switchover happens at a specific λ threshold, and the cost jump at that threshold is the **price of resilience**.

---

## When Optimisation Isn't Enough

Dijkstra solves for the current state of the graph. It gives you the best path *right now*. But:

- The environment changes continuously
- Each routing decision affects congestion on chosen edges
- Uncertainty compounds over long planning horizons

This is where reinforcement learning becomes necessary.

### v1: Tabular Q-Learning

Model the routing problem as a Markov Decision Process:

```
State  s: current node + discretised risk levels across all edges
Action a: choose next node to route through
Reward R: -(cost + 10·risk + 2·time) + 50·[reached target]
```

The Q-learning update rule:

```
Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]
```

The agent learns a policy — a mapping from (node, risk_conditions) → action — that generalises across graph states without re-solving from scratch. **But**: tabular Q-learning stores one entry per (state, action) pair. With 24 edges and risk discretised to 5 levels, the state space is 19 × 5²⁴ — almost entirely unvisited during training.

### v2: Deep Q-Network (DQN)

v2 replaces the lookup table with a **3-layer neural network**:

```
Input:  one-hot(current node, 19 dims) + current risk vector (24 dims) = 43-dim state
Hidden: Linear(43→256) + LayerNorm + ReLU + Dropout(0.1)
        Linear(256→128) + ReLU
Output: Linear(128→19) = Q-value for each possible next node
```

Key stabilisation techniques:
- **Experience replay buffer** (10k capacity) — breaks temporal correlation in transitions
- **Frozen target network** — hard-synced every 100 gradient steps to stabilise Bellman targets
- **Huber loss** — bounded gradient for noisy TD targets
- **Action masking** — Q-values for non-neighbouring nodes are set to -∞ before argmax

This lets the DQN generalise to unseen (node, risk) combinations via interpolation — something tabular Q-learning cannot do.

---

## Beyond Routing: The Economic Cascade

Knowing the optimal route is only part of the picture. Decision-makers need to know: *if Hormuz closes, what happens to the global economy?*

The v2 Economic Cascade model answers this by chaining five transmission mechanisms:

```
Hormuz disruption
    → Supply cut (MBD) × duration multiplier
    → Oil price shock (spot + forward 90-day)
    → Freight premium (tanker rerouting cost)
    → Headline CPI (regional pass-through coefficients from IMF WP/17/53)
    → Food price (fertilizer cost + freight-to-food + panic premium)
    → GDP contraction (direct energy drag + monetary policy drag)
```

Regional pass-through coefficients reflect real import dependency and energy intensity:

| Region | Import Dep. | CPI Pass-Through | GDP Elasticity |
|--------|------------|------------------|----------------|
| East Asia (Japan/Korea/China) | 85% | 0.18 | −0.040 |
| India | 85% | 0.16 | −0.050 |
| Europe | 55% | 0.13 | −0.028 |
| USA | 15% | 0.08 | −0.015 |
| Developing Markets | 80% | 0.22 | −0.060 |

The model is calibrated against seven historical disruptions (1973 Arab Embargo through 2024 Houthi/Red Sea attacks). A Monte Carlo simulation runs 500 scenarios with random severity and duration, producing 95th percentile ("tail risk") estimates alongside median outcomes.

---

## The Counterintuitive Conclusion

Here's what the math reveals that intuition misses:

**The best decision is often not finding a better route — it's committing less to any single route.**

When you run the Monte Carlo stress test across 500 simulated Hormuz disruptions of varying severity, the system adapts — but always at a cost premium. The more you've optimised for the normal case, the more you pay in the tail case.

This is the fundamental tension between efficiency and resilience:

- **Efficiency** = commit to the cheapest path, minimise slack
- **Resilience** = maintain optionality across multiple paths, absorb shocks

The oil network has historically chosen efficiency. The cost of that choice is invisible until a crisis makes it visible — and by then, it's too late to build bypass capacity.

The engineering lesson generalises far beyond oil: **any system optimised entirely for the normal case is fragile by construction**.

---

## Running the POC

### v1 (no ML)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### v2 (LSTM + DQN + Economic Cascade, requires PyTorch)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r v2/requirements_v2.txt
streamlit run v2/app_v2.py
```

The interactive apps let you:

1. **Network Map** — visualise 24-edge oil network on a world map with risk-coloured edges
2. **Route Finder** — sweep λ from 0→50 and watch the path switch from Hormuz to bypass routes
3. **Risk Simulator** — step through risk evolution (OU in v1; LSTM-predicted in v2)
4. **Stress Test** — trigger a Hormuz crisis; run 500 Monte Carlo disruption scenarios
5. **RL Agent** — train the routing agent (Q-table in v1; DQN in v2)
6. **Economic Cascade** *(v2 only)* — simulate macro impacts; 6-month time series; regional breakdown; Monte Carlo tail risk

---

## The Real Question

The question isn't "what's the shortest path?"

The question is: **"What's the smartest path given that the map itself might change — and what happens to the world economy when it does?"**

And the answer, from both the math and the data, is: build redundancy before you need it, price risk honestly, and never mistake optimisation for stability.

The Strait of Hormuz will remain a chokepoint. But the degree to which it's a *vulnerability* is a design choice — one the global energy system continues to make, quietly, every day.

---

*v1: NetworkX · Plotly · Streamlit · Q-Learning*  
*v2: above + PyTorch · LSTM · DQN · Economic Cascade Model*
