# I Built a Simulation That Shows How the World's Oil Supply Could Collapse — Here's What the Math Says

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

## How Risk Evolves: The Rubber Band Model

Risk isn't static. A strait that's perfectly safe today can be under missile fire tomorrow. We model this with the **Ornstein-Uhlenbeck process** — a mean-reverting stochastic differential equation that behaves exactly like a rubber band:

```
dR(t) = θ(μ - R(t))dt + σdW(t)

         ↑─────────────────┘   └──────────┐
    "pull toward baseline"    "random shock"
```

- **θ = 0.3** — how hard the rubber band pulls back (mean reversion speed)
- **μ** — where the band is anchored (baseline risk, calibrated to Lloyd's war-risk premiums)
- **σ** — how strong the random gusts are (volatility; you control this with a slider)
- **dW(t)** — a Wiener process. Pure chaos. N(0,1) drawn fresh every tick.

Why not a random walk? Because a random walk **drifts**. Risk would eventually hit 0 or 1 and stay there forever. Real geopolitical risk mean-reverts — crises happen, escalate, and (usually) de-escalate back toward some grim baseline.

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

---

## The Routing Algorithm: Risk-Aware Dijkstra

Standard Dijkstra is a GPS. It finds the minimum-weight path in O((V+E) log V) using a priority queue — always expanding the cheapest reachable node next.

We keep the algorithm identical. We only change the **weight function**:

```python
def edge_weight(G, u, v, alpha=0.5, lam=10.0):
    e = G[u][v]
    return e["cost"] + alpha * e["time"] + lam * e["risk"]
```

The result: risk becomes **another form of cost**. A risky edge is expensive. At high λ, the Hormuz route becomes so "expensive" (in risk-adjusted terms) that the algorithm prefers the longer, safer bypass.

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

---

## The Graph Has a Structural Flaw: Betweenness Centrality

In network theory, **betweenness centrality** measures how often a node appears on shortest paths between all other node pairs:

```
C_B(v) = Σ σ_st(v) / σ_st    for all s≠v≠t
```

Hormuz has the **highest betweenness centrality in the network**. Remove it, and ~80% of shortest producer-to-consumer paths break. No other node comes close.

This means even a dramatically risk-averse router gets forced through Hormuz unless bypass paths are **complete**. A subtle bug: if the bypass graph isn't fully connected, the algorithm will route through Hormuz even at λ=∞ simply because no alternative complete path exists.

We fixed this by adding three missing edges:

```
Bab-el-Mandeb ──────────────────► Indian Ocean Hub  (southbound Red Sea exit)
Cape of Good Hope ──────────────► India              (round-Cape long bypass)  
Cape of Good Hope ──────────────► Strait of Malacca  (round-Cape to East Asia)
```

Now Saudi Arabia can reach China without Hormuz:

```
Saudi Arabia → Yanbu → Red Sea → Bab-el-Mandeb → Indian Ocean Hub → Malacca → China
```

At base risk: Hormuz still wins (cheaper). After crisis: bypass wins by ~7 weight units. The switchover is automatic — no manual override needed.

---

## When Optimisation Isn't Enough: Q-Learning

Dijkstra answers: *"What's the best route right now?"*

It doesn't remember what worked last week. It doesn't learn. It recomputes from scratch every time the graph changes. In a world where risk evolves continuously, that's reactive rather than adaptive.

**Q-Learning** solves a different problem: learn a *policy* — a mapping from situations to actions — that generalises across graph states.

The framework: a Markov Decision Process.

```
State  s:  (current node, Hormuz risk level)
Action a:  choose next node
Reward R:  -(cost + 10·risk + 2·time)  +  50 if you reach the target

Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') − Q(s,a) ]
                        └────────────────────────────┘
                              Bellman target: what Q should be
                        └──────────────────────────────────────┘
                                    TD error: the surprise
```

After training, the agent has a **policy table**: for every (node, risk_level) pair, it knows the best next move — *without running Dijkstra*. That's a lookup, not a search.

**Exploration vs exploitation:**

```
ε = 1.0 (start)  ──────────decay 0.994/episode──────────►  ε = 0.05 (converged)
[100% random]                                              [95% greedy]
     ↑                                                           ↑
 "Try everything"                                         "Exploit learned policy"
```

---

## The Key Insight The Math Reveals

Here's the counterintuitive conclusion that took a simulation to make obvious:

> **The best decision is often not finding a better route — it's committing less to any single route.**

When you run 500 Monte Carlo disruptions of varying severity across the network, the system always adapts. But it adapts *at a cost premium*. The more tightly optimised for the normal case, the more you pay in the tail.

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

Five tabs in v1:
1. **Network Map** — see the 24-edge graph on a world map, risk-coloured, with the optimal path in gold
2. **Route Finder** — sweep λ and watch the Hormuz→bypass switchover happen live
3. **Risk Simulator** — step through OU risk evolution, watch paths adapt
4. **Stress Test** — 500 Monte Carlo disruption scenarios; how often does the network reroute?
5. **RL Agent** — train Q-learning; compare learned policy vs Dijkstra under normal and crisis conditions

Six tabs in v2 (everything above plus):
6. **Economic Cascade** — Hormuz closure → oil price → CPI → food prices → GDP, across 5 regions

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

*v1 stack: NetworkX · Plotly · Streamlit · Q-Learning*  
*v2 stack: above + PyTorch · LSTM · DQN · Economic Cascade*  
*All numeric parameters sourced from EIA, IEA, Lloyd's, S&P Global, CEIC — see `DATA_SOURCES.md`*
