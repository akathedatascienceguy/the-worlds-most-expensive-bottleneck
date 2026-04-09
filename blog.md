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

**High betweenness centrality.** Remove Hormuz from the graph and the number of shortest paths between Gulf producers and East Asian consumers drops by ~80%. No other single node has this property.

**High flow dependency.** Roughly 18 million barrels per day transit through it. The next largest single-point flow dependency — the Strait of Malacca — handles about 4 million.

**Catastrophic failure impact.** A single edge failure doesn't just increase route cost. It can make the destination *unreachable* within time constraints, because bypass alternatives have insufficient capacity and significantly longer transit times.

This is a textbook example of a **single point of failure in a critical infrastructure graph** — and it exists not because engineers forgot to think about redundancy, but because building and operating redundancy is expensive, and the expected value of a Hormuz disruption was historically low.

That expected value is no longer low.

---

## Where Does Risk Come From?

This is where data science stops being decorative and starts doing actual work.

Risk `r(e, t)` is not directly observed. It's inferred from signals.

### Textual Signals

News headlines, diplomatic statements, shipping advisories, and social media encode conflict escalation before it shows up in market prices. A transformer-based NLP model can estimate a sentiment score — from -1 (peaceful) to +1 (conflictual) — that modulates the prior risk level:

```python
def risk_from_sentiment(baseline_risk, sentiment_score):
    # sentiment in [-1, +1]
    delta = 0.3 * sentiment_score
    return max(0.0, min(1.0, baseline_risk + delta))
```

### Time-Series Market Signals

Oil price volatility, tanker day rates, and war-risk insurance premiums are real-time market aggregations of dispersed information. A sharp spike in insurance premiums for Hormuz-transiting vessels is a Bayesian update: the market knows something.

These signals can be modelled as input features to a classification model (safe / elevated / high risk) or as direct risk multipliers.

### Spatial Signals

AIS (Automatic Identification System) tracking data gives vessel positions in near-real-time. Unusual clustering, vessels deviating from standard lanes, or suspicious transponder gaps are anomaly signals. Satellite imagery adds another layer — visible ship density, military vessel positioning, drone activity.

### Turning Signals into Probabilities

All of this feeds a **probabilistic calibration problem**. The goal isn't accuracy in the conventional sense. It's calibration: when you say 70% risk, bad things should happen 70% of the time.

A well-calibrated 70% is more useful than an overconfident 95%.

Approaches include:

1. **Bayesian updating** — start with a prior (baseline risk), update as signals arrive
2. **Survival analysis** — model "time until disruption" as a survival function
3. **Deep learning classifiers** — predict risk tier given a feature vector of signals

---

## The Graph Is Alive

In textbooks, graphs are static. In reality, the oil network is dynamic:

- Edges appear and disappear (canal closures, pipeline shutdowns)
- Weights fluctuate (insurance premiums change daily)
- Capacity constraints tighten (congestion cascades)

The Ornstein-Uhlenbeck (OU) process is a natural model for risk evolution — it's mean-reverting (risk returns to baseline absent new shocks) and stochastic (exogenous shocks move it away):

```
dR(t) = θ(μ - R(t))dt + σ dW(t)
```

Where:
- `θ` — mean reversion speed
- `μ` — long-run baseline risk
- `σ` — volatility (scales with geopolitical tension)
- `W(t)` — Wiener process (random shocks)

This means you're not solving a routing problem once. You're solving it continuously on a graph whose weights are themselves stochastic processes.

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

The crucial insight: as λ increases, the algorithm switches from Hormuz-dependent routing to bypass routes. This switchover happens at a specific λ threshold, and the cost jump at that threshold is the **price of resilience** — what the market would need to offer to make the safe route economically rational under normal conditions.

---

## When Optimisation Isn't Enough

Dijkstra solves for the current state of the graph. It gives you the best path *right now*. But:

- The environment changes continuously
- Each routing decision affects congestion on chosen edges
- Uncertainty compounds over long planning horizons

This is where reinforcement learning becomes necessary — not as a research curiosity, but as a practical requirement.

### State Space, Actions, Rewards

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

The agent starts with high exploration (ε=0.5) and gradually shifts to exploitation as its Q-table matures.

What's interesting is what the agent *learns* — not just a good path, but a **policy**: a generalised mapping from (node, risk_conditions) → action that works across multiple graph states without re-solving from scratch.

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

```bash
git clone https://github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck
cd the-worlds-most-expensive-bottleneck
pip install -r requirements.txt
streamlit run app.py
```

The interactive app lets you:

1. **Network Map** — visualise the oil network on a world map with risk-coloured edges and optimal path highlighted in gold
2. **Route Finder** — sweep λ from 0→50 and watch the path switch from Hormuz to bypass routes
3. **Risk Simulator** — step through OU process risk evolution and observe path changes
4. **Stress Test** — trigger a Hormuz crisis and run 500 Monte Carlo disruption scenarios
5. **RL Agent** — train a Q-learning agent and compare its learned policy against Dijkstra

---

## The Real Question

The question isn't "what's the shortest path?"

The question is: **"What's the smartest path given that the map itself might change?"**

And the answer, from both the math and the data, is: build redundancy before you need it, price risk honestly, and never mistake optimisation for stability.

The Strait of Hormuz will remain a chokepoint. But the degree to which it's a *vulnerability* is a design choice — one the global energy system continues to make, quietly, every day.

---

*Built with NetworkX · Plotly · Streamlit · Q-Learning*
