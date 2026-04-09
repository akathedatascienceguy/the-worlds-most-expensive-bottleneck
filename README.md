# The World's Most Expensive Bottleneck

> An interactive blog where you can break the global oil network, watch it scramble in real time, and realise the "optimal" system was fragile all along.

**POC:** Risk-aware routing on a dynamic oil network — graph theory meets geopolitical risk.

---

## What This Is

A working proof-of-concept that models global oil logistics as a **dynamic, stochastic graph** and applies:

- **Risk-aware Dijkstra** — redefines edge weights as `cost + α·time + λ·risk(t)`
- **Ornstein-Uhlenbeck risk simulation** — mean-reverting stochastic risk evolution
- **Monte Carlo stress testing** — 500 Hormuz disruption scenarios
- **Q-Learning RL agent** — learns a routing *policy* rather than solving once

The Streamlit app is the interactive blog: read the narrative, then break things.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## App Sections

| Tab | What You Can Do |
|-----|-----------------|
| 🗺️ Network Map | See the oil network on a world map; risk-coloured edges, gold optimal path |
| 🛣️ Route Finder | Sweep λ (risk aversion) and watch the path switch from Hormuz to bypass routes |
| 📡 Risk Simulator | Step through stochastic risk evolution; watch paths adapt |
| 🔥 Stress Test | Trigger a Hormuz crisis; run 500 Monte Carlo disruption scenarios |
| 🤖 RL Agent | Train a Q-learning agent; compare its learned policy to Dijkstra |

---

## The Core Insight

> The system doesn't fail because we lack alternative routes.  
> It fails because we over-commit to the "best" one.

**Redundancy > Efficiency. Optionality > Optimisation.**

---

## The Graph

```
G = (V, E)

V: Saudi Arabia, UAE, Iraq, Kuwait, Qatar     ← producers
   Hormuz, Suez Canal, Strait of Malacca      ← chokepoints
   Yanbu, Fujairah                            ← bypass hubs
   India, China, Japan, Europe, USA           ← consumers

E: shipping lanes + pipelines, each with:
   cost | transit_time | capacity_mbd | risk(t)
```

Routing objective:

```
min Σ [cost(e) + α·time(e) + λ·risk(e, t)]
```

---

## Read the Blog

See [`blog.md`](blog.md) for the full technical write-up.

---

*Stack: NetworkX · Plotly · Streamlit · NumPy · Pandas*
