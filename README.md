# The World's Most Expensive Bottleneck

*Written by Yash Vardhan Gupta and Nikita Gupta*

> An interactive simulation where you can break the global oil network, watch it scramble in real time, and understand exactly what resilience costs.

**Two versions.** v1 is a self-contained POC. v2 adds a trained neural risk predictor, a Deep Q-Network routing agent, and a full macro-economic cascade model.

---

Somewhere between a crude oil tanker leaving the Gulf and your Uber ride getting more expensive, lies a narrow stretch of water called the Strait of Hormuz.

Roughly 1 in every 5 barrels of oil passes through it.

Which is wild, because from a systems design perspective, that's like: routing 20% of global internet traffic through a single server and just… hoping it doesn't crash.

It's not that alternatives don't exist. Pipelines bypass Hormuz. Tankers can round the Cape of Good Hope. The Saudi East-West pipeline terminates at Yanbu on the Red Sea. But these alternatives are underutilised, under-capacity, and significantly more expensive.

The world didn't optimise for resilience. It optimised for cost. And in a world where nothing goes wrong, that's a perfectly rational choice.

The problem is that things go wrong.

Welcome to the world's most expensive bottleneck.

---

## What We've Built

We took that problem — one the energy industry has quietly lived with for decades — and modelled it.

Not with a spreadsheet. Not with a think-tank white paper. With a working simulation: a dynamic, stochastic graph of the global oil supply chain that you can break, stress-test, and watch scramble in real time.

Every barrel of oil has a journey. It starts in a Gulf oilfield, moves through pipelines to a port, crosses open water through straits that have been strategically contested for centuries, and ends up refined into the fuel that powers the city you're reading this in. That journey is a logistics problem. And like every logistics problem, it can be modelled as a graph.

We built that graph — 24 nodes, 24 directed edges, spanning Gulf producers, maritime chokepoints, pipeline bypass hubs, and consuming nations across Asia, Europe, and North America. Every edge carries real numbers: cost, transit time, throughput capacity, and a time-varying risk score calibrated to Lloyd's and S&P war-risk insurance premiums. Then we asked a simple question: what does the optimal route look like when the definition of "optimal" changes?

That's v1. A risk-aware routing algorithm that treats geopolitical instability as a cost — the higher your risk aversion, the more expensive a dangerous edge becomes, until the algorithm quietly abandons Hormuz and reroutes through Bab-el-Mandeb and the Cape of Good Hope. A tabular reinforcement learning agent that doesn't just solve the routing problem once, but learns a policy — a generalised intuition about what to do under any combination of conditions. And a Monte Carlo stress-tester that runs 500 disruption scenarios to put a number on what resilience actually costs.

The answer is 30–50% more per barrel. Every time.

v2 goes further. The risk model — previously a mathematical formula — is replaced by a trained neural network: a two-layer LSTM that learns to predict rising risk from structured signals before the market has finished pricing it in. News sentiment leads a crisis by 3–5 steps. Insurance premiums lag it by 7. The model learns the difference, and routes accordingly. The reinforcement learning agent becomes a Deep Q-Network, operating over a continuous 43-dimensional state space — finally capable of generalising to conditions it has never seen before.

And then, because routing cost alone is still just an abstraction, we built an economic cascade model. A Hormuz disruption doesn't just reroute tankers. It spikes oil prices, inflates freight premiums, passes through into consumer prices, disrupts food supply chains, triggers central bank responses, and ultimately contracts GDP — differently, in each region of the world. We modelled all of it, across five global regions, with Monte Carlo quantification of the tail-risk outcomes that planners should actually be designing for.

The whole thing runs in a Streamlit browser app. Trigger a crisis with a button. The graph re-routes. The cascade unfolds. The numbers update.

The simulation is clean. The underlying problem it's modelling is not.

---

## What This Is

A working proof-of-concept that models global oil logistics as a **dynamic, stochastic graph** and applies:

- **Risk-aware Dijkstra** — redefines edge weights as `cost + α·time + λ·risk(t)`
- **Ornstein-Uhlenbeck / LSTM risk simulation** — mean-reverting and learned risk evolution
- **Monte Carlo stress testing** — 500 Hormuz disruption scenarios
- **Q-Learning / Deep Q-Network** — learns a routing *policy* rather than solving once per state
- **Economic cascade model** — translates a Hormuz disruption into oil price, CPI, food price, and GDP impacts across 5 global regions

All numeric parameters are sourced from real data: EIA throughput figures, CEIC/FRED export volumes, Lloyd's/S&P war-risk insurance premiums, and Signal Group freight rates. See `DATA_SOURCES.md` for full citations.

---

## Quick Start

### v1 (no ML dependencies)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### v2 (PyTorch required)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r v2/requirements_v2.txt
streamlit run v2/app_v2.py
```

> **macOS note:** Do not use `pip install` without a venv — Homebrew Python blocks system-wide installs. Always activate the venv first.

---

## v1 — App Sections

| Tab | What You Can Do |
|-----|-----------------|
| 🗺️ Network Map | See the oil network on a world map; risk-coloured edges, gold optimal path |
| 🛣️ Route Finder | Sweep λ (risk aversion) and watch the path switch from Hormuz to bypass routes |
| 📡 Risk Simulator | Step through stochastic risk evolution; watch paths adapt in real time |
| 🔥 Stress Test | Trigger a Hormuz crisis; run 500 Monte Carlo disruption scenarios |
| 🤖 RL Agent | Train a tabular Q-learning agent; compare its learned policy to Dijkstra |

## v2 — App Sections

| Tab | What You Can Do |
|-----|-----------------|
| 🗺️ Network Map | Same world map with live LSTM-predicted risk overlay |
| 🛣️ Route Finder | λ sweep with DQN routing vs Dijkstra comparison |
| 📡 Risk Simulator | LSTM predicts next-step risk; OU fallback before training |
| 🔥 Stress Test | Crisis scenarios with DQN re-routing |
| 🤖 DQN Agent | Train neural agent with replay buffer + target network; inspect Q-value heatmap |
| 📉 Economic Cascade | Hormuz disruption → oil price → freight → CPI → food → GDP; 5-region breakdown; Monte Carlo tail risk |

---

## The Core Insight

> The system doesn't fail because we lack alternative routes.  
> It fails because we over-commit to the "best" one.

**Redundancy > Efficiency. Optionality > Optimisation.**

---

## The Graph

```
G = (V, E)   — 24 directed edges

V: Saudi Arabia, UAE, Iraq, Kuwait, Qatar      ← producers
   Hormuz, Suez Canal, Bab-el-Mandeb,
   Strait of Malacca, Cape of Good Hope        ← chokepoints
   Yanbu, Fujairah, Indian Ocean Hub,
   Red Sea                                     ← bypass / transit hubs
   India, China, Japan, Europe, USA            ← consumers

E: shipping lanes + pipelines, each with:
   cost | transit_time | capacity_mbd | risk(t) | hormuz_dependent
```

**Bypass graph is fully closed.** Saudi Arabia can reach India, China, and Japan without Hormuz via:

```
Saudi Arabia → Yanbu → Red Sea → Bab-el-Mandeb → Indian Ocean Hub → [India / Malacca → China / Japan]
```

The Cape of Good Hope also connects onward to India and Malacca for extreme-scenario routing.

Routing objective:

```
min Σ [cost(e) + α·time(e) + λ·risk(e, t)]
```

---

## File Structure

```
app.py                  ← v1 Streamlit app (self-contained)
requirements.txt        ← v1 dependencies
blog.md                 ← technical blog post
TECHNICAL.md            ← v1 technical report
DATA_SOURCES.md         ← real data citations for all parameters

v2/
  app_v2.py             ← v2 Streamlit app (self-contained, PyTorch)
  requirements_v2.txt   ← v2 dependencies (adds torch, scikit-learn)
  TECHNICAL_V2.md       ← v2 technical report (LSTM + DQN + Economic Cascade)
```

---

## Read More

- [`blog.md`](blog.md) — full technical narrative
- [`TECHNICAL.md`](TECHNICAL.md) — v1 implementation details
- [`v2/TECHNICAL_V2.md`](v2/TECHNICAL_V2.md) — v2 implementation (LSTM, DQN, economic cascade)
- [`DATA_SOURCES.md`](DATA_SOURCES.md) — all real data citations

---

*v1 stack: NetworkX · Plotly · Streamlit · NumPy · Pandas*  
*v2 stack: above + PyTorch · scikit-learn*
