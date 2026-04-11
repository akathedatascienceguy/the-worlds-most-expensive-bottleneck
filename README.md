# The World's Most Expensive Bottleneck

> An interactive simulation where you can break the global oil network, watch it scramble in real time, and understand exactly what resilience costs.

**Two versions.** v1 is a self-contained POC. v2 adds a trained neural risk predictor, a Deep Q-Network routing agent, and a full macro-economic cascade model.

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
