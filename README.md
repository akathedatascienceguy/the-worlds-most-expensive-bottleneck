# The World's Most Expensive Bottleneck

*Written by Yash Vardhan Gupta and Nikita Gupta*

> An interactive simulation where you can break the global oil network, watch it scramble in real time, and understand exactly what resilience costs.

**Two versions.** v1 is a self-contained POC including a full economic cascade model. v2 upgrades the risk engine to a trained LSTM neural predictor and replaces the Q-learning agent with a Deep Q-Network.

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

We built that graph — 24 nodes, 25 directed edges, spanning Gulf producers, maritime chokepoints, pipeline bypass hubs, and consuming nations across Asia, Europe, and North America. Every edge carries real numbers: cost, transit time, throughput capacity, and a time-varying risk score calibrated to Lloyd's and S&P war-risk insurance premiums. Then we asked a simple question: what does the optimal route look like when the definition of "optimal" changes?

That's v1. A risk-aware routing algorithm that treats geopolitical instability as a cost — the higher your risk aversion, the more expensive a dangerous edge becomes, until the algorithm quietly abandons Hormuz and reroutes through Bab-el-Mandeb and the Cape of Good Hope. A tabular reinforcement learning agent that doesn't just solve the routing problem once, but learns a policy — a generalised intuition about what to do under any combination of conditions. A Monte Carlo stress-tester that runs 500 disruption scenarios to put a number on what resilience actually costs. And an economic cascade model, because routing cost alone is still just an abstraction.

The routing premium is 30–50% more per barrel. Every time. But that's only the tanker's bill. The Hormuz disruption doesn't stop at shipping costs — it spikes oil prices, inflates freight premiums, passes through into consumer prices, disrupts food supply chains, triggers central bank responses, and ultimately contracts GDP — differently, in each region of the world. We modelled all of it, across five global regions, with Monte Carlo quantification of the tail-risk outcomes that planners should actually be designing for.

v2 goes further on the algorithmic side. The risk model — previously a mathematical formula — is replaced by a trained neural network: a two-layer LSTM that learns to predict rising risk from structured signals before the market has finished pricing it in. News sentiment leads a crisis by 3–5 steps. Insurance premiums lag it by 7. The model learns the difference, and routes accordingly. The reinforcement learning agent becomes a Deep Q-Network, operating over a continuous 43-dimensional state space — finally capable of generalising to conditions it has never seen before.

The whole thing runs in a Streamlit browser app. Trigger a crisis with a button. The graph re-routes. The cascade unfolds. The numbers update.

The simulation is clean. The underlying problem it's modelling is not.

> **Try it live (v1):** [the-worlds-most-expensive-bottleneck.streamlit.app](https://the-worlds-most-expensive-bottleneck.streamlit.app) — no setup required.
>
> **Source code:** [github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck](https://github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck/tree/main)

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

## The Graph — G = (V, E)

Strip away the geopolitics and describe the oil supply chain the way a mathematician would — as a *directed graph*.

- **Nodes (V):** anything that oil passes *through* — a country, a strait, a pipeline terminal
- **Edges (E):** the connections between them — shipping lanes and pipelines

```
Producers  →  Chokepoints  →  Transit Hubs  →  Consumers
Saudi Arabia     Hormuz       Indian Ocean      China
UAE              Malacca      Red Sea           Japan
Iraq             Suez Canal   Cape of GH        Europe
```

```
G = (V, E)   — 25 directed edges

V: Saudi Arabia, UAE, Iraq, Kuwait, Qatar      ← producers
   Hormuz, Suez Canal, Bab-el-Mandeb,
   Strait of Malacca, Cape of Good Hope        ← chokepoints
   Yanbu, Fujairah, Indian Ocean Hub,
   Red Sea                                     ← bypass / transit hubs
   India, China, Japan, Europe, USA            ← consumers

E: shipping lanes + pipelines, each with:
   cost | transit_time | capacity_mbd | risk(t) | hormuz_dependent
```

Each edge carries four numbers:

| Attribute | What it means |
|-----------|--------------|
| `cost` | Shipping cost index (proportional to real VLCC freight rates) |
| `time` | Transit days (nautical miles ÷ 14 knots) |
| `capacity` | Max throughput in million barrels/day (real EIA figures) |
| `risk(t)` | Current geopolitical risk — changes over time |

Routing objective:

```
min Σ [cost(e) + α·time(e) + λ·risk(e, t)]
```

The conventional approach minimises raw cost (`min Σ c(e)`). The risk-aware approach adds time and risk — where λ is your risk aversion parameter: how much you're willing to pay in extra cost to route around danger. This is no longer the shortest-path problem. It's risk-aware routing on a dynamic graph.

### Edge Topology

The graph has 25 directed edges. Key structural properties:

**Hormuz funnel:** All five Gulf producers have a direct edge into the Hormuz node. Hormuz has a single outbound edge to Indian Ocean Hub at 20.0 MBD capacity — the bottleneck edge. All six of these edges are flagged `hormuz_dependent=True`.

**Bypass routes:**
- `Saudi Arabia → Yanbu` (Saudi East-West/Petroline, 7.0 MBD) — two onward paths depending on destination:
  - **Northbound** `→ Suez Canal` directly (~3 days, risk 0.10) — for Europe and USA. Yanbu sits at 24°N; Suez at 30°N. Ships sail straight up the Red Sea, bypassing Bab-el-Mandeb and its Houthi exposure entirely.
  - **Southbound** `→ Red Sea → Bab-el-Mandeb → Indian Ocean Hub` — for India, China, and Japan. Ships exit through the southern gate into the Indian Ocean.
- `UAE → Fujairah` (ADCO/Habshan pipeline, 1.5 MBD, ADNOC) `→ Indian Ocean Hub` directly

**Bypass graph is fully closed.** Saudi Arabia can reach all consumers without Hormuz:

```
Europe / USA:  Saudi Arabia → Yanbu → Suez Canal → [Europe / USA]
India / Asia:  Saudi Arabia → Yanbu → Red Sea → Bab-el-Mandeb → Indian Ocean Hub → [India / Malacca → China / Japan]
```

The Cape of Good Hope also connects onward to India and Malacca for extreme-scenario routing.

**Critical connectivity note:** The Yanbu bypass serves two structurally distinct functions. The northbound leg to Suez avoids both Hormuz and Bab-el-Mandeb risk — the cleanest bypass in the network. The southbound leg through Bab-el-Mandeb is the only Hormuz-free path to Asian consumers; `Bab-el-Mandeb → Indian Ocean Hub` is the edge that makes it possible.

**Capacity constraints:** Capacity is stored per edge but is not enforced as a hard constraint in the current routing implementation — it is surfaced as a metric (bottleneck capacity along the chosen path). Full capacity-constrained routing would require min-cost max-flow.

### Why Hormuz is a Single Point of Failure — Betweenness Centrality

Betweenness centrality measures how often a node appears on the shortest path between every pair of other nodes:

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

In network theory, Hormuz has several notable properties:

- **High betweenness centrality.** Remove Hormuz from the graph and ~80% of shortest producer→consumer paths break. No other node comes close.
- **High flow dependency.** Roughly 20 million barrels per day transit through it (EIA 2024). The next largest single-point dependency, the Strait of Malacca, handles 16.6 MBD — but *receives* oil from Hormuz, so it is downstream of the bottleneck, not parallel to it.
- **Catastrophic failure impact.** A single node failure doesn't just increase route cost — it can make the destination unreachable within time and capacity constraints, unless bypass alternatives with sufficient throughput exist.

This is why the λ-switchover requires a large risk penalty to trigger — the bypass is economically irrational under normal conditions. High betweenness centrality = high systemic risk. Redundancy means building *parallel paths* that reduce betweenness, not just adding capacity to the bottleneck.

---

## Risk Modelling — Ornstein-Uhlenbeck Process

Risk is not a static scalar — it evolves continuously in response to geopolitical events, market signals, and random shocks. The chosen model is the Ornstein-Uhlenbeck (OU) process: a mean-reverting stochastic differential equation.

Imagine a rubber band stretched between your hand and a wall. Let go — it snaps back toward the wall. That's mean reversion. Now add random gusts of wind that push it in unpredictable directions. That's the OU process.

$$dR(t) = \theta(\mu - R(t))\,dt + \sigma\,dW(t)$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| θ | Mean reversion speed | 0.3 |
| μ | Long-run baseline risk (calibrated per edge to Lloyd's / S&P war-risk insurance premiums) | varies |
| σ | Volatility (scales random shock magnitude) | 0.12 × slider |
| dW(t) | Wiener process — N(0,1) random shock | random |

**Why OU and not a random walk?** A random walk drifts forever — risk would eventually hit 0 or 1 permanently. OU always pulls back toward baseline: crises happen, escalate, and usually de-escalate. This mirrors real geopolitical risk dynamics. The window to reroute is not zero — there is time between a risk spike and a crisis peak. The question is whether your routing system is reactive (responds after the spike) or adaptive (responds during it).

### Hormuz Crisis Shock

A separate function applies a sudden, large risk spike to all Hormuz-dependent edges. In Monte Carlo runs, `severity` is sampled uniformly in [0.45, 0.95]. For the deterministic crisis button, it is set to 0.90. A ±0.05 jitter prevents unrealistic uniformity across edges.

**Results:**
- Mean reversion speed θ=0.3: a crisis spike decays back to baseline in ~10–15 simulation ticks
- At volatility=0.3 (default), risk standard deviation per tick ≈ 0.036 — consistent with observed daily war-risk premium fluctuations during moderate tension
- Bab-el-Mandeb base risk 0.35 reflects real 2024 post-Houthi premiums (down from 2.0% hull peak in 2023)
- A simulated Hormuz crisis (severity=0.90) raises route cost from 16 → 107 weighted units at λ=40, forcing automatic rerouting

---

## Routing — Risk-Aware Dijkstra

Dijkstra is like a GPS that always finds the fastest route — except instead of time, we minimise a custom edge weight that combines cost, transit time, and risk. It finds the minimum-weight path in O((V+E) log V) by always expanding the cheapest reachable node next. We keep the algorithm identical — we only change the weight function:

```python
def edge_weight(G, u, v, alpha=0.5, lam=20.0):
    return e["cost"] + alpha * e["time"] + lam * e["risk"]
```

**The λ effect:** At λ=0 the algorithm ignores risk and picks the cheapest route (always Hormuz). As λ increases, the risk penalty grows until it outweighs the cost saving — at that threshold the algorithm switches to a bypass route. The cost jump at that threshold is the **price of resilience**.

**The conclusion:** There is a precise price at which safe routing becomes economically rational. Below that λ threshold, shippers will always choose Hormuz — not out of ignorance, but because the math tells them to. Above it, the bypass wins automatically. No human judgment required.

**The usefulness:** It quantifies the resilience premium. You can tell a policymaker, a shipper, or an insurer exactly how much more expensive the safe route is under any given risk level. That number is the conversation. Everything else is politics.

**Results:**
- At base risk (λ=20): optimal route is `Saudi Arabia → Hormuz → Indian Ocean Hub → Malacca → Japan` — cost index 16, transit 30.3 days
- Switchover occurs at **λ ≈ 15.2** under crisis severity 0.90
- Above switchover: route flips to `Saudi Arabia → Yanbu → Red Sea → Bab-el-Mandeb → Indian Ocean Hub → Malacca → Japan` — cost index 26, transit 38.5 days
- **Cost of resilience: +62% in cost, +8.2 days in transit** — the exact premium the market refuses to pay upfront

---

## Reinforcement Learning — Q-Learning (v1)

Dijkstra solves a static snapshot of the graph. In practice, risk values change continuously, re-running Dijkstra on every tick is reactive not adaptive, and the algorithm has no memory of what worked under similar conditions in the past. A Q-learning agent addresses this by learning a *policy* — a mapping from (state, action) → expected cumulative reward — that generalises across graph states.

Like learning to drive in a city you've never seen. At first you turn randomly. Over time you build an *intuition* for every intersection you might face. That's Q-learning — not a memorised path, but a policy.

| Component | Definition |
|-----------|-----------|
| **State** s | Current node + Hormuz risk bucket |
| **Action** a | Which node to move to next |
| **Reward** R | −(cost + 40·risk + 2·time) + 100 if target reached |
| **Goal** | Maximise total reward across the journey |

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \cdot \max_{a'} Q(s', a') - Q(s,a)\Big]$$

ε-greedy exploration: starts at 50% random, decays to 5% as training progresses. The agent builds a policy table — a lookup that maps (node, risk_level) → best_next_node. At inference time, routing is a table lookup, not a graph search.

**The conclusion:** A system that learns from experience routes differently from one that optimises from scratch each time. The Q-learning agent develops something close to institutional memory — it knows to avoid Hormuz before the risk metric peaks, not after.

**The limitation:** The state space is effectively infinite (19 × 5²⁵). The agent can only learn what it has seen. Anything outside its training experience gets a Q-value of zero — random behaviour precisely when the situation is most unusual and the stakes are highest.

**Results:**
- Converges in ~300 episodes; reward curve plateaus and stabilises by episode 200–250
- At normal risk: agent matches Dijkstra — routes through Hormuz
- At crisis (Hormuz bucket = 1.0): agent correctly reroutes via Yanbu bypass, avoiding Hormuz exposure
- Key training insight: risk coefficient must be **40×** (not 10×) for the bypass's extra hops to be worth it in cumulative reward terms; and training must explicitly cover state bucket 1.0 (risk ≈ 0.93) to match inference at severity 0.90

---

## Monte Carlo Stress Testing

Runs 500 disruption scenarios at random severities and measures cost premiums and rerouting rates across the distribution.

**The conclusion:** At severity above 0.75, the network reroutes in almost every scenario — but pays 30–50% more every time. The tail risk is not rare. It is *priced but unpaid*. The market knows the bypass exists. It just refuses to pay for it preemptively.

**The usefulness:** It gives you a distribution, not a point estimate. Planners don't need to know what *will* happen — they need to know what the 95th percentile looks like and whether their system can absorb it.

**Results:**
- Severity 0.30–0.50: ~0% rerouting, negligible cost increase
- Severity 0.50–0.75: partial rerouting, +10–30% cost premium
- Severity 0.75+: ~100% of scenarios reroute away from Hormuz
- Cost premium at full rerouting: **+30–50% per route**
- Worst-case transit time increase: +8–12 days (adds 10–34 days total depending on destination)
- Bottleneck capacity drops from 20 MBD (Hormuz) to 7 MBD (Yanbu bypass) — a **65% throughput reduction** on the bypass corridor

---

## The V2 Evolution

### LSTM Risk Predictor — Changing Where the Risk Comes From

This is where data stops being decorative and starts doing actual work. Unlike in v1 where risk `r(e, t)` is directly observed, in v2 it's *inferred from signals*.

**Textual Signals**

News headlines, diplomatic statements, shipping advisories, and social media encode conflict escalation before it shows up in market prices. A transformer-based NLP model estimates a sentiment score from −1 (peaceful) to +1 (conflictual) that modulates the prior risk level.

**Time-Series Market Signals**

Oil price volatility, tanker day rates, and war-risk insurance premiums are real-time market aggregations of dispersed information. A sharp spike in insurance premiums for Hormuz-transiting vessels is a Bayesian update — the market knows something.

**Spatial Signals**

AIS tracking data gives vessel positions in near-real-time. Unusual clustering, vessels deviating from standard lanes, or suspicious transponder gaps are anomaly signals. Satellite imagery adds another layer.

All three signal types are fed to a two-layer LSTM that learns to predict next-step risk. Insurance premiums take ~7 steps to reprice after a risk event. Sentiment moves first. A model that reads sentiment can anticipate a crisis 7 steps before the market has fully priced it in — and reroute accordingly.

**The conclusion:** The market is slow. A model that reads sentiment can anticipate a crisis 7 steps before the market has fully priced it in — and reroute accordingly. For a VLCC carrying $100M of crude, the difference between rerouting on day 1 of an escalation versus day 8 is not abstract.

**Results:**
- ~251K parameters; 2-layer LSTM trained on 2,000-step synthetic time series
- Train MSE: 0.040 → 0.003 over 120 epochs; Val MSE: 0.045 → 0.005 (no significant overfitting)
- Successfully learns signal lag structure: uses sentiment drop as early warning 3–5 steps ahead of risk peak
- Predicts Hormuz risk elevation **7 steps before insurance premiums fully reprice** — enabling pre-crisis rerouting

---

### Deep Q-Network (v2)

Replaces the Q-table with a neural network that generalises across a continuous 43-dimensional state space. Tabular Q-learning fails at scale because it cannot generalise. The DQN can. It has never seen severity 0.91 before, but it has seen 0.89 and 0.93 — and it interpolates. This is the difference between a policy that works only in training conditions and one that works in the world.

**Results:**
- ~46K parameters; trains in 600 episodes with experience replay (buffer=10,000) and target network sync every 100 steps
- Huber loss stabilises training: TD error variance reduced ~60% vs MSE loss in early episodes
- At crisis: DQN routes via bypass in 100% of tested scenarios — matching or outperforming Dijkstra at equivalent λ
- Generalises cleanly to unseen risk combinations; tabular agent produces random paths on ~40% of novel states

---

### Economic Cascade Model (v1 + v2)

Because routing cost alone is still just an abstraction. A Hormuz disruption doesn't just reroute tankers — it spikes oil prices, inflates freight premiums, passes through into consumer prices, disrupts food supply chains, triggers central bank responses, and ultimately contracts GDP — differently, in each region of the world. Modelled across five global regions, with Monte Carlo quantification of tail-risk outcomes.

**The conclusion:** The cost of a Hormuz closure is not the rerouting premium. East Asia, 85% import-dependent, absorbs four times the GDP shock of the United States. Food prices spike before oil prices even peak. Central banks respond to inflation they cannot supply-side fix. The cascade is non-linear and geographically unequal.

**Results (90-day closure, moderate severity):**

| Region | Oil Import Dependency | Est. CPI Impact | Est. GDP Impact |
|--------|----------------------|-----------------|-----------------|
| East Asia (China, Japan, Korea) | ~85% | +4–6% | −2.5–3.5% |
| South Asia (India) | ~80% | +3–5% | −1.5–2.5% |
| Europe | ~60% | +2–4% | −1.0–2.0% |
| Middle East (non-Gulf) | ~40% | +1–3% | −0.5–1.5% |
| USA | ~15% | +0.5–1.5% | −0.3–0.8% |

- Food price impact leads GDP impact by 15–30 days due to fertiliser and freight pass-through
- Central bank rate responses add a second-order GDP drag 60–90 days post-closure
- 95th percentile (Monte Carlo tail): East Asia GDP impact reaches **−5%+** under extended closure

---

## The Core Insight

> The system doesn't fail because we lack alternative routes.  
> It fails because we over-commit to the "best" one.

**Redundancy > Efficiency. Optionality > Optimisation.**

---

## The Unified Conclusion

Each algorithm illuminates a different dimension of the same problem:

| Algorithm | Version | What it answers | Key result |
|-----------|---------|----------------|------------|
| Dijkstra | v1 + v2 | What is the optimal route, and what does safety cost? | Switchover at λ≈15; bypass costs +62% |
| OU Process | v1 + v2 | How does risk evolve, and how long do you have? | Crisis decays in ~10–15 ticks; window exists |
| Monte Carlo | v1 + v2 | What does the tail look like across 500 scenarios? | 100% rerouting above severity 0.75; +30–50% cost |
| Q-Learning | v1 | Can a system learn crisis-regime behaviour? | Yes — but only if trained on the right risk buckets |
| Economic Cascade | v1 + v2 | What does failure do to the real economy? | East Asia takes 4× the GDP hit of the USA |
| LSTM | v2 | Can you see a crisis before the market does? | 7 steps of advance warning from sentiment signal |
| DQN | v2 | Does the policy generalise to novel conditions? | Yes — 100% bypass rate vs ~60% for tabular agent |

Taken together: **the system works, the rerouting is possible, the bypass exists — and the market will not pay for it until it has no choice.** Every algorithm confirms a different facet of that same structural problem.

---

## Quick Start

### v1 — Live Demo

**[the-worlds-most-expensive-bottleneck.streamlit.app](https://the-worlds-most-expensive-bottleneck.streamlit.app)**, runs in your browser, no install needed.

### v1 — Run Locally

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
| 📉 Economic Cascade | Hormuz disruption → oil price → freight → CPI → food → GDP; 5-region breakdown; Sankey transmission chain; historical calibration; Monte Carlo tail risk |
| 📖 How It Works | First-principles explainers for every model in the app |

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

*v1 stack: NetworkX · Plotly · Streamlit · NumPy · Pandas — includes economic cascade model*  
*v2 stack: above + PyTorch · scikit-learn — upgrades to LSTM risk engine and DQN routing agent*
