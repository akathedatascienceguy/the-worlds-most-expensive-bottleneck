# Seven Steps Ahead: Teaching the Bottleneck to Think

*Written by Yash Vardhan Gupta and Nikita Gupta*

---

The first version ended with a confession.

v1 could route around a crisis. It could quantify the price of resilience down to the percentage point. It could run five hundred disruption scenarios and tell you what the 95th percentile looks like. What it couldn't do was *see it coming*.

That is a bigger problem than it sounds. The shipping industry doesn't lose money when a crisis peaks — it loses money in the window between a crisis beginning and a routing system noticing. By the time the algorithm reacted, the decision had already been made for you.

Two things were holding v1 back. The risk model — the Ornstein-Uhlenbeck process — generated risk from a formula. It had no awareness of the world. It didn't know about the Houthi attack that sent Bab-el-Mandeb premiums from 0.05% to 2.0% hull in a week. It couldn't distinguish a random volatility spike from a genuine geopolitical escalation. It just drew from a distribution and moved on.

And the reinforcement learning agent had a deeper structural problem. It stored everything it knew in a table. A Q-table maps every (state, action) pair to an expected reward — and it only knows what it has seen. In a network with 24 edges and 5 risk levels per edge, the theoretical table has 19 × 5²⁴ entries. That is approximately 60 trillion states. In 600 training episodes of 30 steps each, the agent visited roughly 18,000. That is 0.00003% of the space. Everything else returned a Q-value of zero — which meant the agent fell back to random selection, precisely when the situation was most novel and the stakes were highest.

v2 fixes both of those things. One fix is about *perception* — learning to read signals before a crisis peaks. The other is about *generalisation* — routing intelligently through conditions the agent has never seen before. This is the story of how.

| Problem | v1 Approach | v2 Approach |
|---------|-------------|-------------|
| Risk prediction | OU formula — generates, doesn't learn | LSTM trained on structured synthetic signals |
| Q-function | Dict lookup — zero for unseen states | Neural network — interpolates across state space |
| Training stability | None | Experience replay + target network |
| State space | Discrete: 19 × 5²⁴ buckets | Continuous: ℝ⁴³ |
| Unseen states | Q = 0 → random fallback | Forward pass → learned estimate |
| Risk generalisation | None | LSTM interpolates unseen signal sequences |

*[Image: Side-by-side architecture overview — v1 showing OU process feeding a Q-table; v2 showing four signal inputs feeding an LSTM, which feeds predicted risks into the graph, which feeds the DQN policy network.]*

---

## The Signal Problem

Before replacing the risk model, we had to understand what risk actually *is* — not philosophically, but mechanically. What does geopolitical risk look like to a system that has to measure it in real time?

It doesn't announce itself. It leaks.

In the days and weeks before a major disruption, four types of signals move — at different speeds, in different directions, with different lead times. Think of them as four clocks, each running slightly out of sync with the truth.

**Sentiment moves first.** News headlines, diplomatic cables, shipping advisories. Language tightens around an escalation before it becomes one. A well-calibrated sentiment classifier can detect this tightening 3–5 steps before market prices have fully responded. Sentiment is a *leading* indicator — it won't tell you how bad things will get, but it points in the right direction before anyone has finished pricing it in.

**Oil volatility moves concurrently.** When Hormuz risk rises, crude futures volatility co-moves almost immediately. The futures market is fast. Volatility is a *concurrent* indicator — useful for confirmation, but it arrives at the same time as the event, not before it.

**Insurance premiums lag.** Lloyd's of London reprices war-risk premiums based on rolling incident data. In a crisis, premiums are 7–10 steps *behind* the actual risk peak. They're a *lagging* indicator — eventually accurate, but by definition not actionable early.

**Risk itself is the thing you're trying to infer.** It's not directly observable. You back it out from these signals, the way you'd estimate a runner's fitness from heart rate, pace, and recovery time — never from the number itself, because that number doesn't exist on a public feed.

| Feature | Timing Relative to Risk Peak | How It's Generated |
|---------|-----------------------------|--------------------|
| `sentiment` | **Leads 3–5 steps** | Inverse of risk; high sentiment = calm conditions |
| `oil_vol` | **Concurrent** | Mean-reverts toward Hormuz risk average |
| `insurance` | **Lags ~7 steps** | 85% autoregression + 15% current risk level |
| `risk` | Ground truth target | OU process + injected disruption events |

*[Image: Time-series chart across ~40 timesteps showing a simulated Hormuz crisis. Four lines: sentiment (drops first), actual risk (spikes at the event), oil volatility (co-moves with risk), insurance premium (catches up last). Vertical dashed lines annotate the lead window before the crisis and the lag window after. The visual makes the temporal structure of the signals immediate.]*

In v1, risk was generated directly and treated as observable. In v2, we generate all four signals together — encoding the real lead and lag relationships between them — and train a neural network to predict next-step risk from the signal window. The model never sees the ground truth at inference time. It has to infer risk from what it can observe, just like the real world.

This is the key shift. v1's OU process *was* the risk model. v2's LSTM *learns* the relationship between observable signals and the latent risk they encode. The difference is not just technical. It is the difference between a map drawn once from memory and one that updates as you walk.

---

## The Architecture of Anticipation

Let's make the LSTM concrete before making it precise.

An LSTM — Long Short-Term Memory network — is a type of neural network designed specifically for sequences. It reads one step at a time, carries a memory of what it has seen, and uses gated mechanisms to decide what to remember and what to forget. Unlike a standard network that treats every input independently, an LSTM's prediction at step 10 is informed by everything that happened at steps 1 through 9. That memory is exactly what we need: the lag between sentiment and insurance only becomes learnable if the model can hold one in memory while waiting for the other to move.

The LSTM in v2 takes a rolling window of the last 10 timesteps as input. At each step, it sees all 24 edges simultaneously — each with its four feature values. The input is a 96-dimensional vector per timestep (24 edges × 4 features). The model processes this sequence through two stacked LSTM layers of 128 hidden units each, then passes the final hidden state through two linear layers to produce a single risk prediction for each of the 24 edges.

```
Input:  (Batch, 10 timesteps, 24 edges × 4 features = 96)
              ↓
   LSTM Layer 1 — hidden size 128, dropout 0.2
              ↓
   LSTM Layer 2 — hidden size 128
              ↓  take last hidden state
   Linear(128 → 64) → ReLU → Dropout(0.1)
              ↓
   Linear(64 → 24) → Sigmoid
              ↓
Output: 24 predicted risk values, each in (0, 1)
```

*[Image: LSTM architecture diagram — left side shows the 10-step rolling input window with four colour-coded signal lanes (sentiment, oil vol, insurance, risk) per edge; centre shows two stacked LSTM layers with gate annotations (forget / input / output / cell state); right side shows the linear head projecting to 24 edge risk predictions with sigmoid bounding. Annotate the ~251,000 parameter count.]*

Two design decisions are worth naming explicitly.

**Why sigmoid output?** Risk lives in [0, 1] by construction. Sigmoid enforces this hard constraint from the output layer, rather than relying on clipping after the fact. A model predicting 1.3 risk on a 0–1 scale is not just wrong — it is semantically incoherent.

**Why two LSTM layers instead of one?** The first layer extracts per-timestep patterns from the signal structure. The second layer learns temporal dependencies *across* the sequence — trends, oscillation periods, the multi-step lag between sentiment and insurance. Single-layer LSTMs underfit the multi-signal structure in testing.

Training runs for 120 epochs on 1,989 sequences drawn from a 2,000-step synthetic dataset:

| Epoch | Train MSE | Val MSE | What's Happening |
|-------|-----------|---------|-----------------|
| 0 | ~0.040 | ~0.045 | Random initialisation — guessing |
| 20 | ~0.012 | ~0.015 | Early convergence — learning signal structure |
| 80 | ~0.005 | ~0.007 | Learning rate reduced by scheduler |
| 120 | ~0.003 | ~0.005 | Convergence plateau |

The train and validation curves track closely throughout — no diverging gap, no signs of overfitting. The synthetic data's lag-lead structure is rich enough that the model generalises without memorising.

*[Image: Dual-line loss curve — training MSE in blue, validation MSE in orange, over 120 epochs. Both curves decline steadily; step-down kinks visible around epochs 40–60 where ReduceLROnPlateau fires. The narrow train-val gap throughout confirms clean generalisation.]*

The real test, though, is not the loss number. It is whether the model learned the *structure*. When sentiment drops in the signal window, does the LSTM predict rising risk — before insurance premiums have moved? Yes. Diagnostic inspection confirms it. The model learned to use the leading signal as an early warning and to discount the lagging one as confirmatory-but-late. That causal structure, extracted from data, is the whole reason to train a model rather than write a formula.

*[Image: Predicted vs. actual risk overlay for three edges — Hormuz → Indian Ocean Hub (high-risk, large spike), Red Sea → Bab-el-Mandeb (mid-risk), Cape of Good Hope → Europe (low-risk, near-flat). LSTM predictions as dashed lines on top of ground truth. On the Hormuz edge, the prediction begins rising 4–5 steps before the actual peak — the anticipation window made visible.]*

---

## The Tabular Ceiling

Now for the routing agent. And for that, we need to understand exactly why v1's agent was limited — not because Q-learning is a bad algorithm, but because it was applied to a problem that was the wrong shape for it.

Tabular Q-learning builds a lookup table. Every time the agent visits a (state, action) pair, it updates the Q-value for that entry. At inference time, it retrieves the stored value and picks the best action. Simple, interpretable, mathematically sound — and completely dependent on having visited the states you care about.

Here is the problem. The state in our routing problem is `(current_node, risk_on_every_edge)`. In v1, we simplified this to just the Hormuz risk bucket: 19 nodes × 5 buckets = 95 states. The agent could cover that in a few hundred episodes.

In v2, the state includes all 24 edges. Even if we discretise each risk value to just 5 levels, the state space is 19 × 5²⁴ ≈ 60 trillion entries. In 600 training episodes of 30 steps each, the agent visits ~18,000 states:

| Version | State Space Size | States Visited | Coverage |
|---------|-----------------|---------------|---------|
| v1 | 95 | ~95 | ~100% |
| v2 tabular | ~60 trillion | ~18,000 | 0.00003% |
| v2 DQN | Continuous ℝ⁴³ | — | Interpolates |

*[Image: Logarithmic bar chart — three bars representing v1 (95 states, bar nearly full), v2 tabular (60 trillion, bar almost invisible at scale), and DQN (continuous, shown as a smooth gradient rather than a discrete bar to communicate that it covers by interpolation rather than enumeration).]*

For the 99.99997% of states the tabular agent has never seen, Q = 0. When all Q-values are equal, argmax returns the first element by iteration order — deterministic, but meaningless. The agent doesn't explore or reason. It just happens to pick the first neighbour in the dictionary. That is the failure mode: not chaos, but *frozen indifference*, deployed in the exact moments when a well-considered decision matters most.

The DQN replaces the table with a neural network. Instead of storing Q-values, it *approximates* the Q-function as a continuous mapping from state to Q-values. A state the network has never seen is not a cold miss — it receives a forward pass, and the network interpolates from similar states it has seen during training. A crisis at severity 0.91 benefits from what the network learned at 0.88 and 0.94. The generalisation degrades gracefully rather than collapsing to zero.

---

## The Two Stabilisers

Training a deep Q-network sounds straightforward — run episodes, observe transitions, update the network toward better Q-estimates. In practice, doing this naively diverges. Two specific failure modes appear, and DQN was designed precisely to eliminate them.

**The correlation problem.** Consecutive transitions in a routing episode are not independent data points. Step 4 shares context with step 3: same episode, same graph, adjacent nodes. Training on these correlated transitions pushes gradient updates repeatedly in the same direction — the network memorises recent trajectories rather than learning a general policy. It oscillates instead of converging.

The fix is an **experience replay buffer**: a circular deque holding the last 10,000 transitions. At each training step, 64 transitions are sampled *uniformly at random* from this buffer, not sequentially. Random sampling breaks the correlations. Each mini-batch contains transitions from across ~700 different episodes, giving the gradient a diverse, decorrelated signal to learn from.

*[Image: Experience replay diagram — left: a single episode shown as a chain of correlated transitions (s₁ → s₂ → s₃ → s₄...) with arrows showing how they all share context; right: the circular replay buffer as a deque with random sampling arrows drawing non-adjacent transitions into a mini-batch. Annotate: capacity = 10,000, batch size = 64.]*

**The moving target problem.** The standard Bellman update is:

```
target y = r + γ · max Q(s', a'; θ)
```

The target `y` depends on the same parameters θ that the network is actively updating. Every gradient step changes θ, which changes the target, which changes the gradient — a feedback loop that causes Q-values to oscillate wildly and often diverge entirely.

The fix is a **frozen target network**: a second copy of the network, `Q(s', a'; θ⁻)`, whose parameters are not updated on every step. The Bellman target now uses the frozen copy: `y = r + γ · max Q(s', a'; θ⁻)`. The target is stable for 100 gradient steps, long enough for the policy network to make meaningful progress against it. Every 100 steps, the target network is hard-copied from the policy network, and the cycle repeats.

*[Image: Target network diagram — two identical network boxes: "Policy Net θ" (receives gradient updates every step) and "Target Net θ⁻" (frozen). A solid arrow from transitions to Policy Net labelled "gradient update." A dashed arrow from Policy Net to Target Net labelled "hard copy every 100 steps." Bellman equation annotated: y = r + γ · Q_θ⁻(s', a').]*

Together, these two mechanisms transform an unstable loop into a tractable training problem. They are not engineering tricks — they are the exact structural fixes for the two failure modes of naive online deep Q-learning.

The DQN architecture itself is deliberately compact:

```
State vector (43 dimensions):
[ one-hot node encoding (19) | LSTM-predicted edge risks (24) ]

Network:
Input (43) → Linear(43 → 256) → LayerNorm → ReLU → Dropout(0.1)
           → Linear(256 → 128) → ReLU
           → Linear(128 → 19)    ← one Q-value per possible next node
           → Mask unreachable nodes to −∞
           → argmax → routing decision
```

~47,000 parameters — a fraction of the LSTM. LayerNorm after the first layer prevents the internal covariate shift that otherwise destabilises training on a continuously changing input distribution. The action mask ensures the agent never selects a node it cannot actually reach from its current position.

One last choice: Huber loss instead of MSE. Early in training, TD errors — the gap between predicted Q-values and Bellman targets — are large and noisy. MSE squares these errors, amplifying large mistakes and producing unstable gradient steps. Huber loss behaves like MSE for small errors and like MAE for large ones, bounding the gradient magnitude during the chaotic early phase without sacrificing precision once training stabilises. In practice, Huber loss reduced TD error variance by approximately 60% in the first 100 episodes compared to MSE.

*[Image: Two-panel training chart — left panel: episode reward curve over 600 episodes, raw rewards thin and noisy (red), rolling mean trending upward from large-negative to a stable plateau (orange); right panel: Huber loss curve declining from ~0.5 to ~0.05, with periodic spikes at target network sync points every 100 steps (annotated). Both curves confirm the agent is learning, not oscillating.]*

---

## The Interaction

The LSTM and the DQN are not two separate systems running in parallel. They share a single data structure — the graph's risk state — and this coupling is where the system's most important behaviour emerges.

Here is what happens on every simulation tick:

```
Rolling window W  (10 timesteps × 24 edges × 4 signals)
        ↓
      LSTM
        ↓
Predicted risks r̂  (one per edge, shape: 24)
        ↓
Graph updated:  G[u][v]["risk"] ← r̂[i]
        ↓
      ┌─────────────────────┐       ┌─────────────────────────┐
      │  Dijkstra           │       │  DQN                    │
      │  reads current r̂   │       │  state = [one_hot || r̂] │
      │  no memory          │       │  routes on forecast      │
      └─────────────────────┘       └─────────────────────────┘
```

*[Image: Three swim-lane pipeline diagram — top lane (LSTM): signal window → LSTM → risk forecast vector with sentiment dropping visibly 5 steps before the crisis; middle lane (Graph): risk vector written into edge weights, edges on the map transitioning from green to amber to red over time; bottom lane (DQN): state encoding → DQN → masked argmax → routing decision. A vertical dashed line marks "crisis onset" — left of it, DQN is already routing via bypass while Dijkstra still routes through Hormuz.]*

The DQN's routing decisions are always conditioned on *what the LSTM predicts risks will be*, not on what they currently are. When sentiment begins dropping 5 steps before a Hormuz crisis peaks, the LSTM starts forecasting rising risk on Hormuz-dependent edges. The DQN's state vector reflects that forecast. The DQN, trained to route away from high-risk edges, begins preferring bypass paths — before the crisis has even arrived.

Dijkstra, reading the same graph, sees only the current risk values. It switches to bypass only once those values have actually risen. By that point, the LSTM-DQN system has already rerouted.

| Steps from Crisis Peak | Sentiment | Insurance | Actual Risk | LSTM Forecast | DQN Routing | Dijkstra Routing |
|------------------------|-----------|-----------|-------------|---------------|-------------|-----------------|
| −5 | Dropping | Flat | Low | Rising | Starts preferring bypass | Hormuz |
| −3 | Low | Slightly rising | Moderate | High | Bypass | Hormuz |
| 0 (peak) | Low | Lagging behind | High | High | Bypass | Bypass |
| +7 | Recovering | Still elevated | Declining | Declining | Returns to Hormuz | Hormuz |

The 7-step anticipation window is not an abstract model property. For a VLCC carrying $100 million of crude transiting from Ras Tanura, seven steps is the difference between a course correction made at open ocean and one made while already entering the contested channel. It is the difference between a schedule adjustment and a war-zone passage.

---

## The Economic Cascade, in More Detail

Both versions of the app include an economic cascade model. v1 introduced it. v2 extended the calibration and added Monte Carlo tail-risk quantification. What neither document has fully described is *what the model actually computes* — and that is worth doing, because the numbers are not abstractions. They are estimates of what happens to ordinary people when a shipping route closes.

A Hormuz disruption does not arrive as a line item on a logistics invoice. It propagates through the global economy in six distinct phases, each with a measurable onset lag.

| Phase | Days | Oil Price Response | What's Driving It |
|-------|------|--------------------|-------------------|
| Shock | 0–3 | 0 → 100% of peak | Panic buying, futures squeeze |
| Peak | 4–7 | 100% | Maximum uncertainty; reserves not yet deployed |
| Reserve Deployment | 8–30 | → 70% | OPEC spare capacity + IEA SPR releases |
| Cape Rerouting | 31–90 | → 50% | New shipping equilibrium via longer routes |
| New Equilibrium | 91–180 | → 40% | Demand destruction; structural rebalancing |
| Recovery | Post-disruption | Exponential decay | Market normalisation |

How large the initial spike is depends almost entirely on duration. A 7-day disruption is bad. A 90-day disruption is an order of magnitude worse:

| Duration | Price Multiplier | Real Calibration Event |
|----------|-----------------|----------------------|
| ≤ 7 days | 3.5× | 2019 Abqaiq attack — oil +15% |
| ≤ 30 days | 5.5× | 2005 Hurricane Katrina — oil +25% |
| ≤ 90 days | 8.0× | 1990 Gulf War — oil ~+60% |
| > 90 days | 12.0× | 1973 Arab Embargo — oil +400% |

**Days 1–7.** Oil prices spike. Strategic reserves exist — OPEC can release up to ~3.5 MBD of spare capacity, and the IEA's SPR can cover roughly 17 days of full Hormuz flow — but above crisis severity 0.5, a behavioural panic premium kicks in: inventory hoarding, risk-off futures positioning, overshoot beyond the fundamental supply shortfall.

**Days 8–30.** Freight rates reprice. Tankers diverting to the Cape of Good Hope add 14 days of transit and roughly $630,000 in additional bunker costs per voyage. In the 2019 Abqaiq attacks, TD3C spot rates went from WS 60 to WS 300 — a 400% move — in days.

**Days 30–60.** Consumer prices begin reflecting the oil shock, lagged by approximately 30 days: wholesale repricing, retail shelf lag, statistical collection delay. But here is the structural inequality: the pass-through coefficient — how much of the oil price change reaches consumer prices — varies enormously by region.

| Region | Oil Import Dependency | CPI Pass-Through | GDP Elasticity | Food Import Dependency |
|--------|----------------------|-----------------|----------------|----------------------|
| East Asia (Japan/Korea/China) | 85% | 0.18 | −0.040 per 10% oil | 0.55 |
| India | 85% | 0.16 | −0.050 per 10% oil | 0.48 |
| Europe | 55% | 0.13 | −0.028 per 10% oil | 0.30 |
| USA | 15% | 0.08 | −0.015 per 10% oil | 0.20 |
| Developing Markets | 80% | 0.22 | −0.060 per 10% oil | 0.65 |

*Source: IMF Working Paper 17/53 (Gelos & Ustyugova 2017); IEA Energy Security 2023; World Bank Commodity Markets Outlook 2022.*

The United States, 15% import-dependent with deep strategic reserves, has a CPI pass-through of 0.08. Developing Markets, 80% import-dependent with no SPR and food-energy-poor households, have a pass-through of 0.22. The same oil shock. Very different economies on the other end of it.

**Days 45–90.** Food prices rise — through three compounding channels. Direct energy costs in agriculture (fuel, irrigation). Fertiliser costs, because nitrogen fertiliser is natural gas-based and oil-gas correlation runs ~0.60 in energy crisis periods. And freight costs on food imports, because the same ships carrying food are now paying war-risk premiums. The 2022 Russia sanctions validate the model: oil +60% → FAO food price index +34%, consistent with the calibration to within a few percentage points.

**Months 2–6.** Central banks respond. Supply-side inflation cannot be fixed by raising interest rates — but rates go up anyway, because that is the only tool available. Each 1% of unexpected CPI implies approximately 50 basis points of tightening via the Taylor rule, which via IS-LM approximation implies approximately 0.15% GDP contraction. This second-order monetary drag arrives after the direct energy drag, amplifies it, and persists well after oil prices have begun to normalise.

*[Image: 180-day time series chart — four lines (oil price, headline CPI, food price index, freight premium) plotted daily across all six phases. Phase boundaries annotated with vertical dividers. CPI line starts rising ~30 days after oil; food price ~45 days after. A dotted vertical line marks end of disruption; exponential recovery begins. The visual makes the cascade temporal structure immediately readable.]*

*[Image: Sankey diagram of the transmission chain — flows from "Hormuz Disruption" splitting into "Oil Supply Shock" and "Freight Rate Spike"; Oil Supply Shock branching into "Energy Cost Rise" and "Fertilizer Cost Rise"; converging through "Manufacturing Input Costs" and "Food Import Costs" into "CPI Inflation"; CPI splitting to "Central Bank Rate Hikes" and direct "GDP Contraction"; Rate Hikes feeding a second arrow into GDP Contraction. Flow widths proportional to impact magnitude. The compounding structure becomes visible as a shape, not just a list.]*

Across 500 Monte Carlo scenarios — severity drawn from U(0.30, 0.95), duration drawn uniformly from {3, 7, 14, 30, 60, 90, 180} days — the full cascade runs for each scenario across all five regions. The 95th percentile outcomes:

- Oil price: +90–120%
- Global CPI: +9–12%
- Global food prices: +45–65%
- Global GDP: −2.5% to −4.0%

*[Image: Three-panel Monte Carlo histogram — left: oil price change (%) across 500 scenarios; centre: global CPI impact; right: global food price change. Each histogram has a median marker and a 95th percentile marker. A vertical orange line shows the current scenario. The tail is clearly visible and extends beyond most historical events.]*

East Asia absorbs a GDP shock four times larger than the United States from the identical disruption. That is not a rounding error. It is a structural feature of the global trade architecture — a consequence of import dependency, strategic reserve depth, and food system fragility — and it is precisely why Hormuz is a different category of risk for Tokyo and Seoul than it is for Houston.

*[Image: Grouped bar chart — five regions on the x-axis, three bars per region: headline CPI impact, food price change, and GDP contraction, for a 90-day moderate-severity scenario. East Asia bars tower; USA bars are barely visible at the same scale. The geographic asymmetry of the cascade is impossible to miss.]*

---

## What the Numbers Say

v2 runs as a Streamlit app. The LSTM trains in-browser. The DQN trains against the live graph. You can watch the routing decisions diverge from Dijkstra in real time, from the moment the LSTM starts forecasting rising risk.

Here is what training produces:

| Component | Parameters | Key Result |
|-----------|-----------|------------|
| LSTM Risk Predictor | ~251,000 | Val MSE 0.005; detects Hormuz risk rise 7 steps before insurance repricing |
| DQN Policy Network | ~47,000 | 100% bypass routing at crisis; generalises to unseen risk combinations |
| DQN Target Network | ~47,000 | Frozen copy; hard-synced from policy net every 100 steps |
| **Total** | **~350,000** | **Full anticipatory routing system** |

And here is how the two systems compare across the scenarios that actually matter:

| Scenario | v1 Tabular Agent | v2 LSTM + DQN |
|----------|-----------------|---------------|
| Normal conditions | Correct — routes via Hormuz | Correct — routes via Hormuz |
| Seen crisis severity | Correct — reroutes via bypass | Correct — reroutes via bypass |
| Unseen risk combination | ~40% random paths | Principled interpolated estimate |
| 7 steps before crisis peak | No rerouting — unaware | Begins preferring bypass |
| Post-crisis recovery | Abrupt switch back to Hormuz | Smooth decay following LSTM forecast |
| Route never seen in training | Q = 0, meaningless selection | Non-zero Q from similar neighbouring states |

*[Image: Q-value heatmap — rows are current nodes (19), columns are possible next nodes (19), cell colour encodes Q-value magnitude from blue (low) to red (high), for the current graph risk state. Under normal conditions: producer rows show high Q-values toward Hormuz. Under crisis: those same cells dim, and Q-values toward Yanbu and Fujairah brighten. The agent's learned routing preference is visible as a colour pattern that shifts with risk.]*

The LSTM converges in 120 epochs. The DQN converges meaningfully in 600 episodes, though 1,200 produces a fully stable greedy policy — at 600 episodes the exploration rate ε is still 0.16, meaning one in six actions is still random. Training longer closes that gap.

The core comparison: at crisis conditions, the v1 tabular agent produced random paths for roughly 40% of novel risk combinations. The v2 DQN produced the correct bypass route in 100% of tested crisis scenarios, including combinations it had never seen during training. That is the generalisation property neural approximation provides — and that tabular methods structurally cannot.

---

## The Limitations That Remain

v2 is better than v1. It is not finished. Naming what is still missing is part of taking the work seriously.

| Limitation | What It Means in Practice | What Would Fix It |
|------------|--------------------------|-------------------|
| Synthetic training data | LSTM learns signal structure correctly, but calibration isn't grounded in real historical data | Live AIS feeds, Lloyd's premium series, Bloomberg/Reuters sentiment, GARCH vol from futures |
| Fixed graph topology | Canal closures, port blockades, pipeline shutdowns can't be represented | Dynamic edge removal and reconnection on event trigger |
| One-hot node encoding | DQN sees *which node* you're at, but not its structural role in the network | GNN encoder capturing betweenness centrality, degree, proximity to bypass hubs |
| Single-commodity routing | One tanker, one route — no congestion, no capacity competition | Min-cost max-flow formulation across multiple simultaneous flows |
| Point predictions only | LSTM gives a single risk estimate per edge — no confidence interval | MC Dropout or ensemble methods for calibrated uncertainty |
| DQN convergence at 600 episodes | ε ≈ 0.16 — one in six actions still random | 1,200–1,500 episodes for a fully greedy, stable policy |

The most important of these is the first. The LSTM's causal structure is correct — sentiment leads, insurance lags, oil volatility co-moves. But the specific parameter values come from synthetic data generated by a formula, not from historical incident logs. A production version would train on real signal time series and the model would be empirically calibrated, not just structurally plausible.

The others are extensions that trade off complexity for realism. Each is tractable. None is trivial. The roadmap is clear.

---

## What This Is, Actually

Strip away the implementation detail and what v2 is doing is this: it is trying to answer a question the energy market has never fully answered.

*Can you see a Hormuz disruption coming before you're already inside it?*

The evidence from v2 is: yes — under one specific assumption. If leading signals are available and correlated with actual risk evolution, a trained LSTM can anticipate the risk trajectory 7 steps before lagging market prices have finished repricing. For a tanker transiting from the Persian Gulf, that window is operationally real. It is the difference between a routing update received at Fujairah, with sea room to change course, and one received halfway through the Strait, with none.

The DQN answers a different question: not *what do I see* but *what do I do with it*. Its answer is a policy that generalises across an effectively infinite state space, degrades gracefully on novel inputs, and was trained in an environment where risk is costly and the bypass premium is a number — not a judgment call.

Together — the LSTM feeding its predictions into the DQN's state vector, the DQN routing against a forecast rather than a snapshot — the system does something neither component could do alone. It routes *ahead of the crisis*, not *into* it.

---

## The Core Insight, Unchanged

v1 ended with a claim: the system doesn't fail because alternatives don't exist. It fails because we over-commit to the cheapest route.

v2 doesn't change that claim. It adds a layer beneath it.

The market doesn't just fail to pay the resilience premium. It also fails to *see the crisis coming* in time to make a choice. By the time insurance premiums have repriced, by the time freight rates have spiked, by the time the algorithm would naturally switch routes — the decision window has narrowed. The optimal moment to reroute is not when the crisis peaks. It is several steps before the lagging signals have caught up with reality.

A system that reads the leading indicators — that has learned the 7-step gap between sentiment and insurance, that routes on predicted risk rather than observed risk — has a different decision horizon. Not infinite. Not omniscient. But earlier than the market. In logistics, that is the only advantage that actually matters.

> Redundancy beats efficiency.  
> Optionality beats optimisation.  
> And anticipation beats reaction.

---

*v2 stack: NetworkX · Plotly · Streamlit · NumPy · PyTorch · scikit-learn*  
*~350,000 parameters. LSTM risk engine + DQN routing agent + economic cascade across five global regions.*

> **Try v1 live:** [the-worlds-most-expensive-bottleneck.streamlit.app](https://the-worlds-most-expensive-bottleneck.streamlit.app)  
> **Source code:** [github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck](https://github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck/tree/main)
