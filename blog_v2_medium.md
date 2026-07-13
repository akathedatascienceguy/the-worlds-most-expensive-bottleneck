# The World's Most Expensive Bottleneck: Second Edition

*Written by Yash Vardhan Gupta and Nikita Gupta*

---

*A note before you read: this project began as a simulation of a hypothetical crisis. By the time we published it, the crisis had arrived. What follows is the story of what we built — and what the world confirmed.*

---

## When the Simulation Became Real

*Written in late 2025. This section added June 2026.*

On February 28, 2026, the United States and Israel launched airstrikes on Iran. Within days, the IRGC issued warnings forbidding passage through the Strait of Hormuz, boarded merchant vessels, and laid sea mines in the channel. By May, open transits had fallen to near zero. The strait that carried 20% of the world's seaborne oil — the one we had been modelling as a hypothetical catastrophe — closed.

We built a simulation. The world ran the real thing.

Here is what we got right.

**The bypass gap was real, and it showed immediately.** The Saudi East-West pipeline hit its 7 MBD capacity milestone in March 2026 — exactly the figure we used. Saudi Arabia maxed it out within days of the crisis beginning. But here is what our model underestimated: of those 7 MBD arriving at Yanbu, approximately 3 MBD are consumed by domestic refineries and power plants. Net export capacity through the Yanbu terminals is closer to 4 MBD. Against 20 MBD of disrupted Hormuz flow, the real bypass gap is not 65%. It is closer to 80%.

**The cascade unfolded in the sequence we modelled.** Oil prices surged toward $98–132 per barrel. War-risk insurance premiums moved from 0.125% to 2.5% of hull value per seven-day period — and for some stranded tankers in mid-March, to 10% for a single voyage. Consumer prices followed. The IMF cut global growth forecasts. East Asia scrambled for alternative supply. The United States, producing 13 million barrels a day domestically, remained comparatively insulated. The geographic asymmetry we modelled was not a model artefact. It was the story on the front page.

**The λ switchover happened in hours, not weeks.** Every major carrier went to Cape of Good Hope routing the moment Hormuz became genuinely dangerous. The threshold the algorithm predicted — the precise point where bypass becomes cheaper in risk-adjusted terms — was crossed in real time, by real shipping companies, making real decisions with real money.

**The signals were there before the crisis peaked.** In the weeks before February 28, sentiment indicators in Gulf-related news were deteriorating. AIS data showed unusual vessel clustering near Omani coastal waters. Insurance premiums were already rising before the strikes happened. The exact leading-signal structure the LSTM was designed to read was present, in real data, in the real world. Whether any system actually used it to reroute early is a different question. Most didn't.

---

Here is what we did not model — and what we are building into v2 next.

**Private insurance markets didn't just get expensive. They withdrew.** P&I insurance — the liability coverage that protects ship owners against third-party claims — was cancelled for Gulf transits entirely from March 5. Ships that attempted the strait operated without standard P&I cover. This is not a cost increase. It is a different category of risk entirely. Our cascade model accounted for rising premiums. It did not account for the market simply closing.

**Governments became insurers of last resort.** When Lloyd's withdrew, the Trump administration directed the US International Development Finance Corporation to provide up to $40 billion in political risk reinsurance. Sovereign capital backstopping commercial shipping risk is a dynamic that does not exist in our model. It should.

**The pipeline got attacked.** In early April, the Saudi East-West pipeline — the primary bypass, the one we described as the most important alternative to Hormuz — was struck. Saudi Arabia restored it to full capacity within days, but the episode revealed something our static graph topology cannot represent: bypass infrastructure is not guaranteed to remain intact during the crisis it exists to mitigate. Edges can disappear. Our model assumes they don't.

**The economic tail was worse than our 95th percentile.** Our Monte Carlo simulation put the worst-case GDP impact at −2.5% to −4.0%. The Dallas Fed estimated a −2.9% annualised hit to global GDP in Q2 2026 alone, with prolonged disruption projections reaching −4.38% of global GDP at risk — $4.81 trillion. We were in the right range. But the right range for a tail event is cold comfort when you are living inside it.

The simulation was right about the structure. The world provided the numbers. The rest of this blog is the story of how we built it — and what we are updating now that we know what actually happens.

---

On September 14, 2019, two drone strikes hit the Abqaiq oil processing facility in Saudi Arabia. Within hours, 5% of the world's daily oil supply had vanished. By the time markets opened, Brent crude had jumped 15% — the largest single-day move in history.

Here is what is less often discussed: the shipping industry had seven days of warning.

Not seven days to act on it — seven days in which the signals were visible, in news tone, in insurance premium movements, in the quiet repricing of risk on routes that transited the Gulf. Seven days in which a system smart enough to read those signals could have rerouted. Most didn't. They were waiting for the price to move. By the time it did, the decision window had closed.

That is the problem v2 is built to solve.

v1 — the version we built first and wrote about [here](https://medium.com/culture-data-science/the-worlds-most-expensive-bottleneck-3d86aec769cc) — could do a great many things. It could model the global oil network as a dynamic graph. It could run five hundred disruption scenarios. It could tell you, to the percentage point, what a Hormuz closure costs. What it couldn't do was *see it coming*. It reacted to risk values it was told. It had no way to read the signals that precede a crisis — the sentiment drop, the insurance repricing, the futures volatility — and translate them into a routing decision before the crisis peaked.

v2 is the attempt to build that. Two upgrades. One to *perception* — replacing the hand-coded risk formula with a neural network that learns to predict rising risk from signals, the way a Lloyd's underwriter reads the same signals before pricing a Gulf transit. One to *generalisation* — replacing the Q-table that knew only what it had seen with a Deep Q-Network that reasons about conditions it has never encountered.

The goal, in one sentence: route seven steps before the market knows it's time to.

---

## Where V1 Left Off

If you haven't read v1, a quick orientation.

We built the global oil supply chain as a directed graph: 19 nodes (Gulf producers, maritime chokepoints, ocean waypoints, consuming regions) connected by 25 directed edges (shipping lanes and pipelines), each carrying real numbers for cost, transit time, throughput capacity, and a time-varying risk score. We then asked: what does optimal routing look like when "optimal" includes risk, not just cost?

The answer was a modified Dijkstra algorithm — the same shortest-path logic used by every GPS, but with edge weights redefined as cost + α·time + λ·risk. As the risk aversion parameter λ rises, dangerous routes become "expensive" in the algorithm's view, until at a precise threshold λ* the algorithm abandons Hormuz entirely and reroutes via Yanbu and the Cape of Good Hope. That threshold — the exact point where the bypass becomes cheaper in risk-adjusted terms — is the price of resilience.

We added a Monte Carlo stress-tester (500 random crisis scenarios) and an economic cascade model tracing a Hormuz disruption through oil prices, freight premiums, consumer inflation, food prices, and GDP across five global regions. v1's app also quietly shipped an experimental Q-learning routing agent — we never wrote about how it worked. That gap gets closed below, right before we replace it.

V1 answered the question: *given a crisis, what do you do?*

V2 asks: *can you see the crisis coming before it arrives?*

---

## What V2 Adds — and Why It Matters

Here is the honest summary of what changed:

**The risk model** — previously the Ornstein-Uhlenbeck stochastic process, which generated risk values from a mathematical formula with no connection to the outside world — is replaced by a two-layer LSTM neural network. The LSTM is trained on structured signals: oil price volatility, war-risk insurance premiums, and geopolitical sentiment. It learns the temporal relationships between these signals and the risk they collectively encode. When sentiment drops, the LSTM predicts rising risk — days before insurance premiums have caught up.

**The routing agent** — previously a tabular Q-learner storing one number per (state, action) pair — is replaced by a Deep Q-Network. The Q-table had a fundamental scaling problem: in a network with 24 edges and 5 risk levels per edge, the theoretical state space is 19 × 5²⁴ ≈ 60 trillion entries. The agent could visit ~18,000 of them in 600 training episodes. Everything else returned Q = 0 — random behaviour, precisely when it was needed most. The DQN replaces the table with a neural network that interpolates across continuous state space. It has never seen severity 0.91 before, but it has seen 0.88 and 0.94, and it reasons between them.

The two upgrades are not independent. The LSTM feeds its predicted risks into the DQN's state vector. The DQN routes based on what the LSTM *forecasts* risks will be — not what they currently are. When the LSTM detects rising Hormuz risk seven steps before the peak, the DQN is already preferring bypass routes. Dijkstra, reading only current values, is still routing through Hormuz.

Seven steps of advance warning. That is the number.

For a VLCC carrying $100 million of crude oil, seven steps is the difference between a course correction made in open ocean and one made halfway through a contested strait. It is the difference between a schedule adjustment and a war-zone passage.

---

## What V2 Looks Like in Practice

Like v1, the whole thing runs in a Streamlit browser app. The additions are in three tabs.

**The Training tab** is where the LSTM learns. You generate a synthetic dataset — 2,000 timesteps of risk, oil volatility, insurance premiums, and sentiment across all 24 edges — then train the network over 120 epochs. Watch the loss curves converge. When training is done, a per-edge RMSE table shows you which edges the model predicts most accurately (Cape of Good Hope: very well, since it barely moves) and which are hardest (Hormuz, because the crisis events are sharp and large).

**The DQN Agent tab** is where both routing agents live, in sequence. First, train the Q-learning baseline and watch it fit its 95-state table in a few hundred episodes. Then train the DQN for 600 episodes and watch the reward curve climb — from large-negative values when the agent wanders randomly, to a stable plateau when it has learned to reach the target reliably via low-risk paths. Run the greedy policy for both and compare them against Dijkstra on the current graph, side by side. Under normal conditions, all three tend to agree. Under crisis conditions, the DQN reroutes earlier — and the Q-learning agent's table-lookup limits become visible.

**The Model Internals tab** shows you inside the system while it runs: the LSTM's per-edge risk forecast alongside the current graph risk (the gap between them is the actionable signal), the DQN's Q-value heatmap (which nodes does the agent currently prefer from each position?), and the exploration rate ε decaying toward zero as training progresses.

The rest of the tabs — network map, route finder, risk simulator, stress test, economic cascade — are unchanged from v1. The graph is the same. The economic model is the same. Only the risk engine and the routing agent have been upgraded.

---

## Step 1: Learning to Read Signals

Before building the LSTM, we had to decide what it would read. And for that, we had to answer a prior question: what does geopolitical risk actually look like before it shows up as a number?

It doesn't announce itself. It leaks — through four types of signals, each moving at a different speed.

Think of four news feeds running simultaneously in the background of a shipping operations room. They all describe the same underlying reality, but none of them update at the same time.

**Sentiment moves first.** Language in news headlines and diplomatic statements tightens before a crisis materialises. A trained classifier reading Reuters and shipping advisories can detect escalatory language 3 to 5 timesteps before market prices fully reflect it. Sentiment is a *leading indicator*: it doesn't tell you how bad things will get, but it points in the right direction before anyone has finished pricing it in.

**Oil volatility moves concurrently.** When the risk of a Hormuz disruption rises, crude futures volatility co-moves almost immediately. Options markets are fast. Volatility is a *concurrent indicator*: it confirms the event but doesn't anticipate it.

**Insurance premiums lag.** Lloyd's of London and the P&I clubs reprice war-risk premiums based on a rolling window of incident data. The update mechanism is inherently backward-looking — premiums reflect what happened, smoothed over recent history. In a crisis, premiums are 7 to 10 timesteps *behind* the actual risk peak. A lagging indicator: accurate eventually, but the opportunity to act on it has already passed.

**Risk itself is latent — you can't observe it directly.** You back it out from these signals, the way a doctor estimates cardiovascular risk from blood pressure, cholesterol, and lifestyle data rather than from a number stamped on the patient. There is no public feed that says "Hormuz risk: 0.73."

```
Signal              Timing             What It Captures
─────────────────────────────────────────────────────────────────────────
Sentiment           Leads 3–5 steps    Diplomatic escalation, news tone,
                                        shipping advisories
Oil volatility      Concurrent         Futures market reaction to
                                        current risk level
Insurance premium   Lags 7–10 steps    Lloyd's rolling reprice from
                                        incident history
Risk (latent)       —                  What the model is trained to predict
```

*Image: a time-series chart with sentiment dropping first ("early warning"), risk and oil volatility spiking together, insurance catching up 7+ steps later — the gap between them is the anticipation window the LSTM exploits.*

Here is the insight this structure gives you: a model that uses sentiment as an input can, in principle, begin predicting rising risk before insurance markets have repriced. And a routing agent that acts on predicted risk, rather than current risk, begins rerouting before the market tells it to.

That is seven steps of advance warning. In practice.

---

## Step 2: Building a Memory

The right model for this problem is one that can hold information across time. Not a model that looks at the current signal values and makes a prediction. A model that remembers what sentiment looked like 5 steps ago, and what that usually means for insurance premiums 7 steps from now.

That is precisely what an LSTM — Long Short-Term Memory network — is designed to do.

A standard neural network processes each input independently. Feed it today's signal values and it gives you a prediction. Feed it tomorrow's values and it has no memory of today. For independent inputs this is fine. For time series with lag structure, it is exactly wrong.

An LSTM reads a sequence one step at a time and carries a *memory cell* — a learned internal state that persists across timesteps. At each step, three gates decide what to do with that memory:

- The **forget gate** decides what fraction of the old memory to erase
- The **input gate** decides what new information from this timestep to store
- The **output gate** decides what part of the memory to expose as the prediction

The result is a network that can learn to hold sentiment information for 5 timesteps, weight it appropriately when predicting future risk, and discount the insurance signal as a lagging confirmatory indicator rather than a predictive one. The lag structure is not hand-coded. It is learned from data.

Here is how the LSTM is built for this problem specifically:

```
Input:  10 timesteps × 24 edges × 4 signals  →  sequence of 10 × 96-dimensional vectors
              ↓
   LSTM Layer 1  —  hidden size 128,  dropout 0.2 on output
              ↓
   LSTM Layer 2  —  hidden size 128   (captures trends across the full window)
              ↓  last hidden state only
   Linear(128 → 64)  →  ReLU  →  Dropout(0.1)
              ↓
   Linear(64 → 24)  →  Sigmoid    (one risk prediction per edge, bounded in [0,1])
```

*Image: architecture diagram — 10-step input window → two stacked LSTM boxes with gate icons → linear head narrowing 128 → 64 → 24. ~251,000 parameters.*

Two design choices matter enough to name.

**Why sigmoid output?** Risk lives in [0, 1] by construction. Sigmoid enforces this constraint at the output layer — the model cannot predict 1.3, only values in the valid range. This is not a detail. A model that routinely saturates at the boundary gives a different qualitative signal from one that is calibrated within the range.

**Why two LSTM layers?** The first layer extracts local patterns from the signal values at each timestep. The second captures longer-range temporal dependencies — the multi-step relationship between a sentiment drop and an eventual insurance reprice. In testing, single-layer LSTMs underfit this structure.

The model trains on 1,989 sequences constructed from a 2,000-step synthetic dataset:

```
Epoch   Train MSE   Val MSE   What's Happening
──────────────────────────────────────────────────────────
0       0.040       0.045     Random initialisation
20      0.012       0.015     Signal structure starting to emerge
80      0.005       0.007     Learning rate reduced by scheduler
120     0.003       0.005     Convergence — no overfitting
```

*Image: loss curve — training MSE (blue) and validation MSE (orange) over 120 epochs, declining smoothly and staying close together (no overfitting).*

The real test is not the loss. It is whether the lag structure was actually learned. Diagnostic inspection of the model's predictions confirms: when sentiment drops in the input window, the LSTM begins predicting elevated Hormuz risk before the insurance premium in the same window has moved. It learned when to pay attention to which signal. That is the whole point.

*Image: three-panel overlay — actual risk (solid) vs. LSTM prediction (dashed) for Hormuz, Bab-el-Mandeb, and Cape of Good Hope. On Hormuz, the prediction rises 4–5 steps before the actual peak.*

**The conclusion:** the LSTM turns four observable signals into a calibrated forecast of next-step risk across all 24 edges. It does not know what will happen. It knows what the signals have historically meant — and it acts on that.

---

## Step 3: Teaching the Agent to Generalise

Routing is a decision problem. At each node, the agent must choose which neighbouring node to move to next, balancing cost, time, and risk, with the goal of reaching the destination. v1's app quietly shipped an experimental agent for this — tabular Q-learning — but we never actually wrote about how it worked. That omission gets fixed here, first, before we explain why v2 replaces it with a Deep Q-Network. The reason for the replacement is a single word: scale.

### How Q-Learning Actually Works

Think of it the way you'd learn to drive in a city you've never seen. At first you turn randomly. Over time you learn which turns lead to fast routes and which lead to dead ends. You don't memorise a single path — you build an *intuition* (a policy) for every intersection you might face. That is Q-learning.

Put in driving terms before the formalism: the *state* is which corner you're standing on and how bad the traffic looks from there; the *action* is which street you turn onto; the *reward* is how that particular turn actually worked out — a fast, cheap, safe stretch feels good, a slow, expensive, risky one feels bad, and finally arriving is worth a large bonus on top; the *goal* is to end the whole trip having felt as good as possible along the way, not to win any single turn in isolation. Formally, that is a Markov Decision Process:

```
Component     Definition
──────────────────────────────────────────────────────────────
State (s)     Where the agent is + the current Hormuz risk level
Action (a)    Which node to move to next
Reward (R)    −(cost + 40·risk + 2·time), +100 if target reached
Goal          Maximise total reward across the journey
```

The agent learns a **Q-function**: Q(s, a) — in driving terms, your gut-feel rating for "turning this way, from this corner, given today's traffic." Every time you actually take a turn and see how it goes, you don't discard your old gut-feel and replace it wholesale. You nudge it, a little, toward what you just learned. After taking action *a*, landing in state *s′*, and collecting reward *r*, that nudge is the Bellman update:

```
Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s′,a′) − Q(s,a) ]
```

Read the nudge as three ingredients, each with a driving-world meaning:

```
Symbol                  Meaning
────────────────────────────────────────────────────────────────
α (alpha)               Learning rate — how much one trip is
                          allowed to move your opinion. A stubborn
                          driver (low α) barely updates after a
                          single experience; a jumpy one (high α)
                          overreacts to every fluke. Set to 0.15.
γ (gamma)               Discount factor — how much weight you give
                          the rest of the journey beyond just this
                          turn. Low γ only cares about the next
                          block; high γ is already thinking about
                          the airport. Set to 0.9.
r + γ·max Q(s′,a′)      Bellman target — what Q *should* be, given
                          what you just experienced plus your best
                          guess about everything still ahead.
The bracket term        TD error — the surprise. The gap between
                          what you expected this turn to be worth
                          and what it turned out to be worth once
                          you saw where it led. A big surprise means
                          a big correction; no surprise means the
                          hunch barely moves.
```

Back to the learner-driver: at every intersection, you flip a weighted coin before deciding which way to turn. Heads, with probability ε, you take a random turn anyway — just to see where it leads. Tails, you take the turn you currently believe is fastest. On day one you don't trust your own judgment yet, so the coin is heavily weighted toward "explore anything" — ε starts at 0.5, meaning half your turns are deliberate detours into streets you haven't tried. As the trips pile up, you re-weight the coin toward "drive the route I already trust" — ε decays to 0.05, so by the end you're almost always taking the turn you believe in, only rarely still poking down an unfamiliar street just in case the city changed. That is ε-greedy: wander when you know little, commit once you know more.

What the agent ends up with, after enough trips, is not a memorised single best route from source to destination. It is closer to a driver's accumulated hunches at every intersection they've ever stood at — "from this corner, in this traffic, go left" — instantly recalled, never recalculated. Formally, that is a *policy table*: a lookup mapping (node, risk_level) → best_next_node. At inference time, routing is a table lookup, not a graph search. That is the appeal: instant decisions, no recomputation.

It is also the limitation. The table only knows what it visited during training.

### Why the Q-table Broke

Stretch the driving analogy one step further. The learner-driver above never actually learned the whole city's traffic — they learned one road's traffic report (how bad is the Hormuz route right now?) and used it as a stand-in for conditions everywhere. That is why the notebook of hunches stayed thin: 19 intersections × 5 traffic levels = 95 pages, memorisable in an afternoon of driving.

Now imagine the city instead hands you a live, independently-updating traffic report for 24 different roads at once, and asks for a hunch covering every combination of conditions across all of them. That is not a notebook anymore — it is 19 × 5²⁴ ≈ 60 trillion pages. No driver fills that notebook in a lifetime of trips, let alone a few hundred training episodes.

That is exactly the trap the Q-learning agent above was built to avoid, and exactly the wall it hits the moment you stop letting it avoid it. The agent built a lookup table — one entry per (state, action) pair, updated every time that pair was visited. After training, it simply looked up the Q-value for every available action and chose the highest. This works when the state space is small, which is exactly why the agent above deliberately kept its state compact: (current_node, Hormuz_risk_bucket) — 95 states, coverable in a few hundred episodes.

The compactness is the tell. A table can only stay small if you throw away information — here, all risk data except the Hormuz average. v2's graph carries a live, independently evolving risk value on all 24 edges, not just Hormuz's. A state that actually captures that is (current_node, risk_on_all_24_edges) — the 60-trillion-page notebook above.

Here is what 60 trillion looks like in training terms:

```
Agent                                        State Space     States Visited    Coverage
                                                              (600 episodes)
─────────────────────────────────────────────────────────────────────────────────────────
Q-Learning (as trained, compact state)       95              ~95               ~100%
Q-Learning (hypothetical, full 24-edge)      60 trillion     ~18,000           0.00003%
DQN                                          Continuous ℝ⁴³  —                 Interpolates
```

*Image: three state-space representations side by side — a fully-explored small grid, a vast sparse grid, and a smooth interpolated surface.*

For every unvisited state, the Q-table returns Q = 0. When all actions have Q = 0, argmax picks the first neighbour in dictionary order. Not random — *deterministic*, but meaningless. It is the driver arriving at an intersection with a blank page in the notebook, and turning left anyway, every single time, simply because "left" happens to be listed first — not because it's ever been right. The driver doesn't know they're lost. They just happen to always pick the same direction. In a crisis, when the routing decision matters most, this is exactly when novel risk combinations appear.

At test time, a tabular agent run against the full 24-edge state produced random-equivalent paths for roughly 40% of novel risk configurations.

### How the DQN Fixes It

The Deep Q-Network replaces the table with a neural network. Instead of storing Q-values, it *approximates* the Q-function as a learned continuous mapping from state to Q-values for all possible actions.

```
State vector (43 dimensions):
[ one-hot node encoding (19) | LSTM-predicted edge risks (24) ]

Network:
  Linear(43 → 256)  →  LayerNorm  →  ReLU  →  Dropout(0.1)
  Linear(256 → 128) →  ReLU
  Linear(128 → 19)                  ← one Q-value per possible next node
  Mask unreachable nodes to −∞
  argmax  →  routing decision
```

*Image: network diagram — 43-dim input widening to 256, narrowing to 128, then 19 Q-values as a bar chart with the highest bar highlighted. ~47,000 parameters.*

A state the network has never seen is not a cold miss. It passes through the network and produces Q-value estimates by interpolating from similar states in the high-dimensional space where it has trained. A crisis at severity 0.91 benefits from what the network learned at 0.88 and 0.94. The generalisation is not guaranteed to be perfect — but it degrades gracefully rather than collapsing to zero.

At test time, the DQN produced the correct bypass route in 100% of tested crisis scenarios, including risk combinations it had never encountered during training.

### Making Training Stable

Training a deep Q-network naively — update the network on each transition as it occurs — fails. Two specific problems appear, and DQN was designed with two specific fixes.

**The correlation problem.** Consecutive transitions in a routing episode share context: same graph, same episode, adjacent nodes. Training on them sequentially biases the gradient, causing the network to overfit recent trajectories rather than learning a general policy.

The fix: an **experience replay buffer**. A circular deque holding 10,000 past transitions. At each training step, 64 are sampled uniformly at random — decorrelated, diverse, drawn from ~700 different past episodes.

*Image: a linear episode chain next to a circular buffer with a random sampling arrow pulling non-adjacent transitions into a mini-batch.*

**The moving target problem.** The Bellman update target is r + γ · max Q(s′, a′; θ) — but θ is the same set of parameters being updated. Every gradient step changes the target, which changes the gradient, which changes θ again. The network chases a target that moves with every step. Q-values oscillate and diverge.

The fix: a **frozen target network**. A second copy of the DQN whose parameters are not updated during gradient steps. The Bellman target uses the frozen copy. Every 100 steps, the frozen copy is hard-updated from the current policy network. For those 100 steps, the target is stationary — the network has something stable to converge toward.

*Image: two network boxes — "Policy Net θ" updated by gradient descent, "Target Net θ⁻" hard-copied from it every 100 steps.*

One more choice: **Huber loss** instead of MSE. Early in training, TD errors — the gap between predicted Q-values and Bellman targets — are large and noisy. MSE squares these errors and amplifies them, causing large gradient spikes. Huber loss behaves like MSE for small errors and like MAE for large ones, capping the gradient during the chaotic early phase. In practice, Huber loss reduced TD error variance by ~60% in the first 100 episodes.

*Image: two-panel training chart — episode reward climbing from large-negative to a stable plateau; Huber loss declining from ~0.5 to ~0.05.*

**The conclusion:** the DQN solves the scaling problem the Q-table couldn't. It generalises by interpolation rather than enumeration, and it trains stably because experience replay and the target network eliminate the two root causes of naive deep RL instability.

---

## Step 4: Connecting the Pieces

The LSTM and the DQN are not two independent modules. They are coupled through the graph's risk state — and this coupling is where the system's most important behaviour lives.

On every simulation tick, this is what happens:

1. The LSTM reads the rolling window of the last 10 timesteps — risk, oil volatility, insurance premium, and sentiment across all 24 edges — and produces a predicted risk vector: one number per edge, for the *next* timestep.

2. Those predicted values are written directly into the graph: `G[u][v]["risk"] = LSTM_prediction`. The graph now reflects forecast risk, not current risk.

3. The DQN reads the updated graph and constructs its state vector: one-hot(current node) plus predicted edge risks. It selects the highest-Q action among reachable neighbours.

4. Dijkstra, running in parallel on the same graph, also reads the updated edge weights and finds the minimum-weight path.

The difference is that the DQN's policy was trained to route based on predicted risk, and has built up an implicit understanding of which states predict crises. Dijkstra simply minimises the current weighted cost. Both systems are reading the LSTM's forecast — but the DQN has *learned* what that forecast means for routing, across 600 training episodes.

*Image: swimlane diagram — LSTM signal window → risk forecast → graph edges shifting green → amber → red → DQN and Dijkstra diverging at crisis onset, then reconverging.*

Here is what this looks like at the level of a single crisis event:

```
Steps from Peak   Sentiment    Insurance          Actual Risk   LSTM Forecast   DQN Routing                Dijkstra
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
−5                Dropping     Flat               Low           Rising          Begins preferring bypass   Hormuz
−3                Low          Slightly rising    Moderate      High            Bypass                     Hormuz
0 (peak)          Low          Still catching up  High          High            Bypass                     Bypass
+7                Recovering   Still elevated     Declining     Declining       Returns to Hormuz           Hormuz
```

The market — represented here by the insurance premium — doesn't finish catching up until after the peak. The DQN rerouted three steps before the market finished deciding.

**The conclusion:** the LSTM gives the DQN an information advantage — predicted risk rather than current risk. The DQN has learned to exploit that advantage. Together, they do something neither can do alone: route ahead of the crisis, not into it.

---

## Step 5: What the Disruption Actually Costs

v1 introduced the economic cascade model as five broad stages, without day counts, calibrated against seven historical shocks. V2 doesn't re-derive that model — it does two things v1 didn't: pins the stages to an explicit day-by-day timeline, and, more substantively, runs 500 Monte Carlo scenarios through the cascade's *economic* outputs (oil, CPI, food, GDP), not just through routing cost the way v1's Monte Carlo did.

The timeline: oil spikes within days 1–7, with the size of the spike a function of expected duration —

```
Duration    Price Multiplier   Historical Calibration
─────────────────────────────────────────────────────────────
≤ 7 days    3.5×               2019 Abqaiq attack: oil +15%
≤ 30 days   5.5×               2005 Hurricane Katrina: oil +25%
≤ 90 days   8.0×               1990 Gulf War: oil +60%
> 90 days   12.0×              1973 Arab Embargo: oil +400%
```

— buffered partly by OPEC spare capacity (~35% offset) and the IEA's Strategic Petroleum Reserve (~17 days of full Hormuz flow), though a panic premium kicks in above severity 0.5 as inventory hoarding overshoots. Freight rates reprice in days 8–30 (Cape rerouting adds 14 days and ~$630k in bunker costs per voyage; in the 2019 Abqaiq episode the TD3C spot rate moved WS 60→300 in days). Consumer prices follow with a ~30-day lag, and food prices with a ~45-day lag through energy, fertiliser, and freight costs compounding together. Central banks respond in months 2–6, tightening into a supply shock they can't otherwise fix — each 1% of unexpected CPI implies roughly 50bps of tightening and 0.15% of GDP contraction through the credit channel, arriving after the direct shock and outlasting it.

The regional breakdown is refined from v1's version, now sourced to an IMF working paper and expressed as pass-through coefficients rather than ranges:

```
Region                          Oil Import      CPI Pass-   GDP Impact per
                                 Dependency      Through     10% Oil Rise
─────────────────────────────────────────────────────────────────────────
East Asia (Japan/Korea/China)   85%             0.18        −0.40%
India                           85%             0.16        −0.50%
Europe                          55%             0.13        −0.28%
USA                             15%             0.08        −0.15%
Developing Markets              80%             0.22        −0.60%
```
*Source: IMF Working Paper 17/53 (Gelos & Ustyugova 2017)*

East Asia absorbs roughly four times the GDP shock the USA does from an identical oil price move — same physical event, five different economic experiences.

**What's actually new: Monte Carlo on the cascade itself.** We ran 500 scenarios across the full range of severities and durations and looked at the distribution of *economic* outcomes, not routing cost. The 95th-percentile tail: oil +90–120%, global CPI +9–12%, global food prices +45–65%, GDP −2.5% to −4.0%. These aren't worst-case hypotheticals — they're the top five percent of a realistic disruption distribution, and unlike v1's routing-focused Monte Carlo, they're the number a planner would actually need to size a reserve or a hedge against.

*Image: Monte Carlo histograms for oil price, CPI, and food price change across 500 scenarios, with median and 95th-percentile markers.*

*Image: grouped bar chart by region (CPI, food price, GDP impact) — East Asia tallest, USA barely visible at the same scale.*

**The conclusion, unchanged from v1 but now quantified at the tail:** the cost of a Hormuz closure isn't the rerouting surcharge — it's oil volatility compounding into freight, into consumer inflation, into food prices, into central bank tightening, unequally by geography. The routing model tells you what to do. The cascade model, now with a tail distribution attached, tells you what's at stake if you don't.

---

## The Numbers, Plainly

Here is what the full v2 system trains to:

```
Component               Parameters   What It Does
─────────────────────────────────────────────────────────────────────
LSTM Risk Predictor      ~251,000    Reads 10-step signal window; predicts
                                       next-step risk across 24 edges
DQN Policy Network       ~47,000     Maps 43-dim state to Q-values;
                                       selects routing action
DQN Target Network       ~47,000     Frozen copy of policy net; provides
                                       stable Bellman targets
──────────────────────────────────────────────────────────────────────
Total                    ~350,000    Anticipatory routing: perceives
                                       before the market, routes before
                                       the crisis
```

And here is the comparison that matters:

```
Scenario                        Q-Learning (baseline)     LSTM + DQN (v2)
──────────────────────────────────────────────────────────────────────────
Normal conditions                Correct — Hormuz route    Correct — Hormuz route
Seen crisis severity              Correct bypass            Correct bypass
Unseen risk combination           ~40% random paths         Principled interpolated estimate
7 steps before crisis peak        No rerouting              Begins preferring bypass
Post-crisis recovery              Abrupt switch back        Smooth decay following LSTM forecast
Novel route not in training       Q = 0, meaningless        Non-zero Q from similar states
```

*Image: Q-value heatmap (rows = current node, columns = next node), shown at low vs. high Hormuz risk — the colour pattern shift is the agent's learned crisis response.*

---

## What V2 Is, Actually

Strip away the implementation and the question v2 is really asking is this: can you see a Hormuz disruption coming before you're already inside it?

The evidence is: yes — under one specific condition. If the leading signals (sentiment, AIS anomalies, diplomatic tension) are available and correlated with actual risk evolution, a trained LSTM can anticipate the risk trajectory 7 steps before lagging market indicators have finished repricing. In operational terms, for a VLCC transiting from Ras Tanura, that window is real. It is the difference between a routing update received at open sea and one received at the choke point.

The DQN answers the second question — not *what do I see* but *what do I do with it*. Its answer is a policy that generalises across an effectively infinite state space, trained in an environment where the bypass premium is a number and the cost of being wrong is quantifiable.

Together, the LSTM and the DQN do something neither can do alone. They route ahead of the crisis. Not into it.

---

## The Core Insight, Unchanged

v1 ended with a claim: the system fails not because alternatives don't exist, but because we over-commit to the cheapest route.

v2 doesn't change that claim. It goes one layer deeper.

The market doesn't just fail to pay the resilience premium. It also fails to see the crisis coming in time to make a choice. By the time insurance premiums have repriced, by the time freight rates have spiked, by the time the algorithm would naturally switch — the window has narrowed. The optimal moment to reroute is not when the crisis peaks. It is before the lagging signals have finished catching up.

A system that reads the leading indicators — that has learned the 7-step gap between sentiment dropping and insurance repricing, that routes on predicted risk rather than observed risk — has a different decision horizon. Not infinite. Not omniscient. But earlier than the market.

In logistics, earlier than the market is the only advantage that actually matters.

> Redundancy beats efficiency.
> Optionality beats optimisation.
> And anticipation beats reaction.

---

*v2 stack: NetworkX · Plotly · Streamlit · NumPy · PyTorch · scikit-learn*
*~350,000 parameters. LSTM risk engine + DQN routing agent + economic cascade across five global regions.*

> **Try V1 live:** [the-worlds-most-expensive-bottleneck.streamlit.app](https://the-worlds-most-expensive-bottleneck.streamlit.app)
> **Source code:** [github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck](https://github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck/tree/main)
