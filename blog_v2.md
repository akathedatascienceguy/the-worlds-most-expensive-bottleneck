# The World's Most Expensive Bottleneck: Second Edition

Written by Yash Vardhan Gupta and Nikita Gupta

> Haven't read v1 yet? Start here: **[The World's Most Expensive Bottleneck](https://medium.com/culture-data-science/the-worlds-most-expensive-bottleneck-3d86aec769cc)**. This piece builds directly on it.

---

A note before you read: we built this after the Strait of Hormuz had already closed. There was no hypothetical to test, the crisis came first and the model came after, built from general structural data rather than this specific event. What follows is how well a structural model, never tuned to this crisis, held up against what actually happened.

---

## Built After the Fact

v1 was written in March 2026, shortly after the crisis began. This section added June 2026.

On February 28, 2026, the United States and Israel launched airstrikes on Iran. Within days, the IRGC forbade passage through the Strait of Hormuz, boarded merchant vessels, and laid sea mines in the channel. By May, open transits had fallen to near zero. The strait that carries 20% of the world's seaborne oil closed.

We did not see it coming. What we did was sit down afterward and ask a narrower question: could a structural model of this network, built from general data about pipeline capacity, historical war-risk premiums, and shipping routes, not from this specific event, say anything useful about a crisis it was never trained on?

## Where V1 Left Off

If you haven't read v1: we modelled the global oil supply chain as a directed graph, 19 nodes and 25 edges, each carrying cost, transit time, capacity, and a time-varying risk score, and asked what optimal routing looks like when "optimal" includes risk, not just cost.

The answer was modified Dijkstra, edge weights redefined as `cost + α·time + λ·risk`. As the risk-aversion parameter λ rises, dangerous routes get "expensive" until, at a precise threshold λ*, the algorithm abandons Hormuz entirely for Yanbu and the Cape of Good Hope. That threshold, the exact point where the bypass becomes cheaper in risk-adjusted terms, is the price of resilience. We added a 500-scenario Monte Carlo stress-tester and an economic cascade model tracing a disruption through oil, freight, inflation, food, and GDP across five regions. v1's app also quietly shipped an experimental Q-learning agent we never wrote about, and that gets fixed below, right before we replace it.

V1 answered: given a crisis, what do you do? V2 asks: can you see it coming before it arrives?

## What Actually Matched

The bypass gap was real and showed immediately. The Saudi East-West pipeline hit its 7 MBD capacity milestone in March, exactly the figure we used from general EIA data. Our model still underestimated it, though: about 3 of those 7 MBD are consumed domestically, so real net export capacity is closer to 4 MBD. Against 20 MBD of disrupted Hormuz flow, the real bypass gap isn't 65%. It's closer to 80%.

The cascade unfolded in the sequence we modelled too: oil toward $98 to $132 a barrel, war-risk premiums from 0.125% to 2.5% of hull value (10% for some stranded tankers), then consumer prices, then the IMF cutting growth forecasts, East Asia scrambling for supply while the U.S. stayed comparatively insulated. The λ switchover happened in hours, not weeks: every major carrier went to Cape of Good Hope routing the moment Hormuz became genuinely dangerous, crossing the same risk-adjusted threshold our algorithm identifies.

And the leading signals were there, visible only in hindsight: sentiment deteriorating, AIS data showing unusual vessel clustering, premiums already rising before the strikes. We only saw this by looking backward. It's exactly what convinced us to build the LSTM into v2.

## Where a Structural Model Still Falls Short

Private insurers didn't just raise prices, they withdrew entirely. P&I cover for Gulf transits was cancelled from March 5, a different category of risk than our cascade model accounts for. Governments became insurers of last resort, with the U.S. International Development Finance Corporation offering up to $40B in political risk reinsurance, sovereign capital backstopping commercial shipping risk, a dynamic our model doesn't have. The Saudi East-West pipeline, the bypass we called the most important alternative, got attacked itself in early April, revealing that our static graph assumes edges can't disappear during the crisis they exist to mitigate. And the economic tail was worse than our 95th percentile: we modelled worst-case GDP impact at -2.5% to -4.0%; the Dallas Fed's Q2 2026 estimate came in at -2.9% annualised, with prolonged-disruption projections reaching -4.38% ($4.81T) at risk.

The structure generalised further than we expected for a model that had never seen this crisis. The exact numbers we still had to borrow from general data: close enough to be useful, not close enough to trust blindly.

---

## The V2 Story

On September 14, 2019, two drone strikes hit the Abqaiq oil facility in Saudi Arabia. Within hours, 5% of the world's daily oil supply vanished, and Brent crude jumped 15% by the time markets opened, the largest single-day move in history.

Less often discussed: the shipping industry had seven days of warning beforehand, in news tone, insurance movements, quiet repricing on Gulf routes. Most didn't act. They waited for the price to move, and by the time it did, the window had closed.

That's the problem v2 is built to solve. v1 reacted to risk values it was told, with no way to read the sentiment drop, the insurance repricing, or the futures volatility that precede a crisis, and turn them into a routing decision before the peak.

v2 is the attempt to build that: two upgrades, not independent of each other. One to perception, a neural network that predicts rising risk from signals, the way a Lloyd's underwriter reads the same signals before pricing a transit. One to generalisation, a Deep Q-Network that reasons about risk conditions it's never seen, replacing a Q-table that only knew what it had visited. The goal, in one sentence: route seven steps before the market knows it's time to.

---

## What V2 Looks Like in Practice

Like v1, it runs in a Streamlit browser app, with the upgrades in three tabs. **Training** is where the LSTM learns: generate a synthetic dataset, train over 120 epochs, watch the loss converge, then see a per-edge RMSE table (Cape of Good Hope: easy, barely moves; Hormuz: hard, sharp crisis spikes). **DQN Agent** is where both routing agents live in sequence: train the Q-learning baseline first and watch its 95-state table saturate in a few hundred episodes, then train the DQN for 600 episodes and compare all three (Q-learning, DQN, Dijkstra) side by side, under normal and crisis conditions. **Model Internals** shows the LSTM's forecast against current graph risk, the DQN's Q-value heatmap, and ε decaying as training progresses.

Everything else, network map, route finder, risk simulator, stress test, cascade, is unchanged from v1. Only the risk engine and routing agent are new.

---

## Step 1: Learning to Read Signals

Before building the LSTM, we needed to know what it would read: what does geopolitical risk look like before it becomes a number? It doesn't announce itself. It leaks through four signals, each moving at a different speed, like four news feeds describing the same reality without ever updating in sync.

**Sentiment moves first.** Language in headlines and diplomatic statements tightens before a crisis materialises. A classifier reading Reuters and shipping advisories can detect escalation 3 to 5 timesteps before prices reflect it. It's a leading indicator. **Oil volatility moves concurrently.** Futures react almost immediately, confirming the event without anticipating it. **Insurance premiums lag.** Lloyd's and the P&I clubs reprice off a rolling window of incident data, 7 to 10 timesteps behind the actual peak: accurate eventually, useless for timing. **Risk itself is latent.** You back it out from these signals the way a doctor infers cardiovascular risk from blood pressure and cholesterol rather than a number stamped on the patient. There's no feed that says "Hormuz risk: 0.73."

```
Signal              Timing             What It Captures
─────────────────────────────────────────────────────────────────────────
Sentiment           Leads 3-5 steps    Diplomatic escalation, news tone
Oil volatility      Concurrent         Futures market reaction
Insurance premium   Lags 7-10 steps    Lloyd's rolling reprice
Risk (latent)       n/a                  What the model predicts
```

The insight this gives you: a model using sentiment as input can begin predicting rising risk before insurance has repriced, and a routing agent acting on predicted risk, not current risk, reroutes before the market tells it to. That's seven steps of advance warning, in practice.

---

## Step 2: Building a Memory

The right model here holds information across time. It remembers what sentiment looked like 5 steps ago and what that usually means for premiums 7 steps out. That's what an LSTM (Long Short-Term Memory network) is for. A standard network processes each input independently; feed it tomorrow's values and it has no memory of today. Wrong for lag structure.

An LSTM reads a sequence one step at a time and carries a memory cell across timesteps, governed by three gates: **forget** (what fraction of old memory to erase), **input** (what new information to store), **output** (what to expose as the prediction). The result learns to hold sentiment for 5 timesteps, weight it appropriately, and discount insurance as a lagging confirmatory signal rather than a predictive one, none of it hand-coded, all of it learned from data.

```
Input:  10 timesteps x 24 edges x 4 signals  ->  sequence of 10 x 96-dim vectors
              |
   LSTM Layer 1  -  hidden size 128,  dropout 0.2
              |
   LSTM Layer 2  -  hidden size 128   (longer-range trends)
              |  last hidden state only
   Linear(128 -> 64)  ->  ReLU  ->  Dropout(0.1)
              |
   Linear(64 -> 24)  ->  Sigmoid    (one risk prediction per edge, in [0,1])
```

Two choices worth naming. **Sigmoid output**: risk lives in [0,1] by construction, and a saturating model gives a qualitatively different signal than a calibrated one. **Two LSTM layers**: the first extracts local patterns while the second captures the multi-step relationship between a sentiment drop and an eventual reprice; single-layer LSTMs underfit this in testing.

The model trains on 1,989 sequences from a 2,000-step synthetic dataset, converging from a validation MSE of 0.045 to 0.005 over 120 epochs with no overfitting. The real test isn't the loss, though. It's whether the lag structure was actually learned. Diagnostic inspection confirms it was: when sentiment drops in the input window, the LSTM predicts elevated Hormuz risk before the insurance premium in that same window has moved.

**The conclusion:** the LSTM turns four observable signals into a calibrated forecast of next-step risk across all 24 edges. It doesn't know what will happen. It knows what the signals have historically meant, and acts on that.

---

## Step 3: Teaching the Agent to Generalise

At each node, the agent chooses which neighbour to move to next, balancing cost, time, and risk. v1's app quietly shipped a tabular Q-learning agent for this, never written about, before v2 replaces it with a Deep Q-Network. The reason for the replacement is one word: scale.

### How Q-Learning Actually Works

Think of it like learning to drive in a city you've never seen: you turn randomly at first, and over time build an intuition for every intersection, not a memorised path. In driving terms, the state is your corner and how bad the traffic looks; the action is which street you take; the reward is how that turn worked out (cheap and safe feels good, arriving is worth a bonus); the goal is to feel good across the whole trip, not win any one turn. Formally, that's a Markov Decision Process: state, action, reward `-(cost + 40·risk + 2·time)` plus 100 for reaching the target, and the goal of maximising total reward.

The agent learns a **Q-function** Q(s,a), your gut-feel rating for "turning this way, given today's traffic." Every actual turn nudges that rating, rather than replacing it outright. That nudge is the Bellman update:

```
Q(s,a) <- Q(s,a) + α [ r + γ · max Q(s',a') - Q(s,a) ]
```

α (learning rate, 0.15) is how much one trip can move your opinion: low α is a stubborn driver, high α overreacts to every fluke. γ (discount factor, 0.9) is how much weight you give the rest of the journey beyond this turn. The bracketed term is the TD (Temporal Difference) error, the surprise, the gap between what you expected the turn to be worth and what it actually was, which drives the correction.

Notice Q appears on both sides of its own update. Today's rating for a turn depends on tomorrow's rating for the turns after it. There's no clean formula to solve for Q directly; you just apply the update, turn after turn, until the numbers stop moving. That's what "training" means here, mechanically. The equation is named for Richard Bellman, who formalised this recursive structure for dynamic programming in the 1950s, long before anyone used it to route oil tankers.

Training itself balances explore versus exploit: with probability ε you take a random turn just to see where it leads, otherwise you take the turn you already trust. ε starts at 0.5 and decays to 0.05, so the agent wanders early and commits later. What it ends up with is a policy table, a lookup mapping (node, risk level) to best next node, instantly recalled rather than recalculated.

### Why the Q-table Broke

The learner-driver above never learned the whole city's traffic, only one road's report (how bad is Hormuz right now?), used as a stand-in for everywhere. That's why the notebook stayed thin: 19 intersections times 5 traffic levels equals 95 pages, memorisable in an afternoon.

Now imagine the city hands you a live report for 24 different roads at once, and asks for a hunch covering every combination. That's not a notebook anymore. It's 19 × 5²⁴ ≈ 60 trillion pages, and no driver fills that in a lifetime of trips, let alone a few hundred training episodes:

```
Agent                                        State Space     Visited (600 ep)   Coverage
──────────────────────────────────────────────────────────────────────────────────────
Q-Learning (as trained, compact state)       95              ~95                ~100%
Q-Learning (hypothetical, full 24-edge)      60 trillion     ~18,000            0.00003%
DQN                                          Continuous ℝ⁴³  n/a                  Interpolates
```

For every unvisited state, the Q-table returns 0, and argmax picks the first neighbour in dictionary order: deterministic, but meaningless, like a driver hitting a blank notebook page and turning left every time simply because "left" is listed first. In a crisis, when novel risk combinations are most likely, this is exactly when it matters. At test time, a tabular agent run against the full 24-edge state produced random-equivalent paths for roughly 40% of novel configurations.

### How the DQN Fixes It

The DQN replaces the table with a neural network that approximates Q(s,a) as a continuous mapping instead of storing it:

```
State (43-dim): one-hot node (19) | LSTM-predicted edge risks (24)
Linear(43->256) -> LayerNorm -> ReLU -> Dropout(0.1)
Linear(256->128) -> ReLU
Linear(128->19)  <- one Q-value per possible next node, mask unreachable to -inf, argmax
```

A state it's never seen isn't a cold miss. It interpolates from similar states in the space where it trained. Severity 0.91 benefits from what the network learned at 0.88 and 0.94, degrading gracefully instead of collapsing to zero. At test time, the DQN produced the correct bypass route in 100% of tested crisis scenarios, including risk combinations never seen during training.

### Making Training Stable

Training naively, updating on each transition as it happens, fails for two reasons, each with a standard fix. **Correlation:** consecutive transitions share context (same graph, same episode), biasing the gradient toward recent trajectories. Fixed with an **experience replay buffer**, a 10,000-transition circular deque, sampling 64 at random per step, decorrelated and drawn from around 700 different episodes. **Moving targets:** the Bellman target uses the same parameters being updated, so the target shifts with every gradient step and Q-values can oscillate or diverge. Fixed with a **frozen target network**, a second copy, hard-synced from the policy net every 100 steps, giving the network something stationary to converge toward.

One more choice: **Huber loss** instead of MSE, since early TD errors are large and noisy. MSE would amplify them into gradient spikes, while Huber caps the gradient during that chaotic early phase, reducing TD error variance by about 60% in the first 100 episodes.

**The conclusion:** the DQN solves the scaling problem the table couldn't, generalising by interpolation instead of enumeration, and trains stably because replay and the target network eliminate the two root causes of naive deep RL instability.

---

## Step 4: Connecting the Pieces

The LSTM and DQN are coupled, not independent. On every tick: the LSTM reads the last 10 timesteps of risk, volatility, insurance, and sentiment across all 24 edges, and produces a predicted risk vector for the next step. That vector gets written directly into the graph, `G[u][v]["risk"] = LSTM_prediction`, so the graph reflects forecast risk, not current risk. The DQN builds its state vector from the updated graph and picks the highest-Q reachable action. Dijkstra, running in parallel, reads the same updated weights and finds the minimum-weight path, but only the DQN was trained to route on what the forecast implies, across 600 episodes. Dijkstra just minimises current cost.

```
Steps from Peak   Sentiment    Insurance          Actual Risk   LSTM Forecast   DQN Routing                Dijkstra
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
-5                Dropping     Flat               Low           Rising          Begins preferring bypass   Hormuz
0 (peak)          Low          Still catching up  High          High            Bypass                     Bypass
+7                Recovering   Still elevated     Declining     Declining       Returns to Hormuz           Hormuz
```

The market, represented by the insurance premium, doesn't finish catching up until after the peak. The DQN rerouted three steps before it finished deciding. **The conclusion:** the LSTM gives the DQN an information advantage, predicted risk instead of current risk, and the DQN has learned to exploit it.

---

## Step 5: What the Disruption Actually Costs

v1 introduced the economic cascade model as five broad stages with no day counts. V2 pins it to an explicit timeline and, more substantively, runs 500 Monte Carlo scenarios through the cascade's economic outputs, not just routing cost the way v1's Monte Carlo did.

Oil spikes within days 1 to 7, sized by expected duration:

```
Duration    Price Multiplier   Historical Calibration
─────────────────────────────────────────────────────────────
<= 7 days   3.5x               2019 Abqaiq attack: oil +15%
<= 30 days  5.5x               2005 Hurricane Katrina: oil +25%
<= 90 days  8.0x               1990 Gulf War: oil +60%
> 90 days   12.0x              1973 Arab Embargo: oil +400%
```

Buffered partly by OPEC spare capacity (about 35% offset) and the SPR (about 17 days of full Hormuz flow), though a panic premium kicks in above severity 0.5. Freight reprices in days 8 to 30 (Cape rerouting: 14 extra days, about $630k per voyage). Consumer prices follow with a 30-day lag, food with 45 days, and central banks tighten in months 2 to 6, each 1% of unexpected CPI implying roughly 50bps of tightening and 0.15% of GDP contraction, arriving after the direct shock and outlasting it.

The regional breakdown is now sourced to an IMF working paper, as pass-through coefficients rather than ranges:

```
Region                          Oil Import      CPI Pass-   GDP Impact per
                                 Dependency      Through     10% Oil Rise
─────────────────────────────────────────────────────────────────────────
East Asia (Japan/Korea/China)   85%             0.18        -0.40%
India                           85%             0.16        -0.50%
Europe                          55%             0.13        -0.28%
USA                             15%             0.08        -0.15%
Developing Markets              80%             0.22        -0.60%
```
Source: IMF Working Paper 17/53 (Gelos & Ustyugova 2017)

East Asia absorbs roughly four times the GDP shock the USA does from an identical oil price move.

**What's actually new:** Monte Carlo on the cascade itself. 500 scenarios across the full range of severities and durations, looking at economic outcome distributions rather than routing cost. The 95th-percentile tail: oil +90 to 120%, global CPI +9 to 12%, food +45 to 65%, GDP -2.5% to -4.0%. That's the top five percent of a realistic distribution, and the number a planner would actually need to size a reserve or hedge against.

**The conclusion, unchanged from v1 but now quantified at the tail:** the cost of a Hormuz closure isn't the rerouting surcharge. It's oil volatility compounding into freight, inflation, food prices, and central bank tightening, unequally by geography. The routing model tells you what to do. The cascade model tells you what's at stake if you don't.

---

## The Numbers, Plainly

```
Component               Parameters   What It Does
─────────────────────────────────────────────────────────────────────
LSTM Risk Predictor      ~251,000    Predicts next-step risk, 24 edges
DQN Policy Network       ~47,000     Maps 43-dim state to Q-values
DQN Target Network       ~47,000     Frozen copy; stable Bellman targets
──────────────────────────────────────────────────────────────────────
Total                    ~350,000    Anticipatory routing
```

```
Scenario                        Q-Learning (baseline)     LSTM + DQN (v2)
──────────────────────────────────────────────────────────────────────────
Normal conditions                Correct, Hormuz route     Correct, Hormuz route
Unseen risk combination           ~40% random paths         Principled interpolated estimate
7 steps before crisis peak        No rerouting              Begins preferring bypass
Post-crisis recovery              Abrupt switch back        Smooth decay following forecast
Novel route not in training       Q = 0, meaningless        Non-zero Q from similar states
```

---

## What V2 Is, Actually

Strip away the implementation and the question is: can you see a disruption coming before you're already inside it? Yes, under one condition. If leading signals (sentiment, AIS anomalies, diplomatic tension) are available and correlated with actual risk, a trained LSTM can anticipate the trajectory 7 steps before lagging indicators finish repricing. The DQN answers the second question, not what do I see but what do I do with it: a policy that generalises across an effectively infinite state space.

v1 ended with a claim: the system fails not because alternatives don't exist, but because we over-commit to the cheapest route. v2 goes one layer deeper. The market doesn't just fail to pay the resilience premium, it fails to see the crisis coming in time to act. By the time premiums have repriced and freight has spiked, the window has narrowed. A system that reads leading indicators and routes on predicted risk rather than observed risk has a different decision horizon: not infinite, not omniscient, but earlier than the market. In logistics, earlier than the market is the only advantage that actually matters.

> Redundancy beats efficiency.
> Optionality beats optimisation.
> And anticipation beats reaction.

---

v2 stack: NetworkX, Plotly, Streamlit, NumPy, PyTorch, scikit-learn
~350,000 parameters. LSTM risk engine + DQN routing agent + economic cascade across five global regions.

> **Try V1 live:** [the-worlds-most-expensive-bottleneck.streamlit.app](https://the-worlds-most-expensive-bottleneck.streamlit.app)
> **Source code:** [github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck](https://github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck/tree/main)
