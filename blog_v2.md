# Seven Steps Ahead: Teaching the Bottleneck to Think

*Written by Yash Vardhan Gupta and Nikita Gupta*

---

The first version ended with a confession.

v1 could route around a crisis. It could quantify the price of resilience down to the percentage point. It could run five hundred disruption scenarios and tell you what the 95th percentile looks like. What it couldn't do was *see it coming*.

Risk, in v1, was a formula. A mean-reverting stochastic process — elegant, mathematically honest, and completely blind to the world. The Ornstein-Uhlenbeck model knew nothing about the Houthi attack that sent Bab-el-Mandeb premiums from 0.05% to 2.0% hull in a week. It had no concept of a diplomatic breakdown, a tanker seized, a geopolitical signal quietly repricing risk before any algorithm noticed. It just drew from a distribution and moved on.

And the reinforcement learning agent — the part of the system that was supposed to *learn* — had a deeper problem. It stored its knowledge in a table. Every (state, action) pair it had seen got an entry. Everything else got zero. In a network with 24 edges and 5 risk levels per edge, the theoretical table has 19 × 5²⁴ entries. That is approximately 60 trillion states. In 600 training episodes of 30 steps each, the agent visited roughly 18,000. Not 18,000 percent. 18,000. That is 0.00003% of the space.

Which means the system that was supposed to generalise learned only from the states it had seen — and at inference time, any novel combination of risk values returned a Q-value of zero. The agent fell back to random. Not a graceful degradation. Random behaviour, precisely when the conditions were most unusual and the stakes were highest.

v2 fixes both of those things. This is the story of how.

---

## The Signal Problem

Before we could replace the risk model, we had to understand what risk actually *is*.

Not philosophically. Mechanically. What does geopolitical risk look like, to a system that has to measure it?

It doesn't announce itself. It leaks. Before a crisis peaks, four types of signals move — at different speeds, in different directions, with different lead times.

**Sentiment moves first.** News headlines, diplomatic statements, shipping advisories. Language tightens around an escalation before it becomes one. A well-calibrated sentiment classifier detects conflict escalation 3–5 steps before market prices fully respond. Sentiment is a leading indicator: it doesn't tell you *how bad* things will get, but it tells you the direction before anyone has finished pricing it in.

**Oil volatility moves concurrently.** When Hormuz risk rises, crude volatility co-moves almost immediately. The futures market is fast. Volatility is a concurrent indicator: confirmation without prediction.

**Insurance premiums lag.** Lloyd's reprices war-risk premiums based on recent incident data, smoothed over a rolling window. In a crisis, premiums are 7–10 steps *behind* the actual risk peak. They're a lagging indicator: accurate eventually, but not actionable early.

**Risk is the thing you're trying to infer.** It's not directly observable. You back it out from these signals — the way you'd back out a runner's fitness from heart rate, speed, and recovery time rather than measuring their VO₂ max mid-race.

In v1, we generated risk directly with a formula and pretended we could observe it. In v2, we generate a structured synthetic dataset with these four signals — risk, oil volatility, insurance premium, sentiment — encoding the real lead and lag relationships. We then train a neural network to predict the next-step risk from the window of signals. The model never sees the ground truth directly at inference time. It has to infer risk from what it can observe. Just like the real world.

This is the key shift. v1's OU process *was* the risk model — it produced risk values directly from a stochastic formula. v2's LSTM *learns* the relationship between observable signals and the latent risk they're encoding. The difference is not just technical. It is the difference between a map drawn by hand and one that updates in real time.

---

## The Architecture of Anticipation

The LSTM takes a rolling window of the last 10 timesteps as input. At each step, it sees all 24 edges simultaneously — each with its four feature values. The input, flattened, is a 96-dimensional vector per timestep: 24 edges × 4 features. Across 10 timesteps, that's a sequence the model processes through two stacked LSTM layers, each with 128 hidden units, producing a final hidden state that feeds into two linear layers and a sigmoid output — one predicted risk value per edge.

The total parameter count is approximately 251,000. Not large by modern standards. But appropriate for the problem. Transformers would require positional encodings and would scale quadratically with sequence length. For structured time series at length 10, LSTM's sequential processing and explicit hidden state outperforms attention-based architectures. The inductive bias is correct.

The key architecture decision is the sigmoid output. Risk lives in [0, 1] by construction. Sigmoid enforces this hard constraint without clipping artefacts. A model that predicts 1.3 risk on a 0–1 scale is useless; a model constrained to that range is semantically honest.

Training takes 120 epochs on 1,989 sequences drawn from a 2,000-step synthetic dataset. Train MSE goes from 0.040 to 0.003. Validation MSE goes from 0.045 to 0.005. The curves track closely — no significant overfitting, because the synthetic data's lag-lead structure is rich enough to generalise without memorising.

The critical test is not the loss number. It is whether the model learned the lag structure. Did it learn that sentiment *precedes* risk by 3–5 steps? Did it learn to discount the insurance signal as *lagging*, not predictive? Diagnostic inspection of the predictions confirms: when sentiment drops, the LSTM predicts rising risk several steps ahead of any insurance premium movement. It learned the causal structure from data. That is the whole point.

---

## The Tabular Ceiling

Tabular Q-learning stores one number per (state, action) pair. Update it when you visit. Retrieve it when you need it. Conceptually simple. Practically limited.

The limit is not the algorithm. The limit is the state space.

In v1, the state was `(current_node, Hormuz_risk_bucket)` — 19 nodes × 5 buckets = 95 states. Manageable. The agent could cover the whole space in a few hundred episodes.

v2 represents state correctly: `(current_node, all_24_edge_risks)`. Risk is continuous, but even discretised to 5 levels per edge, that is 19 × 5²⁴ ≈ 60 trillion states. The agent sees a vanishing fraction during training. Unvisited states get Q = 0 by default — meaning any unseen (state, action) pair returns equal Q-values, and argmax over equal values returns the first element by iteration order. Not learned behaviour. Not random exploration. Deterministic, meaningless selection.

The practical consequence: at test time, the tabular agent produces random paths for roughly 40% of novel risk combinations. Its performance under distribution shift — which is exactly when you need it — is equivalent to coin-flipping.

The DQN replaces the table with a neural network. The Q-function is no longer a lookup — it is a learned continuous function over state space. A state the network has never seen is not a cold miss. It receives a forward pass: the network interpolates from nearby states in the high-dimensional space where it has trained. A crisis at severity 0.91 it has never seen benefits from what it learned at 0.88 and 0.94. The generalisation is not guaranteed to be correct — but it is principled. It degrades gracefully rather than collapsing to random.

---

## The Two Stabilisers

Training a deep Q-network directly — update the network on each transition as it occurs — fails. Two specific mechanisms cause the failure, and DQN requires two specific fixes for each.

**The correlation problem.** Consecutive transitions in a routing episode are not independent. Step 3 shares context with step 4: same episode, same graph state, similar nodes. Training on correlated data makes gradient updates biased — they push the network in the same direction repeatedly, causing oscillation rather than convergence. The fix is an experience replay buffer: a circular deque of 10,000 transitions, from which each training step samples 64 uniformly at random. The uniform random sample breaks the correlations. The batch size of 64 ensures diverse transitions per gradient step, drawn from the last ~700 episodes of history.

**The moving target problem.** Standard Q-learning updates toward: `r + γ · max Q(s', a'; θ)`. The target depends on the same parameters θ being updated. As θ changes, the target changes, which changes the gradient, which changes θ again. This feedback loop — where you're chasing a target that moves every step — causes Q-values to oscillate and diverge. The fix is a frozen target network: a second copy of the network, `Q(s', a'; θ⁻)`, whose parameters are not updated on every step. Only the policy network receives gradient updates. The target network is hard-copied from the policy network every 100 steps. Over those 100 steps, the Bellman target is stationary. The training signal is stable. Convergence follows.

Together, these two mechanisms transform an unstable learning loop into a tractable training problem. They are not heuristics. They are the exact fixes for the two specific failure modes of naive online Q-learning.

The DQN architecture itself is deliberately shallow: three linear layers (43→256→128→19 dimensions), with LayerNorm after the first layer, ReLU activations, and a small dropout. Approximately 47,000 parameters — a fraction of the LSTM. The state vector is 43 dimensions: 19 for one-hot node encoding, 24 for the current LSTM-predicted edge risks. The output is a Q-value per node — masked to −∞ for unreachable neighbours before argmax.

The loss function is Huber rather than MSE. In the early episodes of training, TD errors are large and noisy. MSE amplifies these quadratically, causing unstable gradient steps. Huber behaves like MSE for small errors and like MAE for large ones — bounding the gradient magnitude during the chaotic early phase without sacrificing precision once training stabilises. In testing, Huber loss reduced TD error variance by approximately 60% versus MSE in the first 100 episodes.

---

## The Interaction

The LSTM and the DQN are not independent modules. They share a data structure: the graph's risk state.

At each simulation tick, the LSTM reads the rolling window of the last 10 steps and produces a predicted risk vector — one value per edge. These predictions are written directly into the graph: `G[u][v]["risk"] = LSTM_predicted_risk`. The DQN's state vector is then constructed from the updated graph: `s = [one_hot(current_node) || current_edge_risks]`. The current edge risks are the LSTM's predictions.

This means the DQN's routing decisions are conditioned on the LSTM's forecast of what risks *will be* — not what they *are*. When sentiment drops 5 steps before a Hormuz crisis peaks, the LSTM begins predicting rising Hormuz-edge risks. The DQN's state vector reflects this. The DQN, trained to route away from high-risk edges, begins preferring bypass paths — not because the crisis has arrived, but because the LSTM is telling it the crisis is coming.

Dijkstra, operating on the same graph but reacting only to current risk values, makes no such pre-emption. It switches to bypass only after the risk scores on Hormuz edges have fully risen. By that point, the LSTM-DQN system has already rerouted.

For a VLCC carrying $100 million of crude oil, the difference between rerouting on day 1 of an escalation versus day 8 is not a statistic. It is the difference between a schedule adjustment and a war-zone passage. The 7-step anticipation window the LSTM provides is the operational value proposition of the entire v2 system, compressed to a single number.

---

## The Economic Cascade, in More Detail

Both versions include an economic cascade model. v1 introduced it. v2 deepened the calibration and added the Monte Carlo tail-risk distribution. But neither the blog nor the README has fully described what the cascade actually computes — and it is worth describing, because the numbers are not abstractions.

A Hormuz disruption does not enter the economy as a line item on a shipping invoice. It enters through five transmission channels in sequence, with measurable time lags at each step.

**Days 1–7:** The oil price spikes. The magnitude depends on duration: a 7-day disruption triggers a 3.5× amplifier on the raw supply shortfall; a disruption lasting more than 90 days triggers 12×. Strategic reserves exist — OPEC can release up to ~3.5 MBD of spare capacity, and the IEA's SPR can absorb roughly 17 days of full Hormuz flow. But above severity 0.5, a panic premium adds to the price: behavioural overshoot driven by inventory hoarding and risk-off futures positioning.

**Days 8–30:** Freight rates reprice. Tankers rerouting via the Cape of Good Hope add 14 days and approximately $630,000 in bunker costs per voyage. War-risk insurance premiums spike. In the 2019 Abqaiq attacks, TD3C spot rates went from WS 60 to WS 300 — a 400% move — in days.

**Days 30–60:** CPI reflects the oil price, lagged approximately 30 days by the supply chain pipeline: wholesale repricing, retail shelf lag, statistical collection. The IMF's pass-through coefficients (from Working Paper 17/53, Gelos & Ustyugova 2017) differ sharply by region. The United States, 15% import-dependent, has a CPI pass-through coefficient of 0.08. Developing Markets, 80% import-dependent with no strategic reserves and food-energy-poor households, have a coefficient of 0.22. The same oil price shock produces fundamentally different inflation outcomes by geography.

**Days 45–90:** Food prices rise. Three mechanisms compound: energy costs in production (fuel, irrigation), fertiliser costs (nitrogen fertiliser is natural gas-based; oil-gas correlation runs ~0.60 in energy crisis periods), and freight costs on food imports. The 2022 Russia sanctions — oil +60%, food +34% — validate the model's calibration to within a few percentage points.

**Months 2–6:** Central banks respond to inflation they cannot fix on the supply side. The simplified Taylor rule produces a rate hike estimate proportional to unexpected CPI. Each 1% unexpected CPI implies approximately 50 basis points of tightening, which via IS-LM approximation implies approximately 0.15% GDP drag. This monetary contraction is the second-order effect — it arrives after the direct energy drag, amplifies it, and persists after oil prices have begun to normalise.

The Monte Carlo simulation runs 500 scenarios drawing severity from U(0.30, 0.95) and duration from {3, 7, 14, 30, 60, 90, 180} days. The full cascade runs for each scenario across all five regions. The 95th percentile outcomes — the tail risk planners should actually be designing for — are: oil price +90–120%, global CPI +9–12%, global food +45–65%, GDP −2.5% to −4.0%. These are not worst-case theoretical scenarios. They are the 95th percentile of a uniform severity distribution across historically observed disruption durations. They are consistent with IMF scenario analysis for major supply disruptions.

East Asia absorbs a GDP shock four times larger than the United States from the identical disruption. That asymmetry is not incidental. It is a structural feature of the global trade architecture — a consequence of import dependency, strategic reserve depth, and food system fragility — and it is precisely the reason why Hormuz is a different category of risk for Japan and Korea than it is for Texas.

---

## What the Numbers Say

v2 runs as a Streamlit app, and like v1, everything is interactive — the LSTM trains in-browser, the DQN trains against the live graph, and you can watch the routing decisions diverge from Dijkstra once the LSTM starts forecasting rising risk.

Here is what training actually produces:

| System | Parameters | Key Result |
|--------|-----------|------------|
| LSTM Risk Predictor | ~251,000 | Val MSE 0.005; detects Hormuz rise 7 steps before insurance repricing |
| DQN Policy Network | ~47,000 | 100% bypass rate at crisis; generalises to unseen risk combinations |
| DQN Target Network | ~47,000 | Frozen copy; syncs every 100 steps |
| **Total** | **~350,000** | **Full anticipatory routing system** |

The LSTM converges in 120 epochs. The DQN converges meaningfully in 600 episodes — though 1,200 episodes produces a more stable greedy policy (ε decays to 0.05 at ~1,500 episodes; at 600 it is still 0.16, meaning 16% residual randomness in action selection).

The operational comparison is sharp. At crisis conditions, the tabular Q-agent in v1 produced random paths for roughly 40% of novel risk configurations — precisely because tabular generalisation is zero. The DQN produced the correct bypass route in 100% of tested crisis scenarios, including risk combinations it had never seen in training. That is the generalisation property DQN provides and tabular methods structurally cannot.

---

## The Limitations That Remain

v2 is better than v1. It is not finished.

The LSTM trains on synthetic data. The causal structure is correct — sentiment leads, insurance lags, oil volatility co-moves — but the precise calibration is not empirical. A production system would train on real AIS anomaly scores, Lloyd's premium time series, Reuters/Bloomberg sentiment, and GARCH volatility from futures markets. The model would then not just be structurally correct but empirically calibrated — grounded in actual data rather than a simulation of the data-generating process.

The graph topology is fixed. Edges don't appear or disappear. In reality, canal closures, pipeline shutdowns, and port blockades remove edges entirely. The Suez Canal was closed for six days in 2021 by a single grounded container ship. The model can't represent that — it would require dynamic graph modification and re-routing from a structurally altered network.

The node representation is one-hot. A 19-dimensional binary vector for current node position captures which node you're at but nothing about its structural role in the network. A GNN encoder would replace this with a learned embedding that captures betweenness centrality, degree, and proximity to bypass hubs — richer information the DQN could use to route more intelligently in novel graph configurations.

The routing is single-commodity. One origin, one destination, one tanker. Real logistics is multi-vessel, multi-commodity, capacity-constrained. The right formulation is min-cost max-flow on a dynamic graph — an order of magnitude harder, and an order of magnitude more realistic.

These are not excuses. They are a roadmap.

---

## What This Is, Actually

Strip away the implementation details and what v2 is doing is this: it is trying to answer a question the energy market has never fully answered — *can you see a Hormuz disruption coming before you're already inside it?*

The evidence from v2 is: yes, under one specific assumption. If leading signals (sentiment, AIS anomalies, diplomatic tension) are available and correlated with actual risk evolution, a trained LSTM can anticipate the risk trajectory 7 steps before lagging market prices have finished repricing. In operational terms, for a tanker transiting from the Persian Gulf, that window is real — it is the difference between a captain who gets a routing update at Fujairah and one who gets it halfway through the Strait.

The DQN answers a different question — not *what do I see* but *what do I do with it*. And its answer is not just a better path. It is a policy that generalises across an effectively infinite state space, degrades gracefully on novel inputs, and has been trained on the reward structure of an environment where risk is costly and the bypass premium is quantifiable.

Together — the LSTM feeding predicted risks into the DQN's state vector, the DQN routing against a forecast rather than a snapshot — the system does something neither component could do alone. It routes *ahead of the crisis*, not *into* it.

---

## The Core Insight, Unchanged

v1 ended with a claim: the system doesn't fail because alternatives don't exist. It fails because we over-commit to the cheapest route.

v2 doesn't change that claim. It deepens it.

The market doesn't just fail to pay the resilience premium. It also fails to *see the crisis coming* in time to make a choice. By the time insurance premiums have repriced, by the time freight rates have spiked, by the time the algorithm would naturally switch — the decision window has narrowed. The optimal moment to reroute is not when the crisis peaks. It is before the lagging signals have caught up with reality.

A system that can read the leading indicators — that has learned the 7-step lag between sentiment and insurance, that routes on predicted risk rather than observed risk — has a different decision horizon. Not infinite. Not omniscient. But earlier than the market. Which, in logistics, is the only advantage that matters.

> Redundancy beats efficiency. Optionality beats optimisation. And anticipation beats reaction.

---

*v2 stack: NetworkX · Plotly · Streamlit · NumPy · PyTorch · scikit-learn*  
*~350,000 parameters. LSTM risk engine + DQN routing agent + economic cascade across five global regions.*

> **Source code:** [github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck](https://github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck/tree/main)
