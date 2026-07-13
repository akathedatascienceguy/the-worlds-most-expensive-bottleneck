# The World's Most Expensive Bottleneck

Somewhere between a crude oil tanker leaving the Gulf and your Uber ride getting more expensive, lies a narrow stretch of water called the Strait of Hormuz.

Roughly 1 in every 5 barrels of oil passes through it.

Which is a remarkable thing to be true about any single piece of geography. From a systems design perspective, this is the equivalent of routing 20% of global internet traffic through a single server and just… hoping it doesn't crash. No redundancy. No failover. Just the quiet assumption that the server stays on.

It's not that alternatives don't exist. Pipelines bypass Hormuz. Tankers can round the Cape of Good Hope. The Saudi East-West pipeline terminates at Yanbu on the Red Sea. But these alternatives are underutilised, under-capacity, and significantly more expensive.

The world didn't optimise for resilience. It optimised for cost. And in a world where nothing goes wrong, that's a perfectly rational choice.

The problem is that things go wrong.

Welcome to the world's most expensive bottleneck — The Strait of Hormuz.

---

## What Exactly Is the Strait of Hormuz?

The Strait of Hormuz is a narrow passage of water, roughly 33 kilometres wide at its narrowest, sitting between Oman to the south and Iran to the north. It connects the Persian Gulf to the Gulf of Oman, and from there to the Arabian Sea and the wider world.

Every supertanker leaving Saudi Arabia, Iraq, UAE, Kuwait, or Qatar passes through it on the way out. The strait handles approximately 20 million barrels per day of crude oil and petroleum products. One-fifth of the world's total oil consumption, moving through a channel you could drive across in under an hour.

Japan gets about 90% of its oil through Hormuz. South Korea, nearly the same. China gets over half its imports through it.

The scale of this dependency is not accidental. It is the accumulated result of seventy years of decisions: where to drill, where to build ports, where to lay pipelines. Each decision was individually rational. Collectively, they produced a single point of failure with no modern equivalent.

This is not a theoretical risk. In 2019, drone strikes destroyed 5% of the world's daily oil production overnight. Brent crude jumped 15% in a single day, the largest single-day percentage move in history. A few years later, Houthi attacks on Red Sea shipping sent war-risk insurance premiums from 0.05% of hull value to over 2%, a 2,700% increase in a matter of months. Major shipping lines rerouted around the Cape of Good Hope, absorbing weeks of additional transit and hundreds of millions in added costs.

That was a partial disruption of adjacent shipping. A Hormuz closure would be an order of magnitude larger.

---

## Why Can't You Just Go Around?

You can. The bypass routes exist. They just have problems.

The most important is the Saudi East-West Pipeline, running 1,200km from the Eastern Province to the port of Yanbu on the Red Sea, with a capacity of 7 million barrels per day. The UAE's Fujairah Pipeline adds another 1.5 MBD via an onshore route to the Gulf of Oman. And any tanker can round the Cape of Good Hope: no agreements required, just an extra 14 days of transit and 30% to 50% more cost per barrel.

Here's the honest summary:

Combined bypass capacity: 8.5 MBD against Hormuz's 20 MBD. That's a 65% throughput gap. The bypass exists. The math doesn't.

The market knows this. It just refuses to pay the premium preemptively, and waits until the strait is burning before doing the math again — this time with less time to prepare.

---

## What We've Built

We took that problem — one the energy industry has quietly lived with for decades — and modelled it.

Not with a spreadsheet. Not with a think-tank white paper. With a working simulation: a dynamic, stochastic graph of the global oil supply chain that you can break, stress-test, and watch scramble in real time.

The whole thing runs in a Streamlit browser app. No installation, no setup, no code.

1. Start with the sidebar. Pick a producer: Saudi Arabia, UAE, Iraq, Kuwait, or Qatar — and a destination: India, China, Japan, or the United States. Every tab responds to this selection. This is your baseline route. Then break something. Hit **Hormuz Crisis** and watch the network scramble in real time. The optimal path shifts, the cost jumps, and the map redraws itself around the crisis. That single button is the entire argument of this project made visible.

2. Once you have seen the crisis, explore what caused the reroute. Go to **Route Finder** and move the λ slider from 0 to 50. At low λ the algorithm ignores risk entirely and routes straight through Hormuz: cheapest, fastest, most exposed. As λ rises, watch the path switch. The exact moment it switches is the price of resilience.

3. Go to **Risk Simulator** and step the simulation forward. Risk evolves, the optimal path adapts, and the history of both accumulates in the chart below. The mean-reverting process ensures crises spike and decay — it does not stay broken forever, but it stays broken long enough to matter.

4. Go to the **Economic Cascade** and set the disruption duration. Watch a shipping event turn into a CPI number, a food price spike, a GDP contraction — region by region, day by day, across 500 simulated scenarios.

All numeric parameters are sourced from real data: EIA throughput figures, CEIC/FRED export volumes, Lloyd's/S&P war-risk insurance premiums, and Signal Group freight rates. See `DATA_SOURCES.md` for full citations.

**Give it a go yourself:** [the-worlds-most-expensive-bottleneck.streamlit.app](https://the-worlds-most-expensive-bottleneck.streamlit.app)

---

## Modelling It: A Graph Is the Right Abstraction

Strip away the geopolitics and what remains is a logistics problem. Oil moves from producers to consumers through a network of routes, pipelines, straits, shipping lanes — each with its own cost, capacity, and risk. The natural language for that kind of problem is a directed graph.

So we built one.

Formally, a directed graph: G = (V, E) — where nodes (V) are the entities oil passes through (producing countries, straits, pipeline terminals, ocean waypoints) and edges (E) are the connections between them. Every edge carries four attributes:

| Attribute | What it means |
|-----------|--------------|
| `cost` | Shipping cost index (proportional to real VLCC freight rates) |
| `time` | Transit days (nautical miles ÷ 14 knots) |
| `capacity` | Max throughput in million barrels/day (real EIA figures) |
| `risk(t)` | Current geopolitical risk — changes over time |

The graph has 19 nodes and 25 directed edges. All five Gulf producers feed into a single Hormuz node, which carries one outbound edge to the Indian Ocean Hub with a capacity ceiling of 20 MBD. One edge. The bottleneck of the entire global oil supply chain.

The conventional routing objective is simple — minimise total shipping cost:

```
min Σ c(e)   for e in path(source, target)
```

Under this objective, Hormuz wins every time. It is the cheapest path. It will always be the cheapest path — until something forces the math to change.

The question we asked is: what does optimal routing look like when "optimal" stops meaning cheapest?

---

## Why Hormuz Is Structurally Irreplaceable

Before the algorithms, it helps to understand precisely why Hormuz is so hard to route around. There is a formal measure for this in network theory called betweenness centrality.

It asks a deceptively simple question: for every pair of nodes in the network, how often does a given node appear on the shortest path between them?

$$C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$

- σst: total number of shortest paths from node s to node t
- σst(v): number of those paths that pass through node v

A node with betweenness centrality close to 1 sits on almost every shortest path in the network. Remove it, and the network doesn't slow down. It fragments.

Hormuz has the highest betweenness centrality in this graph by a considerable margin. Remove it, and approximately 80% of the shortest producer-to-consumer paths break. Not Malacca, not Suez, not Bab-el-Mandeb comes close to this structural position.

This is why the intuitive fix — building more capacity through bypass routes — doesn't actually solve the problem. Reducing betweenness centrality requires structurally parallel paths that genuinely compete with Hormuz end-to-end. At 8.5 MBD combined against Hormuz's 20 MBD, the bypasses don't qualify. They are emergency detours, not alternatives.

Which brings us to the real engineering challenge: if we cannot easily replace Hormuz in the network, can we at least build a system that routes around it intelligently when the risk of using it becomes too high? That requires making risk computable.

---

## Step 1: Making Risk Computable

So we have a graph. We know Hormuz dominates it. Now the real engineering question begins: how do you represent risk in a way a routing algorithm can actually use?

Risk is not a fixed number. It spikes when a tanker gets seized, then gradually settles. It is never truly zero, but it does not stay elevated forever either. Any honest model of geopolitical risk needs to capture three things simultaneously: it fluctuates randomly, it tends to drift back toward a baseline over time, and it can jump sharply when a discrete event occurs.

The model that captures all three is the Ornstein-Uhlenbeck process, a mean-reverting stochastic differential equation:

$$dR(t) = \theta(\mu - R(t))\,dt + \sigma\,dW(t)$$

It looks dense, but the intuition is clean.

Picture a pendulum. Disturb it and it swings, but gravity always pulls it back to centre. That pull toward the centre is the first term: θ(μ − R(t)) dt. Here μ is the long-run baseline risk — the resting position the system wants to return to — and θ controls the strength of that pull. A high θ snaps risk back quickly after a shock. A low θ allows it to linger.

Now imagine the pendulum is hanging in a room with unpredictable air currents — small random forces nudging it continuously in every direction. That is the second term: σ dW(t), a random shock drawn from a normal distribution at each time step. σ controls how violent those nudges are.

Together, the equation describes something that behaves remarkably like real geopolitical risk: always drifting toward a baseline, never quite settling, and occasionally jolted hard by events outside the model's control.

In discrete time — what actually runs in the simulation — this becomes:

```
new_risk = current_risk + θ * (μ — current_risk) + σ * N(0, 1)
new_risk = clip(new_risk, 0, 1)
```

Now, why not just use a random walk — `new_risk = current_risk + σ * N(0, 1)`, dropping the first term entirely?

Because a random walk has no sense of where it belongs. Without the mean-reverting term, risk accumulates its random shocks indefinitely. It can drift to zero and stay there, implying a world of permanent peace, or drift to one and stay there, implying permanent crisis. Neither is how the world works.

Consider the 2023 Houthi attacks. War-risk premiums on Bab-el-Mandeb spiked to 2,700% of their baseline almost overnight. A random walk would have no mechanism to bring them back down. But over the following twelve months, they did come back down — partially, gradually, exactly as mean reversion would predict. The crisis was real, the shock was real, and so was the decay.

Mean reversion is not a mathematical convenience. It is a structural feature of how geopolitical risk actually behaves: it escalates, it peaks, and absent new shocks, it slowly returns to something resembling normal.

The parameters we calibrated:

| Symbol | Meaning | Value |
|--------|---------|-------|
| θ | Mean reversion speed | 0.3 |
| μ | Long-run baseline (per edge, calibrated to Lloyd's / S&P war-risk insurance premiums) | varies |
| σ | Volatility — scales random shock magnitude | 0.12 × slider |
| dW(t) | Wiener process — N(0,1) random shock | random |

The baseline risk on each edge is not invented. It is read directly from the war-risk insurance market — which is the most honest signal available: it represents what underwriters are actually willing to charge to put capital at risk on each route.

| Route Segment | Base Risk | Insurance Premium Band |
|---------------|-----------|----------------------|
| Producers → Hormuz | 0.28 | ~0.25–0.50% hull |
| Bab-el-Mandeb | 0.35 | ~0.70% hull (2024) |
| Suez Canal | 0.18 | ~0.10–0.20% hull |
| Strait of Malacca | 0.05 | ~0.05% hull |
| Cape of Good Hope | 0.02 | ~0.01% hull |

The most important implication of this model is not the equation. It is what the equation implies about time. With θ = 0.3, a crisis spike takes roughly 10 to 15 simulation ticks to decay back toward baseline. That gap is your window to reroute. The question is whether your routing system is reactive — responding after the spike has already peaked — or adaptive, adjusting while the crisis is still developing.

That distinction is exactly what the next step is designed to address.

---

## Step 2: Risk-Aware Routing — Dijkstra with a Conscience

Now that risk is a live, evolving number on every edge, we need a routing algorithm that actually uses it.

The conventional objective — minimise cost — is equivalent to treating risk as zero. Our modified objective adds two terms:

```
min Σ [c(e) + α·t(e) + λ·r(e, t)]   for e in path(source, target)
```

Where α penalises transit time, λ penalises risk, and r(e, t) is the current risk on each edge from the OU process. The higher λ is, the more the algorithm is willing to pay in cost and time to avoid a dangerous route.

The algorithm that solves this is Dijkstra's, with one modification: a smarter definition of "cheap."

**How Dijkstra actually works:**

Think of it as the algorithm a cautious traveller uses instinctively. You are in an unfamiliar city, trying to reach the airport cheaply. You do not wander randomly. You look at every road leaving your current position and always move next toward whichever nearby point is cheapest to reach so far. You repeat this, never revisiting a point you have already settled, because by the time you settled it, you had already found the cheapest way there.

That guarantee — always expanding the cheapest reachable node next — is what makes Dijkstra both correct and efficient at O((V+E) log V).

In our version, the edge weight becomes:

```
w(e, t) = cost(e) + α·time(e) + λ·risk(e, t)
```

Everything else about Dijkstra is unchanged. Only the definition of "cheapest" has shifted — it now accounts for risk.

**The λ switchover**

Think of λ as a toll on risk. As it rises, three things happen in sequence:

- At **low λ**: risk barely registers in the weight function. Hormuz is the cheapest edge and the algorithm finds it immediately.
- At the **critical threshold λ***: the accumulated risk penalty on Hormuz-dependent edges exactly outweighs their cost advantage. The algorithm switches routes entirely.
- **Above λ***: bypass routes are systematically preferred. Hormuz becomes the path of last resort.

The toll road analogy holds here. Raise the toll gradually and most drivers stay on the highway. But at some precise price point, the back road becomes the rational choice. λ* is that price point for geopolitical risk — and crucially, it is not set by hand. It emerges from the current risk values in the graph.

Under crisis severity 0.90, calibrated to the Lloyd's March 2026 Gulf escalation scenario:

**Cost of resilience: +62% in cost, +8.2 days in transit. The switchover occurs at λ ≈ 15.2.**

Below that threshold, every rational shipper chooses Hormuz — not out of ignorance, but because the math tells them to. Above it, the bypass wins automatically. No human judgment required.

This reframes the policy conversation entirely: not "should we use Hormuz?" but "at what precise risk level do we stop, and what does it cost when we do?"

---

## Step 3: Monte Carlo — The Distribution of Futures

Dijkstra gives us the optimal path for any given risk level. But risk levels are not fixed. They vary. So the natural next question is: across the full range of possible crises, what does the system actually do?

We ran 500 independent disruption scenarios at randomly sampled crisis severities across [0.45, 0.95]. For each: a fresh graph, a Hormuz crisis at a random severity, risk-aware Dijkstra, and a record of whether the network rerouted and what it cost.

At severity above 0.75, the network reroutes in essentially every scenario — but pays 30 to 50% more every time. The tail risk is not rare. It is priced but unpaid.

When rerouting kicks in, the bottleneck capacity on the bypass corridor drops from 20 MBD at Hormuz to 7 MBD via the Yanbu pipeline — a 65% throughput reduction. The oil does not disappear. It simply cannot move as fast. For economies running on just-in-time inventory, that is a supply shock before a single barrel has even arrived late.

This is what Monte Carlo gives you that Dijkstra alone cannot: a distribution, not a point estimate. Planners do not need to know exactly what will happen. They need to know what the 95th percentile looks like, and whether their system can absorb it.

| Severity Range | Rerouting Rate | Cost Premium |
|---------------|---------------|-------------|
| 0.30–0.50 | ~0% | Negligible |
| 0.50–0.75 | Partial | +10–30% |
| 0.75+ | ~100% | +30–50% |

---

## Step 4: The Economic Cascade Model

The cost of a Hormuz closure is not the rerouting premium. That is just the opening act. A disruption propagates, stage by stage, through the entire global economy.

We modelled all of it, across five global regions, alongside a comparison study with historical crises.

**Stage 1: Oil Price Shock.** The 65% throughput reduction hits supply almost immediately. Oil prices spike, with magnitude depending on how quickly strategic reserves are deployed.

**Stage 2: Freight Premium Inflation.** War-risk insurance premiums reprice across all adjacent routes simultaneously. Ships rerouting via the Cape of Good Hope add 14 days and 30 to 50% in voyage costs.

**Stage 3: Consumer Price Pass-Through.** Energy costs cascade into petrol, plastics, fertilisers, and food. Food prices lead the curve by 15 to 30 days before GDP impact even peaks. This is the primary mechanism through which an oil shock becomes a living-standards crisis.

**Stage 4: Central Bank Response.** Supply-side inflation cannot be fixed by raising interest rates. Rates go up anyway, adding a second-order GDP drag 60 to 90 days post-closure. The medicine makes the patient sicker.

**Stage 5: Regional GDP Contraction.** The damage is geographically unequal.

| Region | Oil Import Dependency | Est. CPI Impact | Est. GDP Impact |
|--------|----------------------|-----------------|-----------------|
| East Asia (China, Japan, Korea) | ~85% | +4–6% | −2.5–3.5% |
| South Asia (India) | ~80% | +3–5% | −1.5–2.5% |
| Europe | ~60% | +2–4% | −1.0–2.0% |
| Middle East (non-Gulf) | ~40% | +1–3% | −0.5–1.5% |
| USA | ~15% | +0.5–1.5% | −0.3–0.8% |

East Asia — 85% import-dependent, with no meaningful bypass alternative — absorbs a CPI hit four times larger than the United States from the same disruption. Developing markets fare worse still: highest import dependency, weakest central bank credibility, and food import bills denominated in a currency that weakens as oil spikes. Europe sits in the middle, partially insulated by North Sea supply and Atlantic routing alternatives. The United States, producing roughly 13 million barrels a day domestically, barely registers in the macro numbers. One closure, five different crises.

We then calibrated the model against every major oil shock on record:

| Event | Duration | Oil Δ | CPI Peak | GDP Δ |
|-------|----------|-------|----------|-------|
| 1973 Arab Embargo | 150 days | +400% | +11.0% | −2.5% |
| 1979 Iranian Revolution | 365 days | +150% | +13.5% | −3.5% |
| 1990 Gulf War | 180 days | +100% | +6.2% | −1.5% |
| 2019 Abqaiq Attack | 14 days | +15% | +0.2% | −0.1% |
| 2022 Russia Sanctions | 365 days | +60% | +9.1% | −1.0% |
| 2023–24 Houthi/Red Sea | Ongoing | +8% | +0.3% | −0.3% |

The relationship between supply shock magnitude and consumer price pass-through is empirically stable across fifty years and seven crises. A 30-day Hormuz closure at moderate severity lands between Katrina and the Gulf War on that curve. Neither of those was contained quietly.

What makes this cascade dangerous is not any single stage. It is the feedback between them. Higher oil prices raise freight costs. Higher freight costs raise food prices. Higher food prices generate instability that prolongs the crisis. Prolonged crisis sustains elevated premiums.

The system feeds back on itself.

---

## The Structural Conclusion

Return for a moment to the opening image: 20% of global internet traffic routed through a single server, with no redundancy, no failover, and no backup plan.

The model makes that image precise. The graph quantifies exactly how central Hormuz is. The OU process captures how risk evolves and how much time exists to respond. Dijkstra identifies the precise threshold at which rerouting becomes rational. Monte Carlo maps the full distribution of outcomes. And the cascade model traces what happens to ordinary people, in ordinary economies, when the server finally goes down.

The system does not fail because we lack alternative routes. It fails because we over-commit to the best one. Redundancy beats efficiency. Optionality beats optimisation. The bypass exists, the rerouting is possible, and the cost is known. The market will not pay it until it has no choice — and by then the cost is not a freight surcharge. It is oil prices, food prices, inflation, and GDP contraction cascading across five regions at different speeds, in amounts the world did not budget for because it was too busy optimising.

---

## Next In V2

Currently we try to answer a precise question: given a graph, given current risk, what is the optimal route and what does it cost the world if that route breaks?

However, risk is inferred, not observed. It arrives as a headline, a premium repricing, a vessel deviating from its standard lane at 2am. By the time the number is obvious, the market has already moved. V2 changes where the risk comes from.

The risk model stops being a formula and becomes a trained neural network. A two-layer LSTM learns to predict next-step edge risk from structured signals:

- oil price volatility
- war-risk insurance premiums
- news sentiment

The same inputs a Lloyd's underwriter reads before pricing a Gulf transit.

Moreover, previously, once Dijkstra found the optimal path, that was the end of the decision. The algorithm recomputed from scratch every time the graph changed, had no memory of what worked before, and could not generalise across conditions it had not explicitly seen. Optimal for the moment. Blind to everything else.

In the next version, we are building a reinforcement learning agent. It starts with Q-learning: a cheat sheet the agent memorises during training. Every situation it has seen before, it handles well. Every situation it hasn't, it guesses. And in a crisis, it is almost always seeing something new. So we replace the cheat sheet with a Deep Q-Network — one that doesn't just memorise answers but learns the underlying pattern: so when it sees a risk level it has never encountered, it reasons its way to the right decision rather than guessing.

See you at V2!
