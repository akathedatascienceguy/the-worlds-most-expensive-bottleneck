The Strait of Hormuz actually closed in February 2026. We built this after that happened, not before, no prediction here, just two people who watched a crisis unfold and asked: could a structural model of this network have told us anything useful, even one that had never seen this specific event?

Turns out: mostly yes, using general data instead of anything crisis-specific. Our bypass-gap estimate (65%) landed close to reality (80%). The cascade sequence, oil, freight, CPI, food, GDP, unfolded in the order we modeled. Carriers abandoned Hormuz the moment the risk-adjusted math said to, no committee meeting required.

What the model missed, because nobody builds this stuff for free: insurers didn't just raise prices, they left outright. Governments ended up underwriting commercial shipping risk. And the "safe" backup pipeline we called the alternative got bombed a month into the crisis, turns out the escape route wasn't immune to the crisis it existed to escape.

So v2 isn't a sequel, it's a correction: v1 could tell you what a crisis costs, but it had no way to read the leading signals, sentiment, insurance lag, futures vol, before they became a lagging market number. v2 does, via an LSTM feeding a routing agent that reroutes ~7 steps ahead of the market.

Also retired our old Q-learning agent along the way. A lookup table covering 0.00003% of possible states was, technically, mostly guessing.

The strait will close again. We'd rather not be the last ones to notice. Full writeup + the live app (break it yourself) in the comments. 🛢️
