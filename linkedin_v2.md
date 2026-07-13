In 2025 we built a simulation of a Hormuz oil crisis. "Purely hypothetical," we said. "Fun graph theory project," we said.

In Feb 2026, Iran actually mined and closed the strait. So in 2026 — after the fact, no false modesty here — we sat down and built v2, with reality finally available to grade our homework.

What we got right: the bypass gap (called 65%, reality said 80% — close enough to be smug). The cascade sequence — oil → freight → CPI → food → GDP, exact order. Carriers abandoning Hormuz the second the risk-adjusted math said to, no committee meeting required.

What we completely missed: insurers wouldn't just raise prices, they'd leave outright. Governments would end up underwriting commercial shipping risk. And the "safe" backup pipeline we called the alternative got bombed a month in — turns out the escape route wasn't immune to the crisis it existed to escape.

So v2 isn't a victory lap, it's a correction: v1 could tell you what to do in a crisis, but only once the crisis had already become a number. v2 reads the leading signals — sentiment, insurance lag, futures vol — through an LSTM, and routes ~7 steps before the market finishes pricing it in.

Also retired our old Q-learning agent along the way. A lookup table covering 0.00003% of possible states was, technically, mostly guessing.

Model → reality's rebuttal → better model. Full writeup + the live app (break it yourself) in the comments. 🛢️
