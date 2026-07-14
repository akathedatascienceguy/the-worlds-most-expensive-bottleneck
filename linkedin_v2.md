We're back. 🛢️

Nikita Gupta and I just shipped v2 of The World's Most Expensive Bottleneck, published under Culture X Data Science.

Quick context: the Strait of Hormuz moves about 20% of the world's seaborne oil through a channel 33km wide, no backup, no redundancy. In February 2026, it actually closed. We'd modeled exactly this scenario, so we went back, graded our own homework, and rebuilt the engine underneath it.

v1 could tell you what a Hormuz crisis costs. It had no way to tell you one was coming.

v2 fixes that with two upgrades:

→ An LSTM that reads the signals the market reads too late: sentiment, insurance premiums, oil volatility, and forecasts rising risk before it becomes a price. Underwriters do this by instinct. We taught a neural net to do it from data.

→ A DQN that replaces our old Q-learning agent, which quietly broke the moment risk stopped fitting into a lookup table (60 trillion possible states will do that to you). The DQN doesn't memorize a table, it generalizes, so it can reason through a risk combination it's never seen before.

Chain them together and the system reroutes a tanker away from Hormuz about 7 steps before the market finishes pricing in the danger. For a vessel carrying $100M of crude, that's the difference between a course correction in open water and a decision made mid-strait.

Go break it yourself:
Live app (v2): https://the-worlds-most-expensive-bottleneck-v2.streamlit.app
Full writeup: https://github.com/akathedatascienceguy/the-worlds-most-expensive-bottleneck/blob/main/blog_v2.md
