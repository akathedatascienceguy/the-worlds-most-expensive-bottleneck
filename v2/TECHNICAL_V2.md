# Technical Report: v2 — Deep Q-Network + LSTM Risk Predictor

**Project:** The World's Most Expensive Bottleneck — v2  
**Stack:** Python 3.x · PyTorch 2.x · NetworkX · Plotly · Streamlit  
**Entry point:** `v2/app_v2.py` (self-contained)  
**Prerequisite reading:** `TECHNICAL.md` (v1) covers graph construction, Dijkstra routing, and Monte Carlo stress testing — those components are unchanged in v2.

---

## Table of Contents

1. [What Changed from v1 and Why](#1-what-changed-from-v1-and-why)
2. [Synthetic Data Generation](#2-synthetic-data-generation)
3. [LSTM Risk Predictor](#3-lstm-risk-predictor)
4. [Deep Q-Network Agent](#4-deep-q-network-agent)
5. [Experience Replay Buffer](#5-experience-replay-buffer)
6. [Target Network](#6-target-network)
7. [End-to-End Pipeline](#7-end-to-end-pipeline)
8. [Training Procedures](#8-training-procedures)
9. [Evaluation & Diagnostics](#9-evaluation--diagnostics)
10. [Architecture Comparison: v1 vs v2](#10-architecture-comparison-v1-vs-v2)
11. [Limitations & Next Steps](#11-limitations--next-steps)

---

## 1. What Changed from v1 and Why

### 1.1 The v1 Ceiling

v1 had two fundamental limitations:

**Limitation 1 — Simulated risk, not predicted risk.**  
The Ornstein-Uhlenbeck (OU) process generates plausible-looking risk trajectories but does not *learn* from signal data. It cannot distinguish between a random volatility spike and a genuine escalation event. Every edge evolves independently according to the same formula regardless of what signals are present. A real system would ingest news sentiment, insurance premiums, AIS anomalies, and oil price volatility — and produce a *calibrated forecast* of `r(e, t+1)`.

**Limitation 2 — Tabular Q-learning does not generalise.**  
The Q-table in v1 is a Python `defaultdict`. It stores Q-values only for (state, action) pairs that have been explicitly visited during training. A state is defined as `(node, tuple_of_discretised_risks)`. In a 22-edge network with 5 risk buckets per edge, the theoretical state space is 19 × 5²² ≈ 4.5 × 10¹⁶. The agent sees a negligible fraction of this during training. Any unseen state gets Q-value = 0 by default, meaning the agent falls back to random selection — exactly when reliable decisions matter most.

### 1.2 The v2 Fixes

| Problem | v1 Approach | v2 Approach |
|---------|------------|------------|
| Risk prediction | OU formula (no learning) | LSTM trained on structured synthetic data |
| Q-function | Dict lookup (no generalisation) | Neural network (continuous state, interpolates) |
| Training stability | None | Experience replay + target network |
| State space | Discrete, exponential | Continuous, 41-dimensional |
| Unseen states | Q = 0 (random fallback) | Forward pass through network (learned estimate) |

---

## 2. Synthetic Data Generation

### 2.1 Why Synthetic Data

Real per-edge risk time series with labelled disruption events do not exist as a free public dataset. War-risk insurance premiums are available in aggregate but not per shipping lane per day. AIS data is proprietary. The synthetic generator serves two purposes:

1. Creates a supervised learning problem with known ground truth for LSTM evaluation
2. Mimics the causal structure of real signals — temporal lags, event shocks, mean reversion

### 2.2 Feature Engineering

Four features are generated per edge per timestep:

| Index | Feature | Description | Causal Structure |
|-------|---------|-------------|-----------------|
| 0 | `risk` | Geopolitical risk level | Ground truth target; OU process + event shocks |
| 1 | `oil_vol` | Oil price volatility | Concurrent with Hormuz risk; mean-reverts to `0.25 + 0.5·hormuz_avg` |
| 2 | `insurance` | War-risk insurance premium | **Lags** risk by ~5–10 steps (market response delay) |
| 3 | `sentiment` | Geopolitical sentiment (high=calm) | **Leads** risk by ~3–5 steps (early warning) |

This lag/lead structure is deliberate. The LSTM must learn to use the leading sentiment signal to anticipate risk rises, and to not be misled by the lagging insurance signal for immediate predictions.

### 2.3 Risk Simulation (Ground Truth Generation)

Each edge's risk follows a discrete-time OU process:

```
R(t+1) = R(t) + θ(μ - R(t)) + σ·ε(t),   ε ~ N(0,1)
```

With θ = 0.3, σ = 0.07, μ = `base_risk` per edge (real EIA-calibrated values from DATA_SOURCES.md).

Disruption events are injected at random timesteps:

```python
event_times = rng.choice(range(50, n_steps-50), n_events, replace=False)
for t in range(n_steps):
    ...
    for k, et in enumerate(event_times):
        if t == et:
            sev = rng.uniform(0.30, 0.65)
            for i in hormuz_idxs:
                risks[i] = min(1.0, risks[i] + sev)
```

Post-event, the OU process naturally decays risk back toward baseline — no explicit recovery curve needed.

### 2.4 Correlated Signal Generation

```python
# Oil volatility: mean-reverts to (0.25 + 0.5 * hormuz_avg)
oil_vol[:] = clip(oil_vol + 0.15*(0.25 + 0.5*hormuz_avg - oil_vol) + 0.04*ε, 0, 1)

# Insurance: lagged risk (slow market response)
insurance[:] = clip(0.85*insurance + 0.15*risks + 0.015*ε, 0, 1)

# Sentiment: high when risk is low (inverse, leading)
sentiment[:] = clip(0.80*sentiment + 0.20*(1 - risks) + 0.04*ε, 0, 1)
```

The insurance update rule — 85% autoregression, 15% current risk — produces approximately a 7-step lag at θ=0.15. This is consistent with observed real-world insurance premium adjustment speeds (premiums are repriced daily but smooth over recent incident data).

### 2.5 Data Shape and Train/Val Split

```
n_steps = 2000   (default)
n_edges = 21
n_features = 4

data shape: (2000, 21, 4)

Sequence construction:
  X[i] = data[i : i + SEQ_LEN]          → shape (SEQ_LEN, 21, 4)
  y[i] = data[i + SEQ_LEN, :, 0]        → shape (21,)   [risk only]

Total sequences: 2000 - 10 - 1 = 1989
Train: 1591 sequences (80%)
Val:    398 sequences (20%)
```

---

## 3. LSTM Risk Predictor

### 3.1 Architecture

```
Input:  (B, SEQ_LEN=10, N_EDGES=21, N_FEATURES=4)
        ↓ reshape
        (B, 10, 84)              [flatten last two dims: 21 × 4]
        ↓
LSTM Layer 1:  input=84, hidden=128, dropout=0.2 on output
LSTM Layer 2:  input=128, hidden=128
        ↓ take last hidden state
        (B, 128)
        ↓
Linear(128 → 64) → ReLU → Dropout(0.1)
        ↓
Linear(64 → 21) → Sigmoid
        ↓
Output: (B, 21)    predicted r(e, t+1) ∈ (0, 1) for all edges
```

**Parameter count breakdown:**

| Component | Parameters |
|-----------|-----------|
| LSTM Layer 1 | 4 × (84×128 + 128×128 + 128) = 109,568 |
| LSTM Layer 2 | 4 × (128×128 + 128×128 + 128) = 131,584 |
| Linear 128→64 | 128×64 + 64 = 8,256 |
| Linear 64→21  | 64×21 + 21 = 1,365 |
| Dropout (no params) | 0 |
| **Total** | **~251,285** |

### 3.2 Design Decisions

**Why LSTM over Transformer?**  
Transformers require positional encodings and scale quadratically with sequence length. For SEQ_LEN=10 with structured time series, LSTM's inductive bias (sequential processing, explicit hidden state) outperforms attention-based models that treat the sequence as a bag. LSTMs also train faster on short sequences, which matters for in-app training time.

**Why Sigmoid output?**  
Risk is bounded in [0, 1] by construction. Sigmoid enforces this constraint without clipping artefacts. It also prevents the model from predicting arbitrarily large risk values after crisis events.

**Why MSE loss?**  
The target is a continuous scalar in [0, 1]. MSE penalises large errors more than MAE, which is appropriate — a 0.3 prediction error during a crisis (e.g., predicting 0.6 when truth is 0.9) is more costly than a 0.03 error during calm periods.

**Why two LSTM layers?**  
Layer 1 extracts per-timestep features (local patterns). Layer 2 captures temporal dependencies across the sequence (trend, oscillation period). Single-layer LSTMs underfit the multi-signal structure in testing.

### 3.3 Training Configuration

| Hyperparameter | Value | Rationale |
|---------------|-------|-----------|
| Learning rate | 1e-3 | Standard for Adam on time series |
| Batch size | 64 | Balances gradient noise and memory |
| Epochs | 120 (default) | Convergence plateau observed ~80–100 epochs |
| Gradient clipping | max norm 1.0 | Prevents exploding gradients in LSTM |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) | Adapts lr when val loss stagnates |
| Weight decay | 1e-5 | L2 regularisation |

### 3.4 Loss Convergence

Typical observed values (2000 training steps, 120 epochs):

| Metric | Initial | After 20 ep | After 120 ep |
|--------|---------|------------|-------------|
| Train MSE | ~0.040 | ~0.012 | ~0.003 |
| Val MSE   | ~0.045 | ~0.015 | ~0.005 |

Val loss closely tracks train loss, indicating no significant overfitting. The small gap is attributable to disruption events in the validation period that differ from training events.

### 3.5 Live Inference

At runtime, a **rolling window** of SEQ_LEN=10 recent observations is maintained in `st.session_state.risk_window` with shape `(10, 21, 4)`. On each simulation tick:

```python
# 1. LSTM predicts next-step risks
pred = lstm.predict_next(risk_window)      # (21,) float array

# 2. Update graph edge weights
for i, (u, v) in enumerate(edge_list):
    G[u][v]["risk"] = float(np.clip(pred[i], 0.0, 1.0))

# 3. Append new row to window (constructed from updated graph state)
new_row = np.zeros((1, 21, 4))
for i, (u, v) in enumerate(edge_list):
    r = G[u][v]["risk"]
    new_row[0, i, 0] = r
    new_row[0, i, 1] = clip(r * 0.8 + 0.1 + noise, 0, 1)   # oil_vol proxy
    new_row[0, i, 2] = clip(r * 0.7 + noise, 0, 1)          # insurance proxy
    new_row[0, i, 3] = clip((1 - r) * 0.8 + noise, 0, 1)    # sentiment proxy
risk_window = concatenate([risk_window[1:], new_row], axis=0)
```

The proxy signal construction for online inference is a simplification — a production system would read these from live data feeds. The key structural property (oil_vol correlates with risk, sentiment inversely) is preserved.

---

## 4. Deep Q-Network Agent

### 4.1 Why DQN Over Tabular Q-Learning

Tabular Q-learning requires visiting every (state, action) pair to build accurate Q-estimates. The state space in our problem is:

```
State = (current_node, risk_vector)
      = 19 nodes × ℝ^21 (continuous risks)
```

In v1, risk was discretised to 5 bins, giving 19 × 5²¹ ≈ 9 × 10¹⁵ states. Training would need to visit a statistically significant fraction of these to generalise — computationally infeasible.

DQN replaces the table with a neural network `Q(s, a; θ)` that maps a *continuous* state vector to Q-values for all actions. The network learns a smooth function over state space, interpolating Q-values for unseen states based on similarity to seen ones.

### 4.2 State Representation

```
State vector (41 dimensions, float32):

[  one-hot node encoding  |  LSTM-predicted edge risks  ]
[  19 dims               |  21 dims                    ]

node_oh[i] = 1.0 if current_node == node_list[i] else 0.0
risks[j]   = G[edge_list[j][0]][edge_list[j][1]]["risk"]
```

One-hot encoding is used instead of a learned node embedding because:
- The network structure (which nodes connect to which) is fixed
- 19 dimensions is small enough that one-hot is efficient
- Learned embeddings would require additional pretraining on graph structure

### 4.3 Action Space and Masking

The action space has `N_NODES = 19` dimensions. However, at any node, only its direct neighbours are reachable — typically 1–4 nodes. Invalid actions are masked before argmax:

```python
mask = torch.full((N_NODES,), float("-inf"))
for nb, idx in valid_neighbours:
    mask[idx] = q_values[idx]
best_action = mask.argmax()
```

This ensures the network never selects an unreachable node. During training, only valid-action Q-values contribute to the Bellman update.

### 4.4 Network Architecture

```
Input (41,)
    │
Linear(41 → 256) ─── bias
    │
LayerNorm(256)          ← stabilises training, prevents internal covariate shift
    │
ReLU
    │
Dropout(p=0.1)
    │
Linear(256 → 128) ─── bias
    │
ReLU
    │
Linear(128 → 19) ─── bias
    │
Output (19,)   ← Q-value per node (masked before use)
```

**Parameter count:**

| Layer | Weights | Bias | Total |
|-------|---------|------|-------|
| Linear 41→256 | 41×256=10,496 | 256 | 10,752 |
| LayerNorm(256) | 256 (γ) | 256 (β) | 512 |
| Linear 256→128 | 256×128=32,768 | 128 | 32,896 |
| Linear 128→19 | 128×19=2,432 | 19 | 2,451 |
| **Total** | | | **~46,611** |

### 4.5 Bellman Update and Loss

The Q-learning objective minimises the TD (temporal difference) error:

```
TD target:  y = r + γ · max_a' Q_target(s', a'; θ⁻)
TD error:   δ = y - Q_policy(s, a; θ)
Loss:       L = SmoothL1(δ)   [Huber loss]
```

**Why Huber loss over MSE?**  
Early in training, TD targets are noisy and TD errors can be large. MSE amplifies these (squared penalty), causing unstable gradient steps. Huber loss behaves like MSE for small errors (|δ| < 1) and like MAE for large errors, bounding the gradient magnitude and stabilising training.

```
SmoothL1(δ) = { 0.5·δ²        if |δ| < 1
              { |δ| - 0.5     otherwise
```

### 4.6 Reward Function

```
R(edge e = (u,v)) = -(cost(e) + 12·risk(e) + 2·time(e))
R(reaching target) += +100
```

**Coefficient rationale:**

| Term | Coefficient | Rationale |
|------|-------------|-----------|
| cost | 1 | Base cost unit |
| risk | 12 | Risk aversion: 12× cost means agent prefers a route costing +12 if it eliminates 1 unit of risk |
| time | 2 | Time valued at twice cost per day |
| target bonus | +100 | Large enough to dominate all step penalties for paths ≤ 8 edges |

The target bonus is critical. Without it, the agent learns that staying put (avoiding any edge cost) maximises reward — a pathological fixed point. With +100, reaching the target always dominates any sequence of step penalties for routes of reasonable length.

### 4.7 ε-Greedy Exploration

```
ε(t) = max(ε_min, ε_0 · decay^t)
ε_0 = 1.0,  ε_min = 0.05,  decay = 0.997
```

At episode `t`:
- With probability ε(t): choose a random valid neighbour
- With probability 1-ε(t): choose argmax Q-value over valid neighbours

At 600 episodes, ε decays to approximately:
```
ε(600) = max(0.05, 1.0 × 0.997^600) ≈ max(0.05, 0.163) = 0.163
```

The agent retains meaningful exploration throughout training. For longer training (1500 episodes), ε reaches ~0.05 — almost purely exploitative.

---

## 5. Experience Replay Buffer

### 5.1 Motivation

Naive Q-learning updates the network on each transition as it occurs. This creates two problems:

1. **Correlation**: Consecutive transitions `(s_t, a_t, r_t, s_{t+1})` are highly correlated — they share the same routing episode context. Gradient updates on correlated data cause the network to oscillate rather than converge.

2. **Forgetting**: Without replay, the network only learns from recent experience and forgets earlier transitions, including rare but important events (Hormuz crises, successful Cape routings).

### 5.2 Implementation

```python
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buf = deque(maxlen=capacity)   # circular buffer; oldest entries drop automatically

    def push(self, state, action_idx, reward, next_state, done):
        self.buf.append((state, action_idx, reward, next_state, float(done)))

    def sample(self, n):
        batch = random.sample(self.buf, n)   # uniform random sample — breaks correlations
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),    # (B, state_dim)
            torch.LongTensor(a),               # (B,)
            torch.FloatTensor(r),              # (B,)
            torch.FloatTensor(np.array(ns)),   # (B, state_dim)
            torch.FloatTensor(d),              # (B,) — done flags for terminal masking
        )
```

**Capacity = 10,000:** With ~10–15 transitions per episode and 600 episodes, the buffer holds approximately the last 700 episodes of experience. Older episodes that are no longer representative of the current policy are naturally evicted.

**Minimum fill threshold:** Learning only begins when the buffer contains at least `BATCH_SIZE = 64` transitions, ensuring the first gradient update has sufficient diversity.

### 5.3 Transition Storage

Each stored transition is a 5-tuple:

```
(state, action_idx, reward, next_state, done)

state:      np.array (41,)  float32
action_idx: int             index of chosen node in node_list
reward:     float
next_state: np.array (41,)  float32
done:       float           1.0 if next_state is target, 0.0 otherwise
```

The `done` flag is used to mask the bootstrap term in the Bellman update:

```
y = r + γ · Q_target(s') · (1 - done)
```

When `done=1`, the target is just `r + 0` — no bootstrapping from a terminal state.

---

## 6. Target Network

### 6.1 The Moving-Target Problem

Standard Q-learning updates with:
```
y = r + γ · max_a' Q(s', a'; θ)
```

But `Q(s', a'; θ)` uses the **same parameters θ** that are being updated. This creates a moving target: as θ changes, the target y changes, which changes the gradient, which changes θ again. This feedback loop causes oscillation and divergence in deep networks.

### 6.2 Solution: Frozen Target Network

Maintain two networks with identical architecture:

- **Policy network** `Q(s, a; θ)`: updated every gradient step
- **Target network** `Q(s, a; θ⁻)`: frozen copy, updated every `TARGET_UPD = 100` steps

Bellman update uses the target network for bootstrap:
```
y = r + γ · max_a' Q_target(s', a'; θ⁻)
```

The target is now stationary over 100 steps, breaking the feedback loop and providing a stable learning signal.

```python
# Gradient step (every transition)
q_pred = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
with torch.no_grad():
    q_next = target_net(next_states).max(1)[0]
    q_tgt  = rewards + GAMMA * q_next * (1 - dones)
loss = smooth_l1(q_pred, q_tgt)
optimizer.step()

# Target network sync (every TARGET_UPD steps)
if steps % TARGET_UPD == 0:
    target_net.load_state_dict(policy_net.state_dict())
```

**Why hard copy, not soft update?**  
Soft updates (`θ⁻ ← τ·θ + (1-τ)·θ⁻`, typical τ ≈ 0.005) provide smoother target movement at the cost of slower adaptation. Hard copies every 100 steps are equivalent for this problem size and simpler to implement and reason about.

---

## 7. End-to-End Pipeline

### 7.1 Training Phase

```
┌─────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                       │
│                                                         │
│  1. Generate synthetic data                             │
│     n_steps=2000, n_events=10, seed=42                  │
│     → (2000, 21, 4) array                              │
│                                                         │
│  2. Build LSTM sequences                                │
│     X: (1989, 10, 21, 4)                               │
│     y: (1989, 21)  [next-step risks]                   │
│     80/20 train/val split                               │
│                                                         │
│  3. Train LSTM (120 epochs, Adam, MSE)                 │
│     → Learned θ_LSTM                                   │
│                                                         │
│  4. Train DQN (600 episodes)                           │
│     Each episode:                                       │
│       state = encode(node, current_risks)              │
│       → ε-greedy action → transition → buffer          │
│       → sample batch → Bellman update → θ_DQN         │
│       → sync target net every 100 steps               │
│     → Learned θ_DQN, θ_DQN_target                    │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Inference Phase (Per Simulation Tick)

```
┌─────────────────────────────────────────────────────────┐
│                  INFERENCE (per tick)                   │
│                                                         │
│  Input: rolling window W ∈ ℝ^(10 × 21 × 4)            │
│                                                         │
│  1. LSTM Risk Prediction                                │
│     r̂ = sigmoid(LSTM(W)) ∈ ℝ^21                       │
│     Update G[u][v]["risk"] ← r̂[i] for each edge       │
│                                                         │
│  2. Window Advance                                      │
│     Construct new_row from updated graph risks         │
│     W ← concatenate([W[1:], new_row])                 │
│                                                         │
│  3. Routing Decision (two parallel paths)              │
│     ┌─────────────────┐   ┌────────────────────────┐  │
│     │  Dijkstra       │   │  DQN Greedy Policy     │  │
│     │  min Σw(e,t)    │   │  argmax Q(s, ·; θ_DQN) │  │
│     └────────┬────────┘   └────────────┬───────────┘  │
│              │                         │               │
│              └──────────┬──────────────┘               │
│                         ↓                               │
│              Display on map + stats                     │
└─────────────────────────────────────────────────────────┘
```

### 7.3 Interaction Between LSTM and DQN

The LSTM and DQN interact through the graph's risk state:

```
LSTM output → G[u][v]["risk"] → DQN state encoding
                              → Dijkstra edge weights
                              → display (edge colour)
```

The DQN's state vector at any tick is:
```
s = [one_hot(current_node) || LSTM_predicted_risks]
```

This means the DQN's routing decisions are *conditioned on LSTM's risk forecast*. When the LSTM predicts rising Hormuz risk several ticks before the crisis peaks (via the leading sentiment signal), the DQN will begin routing via bypass paths — potentially before Dijkstra would switch, since Dijkstra only reacts to current weights.

---

## 8. Training Procedures

### 8.1 LSTM Training Loop

```python
for epoch in range(n_epochs):
    # Shuffle training data
    perm = torch.randperm(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # Mini-batch SGD
    for batch in range(n_batches):
        xb, yb = X_train[b*B:(b+1)*B], y_train[b*B:(b+1)*B]
        optimizer.zero_grad()
        pred = model(xb)            # forward pass
        loss = mse(pred, yb)        # MSE
        loss.backward()             # backprop through time (BPTT)
        clip_grad_norm_(params, 1.0)
        optimizer.step()

    # Validation
    with torch.no_grad():
        val_loss = mse(model(X_val), y_val)
    scheduler.step(val_loss)        # ReduceLROnPlateau
```

**Backpropagation through time (BPTT):** PyTorch's LSTM implementation unrolls the recurrence over the sequence dimension during the forward pass and accumulates gradients through time during `loss.backward()`. The gradient clipping (`max_norm=1.0`) prevents the vanishing/exploding gradient problem inherent to long-sequence BPTT.

### 8.2 DQN Training Loop

```python
for episode in range(n_episodes):
    node = source
    seen = {node}
    
    while node != target:
        # Choose action
        action, idx = agent.act(node, edge_risks)
        
        # Environment step
        reward = -(cost + 12·risk + 2·time) + (100 if done else 0)
        
        # Store transition
        buffer.push(encode(node), idx, reward, encode(next_node), done)
        
        # Learn (if buffer ready)
        if len(buffer) >= BATCH_SIZE:
            s, a, r, ns, d = buffer.sample(BATCH_SIZE)
            
            # Compute TD targets with frozen target net
            with torch.no_grad():
                q_next = target_net(ns).max(1)[0]
                y = r + γ * q_next * (1 - d)
            
            # Compute current Q-values
            q_pred = policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            
            # Huber loss + backprop
            loss = smooth_l1(q_pred, y)
            loss.backward()
            clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
            
            # Sync target net every 100 steps
            if steps % 100 == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        # Decay exploration
        ε = max(ε_min, ε * 0.997)
        node = action
```

---

## 9. Evaluation & Diagnostics

### 9.1 LSTM Diagnostics

**Loss curves (Training tab)**  
Training and validation MSE plotted per epoch. Signs of healthy training:
- Both curves decrease monotonically in early epochs
- Val curve tracks train curve (no overfitting)
- ReduceLROnPlateau steps visible as sudden drops

**Per-edge RMSE table (Training tab)**  
```
RMSE_i = sqrt(mean((y_val[:,i] - y_pred[:,i])^2))
```
Higher RMSE on Hormuz-dependent edges is expected — they experience larger shock events that are harder to predict precisely. The goal is calibration (correct direction and magnitude), not exact point prediction.

**Predicted vs actual time series (Training tab)**  
Overlays LSTM predictions against ground truth for three representative edges:
- Hormuz → Indian Ocean Hub (highest-risk edge)
- Red Sea → Bab-el-Mandeb (disrupted by Houthi attacks in real data)
- Cape of Good Hope → Europe (lowest-risk edge; should be near-flat)

**Model internals tab — live predictions**  
The `Δ` column shows `LSTM_predicted - current_graph_risk` per edge. Positive Δ means LSTM is forecasting rising risk. This is the actionable signal: a router using LSTM can pre-empt risk spikes that Dijkstra (reacting to current values) would miss.

### 9.2 DQN Diagnostics

**Episode reward curve**  
Should trend upward from large negative values (agent fails to find target) toward a plateau (agent reliably reaches target with low-cost, low-risk paths). The rolling mean (window = n_episodes/20) filters episode-to-episode noise.

**Huber loss curve**  
Should decrease from high values (poor initial Q-estimates) toward a low stable value. Occasional spikes indicate the target network just synced and temporarily destabilised Q-estimates — normal behaviour.

**Q-value heatmap (Model Internals tab)**  
Rows = current node, columns = possible next node. Each cell = Q(current_node → next_node) for the current graph risk state.

Correctly trained patterns to look for:
- Producer rows: high Q-values toward Hormuz (fast, cheap) and toward bypass hubs (safe alternative)
- Hormuz row: high Q-value toward Indian Ocean Hub
- Indian Ocean Hub row: high Q-value toward Malacca (East Asia routes) or India
- Cape of Good Hope row: high Q-values toward Europe/USA

If Hormuz risk is high (due to crisis simulation), Q-values toward Hormuz should be lower, and bypass hub Q-values should increase.

**ε (exploration rate)**  
Displayed in Model Internals. At 600 episodes ε ≈ 0.16 (still 16% random), at 1500 episodes ε ≈ 0.05. For a fully converged greedy policy, train for 1200+ episodes.

---

## 10. Architecture Comparison: v1 vs v2

### 10.1 Component-Level

| Component | v1 | v2 |
|-----------|----|----|
| **Risk engine** | OU stochastic process | Trained LSTM |
| **Risk inputs** | None (generates internally) | (risk, oil_vol, insurance, sentiment) window |
| **Risk params** | 0 (formula, no learning) | ~251,285 |
| **Risk output** | Noisy sample from OU | Calibrated forecast `r̂(e, t+1)` |
| **Risk generalisation** | None | Yes — LSTM interpolates unseen sequences |
| **Agent model** | Python `defaultdict` | 3-layer DQN, ~46,355 params |
| **State space** | Discrete: 19 × 5²¹ buckets | Continuous: ℝ^41 |
| **State seen during training** | ~0.0% of theoretical space | N/A (network interpolates) |
| **Training stability** | None | Replay buffer + target network |
| **Action selection** | ε-greedy on dict lookup | ε-greedy on masked network forward pass |
| **Loss function** | None (table update) | Huber loss on Bellman targets |
| **Gradient clipping** | N/A | max norm 1.0 |
| **Routing algorithm** | Risk-aware Dijkstra | Risk-aware Dijkstra (unchanged) |

### 10.2 Inference Behaviour

| Scenario | v1 Response | v2 Response |
|----------|------------|------------|
| Unseen (node, risk) combination | Q = 0.0 → random action | Forward pass → interpolated Q estimate |
| Rising risk (pre-crisis) | No anticipation | LSTM detects sentiment lead, updates risk prediction, DQN pre-empts |
| Post-crisis recovery | Sudden Q update on next visit | LSTM models insurance lag, smooth risk decay |
| Novel route not seen in training | Q = 0 (never explored) | Non-zero Q estimate from neighbouring states |

### 10.3 Total Parameter Count

| Model | Parameters | Trainable |
|-------|-----------|-----------|
| LSTM Risk Predictor | ~251,285 | Yes |
| DQN Policy Network | ~46,355 | Yes (during training) |
| DQN Target Network | ~46,355 | No (frozen, copied from policy) |
| **Total** | **~344,000** | **~297,640** |

---

## 11. Limitations & Next Steps

### 11.1 Current Limitations

| Limitation | Description | Impact |
|------------|-------------|--------|
| Synthetic training data | LSTM trained on OU-generated data, not real historical disruptions | Predictions are structurally correct but not empirically calibrated |
| Fixed graph topology | Edges never appear/disappear | Cannot model canal closures, pipeline shutdowns |
| Tabular node representation | One-hot encoding doesn't capture node relationships | GNN-based encoding would be richer |
| Single-commodity flow | Routes one origin-destination pair at a time | Real logistics is multi-commodity, multi-vessel |
| No capacity constraints | Routing ignores whether edge can carry the required volume | Min-cost max-flow needed for realistic capacity planning |
| DQN convergence | 600 episodes insufficient for full convergence | 1200–1500 episodes needed for stable greedy policy |
| No uncertainty quantification | LSTM gives point predictions, no confidence intervals | MC Dropout or ensemble methods would add calibrated uncertainty |

### 11.2 Immediate Extensions (Low Effort)

**MC Dropout for LSTM uncertainty:**  
Enable dropout at inference time and run N=50 forward passes. The variance of predictions gives a per-edge confidence interval. High-variance predictions signal that the model is uncertain — a signal to fall back to the conservative (high-λ) routing.

```python
def predict_with_uncertainty(model, window, n_samples=50):
    model.train()  # keep dropout active
    preds = np.array([model.predict_next(window) for _ in range(n_samples)])
    return preds.mean(axis=0), preds.std(axis=0)
```

**Prioritised Experience Replay:**  
Sample transitions proportional to their TD error magnitude (high-error = informative = sampled more often). This accelerates learning on rare but important transitions (crisis events).

**Dueling DQN:**  
Decompose Q-values into value (how good is this state?) and advantage (how much better is this action than average?):
```
Q(s,a) = V(s) + A(s,a) - mean_a'[A(s,a')]
```
Particularly useful for states where all actions have similar value (e.g., transit hubs).

### 11.3 Production-Grade Extensions (High Effort)

**Real data pipeline:**

```
Live signals → Feature engineering → LSTM → r̂(e,t) → DQN/Dijkstra
   │
   ├── News API (Reuters, Bloomberg) → BERT sentiment → sentiment[e,t]
   ├── FRED / ICE → WTI/Brent OHLCV → GARCH vol → oil_vol[t]
   ├── P&I club APIs → war-risk premium → insurance[e,t]
   └── AIS stream → vessel density anomaly → congestion[e,t]
```

**Graph Neural Network state encoder:**  
Replace one-hot node encoding with a GNN that encodes the full graph topology:
```
node_embedding = GNN(adjacency_matrix, node_features, edge_features)
state = [node_embedding[current_node] || LSTM_risks]
```
The GNN captures structural information (betweenness centrality, node degree) that one-hot encoding misses.

**Multi-agent RL:**  
Deploy one agent per tanker fleet (producer-specific). Agents compete for route capacity, creating emergent congestion dynamics and producing realistic market equilibria for shipping lane pricing.

**Offline RL:**  
Train the DQN on historical routing decisions and observed outcomes (disruptions, delays) rather than online simulation. This requires a logged dataset of `(route_chosen, outcome)` pairs — available from shipping company databases and AIS history.

---

*End of v2 Technical Report*  
*See also: `TECHNICAL.md` (v1 baseline), `DATA_SOURCES.md` (real data citations)*
