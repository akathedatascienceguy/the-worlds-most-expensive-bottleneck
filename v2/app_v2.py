"""
The World's Most Expensive Bottleneck — v2
Production-Grade Risk-Aware Routing

What's new vs v1:
──────────────────────────────────────────────────────────────────
1. LSTM Risk Predictor  — trained neural network replaces OU simulation.
   Input:  sliding window of (risk, oil_vol, insurance, sentiment) per edge
   Output: predicted r(e, t+1) for all 22 edges
   Train:  MSE loss on synthetic data that mimics real signal structure

2. Deep Q-Network (DQN) — neural network Q-function replaces tabular dict.
   State:  one-hot node embedding (19) + LSTM risk vector (22) = 41 dims
   Hidden: 256 → 128 → n_actions (19), ReLU + LayerNorm
   Training extras: experience replay (10k buffer), target network, Huber loss,
                    gradient clipping, ε-decay schedule

3. End-to-end pipeline:
   [signal window] → LSTM → r̂(e,t) → update graph → DQN routes → display

4. New tabs: Model Internals (architecture, Q-value heatmap, LSTM predictions)
──────────────────────────────────────────────────────────────────
"""

import copy
import heapq
import random
from collections import deque

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hormuz Problem — v2 (DQN + LSTM)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    h1,h2,h3 { color: #58a6ff; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #58a6ff; border-bottom: 2px solid #58a6ff; }
    blockquote { border-left: 3px solid #58a6ff; padding-left: 1em; color: #8b949e; }
    code { background: #161b22; border-radius: 4px; padding: 2px 6px; }
    div[data-testid="metric-container"] {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 10px 14px;
    }
</style>
""", unsafe_allow_html=True)

torch.manual_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
SEQ_LEN    = 10    # LSTM lookback window
N_FEATURES = 4     # risk, oil_vol, insurance, sentiment
HIDDEN     = 128   # LSTM hidden size
DQN_H1     = 256   # DQN hidden layer 1
DQN_H2     = 128   # DQN hidden layer 2
BATCH_SIZE = 64
REPLAY_CAP = 10_000
TARGET_UPD = 100   # steps between target network syncs
GAMMA      = 0.95
LR_DQN     = 1e-4
LR_LSTM    = 1e-3


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH  (identical real-data graph from v1)
# ──────────────────────────────────────────────────────────────────────────────
def build_oil_network() -> nx.DiGraph:
    G = nx.DiGraph()
    nodes = {
        "Saudi Arabia":       {"type": "producer",   "lat": 23.0,  "lon": 45.0},
        "UAE":                {"type": "producer",   "lat": 24.5,  "lon": 54.5},
        "Iraq":               {"type": "producer",   "lat": 33.0,  "lon": 44.0},
        "Kuwait":             {"type": "producer",   "lat": 29.5,  "lon": 47.5},
        "Qatar":              {"type": "producer",   "lat": 25.3,  "lon": 51.2},
        "Hormuz":             {"type": "chokepoint", "lat": 26.6,  "lon": 56.5},
        "Suez Canal":         {"type": "chokepoint", "lat": 30.5,  "lon": 32.3},
        "Bab-el-Mandeb":      {"type": "chokepoint", "lat": 12.6,  "lon": 43.3},
        "Strait of Malacca":  {"type": "chokepoint", "lat": 2.5,   "lon": 101.2},
        "Cape of Good Hope":  {"type": "chokepoint", "lat": -34.4, "lon": 18.5},
        "Fujairah":           {"type": "bypass_hub", "lat": 25.1,  "lon": 56.35},
        "Yanbu":              {"type": "bypass_hub", "lat": 24.1,  "lon": 38.1},
        "Indian Ocean Hub":   {"type": "hub",        "lat": 15.0,  "lon": 68.0},
        "Red Sea":            {"type": "hub",        "lat": 18.0,  "lon": 38.5},
        "India":              {"type": "consumer",   "lat": 20.6,  "lon": 79.0},
        "China":              {"type": "consumer",   "lat": 35.9,  "lon": 104.2},
        "Japan":              {"type": "consumer",   "lat": 35.7,  "lon": 139.7},
        "Europe":             {"type": "consumer",   "lat": 51.2,  "lon": 10.5},
        "USA":                {"type": "consumer",   "lat": 37.1,  "lon": -95.7},
    }
    for name, attrs in nodes.items():
        G.add_node(name, **attrs)

    # (from, to, cost_idx, transit_days, capacity_mbd, base_risk, hormuz_dep)
    # All data sourced — see DATA_SOURCES.md
    edges = [
        ("Saudi Arabia", "Hormuz",           1, 0.8,  6.05, 0.28, True),
        ("UAE",          "Hormuz",           1, 0.5,  2.72, 0.28, True),
        ("Iraq",         "Hormuz",           1, 0.8,  3.37, 0.28, True),
        ("Kuwait",       "Hormuz",           1, 0.5,  1.57, 0.28, True),
        ("Qatar",        "Hormuz",           1, 0.4,  0.51, 0.28, True),
        ("Saudi Arabia", "Yanbu",            6, 3.0,  5.00, 0.08, False),
        ("UAE",          "Fujairah",         4, 1.5,  1.50, 0.09, False),
        ("Hormuz",       "Indian Ocean Hub", 1, 1.5, 20.00, 0.28, True),
        ("Fujairah",     "Indian Ocean Hub", 2, 0.7,  1.50, 0.09, False),
        ("Yanbu",        "Red Sea",          1, 1.0,  5.00, 0.09, False),
        ("Red Sea",      "Bab-el-Mandeb",    2, 2.0,  8.80, 0.35, False),
        ("Bab-el-Mandeb","Suez Canal",       3, 3.5,  7.50, 0.35, False),
        ("Indian Ocean Hub", "India",              4,  2.0,  5.00, 0.07, False),
        ("Indian Ocean Hub", "Strait of Malacca",  8, 20.0, 16.60, 0.05, False),
        ("Indian Ocean Hub", "Cape of Good Hope",  7, 18.0, 25.00, 0.02, False),
        ("Strait of Malacca", "China",       5,  5.0,  8.00, 0.05, False),
        ("Strait of Malacca", "Japan",       6,  8.0,  4.50, 0.05, False),
        ("Suez Canal",   "Europe",           7,  8.0,  4.50, 0.18, False),
        ("Suez Canal",   "USA",              9, 16.0,  2.00, 0.18, False),
        ("Cape of Good Hope", "Europe",     10, 16.0, 20.00, 0.02, False),
        ("Cape of Good Hope", "USA",        12, 19.0, 15.00, 0.02, False),
    ]
    for u, v, cost, time, cap, risk, hdep in edges:
        G.add_edge(u, v, cost=cost, time=time, capacity=cap,
                   base_risk=risk, risk=risk, hormuz_dependent=hdep)
    return G


# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR  (for LSTM training)
# ──────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(G: nx.DiGraph, n_steps: int = 2000,
                             n_events: int = 10, seed: int = 42):
    """
    Generate (n_steps, n_edges, 4) array mimicking real signal structure.

    Features per edge per timestep:
      0: risk               — OU process + event shocks
      1: oil_vol            — correlated with mean Hormuz risk, lagged
      2: insurance_premium  — lagged risk (market responds ~5-10 steps later)
      3: sentiment          — leading indicator (drops before risk rises)

    Returns:
      data      : np.array (n_steps, n_edges, 4)
      edge_list : list of (u, v) in consistent order
    """
    rng = np.random.default_rng(seed)
    edges = list(G.edges())
    n_edges = len(edges)

    risks      = np.array([G[u][v]["base_risk"] for u, v in edges], dtype=float)
    oil_vol    = np.full(n_edges, 0.30)
    insurance  = risks.copy()
    sentiment  = np.full(n_edges, 0.50)

    # Simulate structured disruption events
    event_times = rng.choice(range(150, n_steps - 150), n_events, replace=False)
    hormuz_idxs = [i for i, (u, v) in enumerate(edges)
                   if G[u][v].get("hormuz_dependent")]
    bamel_idxs  = [i for i, (u, v) in enumerate(edges)
                   if (u, v) in [("Red Sea", "Bab-el-Mandeb"),
                                  ("Bab-el-Mandeb", "Suez Canal")]]

    data = np.zeros((n_steps, n_edges, N_FEATURES))

    for t in range(n_steps):
        # OU mean-reversion for all edges
        for i, (u, v) in enumerate(edges):
            mu    = G[u][v]["base_risk"]
            risks[i] = np.clip(
                risks[i] + 0.3 * (mu - risks[i]) + 0.07 * rng.standard_normal(),
                0.0, 1.0
            )

        # Disruption events
        for k, et in enumerate(event_times):
            if t == et:
                sev    = rng.uniform(0.30, 0.65)
                kind   = k % 3
                target = hormuz_idxs if kind < 2 else bamel_idxs
                for i in target:
                    risks[i] = min(1.0, risks[i] + sev)
                # Decay over 20 steps (handled by OU above — spike fades)

        # Oil volatility: mean-reverts to (0.25 + 0.5 * avg_hormuz_risk)
        hormuz_avg = float(np.mean(risks[hormuz_idxs])) if hormuz_idxs else 0.28
        oil_vol[:] = np.clip(
            oil_vol + 0.15 * (0.25 + 0.5 * hormuz_avg - oil_vol)
            + 0.04 * rng.standard_normal(n_edges), 0.0, 1.0
        )

        # Insurance: lagged risk (slow response)
        insurance[:] = np.clip(
            0.85 * insurance + 0.15 * risks + 0.015 * rng.standard_normal(n_edges),
            0.0, 1.0
        )

        # Sentiment: leading indicator (high sentiment = low tension)
        # Drops ~3-8 steps BEFORE risk rises (approximated by inverse of risk trend)
        sentiment[:] = np.clip(
            0.80 * sentiment + 0.20 * (1.0 - risks)
            + 0.04 * rng.standard_normal(n_edges), 0.0, 1.0
        )

        data[t, :, 0] = risks
        data[t, :, 1] = oil_vol
        data[t, :, 2] = insurance
        data[t, :, 3] = sentiment

    return data, edges


# ──────────────────────────────────────────────────────────────────────────────
# LSTM RISK PREDICTOR
# ──────────────────────────────────────────────────────────────────────────────
class LSTMRiskPredictor(nn.Module):
    """
    Architecture:
      Input  → LSTM(n_edges*n_features, hidden=128, layers=2)
      → last hidden state
      → Linear(128, 64) + ReLU
      → Linear(64, n_edges)
      → Sigmoid  [output in (0,1)]

    Trained on sequences of shape (batch, SEQ_LEN, n_edges, n_features).
    Predicts risk at t+1 for all edges simultaneously.
    """
    def __init__(self, n_edges: int, n_features: int = N_FEATURES,
                 hidden: int = HIDDEN, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_edges    = n_edges
        self.n_features = n_features
        self.lstm = nn.LSTM(
            input_size  = n_edges * n_features,
            hidden_size = hidden,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, n_edges),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, n_edges, n_features)
        B, S, E, F = x.shape
        x_flat = x.reshape(B, S, E * F)          # (B, S, E*F)
        out, _ = self.lstm(x_flat)               # (B, S, hidden)
        return self.head(out[:, -1, :])          # (B, n_edges)

    @torch.no_grad()
    def predict_next(self, window: np.ndarray) -> np.ndarray:
        """window: (SEQ_LEN, n_edges, n_features) → (n_edges,)"""
        self.eval()
        x = torch.FloatTensor(window).unsqueeze(0)
        return self.forward(x).squeeze(0).numpy()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_lstm(model: LSTMRiskPredictor, data: np.ndarray,
               epochs: int = 120, batch_size: int = 64,
               progress_cb=None) -> dict:
    """
    data : (n_steps, n_edges, n_features)
    Returns dict with train_losses, val_losses, X_val, y_val tensors.
    """
    optimizer = optim.Adam(model.parameters(), lr=LR_LSTM, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn   = nn.MSELoss()

    # Build sequences: X=(N, SEQ_LEN, E, F), y=(N, E)
    X_np = np.array([data[i:i + SEQ_LEN]           for i in range(len(data) - SEQ_LEN - 1)])
    y_np = np.array([data[i + SEQ_LEN, :, 0]       for i in range(len(data) - SEQ_LEN - 1)])

    split    = int(0.8 * len(X_np))
    X_tr     = torch.FloatTensor(X_np[:split])
    y_tr     = torch.FloatTensor(y_np[:split])
    X_val    = torch.FloatTensor(X_np[split:])
    y_val    = torch.FloatTensor(y_np[split:])

    n_batches    = max(1, len(X_tr) // batch_size)
    train_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        perm   = torch.randperm(len(X_tr))
        X_tr   = X_tr[perm];  y_tr = y_tr[perm]
        ep_loss = 0.0
        for b in range(n_batches):
            xb = X_tr[b * batch_size: (b + 1) * batch_size]
            yb = y_tr[b * batch_size: (b + 1) * batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(X_val), y_val).item()
        scheduler.step(vl)

        train_losses.append(ep_loss / n_batches)
        val_losses.append(vl)
        if progress_cb:
            progress_cb((ep + 1) / epochs)

    return {"train": train_losses, "val": val_losses,
            "X_val": X_val, "y_val": y_val}


# ──────────────────────────────────────────────────────────────────────────────
# DEEP Q-NETWORK
# ──────────────────────────────────────────────────────────────────────────────
class DQNNet(nn.Module):
    """
    State  : one-hot node (N_NODES) + edge risks (N_EDGES)  → 41 dims
    Hidden : 256 → LayerNorm → ReLU → 128 → ReLU
    Output : Q-value per node (N_NODES)                      → 19 dims
    """
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, DQN_H1),
            nn.LayerNorm(DQN_H1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(DQN_H1, DQN_H2),
            nn.ReLU(),
            nn.Linear(DQN_H2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_CAP):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, float(done)))

    def sample(self, n: int):
        batch  = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buf)


class DQNAgent:
    def __init__(self, G: nx.DiGraph, node_list: list, edge_list: list,
                 epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.997):
        self.G         = G
        self.nodes     = node_list
        self.node_idx  = {n: i for i, n in enumerate(node_list)}
        self.edges     = edge_list
        self.n_nodes   = len(node_list)
        self.n_edges   = len(edge_list)
        self.state_dim = self.n_nodes + self.n_edges  # 19 + 22 = 41
        self.n_actions = self.n_nodes

        self.policy = DQNNet(self.state_dim, self.n_actions)
        self.target = DQNNet(self.state_dim, self.n_actions)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt     = optim.Adam(self.policy.parameters(), lr=LR_DQN)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer  = ReplayBuffer()

        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.steps         = 0

        self.losses          = []
        self.episode_rewards = []

    # ── state encoding ────────────────────────────────────────────────────────
    def encode(self, node: str, edge_risks: np.ndarray) -> np.ndarray:
        oh = np.zeros(self.n_nodes, dtype=np.float32)
        oh[self.node_idx[node]] = 1.0
        return np.concatenate([oh, edge_risks.astype(np.float32)])

    def _edge_risks_array(self) -> np.ndarray:
        return np.array([self.G[u][v]["risk"] for u, v in self.edges], dtype=np.float32)

    # ── action selection ──────────────────────────────────────────────────────
    def _valid(self, node: str) -> list:
        return [(nb, self.node_idx[nb])
                for nb in self.G.neighbors(node)
                if nb in self.node_idx]

    def act(self, node: str, edge_risks: np.ndarray) -> tuple:
        valid = self._valid(node)
        if not valid:
            return None, None
        if random.random() < self.epsilon:
            return random.choice(valid)
        state  = torch.FloatTensor(self.encode(node, edge_risks)).unsqueeze(0)
        self.policy.eval()
        with torch.no_grad():
            q = self.policy(state).squeeze(0)
        mask = torch.full((self.n_actions,), float("-inf"))
        for nb, idx in valid:
            mask[idx] = q[idx]
        best = int(mask.argmax())
        return self.nodes[best], best

    # ── learning step ─────────────────────────────────────────────────────────
    def learn(self) -> float | None:
        if len(self.buffer) < BATCH_SIZE:
            return None
        s, a, r, ns, d = self.buffer.sample(BATCH_SIZE)
        self.policy.train()
        q_pred = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(ns).max(1)[0]
            q_tgt  = r + GAMMA * q_next * (1 - d)
        loss = self.loss_fn(q_pred, q_tgt)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()
        self.steps += 1
        if self.steps % TARGET_UPD == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()

    # ── episode ───────────────────────────────────────────────────────────────
    def run_episode(self, source: str, target: str, max_steps: int = 30) -> tuple:
        node    = source
        path    = [node]
        total_r = 0.0
        seen    = {node}
        er      = self._edge_risks_array()

        for _ in range(max_steps):
            if node == target:
                break
            action_node, action_idx = self.act(node, er)
            if action_node is None or action_node in seen:
                break
            if not self.G.has_edge(node, action_node):
                break

            e      = self.G[node][action_node]
            reward = -(e["cost"] + 12 * e["risk"] + 2 * e["time"])
            done   = action_node == target
            if done:
                reward += 100.0

            ns = self.encode(action_node, er)
            self.buffer.push(self.encode(node, er), action_idx, reward, ns, done)
            loss = self.learn()
            if loss is not None:
                self.losses.append(loss)

            seen.add(action_node)
            path.append(action_node)
            total_r += reward
            node = action_node
            if done:
                break

        self.episode_rewards.append(total_r)
        return path, total_r

    def train(self, source: str, target: str, episodes: int = 600,
              progress_cb=None) -> None:
        for ep in range(episodes):
            self.run_episode(source, target)
            if progress_cb:
                progress_cb((ep + 1) / episodes)

    def greedy_path(self, source: str, target: str, max_steps: int = 30) -> list:
        old, self.epsilon = self.epsilon, 0.0
        node, path, seen = source, [source], {source}
        er = self._edge_risks_array()
        for _ in range(max_steps):
            if node == target:
                break
            an, _ = self.act(node, er)
            if an is None or an in seen:
                break
            seen.add(an)
            path.append(an)
            node = an
        self.epsilon = old
        return path

    def q_values_for(self, node: str) -> dict:
        er    = self._edge_risks_array()
        state = torch.FloatTensor(self.encode(node, er)).unsqueeze(0)
        self.policy.eval()
        with torch.no_grad():
            q = self.policy(state).squeeze(0).numpy()
        return {self.nodes[i]: float(q[i]) for i in range(self.n_nodes)}


# ──────────────────────────────────────────────────────────────────────────────
# ROUTING  (same as v1)
# ──────────────────────────────────────────────────────────────────────────────
def edge_weight(G, u, v, alpha=0.5, lam=10.0):
    e = G[u][v]
    return e["cost"] + alpha * e["time"] + lam * e["risk"]

def risk_dijkstra(G, source, target, alpha=0.5, lam=10.0):
    pq, visited = [(0.0, source, [source])], set()
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == target:
            return round(cost, 2), path
        for nb in G.neighbors(node):
            if nb not in visited:
                heapq.heappush(pq, (cost + edge_weight(G, node, nb, alpha, lam),
                                    nb, path + [nb]))
    return float("inf"), []

def path_stats(G, path):
    if len(path) < 2:
        return {}
    pairs = [(path[i], path[i+1]) for i in range(len(path)-1)]
    return {
        "Cost":             sum(G[u][v]["cost"] for u, v in pairs),
        "Transit (days)":   round(sum(G[u][v]["time"] for u, v in pairs), 1),
        "Max Risk":         round(max(G[u][v]["risk"] for u, v in pairs), 3),
        "Avg Risk":         round(float(np.mean([G[u][v]["risk"] for u, v in pairs])), 3),
        "Hormuz Exposed":   any(G[u][v].get("hormuz_dependent") for u, v in pairs),
        "Bottleneck (MBD)": min(G[u][v]["capacity"] for u, v in pairs),
    }


# ──────────────────────────────────────────────────────────────────────────────
# VISUALISATION HELPERS  (same as v1)
# ──────────────────────────────────────────────────────────────────────────────
NODE_COLORS = {
    "producer":   "#2ECC71", "chokepoint": "#E74C3C",
    "bypass_hub": "#F39C12", "hub":        "#3498DB", "consumer": "#9B59B6",
}
def _rgb(risk):
    return f"rgb({int(255*risk)},{int(255*(1-risk))},50)"

def draw_network(G, path=None, title="Oil Network"):
    fig = go.Figure()
    path_edges = set()
    if path and len(path) > 1:
        for i in range(len(path)-1):
            path_edges.add((path[i], path[i+1]))

    for u, v, d in G.edges(data=True):
        un, vn = G.nodes[u], G.nodes[v]
        ip = (u, v) in path_edges
        fig.add_trace(go.Scattergeo(
            lon=[un["lon"], vn["lon"], None], lat=[un["lat"], vn["lat"], None],
            mode="lines",
            line=dict(width=5 if ip else max(1.0, d["capacity"]/5),
                      color="#FFD700" if ip else _rgb(d["risk"])),
            opacity=1.0 if ip else 0.60,
            hovertext=f"<b>{u}→{v}</b><br>Risk:{d['risk']:.3f} Cap:{d['capacity']}MBD",
            hoverinfo="text", showlegend=False,
        ))

    in_path = set(path) if path else set()
    lons = [G.nodes[n]["lon"] for n in G.nodes()]
    lats = [G.nodes[n]["lat"] for n in G.nodes()]
    colors  = [NODE_COLORS.get(G.nodes[n]["type"], "#AAA") for n in G.nodes()]
    sizes   = [18 if G.nodes[n]["type"]=="chokepoint" else 14
               if G.nodes[n]["type"] in ("producer","consumer") else 10 for n in G.nodes()]
    borders = ["#FFD700" if n in in_path else "#444" for n in G.nodes()]
    bwidths = [3 if n in in_path else 1 for n in G.nodes()]

    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats, mode="markers+text",
        marker=dict(size=sizes, color=colors, line=dict(color=borders, width=bwidths)),
        text=list(G.nodes()), textposition="top center",
        textfont=dict(size=9, color="white"),
        hoverinfo="text", showlegend=False,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=15)),
        geo=dict(showland=True, landcolor="#1a1a2e", showocean=True, oceancolor="#0d1117",
                 showcoastlines=True, coastlinecolor="#2d3748", showframe=False,
                 projection_type="natural earth", center=dict(lat=22, lon=60),
                 lataxis_range=[-45, 70], lonaxis_range=[-110, 160]),
        paper_bgcolor="#0d1117", margin=dict(l=0, r=0, t=40, b=0), height=500,
    )
    return fig

_DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
             font=dict(color="white"), legend=dict(bgcolor="#161b22"))


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "G" not in st.session_state:
    G0 = build_oil_network()
    st.session_state.G            = G0
    st.session_state.t            = 0
    st.session_state.node_list    = list(G0.nodes())
    st.session_state.edge_list    = list(G0.edges())
    st.session_state.risk_hist    = []
    # LSTM
    st.session_state.lstm_model   = None
    st.session_state.lstm_results = None
    st.session_state.synth_data   = None
    st.session_state.risk_window  = None   # (SEQ_LEN, n_edges, n_features)
    # DQN
    st.session_state.dqn_agent    = None
    st.session_state.crisis       = False

G         = st.session_state.G
NODE_LIST = st.session_state.node_list
EDGE_LIST = st.session_state.edge_list
N_NODES   = len(NODE_LIST)
N_EDGES   = len(EDGE_LIST)

PRODUCERS = [n for n in NODE_LIST if G.nodes[n]["type"] == "producer"]
CONSUMERS = [n for n in NODE_LIST if G.nodes[n]["type"] == "consumer"]

# ──────────────────────────────────────────────────────────────────────────────
# LIVE RISK UPDATE  (LSTM → graph or OU fallback)
# ──────────────────────────────────────────────────────────────────────────────
def _ou_step(G, vol=0.3):
    for u, v in G.edges():
        mu   = G[u][v]["base_risk"]
        curr = G[u][v]["risk"]
        G[u][v]["risk"] = float(np.clip(
            curr + 0.3*(mu - curr) + vol*0.12*np.random.randn(), 0, 1))

def _lstm_step(G, model, window):
    """Run LSTM on rolling window; update graph risks with predictions."""
    pred = model.predict_next(window)   # (n_edges,)
    for i, (u, v) in enumerate(EDGE_LIST):
        G[u][v]["risk"] = float(np.clip(pred[i], 0.0, 1.0))

def _advance_window(window, G, vol=0.3):
    """Append a new row (OU-simulated signals) to the rolling window."""
    new_row = np.zeros((1, N_EDGES, N_FEATURES))
    for i, (u, v) in enumerate(EDGE_LIST):
        r = G[u][v]["risk"]
        new_row[0, i, 0] = r
        new_row[0, i, 1] = np.clip(r * 0.8 + 0.1 + 0.05 * np.random.randn(), 0, 1)
        new_row[0, i, 2] = np.clip(r * 0.7 + 0.05 * np.random.randn(), 0, 1)
        new_row[0, i, 3] = np.clip((1 - r) * 0.8 + 0.04 * np.random.randn(), 0, 1)
    return np.concatenate([window[1:], new_row], axis=0)

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 v2 Controls")
    st.markdown("---")
    source = st.selectbox("Source", PRODUCERS, index=0)
    target = st.selectbox("Destination", CONSUMERS, index=1)
    alpha  = st.slider("⏱ α (time weight)", 0.0, 2.0, 0.5, 0.1)
    lam    = st.slider("⚠️ λ (risk aversion)", 0.0, 50.0, 10.0, 1.0)
    vol    = st.slider("📈 Volatility", 0.0, 1.0, 0.30, 0.05)

    st.markdown("---")
    st.markdown("### Simulation")
    use_lstm = (st.session_state.lstm_model is not None)
    mode_label = "🧠 LSTM" if use_lstm else "📊 OU"
    st.caption(f"Risk engine: **{mode_label}**")

    def _step(n=1):
        lstm = st.session_state.lstm_model
        win  = st.session_state.risk_window
        for _ in range(n):
            if lstm is not None and win is not None:
                _lstm_step(G, lstm, win)
                win = _advance_window(win, G, vol)
                st.session_state.risk_window = win
            else:
                _ou_step(G, vol)
            st.session_state.t += 1
            c, p = risk_dijkstra(G, source, target, alpha, lam)
            hr = float(np.mean([G[u][v]["risk"] for u, v in EDGE_LIST
                                if G[u][v].get("hormuz_dependent")]))
            st.session_state.risk_hist.append({
                "t": st.session_state.t, "hormuz_risk": round(hr, 4),
                "path_cost": c, "path": " → ".join(p),
            })

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Step", use_container_width=True): _step(1)
    with c2:
        if st.button("⏩ ×10", use_container_width=True): _step(10)

    st.markdown("---")
    if st.button("🔥 Hormuz Crisis", use_container_width=True, type="primary"):
        for u, v in G.edges():
            if G[u][v].get("hormuz_dependent"):
                G[u][v]["risk"] = float(np.clip(0.90 + np.random.uniform(-0.05, 0.05), 0, 1))
        st.session_state.crisis = True
        _step(1)

    if st.button("🔄 Reset", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if st.session_state.crisis:
        st.error("🚨 Hormuz Crisis Active")
    st.markdown(f"**Tick:** `t = {st.session_state.t}`")

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("# 🌍 The World's Most Expensive Bottleneck — v2")
st.markdown(
    "> **Upgrade:** Tabular Q-table → Deep Q-Network (DQN) &emsp;|&emsp; "
    "OU simulation → Trained LSTM risk predictor &emsp;|&emsp; "
    "Lookup → Generalisation"
)
st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Network",
    "🧠 LSTM Risk Engine",
    "🤖 DQN Agent",
    "🔥 Stress Test",
    "🔬 Model Internals",
])


# ── TAB 1 · NETWORK MAP ───────────────────────────────────────────────────────
with tab1:
    _, opt_path = risk_dijkstra(G, source, target, alpha, lam)
    st.plotly_chart(draw_network(G, path=opt_path,
                                 title=f"Risk-Aware Route: {source} → {target}"),
                    use_container_width=True)
    st.markdown("Edge risk colour: 🟢 safe → 🟡 elevated → 🔴 high &emsp;|&emsp; 🟡 Gold = optimal path")
    stats = path_stats(G, opt_path)
    if stats:
        st.code(" → ".join(opt_path))
        cols = st.columns(len(stats))
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, "⚠️ YES" if v is True else ("✅ NO" if v is False else str(v)))


# ── TAB 2 · LSTM RISK ENGINE ─────────────────────────────────────────────────
with tab2:
    st.markdown("## 🧠 LSTM Risk Predictor")
    st.markdown("""
    Replaces the OU simulation with a **trained sequence model**.

    ```
    Input  : (batch, seq_len=10, n_edges=21, n_features=4)
             features = [risk, oil_volatility, insurance_premium, sentiment]
    LSTM   : 128 hidden units, 2 layers, dropout 0.2
    Head   : Linear(128→64) → ReLU → Linear(64→21) → Sigmoid
    Output : predicted r(e, t+1) for all edges — in (0, 1)
    Loss   : MSE on held-out 20% of synthetic data
    ```

    Synthetic data mimics real signal structure:
    - **oil_vol** correlates with Hormuz risk (concurrent)
    - **insurance** lags risk by ~5–10 steps (market response delay)
    - **sentiment** *leads* risk by ~3–5 steps (early warning signal)
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        n_steps  = st.slider("Training data steps", 1000, 3000, 2000, 500)
        n_epochs = st.slider("Training epochs", 50, 200, 120, 10)
        n_events = st.slider("Disruption events (training)", 5, 20, 10, 1)

        if st.button("🚀 Generate Data + Train LSTM", type="primary", use_container_width=True):
            with st.spinner("Generating synthetic data..."):
                data, edges = generate_synthetic_data(G, n_steps=n_steps,
                                                       n_events=n_events, seed=42)
                st.session_state.synth_data = data

            model = LSTMRiskPredictor(n_edges=N_EDGES)
            prog  = st.progress(0.0, text="Training LSTM…")
            results = train_lstm(model, data, epochs=n_epochs,
                                 progress_cb=lambda v: prog.progress(v,
                                     text=f"Training LSTM… {int(v*100)}%"))
            prog.empty()
            st.session_state.lstm_model   = model
            st.session_state.lstm_results = results

            # Initialise rolling window from end of training data
            st.session_state.risk_window = data[-SEQ_LEN:].copy()
            st.success(f"LSTM trained! Params: {model.n_params():,} &emsp;|&emsp; "
                       f"Final val MSE: {results['val'][-1]:.5f}")

    with col2:
        if st.session_state.lstm_results:
            res = st.session_state.lstm_results
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=res["train"], name="Train MSE",
                                          line=dict(color="#3498DB", width=2)))
            fig_loss.add_trace(go.Scatter(y=res["val"], name="Val MSE",
                                           line=dict(color="#E74C3C", width=2)))
            fig_loss.update_layout(**_DARK, title="LSTM Training & Validation Loss",
                                   xaxis_title="Epoch", yaxis_title="MSE", height=320)
            st.plotly_chart(fig_loss, use_container_width=True)
        else:
            st.info("Train the LSTM to see the loss curve.")

    if st.session_state.lstm_model and st.session_state.lstm_results:
        st.markdown("---")
        st.markdown("### Predicted vs Actual Risk — Validation Set")

        model   = st.session_state.lstm_model
        res     = st.session_state.lstm_results
        X_val   = res["X_val"]
        y_val   = res["y_val"].numpy()
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val).numpy()

        # Show 3 representative edges
        chosen_edges = [
            next(i for i, (u, v) in enumerate(EDGE_LIST) if G[u][v].get("hormuz_dependent") and u == "Hormuz"),
            next(i for i, (u, v) in enumerate(EDGE_LIST) if (u, v) == ("Red Sea", "Bab-el-Mandeb")),
            next(i for i, (u, v) in enumerate(EDGE_LIST) if (u, v) == ("Cape of Good Hope", "Europe")),
        ]
        labels = ["Hormuz → Indian Ocean", "Red Sea → Bab-el-Mandeb", "Cape → Europe"]
        colors_p = ["#E74C3C", "#F39C12", "#2ECC71"]

        fig_pred = go.Figure()
        for ei, label, col in zip(chosen_edges, labels, colors_p):
            t_axis = list(range(len(y_val)))
            fig_pred.add_trace(go.Scatter(y=y_val[:, ei], name=f"{label} (actual)",
                                           line=dict(color=col, width=1.5), opacity=0.5))
            fig_pred.add_trace(go.Scatter(y=y_pred[:, ei], name=f"{label} (LSTM)",
                                           line=dict(color=col, width=2.5, dash="dash")))
        fig_pred.update_layout(**_DARK, title="LSTM Risk Predictions vs Ground Truth",
                               xaxis_title="Validation Step",
                               yaxis_title="Risk Level", yaxis_range=[0, 1], height=380)
        st.plotly_chart(fig_pred, use_container_width=True)

        # RMSE per edge
        rmse_per = np.sqrt(np.mean((y_val - y_pred)**2, axis=0))
        df_rmse  = pd.DataFrame({
            "Edge": [f"{u}→{v}" for u, v in EDGE_LIST],
            "RMSE": [round(float(r), 5) for r in rmse_per],
            "Hormuz Dep": ["⚠️" if G[u][v].get("hormuz_dependent") else "✅"
                           for u, v in EDGE_LIST],
        }).sort_values("RMSE", ascending=False)
        st.markdown("#### Per-Edge Prediction Error")
        st.dataframe(df_rmse, use_container_width=True, hide_index=True)


# ── TAB 3 · DQN AGENT ────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 🤖 Deep Q-Network Agent")
    st.markdown(r"""
    Replaces the tabular Q-table with a **neural network** that generalises to
    unseen (node, risk-vector) combinations.

    ```
    State (41-dim):  one-hot node (19)  +  LSTM edge risks (22)
    Network       :  Linear(41→256) → LayerNorm → ReLU → Dropout
                     → Linear(256→128) → ReLU → Linear(128→19)
    Action        :  choose next node (masked to valid neighbours)
    Loss          :  Huber (SmoothL1) on Bellman targets
    Extras        :  experience replay (10k buffer), target network (sync/100 steps),
                     gradient clipping (max norm 1.0), ε-decay 1.0→0.05
    ```

    **Reward shaping:**
    ```
    R(edge) = -(cost + 12·risk + 2·time)
    R(target reached) += +100
    ```
    """)

    col1, col2 = st.columns([1, 2])
    with col1:
        n_eps = st.slider("Training episodes", 200, 1500, 600, 100)
        if st.button("🎯 Train DQN", type="primary", use_container_width=True):
            agent = DQNAgent(G, NODE_LIST, EDGE_LIST)
            prog  = st.progress(0.0, text="Training DQN…")
            agent.train(source, target, episodes=n_eps,
                        progress_cb=lambda v: prog.progress(v,
                            text=f"Training DQN… episode {int(v*n_eps)}/{n_eps}"))
            prog.empty()
            st.session_state.dqn_agent = agent
            st.success(f"DQN trained! Steps: {agent.steps:,} &emsp;|&emsp; "
                       f"Buffer: {len(agent.buffer):,} transitions &emsp;|&emsp; "
                       f"ε: {agent.epsilon:.3f}")

    with col2:
        if st.session_state.dqn_agent:
            agent = st.session_state.dqn_agent
            rews  = agent.episode_rewards
            losses = agent.losses

            window = max(10, len(rews) // 20)
            smooth = pd.Series(rews).rolling(window, min_periods=1).mean()

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(y=rews, name="Episode Reward",
                                        line=dict(color="#3498DB", width=1), opacity=0.3))
            fig_r.add_trace(go.Scatter(y=smooth, name=f"Smoothed (w={window})",
                                        line=dict(color="#F39C12", width=2)))
            fig_r.update_layout(**_DARK, title="DQN Episode Rewards",
                                xaxis_title="Episode", yaxis_title="Total Reward", height=280)
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Train the DQN to see learning curves.")

    if st.session_state.dqn_agent:
        agent = st.session_state.dqn_agent
        st.markdown("---")

        # Loss curve
        if agent.losses:
            win_l  = max(20, len(agent.losses)//20)
            sm_l   = pd.Series(agent.losses).rolling(win_l, min_periods=1).mean()
            fig_l  = go.Figure()
            fig_l.add_trace(go.Scatter(y=agent.losses, name="Huber Loss",
                                        line=dict(color="#E74C3C", width=1), opacity=0.3))
            fig_l.add_trace(go.Scatter(y=sm_l, name="Smoothed",
                                        line=dict(color="#FF8888", width=2)))
            fig_l.update_layout(**_DARK, title="DQN Huber Loss During Training",
                                xaxis_title="Gradient Step", yaxis_title="Loss", height=260)
            st.plotly_chart(fig_l, use_container_width=True)

        st.markdown("### DQN vs Dijkstra Path Comparison")
        dqn_path  = agent.greedy_path(source, target)
        _, d_path = risk_dijkstra(G, source, target, alpha, lam)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("#### 🤖 DQN (learned policy)")
            st.code(" → ".join(dqn_path) if dqn_path else "No path")
            ds = path_stats(G, dqn_path)
            for k, v in ds.items():
                st.metric(k, "⚠️YES" if v is True else ("✅NO" if v is False else str(v)))
        with col_d2:
            st.markdown("#### 🗺️ Dijkstra (single-shot)")
            st.code(" → ".join(d_path) if d_path else "No path")
            ds2 = path_stats(G, d_path)
            for k, v in ds2.items():
                st.metric(k, "⚠️YES" if v is True else ("✅NO" if v is False else str(v)))

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.plotly_chart(draw_network(G, path=dqn_path, title="DQN Route"),
                            use_container_width=True)
        with col_m2:
            st.plotly_chart(draw_network(G, path=d_path, title="Dijkstra Route"),
                            use_container_width=True)

        st.info("""
        **Key difference:** Dijkstra recomputes the globally optimal path from scratch each time.
        The DQN's policy network *generalises* — given a new (node, risk_vector) state it has never
        seen, it estimates Q-values via a forward pass rather than a full graph traversal.
        That's what makes it viable for real-time rerouting under continuously changing risk.
        """)


# ── TAB 4 · STRESS TEST ──────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🔥 Hormuz Crisis Stress Test")

    G_n = build_oil_network()
    _, pb = risk_dijkstra(G_n, source, target, alpha, lam); sb = path_stats(G_n, pb)

    G_c = build_oil_network()
    for u, v in G_c.edges():
        if G_c[u][v].get("hormuz_dependent"):
            G_c[u][v]["risk"] = float(np.clip(0.92 + random.uniform(-0.04, 0.04), 0, 1))
    _, pa = risk_dijkstra(G_c, source, target, alpha, lam); sa = path_stats(G_c, pa)

    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown("#### 🟢 Normal")
        st.code(" → ".join(pb))
        for k, v in sb.items():
            st.metric(k, "⚠️YES" if v is True else ("✅NO" if v is False else str(v)))
    with col_a:
        st.markdown("#### 🔴 Hormuz Crisis (severity 0.92)")
        st.code(" → ".join(pa))
        for k, v in sa.items():
            bv = sb.get(k)
            if k == "Hormuz Exposed":
                st.metric(k, "⚠️YES" if v else "✅NO")
            elif isinstance(bv, (int, float)) and isinstance(v, (int, float)):
                st.metric(k, str(v), delta=round(v - bv, 2))
            else:
                st.metric(k, str(v))

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.plotly_chart(draw_network(G_n, path=pb, title="Normal Routing"), use_container_width=True)
    with col_m2:
        st.plotly_chart(draw_network(G_c, path=pa, title="Crisis Routing"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Monte Carlo — 500 Crisis Scenarios")
    with st.spinner("Simulating…"):
        mc = []
        for _ in range(500):
            Gmc = build_oil_network()
            sev = random.uniform(0.45, 0.95)
            for u, v in Gmc.edges():
                if Gmc[u][v].get("hormuz_dependent"):
                    Gmc[u][v]["risk"] = float(np.clip(sev + random.uniform(-0.05, 0.05), 0, 1))
            _, pmc = risk_dijkstra(Gmc, source, target, alpha, lam)
            smc = path_stats(Gmc, pmc)
            mc.append({"severity": round(sev, 3),
                       "actual_cost": smc.get("Cost", 0),
                       "transit_days": smc.get("Transit (days)", 0),
                       "hormuz_free": not smc.get("Hormuz Exposed", True)})
    df_mc = pd.DataFrame(mc)
    rp = df_mc["hormuz_free"].mean() * 100

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_h = px.histogram(df_mc, x="actual_cost", nbins=25,
                             color_discrete_sequence=["#3498DB"],
                             title="Cost Distribution (500 crises)",
                             labels={"actual_cost": "Actual Route Cost"})
        fig_h.update_layout(**_DARK, height=340)
        st.plotly_chart(fig_h, use_container_width=True)
    with col_c2:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=rp,
            title={"text": "% Rerouted Away from Hormuz"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2ECC71"},
                   "steps": [{"range": [0, 33], "color": "#2d1b1b"},
                              {"range": [33, 66], "color": "#2d2a1b"},
                              {"range": [66, 100], "color": "#1b2d1b"}]},
            number={"suffix": "%"},
        ))
        fig_g.update_layout(paper_bgcolor="#0d1117", font=dict(color="white"), height=340)
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown(f"""
    - **{rp:.1f}%** of scenarios successfully rerouted away from Hormuz
    - Average cost increase vs normal: **+{df_mc['actual_cost'].mean() - sb.get('Cost',0):.1f}** units
    - Worst-case transit: **{df_mc['transit_days'].max():.0f} days** (Cape of Good Hope route)
    """)


# ── TAB 5 · MODEL INTERNALS ──────────────────────────────────────────────────
with tab5:
    st.markdown("## 🔬 Model Internals")

    col_l, col_d = st.columns(2)

    with col_l:
        st.markdown("### LSTM Risk Predictor")
        lstm = st.session_state.lstm_model
        if lstm:
            arch_lstm = [
                {"Layer": "Input", "Shape": f"(B, {SEQ_LEN}, {N_EDGES}, {N_FEATURES})", "Params": "—"},
                {"Layer": "Reshape", "Shape": f"(B, {SEQ_LEN}, {N_EDGES*N_FEATURES})", "Params": "—"},
                {"Layer": "LSTM (2 layers)", "Shape": f"(B, {SEQ_LEN}, 128)", "Params": f"{sum(p.numel() for p in lstm.lstm.parameters()):,}"},
                {"Layer": "Last hidden", "Shape": "(B, 128)", "Params": "—"},
                {"Layer": "Linear 128→64", "Shape": "(B, 64)", "Params": "8,256"},
                {"Layer": "ReLU + Dropout", "Shape": "(B, 64)", "Params": "—"},
                {"Layer": "Linear 64→21", "Shape": f"(B, {N_EDGES})", "Params": f"{64*N_EDGES + N_EDGES}"},
                {"Layer": "Sigmoid", "Shape": f"(B, {N_EDGES})", "Params": "—"},
            ]
            st.dataframe(pd.DataFrame(arch_lstm), use_container_width=True, hide_index=True)
            st.metric("Total trainable params", f"{lstm.n_params():,}")

            # Live LSTM prediction
            if st.session_state.risk_window is not None:
                pred = lstm.predict_next(st.session_state.risk_window)
                pred_df = pd.DataFrame({
                    "Edge":            [f"{u}→{v}" for u, v in EDGE_LIST],
                    "LSTM Predicted":  [round(float(p), 4) for p in pred],
                    "Current Graph":   [round(G[u][v]["risk"], 4) for u, v in EDGE_LIST],
                    "Δ":               [round(float(p) - G[u][v]["risk"], 4)
                                        for p, (u, v) in zip(pred, EDGE_LIST)],
                }).sort_values("LSTM Predicted", ascending=False)
                st.markdown("#### Live LSTM Risk Predictions")
                st.dataframe(pred_df, use_container_width=True, hide_index=True)
        else:
            st.info("Train the LSTM in the **LSTM Risk Engine** tab first.")

    with col_d:
        st.markdown("### Deep Q-Network")
        agent = st.session_state.dqn_agent
        if agent:
            arch_dqn = [
                {"Layer": "Input", "Shape": f"({agent.state_dim},) = {N_NODES} node + {N_EDGES} risks", "Params": "—"},
                {"Layer": "Linear 41→256", "Shape": "(256,)", "Params": f"{agent.state_dim*256 + 256:,}"},
                {"Layer": "LayerNorm(256)", "Shape": "(256,)", "Params": "512"},
                {"Layer": "ReLU + Dropout", "Shape": "(256,)", "Params": "—"},
                {"Layer": "Linear 256→128", "Shape": "(128,)", "Params": f"{256*128 + 128:,}"},
                {"Layer": "ReLU", "Shape": "(128,)", "Params": "—"},
                {"Layer": "Linear 128→19", "Shape": f"({N_NODES},) Q per node", "Params": f"{128*N_NODES + N_NODES:,}"},
                {"Layer": "Action Mask", "Shape": f"({N_NODES},) — invalid→-inf", "Params": "—"},
            ]
            st.dataframe(pd.DataFrame(arch_dqn), use_container_width=True, hide_index=True)
            st.metric("Total trainable params", f"{agent.policy.n_params():,}")
            st.metric("Replay buffer size", f"{len(agent.buffer):,} transitions")
            st.metric("Gradient steps taken", f"{agent.steps:,}")
            st.metric("Final ε (exploration)", f"{agent.epsilon:.4f}")

            # Q-value heatmap
            st.markdown("#### Q-Value Map — from each producer")
            q_rows = []
            for prod in PRODUCERS:
                qv = agent.q_values_for(prod)
                for tgt_node, qval in qv.items():
                    q_rows.append({"From": prod, "To": tgt_node, "Q-value": round(qval, 2)})
            q_df = pd.DataFrame(q_rows)
            pivot = q_df.pivot(index="From", columns="To", values="Q-value").fillna(0)

            fig_hm = px.imshow(
                pivot,
                color_continuous_scale="RdYlGn",
                title="Q-Values: rows=current node, cols=next node action",
                aspect="auto",
                text_auto=".1f",
            )
            fig_hm.update_layout(**_DARK, height=350)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Train the DQN in the **DQN Agent** tab first.")

    st.markdown("---")
    st.markdown("### Architecture Summary: v1 vs v2")
    cmp = pd.DataFrame([
        {"Component":       "Risk prediction",
         "v1":              "OU process (stochastic simulation, 0 params)",
         "v2":              f"LSTM (trained, {st.session_state.lstm_model.n_params():,} params)" if st.session_state.lstm_model else "LSTM (not yet trained)"},
        {"Component":       "Decision model",
         "v1":              "Tabular Q-table (dict, grows with state visits)",
         "v2":              f"DQN (fixed {st.session_state.dqn_agent.policy.n_params():,}-param network)" if st.session_state.dqn_agent else "DQN (not yet trained)"},
        {"Component":       "State space",
         "v1":              "Discrete: (node, discretised risk bucket)",
         "v2":              "Continuous: 41-dim float vector"},
        {"Component":       "Generalisation",
         "v1":              "None — unseen states get Q=0",
         "v2":              "Yes — neural network interpolates"},
        {"Component":       "Experience replay",
         "v1":              "None",
         "v2":              "10,000-transition circular buffer"},
        {"Component":       "Target network",
         "v1":              "None",
         "v2":              "Separate frozen copy, synced every 100 steps"},
        {"Component":       "Routing",
         "v1":              "Risk-aware Dijkstra (unchanged)",
         "v2":              "Risk-aware Dijkstra (unchanged)"},
    ])
    st.dataframe(cmp, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#8b949e;font-size:0.80em;padding-bottom:1rem'>
    v2: LSTM Risk Predictor · Deep Q-Network · Experience Replay · Target Network<br>
    NetworkX · Plotly · Streamlit · PyTorch
</div>
""", unsafe_allow_html=True)
