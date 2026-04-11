"""
The World's Most Expensive Bottleneck
Risk-Aware Routing on a Dynamic Oil Network — Interactive POC

DATA SOURCES (all figures are real, cited):
─────────────────────────────────────────────────────────────────────────────
CHOKEPOINT CAPACITIES
  Hormuz:          20 MBD  — EIA, 2024  (eia.gov/todayinenergy/detail.php?id=65504)
  Strait of Malacca: 16.6 MBD crude — EIA, H1 2025
  Suez Canal:      7.5 MBD normal capacity — EIA/IEA Suez Factsheet 2023
  Bab-el-Mandeb:   8.8 MBD pre-Houthi (2023); 4.2 MBD post-attacks (Q1 2025)
  SUMED Pipeline:  2.5 MBD — EIA Red Sea Chokepoints analysis

BYPASS PIPELINES
  Saudi East-West (Petroline → Yanbu): 5 MBD capacity (7 MBD post-2026 upgrade)
    — Fortune / Argus Media, March 2026; CEO Amin Nasser confirmation
  UAE ADCO pipeline (Habshan → Fujairah): 1.5 MBD (expandable to 1.8)
    — Wikipedia / Global Energy Monitor / ADNOC, operational since June 2012

PRODUCER EXPORTS (2023–2024 actuals, MBD)
  Saudi Arabia:  6.05–6.66 MBD  — CEIC Data / FRED / Statista
  UAE:           2.65–2.72 MBD crude exports — CEIC Data
  Iraq:          3.37–3.47 MBD  — Iraq Oil Report / EIA
  Kuwait:        1.57–1.80 MBD  — CEIC Data / EIA
  Qatar:         0.51 MBD crude (+ 9.3 Bcf/d LNG) — Statista / EIA

TRANSIT DISTANCES & TIMES (VLCC at 13–14 knots)
  Gulf → Hormuz:       ~250 nm → ~0.8 days
  Hormuz → India:      ~1,500 nm → ~2 days   (Zee News maritime refs)
  Hormuz → Malacca:    ~6,600 nm → ~20 days  (Maritime Executive)
  Hormuz → Suez (via Indian Ocean): ~3,900 nm → ~12 days
  Suez → NW Europe:    ~3,000 nm → ~8 days   (MyDello Suez Guide)
  Hormuz → Cape → Rotterdam: ~15,500 nm → ~34 days  (S&P Global)
  Cape → US Gulf:      adds ~8 days vs. Suez route

WAR-RISK INSURANCE PREMIUMS (as % of hull value per transit)
  Hormuz / Gulf baseline (2024):    0.25%   — S&P Global / Lloyd's
  Hormuz during 2019 attacks:       0.50%   — Strauss Center
  Hormuz 2026 escalation:           1.00%   — Lloyd's List March 2026
  Bab-el-Mandeb pre-Houthi:        0.05%   — Policyholder Pulse
  Bab-el-Mandeb 2024 peak:         2.00%   — AGBI / Maplecroft (2,700% rise)
  Strait of Malacca (2024–25):     0.05%   — Lloyd's JWC (removed from high-risk May 2024)
  Cape of Good Hope:               ~0.01%  — industry baseline

FREIGHT RATES (VLCC, Worldscale, 2023 averages)
  AG → Japan (TD3C):  WS 50–90 (peak ~WS90 March 2023, ~WS50–60 Dec 2023)
    — Signal Group 2023 Tanker Annual Review / Baltic Exchange
  AG → NW Europe / US Gulf: no free public 2023 averages found;
    costs estimated from TD3C + distance scaling and OPEC MMR context.
─────────────────────────────────────────────────────────────────────────────
COST COLUMN UNITS: normalised index (1 unit ≈ USD 0.30–0.40/barrel at mid-2023
  VLCC rates; derived by dividing WS-implied $/bbl by 0.35 and rounding)
RISK COLUMN: calibrated to war-risk insurance premium bands (see TECHNICAL.md §3.3)
"""

import heapq
import random
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The World's Most Expensive Bottleneck",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    h1, h2, h3 { color: #58a6ff; }
    .stTabs [data-baseweb="tab"] { color: #8b949e; font-size: 0.95rem; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #58a6ff; border-bottom: 2px solid #58a6ff; }
    blockquote { border-left: 3px solid #58a6ff; padding-left: 1em; color: #8b949e; }
    code { background: #161b22; border-radius: 4px; padding: 2px 6px; }
    .stDataFrame { background: #161b22; }
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 10px 14px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

def build_oil_network() -> nx.DiGraph:
    G = nx.DiGraph()

    nodes = {
        # Producers
        "Saudi Arabia":       {"type": "producer",    "lat": 23.0,   "lon": 45.0},
        "UAE":                {"type": "producer",    "lat": 24.5,   "lon": 54.5},
        "Iraq":               {"type": "producer",    "lat": 33.0,   "lon": 44.0},
        "Kuwait":             {"type": "producer",    "lat": 29.5,   "lon": 47.5},
        "Qatar":              {"type": "producer",    "lat": 25.3,   "lon": 51.2},
        # Chokepoints
        "Hormuz":             {"type": "chokepoint",  "lat": 26.6,   "lon": 56.5},
        "Suez Canal":         {"type": "chokepoint",  "lat": 30.5,   "lon": 32.3},
        "Bab-el-Mandeb":      {"type": "chokepoint",  "lat": 12.6,   "lon": 43.3},
        "Strait of Malacca":  {"type": "chokepoint",  "lat": 2.5,    "lon": 101.2},
        "Cape of Good Hope":  {"type": "chokepoint",  "lat": -34.4,  "lon": 18.5},
        # Bypass / transit hubs
        "Fujairah":           {"type": "bypass_hub",  "lat": 25.1,   "lon": 56.35},
        "Yanbu":              {"type": "bypass_hub",  "lat": 24.1,   "lon": 38.1},
        "Indian Ocean Hub":   {"type": "hub",         "lat": 15.0,   "lon": 68.0},
        "Red Sea":            {"type": "hub",         "lat": 18.0,   "lon": 38.5},
        # Consumers
        "India":              {"type": "consumer",    "lat": 20.6,   "lon": 79.0},
        "China":              {"type": "consumer",    "lat": 35.9,   "lon": 104.2},
        "Japan":              {"type": "consumer",    "lat": 35.7,   "lon": 139.7},
        "Europe":             {"type": "consumer",    "lat": 51.2,   "lon": 10.5},
        "USA":                {"type": "consumer",    "lat": 37.1,   "lon": -95.7},
    }

    for name, attrs in nodes.items():
        G.add_node(name, **attrs)

    # ── REAL DATA EDGES ────────────────────────────────────────────────────────
    # Columns: (from, to, cost_index, transit_days, capacity_mbd, base_risk, hormuz_dep)
    #
    # cost_index  : normalised (1 unit ≈ $0.30–0.40/bbl at 2023 VLCC mid-rates)
    # transit_days: derived from real nautical distances at 13–14 knot VLCC speed
    # capacity_mbd: EIA 2023-2024 measured throughput / pipeline nameplate capacity
    # base_risk   : calibrated to Lloyd's / S&P war-risk insurance premium bands
    #               (0.01 = ~0.01% hull premium ≡ open ocean; 0.30 = ~0.25-0.50% ≡ Hormuz)
    #
    edges = [
        # ── Gulf producers → Hormuz ──────────────────────────────────────────
        # Capacity = 2023-24 actual crude exports (EIA/CEIC). Distance ~250nm, ~0.8 days.
        # Risk 0.28 = calibrated to 0.25% hull war-risk premium (S&P Global 2024 baseline)
        ("Saudi Arabia", "Hormuz",           1,  0.8,  6.05, 0.28, True),   # 6.05 MBD: CEIC Dec-2024
        ("UAE",          "Hormuz",           1,  0.5,  2.72, 0.28, True),   # 2.72 MBD: CEIC Dec-2024
        ("Iraq",         "Hormuz",           1,  0.8,  3.37, 0.28, True),   # 3.37 MBD: Iraq Oil Report 2024
        ("Kuwait",       "Hormuz",           1,  0.5,  1.57, 0.28, True),   # 1.57 MBD: CEIC Dec-2023
        ("Qatar",        "Hormuz",           1,  0.4,  0.51, 0.28, True),   # 0.51 MBD crude: Statista 2024

        # ── Bypass routes (Hormuz-free) ───────────────────────────────────────
        # Saudi East-West Pipeline (Petroline): 5 MBD nameplate (Argus Media / Fortune 2026)
        # Pump station transit ~3 days; premium ~+25% vs tanker for same origin-dest pair
        ("Saudi Arabia", "Yanbu",            6,  3.0,  5.00, 0.08, False),

        # UAE ADCO pipeline Habshan→Fujairah: 1.5 MBD (ADNOC / Global Energy Monitor)
        # 360km onshore, ~1.5 days equivalent; low war-risk (onshore UAE)
        ("UAE",          "Fujairah",         4,  1.5,  1.50, 0.09, False),

        # ── Hormuz outbound → Indian Ocean ────────────────────────────────────
        # Hormuz total throughput 20 MBD (EIA 2024). ~500 nm to open Indian Ocean → ~1.5 days
        # Risk matches Hormuz-dependent premium (0.28)
        ("Hormuz",       "Indian Ocean Hub", 1,  1.5, 20.00, 0.28, True),

        # Fujairah direct to Indian Ocean (bypass, no Hormuz risk)
        # 1.5 MBD pipeline capacity; ~200nm offshore → ~0.7 days
        ("Fujairah",     "Indian Ocean Hub", 2,  0.7,  1.50, 0.09, False),

        # ── Yanbu / Red Sea legs ──────────────────────────────────────────────
        # Yanbu loading → Red Sea: short coastal run ~1 day; SUMED parallel pipeline 2.5 MBD
        ("Yanbu",        "Red Sea",          1,  1.0,  5.00, 0.09, False),

        # Red Sea → Bab-el-Mandeb: ~700nm → ~2 days
        # Bab-el-Mandeb pre-Houthi throughput 8.8 MBD (EIA 2023)
        # Risk 0.35 = 2024 Houthi-era elevated premium (down from peak 2.0% hull → 0.70% hull)
        ("Red Sea",      "Bab-el-Mandeb",    2,  2.0,  8.80, 0.35, False),

        # Bab-el-Mandeb → Suez Canal: ~1,200nm → ~3.5 days
        # Suez throughput 7.5 MBD normal capacity (IEA Factsheet); 3.9 MBD during Houthi disruption
        # Risk 0.35 same corridor; SUMED pipeline (2.5 MBD) adds parallel capacity counted here
        ("Bab-el-Mandeb","Suez Canal",       3,  3.5,  7.50, 0.35, False),

        # ── Indian Ocean Hub → consumer legs ─────────────────────────────────
        # IOH → India: Ras Tanura → Mumbai ~1,500nm → ~2 days at 14kn (Zee News maritime)
        # India VLCC berth capacity; freight cost ~$1.50/bbl → index 4
        ("Indian Ocean Hub", "India",            4,  2.0,  5.00, 0.07, False),

        # IOH → Strait of Malacca: ~6,600nm → ~20 days (Maritime Executive confirmed)
        # Malacca crude throughput 16.6 MBD H1-2025 (EIA); cost index 8 (proportional to TD3C)
        # Risk 0.05 = Lloyd's JWC removed from high-risk list May 2024
        ("Indian Ocean Hub", "Strait of Malacca", 8, 20.0, 16.60, 0.05, False),

        # IOH → Cape of Good Hope: ~6,000nm → ~18 days
        # No capacity ceiling (open ocean); very low risk (0.02 ≈ 0.01% hull premium)
        ("Indian Ocean Hub", "Cape of Good Hope", 7, 18.0, 25.00, 0.02, False),

        # ── Malacca → East Asia ───────────────────────────────────────────────
        # Malacca → China (Ningbo/Qingdao): ~1,600nm → ~5 days
        # Malacca → Japan (Chiba/Yokohama): ~2,800nm → ~8 days
        ("Strait of Malacca", "China",       5,  5.0,  8.00, 0.05, False),
        ("Strait of Malacca", "Japan",       6,  8.0,  4.50, 0.05, False),

        # ── Suez Canal → Western markets ─────────────────────────────────────
        # Suez → NW Europe (Rotterdam): ~3,000nm → ~8 days (MyDello Suez Guide)
        # Total PG→Rotterdam via Suez ~11,200nm; cost index 7 (TD3C-equivalent scaling)
        ("Suez Canal",   "Europe",           7,  8.0,  4.50, 0.18, False),

        # Suez → US Gulf: ~5,500nm → ~16 days (EIA Strauss Center routing analysis)
        ("Suez Canal",   "USA",              9, 16.0,  2.00, 0.18, False),

        # ── Cape of Good Hope → Western markets ──────────────────────────────
        # Cape → NW Europe: total PG→Rotterdam ~15,500nm (S&P Global); Cape leg ~5,500nm → ~16 days
        # 7–10 extra days vs Suez; very low risk; higher cost due to bunker consumption
        ("Cape of Good Hope", "Europe",     10, 16.0, 20.00, 0.02, False),

        # Cape → US Gulf: additional ~4,500nm beyond Cape-Europe → ~3 more days
        ("Cape of Good Hope", "USA",        12, 19.0, 15.00, 0.02, False),

        # Cape → India: Cape Town to Mumbai ~5,200nm → ~16 days (open Indian Ocean)
        # Used when Hormuz is blocked; low risk, high cost due to extra distance
        ("Cape of Good Hope", "India",       9, 16.0, 20.00, 0.02, False),

        # Cape → Strait of Malacca: Cape to Malacca ~5,800nm → ~18 days
        # Extreme bypass for China/Japan when both Hormuz and Red Sea routes are high-risk
        ("Cape of Good Hope", "Strait of Malacca", 10, 18.0, 16.60, 0.02, False),

        # ── Bab-el-Mandeb southbound → Indian Ocean ───────────────────────────
        # Ships exiting the Red Sea southward (from Yanbu bypass) into the Indian Ocean.
        # Gulf of Aden → northern Indian Ocean: ~1,500nm → ~4.5 days
        # Enables: Saudi Arabia → Yanbu → Red Sea → Bab-el-Mandeb → IOH → India/Asia
        # without transiting Hormuz. Risk 0.35 (same Bab/Gulf of Aden corridor).
        ("Bab-el-Mandeb", "Indian Ocean Hub", 3, 4.5, 8.80, 0.35, False),
    ]

    for u, v, cost, time, cap, risk, hdep in edges:
        G.add_edge(u, v,
                   cost=cost, time=time, capacity=cap,
                   base_risk=risk, risk=risk,
                   hormuz_dependent=hdep)
    return G


# ──────────────────────────────────────────────────────────────────────────────
# RISK ENGINE  (Ornstein-Uhlenbeck mean-reverting stochastic process)
# ──────────────────────────────────────────────────────────────────────────────

def simulate_step(G: nx.DiGraph, volatility: float = 0.3) -> None:
    for u, v in G.edges():
        mu      = G[u][v]["base_risk"]
        current = G[u][v]["risk"]
        theta   = 0.3
        sigma   = volatility * 0.12
        shock   = np.random.normal(0, 1)
        new_r   = current + theta * (mu - current) + sigma * shock
        G[u][v]["risk"] = float(np.clip(new_r, 0.0, 1.0))


def apply_hormuz_crisis(G: nx.DiGraph, severity: float = 0.88) -> None:
    for u, v in G.edges():
        if G[u][v].get("hormuz_dependent", False):
            G[u][v]["risk"] = float(np.clip(severity + random.uniform(-0.05, 0.05), 0, 1))


def reset_risks(G: nx.DiGraph) -> None:
    for u, v in G.edges():
        G[u][v]["risk"] = G[u][v]["base_risk"]


# ──────────────────────────────────────────────────────────────────────────────
# ROUTING  (risk-aware Dijkstra)
# ──────────────────────────────────────────────────────────────────────────────

def edge_weight(G: nx.DiGraph, u: str, v: str,
                alpha: float = 0.5, lam: float = 10.0) -> float:
    e = G[u][v]
    return e["cost"] + alpha * e["time"] + lam * e["risk"]


def risk_dijkstra(G: nx.DiGraph, source: str, target: str,
                  alpha: float = 0.5, lam: float = 10.0):
    pq = [(0.0, source, [source])]
    visited: set = set()
    while pq:
        cost, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == target:
            return round(cost, 2), path
        for nb in G.neighbors(node):
            if nb not in visited:
                w = edge_weight(G, node, nb, alpha, lam)
                heapq.heappush(pq, (cost + w, nb, path + [nb]))
    return float("inf"), []


def path_stats(G: nx.DiGraph, path: list) -> dict:
    if len(path) < 2:
        return {}
    pairs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return {
        "Cost":              sum(G[u][v]["cost"] for u, v in pairs),
        "Transit (days)":    round(sum(G[u][v]["time"] for u, v in pairs), 1),
        "Max Risk":          round(max(G[u][v]["risk"] for u, v in pairs), 3),
        "Avg Risk":          round(float(np.mean([G[u][v]["risk"] for u, v in pairs])), 3),
        "Hormuz Exposed":    any(G[u][v].get("hormuz_dependent") for u, v in pairs),
        "Bottleneck (MBD)":  min(G[u][v]["capacity"] for u, v in pairs),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Q-LEARNING AGENT
# ──────────────────────────────────────────────────────────────────────────────

class QLearningAgent:
    def __init__(self, alpha: float = 0.15, gamma: float = 0.9, epsilon: float = 0.5):
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon
        self.Q: dict = defaultdict(float)

    def _state(self, G: nx.DiGraph, node: str) -> tuple:
        risks = tuple(round(G[u][v]["risk"] * 4) / 4 for u, v in sorted(G.edges()))
        return (node, risks)

    def _act(self, G: nx.DiGraph, node: str) -> str | None:
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return None
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        state = self._state(G, node)
        return max(neighbors, key=lambda a: self.Q[(state, a)])

    def _update(self, G: nx.DiGraph, s: str, action: str, reward: float, ns: str) -> None:
        state      = self._state(G, s)
        next_state = self._state(G, ns)
        next_nbrs  = list(G.neighbors(ns))
        best_next  = max((self.Q[(next_state, a)] for a in next_nbrs), default=0.0)
        self.Q[(state, action)] += self.alpha * (
            reward + self.gamma * best_next - self.Q[(state, action)]
        )

    def _episode(self, G: nx.DiGraph, source: str, target: str,
                 max_steps: int = 30) -> tuple[list, float]:
        node   = source
        path   = [node]
        total  = 0.0
        seen   = {node}
        for _ in range(max_steps):
            if node == target:
                break
            action = self._act(G, node)
            if action is None or action in seen or not G.has_edge(node, action):
                break
            e      = G[node][action]
            reward = -(e["cost"] + 10 * e["risk"] + 2 * e["time"])
            if action == target:
                reward += 50.0
            self._update(G, node, action, reward, action)
            seen.add(action)
            path.append(action)
            total += reward
            node = action
        return path, total

    def train(self, G: nx.DiGraph, source: str, target: str,
              episodes: int = 300) -> list[float]:
        rewards = []
        for _ in range(episodes):
            self.epsilon = max(0.05, self.epsilon * 0.994)
            _, r = self._episode(G, source, target)
            rewards.append(r)
        return rewards

    def greedy_path(self, G: nx.DiGraph, source: str, target: str) -> list:
        old, self.epsilon = self.epsilon, 0.0
        path, _ = self._episode(G, source, target)
        self.epsilon = old
        return path


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────

NODE_COLORS = {
    "producer":    "#2ECC71",
    "chokepoint":  "#E74C3C",
    "bypass_hub":  "#F39C12",
    "hub":         "#3498DB",
    "consumer":    "#9B59B6",
}

def _risk_rgb(risk: float) -> str:
    r = int(255 * risk)
    g = int(255 * (1.0 - risk))
    return f"rgb({r},{g},50)"


def draw_network(G: nx.DiGraph, path: list | None = None,
                 title: str = "Oil Network") -> go.Figure:
    fig = go.Figure()
    path_edges: set = set()
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i + 1]))

    # Edges
    for u, v, data in G.edges(data=True):
        un, vn    = G.nodes[u], G.nodes[v]
        is_path   = (u, v) in path_edges
        risk      = data["risk"]
        width     = 5 if is_path else max(1.0, data["capacity"] / 5)
        color     = "#FFD700" if is_path else _risk_rgb(risk)
        opacity   = 1.0 if is_path else 0.60
        fig.add_trace(go.Scattergeo(
            lon=[un["lon"], vn["lon"], None],
            lat=[un["lat"], vn["lat"], None],
            mode="lines",
            line=dict(width=width, color=color),
            opacity=opacity,
            hoverinfo="text",
            hovertext=(f"<b>{u} → {v}</b><br>"
                       f"Cost: {data['cost']} | Time: {data['time']}d<br>"
                       f"Risk: {risk:.3f} | Capacity: {data['capacity']} MBD<br>"
                       f"Hormuz-dependent: {data.get('hormuz_dependent', False)}"),
            showlegend=False,
        ))

    # Nodes
    in_path = set(path) if path else set()
    lons, lats, colors, sizes, borders, bwidths, texts, hovers = [], [], [], [], [], [], [], []
    for n in G.nodes():
        nd = G.nodes[n]
        lons.append(nd["lon"])
        lats.append(nd["lat"])
        colors.append(NODE_COLORS.get(nd["type"], "#AAA"))
        t = nd["type"]
        sizes.append(18 if t == "chokepoint" else (14 if t in ("producer", "consumer") else 10))
        borders.append("#FFD700" if n in in_path else "#444")
        bwidths.append(3 if n in in_path else 1)
        texts.append(n)
        hovers.append(f"<b>{n}</b><br>Type: {t}")

    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats,
        mode="markers+text",
        marker=dict(size=sizes, color=colors,
                    line=dict(color=borders, width=bwidths)),
        text=texts,
        textposition="top center",
        textfont=dict(size=9, color="white"),
        hovertext=hovers,
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="white", size=16)),
        geo=dict(
            showland=True,     landcolor="#1a1a2e",
            showocean=True,    oceancolor="#0d1117",
            showcoastlines=True, coastlinecolor="#2d3748",
            showframe=False,
            projection_type="natural earth",
            center=dict(lat=22, lon=60),
            lataxis_range=[-45, 70],
            lonaxis_range=[-110, 160],
        ),
        paper_bgcolor="#0d1117",
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig


_DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
             font=dict(color="white"), legend=dict(bgcolor="#161b22"))


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────

if "G" not in st.session_state:
    st.session_state.G           = build_oil_network()
    st.session_state.t           = 0
    st.session_state.risk_hist   = []
    st.session_state.rl_agent    = None
    st.session_state.rl_rewards  = []
    st.session_state.crisis      = False

G = st.session_state.G

PRODUCERS = ["Saudi Arabia", "UAE", "Iraq", "Kuwait", "Qatar"]
CONSUMERS = ["India", "China", "Japan", "Europe", "USA"]


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🛢️ Network Controls")
    st.markdown("---")

    source = st.selectbox("Source (Producer)", PRODUCERS, index=0)
    target = st.selectbox("Destination (Consumer)", CONSUMERS, index=1)

    st.markdown("---")
    st.markdown("### Routing Parameters")
    alpha = st.slider("⏱ Time Weight α", 0.0, 2.0, 0.5, 0.1,
                      help="Cost weight on transit time")
    lam   = st.slider("⚠️ Risk Aversion λ", 0.0, 50.0, 10.0, 1.0,
                      help="Higher = pay more to avoid risky edges")

    st.markdown("---")
    st.markdown("### Simulation")
    volatility = st.slider("📈 Volatility", 0.0, 1.0, 0.30, 0.05)

    c1, c2 = st.columns(2)

    def _advance(n=1):
        for _ in range(n):
            simulate_step(G, volatility)
            st.session_state.t += 1
            c, p = risk_dijkstra(G, source, target, alpha, lam)
            h_risk = float(np.mean([G[u][v]["risk"]
                                    for u, v in G.edges()
                                    if G[u][v].get("hormuz_dependent")]))
            st.session_state.risk_hist.append({
                "t":           st.session_state.t,
                "hormuz_risk": round(h_risk, 4),
                "path_cost":   c,
                "path":        " → ".join(p),
            })

    with c1:
        if st.button("▶ Step", use_container_width=True):
            _advance(1)
    with c2:
        if st.button("⏩ ×10", use_container_width=True):
            _advance(10)

    st.markdown("---")
    if st.button("🔥 Hormuz Crisis", use_container_width=True, type="primary"):
        apply_hormuz_crisis(G, severity=0.90)
        st.session_state.crisis = True
        st.session_state.t += 1
        c, p = risk_dijkstra(G, source, target, alpha, lam)
        st.session_state.risk_hist.append({
            "t": st.session_state.t, "hormuz_risk": 0.90,
            "path_cost": c, "path": " → ".join(p),
        })

    if st.button("🔄 Reset", use_container_width=True):
        for key in ["G", "t", "risk_hist", "rl_agent", "rl_rewards", "crisis"]:
            del st.session_state[key]
        st.rerun()

    if st.session_state.crisis:
        st.error("🚨 Hormuz Crisis Active")

    st.markdown(f"**Tick:** `t = {st.session_state.t}`")


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("# 🌍 The World's Most Expensive Bottleneck")
st.markdown(
    "> *An interactive proof-of-concept: graph theory meets geopolitical risk "
    "in global oil routing. Break the network. Watch it scramble.*"
)
st.markdown("---")


# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Network Map",
    "🛣️ Route Finder",
    "📡 Risk Simulator",
    "🔥 Stress Test",
    "🤖 RL Agent",
])


# ── TAB 1 · NETWORK MAP ───────────────────────────────────────────────────────
with tab1:
    st.markdown("## The World as a Graph")
    st.markdown("""
    Every node is a port, refinery hub, or chokepoint.
    Every edge carries **cost**, **transit time**, **capacity**, and **risk**.
    The gold path is the current risk-aware optimal route.

    > Edge color: 🟢 safe → 🟡 elevated → 🔴 high risk &emsp;|&emsp; Edge width ∝ capacity
    """)

    _, opt_path = risk_dijkstra(G, source, target, alpha, lam)
    st.plotly_chart(draw_network(G, path=opt_path,
                                 title=f"Risk-Aware Route: {source} → {target}"),
                    use_container_width=True)

    # Legend
    legend = [("🟢 Producer", ""), ("🔴 Chokepoint", ""),
              ("🟠 Bypass Hub", ""), ("🔵 Transit Hub", ""), ("🟣 Consumer", "")]
    for col, (label, _) in zip(st.columns(5), legend):
        col.caption(label)

    # Stats row
    stats = path_stats(G, opt_path)
    if stats:
        st.markdown("### Current Optimal Path")
        st.code(" → ".join(opt_path))
        cols = st.columns(len(stats))
        for col, (k, v) in zip(cols, stats.items()):
            col.metric(k, "⚠️ YES" if v is True else ("✅ NO" if v is False else str(v)))


# ── TAB 2 · ROUTE FINDER ─────────────────────────────────────────────────────
with tab2:
    st.markdown("## Route Finder: Cost vs Risk Tradeoff")
    st.markdown("""
    The routing objective isn't shortest path — it's minimum **risk-adjusted cost**:

    ```
    w(e) = cost(e) + α·time(e) + λ·risk(e, t)
    ```

    Move λ from 0 → 50 and watch the system switch from Hormuz-dependent to bypass routes.
    """)

    lambdas = [0, 5, 10, 20, 35, 50]
    rows = []
    for l in lambdas:
        c, p = risk_dijkstra(G, source, target, alpha, float(l))
        s    = path_stats(G, p)
        rows.append({
            "λ":                   l,
            "Path":                " → ".join(p) if p else "No path",
            "Actual Cost":         s.get("Cost", "-"),
            "Transit (days)":      s.get("Transit (days)", "-"),
            "Max Risk":            s.get("Max Risk", "-"),
            "Hormuz Exposed":      "⚠️" if s.get("Hormuz Exposed") else "✅",
            "Bottleneck (MBD)":    s.get("Bottleneck (MBD)", "-"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Cost vs Risk Frontier")

    lam_sweep   = np.linspace(0, 50, 40)
    actual_costs, max_risks, transits = [], [], []
    for l in lam_sweep:
        _, p = risk_dijkstra(G, source, target, alpha, float(l))
        s = path_stats(G, p)
        actual_costs.append(s.get("Cost", 0))
        max_risks.append(s.get("Max Risk", 0))
        transits.append(s.get("Transit (days)", 0))

    fig_tradeoff = go.Figure()
    fig_tradeoff.add_trace(go.Scatter(
        x=lam_sweep, y=actual_costs, name="Actual Cost",
        line=dict(color="#3498DB", width=2), mode="lines",
    ))
    fig_tradeoff.add_trace(go.Scatter(
        x=lam_sweep, y=[r * max(actual_costs) for r in max_risks],
        name="Max Risk (scaled)", yaxis="y2",
        line=dict(color="#E74C3C", width=2, dash="dash"), mode="lines",
    ))
    fig_tradeoff.update_layout(
        **_DARK,
        title="As λ rises: cost goes up, risk goes down — but the jump marks the bypass switch-over",
        xaxis_title="Risk Aversion λ",
        yaxis=dict(title="Actual Route Cost"),
        yaxis2=dict(title="Max Edge Risk (scaled)", overlaying="y", side="right"),
        height=380,
    )
    st.plotly_chart(fig_tradeoff, use_container_width=True)

    st.info(
        "💡 **The jump in cost** at a certain λ marks the exact point where the algorithm "
        "switches from the Hormuz route to a longer but safer bypass. "
        "That cost premium = the price of resilience."
    )

    # Show both routes side by side on map
    st.markdown("### Route Visualised: λ=0 (cost-only) vs Current λ")
    _, cheap_path = risk_dijkstra(G, source, target, 0.5, 0.0)
    _, safe_path  = risk_dijkstra(G, source, target, alpha, lam)
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(f"**λ=0 (pure cost):** `{' → '.join(cheap_path)}`")
        st.plotly_chart(draw_network(G, path=cheap_path, title="λ=0: Cheapest Route"),
                        use_container_width=True)
    with col_m2:
        st.markdown(f"**λ={lam} (risk-aware):** `{' → '.join(safe_path)}`")
        st.plotly_chart(draw_network(G, path=safe_path, title=f"λ={lam}: Risk-Aware Route"),
                        use_container_width=True)


# ── TAB 3 · RISK SIMULATOR ───────────────────────────────────────────────────
with tab3:
    st.markdown("## Dynamic Risk Evolution")
    st.markdown(r"""
    Risk isn't static — it evolves as a **mean-reverting stochastic process**
    (Ornstein-Uhlenbeck):

    ```
    dR(t) = θ(μ - R(t))dt + σ dW(t)
    ```

    θ = mean-reversion speed, μ = baseline risk, σ = volatility, W = Wiener process.

    Use **▶ Step** or **⏩ ×10** in the sidebar to advance the simulation.
    """)

    if not st.session_state.risk_hist:
        st.info("Use the sidebar controls to step the simulation forward.")
        # Show static snapshot
        risk_rows = []
        for u, v, d in G.edges(data=True):
            risk_rows.append({
                "Edge":              f"{u} → {v}",
                "Current Risk":      round(d["risk"], 3),
                "Capacity (MBD)":    d["capacity"],
                "Hormuz Dependent":  "⚠️" if d.get("hormuz_dependent") else "✅",
            })
        df_snap = pd.DataFrame(risk_rows).sort_values("Current Risk", ascending=False)
        st.markdown("### Current Edge Risk Snapshot")
        st.dataframe(df_snap, use_container_width=True, hide_index=True)
    else:
        df_h = pd.DataFrame(st.session_state.risk_hist)

        # Risk evolution
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(
            x=df_h["t"], y=df_h["hormuz_risk"],
            name="Avg Hormuz-Route Risk",
            line=dict(color="#E74C3C", width=2),
            fill="tozeroy", fillcolor="rgba(231,76,60,0.12)",
        ))
        fig_risk.add_shape(type="line",
                           x0=df_h["t"].min(), x1=df_h["t"].max(), y0=0.25, y1=0.25,
                           line=dict(color="#F39C12", dash="dash", width=1))
        fig_risk.add_annotation(
            x=df_h["t"].max(), y=0.27,
            text="Baseline risk (0.25)", showarrow=False, font=dict(color="#F39C12")
        )

        # Mark crisis events
        for i, row in df_h[df_h["hormuz_risk"] > 0.70].iterrows():
            fig_risk.add_vline(x=row["t"], line=dict(color="#FF4444", dash="dot", width=1))

        fig_risk.update_layout(
            **_DARK,
            title="Hormuz-Route Risk Over Time",
            xaxis_title="Simulation Tick", yaxis_title="Risk Level",
            yaxis_range=[0, 1], height=350,
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # Weighted path cost evolution
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Scatter(
            x=df_h["t"], y=df_h["path_cost"],
            name="Weighted Route Cost",
            line=dict(color="#3498DB", width=2),
        ))
        fig_cost.update_layout(
            **_DARK,
            title="Optimal Route Weighted Cost Over Time",
            xaxis_title="Simulation Tick", yaxis_title="Weighted Cost",
            height=300,
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        st.markdown("### Path History (last 10 ticks)")
        st.dataframe(df_h.tail(10)[["t", "hormuz_risk", "path_cost", "path"]],
                     use_container_width=True, hide_index=True)


# ── TAB 4 · STRESS TEST ──────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🔥 Hormuz Crisis Stress Test")
    st.markdown("""
    **Scenario:** Geopolitical escalation blocks the Strait of Hormuz.
    ~20 million barrels/day suddenly have nowhere to go.

    The routing algorithm is forced to re-solve on a degraded graph.
    What does it find? What does resilience *cost*?
    """)

    G_normal = build_oil_network()
    _, path_before = risk_dijkstra(G_normal, source, target, alpha, lam)
    sb = path_stats(G_normal, path_before)

    G_crisis = build_oil_network()
    apply_hormuz_crisis(G_crisis, severity=0.92)
    _, path_after = risk_dijkstra(G_crisis, source, target, alpha, lam)
    sa = path_stats(G_crisis, path_after)

    st.markdown("### Before vs After")
    col_b, col_a = st.columns(2)
    with col_b:
        st.markdown("#### 🟢 Normal Conditions")
        st.code(" → ".join(path_before))
        for k, v in sb.items():
            st.metric(k, "⚠️ YES" if v is True else ("✅ NO" if v is False else str(v)))
    with col_a:
        st.markdown("#### 🔴 Hormuz Crisis (severity = 0.92)")
        st.code(" → ".join(path_after))
        for k, v in sa.items():
            bv = sb.get(k)
            if k == "Hormuz Exposed":
                st.metric(k, "⚠️ YES" if v else "✅ NO")
            elif isinstance(bv, (int, float)) and isinstance(v, (int, float)):
                st.metric(k, str(v), delta=round(v - bv, 2))
            else:
                st.metric(k, str(v))

    st.markdown("---")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.plotly_chart(draw_network(G_normal, path=path_before, title="Normal Routing"),
                        use_container_width=True)
    with col_m2:
        st.plotly_chart(draw_network(G_crisis, path=path_after, title="Crisis Routing"),
                        use_container_width=True)

    st.markdown("---")
    st.markdown("### Monte Carlo: Disruption Cost Distribution (500 scenarios)")

    with st.spinner("Running Monte Carlo simulation..."):
        mc = []
        for _ in range(500):
            Gmc = build_oil_network()
            sev = random.uniform(0.45, 0.95)
            apply_hormuz_crisis(Gmc, severity=sev)
            _, pmc = risk_dijkstra(Gmc, source, target, alpha, lam)
            smc = path_stats(Gmc, pmc)
            mc.append({
                "severity":       round(sev, 3),
                "actual_cost":    smc.get("Cost", 0),
                "transit_days":   smc.get("Transit (days)", 0),
                "hormuz_free":    not smc.get("Hormuz Exposed", True),
            })
    df_mc = pd.DataFrame(mc)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_mc = px.histogram(
            df_mc, x="actual_cost", nbins=25,
            color_discrete_sequence=["#3498DB"],
            title="Route Cost Distribution Across Crisis Scenarios",
            labels={"actual_cost": "Actual Route Cost"},
        )
        fig_mc.update_layout(**_DARK, height=350)
        st.plotly_chart(fig_mc, use_container_width=True)

    with col_c2:
        rerouted_pct = df_mc["hormuz_free"].mean() * 100
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rerouted_pct,
            title={"text": "% Scenarios Rerouted Away from Hormuz"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#2ECC71"},
                "steps": [
                    {"range": [0,  33], "color": "#2d1b1b"},
                    {"range": [33, 66], "color": "#2d2a1b"},
                    {"range": [66, 100], "color": "#1b2d1b"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 66},
            },
            number={"suffix": "%", "font": {"color": "white"}},
        ))
        fig_gauge.update_layout(paper_bgcolor="#0d1117", font=dict(color="white"), height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

    cost_increase = df_mc["actual_cost"].mean() - sb.get("Cost", 0)
    st.markdown(f"""
    **Findings across 500 simulated crises:**
    - **{rerouted_pct:.1f}%** of scenarios forced a Hormuz-free reroute
    - Average cost increase: **+{cost_increase:.1f}** units
    - Worst-case transit time: **{df_mc['transit_days'].max():.0f} days** (Cape of Good Hope)
    - At severity > 0.75, {df_mc[df_mc['severity'] > 0.75]['hormuz_free'].mean() * 100:.0f}% of routes avoid Hormuz entirely

    > The network *can* adapt — but resilience is expensive. Every extra day and dollar is the premium you pay for not having built redundancy upfront.
    """)


# ── TAB 5 · RL AGENT ─────────────────────────────────────────────────────────
with tab5:
    st.markdown("## 🤖 Reinforcement Learning: From Pathfinding to Policy")
    st.markdown(r"""
    Dijkstra answers: *"What's the best route right now?"*

    The RL agent answers: *"How should I route in general, across many risk conditions?"*

    It learns a **Q-function** — a value estimate for every (state, action) pair:

    ```
    Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') − Q(s,a)]
    ```

    **Reward signal:**
    ```
    R(edge) = -(cost + 10·risk + 2·time)
    R(reaching target) += +50
    ```

    The agent explores randomly at first (high ε), then exploits learned knowledge (ε → 0.05).
    """)

    c1, c2 = st.columns([1, 2])
    with c1:
        n_episodes = st.slider("Training Episodes", 100, 1000, 300, 50)
        if st.button("🎯 Train Agent", type="primary", use_container_width=True):
            with st.spinner(f"Training Q-Learning agent for {n_episodes} episodes..."):
                agent   = QLearningAgent(alpha=0.15, gamma=0.9, epsilon=0.5)
                rewards = agent.train(G, source, target, episodes=n_episodes)
                st.session_state.rl_agent   = agent
                st.session_state.rl_rewards = rewards
            st.success("Training complete!")

    with c2:
        if st.session_state.rl_rewards:
            rews     = st.session_state.rl_rewards
            window   = max(10, len(rews) // 20)
            smoothed = pd.Series(rews).rolling(window, min_periods=1).mean()

            fig_rl = go.Figure()
            fig_rl.add_trace(go.Scatter(
                y=rews, mode="lines", name="Episode Reward",
                line=dict(color="#3498DB", width=1), opacity=0.35,
            ))
            fig_rl.add_trace(go.Scatter(
                y=smoothed, mode="lines", name=f"Rolling Mean (w={window})",
                line=dict(color="#F39C12", width=2),
            ))
            fig_rl.update_layout(
                **_DARK,
                title="Training Reward Curve — Agent Improving Over Episodes",
                xaxis_title="Episode", yaxis_title="Total Episode Reward",
                height=320,
            )
            st.plotly_chart(fig_rl, use_container_width=True)
        else:
            st.info("Train the agent above to see the learning curve.")

    if st.session_state.rl_agent:
        st.markdown("---")
        st.markdown("### Learned Policy vs Dijkstra")

        agent     = st.session_state.rl_agent
        rl_path   = agent.greedy_path(G, source, target)
        _, d_path = risk_dijkstra(G, source, target, alpha, lam)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### 🤖 RL Agent (learned policy)")
            st.code(" → ".join(rl_path) if rl_path else "No path found")
            rl_s = path_stats(G, rl_path)
            for k, v in rl_s.items():
                st.metric(k, "⚠️ YES" if v is True else ("✅ NO" if v is False else str(v)))
        with col_r2:
            st.markdown("#### 🗺️ Dijkstra (single-shot optimal)")
            st.code(" → ".join(d_path) if d_path else "No path found")
            d_s = path_stats(G, d_path)
            for k, v in d_s.items():
                st.metric(k, "⚠️ YES" if v is True else ("✅ NO" if v is False else str(v)))

        col_map_rl, col_map_dijk = st.columns(2)
        with col_map_rl:
            st.plotly_chart(draw_network(G, path=rl_path,  title="RL Agent Route"),
                            use_container_width=True)
        with col_map_dijk:
            st.plotly_chart(draw_network(G, path=d_path, title="Dijkstra Route"),
                            use_container_width=True)

        st.info("""
        💡 **Why RL matters beyond Dijkstra:**
        The Q-learning agent stores a *generalised policy* — it doesn't re-solve the graph each time.
        In a real deployment it could make near-instant rerouting decisions as new risk signals arrive,
        without running a full graph search. It also naturally discovers routes that balance cost,
        risk, and time through experience rather than through a hand-tuned objective function.
        """)

    else:
        st.info("👆 Train the agent first to compare it against Dijkstra.")


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8b949e; font-size:0.82em; padding-bottom:1rem'>
    Built with NetworkX · Plotly · Streamlit · Q-Learning<br>
    Graph: G=(V,E) where V = ports/chokepoints, E = shipping lanes with dynamic risk weights<br>
    Objective: min Σ[cost(e) + α·time(e) + λ·risk(e,t)]
</div>
""", unsafe_allow_html=True)
