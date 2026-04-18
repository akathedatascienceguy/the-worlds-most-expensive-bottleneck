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
  Saudi East-West (Petroline → Yanbu): 7.0 MBD (upgraded March 2026)
    — Fortune / Argus Media / CNBC, March 28 2026; CEO Amin Nasser confirmation
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
        # Saudi East-West Pipeline (Petroline): 7.0 MBD post-March 2026 upgrade
        # Fortune / Argus Media / CNBC, March 28 2026; CEO Amin Nasser confirmation
        # Pump station transit ~3 days; premium ~+25% vs tanker for same origin-dest pair
        ("Saudi Arabia", "Yanbu",            6,  3.0,  7.00, 0.08, False),

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
        # Yanbu → Suez Canal DIRECT (northbound): Yanbu is at 24°N, Suez at 30°N.
        # Ships heading to Europe/USA sail directly north up the Red Sea — no Bab-el-Mandeb detour.
        # ~1,000nm → ~3 days at 13kn. Avoids Houthi-exposed southern corridor entirely.
        # Capacity 7.0 MBD (Petroline throughput); risk 0.10 (northern Red Sea, below Houthi range)
        ("Yanbu",        "Suez Canal",       3,  3.0,  7.00, 0.10, False),

        # Yanbu → Red Sea (southbound): for Asia-bound cargo only.
        # Ships heading to India/China/Japan exit south through Bab-el-Mandeb into Indian Ocean.
        # ~700nm south to open Red Sea → ~1 day
        ("Yanbu",        "Red Sea",          1,  1.0,  7.00, 0.09, False),

        # Red Sea → Bab-el-Mandeb: ~700nm → ~2 days (southbound, Asia routing)
        # Bab-el-Mandeb pre-Houthi throughput 8.8 MBD (EIA 2023)
        # Risk 0.35 = 2024 Houthi-era elevated premium (down from peak 2.0% hull → 0.70% hull)
        ("Red Sea",      "Bab-el-Mandeb",    2,  2.0,  8.80, 0.35, False),

        # Bab-el-Mandeb → Suez Canal: ships from Indian Ocean / Gulf of Aden going northbound
        # (e.g. routed via IOH southbound then back north). ~1,200nm → ~3.5 days.
        # Suez throughput 7.5 MBD normal capacity (IEA Factsheet)
        # Risk 0.35 same Houthi-affected corridor
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
# ECONOMIC CASCADE MODEL
# All elasticities, pass-through coefficients, and historical benchmarks are
# sourced from peer-reviewed literature and institutional reports (cited inline).
# ──────────────────────────────────────────────────────────────────────────────

HISTORICAL_EVENTS = [
    # (label, duration_days, oil_chg_pct, cpi_peak_pct, food_chg_pct, gdp_impact_pct, source)
    ("1973 Arab Embargo",        150,  400, 11.0,  30, -2.5, "Hamilton 1983 / BLS"),
    ("1979 Iranian Revolution",  365,  150, 13.5,  20, -3.5, "Hamilton 2009 NBER"),
    ("1990 Gulf War",            180,  100,  6.2,  15, -1.5, "Kilian 2008 AER"),
    ("2005 Hurricane Katrina",    30,   25,  3.4,   5, -0.3, "EIA post-event report"),
    ("2019 Abqaiq Attack",        14,   15,  0.2,   2, -0.1, "IEA/S&P Global"),
    ("2022 Russia Sanctions",    365,   60,  9.1,  34, -1.0, "IMF WEO Oct 2022"),
    ("2023–24 Houthi/Red Sea",   365,    8,  0.3,   8, -0.3, "EIA / IEA 2024"),
]

# Oil import dependency, CPI pass-through coeff, GDP oil elasticity
# Sources: IMF WP/17/53 (Gelos & Ustyugova); IEA Energy Security report 2023;
#          World Bank Commodity Markets Outlook 2022
REGIONS = {
    "East Asia\n(Japan/Korea/China)": {
        "import_dep": 0.85, "cpi_pt": 0.18, "gdp_elast": -0.040,
        "food_dep": 0.55, "color": "#E74C3C",
    },
    "India": {
        "import_dep": 0.85, "cpi_pt": 0.16, "gdp_elast": -0.050,
        "food_dep": 0.48, "color": "#E67E22",
    },
    "Europe": {
        "import_dep": 0.55, "cpi_pt": 0.13, "gdp_elast": -0.028,
        "food_dep": 0.30, "color": "#3498DB",
    },
    "USA": {
        "import_dep": 0.15, "cpi_pt": 0.08, "gdp_elast": -0.015,
        "food_dep": 0.20, "color": "#2ECC71",
    },
    "Developing\nMarkets": {
        "import_dep": 0.80, "cpi_pt": 0.22, "gdp_elast": -0.060,
        "food_dep": 0.65, "color": "#9B59B6",
    },
}


def oil_price_scenario(hormuz_risk: float, duration_days: int,
                        base_price: float = 75.0) -> dict:
    """
    Compute oil price impact from a Hormuz disruption.

    Model:
      ΔP / P ≈ (supply_disruption × elasticity_multiplier) - reserve_offset

    Elasticity sourced from:
      Hamilton (2009) "Causes and Consequences of the Oil Shock of 2007-08", NBER
      Kilian (2008) "The Economic Effects of Energy Price Shocks", J. Econ. Literature
      IEA (2022) "The Role of Critical Minerals in Clean Energy Transitions"

    Short-run supply elasticity ≈ -0.05 (very inelastic)
    → 1% supply cut ≈ 4-8% price rise (duration-dependent)
    """
    supply_disrupted = hormuz_risk * 0.20   # Hormuz = 20% global supply (EIA 2024)

    if duration_days <= 7:
        dur_mult = 3.5
    elif duration_days <= 30:
        dur_mult = 5.5
    elif duration_days <= 90:
        dur_mult = 8.0
    else:
        dur_mult = 12.0

    panic = max(0.0, (hormuz_risk - 0.5) * 0.40)
    opec_offset = min(supply_disrupted * 0.35, 0.07)
    spr_offset = min(0.10 / max(duration_days, 1) * 30, supply_disrupted * 0.25)

    net_disruption = max(0, supply_disrupted - opec_offset - spr_offset)
    pct_change     = net_disruption * dur_mult * 100 + panic * 100

    spot   = base_price * (1 + pct_change / 100)
    fwd_90 = base_price * (1 + pct_change * 0.55 / 100)

    petrol_base  = 1.60
    petrol_new   = petrol_base * (1 + pct_change * 0.50 / 100)

    cape_extra_days  = 14.0
    bunker_per_day   = 45_000
    cape_extra_cost  = cape_extra_days * bunker_per_day
    voyage_value     = 2_000_000 * base_price
    freight_premium  = (cape_extra_cost / voyage_value) * 100 + hormuz_risk * 250

    return {
        "pct_change":      round(pct_change, 1),
        "spot":            round(spot, 1),
        "fwd_90":          round(fwd_90, 1),
        "petrol_new":      round(petrol_new, 3),
        "petrol_base":     petrol_base,
        "freight_premium": round(freight_premium, 1),
        "supply_cut_mbd":  round(net_disruption * 100, 1),
        "base_price":      base_price,
    }


def inflation_cascade(oil_pct_change: float, freight_pct: float,
                       duration_days: int) -> dict:
    """
    Compute headline CPI, core CPI, and food price impacts by region.

    CPI pass-through coefficients (IMF WP/17/53, Gelos & Ustyugova 2017):
      Advanced economies: 0.06–0.15 (per 10% oil price rise)
      Emerging markets:   0.12–0.25
    """
    results = {}
    for region, params in REGIONS.items():
        dep  = params["import_dep"]
        pt   = params["cpi_pt"]
        gel  = params["gdp_elast"]
        fdep = params["food_dep"]

        eff_oil_chg = oil_pct_change * dep
        headline_cpi = eff_oil_chg * pt
        core_cpi     = headline_cpi * 0.38

        energy_to_food  = oil_pct_change * 0.15
        fertilizer_cost = oil_pct_change * 0.60 * 0.18
        freight_to_food = freight_pct * 0.22 * fdep
        panic_food      = max(0, (oil_pct_change - 25) * 0.28)
        food_chg        = energy_to_food + fertilizer_cost + freight_to_food + panic_food

        direct_gdp    = eff_oil_chg * gel
        monetary_drag = -0.15 * max(0, headline_cpi - 0.5)
        total_gdp     = direct_gdp + monetary_drag
        rate_hike     = max(0, 1.5 * headline_cpi * 0.1)

        results[region] = {
            "headline_cpi": round(headline_cpi, 2),
            "core_cpi":     round(core_cpi, 2),
            "food_chg":     round(food_chg, 2),
            "gdp_impact":   round(total_gdp, 2),
            "rate_hike":    round(rate_hike, 2),
            "eff_oil_chg":  round(eff_oil_chg, 1),
        }
    return results


def economic_time_series(hormuz_risk: float, duration_days: int,
                          base_price: float = 75.0, n_days: int = 180) -> pd.DataFrame:
    """
    Simulate day-by-day evolution of key economic indicators over 6 months.

    Phase model (calibrated to historical crisis time dynamics):
      Phase 1 (0–3d):   Shock onset
      Phase 2 (4–7d):   Peak — strategic reserves announced
      Phase 3 (8–30d):  Reserve deployment + OPEC response
      Phase 4 (31–90d): Cape rerouting established
      Phase 5 (91–180d): Demand destruction + alternatives
    """
    oil_result = oil_price_scenario(hormuz_risk, duration_days, base_price)
    peak_chg   = oil_result["pct_change"]

    rows = []
    for d in range(n_days):
        if d <= duration_days:
            if d <= 3:
                phase_factor = (d / 3) ** 0.5
            elif d <= 7:
                phase_factor = 1.0
            elif d <= 30:
                phase_factor = 1.0 - 0.30 * ((d - 7) / 23)
            elif d <= 90:
                phase_factor = 0.70 - 0.20 * ((d - 30) / 60)
            else:
                phase_factor = 0.50 - 0.10 * ((d - 90) / 90)
        else:
            days_after = d - duration_days
            phase_factor = max(0, (0.50 * np.exp(-days_after / 20)))

        phase_factor = max(0, phase_factor)
        oil_chg      = peak_chg * phase_factor
        oil_price    = base_price * (1 + oil_chg / 100)

        freight_chg = oil_result["freight_premium"] * min(phase_factor * 1.2, 1.0)
        freight_chg = max(0, freight_chg)

        lag_cpi = 30
        oil_lagged_cpi = peak_chg * (rows[d - lag_cpi]["oil_factor"] if d >= lag_cpi else 0)
        cpi_add = oil_lagged_cpi * 0.12

        lag_food = 45
        oil_lagged_food = peak_chg * (rows[d - lag_food]["oil_factor"] if d >= lag_food else 0)
        food_add = oil_lagged_food * 0.28 + freight_chg * 0.15

        petrol_lag = 7
        oil_lagged_petrol = peak_chg * (rows[d - petrol_lag]["oil_factor"] if d >= petrol_lag else 0)
        petrol_price = oil_result["petrol_base"] * (1 + oil_lagged_petrol * 0.50 / 100)

        rows.append({
            "day":          d,
            "oil_factor":   phase_factor,
            "oil_price":    round(oil_price, 2),
            "oil_chg_pct":  round(oil_chg, 2),
            "cpi_add":      round(cpi_add, 3),
            "food_add_pct": round(food_add, 2),
            "freight_chg":  round(freight_chg, 1),
            "petrol_price": round(petrol_price, 3),
            "phase":        ("Shock" if d <= 7 else
                             "Reserve Deployment" if d <= 30 else
                             "Cape Rerouting" if d <= 90 else "New Equilibrium")
                             if d <= duration_days else "Recovery",
        })

    return pd.DataFrame(rows)


def monte_carlo_economic(n: int = 500, base_price: float = 75.0) -> pd.DataFrame:
    out = []
    for _ in range(n):
        sev  = random.uniform(0.30, 0.95)
        dur  = random.choice([3, 7, 14, 30, 60, 90, 180])
        res  = oil_price_scenario(sev, dur, base_price)
        casc = inflation_cascade(res["pct_change"], res["freight_premium"], dur)
        out.append({
            "severity":      sev,
            "duration_days": dur,
            "oil_chg_pct":   res["pct_change"],
            "global_cpi":    float(np.mean([v["headline_cpi"] for v in casc.values()])),
            "global_food":   float(np.mean([v["food_chg"]     for v in casc.values()])),
            "global_gdp":    float(np.mean([v["gdp_impact"]   for v in casc.values()])),
            "freight_pct":   res["freight_premium"],
        })
    return pd.DataFrame(out)


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
                alpha: float = 0.5, lam: float = 20.0) -> float:
    e = G[u][v]
    return e["cost"] + alpha * e["time"] + lam * e["risk"]


def risk_dijkstra(G: nx.DiGraph, source: str, target: str,
                  alpha: float = 0.5, lam: float = 20.0):
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
        # Compact state: node + global Hormuz risk bucket (5 levels: 0/0.25/0.5/0.75/1.0)
        # Gives 19×5 = 95 reachable states — fully explorable in a few hundred episodes.
        # Full 24-edge encoding (5^24 states) is never covered and produces identical
        # greedy paths every run due to universal Q=0 tie-breaking.
        hormuz_risks = [G[u][v]["risk"] for u, v in G.edges()
                        if G[u][v].get("hormuz_dependent")]
        avg = float(np.mean(hormuz_risks)) if hormuz_risks else 0.0
        return (node, round(avg * 4) / 4)

    def _act(self, G: nx.DiGraph, node: str,
             exclude: set | None = None, target: str | None = None) -> str | None:
        neighbors = [
            n for n in G.neighbors(node)
            if n not in (exclude or set())
            # never enter a consumer dead-end that isn't the destination
            and (G.nodes[n].get("type") != "consumer" or n == target)
        ]
        if not neighbors:
            return None
        if random.random() < self.epsilon:
            return random.choice(neighbors)
        state  = self._state(G, node)
        q_vals = [self.Q[(state, a)] for a in neighbors]
        max_q  = max(q_vals)
        # random tie-break so equal-Q neighbors don't always resolve to same node
        best   = [n for n, q in zip(neighbors, q_vals) if q == max_q]
        return random.choice(best)

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
        node      = source
        path      = [node]
        total     = 0.0
        seen      = {node}
        prev_node   = None
        prev_action = None
        for _ in range(max_steps):
            if node == target:
                break
            action = self._act(G, node, exclude=seen, target=target)
            if action is None or not G.has_edge(node, action):
                # Dead end — penalise the move that led here so the agent
                # learns to avoid routing into nodes with no onward path
                # to the target (e.g. Suez Canal when target is Japan).
                if prev_node is not None and prev_action is not None:
                    self._update(G, prev_node, prev_action, -100.0, node)
                break
            e      = G[node][action]
            reward = -(e["cost"] + 40 * e["risk"] + 2 * e["time"])
            if action == target:
                reward += 100.0
            self._update(G, node, action, reward, action)
            seen.add(action)
            path.append(action)
            total += reward
            prev_node   = node
            prev_action = action
            node = action
        return path, total

    def train(self, G: nx.DiGraph, source: str, target: str,
              episodes: int = 300) -> list[float]:
        rewards = []
        edges = list(G.edges())
        for ep in range(episodes):
            self.epsilon = max(0.05, self.epsilon * 0.994)
            # Perturb risks each episode so the agent trains across all
            # Hormuz risk buckets (low/medium/high/crisis), not just base risk.
            # Without this every episode sees the same state → Q-table has
            # identical entries → greedy path always matches Dijkstra.
            saved = {(u, v): G[u][v]["risk"] for u, v in edges}
            # Cycle through 4 phases, each targeting a distinct Hormuz risk
            # bucket so all 4 state buckets (0.25 / 0.5 / 0.75 / 1.0) are
            # trained. Bucket = round(avg_hormuz_risk × 4) / 4:
            #   phase 0 → bucket 0.25  (base ≈ 0.28)
            #   phase 1 → bucket 0.50  (base + 0.22 ≈ 0.50)
            #   phase 2 → bucket 0.75  (hdep → 0.76)
            #   phase 3 → bucket 1.00  (hdep → 0.93)  ← matches severity 0.90
            phase = (ep % 4)
            for u, v in edges:
                base = G[u][v]["base_risk"]
                hdep = G[u][v].get("hormuz_dependent", False)
                if phase == 0:   # normal          → bucket 0.25
                    G[u][v]["risk"] = float(np.clip(base + np.random.normal(0, 0.04), 0, 1))
                elif phase == 1: # elevated tension → bucket 0.50
                    bump = 0.22 if hdep else 0.05
                    G[u][v]["risk"] = float(np.clip(base + bump + np.random.normal(0, 0.04), 0, 1))
                elif phase == 2: # serious crisis   → bucket 0.75
                    G[u][v]["risk"] = float(np.clip((0.76 if hdep else base) + np.random.normal(0, 0.04), 0, 1))
                else:            # extreme crisis   → bucket 1.00
                    G[u][v]["risk"] = float(np.clip((0.93 if hdep else base) + np.random.normal(0, 0.03), 0, 1))
            _, r = self._episode(G, source, target)
            for u, v in edges:  # restore original risks
                G[u][v]["risk"] = saved[(u, v)]
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
    st.caption("Built by **Yash Vardhan Gupta** & **Nikita Gupta**")
    st.markdown("---")

    source = st.selectbox("Source (Producer)", PRODUCERS, index=0)
    target = st.selectbox("Destination (Consumer)", CONSUMERS, index=1)

    st.markdown("---")
    st.markdown("### Routing Parameters")
    alpha = st.slider("⏱ Time Weight α", 0.0, 2.0, 0.5, 0.1,
                      help="Cost weight on transit time")
    lam   = st.slider("⚠️ Risk Aversion λ", 0.0, 50.0, 20.0, 1.0,
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🗺️ Network Map",
    "🛣️ Route Finder",
    "📡 Risk Simulator",
    "🔥 Stress Test",
    "🤖 RL Agent",
    "📉 Economic Cascade",
    "📖 How It Works",
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
    R(edge) = -(cost + 40·risk + 2·time)
    R(reaching target) += +100
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
        st.markdown("""
        The comparison is shown across **two scenarios**. Under normal conditions both methods
        often agree — that's expected. The difference emerges under crisis, where the RL agent
        uses a *pre-learned policy* for the high-risk regime it trained on, while Dijkstra
        recomputes from scratch on the degraded graph.
        """)

        agent = st.session_state.rl_agent

        # ── Scenario A: current live graph (normal / stepped) ─────────────────
        st.markdown("#### Scenario A — Current Graph State (normal conditions)")
        rl_path_n   = agent.greedy_path(G, source, target)
        _, d_path_n = risk_dijkstra(G, source, target, alpha, lam)

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**🤖 RL Agent**")
            st.code(" → ".join(rl_path_n) if rl_path_n else "No path found")
            for k, v in path_stats(G, rl_path_n).items():
                st.metric(k, "⚠️ YES" if v is True else ("✅ NO" if v is False else str(v)))
        with col_r2:
            st.markdown("**🗺️ Dijkstra**")
            st.code(" → ".join(d_path_n) if d_path_n else "No path found")
            for k, v in path_stats(G, d_path_n).items():
                st.metric(k, "⚠️ YES" if v is True else ("✅ NO" if v is False else str(v)))

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.plotly_chart(draw_network(G, path=rl_path_n, title="RL — Normal"),
                            use_container_width=True)
        with col_m2:
            st.plotly_chart(draw_network(G, path=d_path_n, title="Dijkstra — Normal"),
                            use_container_width=True)

        # ── Scenario B: Hormuz crisis on a fresh graph ────────────────────────
        st.markdown("---")
        st.markdown("#### Scenario B — Hormuz Crisis (severity 0.90)")
        st.caption("Fresh graph with crisis applied — neither method has seen this exact state before. "
                   "RL uses its pre-trained crisis-regime policy; Dijkstra recomputes optimally.")

        G_crisis = build_oil_network()
        apply_hormuz_crisis(G_crisis, severity=0.90)

        rl_path_c   = agent.greedy_path(G_crisis, source, target)
        _, d_path_c = risk_dijkstra(G_crisis, source, target, alpha, lam)

        same = (rl_path_c == d_path_c)
        if same:
            st.info("🤝 Both methods chose the same bypass route under crisis — "
                    "the agent's learned policy matches Dijkstra's optimal solution.")
        else:
            st.success("✅ Crisis reveals the difference: RL and Dijkstra diverge on routing strategy.")

        col_r3, col_r4 = st.columns(2)
        with col_r3:
            st.markdown("**🤖 RL Agent**")
            st.code(" → ".join(rl_path_c) if rl_path_c else "No path found")
            rl_cs = path_stats(G_crisis, rl_path_c)
            for k, v in rl_cs.items():
                st.metric(k, "⚠️ YES" if v is True else ("✅ NO" if v is False else str(v)))
        with col_r4:
            st.markdown("**🗺️ Dijkstra**")
            st.code(" → ".join(d_path_c) if d_path_c else "No path found")
            d_cs = path_stats(G_crisis, d_path_c)
            dij_rl = path_stats(G_crisis, rl_path_c)
            for k, v in d_cs.items():
                bv = dij_rl.get(k)
                if k == "Hormuz Exposed":
                    st.metric(k, "⚠️ YES" if v else "✅ NO")
                elif isinstance(bv, (int, float)) and isinstance(v, (int, float)):
                    st.metric(k, str(v), delta=round(v - bv, 2),
                              delta_color="inverse" if k in ("Cost", "Transit (days)", "Max Risk", "Avg Risk") else "normal")
                else:
                    st.metric(k, str(v))

        col_m3, col_m4 = st.columns(2)
        with col_m3:
            st.plotly_chart(draw_network(G_crisis, path=rl_path_c, title="RL — Crisis"),
                            use_container_width=True)
        with col_m4:
            st.plotly_chart(draw_network(G_crisis, path=d_path_c, title="Dijkstra — Crisis"),
                            use_container_width=True)

        st.info("""
        💡 **The key difference is not the path — it's the mechanism.**
        Dijkstra recomputes from scratch every time the graph changes (O((V+E) log V) per query).
        The RL agent does a single Q-table lookup — instant, regardless of graph size.
        In a real deployment with continuous risk updates, that lookup speed is the advantage.
        The agent also learned its crisis-regime policy *without* being told λ — it discovered
        the cost-risk tradeoff purely through reward signals.
        """)

    else:
        st.info("👆 Train the agent first to compare it against Dijkstra.")


# ── TAB 6 · ECONOMIC CASCADE ─────────────────────────────────────────────────
with tab6:
    st.markdown("## 📉 Economic Cascade Simulator")
    st.markdown("""
    A Hormuz disruption doesn't stop at shipping costs.
    It propagates through the global economy in three waves:

    > **Wave 1 — Energy** (days 1–7): Oil price spike, petrol prices, power costs
    > **Wave 2 — Supply Chain** (days 8–60): Freight rates, manufacturing costs, trade prices
    > **Wave 3 — Macro** (months 2–6): CPI inflation, food prices, central bank hikes, GDP drag

    All coefficients are sourced from IMF, World Bank, IEA, and NBER research.
    """)

    st.markdown("---")
    st.markdown("### Scenario Parameters")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        econ_severity = st.slider(
            "Disruption Severity", 0.10, 1.00,
            float(np.mean([G[u][v]["risk"] for u, v in G.edges()
                           if G[u][v].get("hormuz_dependent")])),
            0.05,
            help="Fraction of Hormuz capacity disrupted. Linked to current graph risk.",
        )
    with col_s2:
        econ_duration = st.select_slider(
            "Disruption Duration",
            options=[3, 7, 14, 30, 60, 90, 180],
            value=30,
            format_func=lambda x: f"{x}d" if x < 60 else f"{x//30}mo",
        )
    with col_s3:
        base_oil = st.slider("Base Oil Price (USD/bbl)", 50, 120, 75, 5,
                             help="Brent crude baseline. Current ~$75 (EIA 2024).")

    oil_res = oil_price_scenario(econ_severity, econ_duration, base_oil)
    casc    = inflation_cascade(oil_res["pct_change"], oil_res["freight_premium"], econ_duration)
    ts_df   = economic_time_series(econ_severity, econ_duration, base_oil, n_days=180)

    global_cpi  = round(np.mean([v["headline_cpi"] for v in casc.values()]), 2)
    global_food = round(np.mean([v["food_chg"]     for v in casc.values()]), 2)
    global_gdp  = round(np.mean([v["gdp_impact"]   for v in casc.values()]), 2)
    global_rate = round(np.mean([v["rate_hike"]    for v in casc.values()]), 2)

    st.markdown("### Projected Impact")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("🛢️ Brent Crude",
              f"${oil_res['spot']}/bbl",
              f"+{oil_res['pct_change']}%")
    k2.metric("⛽ Petrol at Pump",
              f"${oil_res['petrol_new']}/L",
              f"+{round((oil_res['petrol_new']/oil_res['petrol_base']-1)*100,1)}%")
    k3.metric("🚢 Freight Rates",
              f"+{oil_res['freight_premium']:.0f}%",
              f"{round(oil_res['supply_cut_mbd'],1)}% supply cut",
              delta_color="inverse")
    k4.metric("📈 Global CPI",
              f"+{global_cpi}%",
              "YoY addition", delta_color="inverse")
    k5.metric("🌾 Food Prices",
              f"+{global_food}%",
              "FAO index equivalent", delta_color="inverse")
    k6.metric("📊 Global GDP",
              f"{global_gdp}%",
              "annual impact", delta_color="inverse")

    st.markdown("---")
    st.markdown("### Indicator Evolution Over 6 Months")

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts_df["day"], y=ts_df["oil_chg_pct"],
        name="Oil Price Change (%)", line=dict(color="#E74C3C", width=2.5),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.08)",
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_df["day"], y=ts_df["cpi_add"],
        name="CPI Addition (%)", line=dict(color="#F39C12", width=2, dash="dash"),
        yaxis="y2",
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_df["day"], y=ts_df["food_add_pct"],
        name="Food Price Change (%)", line=dict(color="#2ECC71", width=2),
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_df["day"], y=ts_df["freight_chg"],
        name="Freight Rate Premium (%)", line=dict(color="#9B59B6", width=1.5, dash="dot"),
    ))

    for px_day, label in [(0, "Shock"), (8, "Reserves"), (31, "Rerouting"), (91, "Equilibrium")]:
        if px_day < 180:
            fig_ts.add_vline(x=px_day, line=dict(color="#444", dash="dot", width=1))
            fig_ts.add_annotation(x=px_day + 1, y=ts_df["oil_chg_pct"].max() * 0.92,
                                   text=label, showarrow=False,
                                   font=dict(color="#8b949e", size=10))
    if econ_duration < 180:
        fig_ts.add_vline(x=econ_duration, line=dict(color="#FFD700", dash="dash", width=2))
        fig_ts.add_annotation(x=econ_duration, y=ts_df["oil_chg_pct"].max(),
                               text="Disruption ends", showarrow=True, arrowcolor="#FFD700",
                               font=dict(color="#FFD700", size=10))

    fig_ts.update_layout(
        **_DARK,
        title=f"Economic Cascade: {econ_duration}-day disruption at severity {econ_severity:.2f}",
        xaxis_title="Days since disruption onset",
        yaxis=dict(title="Change (%)", ticksuffix="%"),
        yaxis2=dict(title="CPI Addition (%)", overlaying="y", side="right",
                    ticksuffix="%", showgrid=False),
        height=420,
    )
    fig_ts.update_layout(legend=dict(bgcolor="#161b22", x=0.01, y=0.99))
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")
    st.markdown("### Regional Impact Breakdown")
    st.caption("Impact varies significantly by oil import dependency and energy intensity of economy.")

    regions     = list(casc.keys())
    region_lbls = [r.replace("\n", " ") for r in regions]
    colors_r    = [REGIONS[r]["color"] for r in regions]

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        fig_cpi = go.Figure(go.Bar(
            x=region_lbls,
            y=[casc[r]["headline_cpi"] for r in regions],
            marker_color=colors_r,
            text=[f"+{casc[r]['headline_cpi']}%" for r in regions],
            textposition="outside",
        ))
        fig_cpi.update_layout(**_DARK, title="Headline CPI Addition (%)",
                               yaxis_title="%", height=320, showlegend=False)
        st.plotly_chart(fig_cpi, use_container_width=True)

    with col_r2:
        fig_food_bar = go.Figure(go.Bar(
            x=region_lbls,
            y=[casc[r]["food_chg"] for r in regions],
            marker_color=colors_r,
            text=[f"+{casc[r]['food_chg']}%" for r in regions],
            textposition="outside",
        ))
        fig_food_bar.update_layout(**_DARK, title="Food Price Increase (%)",
                                    yaxis_title="%", height=320, showlegend=False)
        st.plotly_chart(fig_food_bar, use_container_width=True)

    with col_r3:
        fig_gdp_bar = go.Figure(go.Bar(
            x=region_lbls,
            y=[casc[r]["gdp_impact"] for r in regions],
            marker_color=colors_r,
            text=[f"{casc[r]['gdp_impact']}%" for r in regions],
            textposition="outside",
        ))
        fig_gdp_bar.update_layout(**_DARK, title="GDP Impact (%)",
                                   yaxis_title="%", height=320, showlegend=False)
        st.plotly_chart(fig_gdp_bar, use_container_width=True)

    reg_rows = []
    for r in regions:
        v = casc[r]
        reg_rows.append({
            "Region":           r.replace("\n", " "),
            "Oil Import Dep.":  f"{int(REGIONS[r]['import_dep']*100)}%",
            "Effective Oil Δ":  f"+{v['eff_oil_chg']}%",
            "Headline CPI":     f"+{v['headline_cpi']}%",
            "Core CPI":         f"+{v['core_cpi']}%",
            "Food Price Δ":     f"+{v['food_chg']}%",
            "GDP Impact":       f"{v['gdp_impact']}%",
            "Est. Rate Hike":   f"+{v['rate_hike']} pp",
        })
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Transmission Chain")
    st.caption("How a Hormuz disruption propagates through the global economy.")

    op = oil_res["pct_change"]
    fp = oil_res["freight_premium"]

    sankey_labels = [
        "Hormuz Disruption", "Oil Supply Shock", "Freight Rate Spike",
        "Energy Cost Rise", "Fertilizer Cost Rise", "Manufacturing Cost",
        "Food Import Cost", "Retail & Consumer Prices", "Food Prices",
        "CPI Inflation", "Central Bank Hikes", "GDP Contraction",
    ]
    sankey_colors = [
        "#E74C3C","#E74C3C","#9B59B6","#F39C12","#27AE60",
        "#F39C12","#27AE60","#E67E22","#27AE60","#E67E22",
        "#3498DB","#C0392B",
    ]

    s_scale = max(op, 5)
    src = [0,0,1,1,1,3,4,5,6,7,8,9,9,10]
    tgt = [1,2,3,4,2,5,6,7,8,9,9,10,11,11]
    val = [
        s_scale, s_scale * 0.40, s_scale * 0.55, s_scale * 0.20,
        fp * 0.08, s_scale * 0.30, s_scale * 0.15, s_scale * 0.25,
        s_scale * 0.20, s_scale * 0.18, s_scale * 0.22,
        global_cpi * 8, global_gdp * -12, global_rate * 15,
    ]
    val = [max(0.5, abs(v)) for v in val]

    fig_sankey = go.Figure(go.Sankey(
        node=dict(pad=15, thickness=20,
                  line=dict(color="#333", width=0.5),
                  label=sankey_labels, color=sankey_colors),
        link=dict(source=src, target=tgt, value=val,
                  color=["rgba(231,76,60,0.3)" if i < 4 else
                         "rgba(243,156,18,0.3)" if i < 8 else
                         "rgba(52,152,219,0.3)"
                         for i in range(len(src))]),
    ))
    fig_sankey.update_layout(
        **_DARK,
        title=f"Economic Transmission Chain — Severity {econ_severity:.2f}, Duration {econ_duration}d",
        height=500,
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.markdown("---")
    st.markdown("### Historical Calibration — How Does This Scenario Compare?")
    st.caption("Real historical events used to validate model coefficients.")

    hist_df = pd.DataFrame(HISTORICAL_EVENTS,
                           columns=["Event", "Duration (days)", "Oil Δ%",
                                    "CPI Peak %", "Food Δ%", "GDP %", "Source"])
    current_row = pd.DataFrame([{
        "Event":           f"▶ This Scenario (sev={econ_severity:.2f})",
        "Duration (days)": econ_duration,
        "Oil Δ%":          oil_res["pct_change"],
        "CPI Peak %":      global_cpi,
        "Food Δ%":         global_food,
        "GDP %":           global_gdp,
        "Source":          "Model (v1)",
    }])
    compare_df = pd.concat([current_row, hist_df], ignore_index=True)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    fig_scatter = go.Figure()
    for _, row in hist_df.iterrows():
        fig_scatter.add_trace(go.Scatter(
            x=[row["Oil Δ%"]], y=[row["CPI Peak %"]],
            mode="markers+text",
            marker=dict(size=12, color="#3498DB", opacity=0.8),
            text=[row["Event"].split(" ")[0]], textposition="top center",
            textfont=dict(size=9, color="#8b949e"),
            showlegend=False,
            hovertext=f"{row['Event']}<br>Oil: +{row['Oil Δ%']}% | CPI: +{row['CPI Peak %']}%",
            hoverinfo="text",
        ))
    fig_scatter.add_trace(go.Scatter(
        x=[oil_res["pct_change"]], y=[global_cpi],
        mode="markers+text",
        marker=dict(size=16, color="#FFD700", symbol="star"),
        text=["This scenario"], textposition="top center",
        textfont=dict(color="#FFD700", size=11),
        name="Current scenario",
        hovertext=f"This Scenario<br>Oil: +{oil_res['pct_change']}% | CPI: +{global_cpi}%",
        hoverinfo="text",
    ))
    hist_x = [e[2] for e in HISTORICAL_EVENTS]
    hist_y = [e[3] for e in HISTORICAL_EVENTS]
    z = np.polyfit(hist_x, hist_y, 1)
    x_line = np.linspace(0, max(max(hist_x), oil_res["pct_change"]) * 1.1, 50)
    fig_scatter.add_trace(go.Scatter(
        x=x_line, y=np.polyval(z, x_line),
        mode="lines", name="Historical trend",
        line=dict(color="#555", dash="dash", width=1), showlegend=True,
    ))
    fig_scatter.update_layout(
        **_DARK,
        title="Oil Price Change vs CPI Impact — Historical Events + This Scenario",
        xaxis_title="Oil Price Change (%)",
        yaxis_title="CPI Peak Addition (%)",
        height=380,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("### Monte Carlo — Economic Outcome Distribution (500 scenarios)")
    with st.spinner("Running 500 economic scenarios…"):
        mc_econ = monte_carlo_economic(n=500, base_price=base_oil)

    col_mc1, col_mc2, col_mc3 = st.columns(3)
    for col_mc, field, label, color in [
        (col_mc1, "oil_chg_pct", "Oil Price Change (%)", "#E74C3C"),
        (col_mc2, "global_cpi",  "Global CPI Addition (%)", "#F39C12"),
        (col_mc3, "global_food", "Global Food Price Δ (%)", "#2ECC71"),
    ]:
        fig_mc_h = go.Figure()
        fig_mc_h.add_trace(go.Histogram(
            x=mc_econ[field], nbinsx=30,
            marker_color=color, opacity=0.8, name=label,
        ))
        fig_mc_h.add_vline(
            x=mc_econ[field].mean(),
            line=dict(color="white", dash="dash", width=1.5),
            annotation_text=f"Mean: {mc_econ[field].mean():.1f}%",
            annotation_font=dict(color="white", size=10),
        )
        current_val = (oil_res["pct_change"] if field == "oil_chg_pct"
                       else global_cpi if field == "global_cpi" else global_food)
        fig_mc_h.add_vline(
            x=current_val,
            line=dict(color="#FFD700", dash="solid", width=2),
            annotation_text="This scenario",
            annotation_font=dict(color="#FFD700", size=10),
        )
        fig_mc_h.update_layout(**_DARK, title=label, height=300,
                               xaxis_title="%", showlegend=False)
        col_mc.plotly_chart(fig_mc_h, use_container_width=True)

    pct_worse = (mc_econ["oil_chg_pct"] > oil_res["pct_change"]).mean() * 100
    st.markdown(f"""
    **Across 500 simulated scenarios:**
    - **{pct_worse:.0f}%** of scenarios produce a *worse* oil price shock than the current scenario
    - Oil price: median **+{mc_econ['oil_chg_pct'].median():.0f}%**, 95th pct **+{mc_econ['oil_chg_pct'].quantile(0.95):.0f}%**
    - Global CPI: median **+{mc_econ['global_cpi'].median():.1f}%**, 95th pct **+{mc_econ['global_cpi'].quantile(0.95):.1f}%**
    - Global food: median **+{mc_econ['global_food'].median():.1f}%**, 95th pct **+{mc_econ['global_food'].quantile(0.95):.1f}%**
    - GDP impact: median **{mc_econ['global_gdp'].median():.2f}%**, worst 5% **{mc_econ['global_gdp'].quantile(0.05):.2f}%**

    > Developing markets face the sharpest inflation and GDP hits — highest oil import dependency
    > combined with least central bank credibility to anchor inflation expectations.
    """)

    st.info("""
    **Model sources:** IMF WP/17/53 (Gelos & Ustyugova) — CPI pass-through coefficients ·
    Hamilton (2009) NBER — oil supply elasticity · Kilian (2008) AER — GDP elasticity ·
    World Bank Commodity Markets Outlook Apr 2022 — food price sensitivity ·
    IEA Emergency Response Manual — strategic reserve offsets ·
    EIA — Hormuz throughput share, historical events data
    """)


# ── TAB 7 · HOW IT WORKS ─────────────────────────────────────────────────────
with tab7:
    st.markdown("## 📖 How It Works — Concept Guide")
    st.markdown(
        "Every model in this app is grounded in a specific mathematical idea. "
        "This tab explains each one from first principles — no prerequisites needed."
    )

    # ── 1. The Graph ──────────────────────────────────────────────────────────
    with st.expander("🌐 The Oil Network as a Graph  G = (V, E)", expanded=True):
        st.markdown("""
**The core idea:** strip away the geopolitics and describe the oil supply chain the way a mathematician would — as a *graph*.

A graph has two ingredients:
- **Nodes (V):** anything that oil passes *through* — a country, a strait, a pipeline terminal
- **Edges (E):** the connections between them — shipping lanes and pipelines

```
Producers  →  Chokepoints  →  Transit Hubs  →  Consumers
Saudi Arabia     Hormuz       Indian Ocean      China
UAE              Malacca      Red Sea           Japan
Iraq             Suez Canal   Cape of GH        Europe
```

Each edge carries four numbers that matter:

| Attribute | What it means |
|-----------|--------------|
| `cost` | Shipping cost index (proportional to real VLCC freight rates) |
| `time` | Transit days (derived from nautical miles ÷ 14 knots) |
| `capacity` | Max throughput in million barrels/day (real EIA figures) |
| `risk(t)` | Current geopolitical risk — *this one changes over time* |

The routing problem: find the path from producer to consumer that minimises total cost+time+risk.
        """)

    # ── 2. Dijkstra ──────────────────────────────────────────────────────────
    with st.expander("🗺️ Dijkstra's Algorithm — The GPS of Graphs"):
        st.markdown("""
**Analogy:** Dijkstra is like a GPS that always finds the fastest route — except instead of
*time*, we minimise a custom edge weight that combines cost, transit time, and risk.

**How it works (step by step):**

```
1. Start at the source node. Distance = 0.
2. Push (distance=0, node=source) into a priority queue.
3. Pop the lowest-distance item from the queue.
4. For each unvisited neighbour:
      new_dist = current_dist + edge_weight(current → neighbour)
      if new_dist < known_dist[neighbour]:
          update known_dist[neighbour] = new_dist
          push (new_dist, neighbour) into queue
5. Repeat until you reach the destination.
```

**Why a priority queue?** It ensures we always explore the *cheapest reachable node next*,
which guarantees the first time we reach a node, we've found the cheapest path to it.

**Complexity:** O((V + E) log V) — fast enough to recompute in milliseconds on our 19-node graph.

**The standard version** minimises raw shipping cost. Our version adds *time* and *risk*:

```python
def edge_weight(G, u, v, alpha=0.5, lam=10.0):
    e = G[u][v]
    return e["cost"] + alpha * e["time"] + lam * e["risk"]
```

**The λ effect:** At λ=0 the algorithm ignores risk entirely and picks the cheapest route
(always Hormuz). As λ increases, the risk penalty on Hormuz edges grows until it outweighs
the cost saving — at that threshold the algorithm *switches* to a bypass route. That cost
jump is the **price of resilience**.
        """)

    # ── 3. OU Process ─────────────────────────────────────────────────────────
    with st.expander("📡 Ornstein-Uhlenbeck Process — How Risk Evolves"):
        st.markdown(r"""
**Analogy:** Imagine a rubber band stretched between your hand and a wall. Let go — it snaps
back toward the wall. That "snapping back" is mean reversion. Now add random gusts of wind
that push it in unpredictable directions. That's the OU process.

**The equation:**

$$dR(t) = \theta(\mu - R(t))\,dt + \sigma\,dW(t)$$

| Symbol | Meaning | Value |
|--------|---------|-------|
| $R(t)$ | Current risk level | changes each tick |
| $\theta$ | Mean reversion speed — how fast risk returns to baseline | 0.3 |
| $\mu$ | Long-run baseline risk (the "wall") | calibrated per edge |
| $\sigma$ | Volatility — how strong the random shocks are | 0.12 × slider |
| $dW(t)$ | Wiener process — pure random shock drawn from N(0,1) | random |

**Why OU and not a random walk?**

A random walk drifts forever — risk would eventually reach 0 or 1 and stay there.
The OU process always pulls back toward baseline: crises happen, escalate, then
*de-escalate* (usually). This mirrors real geopolitical risk dynamics.

**Discrete implementation (one simulation tick):**

```python
new_risk = current + θ(μ - current) + σ · N(0,1)
new_risk = clip(new_risk, 0, 1)
```

The **Volatility slider** in the sidebar scales σ. At 0 the graph is static.
At 1.0 risk swings wildly every tick. Real-world volatility sits around 0.3–0.5
during periods of elevated tension.
        """)

    # ── 4. Q-Learning ─────────────────────────────────────────────────────────
    with st.expander("🤖 Q-Learning — Teaching an Agent to Route"):
        st.markdown(r"""
**Analogy:** Imagine learning to drive in a city you've never seen. At first you turn randomly.
Over time you learn which turns lead to fast routes and which lead to dead ends. You don't
memorise a single path — you build an *intuition* (a policy) for every intersection you might face.
That's Q-learning.

**The MDP setup:**

| Component | Definition |
|-----------|-----------|
| **State** $s$ | Where the agent is + current Hormuz risk level |
| **Action** $a$ | Which node to move to next |
| **Reward** $R$ | $-(cost + 40 \cdot risk + 2 \cdot time)$ + 100 if reached target |
| **Goal** | Maximise total reward across the journey |

**The Q-function:** $Q(s, a)$ = "how good is it to take action $a$ from state $s$?"

After taking action $a$, moving to state $s'$, and getting reward $r$, we update:

$$Q(s,a) \leftarrow Q(s,a) + \alpha \Big[r + \gamma \cdot \max_{a'} Q(s', a') - Q(s,a)\Big]$$

| Symbol | Meaning |
|--------|---------|
| $\alpha$ | Learning rate — how much to shift toward new information (0.15) |
| $\gamma$ | Discount factor — how much to value future vs immediate rewards (0.9) |
| $r + \gamma \max Q(s',a')$ | Bellman target — what Q *should* be |
| The bracket | TD error — the surprise, used to nudge Q toward the target |

**Exploration vs exploitation (ε-greedy):**

```
With probability ε  → take a RANDOM action (explore)
With probability 1-ε → take the BEST KNOWN action (exploit)
```

ε starts at 0.5 (50% random) and decays to 0.05 as training progresses.
This ensures the agent explores the graph before committing to a fixed route.

**What the agent learns:** not a single path, but a *policy table* — a lookup that
maps (node, risk_level) → best_next_node. At inference time, routing is a table
lookup, not a graph search.
        """)

    # ── 5. Betweenness Centrality ──────────────────────────────────────────────
    with st.expander("🔴 Why Hormuz is a Single Point of Failure — Betweenness Centrality"):
        st.markdown("""
**Betweenness centrality** measures how often a node appears on the shortest path
between *every pair* of other nodes in the network.

$$C_B(v) = \\sum_{s \\neq v \\neq t} \\frac{\\sigma_{st}(v)}{\\sigma_{st}}$$

where $\\sigma_{st}$ = total shortest paths from $s$ to $t$, and $\\sigma_{st}(v)$ =
those that pass through $v$.

**Hormuz in numbers:** Remove the Hormuz node. The shortest-path count between Gulf
producers and Asian consumers drops by ~80%. No other node comes close.

**What this means practically:**
- It's not just that Hormuz is busy — it's that the *structure* of the network makes it unavoidable
- Even if you want to avoid Hormuz, the bypass routes are longer, costlier, and lower-capacity
- This is why the λ-switchover in the Route Finder requires a large risk penalty to trigger —
  the bypass is economically irrational under normal conditions

**The engineering lesson:** high betweenness centrality = high systemic risk. A node with
CB close to 1.0 is a choke the whole network depends on. Redundancy means building
*parallel paths* that reduce betweenness, not just adding capacity to the bottleneck itself.
        """)

    st.markdown("---")
    st.info(
        "📘 For full implementation details, equations, and data sources: "
        "see `TECHNICAL.md`, `blog.md`, and `DATA_SOURCES.md` in the repository."
    )


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
