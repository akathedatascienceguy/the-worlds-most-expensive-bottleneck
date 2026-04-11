# Data Sources & Calibration

Every numeric parameter in the graph model is either directly sourced from a public dataset or derived from one. This document records the source, year, and derivation method for each parameter type.

---

## 1. Chokepoint Throughput Capacities

| Chokepoint | Value (MBD) | Year | Source |
|------------|-------------|------|--------|
| Strait of Hormuz | 20.0 | 2024 | U.S. EIA — *World Oil Transit Chokepoints*, eia.gov/todayinenergy/detail.php?id=65504 |
| Strait of Malacca | 16.6 (crude) | H1 2025 | U.S. EIA chokepoints analysis |
| Suez Canal (oil tankers) | 7.5 normal / 3.9 disrupted | 2023 / 2024 | IEA Suez Canal Factsheet; EIA Red Sea analysis |
| Bab-el-Mandeb | 8.8 pre-Houthi / 4.2 Q1-2025 | 2023 / 2025 | EIA Red Sea Chokepoints — eia.gov/todayinenergy/detail.php?id=61025 |
| SUMED Pipeline (Egypt) | 2.5 | 2023 | EIA SUMED analysis — eia.gov/todayinenergy/detail.php?id=40152 |
| Cape of Good Hope | No ceiling (open ocean) | — | EIA; modelled as 25 MBD (tanker fleet upper bound) |

**Note on Suez capacity:** The model uses 7.5 MBD (nameplate normal capacity) as the `base` edge capacity. The risk engine depresses effective throughput dynamically when `bab_el_mandeb_risk` rises — mirroring the 50% throughput drop observed during the 2024 Houthi disruption.

---

## 2. Bypass Pipeline Capacities

| Pipeline | Capacity (MBD) | Notes | Source |
|----------|----------------|-------|--------|
| Saudi East-West (Petroline → Yanbu) | 7.0 | Upgraded March 2026 (prev. nameplate 5.0 MBD) | Fortune / Argus Media / CNBC, March 28 2026; CEO Amin Nasser confirmation |
| UAE ADCO (Habshan → Fujairah) | 1.5 (expandable to 1.8) | 360 km, 48-inch pipe; operational since June 2012 | Wikipedia / Global Energy Monitor / ADNOC |

Both apps use **7.0 MBD** for Yanbu, reflecting the confirmed March 2026 upgrade. The previous 5.0 MBD nameplate is no longer the operative capacity.

---

## 3. Producer Export Volumes

All figures are actual measured exports (crude + condensate), not production capacity.

| Country | Export Volume (MBD) | Period | Source |
|---------|---------------------|--------|--------|
| Saudi Arabia | 6.05 | December 2024 | CEIC Data / FRED St. Louis Fed |
| Saudi Arabia | 6.66 | December 2023 | CEIC Data / Statista |
| UAE | 2.72 | December 2024 | CEIC Data |
| UAE | 2.65 | December 2023 | CEIC Data |
| Iraq | 3.37 | 2024 average | Iraq Oil Report / EIA |
| Iraq | 3.47 | December 2023 | CEIC Data |
| Kuwait | 1.57 | December 2023 | CEIC Data / EIA |
| Qatar (crude only) | 0.51 | 2024 projection | Statista / EIA |
| Qatar (crude + condensate) | 1.22 | 2024 | Statista |

**Model usage:** Edge capacity from producer → Hormuz is set to the December 2024 crude export figure (most recent). These represent real throughput, not nameplate capacity.

---

## 4. Transit Times

Derived from real nautical distances and a standard VLCC laden speed of **13–14 knots**.

| Leg | Distance (nm) | Speed (kn) | Transit Time | Source / Validation |
|-----|--------------|------------|--------------|---------------------|
| Persian Gulf → Hormuz | ~250 | 14 | ~0.8 days | Standard maritime refs; port-to-strait distance |
| Hormuz → India (Mumbai) | ~1,500 | 14 | ~2 days | Zee News / Sunday Guardian maritime analysis; 53-hour typical |
| Hormuz → Strait of Malacca | ~6,600 | 14 | ~20 days | Maritime Executive; confirmed by shipping broker schedules |
| Hormuz → Suez Canal (via IOH) | ~3,900 | 14 | ~12 days | EIA routing data; Strauss Center analysis |
| Suez Canal → NW Europe (Rotterdam) | ~3,000 | 14 | ~8 days | MyDello Suez Canal Shipping Guide |
| Hormuz → Cape → Rotterdam | ~15,500 total | 14–15 | ~34 days | S&P Global Specifications Guide; 31d 1h calculated at 15kn |
| Cape → US Gulf Coast | Cape leg + ~4,500 | 14 | ~19 days | EIA; Quora maritime references |
| Red Sea → Bab-el-Mandeb | ~700 | 14 | ~2 days | Standard Red Sea routing |
| Bab-el-Mandeb → Suez Canal | ~1,200 | 14 | ~3.5 days | Suez Canal Authority transit data |
| Malacca → China (Ningbo) | ~1,600 | 14 | ~5 days | Port distance tables |
| Malacca → Japan (Yokohama) | ~2,800 | 14 | ~8 days | Port distance tables |

---

## 5. War-Risk Insurance Premiums → Risk Calibration

The `base_risk` field on each edge is calibrated to the **war-risk insurance premium band** for that route segment. The mapping:

| Premium Band (% hull value) | Model Risk Score | Interpretation |
|-----------------------------|-----------------|----------------|
| ~0.01% (open ocean baseline) | 0.01–0.02 | Cape of Good Hope, open Indian Ocean |
| ~0.05–0.10% (low tension) | 0.05–0.09 | Strait of Malacca (post-May 2024 JWC delisting), pipelines |
| ~0.10–0.20% (elevated) | 0.12–0.18 | Suez Canal baseline, Red Sea pipeline routes |
| ~0.25–0.50% (Hormuz baseline) | 0.25–0.30 | Gulf/Hormuz standard (S&P Global 2024) |
| ~0.70–1.00% (active tension) | 0.35–0.45 | Bab-el-Mandeb 2024 (down from 2.0% peak) |
| ~1.00–2.00% (crisis) | 0.60–0.85 | Hormuz 2019 tanker attacks; Hormuz 2026 escalation |
| ~2.00%+ (war) | 0.85–1.00 | Bab-el-Mandeb Nov–Dec 2023 peak |

**Premium sources:**
- Hormuz baseline 0.25% hull: S&P Global / Lloyd's, 2024
- Hormuz 2019 attacks 0.50% hull: Strauss Center comprehensive analysis
- Hormuz 2026 escalation 1.00%: Lloyd's List, March 2026; Caixin Global
- Bab-el-Mandeb pre-Houthi 0.05%: Policyholder Pulse
- Bab-el-Mandeb 2024 peak 2.00% (2,700% rise from baseline): AGBI / Maplecroft / Argus Media
- Malacca 2024–25 ~0.05%: Lloyd's Joint War Committee (removed from high-risk list May 2024)

**Crisis simulation:** The `apply_hormuz_crisis(severity)` function sets Hormuz-dependent edge risks to `severity`. A severity of 0.90 corresponds approximately to the 2026 escalation scenario (~1.0% hull premium). A severity of 0.35 corresponds to the 2024 Bab-el-Mandeb elevated state.

---

## 6. Freight Rate → Cost Index Calibration

The `cost` column in the edge table is a **normalised index** where 1 unit ≈ USD 0.30–0.40 per barrel, derived from 2023 VLCC market rates.

### Source data

| Route | Worldscale (2023 range) | $/barrel (approx) | Source |
|-------|------------------------|-------------------|--------|
| AG → Japan (TD3C) | WS 50–90 | $1.60–$3.20 | Signal Group 2023 Tanker Annual Review; Baltic Exchange archives |
| TD3C peak (March 2023) | ~WS 90 | ~$3.20 | Signal Group |
| TD3C Dec 2023 | WS 50–60 | ~$1.80–2.00 | Signal Group / Breakwave Advisors |
| AG → NW Europe | Not publicly available 2023 | Estimated +15% vs TD3C | — |
| AG → US Gulf | Not publicly available 2023 | Estimated +20% vs TD3C | — |

**Note on data gaps:** Specific route-level 2023 average freight rates for AG→Europe and AG→USGC are proprietary (Clarksons, Fearnleys, OPEC MMR). The cost indices for those legs are estimated by distance-scaling from TD3C and cross-checking against EIA shipping cost analysis.

### Conversion method

```
WS mid-2023 average for TD3C ≈ WS 70
WS 70 for a VLCC carrying 2M barrels ≈ USD 2.00–2.50 / barrel
cost_index = round(USD_per_bbl / 0.35)

Examples:
  Gulf internal leg ($0.30/bbl)  → cost_index = 1
  Hormuz → Indian Ocean ($0.40)  → cost_index = 1
  IOH → India ($1.40)            → cost_index = 4
  IOH → Malacca ($2.80)          → cost_index = 8
  Cape → Europe ($3.50)          → cost_index = 10
```

---

## 7. What Remains Estimated

The following parameters have no freely available primary source and are modelled estimates:

| Parameter | Status | Notes |
|-----------|--------|-------|
| AG → NW Europe freight rate (2023 avg) | Estimated | Proprietary; scaled from TD3C |
| AG → US Gulf freight rate (2023 avg) | Estimated | Proprietary; scaled from TD3C |
| Cape → US Gulf exact distance | Estimated | Combined sources suggest 19 days; no single primary source isolated |
| IOH → Cape exact nm | Estimated at 6,000nm | Derived from total route distance minus sub-legs |
| Per-producer Hormuz transit costs | Estimated | Allocated proportionally to distance and port fees |

---

## 8. Primary Sources Index

| Source | URL / Reference | Used For |
|--------|----------------|----------|
| U.S. EIA Chokepoints Report | eia.gov/todayinenergy/detail.php?id=65504 | Hormuz, Malacca throughput |
| U.S. EIA Red Sea Chokepoints | eia.gov/todayinenergy/detail.php?id=61025 | Bab-el-Mandeb, SUMED |
| IEA Suez Canal Factsheet | iea.blob.core.windows.net/assets/0a3c8da7.../SuezCanal-Factsheet.pdf | Suez capacity |
| Fortune / Argus Media (March 2026) | Saudi Petroline 7 MBD milestone | Yanbu pipeline capacity |
| Global Energy Monitor | globalenergymonitor.org | Fujairah pipeline specs |
| CEIC Data | ceicdata.com | Crude export volumes |
| FRED St. Louis Fed | fred.stlouisfed.org | Saudi export series |
| Signal Group 2023 Tanker Review | thesignalgroup.com/newsroom/tanker-annual-review-2023 | TD3C freight rates |
| MyDello Suez Guide | mydello.com/suez-canal-shipping | Transit times |
| S&P Global Shipping | spglobal.com/energy | War-risk insurance, Cape distances |
| Lloyd's List (March 2026) | lloydslist.com | Gulf war-risk premium spike |
| Strauss Center (UT Austin) | strausscenter.org | Comprehensive Hormuz insurance analysis |
| Maplecroft / AGBI | agbi.com | Bab-el-Mandeb premium data |
| Maritime Executive | maritime-executive.com | Malacca transit times |
| Iraq Oil Report | iraqoilreport.com | Iraq export volumes |
| Policyholder Pulse | — | Bab-el-Mandeb baseline premium |
