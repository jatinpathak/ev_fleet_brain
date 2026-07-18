# 🔋 EV Fleet Intelligence Brain — v3

**ET AI Hackathon 2026 · Problem 3 — AI for Industrial EV Supply Chain & Asset Intelligence**

A single decision-support platform that helps a commercial fleet operator
electrify a diesel/petrol fleet **intelligently, not all at once** — and manage
the battery **supply chain, assets, maintenance and carbon** around it. Seven
domain engines share one data layer, are coordinated by a **bounded multi-agent
planner**, stress-tested by a **scenario simulator**, and shown on one
executive-grade Streamlit dashboard — with a REST API and Docker for integration.

| Engine | Tier | Question it answers |
| --- | --- | --- |
| 🔋 **Battery Health & RUL** | 1 | How healthy is each cell, how many cycles remain (with a confidence interval), which degrade abnormally fast — plus a battery passport. |
| 🚚 **Fleet Readiness** | 1 | Which vehicles switch to EV first, how confident are we, and what's the 5-year TCO? |
| 🔗 **Supply-Chain Risk** | 1 | Where is the battery-material supply fragile, what ₹ is exposed, and what happens if a node is disrupted? |
| 🛠️ **Maintenance & Charging** | 2 (MVP) | How do we schedule maintenance and overnight charging to cut downtime and cost? |
| 🌱 **Carbon Impact** | 2 | Scope 1/2/3 CO₂, the ₹ credit value, and the smart-charging (hourly-grid) opportunity. |
| 🧪 **Scenario Simulation** | ⭐ | What-if: supplier disruption, accelerated degradation, tariff change, fleet expansion. |
| 🛰️ **Digital Twin / Quality** | 3 (MVP) | Unified live fleet-state view + light manufacturing-quality (SPC + root-cause). |
| 🧠 **Multi-agent planner (copilot)** | 1 | Decomposes a question, calls the right agents (loop-guarded), and shows its reasoning. |

> **Honesty first.** Every simplified module carries an **MVP / illustrative**
> badge in the UI. The battery accuracy headline comes from real-data-shaped
> cycling; every other dataset is clearly labelled **synthetic**. See
> [Honest disclosure](#honest-disclosure).

---

## What's new in v3 (final polish — Innovation + UX)

Building on v2's audited 6-engine core, v3 adds the innovation and
user-experience layer, and a cheap-but-credible integration story. See
[`DEMO_QA.md`](DEMO_QA.md) for judge-question answers.

### Roadmap → Resolution map

| Roadmap item | Resolved by | Depth |
| --- | --- | --- |
| True multi-agent planner | `core/orchestrator.py` bounded planner + **visible plan trace** | Full |
| Executive Command Center | new home page + **Today's Highlights** + alert centre | Full |
| Scenario simulation ⭐ | `engines/engine_scenario.py` (4 scenarios) + Scenario Lab page | Full |
| Richer AI recommendations | `core/recommend.py`: confidence + reasoning + ₹/CO₂ impact + alternatives on every engine | Full |
| Supplier knowledge graph + propagation | `engine_supply_chain.propagate_risk()` + interactive graph | Full |
| Manufacturing quality + RCA + traceability | `engine_quality.root_cause_hints()` | MVP (labelled) |
| Battery Passport | `engine_battery.battery_passport()` + panel | Full |
| Hourly grid emission factors | `engine_carbon` 24h profile + smart-charging | Full |
| Geospatial maps | fleet / charging-depot / supplier maps | Full |
| Digital Twin | unified asset view | MVP (labelled) |
| Monitoring stub (drift) | `core/monitoring.py` (PSI) + System page | Full (stub) |
| Containerisation + REST API | `Dockerfile`, `docker-compose.yml`, `api.py` (FastAPI) | Full |
| Feature store | `core/feature_store.py` (SQLite) | Full |
| UI/UX (icons, semantic colours, heatmap, treemap, network, drill-down) | dashboard | Full |
| TFT/LSTM, GNN, message queues, retraining | intentionally skipped per guardrails | — |

**Guardrails honoured:** XGBoost stays the **primary** battery model (real,
defensible accuracy); no enterprise-infra rabbit holes (no queues/streaming/
retraining); everything degrades gracefully; nothing overclaims.

---

## What's new in v2 (audit-hardening)

A technical audit of v1 (3 engines) found gaps; v2 closes them at tiered depth.
The **cross-cutting quality layer** (`core/`) is implemented **once and reused
everywhere**: structured logging, measurable KPIs, explainability (SHAP /
feature-importance) and uncertainty (conformal / bootstrap intervals) on every
prediction, plus unit tests and docstrings throughout.

### Gap → Resolution map (the re-audit checks these)

| Audit gap | Resolved by | Depth |
| --- | --- | --- |
| No supply-chain risk intelligence | `engines/engine_supply_chain.py` + Supply-Chain page | **Full (Tier 1)** |
| No multi-agent orchestration | `core/orchestrator.py` + agentic `copilot.py` | **Full (Tier 1)** |
| Limited explainability / no confidence intervals | `core/explain.py`, `core/uncertainty.py` on the battery & readiness models | **Full (Tier 1)** |
| No measurable KPIs / monitoring / logging / tests / docs | `core/kpis.py`, `core/logging_config.py`, `tests/`, README | **Full (Tier 1)** |
| No maintenance scheduling optimiser | `engines/engine_maintenance.py` (OR-Tools CP-SAT + greedy fallback) | MVP (Tier 2) |
| No charging optimisation | `engines/engine_maintenance.py` charging scheduler | MVP (Tier 2) |
| Limited Scope 3 carbon accounting | `engines/engine_carbon.py` Scope 1/2/3 rework | Full (Tier 2) |
| Scalability unproven | `generate_data.py --n-vehicles 10000` + caching + in-app benchmark | Demonstrated (Tier 2) |
| No manufacturing QMS integration | `engines/engine_quality.py` light SPC + traceability | MVP (Tier 3, labelled) |
| No digital twin | Digital-twin live fleet view | MVP (Tier 3, labelled) |
| Limited geospatial analytics | Supplier network graph + fleet map | Partial (Tier 1/3) |

---

## Quick start

```bash
# 1. Install core dependencies (optional deps documented in requirements.txt)
pip install -r requirements.txt

# 2. (Optional) enable the live LLM copilot
cp .env.example .env        # then paste your ANTHROPIC_API_KEY

# 3. Build everything from scratch — ONE command
python run_pipeline.py      # generates data + trains the battery model

# 4. Launch the dashboard
streamlit run app.py        # opens http://localhost:8501
```

The copilot works **without** an API key — it falls back to a deterministic
routed summary, so the demo never crashes on stage.

### Scalability demo (10,000 vehicles)

```bash
python generate_data.py --n-vehicles 10000   # rebuild the fleet at 10k
```

…or just move the **“Fleet size”** slider in the sidebar to **10,000**: the app
regenerates, caches heavy loads with `@st.cache_data`, and reports the per-page
render time so you can see it stays responsive (vectorised scoring: 10k
vehicles in ~20 ms).

### REST API + Docker (integration story)

```bash
# REST API (interactive docs at http://localhost:8000/docs)
uvicorn api:app --port 8000
#   GET  /fleet/summary
#   POST /battery/predict  {"cell_id": "CELL_010"}
#   POST /scenario/run     {"name": "tariff_change", "params": {"pct": 15}}

# Or the whole stack (dashboard :8501 + API :8000) in one command:
docker compose up
```

---

## Optional dependencies & graceful degradation

Every optional dependency has a documented fallback, so the app **always runs**:

| Optional dep | Enables | Fallback if absent |
| --- | --- | --- |
| `ortools` | CP-SAT maintenance scheduler | greedy earliest-feasible heuristic |
| `shap` | SHAP per-prediction drivers | model feature-importance (global) |
| `mapie` | conformal RUL intervals | built-in split-conformal intervals |

Install any of them to switch on the primary path (see `requirements.txt`).

---

## Architecture

```
ev_fleet_brain/
├── data/                        synthetic datasets (rebuilt by run_pipeline.py)
│   ├── battery_data_synthetic.csv
│   ├── fleet_telematics_synthetic.csv
│   ├── suppliers_synthetic.csv          # battery-material supply chain
│   ├── maintenance_events_synthetic.csv
│   └── emission_factors.json
├── core/                        cross-cutting layer (implement once, reuse everywhere)
│   ├── orchestrator.py          bounded multi-agent PLANNER (visible plan trace)
│   ├── recommend.py             structured recommendations (confidence/impact/alts)
│   ├── kpis.py                  KPI structure + formatting
│   ├── explain.py               SHAP / feature-importance helpers
│   ├── uncertainty.py           conformal / bootstrap confidence intervals
│   ├── feature_store.py         SQLite-backed feature store
│   ├── monitoring.py            model-drift (PSI) monitoring stub
│   └── logging_config.py        structured JSON logging
├── engines/
│   ├── engine_battery.py        anomaly detection, CIs, snapshot, PASSPORT
│   ├── engine_readiness.py      confidence-scored index, full TCO
│   ├── engine_carbon.py         Scope 1/2/3 + HOURLY grid + smart charging
│   ├── engine_supply_chain.py   HHI, geopolitics, graph + risk PROPAGATION
│   ├── engine_maintenance.py    CP-SAT/greedy maintenance + charging optimiser
│   ├── engine_quality.py        SPC + root-cause hints (RCA)
│   └── engine_scenario.py       NEW: 4 what-if scenarios ⭐
├── copilot.py                   planner-driven: plans, calls agents, explains
├── api.py                       FastAPI REST layer (3 endpoints)
├── Dockerfile, docker-compose.yml   one-command containerised run
├── config.py                    paths, constants, EV catalog, reference tables
├── app.py                       9-page dashboard (planner + scenario on each/own page)
├── generate_data.py             parametrized synthetic data (scales to N)
├── run_pipeline.py              ONE command to build everything
├── tests/                       pytest unit + integration + smoke + demo + v3 tests
├── DEMO_QA.md                   judge-question crib sheet
├── requirements.txt
└── README.md
```

---

## How each engine works

**Battery (Engine 1, Tier 1)** — Early-cycle features (log-variance of the ΔQ(V)
curve — the key Severson feature — plus fade slope, capacity at cycle 100, and
range) feed an XGBoost regressor on `log10(cycle_life)`; we report **real RMSE /
MAPE on a 40% held-out set**. Adds: a **prediction interval** on RUL
(split-conformal, or `mapie` if installed), **IsolationForest anomaly detection**
over per-cell degradation signals (fade / thermal / internal-resistance proxies),
an **SoH trend**, and **per-prediction drivers** (SHAP or feature-importance).

**Readiness (Engine 2, Tier 1)** — Matches each vehicle to a real Indian EV,
scores **range (40%) + payload (30%) + ROI (30%)**, adds a **confidence score**
(how decisive the recommendation is) and a full **5-year TCO** (purchase, energy,
maintenance, insurance, residual value).

**Supply chain (Engine 4, Tier 1)** — **Herfindahl (HHI)** concentration per
critical material, **volume-weighted geopolitical exposure** (static illustrative
country-risk table), **cell → pack → vehicle traceability**, and a
**`networkx`** supplier network. `supply_risk_summary()` returns an overall score
+ top-3 vulnerabilities with **₹ exposure**.

**Maintenance & charging (Engine 5, Tier 2 MVP)** — CP-SAT (OR-Tools) or greedy
scheduling that respects workshop bay-hour capacity and minimises
priority-weighted downtime; a cheapest-hours-first charging optimiser against a
time-of-use tariff. Reports downtime reduction %, ₹ saved, charger utilisation.

**Carbon (Engine 3, Tier 2)** — Explicit **Scope 1** (diesel tailpipe), **Scope
2** (grid charging) and **Scope 3** (diesel well-to-tank + amortised
battery-embodied), per vehicle class, plus a ₹ carbon-credit value.

**Quality (Engine 6, Tier 3 MVP)** — A light QMS view: incoming-material defect
trend + an **SPC control chart** with 3σ out-of-control flags + traceability.
Explicitly **not** a full MES/QMS.

**Copilot / Orchestrator (Tier 1)** — `orchestrator.route(query)` does
deterministic **intent routing** to the right engine(s), gathers their structured
outputs, and the copilot turns that into plain English with **one LLM call**.
It is a **router + explainer, not an autonomous agent**, and is labelled so.

---

## Deploy a public demo (Streamlit Community Cloud)

The app **self-bootstraps**: on a fresh machine, if data or the model is missing
it generates and trains them on first run (both are gitignored, not committed).

1. Go to **[share.streamlit.io](https://share.streamlit.io)**, sign in with GitHub.
2. **New app → From existing repo**: repo `jatinpathak/ev_fleet_brain`, branch
   `main`, main file `app.py`, Python **3.11+**.
3. *(Optional — live copilot)* **Advanced settings → Secrets**:
   ```toml
   ANTHROPIC_API_KEY = "your_real_key_here"
   ```
4. **Deploy.** First load takes ~30–60 s to build data + model, then it's cached.

`.env`, `.streamlit/secrets.toml`, `data/` and `models/` are gitignored — no key
or artifact is ever committed.

---

## Running the tests

```bash
pytest -v
```

| Round | File(s) | What it checks |
| --- | --- | --- |
| **1 — Component** | `tests/test_data.py`, `tests/test_engines.py` | Reproducible data (incl. suppliers & maintenance); each engine's outputs in valid ranges; prediction intervals bracket the point estimate; every engine exposes KPIs. |
| **2 — Integration** | `tests/test_integration.py`, `tests/test_new_engines.py` | Full chain end-to-end; orchestrator intent routing; copilot fallback with no key; supply-chain / maintenance / quality / core-service correctness; edge cases. |
| **3 — Dashboard smoke** | `tests/test_dashboard.py` | Every one of the 7 pages renders headless via Streamlit `AppTest`; controls don't crash. |
| **4 — Demo dry-run** | `tests/test_demo_walkthrough.py` | The exact judge path produces sensible, internally consistent output. |

---

## Honest disclosure

- **All datasets are synthetic** (fixed seed, reproducible) and labelled as such
  in the app footer/sidebar — **except** that the battery module is shaped after
  the public **Severson LFP cycling dataset** so the accuracy headline is
  genuine. (Real Severson/MATR data via BatteryML can be dropped in; the model
  and features are unchanged.)
- **Nothing claims to be autonomous, real-time or physics-simulated.** The
  copilot is a router+explainer; the digital twin and QMS are labelled MVPs.
- **All reference tables are illustrative:** the country-risk table, emission
  factors (grid ~0.7 kgCO₂/kWh, CEA India — verify latest before submission),
  TCO inputs, carbon-credit price and production value-at-risk. They are for
  prioritisation and demonstration, **not** a financial or compliance quote.
- **Carbon scopes:** under the GHG Protocol grid charging is formally Scope 2 and
  fuel extraction / embodied battery is Scope 3; we label them accordingly rather
  than lumping them together.
