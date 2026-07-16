# 🔋 EV Fleet Intelligence Brain

**ET AI Hackathon 2026 · Problem 3 — AI for Industrial EV Supply Chain & Asset Intelligence**

A single assistant that helps a commercial fleet operator electrify a
300-vehicle diesel/petrol fleet — intelligently, not all at once. Three
independent engines share one data layer, are explained by one LLM copilot, and
are shown on one clean Streamlit dashboard.

| Engine | Question it answers |
| --- | --- |
| 🔋 **Battery Health & RUL** | How healthy is each battery, and how many cycles remain? |
| 🚚 **Fleet Readiness** | Which vehicles should switch to EV first, and what's the rupee saving? |
| 🌱 **Carbon Savings** | How much CO₂ do we avoid by electrifying? |
| 💬 **Copilot** | Explains every number in plain English for a non-technical manager. |

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) enable the live LLM copilot
cp .env.example .env        # then paste your ANTHROPIC_API_KEY

# 3. Build everything from scratch — ONE command
python run_pipeline.py      # generates data + trains both models

# 4. Launch the dashboard
streamlit run app.py        # opens http://localhost:8501
```

The copilot works **without** an API key — it falls back to clear templated
explanations, so the demo never crashes.

---

## Deploy a public demo (Streamlit Community Cloud)

The code is deploy-ready: on a fresh machine the app **self-bootstraps** — if the
synthetic data or trained model is missing, it generates and trains them on the
first run (they are gitignored, not committed). So no build step is needed.

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
2. **New app → From existing repo**, then:
   - Repository: `jatinpathak/ev_fleet_brain`
   - Branch: `main`
   - Main file path: `app.py`
   - (Advanced) Python version: **3.11** or newer.
3. *(Optional — for the live LLM copilot)* Open **Advanced settings → Secrets**
   and paste (see `.streamlit/secrets.toml.example`):
   ```toml
   ANTHROPIC_API_KEY = "your_real_key_here"
   ```
   Skip this and the copilot still works via templated explanations.
4. Click **Deploy**. First load takes ~30–60 s while it builds the data and
   model; after that it is cached and fast. You get a public URL to share.

Safety: `.env`, `.streamlit/secrets.toml`, `data/`, and `models/` are all
gitignored — no key or artifact is ever committed.

---

## Running the tests

A four-round test suite proves the tool is reliable before it ever reaches a
judge. Run it all with:

```bash
pytest -v
```

| Round | File(s) | What it checks |
| --- | --- | --- |
| **1 — Component** | `tests/test_data.py`, `tests/test_engines.py` | Reproducible data, no NaNs; each engine's outputs are in valid ranges. |
| **2 — Integration** | `tests/test_integration.py` | Full chain runs end-to-end; copilot fallback with no key; edge cases (route beyond every EV's range, brand-new and near-dead batteries). |
| **3 — Dashboard smoke** | `tests/test_dashboard.py` | Every page renders headless via Streamlit `AppTest`; controls don't crash. |
| **4 — Demo dry-run** | `tests/test_demo_walkthrough.py` | The exact judge path produces sensible output and stays internally consistent. |

### Latest test report

```
26 tests collected · 26 passed
Battery RUL model:  RMSE 59.9 cycles · MAPE 3.8%  (60/40 split, 60 held-out cells)
```

---

## Sample outputs (synthetic data, fixed seed)

- **Fleet readiness:** 300/300 vehicles score as viable candidates; top pick
  `VEH_118` scores **96.3/100**, matched to the **Tata Nexon EV**, ~2.98-year payback.
- **5-year fleet savings:** **≈ ₹40.5 Cr** in running-cost reduction.
- **Carbon:** **≈ 2,023 tonnes CO₂ avoided per year** — a **53.6%** reduction
  versus the diesel baseline (~10,100 tonnes over five years).

---

## Project structure

```
ev_fleet_brain/
├── data/                     synthetic datasets (rebuilt by run_pipeline.py)
├── models/                   trained model pickles (rebuilt by run_pipeline.py)
├── tests/                    four-round test suite
├── generate_data.py          creates the synthetic datasets (fixed seed)
├── engine_battery.py         battery health + RUL (XGBoost, Severson features)
├── engine_readiness.py       electrification scoring + EV matching
├── engine_carbon.py          CO₂ and cost savings
├── copilot.py                thin Anthropic wrapper (explains, not an agent)
├── config.py                 constants, paths, real EV catalog, .env key
├── app.py                    Streamlit dashboard (home + 3 pages + copilot)
├── run_pipeline.py           ONE command to build everything from scratch
├── requirements.txt
└── README.md
```

---

## How each engine works

**Battery (Engine 1)** — Extracts *early-cycle* features from a cell's first
100 cycles: the **log-variance of the change in the capacity-vs-voltage curve**
(the key Severson et al. feature), capacity fade slope, capacity at cycle 100,
and capacity range. An XGBoost regressor predicts `log10(cycle_life)`; we report
RMSE and MAPE on a 40% held-out set. `predict_health()` returns state-of-health,
predicted cycle life, and remaining useful life, colour-coded
healthy / degraded / critical.

**Readiness (Engine 2)** — Each vehicle is matched to the best-fit model in the
real Indian `EV_CATALOG` (Tata Ace EV, Mahindra Treo, Tata Nexon EV). The score
is a weighted blend: **range fit (40%) + payload fit (30%) + 5-year ROI (30%)**,
with per-vehicle annual/5-year rupee savings and payback period.

**Carbon (Engine 3)** —
`current_CO₂ = (annual_km / diesel_km_per_liter) × diesel_co2_per_liter` versus
`electrified_CO₂ = (annual_km / ev_efficiency) × grid_co2_per_kwh`. Fleet totals
are the exact sum of per-vehicle values (no double counting).

**Copilot** — A thin wrapper around the Anthropic API (`claude-sonnet-4-6`). It
is deliberately **not** an agent: it takes a structured dict and returns a
plain-English explanation under ~200 words. No key → templated fallback.

---

## Synthetic data & assumptions (honest disclosure)

- **All data is synthetic** and clearly labelled as such in the app footer and
  sidebar. It is generated from a fixed random seed for reproducibility.
- The **battery dataset mimics the public Severson LFP cycling dataset**
  (150 cells, lives spread ~150–2300 cycles, end of life = 80% of 1.1 Ah).
- The **fleet dataset** (300 vehicles) is realistic simulated telematics,
  **anchored to real Indian commercial EVs** for pricing and specs.
- **Carbon accounting:** Scope 1 tailpipe diesel vs Scope 2/3 grid electricity
  at **0.7 kgCO₂/kWh** (CEA India estimate). **Operational emissions only** —
  manufacturing / embodied battery carbon is not modelled.
- Rupee savings use a diesel running-cost baseline of ₹8.5/km and the EV
  catalog's per-km costs; they are directional estimates for prioritisation,
  not a financial quote.
