"""Central configuration: paths, constants, catalogs, and static reference tables.

Everything the engines, the orchestrator and the dashboard need to agree on
lives here so there is a single source of truth. The API key is read lazily
from the .env file.

All reference tables (country-risk weights, emission factors, TCO inputs) are
ILLUSTRATIVE and clearly labelled as such in the UI and README. Swap in
audited numbers before any real decision.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

BATTERY_DATA_CSV = DATA_DIR / "battery_data_synthetic.csv"
FLEET_DATA_CSV = DATA_DIR / "fleet_telematics_synthetic.csv"
SUPPLIERS_CSV = DATA_DIR / "suppliers_synthetic.csv"
MAINTENANCE_CSV = DATA_DIR / "maintenance_events_synthetic.csv"
EMISSION_FACTORS_JSON = DATA_DIR / "emission_factors.json"

BATTERY_MODEL_PKL = MODELS_DIR / "battery_rul_model.pkl"
READINESS_MODEL_PKL = MODELS_DIR / "readiness_scorer.pkl"

# Reproducibility: a single fixed seed used everywhere.
RANDOM_SEED = 42

# Default fleet size. generate_data.py accepts --n-vehicles to scale this; the
# scalability demo proves the app runs at 10,000.
DEFAULT_N_VEHICLES = 300
SCALE_DEMO_N_VEHICLES = 10_000

# ---------------------------------------------------------------------------
# Battery physics constants (LFP chemistry, mimicking the Severson dataset)
# ---------------------------------------------------------------------------
NOMINAL_CAPACITY_AH = 1.1          # nominal capacity of an LFP cell
END_OF_LIFE_FRACTION = 0.80        # end of life defined at 80% of nominal
END_OF_LIFE_CAPACITY_AH = NOMINAL_CAPACITY_AH * END_OF_LIFE_FRACTION  # 0.88 Ah
EARLY_CYCLE_WINDOW = 100           # features come from the first 100 cycles

# Battery uncertainty / anomaly settings.
RUL_PI_COVERAGE = 0.90             # target coverage for the RUL prediction interval
ANOMALY_CONTAMINATION = 0.08       # expected fraction of abnormally-fast degraders

# ---------------------------------------------------------------------------
# Real Indian commercial EV catalog (used by the readiness engine)
# ---------------------------------------------------------------------------
EV_CATALOG = {
    "Tata Ace EV":   {"range_km": 154, "payload_kg": 1000, "price_inr": 1200000, "cost_per_km": 1.2},
    "Mahindra Treo": {"range_km": 141, "payload_kg": 550,  "price_inr": 950000,  "cost_per_km": 1.0},
    "Tata Nexon EV": {"range_km": 312, "payload_kg": 500,  "price_inr": 1500000, "cost_per_km": 1.5},
}

# Diesel/petrol running-cost baseline (INR per km), used for ROI and savings.
DIESEL_COST_PER_KM = 8.5

# Readiness score component weights (must sum to 1.0).
READINESS_WEIGHTS = {"range_fit": 0.40, "payload_fit": 0.30, "roi": 0.30}
ROI_HORIZON_YEARS = 5

# Richer TCO inputs (illustrative). Used by the readiness 5-year TCO breakdown.
TCO_INPUTS = {
    "ev_maintenance_per_km": 0.35,       # EVs: fewer moving parts
    "diesel_maintenance_per_km": 0.95,
    "ev_insurance_per_year": 22000,
    "diesel_insurance_per_year": 18000,
    "ev_residual_value_frac": 0.35,      # resale value after horizon
    "diesel_residual_value_frac": 0.25,
    "discount_rate": 0.09,               # for NPV of savings
}

# Colour thresholds for battery state-of-health status.
SOH_HEALTHY_MIN = 0.90   # >= 90% -> healthy
SOH_CRITICAL_MAX = 0.80  # <  80% -> critical (end of life)

# ---------------------------------------------------------------------------
# Supply-chain risk (engine_supply_chain)
# ---------------------------------------------------------------------------
# Critical battery materials we track concentration risk on.
CRITICAL_MATERIALS = ["Lithium", "Cobalt", "Nickel", "LFP cell", "NMC cell"]

# Static, ILLUSTRATIVE country-risk table (0 = low risk, 1 = high risk).
# Blends geopolitical stability, export-control and logistics exposure. Cite as
# illustrative; replace with an audited index (e.g. World Bank WGI) for real use.
COUNTRY_RISK = {
    "India": 0.25,
    "China": 0.70,
    "Australia": 0.15,
    "Chile": 0.35,
    "Indonesia": 0.55,
    "DR Congo": 0.90,
    "South Korea": 0.20,
    "Japan": 0.15,
    "Argentina": 0.45,
    "Russia": 0.85,
}
DEFAULT_COUNTRY_RISK = 0.50  # used for any country missing from the table

# HHI concentration bands (Herfindahl index on 0..1 scale).
HHI_HIGH = 0.25   # >= this -> highly concentrated / fragile
HHI_MODERATE = 0.15

# Rough production value at risk if a single critical material is disrupted
# (INR). Illustrative order-of-magnitude figure for the ₹-exposure headline.
PRODUCTION_VALUE_AT_RISK_INR = 500_000_000  # ₹50 Cr of annual output exposure

# ---------------------------------------------------------------------------
# Maintenance + charging optimiser (engine_maintenance)
# ---------------------------------------------------------------------------
WORKSHOP_BAYS = 6            # parallel maintenance bays at the depot
WORKSHOP_DAYS = 14          # planning horizon (days)
SERVICE_HOURS = 4           # hours one maintenance job occupies a bay
BAY_HOURS_PER_DAY = 8       # workshop shift length

DEPOT_CHARGERS = 20         # depot charge points
CHARGER_KW = 30             # charger power (kW)
CHARGE_WINDOW_HOURS = 10    # overnight dwell window
# Time-of-use tariff (INR/kWh) by hour bucket across the dwell window.
TOU_TARIFF_INR_PER_KWH = {"off_peak": 6.0, "mid_peak": 9.0, "peak": 12.0}

# ---------------------------------------------------------------------------
# Carbon: Scope 1 & 3 accounting (engine_carbon)
# ---------------------------------------------------------------------------
CARBON_CREDIT_INR_PER_TONNE = 2000   # illustrative voluntary-market price
# Vehicle-class labels derived from payload, used for per-class breakdowns.
VEHICLE_CLASSES = {"light": (0, 500), "medium": (500, 900), "heavy": (900, 10_000)}

# ---------------------------------------------------------------------------
# Manufacturing quality (engine_quality, Tier 3 MVP)
# ---------------------------------------------------------------------------
SPC_TARGET = 100.0       # target value of the synthetic process parameter
SPC_SIGMA = 2.5          # process standard deviation
SPC_CONTROL_K = 3.0      # control limits at +/- k sigma

# ---------------------------------------------------------------------------
# Copilot / LLM
# ---------------------------------------------------------------------------
COPILOT_MODEL = "claude-sonnet-4-6"
COPILOT_MAX_TOKENS = 500

_ENV_LOADED = False


def get_api_key() -> str | None:
    """Return the Anthropic API key from the environment / .env, or None."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(ROOT / ".env")
        _ENV_LOADED = True
    return os.environ.get("ANTHROPIC_API_KEY")


def vehicle_class(payload_kg: float) -> str:
    """Map a payload to a light/medium/heavy vehicle class."""
    for name, (lo, hi) in VEHICLE_CLASSES.items():
        if lo <= payload_kg < hi:
            return name
    return "heavy"
