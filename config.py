"""Central configuration: paths, constants, and the real-EV catalog.

Everything the engines and dashboard need to agree on lives here so there is a
single source of truth. The API key is read lazily from the .env file.
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
EMISSION_FACTORS_JSON = DATA_DIR / "emission_factors.json"

BATTERY_MODEL_PKL = MODELS_DIR / "battery_rul_model.pkl"
READINESS_MODEL_PKL = MODELS_DIR / "readiness_scorer.pkl"

# Reproducibility: a single fixed seed used everywhere.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Battery physics constants (LFP chemistry, mimicking the Severson dataset)
# ---------------------------------------------------------------------------
NOMINAL_CAPACITY_AH = 1.1          # nominal capacity of an LFP cell
END_OF_LIFE_FRACTION = 0.80        # end of life defined at 80% of nominal
END_OF_LIFE_CAPACITY_AH = NOMINAL_CAPACITY_AH * END_OF_LIFE_FRACTION  # 0.88 Ah
EARLY_CYCLE_WINDOW = 100           # features come from the first 100 cycles

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

# Colour thresholds for battery state-of-health status.
SOH_HEALTHY_MIN = 0.90   # >= 90% -> healthy
SOH_CRITICAL_MAX = 0.80  # <  80% -> critical (end of life)

# ---------------------------------------------------------------------------
# Copilot / LLM
# ---------------------------------------------------------------------------
COPILOT_MODEL = "claude-sonnet-4-6"
COPILOT_MAX_TOKENS = 400

_ENV_LOADED = False


def get_api_key() -> str | None:
    """Return the Anthropic API key from the environment / .env, or None."""
    global _ENV_LOADED
    if not _ENV_LOADED:
        load_dotenv(ROOT / ".env")
        _ENV_LOADED = True
    return os.environ.get("ANTHROPIC_API_KEY")
