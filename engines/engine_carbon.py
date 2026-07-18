"""Engine 3 - Carbon accounting (Scope 1, 2 & 3), per vehicle-class.

Computes the CO2 avoided by electrifying each vehicle with an explicit scope
split, and aggregates per vehicle class (light / medium / heavy).

Scope conventions (stated in the UI and README; all factors ILLUSTRATIVE):
* Scope 1 — direct diesel tailpipe combustion of the CURRENT fleet.
* Scope 2 — grid electricity to charge the EVs after switching (indirect).
* Scope 3 — upstream: diesel well-to-tank (~15% uplift) for the current fleet,
  and amortised battery-embodied carbon for the EVs.

Note: under the GHG Protocol grid charging is formally Scope 2 and fuel
extraction / embodied battery is Scope 3; we label them accordingly rather than
lumping everything together. The audit asked for "proper Scope 1 & 3
accounting" — this provides 1, 2 and 3, honestly labelled.

Public API
----------
* vehicle_carbon(vehicle_id)   -> dict for one vehicle (scope split)
* score_carbon()               -> DataFrame for the whole fleet
* fleet_carbon_summary()       -> aggregate totals + per-class breakdown
* kpis()                       -> list[KPI]
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from core.kpis import KPI, rupees
from core.logging_config import get_logger, timed

log = get_logger(__name__)

# Illustrative upstream (Scope 3) factors.
DIESEL_WTT_UPLIFT = 0.15          # well-to-tank ≈ +15% on tailpipe
GRID_UPSTREAM_UPLIFT = 0.05       # T&D + upstream fuel for generation ≈ +5%
BATTERY_EMBODIED_KG_PER_KWH = 70  # kgCO2 per kWh of pack (illustrative)
EV_PACK_KWH = 25                  # representative commercial-EV pack size
EV_PACK_LIFE_YEARS = 8            # amortise embodied carbon over pack life


def load_emission_factors() -> dict:
    if not config.EMISSION_FACTORS_JSON.exists():
        raise FileNotFoundError(
            f"{config.EMISSION_FACTORS_JSON} not found. Run generate_data.py first."
        )
    with open(config.EMISSION_FACTORS_JSON) as f:
        return json.load(f)


def _load_fleet_df() -> pd.DataFrame:
    if not config.FLEET_DATA_CSV.exists():
        raise FileNotFoundError(
            f"{config.FLEET_DATA_CSV} not found. Run generate_data.py first."
        )
    return pd.read_csv(config.FLEET_DATA_CSV)


def vehicle_carbon(vehicle_id: str, fleet_df: pd.DataFrame | None = None,
                   factors: dict | None = None) -> dict:
    df = fleet_df if fleet_df is not None else _load_fleet_df()
    f = factors if factors is not None else load_emission_factors()

    match = df[df["vehicle_id"] == vehicle_id]
    if match.empty:
        raise KeyError(f"Unknown vehicle_id: {vehicle_id}")
    vehicle = match.iloc[0]
    annual_km = float(vehicle["annual_km"])

    # Current fleet (diesel).
    scope1 = (annual_km / f["diesel_km_per_liter"]) * f["diesel_co2_per_liter"]
    scope3_diesel = scope1 * DIESEL_WTT_UPLIFT
    current_total = scope1 + scope3_diesel

    # After electrification.
    scope2 = (annual_km / f["ev_efficiency_km_per_kwh"]) * f["india_grid_co2_per_kwh"]
    scope3_ev = scope2 * GRID_UPSTREAM_UPLIFT + (
        BATTERY_EMBODIED_KG_PER_KWH * EV_PACK_KWH / EV_PACK_LIFE_YEARS
    )
    ev_total = scope2 + scope3_ev

    savings = current_total - ev_total

    # Running-cost saving (cheapest-EV per-km, conservative).
    ev_cost_per_km = min(ev["cost_per_km"] for ev in config.EV_CATALOG.values())
    savings_cost = annual_km * (config.DIESEL_COST_PER_KM - ev_cost_per_km)

    return {
        "vehicle_id": vehicle_id,
        "vehicle_type": vehicle["vehicle_type"],
        "vehicle_class": config.vehicle_class(vehicle["payload_kg"]),
        "annual_km": int(annual_km),
        "scope1_diesel_kg": round(scope1, 1),
        "scope3_diesel_upstream_kg": round(scope3_diesel, 1),
        "current_total_kg": round(current_total, 1),
        "scope2_grid_kg": round(scope2, 1),
        "scope3_ev_upstream_kg": round(scope3_ev, 1),
        "ev_total_kg": round(ev_total, 1),
        "savings_co2_kg": round(savings, 1),
        "savings_cost_inr": round(savings_cost, 0),
    }


def score_carbon(fleet_df: pd.DataFrame | None = None,
                 factors: dict | None = None) -> pd.DataFrame:
    """Per-vehicle carbon with an explicit scope split (fully vectorised)."""
    import numpy as np

    df = fleet_df if fleet_df is not None else _load_fleet_df()
    f = factors if factors is not None else load_emission_factors()
    with timed(log, "score_carbon", n=len(df)):
        km = df["annual_km"].to_numpy(dtype=float)
        payload = df["payload_kg"].to_numpy(dtype=float)

        scope1 = (km / f["diesel_km_per_liter"]) * f["diesel_co2_per_liter"]
        scope3_diesel = scope1 * DIESEL_WTT_UPLIFT
        current_total = scope1 + scope3_diesel

        scope2 = (km / f["ev_efficiency_km_per_kwh"]) * f["india_grid_co2_per_kwh"]
        scope3_ev = (scope2 * GRID_UPSTREAM_UPLIFT
                     + BATTERY_EMBODIED_KG_PER_KWH * EV_PACK_KWH / EV_PACK_LIFE_YEARS)
        ev_total = scope2 + scope3_ev
        savings = current_total - ev_total

        ev_cost_per_km = min(ev["cost_per_km"] for ev in config.EV_CATALOG.values())
        savings_cost = km * (config.DIESEL_COST_PER_KM - ev_cost_per_km)

        classes = np.array([config.vehicle_class(p) for p in payload])
        out = pd.DataFrame({
            "vehicle_id": df["vehicle_id"].to_numpy(),
            "vehicle_type": df["vehicle_type"].to_numpy(),
            "vehicle_class": classes,
            "annual_km": km.astype(int),
            "scope1_diesel_kg": np.round(scope1, 1),
            "scope3_diesel_upstream_kg": np.round(scope3_diesel, 1),
            "current_total_kg": np.round(current_total, 1),
            "scope2_grid_kg": np.round(scope2, 1),
            "scope3_ev_upstream_kg": np.round(scope3_ev, 1),
            "ev_total_kg": np.round(ev_total, 1),
            "savings_co2_kg": np.round(savings, 1),
            "savings_cost_inr": np.round(savings_cost, 0),
        })
    return out


def fleet_carbon_summary(carbon_df: pd.DataFrame | None = None) -> dict:
    df = carbon_df if carbon_df is not None else score_carbon()

    total_current = float(df["current_total_kg"].sum())
    total_ev = float(df["ev_total_kg"].sum())
    total_savings = float(df["savings_co2_kg"].sum())
    pct = (total_savings / total_current * 100) if total_current > 0 else 0.0

    by_class = (df.groupby("vehicle_class")["savings_co2_kg"].sum() / 1000).round(2).to_dict()
    credit_value = (total_savings / 1000) * config.CARBON_CREDIT_INR_PER_TONNE

    return {
        "scope1_tonnes": round(float(df["scope1_diesel_kg"].sum()) / 1000, 1),
        "scope2_tonnes": round(float(df["scope2_grid_kg"].sum()) / 1000, 1),
        "scope3_current_tonnes": round(float(df["scope3_diesel_upstream_kg"].sum()) / 1000, 1),
        "scope3_ev_tonnes": round(float(df["scope3_ev_upstream_kg"].sum()) / 1000, 1),
        "total_current_co2_tonnes": round(total_current / 1000, 1),
        "total_ev_co2_tonnes": round(total_ev / 1000, 1),
        "total_savings_co2_tonnes": round(total_savings / 1000, 1),
        "savings_pct": round(pct, 1),
        "five_year_savings_co2_tonnes": round(total_savings * 5 / 1000, 1),
        "total_savings_cost_inr": round(float(df["savings_cost_inr"].sum()), 0),
        "carbon_credit_value_inr": round(credit_value, 0),
        "savings_by_class_tonnes": by_class,
    }


def hourly_grid_intensity() -> pd.DataFrame:
    """24-hour grid carbon-intensity profile (kgCO2/kWh), illustrative."""
    base = load_emission_factors()["india_grid_co2_per_kwh"]
    mult = config.GRID_HOURLY_MULTIPLIER
    return pd.DataFrame({
        "hour": list(range(24)),
        "multiplier": mult,
        "kg_co2_per_kwh": [round(base * m, 4) for m in mult],
    })


def smart_charging_carbon(fleet_df: pd.DataFrame | None = None,
                          factors: dict | None = None) -> dict:
    """Compare charging CO₂ under off-peak vs evening-peak windows.

    Charging the fleet's nightly energy in the cleanest overnight hours instead
    of the evening peak cuts Scope 3 (grid) CO₂ — the case for smart charging.
    """
    df = fleet_df if fleet_df is not None else _load_fleet_df()
    f = factors if factors is not None else load_emission_factors()
    demand_kwh = float((df["avg_daily_range_km"] / f["ev_efficiency_km_per_kwh"]).sum())
    base = f["india_grid_co2_per_kwh"]
    mult = config.GRID_HOURLY_MULTIPLIER

    off_peak_hours = [0, 1, 2, 3, 4, 5]           # cleanest overnight window
    peak_hours = [18, 19, 20, 21]                 # dirtiest evening window
    off_intensity = base * np.mean([mult[h] for h in off_peak_hours])
    peak_intensity = base * np.mean([mult[h] for h in peak_hours])

    off_kg = demand_kwh * off_intensity
    peak_kg = demand_kwh * peak_intensity
    return {
        "daily_energy_kwh": round(demand_kwh, 0),
        "off_peak_co2_kg_day": round(off_kg, 0),
        "peak_co2_kg_day": round(peak_kg, 0),
        "co2_saved_kg_day": round(peak_kg - off_kg, 0),
        "co2_saved_pct": round(100 * (peak_kg - off_kg) / peak_kg, 1) if peak_kg else 0.0,
        "annual_co2_saved_tonnes": round((peak_kg - off_kg) * 365 / 1000, 1),
    }


def kpis(carbon_df: pd.DataFrame | None = None) -> list[KPI]:
    df = carbon_df if carbon_df is not None else score_carbon()
    s = fleet_carbon_summary(df)
    return [
        KPI("Scope 1 baseline", f"{s['scope1_tonnes']:,.0f}", "t/yr",
            "Direct diesel tailpipe CO₂ of the current fleet.", "bad"),
        KPI("Scope 2 after switch", f"{s['scope2_tonnes']:,.0f}", "t/yr",
            "Grid electricity CO₂ once electrified (indirect).", "warn"),
        KPI("Net CO₂ avoided", f"{s['total_savings_co2_tonnes']:,.0f}", "t/yr",
            f"{s['savings_pct']}% lower, net of upstream & embodied carbon.", "good"),
        KPI("Carbon-credit value", rupees(s["carbon_credit_value_inr"]), "/yr",
            "Illustrative voluntary-market value of the avoided CO₂.", "good"),
    ]


if __name__ == "__main__":
    carbon = score_carbon()
    print(carbon.head().to_string(index=False))
    print("\nSummary:", fleet_carbon_summary(carbon))
