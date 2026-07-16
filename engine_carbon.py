"""Engine 3 - Carbon (CO2) Savings.

Computes the CO2 avoided by electrifying each vehicle, and the fleet total.

Assumptions (stated explicitly on the dashboard and in the README):
* Current emissions are Scope 1 tailpipe diesel: burning fuel directly.
* Electrified emissions are Scope 2/3 grid: the CO2 to generate the charging
  electricity, using the CEA India grid estimate of 0.7 kgCO2/kWh.
* We do NOT model manufacturing / battery embodied carbon -- operational only.

Formulae
--------
current_co2_kg     = (annual_km / diesel_km_per_liter) * diesel_co2_per_liter
electrified_co2_kg = (annual_km / ev_efficiency_km_per_kwh) * india_grid_co2_per_kwh

Public API
----------
* vehicle_carbon(vehicle_id)   -> dict for one vehicle
* score_carbon()               -> DataFrame for the whole fleet
* fleet_carbon_summary()       -> aggregate totals for the dashboard
"""
from __future__ import annotations

import json

import pandas as pd

import config


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


def vehicle_carbon(
    vehicle_id: str,
    fleet_df: pd.DataFrame | None = None,
    factors: dict | None = None,
) -> dict:
    """CO2 (kg) and cost savings for electrifying a single vehicle."""
    df = fleet_df if fleet_df is not None else _load_fleet_df()
    f = factors if factors is not None else load_emission_factors()

    match = df[df["vehicle_id"] == vehicle_id]
    if match.empty:
        raise KeyError(f"Unknown vehicle_id: {vehicle_id}")
    vehicle = match.iloc[0]
    annual_km = float(vehicle["annual_km"])

    current_co2 = (annual_km / f["diesel_km_per_liter"]) * f["diesel_co2_per_liter"]
    electrified_co2 = (annual_km / f["ev_efficiency_km_per_kwh"]) * f["india_grid_co2_per_kwh"]
    savings_co2 = current_co2 - electrified_co2

    # Simple running-cost saving to keep the carbon page self-contained.
    diesel_cost = annual_km * config.DIESEL_COST_PER_KM
    # Assume the fleet's cheapest EV per-km for a conservative carbon-page cost.
    ev_cost_per_km = min(ev["cost_per_km"] for ev in config.EV_CATALOG.values())
    ev_cost = annual_km * ev_cost_per_km
    savings_cost = diesel_cost - ev_cost

    return {
        "vehicle_id": vehicle_id,
        "vehicle_type": vehicle["vehicle_type"],
        "annual_km": int(annual_km),
        "current_co2_kg": round(current_co2, 1),
        "electrified_co2_kg": round(electrified_co2, 1),
        "savings_co2_kg": round(savings_co2, 1),
        "savings_cost_inr": round(savings_cost, 0),
    }


def score_carbon(
    fleet_df: pd.DataFrame | None = None,
    factors: dict | None = None,
) -> pd.DataFrame:
    """Carbon savings for every vehicle in the fleet."""
    df = fleet_df if fleet_df is not None else _load_fleet_df()
    f = factors if factors is not None else load_emission_factors()
    rows = [vehicle_carbon(vid, df, f) for vid in df["vehicle_id"]]
    return pd.DataFrame(rows)


def fleet_carbon_summary(carbon_df: pd.DataFrame | None = None) -> dict:
    """Aggregate fleet-wide carbon totals (per-vehicle sums -> no double count)."""
    df = carbon_df if carbon_df is not None else score_carbon()

    total_current = float(df["current_co2_kg"].sum())
    total_electrified = float(df["electrified_co2_kg"].sum())
    total_savings = float(df["savings_co2_kg"].sum())
    pct = (total_savings / total_current * 100) if total_current > 0 else 0.0

    # Savings broken down by original vehicle type, for the bar chart.
    by_type = (
        df.groupby("vehicle_type")["savings_co2_kg"].sum().round(1).to_dict()
    )

    return {
        "total_current_co2_tonnes": round(total_current / 1000, 1),
        "total_electrified_co2_tonnes": round(total_electrified / 1000, 1),
        "total_savings_co2_tonnes": round(total_savings / 1000, 1),
        "savings_pct": round(pct, 1),
        "five_year_savings_co2_tonnes": round(total_savings * 5 / 1000, 1),
        "total_savings_cost_inr": round(float(df["savings_cost_inr"].sum()), 0),
        "savings_by_vehicle_type_kg": by_type,
    }


if __name__ == "__main__":
    carbon = score_carbon()
    print(carbon.head().to_string(index=False))
    print("\nSummary:", fleet_carbon_summary(carbon))
