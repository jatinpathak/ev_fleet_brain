"""Engine 2 - Fleet Electrification Readiness.

Scores every vehicle 0-100 for how ready it is to switch to an EV, and matches
each to the best-fit model in the real Indian EV_CATALOG. The score is a
weighted blend of range fit (40%), payload fit (30%) and 5-year ROI (30%).

Public API
----------
* score_fleet()                    -> DataFrame of every vehicle with scores
* vehicle_recommendation(vid)      -> dict for one vehicle
* fleet_summary()                  -> totals for the dashboard
"""
from __future__ import annotations

import pandas as pd

import config


def _load_fleet_df() -> pd.DataFrame:
    if not config.FLEET_DATA_CSV.exists():
        raise FileNotFoundError(
            f"{config.FLEET_DATA_CSV} not found. Run generate_data.py first."
        )
    return pd.read_csv(config.FLEET_DATA_CSV)


def _range_fit(daily_range_km: float, ev_range_km: float) -> float:
    """1.0 if the EV comfortably covers the daily route, decaying past that.

    We keep a 20% buffer for real-world range loss; a route needing more than
    the buffered range scores below 1 and can reach 0.
    """
    usable = ev_range_km * 0.8
    if usable <= 0:
        return 0.0
    ratio = usable / max(daily_range_km, 1.0)
    return float(max(0.0, min(1.0, ratio)))


def _payload_fit(required_kg: float, ev_payload_kg: float) -> float:
    """1.0 if the EV can carry the payload, scaling down if it cannot."""
    if required_kg <= ev_payload_kg:
        return 1.0
    return float(max(0.0, ev_payload_kg / max(required_kg, 1.0)))


def _annual_savings(vehicle: pd.Series, ev: dict) -> float:
    """Annual running-cost saving (INR) of the EV vs the diesel baseline."""
    diesel_cost = vehicle["annual_km"] * config.DIESEL_COST_PER_KM
    ev_cost = vehicle["annual_km"] * ev["cost_per_km"]
    return float(diesel_cost - ev_cost)


def _roi_fit(payback_years: float) -> float:
    """Map payback period to a 0-1 ROI score (faster payback -> higher)."""
    if payback_years <= 0:
        return 0.0
    # Full marks at <=2 years, zero at >=10 years, linear in between.
    if payback_years <= 2:
        return 1.0
    if payback_years >= 10:
        return 0.0
    return float((10 - payback_years) / 8.0)


def _best_ev_for(vehicle: pd.Series) -> tuple[str, dict, float]:
    """Choose the EV that maximises range+payload fit for this vehicle."""
    best_name, best_ev, best_fit = None, None, -1.0
    for name, ev in config.EV_CATALOG.items():
        fit = (
            config.READINESS_WEIGHTS["range_fit"] * _range_fit(vehicle["avg_daily_range_km"], ev["range_km"])
            + config.READINESS_WEIGHTS["payload_fit"] * _payload_fit(vehicle["payload_kg"], ev["payload_kg"])
        )
        if fit > best_fit:
            best_name, best_ev, best_fit = name, ev, fit
    return best_name, best_ev, best_fit


def vehicle_recommendation(vehicle_id: str, fleet_df: pd.DataFrame | None = None) -> dict:
    """Full readiness recommendation for a single vehicle."""
    df = fleet_df if fleet_df is not None else _load_fleet_df()
    match = df[df["vehicle_id"] == vehicle_id]
    if match.empty:
        raise KeyError(f"Unknown vehicle_id: {vehicle_id}")
    vehicle = match.iloc[0]

    ev_name, ev, _ = _best_ev_for(vehicle)

    range_fit = _range_fit(vehicle["avg_daily_range_km"], ev["range_km"])
    payload_fit = _payload_fit(vehicle["payload_kg"], ev["payload_kg"])

    annual_savings = _annual_savings(vehicle, ev)
    # Payback: EV purchase price divided by annual saving. Guard against a
    # non-positive saving (would never pay back) by capping at the horizon+.
    if annual_savings > 0:
        payback_years = ev["price_inr"] / annual_savings
    else:
        payback_years = float(config.ROI_HORIZON_YEARS * 5)  # effectively "never"

    roi_fit = _roi_fit(payback_years)

    score = 100.0 * (
        config.READINESS_WEIGHTS["range_fit"] * range_fit
        + config.READINESS_WEIGHTS["payload_fit"] * payload_fit
        + config.READINESS_WEIGHTS["roi"] * roi_fit
    )
    score = float(max(0.0, min(100.0, score)))

    five_year_savings = annual_savings * config.ROI_HORIZON_YEARS

    return {
        "vehicle_id": vehicle_id,
        "vehicle_type": vehicle["vehicle_type"],
        "annual_km": int(vehicle["annual_km"]),
        "avg_daily_range_km": int(vehicle["avg_daily_range_km"]),
        "payload_kg": int(vehicle["payload_kg"]),
        "duty_cycle": vehicle["duty_cycle"],
        "readiness_score": round(score, 1),
        "ev_match": ev_name,
        "range_fit": round(range_fit, 3),
        "payload_fit": round(payload_fit, 3),
        "payback_years": round(float(payback_years), 2),
        "annual_savings_inr": round(annual_savings, 0),
        "five_year_savings_inr": round(five_year_savings, 0),
    }


def score_fleet(fleet_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score and rank the entire fleet, best candidates first."""
    df = fleet_df if fleet_df is not None else _load_fleet_df()
    recs = [vehicle_recommendation(vid, df) for vid in df["vehicle_id"]]
    out = pd.DataFrame(recs)
    return out.sort_values("readiness_score", ascending=False).reset_index(drop=True)


def fleet_summary(scored: pd.DataFrame | None = None) -> dict:
    """Aggregate readiness totals for the dashboard headline metrics."""
    df = scored if scored is not None else score_fleet()
    ready = df[df["readiness_score"] >= 60]
    return {
        "total_vehicles": int(len(df)),
        "ready_now": int(len(ready)),
        "avg_readiness_score": round(float(df["readiness_score"].mean()), 1),
        "total_annual_savings_inr": round(float(df["annual_savings_inr"].sum()), 0),
        "total_five_year_savings_inr": round(float(df["five_year_savings_inr"].sum()), 0),
        "top_vehicle_id": df.iloc[0]["vehicle_id"],
    }


if __name__ == "__main__":
    scored = score_fleet()
    print(scored.head(10).to_string(index=False))
    print("\nSummary:", fleet_summary(scored))
