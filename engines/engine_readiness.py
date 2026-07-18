"""Engine 2 - Fleet Electrification Readiness (confidence-scored + full TCO).

Scores every vehicle 0-100 for how ready it is to switch to an EV, matches each
to the best-fit real Indian EV, and now adds:

* a confidence score on each readiness index (how decisive the recommendation is);
* a richer 5-year Total Cost of Ownership (purchase, energy, maintenance,
  insurance, residual value) rather than running-cost only.

Public API
----------
* score_fleet()               -> DataFrame of every vehicle with scores + confidence
* vehicle_recommendation(vid)  -> dict incl. a full TCO breakdown
* fleet_summary()              -> totals for the dashboard
* kpis()                       -> list[KPI]
"""
from __future__ import annotations

import pandas as pd

import config
from core.kpis import KPI, rupees, tone_for
from core.logging_config import get_logger, timed

log = get_logger(__name__)


def _load_fleet_df() -> pd.DataFrame:
    if not config.FLEET_DATA_CSV.exists():
        raise FileNotFoundError(
            f"{config.FLEET_DATA_CSV} not found. Run generate_data.py first."
        )
    return pd.read_csv(config.FLEET_DATA_CSV)


def _range_fit(daily_range_km: float, ev_range_km: float) -> float:
    usable = ev_range_km * 0.8
    if usable <= 0:
        return 0.0
    ratio = usable / max(daily_range_km, 1.0)
    return float(max(0.0, min(1.0, ratio)))


def _payload_fit(required_kg: float, ev_payload_kg: float) -> float:
    if required_kg <= ev_payload_kg:
        return 1.0
    return float(max(0.0, ev_payload_kg / max(required_kg, 1.0)))


def _annual_savings(vehicle: pd.Series, ev: dict) -> float:
    diesel_cost = vehicle["annual_km"] * config.DIESEL_COST_PER_KM
    ev_cost = vehicle["annual_km"] * ev["cost_per_km"]
    return float(diesel_cost - ev_cost)


def _roi_fit(payback_years: float) -> float:
    if payback_years <= 0:
        return 0.0
    if payback_years <= 2:
        return 1.0
    if payback_years >= 10:
        return 0.0
    return float((10 - payback_years) / 8.0)


def _best_ev_for(vehicle: pd.Series):
    best = (None, None, -1.0)
    for name, ev in config.EV_CATALOG.items():
        fit = (
            config.READINESS_WEIGHTS["range_fit"] * _range_fit(vehicle["avg_daily_range_km"], ev["range_km"])
            + config.READINESS_WEIGHTS["payload_fit"] * _payload_fit(vehicle["payload_kg"], ev["payload_kg"])
        )
        if fit > best[2]:
            best = (name, ev, fit)
    return best


def _tco_breakdown(vehicle: pd.Series, ev: dict) -> dict:
    """5-year TCO for staying diesel vs switching to the matched EV (INR)."""
    km = float(vehicle["annual_km"])
    yrs = config.ROI_HORIZON_YEARS
    t = config.TCO_INPUTS

    diesel_energy = km * config.DIESEL_COST_PER_KM * yrs
    diesel_maint = km * t["diesel_maintenance_per_km"] * yrs
    diesel_ins = t["diesel_insurance_per_year"] * yrs
    # Diesel vehicle assumed already owned -> no purchase; small residual loss.
    diesel_total = diesel_energy + diesel_maint + diesel_ins

    ev_purchase = ev["price_inr"]
    ev_energy = km * ev["cost_per_km"] * yrs
    ev_maint = km * t["ev_maintenance_per_km"] * yrs
    ev_ins = t["ev_insurance_per_year"] * yrs
    ev_residual = -ev["price_inr"] * t["ev_residual_value_frac"]  # credit at horizon
    ev_total = ev_purchase + ev_energy + ev_maint + ev_ins + ev_residual

    return {
        "diesel": {
            "energy": round(diesel_energy), "maintenance": round(diesel_maint),
            "insurance": round(diesel_ins), "total": round(diesel_total),
        },
        "ev": {
            "purchase": round(ev_purchase), "energy": round(ev_energy),
            "maintenance": round(ev_maint), "insurance": round(ev_ins),
            "residual_credit": round(ev_residual), "total": round(ev_total),
        },
        "tco_savings_5yr": round(diesel_total - ev_total),
    }


def _confidence(range_fit: float, payload_fit: float, payback_years: float) -> float:
    """How decisive is this recommendation? 0..1.

    High when both physical fits are clear (near 0 or near 1, not marginal) and
    the payback sits comfortably inside the horizon. Marginal fits (~0.5) or a
    payback right at the boundary lower confidence. This is a heuristic
    confidence score, labelled as such in the UI.
    """
    def _decisiveness(x: float) -> float:
        return 1.0 - 4.0 * x * (1.0 - x)      # 1 at x=0/1, 0 at x=0.5

    fit_conf = 0.5 * (_decisiveness(range_fit) + _decisiveness(payload_fit))
    # payback comfort: full inside <=3yr, decays to 0 by 10yr
    pay_conf = max(0.0, min(1.0, (10 - payback_years) / 7.0))
    return float(max(0.0, min(1.0, 0.6 * fit_conf + 0.4 * pay_conf)))


def vehicle_recommendation(vehicle_id: str, fleet_df: pd.DataFrame | None = None) -> dict:
    df = fleet_df if fleet_df is not None else _load_fleet_df()
    match = df[df["vehicle_id"] == vehicle_id]
    if match.empty:
        raise KeyError(f"Unknown vehicle_id: {vehicle_id}")
    vehicle = match.iloc[0]

    ev_name, ev, _ = _best_ev_for(vehicle)
    range_fit = _range_fit(vehicle["avg_daily_range_km"], ev["range_km"])
    payload_fit = _payload_fit(vehicle["payload_kg"], ev["payload_kg"])

    annual_savings = _annual_savings(vehicle, ev)
    if annual_savings > 0:
        payback_years = ev["price_inr"] / annual_savings
    else:
        payback_years = float(config.ROI_HORIZON_YEARS * 5)

    roi_fit = _roi_fit(payback_years)
    score = 100.0 * (
        config.READINESS_WEIGHTS["range_fit"] * range_fit
        + config.READINESS_WEIGHTS["payload_fit"] * payload_fit
        + config.READINESS_WEIGHTS["roi"] * roi_fit
    )
    score = float(max(0.0, min(100.0, score)))
    confidence = _confidence(range_fit, payload_fit, payback_years)
    tco = _tco_breakdown(vehicle, ev)

    return {
        "vehicle_id": vehicle_id,
        "vehicle_type": vehicle["vehicle_type"],
        "annual_km": int(vehicle["annual_km"]),
        "avg_daily_range_km": int(vehicle["avg_daily_range_km"]),
        "payload_kg": int(vehicle["payload_kg"]),
        "duty_cycle": vehicle["duty_cycle"],
        "depot": vehicle.get("depot", ""),
        "readiness_score": round(score, 1),
        "confidence": round(confidence, 2),
        "ev_match": ev_name,
        "range_fit": round(range_fit, 3),
        "payload_fit": round(payload_fit, 3),
        "payback_years": round(float(payback_years), 2),
        "annual_savings_inr": round(annual_savings, 0),
        "five_year_savings_inr": round(annual_savings * config.ROI_HORIZON_YEARS, 0),
        "tco_savings_5yr_inr": tco["tco_savings_5yr"],
        "tco_breakdown": tco,
    }


def score_fleet(fleet_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Score and rank the entire fleet, best candidates first.

    Fully vectorised (numpy over the whole frame) so it scales to 10,000+
    vehicles in well under a second — the single-vehicle ``vehicle_recommendation``
    is used only for the per-vehicle detail view.
    """
    import numpy as np

    df = fleet_df if fleet_df is not None else _load_fleet_df()
    with timed(log, "score_fleet", n=len(df)):
        km = df["annual_km"].to_numpy(dtype=float)
        daily = df["avg_daily_range_km"].to_numpy(dtype=float)
        payload = df["payload_kg"].to_numpy(dtype=float)

        names = list(config.EV_CATALOG)
        wr, wp, wroi = (config.READINESS_WEIGHTS[k] for k in ("range_fit", "payload_fit", "roi"))

        # Per-EV fit components, stacked (n_ev, n_vehicles).
        rf_stack, pf_stack, price_row, cpk_row = [], [], [], []
        for name in names:
            ev = config.EV_CATALOG[name]
            rf = np.clip((ev["range_km"] * 0.8) / np.maximum(daily, 1.0), 0.0, 1.0)
            pf = np.where(payload <= ev["payload_kg"], 1.0,
                          np.clip(ev["payload_kg"] / np.maximum(payload, 1.0), 0.0, 1.0))
            rf_stack.append(rf); pf_stack.append(pf)
            price_row.append(ev["price_inr"]); cpk_row.append(ev["cost_per_km"])
        rf_stack = np.vstack(rf_stack); pf_stack = np.vstack(pf_stack)
        fit = wr * rf_stack + wp * pf_stack
        best = np.argmax(fit, axis=0)

        rows_idx = np.arange(len(df))
        rf_sel = rf_stack[best, rows_idx]
        pf_sel = pf_stack[best, rows_idx]
        price_sel = np.array(price_row)[best]
        cpk_sel = np.array(cpk_row)[best]

        annual_savings = km * (config.DIESEL_COST_PER_KM - cpk_sel)
        payback = np.where(annual_savings > 0, price_sel / np.maximum(annual_savings, 1e-9),
                           config.ROI_HORIZON_YEARS * 5.0)
        roi_fit = np.clip(np.where(payback <= 2, 1.0,
                                   np.where(payback >= 10, 0.0, (10 - payback) / 8.0)), 0.0, 1.0)
        score = np.clip(100.0 * (wr * rf_sel + wp * pf_sel + wroi * roi_fit), 0.0, 100.0)

        # Confidence (vectorised form of _confidence()).
        def _dec(x):
            return 1.0 - 4.0 * x * (1.0 - x)
        fit_conf = 0.5 * (_dec(rf_sel) + _dec(pf_sel))
        pay_conf = np.clip((10 - payback) / 7.0, 0.0, 1.0)
        confidence = np.clip(0.6 * fit_conf + 0.4 * pay_conf, 0.0, 1.0)

        # 5-year TCO saving (vectorised).
        t = config.TCO_INPUTS
        yrs = config.ROI_HORIZON_YEARS
        diesel_tco = (km * config.DIESEL_COST_PER_KM * yrs
                      + km * t["diesel_maintenance_per_km"] * yrs
                      + t["diesel_insurance_per_year"] * yrs)
        ev_tco = (price_sel + km * cpk_sel * yrs + km * t["ev_maintenance_per_km"] * yrs
                  + t["ev_insurance_per_year"] * yrs - price_sel * t["ev_residual_value_frac"])
        tco_savings = diesel_tco - ev_tco

        out = pd.DataFrame({
            "vehicle_id": df["vehicle_id"].to_numpy(),
            "vehicle_type": df["vehicle_type"].to_numpy(),
            "annual_km": km.astype(int),
            "avg_daily_range_km": daily.astype(int),
            "payload_kg": payload.astype(int),
            "duty_cycle": df["duty_cycle"].to_numpy(),
            "depot": df["depot"].to_numpy() if "depot" in df else "",
            "readiness_score": np.round(score, 1),
            "confidence": np.round(confidence, 2),
            "ev_match": np.array(names)[best],
            "range_fit": np.round(rf_sel, 3),
            "payload_fit": np.round(pf_sel, 3),
            "payback_years": np.round(payback, 2),
            "annual_savings_inr": np.round(annual_savings, 0),
            "five_year_savings_inr": np.round(annual_savings * yrs, 0),
            "tco_savings_5yr_inr": np.round(tco_savings, 0),
        })
    return out.sort_values("readiness_score", ascending=False).reset_index(drop=True)


def fleet_summary(scored: pd.DataFrame | None = None) -> dict:
    df = scored if scored is not None else score_fleet()
    ready = df[df["readiness_score"] >= 60]
    return {
        "total_vehicles": int(len(df)),
        "ready_now": int(len(ready)),
        "avg_readiness_score": round(float(df["readiness_score"].mean()), 1),
        "avg_confidence": round(float(df["confidence"].mean()), 2),
        "total_annual_savings_inr": round(float(df["annual_savings_inr"].sum()), 0),
        "total_five_year_savings_inr": round(float(df["five_year_savings_inr"].sum()), 0),
        "top_vehicle_id": df.iloc[0]["vehicle_id"],
    }


def kpis(scored: pd.DataFrame | None = None) -> list[KPI]:
    df = scored if scored is not None else score_fleet()
    s = fleet_summary(df)
    return [
        KPI("Ready to electrify now", f"{s['ready_now']}", f"/ {s['total_vehicles']}",
            "Vehicles scoring ≥60 — the immediate switch list.", "good"),
        KPI("5-year running savings", rupees(s["total_five_year_savings_inr"]), "",
            "Diesel-vs-EV fuel saving across the fleet over 5 years.", "good"),
        KPI("Avg readiness confidence", f"{s['avg_confidence']*100:.0f}", "%",
            "How decisive the recommendations are — lower means more borderline cases.",
            tone_for(s["avg_confidence"], 0.6, 0.4, higher_is_worse=False)),
        KPI("Avg readiness score", f"{s['avg_readiness_score']:.0f}", "/ 100",
            "Fleet-wide electrification readiness.",
            tone_for(s["avg_readiness_score"], 50, 35, higher_is_worse=False)),
    ]


if __name__ == "__main__":
    scored = score_fleet()
    print(scored.head(10).to_string(index=False))
    print("\nSummary:", fleet_summary(scored))
