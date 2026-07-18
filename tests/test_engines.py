"""ROUND 1 - Component tests for the core domain engines."""
import math

import numpy as np
import pandas as pd
import pytest

import config
import generate_data
from engines import engine_battery as eb
from engines import engine_carbon as ec
from engines import engine_readiness as er


@pytest.fixture(scope="module", autouse=True)
def ensure_data():
    if not config.FLEET_DATA_CSV.exists() or not config.BATTERY_DATA_CSV.exists():
        generate_data.main()


@pytest.fixture(scope="module")
def battery_df():
    return pd.read_csv(config.BATTERY_DATA_CSV)


@pytest.fixture(scope="module")
def fleet_df():
    return pd.read_csv(config.FLEET_DATA_CSV)


# --------------------------- Battery ---------------------------------------
def test_battery_model_trains_and_reports_rmse():
    _, metrics = eb.train_model(verbose=False)
    assert metrics["rmse_cycles"] > 0
    assert math.isfinite(metrics["rmse_cycles"])
    assert 0 < metrics["mape_pct"] < 100


def test_predict_health_keys_and_prediction_interval(battery_df):
    for cell_id in battery_df["cell_id"].unique()[:10]:
        r = eb.predict_health(battery_df[battery_df["cell_id"] == cell_id])
        assert {"state_of_health", "predicted_cycle_life",
                "predicted_cycle_life_pi", "top_drivers"}.issubset(r)
        assert 0.0 <= r["state_of_health"] <= 1.0
        lo, hi = r["predicted_cycle_life_pi"]
        # The point estimate lies inside its own prediction interval.
        assert lo <= r["predicted_cycle_life"] <= hi
        assert lo <= hi


def test_health_status_thresholds():
    assert eb.health_status(0.95) == "healthy"
    assert eb.health_status(0.85) == "degraded"
    assert eb.health_status(0.70) == "critical"


def test_anomaly_detection_flags_some_cells(battery_df):
    an = eb.detect_anomalies(battery_df)
    assert len(an) == battery_df["cell_id"].nunique()
    # With ~8% contamination we expect at least one flag but not everything.
    assert 0 < int(an["anomaly"].sum()) < len(an)


# --------------------------- Readiness -------------------------------------
def test_every_vehicle_scored_with_confidence(fleet_df):
    scored = er.score_fleet(fleet_df)
    assert len(scored) == len(fleet_df)
    assert scored["readiness_score"].between(0, 100).all()
    assert scored["confidence"].between(0, 1).all()


def test_tco_breakdown_present_and_signed(fleet_df):
    rec = er.vehicle_recommendation(fleet_df["vehicle_id"].iloc[0], fleet_df)
    tco = rec["tco_breakdown"]
    assert {"diesel", "ev", "tco_savings_5yr"}.issubset(tco)
    assert tco["ev"]["residual_credit"] <= 0  # a credit reduces EV cost


def test_higher_score_means_better_fit(fleet_df):
    good = pd.DataFrame([{
        "vehicle_id": "GOOD", "vehicle_type": "diesel", "annual_km": 90000,
        "avg_daily_range_km": 120, "payload_kg": 400, "duty_cycle": "urban", "depot": "Pune",
    }])
    bad = pd.DataFrame([{
        "vehicle_id": "BAD", "vehicle_type": "diesel", "annual_km": 9000,
        "avg_daily_range_km": 480, "payload_kg": 1150, "duty_cycle": "highway", "depot": "Delhi",
    }])
    assert (er.vehicle_recommendation("GOOD", good)["readiness_score"]
            > er.vehicle_recommendation("BAD", bad)["readiness_score"])


# --------------------------- Carbon ----------------------------------------
def test_carbon_scope_split_and_non_negative(fleet_df):
    carbon = ec.score_carbon(fleet_df)
    for col in ("scope1_diesel_kg", "scope2_grid_kg", "scope3_ev_upstream_kg"):
        assert col in carbon.columns
    assert (carbon["savings_co2_kg"] >= 0).all()


def test_carbon_totals_equal_sum_of_parts(fleet_df):
    carbon = ec.score_carbon(fleet_df)
    summary = ec.fleet_carbon_summary(carbon)
    expected = round(carbon["savings_co2_kg"].sum() / 1000, 1)
    assert summary["total_savings_co2_tonnes"] == expected


def test_every_engine_exposes_kpis(fleet_df):
    for kpi_list in (eb.kpis(), er.kpis(er.score_fleet(fleet_df)), ec.kpis()):
        assert len(kpi_list) >= 3
        assert all(k.label and k.why for k in kpi_list)
