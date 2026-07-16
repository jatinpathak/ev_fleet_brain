"""ROUND 1 - Component tests for the three engines."""
import math

import numpy as np
import pandas as pd
import pytest

import config
import engine_battery as eb
import engine_carbon as ec
import engine_readiness as er
import generate_data


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
    assert 0 < metrics["mape_pct"] < 100  # a sane error rate


def test_predict_health_keys_and_ranges(battery_df):
    cells = battery_df["cell_id"].unique()[:10]
    assert len(cells) >= 10
    for cell_id in cells:
        hist = battery_df[battery_df["cell_id"] == cell_id]
        r = eb.predict_health(hist)
        assert {"state_of_health", "predicted_cycle_life",
                "remaining_useful_life"}.issubset(r)
        assert 0.0 <= r["state_of_health"] <= 1.0
        assert r["remaining_useful_life"] >= 0
        assert r["predicted_cycle_life"] > 0


def test_health_status_thresholds():
    assert eb.health_status(0.95) == "healthy"
    assert eb.health_status(0.85) == "degraded"
    assert eb.health_status(0.70) == "critical"


# --------------------------- Readiness -------------------------------------
def test_every_vehicle_scored_in_range(fleet_df):
    scored = er.score_fleet(fleet_df)
    assert len(scored) == len(fleet_df)
    assert scored["readiness_score"].between(0, 100).all()


def test_payback_positive_and_finite(fleet_df):
    scored = er.score_fleet(fleet_df)
    assert (scored["payback_years"] > 0).all()
    assert np.isfinite(scored["payback_years"]).all()


def test_higher_score_means_better_fit(fleet_df):
    """A vehicle with a clearly better route/payload/ROI must outscore a worse one."""
    good = pd.DataFrame([{
        "vehicle_id": "GOOD", "vehicle_type": "diesel", "annual_km": 90000,
        "avg_daily_range_km": 120, "payload_kg": 400, "duty_cycle": "urban",
    }])
    bad = pd.DataFrame([{
        "vehicle_id": "BAD", "vehicle_type": "diesel", "annual_km": 9000,
        "avg_daily_range_km": 480, "payload_kg": 1150, "duty_cycle": "highway",
    }])
    g = er.vehicle_recommendation("GOOD", good)
    b = er.vehicle_recommendation("BAD", bad)
    assert g["readiness_score"] > b["readiness_score"]


# --------------------------- Carbon ----------------------------------------
def test_carbon_savings_non_negative(fleet_df):
    carbon = ec.score_carbon(fleet_df)
    assert (carbon["savings_co2_kg"] >= 0).all()
    assert (carbon["savings_cost_inr"] >= 0).all()


def test_carbon_totals_equal_sum_of_parts(fleet_df):
    """Fleet totals must equal the sum of per-vehicle values (no double count)."""
    carbon = ec.score_carbon(fleet_df)
    summary = ec.fleet_carbon_summary(carbon)
    expected_tonnes = round(carbon["savings_co2_kg"].sum() / 1000, 1)
    assert summary["total_savings_co2_tonnes"] == expected_tonnes
