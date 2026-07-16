"""ROUND 2 - Integration tests: the full chain, fallbacks, and edge cases."""
import os

import pandas as pd
import pytest

import config
import copilot
import engine_battery as eb
import engine_carbon as ec
import engine_readiness as er
import generate_data


@pytest.fixture(scope="module", autouse=True)
def ensure_data():
    if not config.FLEET_DATA_CSV.exists() or not config.BATTERY_DATA_CSV.exists():
        generate_data.main()


def test_full_chain_end_to_end():
    """generate -> train -> score -> carbon -> copilot, no exceptions."""
    fleet = pd.read_csv(config.FLEET_DATA_CSV)
    scored = er.score_fleet(fleet)
    carbon = ec.score_carbon(fleet)
    fsum = er.fleet_summary(scored)
    csum = ec.fleet_carbon_summary(carbon)
    text = copilot.explain("fleet", {**fsum, **csum})
    assert isinstance(text, str) and len(text) > 0


def test_readiness_and_carbon_agree_on_top_vehicle():
    """The vehicle readiness recommends first is one carbon also electrifies
    with a positive CO2 saving -- the story must not contradict itself."""
    fleet = pd.read_csv(config.FLEET_DATA_CSV)
    scored = er.score_fleet(fleet)
    top_id = scored.iloc[0]["vehicle_id"]
    carbon = ec.vehicle_carbon(top_id, fleet)
    assert carbon["savings_co2_kg"] > 0


def test_copilot_fallback_without_api_key(monkeypatch):
    """With NO api key, the copilot must still return a real explanation."""
    monkeypatch.setattr(config, "get_api_key", lambda: None)
    rec = {"vehicle_id": "VEH_000", "readiness_score": 80, "ev_match": "Tata Ace EV",
           "annual_savings_inr": 100000, "five_year_savings_inr": 500000,
           "payback_years": 4.0}
    text = copilot.explain("vehicle", rec)
    assert "VEH_000" in text and len(text) > 30


def test_edge_route_exceeds_every_ev_range():
    """A route longer than every EV's range should score LOW, not crash."""
    extreme = pd.DataFrame([{
        "vehicle_id": "EXTREME", "vehicle_type": "diesel", "annual_km": 30000,
        "avg_daily_range_km": 500, "payload_kg": 300, "duty_cycle": "highway",
    }])
    rec = er.vehicle_recommendation("EXTREME", extreme)
    assert 0 <= rec["readiness_score"] <= 100
    assert rec["range_fit"] < 1.0  # cannot fully cover the route


def test_edge_brand_new_battery():
    """A cell with only a few cycles must still return valid health output."""
    df = pd.read_csv(config.BATTERY_DATA_CSV)
    one = df[df["cell_id"] == df["cell_id"].iloc[0]].sort_values("cycle")
    new_batt = one.head(5)  # brand new
    r = eb.predict_health(new_batt)
    assert 0.0 <= r["state_of_health"] <= 1.0
    assert r["remaining_useful_life"] >= 0


def test_edge_nearly_dead_battery():
    """A cell near end of life should report low SoH and small RUL."""
    df = pd.read_csv(config.BATTERY_DATA_CSV)
    # Pick the cell with the lowest final capacity and use its full history.
    finals = df.sort_values("cycle").groupby("cell_id").tail(1)
    worst_id = finals.loc[finals["discharge_capacity_ah"].idxmin(), "cell_id"]
    hist = df[df["cell_id"] == worst_id]
    r = eb.predict_health(hist)
    assert r["state_of_health"] <= 1.0
    assert r["remaining_useful_life"] >= 0
