"""ROUND 4 - Scripted demo walkthrough (the exact path shown to judges).

home -> battery page (pick a cell, read explanation) -> readiness page
(sort, open top vehicle) -> carbon page (read the story). Each step must
produce sensible output.
"""
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


def test_demo_walkthrough():
    # ---- HOME: headline numbers exist and are sensible ----
    fleet = pd.read_csv(config.FLEET_DATA_CSV)
    scored = er.score_fleet(fleet)
    fsum = er.fleet_summary(scored)
    assert fsum["total_vehicles"] == 300
    assert fsum["total_five_year_savings_inr"] > 0

    # ---- BATTERY: pick a cell, read the explanation ----
    batt = pd.read_csv(config.BATTERY_DATA_CSV)
    cell_id = sorted(batt["cell_id"].unique())[0]
    health = eb.predict_health(batt[batt["cell_id"] == cell_id])
    assert health["predicted_cycle_life"] > 0
    battery_text = copilot.explain("battery", health)
    assert len(battery_text) > 30

    # ---- READINESS: sort, open the top vehicle ----
    top_id = scored.iloc[0]["vehicle_id"]
    rec = er.vehicle_recommendation(top_id, fleet)
    assert rec["readiness_score"] == scored.iloc[0]["readiness_score"]
    assert rec["annual_savings_inr"] > 0
    vehicle_text = copilot.explain("vehicle", rec)
    assert top_id in vehicle_text

    # ---- CARBON: read the story ----
    carbon = ec.score_carbon(fleet)
    csum = ec.fleet_carbon_summary(carbon)
    assert csum["total_savings_co2_tonnes"] > 0
    story = copilot.explain("fleet", {**fsum, **csum})
    assert len(story) > 30


def test_top_vehicle_consistent_across_engines():
    """The demo story stays coherent: the top readiness pick also saves CO2."""
    fleet = pd.read_csv(config.FLEET_DATA_CSV)
    scored = er.score_fleet(fleet)
    top_id = scored.iloc[0]["vehicle_id"]
    assert ec.vehicle_carbon(top_id, fleet)["savings_co2_kg"] > 0
