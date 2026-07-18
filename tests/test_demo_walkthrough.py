"""ROUND 4 - Scripted demo walkthrough (the exact path shown to judges).

home -> battery -> readiness -> supply chain -> maintenance -> carbon,
each step producing sensible, non-contradictory output.
"""
import pandas as pd
import pytest

import config
import copilot
import generate_data
from engines import engine_battery as eb
from engines import engine_carbon as ec
from engines import engine_maintenance as em
from engines import engine_readiness as er
from engines import engine_supply_chain as sc


@pytest.fixture(scope="module", autouse=True)
def ensure_data():
    if not config.FLEET_DATA_CSV.exists() or not config.BATTERY_DATA_CSV.exists():
        generate_data.main()


def test_demo_walkthrough():
    # HOME
    fleet = pd.read_csv(config.FLEET_DATA_CSV)
    scored = er.score_fleet(fleet)
    fsum = er.fleet_summary(scored)
    assert fsum["total_vehicles"] == len(fleet)
    assert fsum["total_five_year_savings_inr"] > 0

    # BATTERY
    batt = pd.read_csv(config.BATTERY_DATA_CSV)
    cell_id = sorted(batt["cell_id"].unique())[0]
    health = eb.predict_health(batt[batt["cell_id"] == cell_id])
    assert health["predicted_cycle_life"] > 0
    assert len(copilot.explain("battery", health)) > 30

    # READINESS
    top_id = scored.iloc[0]["vehicle_id"]
    rec = er.vehicle_recommendation(top_id, fleet)
    assert rec["readiness_score"] == scored.iloc[0]["readiness_score"]
    assert top_id in copilot.explain("vehicle", rec)

    # SUPPLY CHAIN
    summary = sc.supply_risk_summary()
    assert 0 <= summary["overall_risk_score"] <= 100
    assert summary["top_vulnerabilities"]

    # MAINTENANCE
    msum = em.schedule_maintenance()
    assert msum["downtime_reduction_pct"] >= 0

    # CARBON
    csum = ec.fleet_carbon_summary(ec.score_carbon(fleet))
    assert csum["total_savings_co2_tonnes"] > 0
    assert len(copilot.explain("fleet", {**fsum, **csum})) > 30


def test_story_consistent_across_engines():
    fleet = pd.read_csv(config.FLEET_DATA_CSV)
    top_id = er.score_fleet(fleet).iloc[0]["vehicle_id"]
    assert ec.vehicle_carbon(top_id, fleet)["savings_co2_kg"] > 0
