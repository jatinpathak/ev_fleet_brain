"""ROUND 2 - Integration tests: the full chain, fallbacks, and edge cases."""
import pandas as pd
import pytest

import config
import copilot
import generate_data
from core import orchestrator
from engines import engine_battery as eb
from engines import engine_carbon as ec
from engines import engine_readiness as er


@pytest.fixture(scope="module", autouse=True)
def ensure_data():
    if not config.FLEET_DATA_CSV.exists() or not config.BATTERY_DATA_CSV.exists():
        generate_data.main()


def test_full_chain_end_to_end():
    fleet = pd.read_csv(config.FLEET_DATA_CSV)
    scored = er.score_fleet(fleet)
    carbon = ec.score_carbon(fleet)
    text = copilot.explain("fleet", {**er.fleet_summary(scored),
                                     **ec.fleet_carbon_summary(carbon)})
    assert isinstance(text, str) and len(text) > 0


def test_orchestrator_routes_by_intent():
    assert "battery" in orchestrator.classify("how healthy is the battery?")
    assert "supply_chain" in orchestrator.classify("what's our lithium supplier risk?")
    assert "carbon" in orchestrator.classify("how much CO2 do we save?")
    routed = orchestrator.route("which vehicles should we electrify first?")
    assert "readiness" in routed["engines_called"]
    assert routed["data"]


def test_copilot_answer_fallback_without_api_key(monkeypatch):
    monkeypatch.setattr(config, "get_api_key", lambda: None)
    res = copilot.answer("Give me a supply chain and carbon overview")
    assert isinstance(res["answer"], str) and len(res["answer"]) > 20
    assert res["routed"]["engines_called"]


def test_copilot_explain_fallback_without_api_key(monkeypatch):
    monkeypatch.setattr(config, "get_api_key", lambda: None)
    rec = {"vehicle_id": "VEH_00000", "readiness_score": 80, "confidence": 0.8,
           "ev_match": "Tata Ace EV", "annual_savings_inr": 100000,
           "tco_savings_5yr_inr": 500000, "payback_years": 4.0}
    text = copilot.explain("vehicle", rec)
    assert "VEH_00000" in text and len(text) > 30


def test_readiness_and_carbon_agree_on_top_vehicle():
    fleet = pd.read_csv(config.FLEET_DATA_CSV)
    top_id = er.score_fleet(fleet).iloc[0]["vehicle_id"]
    assert ec.vehicle_carbon(top_id, fleet)["savings_co2_kg"] > 0


def test_edge_route_exceeds_every_ev_range():
    extreme = pd.DataFrame([{
        "vehicle_id": "EXTREME", "vehicle_type": "diesel", "annual_km": 30000,
        "avg_daily_range_km": 500, "payload_kg": 300, "duty_cycle": "highway", "depot": "Pune",
    }])
    rec = er.vehicle_recommendation("EXTREME", extreme)
    assert 0 <= rec["readiness_score"] <= 100
    assert rec["range_fit"] < 1.0


def test_edge_brand_new_battery():
    df = pd.read_csv(config.BATTERY_DATA_CSV)
    one = df[df["cell_id"] == df["cell_id"].iloc[0]].sort_values("cycle")
    r = eb.predict_health(one.head(5))
    assert 0.0 <= r["state_of_health"] <= 1.0
    assert r["remaining_useful_life"] >= 0
