"""ROUND 2 (v3) - Planner, scenarios, recommendations, infra."""
import pandas as pd
import pytest

import config
import generate_data
from core import feature_store as fs
from core import monitoring, orchestrator, recommend
from engines import engine_battery as eb
from engines import engine_carbon as ec
from engines import engine_maintenance as em
from engines import engine_quality as eq
from engines import engine_readiness as er
from engines import engine_scenario as es
from engines import engine_supply_chain as sc


@pytest.fixture(scope="module", autouse=True)
def ensure_data():
    if not config.SUPPLIERS_CSV.exists() or not config.MAINTENANCE_CSV.exists():
        generate_data.main()


# --------------------------- Planner ---------------------------------------
def test_planner_produces_bounded_visible_plan():
    p = orchestrator.plan("battery health, supply risk and which vehicles to electrify",
                          max_steps=6)
    assert p["agents_called"]
    # A synthesis step is always appended, and steps are capped by max_steps + 1.
    assert len(p["steps"]) <= p["max_steps"] + 1
    assert p["steps"][-1]["intent"] == "synthesis"
    assert all("result" in s for s in p["steps"])


def test_planner_loop_guard_caps_agents():
    p = orchestrator.plan("battery supply maintenance carbon readiness quality", max_steps=3)
    assert len(p["engines_called"]) <= 3


def test_route_backward_compatible():
    r = orchestrator.route("carbon overview")
    assert {"query", "intent", "engines_called", "data", "plan"} <= set(r)


# --------------------------- Scenarios -------------------------------------
@pytest.mark.parametrize("name", list(es.SCENARIOS))
def test_scenario_returns_signed_deltas(name):
    res = es.run(name)
    assert res["deltas"]
    for d in res["deltas"].values():
        assert {"before", "after", "delta", "pct", "direction"} <= set(d)


def test_supplier_disruption_raises_risk():
    res = es.run("supplier_disruption", material="Lithium", severity=1.0)
    assert res["deltas"]["Overall supply risk"]["after"] >= \
        res["deltas"]["Overall supply risk"]["before"]


def test_degradation_reduces_rul():
    res = es.run("accelerated_degradation", fade_multiplier=2.0)
    d = res["deltas"]["Mean RUL (cycles)"]
    assert d["after"] < d["before"]


# --------------------------- Recommendations -------------------------------
def test_every_engine_recommendation_complete():
    recs = [
        eb.recommendation(eb.predict_health(
            pd.read_csv(config.BATTERY_DATA_CSV).pipe(lambda d: d[d.cell_id == "CELL_010"]))),
        er.recommendation(),
        sc.recommendation(),
        em.recommendation(),
    ]
    for r in recs:
        assert isinstance(r, recommend.Recommendation)
        assert 0.0 <= r.confidence <= 1.0
        assert r.reasoning and r.impact
        assert len(r.alternatives) >= 1


# --------------------------- Graph / passport / RCA ------------------------
def test_risk_propagation_blast_radius():
    prop = sc.propagate_risk("S10")   # a Tier-3 cobalt miner
    assert prop["n_affected"] >= 1
    assert 0.0 <= prop["propagated_risk"] <= 1.0


def test_battery_passport_fields():
    pp = eb.battery_passport("CELL_005")
    assert pp["chemistry"] == "LFP"
    assert pp["warranty_status"] in {"in warranty", "out of warranty"}
    assert "lifecycle_stage" in pp


def test_quality_rca_hints():
    hints = eq.root_cause_hints()
    assert hints and all({"signal", "likely_cause", "action"} <= set(h) for h in hints)


# --------------------------- Hourly grid carbon ----------------------------
def test_smart_charging_saves_offpeak():
    smart = ec.smart_charging_carbon()
    assert smart["off_peak_co2_kg_day"] <= smart["peak_co2_kg_day"]
    assert smart["co2_saved_kg_day"] >= 0
    assert len(ec.hourly_grid_intensity()) == 24


# --------------------------- Feature store / monitoring --------------------
def test_feature_store_build_and_read():
    counts = fs.build()
    assert counts["fleet_scores"] > 0
    assert "suppliers" in fs.tables()
    assert len(fs.read("fleet_scores")) == counts["fleet_scores"]


def test_monitoring_psi_reacts_to_drift():
    stable = monitoring.drift_report("readiness_score", drift_shift=0.0)
    drifted = monitoring.drift_report("readiness_score", drift_shift=0.3)
    assert drifted["psi"] > stable["psi"]
    assert drifted["drifted"]


# --------------------------- REST API --------------------------------------
def test_api_endpoints():
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    assert c.get("/").status_code == 200
    assert c.get("/fleet/summary").status_code == 200
    r = c.post("/battery/predict", json={"cell_id": "CELL_010"})
    assert r.status_code == 200 and "recommendation" in r.json()
    assert c.post("/battery/predict", json={"cell_id": "NOPE"}).status_code == 404
    r = c.post("/scenario/run", json={"name": "tariff_change", "params": {"pct": 10}})
    assert r.status_code == 200 and r.json()["deltas"]
