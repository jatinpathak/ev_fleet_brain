"""ROUND 2 - Tests for the v2 engines: supply chain, maintenance, quality, core."""
import numpy as np
import pytest

import config
import generate_data
from core import explain, uncertainty
from engines import engine_maintenance as em
from engines import engine_quality as eq
from engines import engine_supply_chain as sc


@pytest.fixture(scope="module", autouse=True)
def ensure_data():
    if not config.SUPPLIERS_CSV.exists() or not config.MAINTENANCE_CSV.exists():
        generate_data.main()


# --------------------------- Supply chain ----------------------------------
def test_concentration_index_in_range():
    conc = sc.supplier_concentration_risk()
    assert (conc["hhi"] >= 0).all() and (conc["hhi"] <= 1).all()
    # A single-source material must have HHI == 1.
    single = conc[conc["single_source"]]
    if len(single):
        assert (single["hhi"] == 1.0).all()


def test_geopolitical_exposure_bounded():
    geo = sc.geopolitical_exposure_score()
    assert 0.0 <= geo["weighted_exposure"] <= 1.0


def test_traceability_walks_tiers():
    tr = sc.material_traceability("CELL_010")
    assert tr["cell_maker"] is not None
    assert isinstance(tr["upstream"], list) and len(tr["upstream"]) >= 1
    # Every upstream node carries a country risk in [0, 1].
    for u in tr["upstream"]:
        assert 0.0 <= u["country_risk"] <= 1.0


def test_supply_summary_and_graph():
    summary = sc.supply_risk_summary()
    assert 0 <= summary["overall_risk_score"] <= 100
    assert len(summary["top_vulnerabilities"]) <= 3
    graph = sc.build_graph()
    assert graph["n_nodes"] > 0 and graph["n_edges"] > 0


# --------------------------- Maintenance -----------------------------------
def test_maintenance_schedule_respects_horizon():
    res = em.schedule_maintenance()
    assert res["downtime_reduction_pct"] >= 0
    assert (res["schedule"]["assigned_day"] <= config.WORKSHOP_DAYS).all()
    assert (res["schedule"]["lateness"] >= 0).all()


def test_charging_optimisation_beats_peak_baseline():
    c = em.optimise_charging()
    assert c["optimised_cost_inr"] <= c["baseline_cost_inr"]
    assert 0 <= c["charger_utilisation_pct"] <= 100


# --------------------------- Quality (Tier 3) ------------------------------
def test_spc_flags_out_of_control_points():
    spc = eq.spc_series()
    assert "out_of_control" in spc.columns
    # The injected late drift should trip at least one control-limit breach.
    assert int(spc["out_of_control"].sum()) >= 1


def test_incoming_quality_summary():
    iq = eq.incoming_quality()
    assert "ppm" in iq.columns and (iq["ppm"] >= 0).all()


# --------------------------- Core services ---------------------------------
def test_conformal_halfwidth_covers():
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 1.0, 500)
    hw = uncertainty.conformal_halfwidth(resid, 0.90)
    covered = np.mean(np.abs(resid) <= hw)
    assert covered >= 0.85  # approximately the requested coverage


def test_explain_top_drivers_fallback():
    class _Dummy:
        feature_importances_ = np.array([0.6, 0.3, 0.1])
    out = explain.top_drivers(_Dummy(), np.array([1.0, 2.0, 3.0]),
                              ["a", "b", "c"], k=2)
    assert len(out["drivers"]) == 2
    assert out["drivers"][0][0] == "a"  # highest importance first
