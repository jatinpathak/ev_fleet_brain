"""Engine 7 (NEW, v3) - Scenario Simulation Engine  ⭐ innovation centerpiece.

A decision-support simulator: pick a scenario and a magnitude, and every
downstream KPI is recomputed from the real engines so the operator sees the
before/after impact. This maps directly to econometric scenario modelling and
is the headline "what-if" capability.

Scenarios
---------
* supplier_disruption(material, severity)      -> supply risk + ₹ exposure
* accelerated_degradation(fade_multiplier)     -> RUL, replacement cost, availability
* tariff_change(pct)                           -> charging cost + carbon-cost
* fleet_expansion(add_vehicles)                -> readiness, capex, CO₂, savings

Every scenario returns baseline vs scenario KPIs with signed deltas. All figures
are illustrative decision-support estimates, labelled as such in the UI — this
is a simulation, not a forecast.

Public API
----------
* run(name, **params)  -> {"scenario","params","deltas","narrative_facts", ...}
* SCENARIOS            -> registry of {name: (label, param_spec)}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from core.logging_config import get_logger, timed

log = get_logger(__name__)


def _delta(before: float, after: float, lower_is_better: bool = False) -> dict:
    change = after - before
    pct = (change / before * 100.0) if before not in (0, 0.0) else 0.0
    good = (change < 0) if lower_is_better else (change > 0)
    return {
        "before": round(before, 2), "after": round(after, 2),
        "delta": round(change, 2), "pct": round(pct, 1),
        "direction": "up" if change > 0 else "down" if change < 0 else "flat",
        "favourable": bool(good) if change != 0 else None,
    }


# ---------------------------------------------------------------------------
# 1. Supplier disruption
# ---------------------------------------------------------------------------
def supplier_disruption(material: str = "Lithium", severity: float = 1.0) -> dict:
    """A critical material (partly/fully) goes offline -> supply-risk impact."""
    from engines import engine_supply_chain as sc

    base_df = sc.load_suppliers()
    base = sc.supply_risk_summary(base_df)

    # Remove `severity` fraction of that material's supply capacity. At full
    # severity the material's suppliers are effectively lost (single/zero source).
    df = base_df.copy()
    df["annual_volume"] = df["annual_volume"].astype(float)   # allow fractional scaling
    mask = df["material"] == material
    df.loc[mask, "annual_volume"] = (df.loc[mask, "annual_volume"] * (1 - severity)).clip(lower=1)
    if severity >= 0.99:  # fully offline -> collapse to a single stub supplier
        keep = df[~mask]
        stub = df[mask].head(1)
        df = pd.concat([keep, stub], ignore_index=True)
    scen = sc.supply_risk_summary(df)

    return {
        "scenario": "supplier_disruption",
        "params": {"material": material, "severity": severity},
        "deltas": {
            "Overall supply risk": _delta(base["overall_risk_score"],
                                          scen["overall_risk_score"], lower_is_better=True),
            "Value at risk (₹)": _delta(base["value_at_risk_inr"],
                                        scen["value_at_risk_inr"], lower_is_better=True),
            "% single-sourced": _delta(base["pct_single_sourced"],
                                       scen["pct_single_sourced"], lower_is_better=True),
        },
        "narrative_facts": {
            "material": material, "severity_pct": round(severity * 100),
            "new_single_sourced": scen["single_sourced_materials"],
            "risk_before": base["overall_risk_score"], "risk_after": scen["overall_risk_score"],
        },
    }


# ---------------------------------------------------------------------------
# 2. Accelerated battery degradation
# ---------------------------------------------------------------------------
def accelerated_degradation(fade_multiplier: float = 1.5) -> dict:
    """Higher fade rate -> shorter RUL, more replacements, lower availability."""
    from engines import engine_battery as eb

    snap = eb.operational_snapshot()
    thresh = config.RUL_REPLACE_THRESHOLD_CYCLES
    ruls = snap["remaining_useful_life"].to_numpy(dtype=float)

    scen_ruls = ruls / max(fade_multiplier, 1e-6)   # faster fade => less life left
    base_repl = int((ruls < thresh).sum())
    scen_repl = int((scen_ruls < thresh).sum())
    cost = config.PACK_REPLACEMENT_COST_INR

    # Availability proxy: fraction of packs above the replacement threshold.
    base_avail = 100.0 * (ruls >= thresh).mean()
    scen_avail = 100.0 * (scen_ruls >= thresh).mean()

    return {
        "scenario": "accelerated_degradation",
        "params": {"fade_multiplier": fade_multiplier},
        "deltas": {
            "Mean RUL (cycles)": _delta(float(ruls.mean()), float(scen_ruls.mean())),
            "Packs to replace": _delta(base_repl, scen_repl, lower_is_better=True),
            "Replacement cost (₹)": _delta(base_repl * cost, scen_repl * cost, lower_is_better=True),
            "Fleet availability (%)": _delta(base_avail, scen_avail),
        },
        "narrative_facts": {
            "fade_multiplier": fade_multiplier,
            "extra_replacements": scen_repl - base_repl,
            "extra_cost_inr": (scen_repl - base_repl) * cost,
        },
    }


# ---------------------------------------------------------------------------
# 3. Electricity tariff change
# ---------------------------------------------------------------------------
def tariff_change(pct: float = 20.0) -> dict:
    """Shift all charging tariffs by `pct`% -> charging cost impact."""
    from engines import engine_maintenance as em

    mult = 1 + pct / 100.0
    base = em.optimise_charging()
    scen = em.optimise_charging(tariff_multiplier=mult)

    base_annual = base["optimised_cost_inr"] * 365
    scen_annual = scen["optimised_cost_inr"] * 365
    return {
        "scenario": "tariff_change",
        "params": {"pct": pct},
        "deltas": {
            "Charging cost (₹/day)": _delta(base["optimised_cost_inr"],
                                            scen["optimised_cost_inr"], lower_is_better=True),
            "Charging cost (₹/yr)": _delta(base_annual, scen_annual, lower_is_better=True),
            "Cost saved vs peak (%)": _delta(base["cost_saved_pct"], scen["cost_saved_pct"]),
        },
        "narrative_facts": {
            "pct": pct, "extra_annual_cost_inr": scen_annual - base_annual,
        },
    }


# ---------------------------------------------------------------------------
# 4. Fleet expansion
# ---------------------------------------------------------------------------
def fleet_expansion(add_vehicles: int = 100) -> dict:
    """Add N vehicles -> readiness, capex, CO₂ and savings trajectory.

    Linear projection from current per-vehicle averages (a labelled estimate,
    not a re-simulation of individual new vehicles).
    """
    from engines import engine_readiness as er
    from engines import engine_carbon as ec

    scored = er.score_fleet()
    carbon = ec.score_carbon()
    n = len(scored)
    ready_frac = (scored["readiness_score"] >= 60).mean()
    avg_save = scored["five_year_savings_inr"].mean()
    avg_co2 = carbon["savings_co2_kg"].mean() / 1000.0   # tonnes/yr per vehicle

    new_n = n + add_vehicles
    capex = ready_frac * add_vehicles * config.AVG_EV_CAPEX_INR

    return {
        "scenario": "fleet_expansion",
        "params": {"add_vehicles": add_vehicles},
        "deltas": {
            "Fleet size": _delta(n, new_n),
            "5-yr savings (₹)": _delta(avg_save * n, avg_save * new_n),
            "CO₂ avoided (t/yr)": _delta(avg_co2 * n, avg_co2 * new_n),
            "EV capex required (₹)": _delta(0, capex, lower_is_better=True),
        },
        "narrative_facts": {
            "add_vehicles": add_vehicles, "ready_frac_pct": round(ready_frac * 100),
            "capex_inr": capex, "extra_co2_t": avg_co2 * add_vehicles,
        },
    }


SCENARIOS = {
    "supplier_disruption": ("Supplier disruption",
                            {"material": config.CRITICAL_MATERIALS, "severity": (0.1, 1.0)}),
    "accelerated_degradation": ("Accelerated battery degradation",
                                {"fade_multiplier": (1.0, 3.0)}),
    "tariff_change": ("Electricity tariff change", {"pct": (-30.0, 50.0)}),
    "fleet_expansion": ("Fleet expansion", {"add_vehicles": (50, 5000)}),
}

_DISPATCH = {
    "supplier_disruption": supplier_disruption,
    "accelerated_degradation": accelerated_degradation,
    "tariff_change": tariff_change,
    "fleet_expansion": fleet_expansion,
}


def run(name: str, **params) -> dict:
    """Run a named scenario with keyword params; returns baseline/scenario deltas."""
    if name not in _DISPATCH:
        raise KeyError(f"Unknown scenario: {name}")
    with timed(log, "scenario", name=name):
        return _DISPATCH[name](**params)


if __name__ == "__main__":
    for nm in SCENARIOS:
        r = run(nm)
        print(nm, "->", {k: v["delta"] for k, v in r["deltas"].items()})
