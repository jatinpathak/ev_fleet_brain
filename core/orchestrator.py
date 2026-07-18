"""Agent Orchestrator - routes a natural-language query across the engines.

This is what makes the "multi-agent / agentic" claim TRUE rather than cosmetic:
the orchestrator reads a query, decides which engine(s) to call (deterministic
intent routing + tool calls — NOT an open-ended LLM loop), gathers their
structured outputs, and returns a unified result. The copilot then turns that
structured result into plain English with one LLM call.

Honesty: this is a router + explainer, not an autonomous agent. It is labelled
that way in the UI.

Public API
----------
* route(query)          -> {"query","intent","engines_called","data","summary"}
* answer_payload(query)  -> the structured dict the copilot explains
"""
from __future__ import annotations

import re

from core.logging_config import get_logger, timed

log = get_logger(__name__)

# Intent -> trigger keywords (checked in priority order).
INTENT_KEYWORDS = {
    "battery": ["battery", "cell", "soh", "state of health", "rul", "remaining",
                "degrad", "anomaly", "capacity", "cycle life"],
    "supply_chain": ["supply", "supplier", "lithium", "cobalt", "nickel",
                     "material", "geopolit", "traceab", "concentration",
                     "single source", "sourcing"],
    "maintenance": ["maintenance", "service", "workshop", "downtime", "schedule",
                    "charging", "charger", "tariff", "depot"],
    "carbon": ["carbon", "co2", "emission", "scope", "green", "credit",
               "decarbon"],
    "readiness": ["readiness", "electrify", "electrification", "switch", "ev",
                  "candidate", "which vehicle", "tco", "payback", "roi"],
    "quality": ["quality", "defect", "spc", "manufactur", "control chart", "ppm"],
}

_CELL_RE = re.compile(r"CELL[_ ]?(\d+)", re.IGNORECASE)
_VEH_RE = re.compile(r"VEH[_ ]?(\d+)", re.IGNORECASE)


def classify(query: str) -> list[str]:
    """Return the intents a query matches, most specific first (may be several)."""
    q = query.lower()
    hits = [intent for intent, kws in INTENT_KEYWORDS.items()
            if any(k in q for k in kws)]
    return hits


def _battery_payload(query: str) -> dict:
    from engines import engine_battery as eb
    import pandas as pd
    import config

    df = pd.read_csv(config.BATTERY_DATA_CSV)
    m = _CELL_RE.search(query)
    if m:
        cell_id = f"CELL_{int(m.group(1)):03d}"
        if cell_id in set(df["cell_id"]):
            return {"scope": "cell", "cell": eb.predict_health(df[df.cell_id == cell_id])}
    anomalies = eb.detect_anomalies(df)
    return {"scope": "fleet",
            "kpis": [k.as_dict() for k in eb.kpis(df)],
            "n_anomalies": int(anomalies["anomaly"].sum()),
            "top_anomalies": anomalies.head(3)[["cell_id", "anomaly_score"]].to_dict("records")}


def _readiness_payload(query: str) -> dict:
    from engines import engine_readiness as er
    scored = er.score_fleet()
    m = _VEH_RE.search(query)
    if m:
        vid = f"VEH_{int(m.group(1)):05d}"
        try:
            return {"scope": "vehicle", "vehicle": er.vehicle_recommendation(vid)}
        except KeyError:
            pass
    return {"scope": "fleet", "summary": er.fleet_summary(scored),
            "kpis": [k.as_dict() for k in er.kpis(scored)],
            "top5": scored.head(5)[["vehicle_id", "readiness_score", "ev_match",
                                    "confidence", "payback_years"]].to_dict("records")}


def _carbon_payload(query: str) -> dict:
    from engines import engine_carbon as ec
    return {"summary": ec.fleet_carbon_summary(),
            "kpis": [k.as_dict() for k in ec.kpis()]}


def _supply_payload(query: str) -> dict:
    from engines import engine_supply_chain as sc
    m = _CELL_RE.search(query)
    out = {"summary": sc.supply_risk_summary(),
           "kpis": [k.as_dict() for k in sc.kpis()]}
    if m:
        out["traceability"] = sc.material_traceability(f"CELL_{int(m.group(1)):03d}")
    return out


def _maintenance_payload(query: str) -> dict:
    from engines import engine_maintenance as em
    sched = em.schedule_maintenance()
    return {"maintenance": {k: v for k, v in sched.items() if k != "schedule"},
            "charging": em.optimise_charging(),
            "kpis": [k.as_dict() for k in em.kpis()]}


def _quality_payload(query: str) -> dict:
    from engines import engine_quality as eq
    return {"kpis": [k.as_dict() for k in eq.kpis()],
            "incoming_quality": eq.incoming_quality().to_dict("records")}


_DISPATCH = {
    "battery": _battery_payload,
    "readiness": _readiness_payload,
    "carbon": _carbon_payload,
    "supply_chain": _supply_payload,
    "maintenance": _maintenance_payload,
    "quality": _quality_payload,
}


def route(query: str) -> dict:
    """Route a query to one or more engines and gather their structured output."""
    with timed(log, "route", query=query[:80]):
        intents = classify(query)
        if not intents:
            intents = ["readiness", "carbon"]      # sensible default overview
        # Cap at the two strongest matches to keep answers focused.
        intents = intents[:2]
        data = {}
        for intent in intents:
            try:
                data[intent] = _DISPATCH[intent](query)
            except Exception as exc:  # pragma: no cover - defensive
                log.info("engine_error", extra={"kv": {"intent": intent, "error": str(exc)}})
                data[intent] = {"error": str(exc)}
    return {"query": query, "intent": intents[0], "engines_called": intents,
            "data": data}


def answer_payload(query: str) -> dict:
    """The structured result the copilot explains (alias for route)."""
    return route(query)
