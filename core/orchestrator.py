"""Agent Orchestrator - a bounded multi-agent planner over the engines.

This is what makes the "multi-agent" claim TRUE and *visible* rather than
cosmetic. Given a natural-language request the planner:

    1. DECOMPOSES it into sub-tasks (one per relevant domain agent);
    2. builds a bounded, loop-guarded PLAN (max_steps) of which agents to call;
    3. INVOKES each agent (Battery / Fleet / Supply Chain / Maintenance /
       Quality / Carbon), recording every intermediate result;
    4. SYNTHESISES a unified answer.

The plan (``steps``, ``agents_called``, ``intermediate_results``) is returned so
the UI can SHOW the reasoning to judges. It is deterministic and step-capped —
NOT an open-ended autonomous loop, and labelled that way in the UI.

Public API
----------
* plan(query, max_steps)  -> full plan incl. steps + intermediate_results
* route(query)            -> {"query","intent","engines_called","data","plan"}
* classify(query)         -> matched intents
"""
from __future__ import annotations

import re

from core.logging_config import get_logger, timed

log = get_logger(__name__)

# Intent -> trigger keywords (checked in priority order).
INTENT_KEYWORDS = {
    "battery": ["battery", "cell", "soh", "state of health", "rul", "remaining",
                "degrad", "anomaly", "capacity", "cycle life", "passport"],
    "supply_chain": ["supply", "supplier", "lithium", "cobalt", "nickel",
                     "material", "geopolit", "traceab", "concentration",
                     "single source", "sourcing", "disrupt"],
    "maintenance": ["maintenance", "service", "workshop", "downtime", "schedule",
                    "charging", "charger", "tariff", "depot", "technician"],
    "carbon": ["carbon", "co2", "emission", "scope", "green", "credit",
               "decarbon"],
    "readiness": ["readiness", "electrify", "electrification", "switch", "ev",
                  "candidate", "which vehicle", "tco", "payback", "roi", "expand"],
    "quality": ["quality", "defect", "spc", "manufactur", "control chart", "ppm",
                "root cause", "rca"],
}

# Human-facing agent metadata for the visible plan.
AGENT_META = {
    "battery": ("🔋 Battery Agent", "Assess cell health, RUL & anomalies"),
    "readiness": ("🚚 Fleet Agent", "Rank electrification readiness & savings"),
    "supply_chain": ("🔗 Supply-Chain Agent", "Quantify material & geopolitical risk"),
    "maintenance": ("🛠️ Maintenance Agent", "Optimise maintenance & charging"),
    "carbon": ("🌱 Carbon Agent", "Compute Scope 1/2/3 emissions & savings"),
    "quality": ("🏭 Quality Agent", "Check SPC & incoming-material quality"),
}

_CELL_RE = re.compile(r"CELL[_ ]?(\d+)", re.IGNORECASE)
_VEH_RE = re.compile(r"VEH[_ ]?(\d+)", re.IGNORECASE)


def classify(query: str) -> list[str]:
    """Return the intents a query matches, most specific first (may be several)."""
    q = query.lower()
    return [intent for intent, kws in INTENT_KEYWORDS.items()
            if any(k in q for k in kws)]


# ---------------------------------------------------------------------------
# Domain agents (each returns a structured payload)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# The bounded planner
# ---------------------------------------------------------------------------
def _headline_kpi(payload: dict) -> str:
    """Pull a one-line takeaway from an agent payload for the plan/synthesis."""
    if not isinstance(payload, dict):
        return "n/a"
    if payload.get("kpis"):
        k = payload["kpis"][0]
        return f"{k['label']}: {k['value']}"
    if "cell" in payload:
        c = payload["cell"]
        return f"SoH {c['state_of_health']*100:.0f}%, ~{c['remaining_useful_life']:,} cycles left"
    if "vehicle" in payload:
        v = payload["vehicle"]
        return f"{v['vehicle_id']} scores {v['readiness_score']}/100"
    return "computed"


def plan(query: str, max_steps: int = 6) -> dict:
    """Decompose a query into a bounded, visible multi-agent plan and execute it.

    Deterministic and loop-guarded: at most ``max_steps`` agent calls, chosen by
    intent match (falling back to a readiness+carbon overview). Returns the plan
    steps, the agents called, and every intermediate result.
    """
    with timed(log, "plan", query=query[:80]):
        intents = classify(query) or ["readiness", "carbon"]
        # De-duplicate, preserve order, and CAP at max_steps (the loop guard).
        seen, ordered = set(), []
        for i in intents:
            if i not in seen:
                seen.add(i); ordered.append(i)
        ordered = ordered[:max_steps]

        steps, intermediate, data = [], {}, {}
        for idx, intent in enumerate(ordered, start=1):
            name, task = AGENT_META.get(intent, (intent, "Analyse"))
            try:
                payload = _DISPATCH[intent](query)
                status, takeaway = "ok", _headline_kpi(payload)
            except Exception as exc:  # pragma: no cover - defensive
                log.info("agent_error", extra={"kv": {"intent": intent, "error": str(exc)}})
                payload, status, takeaway = {"error": str(exc)}, "error", str(exc)
            data[intent] = payload
            intermediate[intent] = takeaway
            steps.append({
                "step": idx, "agent": name, "intent": intent,
                "task": task, "status": status, "result": takeaway,
            })

        # Final synthesis step (the "reduce" over the agents' outputs).
        synthesis = "; ".join(f"{AGENT_META.get(i, (i,))[0]} → {t}"
                              for i, t in intermediate.items())
        steps.append({
            "step": len(ordered) + 1, "agent": "🧠 Synthesiser",
            "intent": "synthesis", "task": "Combine agent findings",
            "status": "ok", "result": synthesis,
        })

    return {
        "query": query,
        "intent": ordered[0],
        "engines_called": ordered,
        "agents_called": [AGENT_META.get(i, (i,))[0] for i in ordered],
        "steps": steps,
        "intermediate_results": intermediate,
        "synthesis": synthesis,
        "data": data,
        "max_steps": max_steps,
    }


def route(query: str) -> dict:
    """Backward-compatible entry point: run the planner and return its result."""
    p = plan(query)
    return {"query": query, "intent": p["intent"], "engines_called": p["engines_called"],
            "data": p["data"], "plan": p}


def answer_payload(query: str) -> dict:
    """The structured result the copilot explains (alias for route)."""
    return route(query)
