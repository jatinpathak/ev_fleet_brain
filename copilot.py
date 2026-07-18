"""The Copilot - a router + explainer over the engines.

Two entry points:

* ``answer(query)`` — the agentic path: the orchestrator routes the query to the
  right engine(s) and gathers structured data; the copilot turns that into a
  plain-English answer with one LLM call.
* ``explain(kind, data)`` — explains a single structured result (used by the
  per-page "explain this" panels).

Honesty: this is NOT an autonomous agent. It is deterministic intent routing
plus a single explanation call, and is labelled that way in the UI. If no API
key is configured (or the call fails) it falls back to a clear templated answer
so the dashboard NEVER crashes on stage.
"""
from __future__ import annotations

import json

import config
from core import orchestrator
from core.logging_config import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are an advisor for a commercial EV fleet operator's asset-intelligence "
    "platform. You are given STRUCTURED data already computed by domain engines "
    "(battery health, electrification readiness, supply-chain risk, maintenance/"
    "charging, carbon). Explain it in plain, non-technical language for a fleet "
    "manager. Be specific with numbers, ₹ amounts, timelines and business impact. "
    "Do not invent numbers not present in the data. Keep the answer under 200 words."
)


# ---------------------------------------------------------------------------
# Templated fallbacks (used when there is no API key or the call fails)
# ---------------------------------------------------------------------------
def _fallback_battery(d: dict) -> str:
    soh_pct = d.get("state_of_health", 0) * 100
    pi = d.get("predicted_cycle_life_pi") or [None, None]
    pi_txt = f" (90% interval {pi[0]:,}–{pi[1]:,})" if pi[0] is not None else ""
    return (
        f"This battery is at {soh_pct:.0f}% state of health ({d.get('status', 'unknown')}). "
        f"We predict about {d.get('predicted_cycle_life', '?'):,} total cycles{pi_txt}, "
        f"with roughly {d.get('remaining_useful_life', '?'):,} remaining. Plan a "
        f"replacement before it drops below 80% health to avoid a field failure."
    )


def _fallback_vehicle(d: dict) -> str:
    return (
        f"Vehicle {d.get('vehicle_id')} scores {d.get('readiness_score')}/100 for going "
        f"electric (confidence {int(d.get('confidence', 0)*100)}%). Best match: "
        f"{d.get('ev_match')}. Switching saves about "
        f"₹{d.get('annual_savings_inr', 0):,.0f}/year and about "
        f"₹{d.get('tco_savings_5yr_inr', 0):,.0f} over five years on total cost of "
        f"ownership, paying back in ~{d.get('payback_years')} years."
    )


def _fallback_fleet(d: dict) -> str:
    return (
        f"Across {d.get('total_vehicles', '?')} vehicles, {d.get('ready_now', '?')} are "
        f"ready to electrify now (avg readiness {d.get('avg_readiness_score', '?')}/100). "
        f"That saves about ₹{d.get('total_annual_savings_inr', 0):,.0f}/year and cuts "
        f"roughly {d.get('total_savings_co2_tonnes', d.get('savings_co2_tonnes', '?'))} "
        f"tonnes of CO₂ annually. Start with the highest-scoring vehicles."
    )


def _fallback(kind: str, data: dict) -> str:
    if kind == "battery":
        return _fallback_battery(data)
    if kind == "vehicle":
        return _fallback_vehicle(data)
    return _fallback_fleet(data)


def _fallback_answer(routed: dict) -> str:
    """Readable summary of a routed multi-engine payload, no LLM needed."""
    engines = ", ".join(routed.get("engines_called", []))
    lines = [f"Engines consulted: **{engines}**."]
    for intent, payload in routed.get("data", {}).items():
        if not isinstance(payload, dict):
            continue
        kpis = payload.get("kpis")
        if kpis:
            top = "; ".join(f"{k['label']}: {k['value']}" for k in kpis[:3])
            lines.append(f"- **{intent}** — {top}.")
        elif "cell" in payload:
            c = payload["cell"]
            lines.append(f"- **battery** — cell at {c['state_of_health']*100:.0f}% SoH, "
                         f"~{c['remaining_useful_life']:,} cycles remaining.")
        elif "vehicle" in payload:
            v = payload["vehicle"]
            lines.append(f"- **readiness** — {v['vehicle_id']} scores "
                         f"{v['readiness_score']}/100 ({v['ev_match']}).")
    return "\n".join(lines)


def _llm_call(system: str, user_msg: str) -> str | None:
    api_key = config.get_api_key()
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=config.COPILOT_MODEL, max_tokens=config.COPILOT_MAX_TOKENS,
            system=system, messages=[{"role": "user", "content": user_msg}],
        )
        text = "".join(b.text for b in resp.content
                       if getattr(b, "type", "") == "text").strip()
        return text or None
    except Exception as exc:
        log.info("llm_failed_fallback", extra={"kv": {"error": str(exc)}})
        return None


def explain(kind: str, data: dict) -> str:
    """Plain-English explanation of a single structured result."""
    text = _llm_call(
        SYSTEM_PROMPT,
        f"Explain this {kind} data for a fleet manager:\n{json.dumps(data, indent=2, default=str)}",
    )
    return text or _fallback(kind, data)


def answer(query: str) -> dict:
    """Route a natural-language query and return {answer, routed}.

    The agentic path: orchestrator routes -> engines compute -> one LLM call
    explains. Falls back to a deterministic summary if no key/LLM is available.
    """
    routed = orchestrator.route(query)
    text = _llm_call(
        SYSTEM_PROMPT,
        f"A fleet manager asked: {query!r}\n\n"
        f"The orchestrator called these engines and returned this structured "
        f"data:\n{json.dumps(routed['data'], indent=2, default=str)}\n\n"
        f"Answer the question using only these numbers.",
    )
    return {"answer": text or _fallback_answer(routed), "routed": routed}


if __name__ == "__main__":
    print(answer("Which vehicles should we electrify first and what's the carbon impact?")["answer"])
