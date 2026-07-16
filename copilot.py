"""The Copilot - a thin Anthropic wrapper that explains structured data.

This is deliberately NOT an agent: no chains, no tool loops. It takes a
structured dict (battery health, a vehicle recommendation, or a fleet summary)
and returns a plain-English explanation of under ~200 words for a non-technical
fleet manager.

If no API key is configured (or the API call fails), it falls back to a clear
templated explanation so the dashboard NEVER crashes on stage.
"""
from __future__ import annotations

import json

import config

SYSTEM_PROMPT = (
    "You are an advisor for commercial EV fleet electrification. Given "
    "structured data, explain it in plain, non-technical language for a fleet "
    "manager. Be specific with numbers, timelines, and business impact. Keep "
    "your answer under 200 words."
)


# ---------------------------------------------------------------------------
# Templated fallbacks (used when there is no API key or the call fails)
# ---------------------------------------------------------------------------
def _fallback_battery(d: dict) -> str:
    soh_pct = d.get("state_of_health", 0) * 100
    return (
        f"This battery is currently at {soh_pct:.0f}% state of health "
        f"({d.get('status', 'unknown')}). We predict a total life of about "
        f"{d.get('predicted_cycle_life', '?')} charge cycles, of which roughly "
        f"{d.get('remaining_useful_life', '?')} cycles remain. In plain terms, "
        f"the cell still has useful life left, but plan a replacement before it "
        f"drops below 80% health to avoid a failure in the field."
    )


def _fallback_vehicle(d: dict) -> str:
    return (
        f"Vehicle {d.get('vehicle_id')} scores "
        f"{d.get('readiness_score')}/100 for going electric. The best match is "
        f"the {d.get('ev_match')}. Switching would save about "
        f"Rs {d.get('annual_savings_inr', 0):,.0f} per year "
        f"(Rs {d.get('five_year_savings_inr', 0):,.0f} over five years), paying "
        f"back the vehicle in roughly {d.get('payback_years')} years. This makes "
        f"it a strong early candidate for electrification."
    )


def _fallback_fleet(d: dict) -> str:
    return (
        f"Across {d.get('total_vehicles', '?')} vehicles, {d.get('ready_now', '?')} "
        f"are ready to electrify now (average readiness "
        f"{d.get('avg_readiness_score', '?')}/100). Doing so would save about "
        f"Rs {d.get('total_annual_savings_inr', 0):,.0f} a year and cut roughly "
        f"{d.get('total_savings_co2_tonnes', d.get('savings_co2_tonnes', '?'))} "
        f"tonnes of CO2 annually. Start with the highest-scoring vehicles for the "
        f"fastest payback and biggest impact."
    )


def _fallback(kind: str, data: dict) -> str:
    if kind == "battery":
        return _fallback_battery(data)
    if kind == "vehicle":
        return _fallback_vehicle(data)
    return _fallback_fleet(data)


def explain(kind: str, data: dict) -> str:
    """Return a plain-English explanation of `data`.

    Parameters
    ----------
    kind : "battery" | "vehicle" | "fleet"
    data : the structured dict from the relevant engine.
    """
    api_key = config.get_api_key()
    if not api_key:
        return _fallback(kind, data)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        user_msg = (
            f"Explain this {kind} data for a fleet manager:\n"
            f"{json.dumps(data, indent=2)}"
        )
        resp = client.messages.create(
            model=config.COPILOT_MODEL,
            max_tokens=config.COPILOT_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        # Concatenate any text blocks in the response.
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        ).strip()
        return text or _fallback(kind, data)
    except Exception:
        # Any failure (network, auth, quota) -> graceful templated fallback.
        return _fallback(kind, data)


if __name__ == "__main__":
    demo = {
        "vehicle_id": "VEH_118", "readiness_score": 96.3, "ev_match": "Tata Nexon EV",
        "annual_savings_inr": 503734, "five_year_savings_inr": 2518670, "payback_years": 2.98,
    }
    print(explain("vehicle", demo))
