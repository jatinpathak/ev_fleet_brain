"""Structured AI recommendations shared across engines.

Every engine's recommendation carries the four things a decision-maker needs
(the audit's "richer recommendations" requirement):

    * a confidence score (0..1),
    * plain reasoning,
    * quantified business impact (₹ / CO₂ / downtime),
    * 1–2 alternative options.

Engines build a ``Recommendation`` and the dashboard renders it consistently.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Recommendation:
    title: str                      # what we recommend, one line
    action: str                     # the concrete next step
    confidence: float               # 0..1
    reasoning: str                  # why (plain English)
    impact: dict = field(default_factory=dict)      # {"₹ saved": "...", "CO₂": "..."}
    alternatives: list = field(default_factory=list)  # [{"option":..., "note":...}]

    def confidence_pct(self) -> int:
        return int(round(max(0.0, min(1.0, self.confidence)) * 100))

    def confidence_band(self) -> str:
        c = self.confidence
        return "high" if c >= 0.66 else "medium" if c >= 0.4 else "low"

    def as_dict(self) -> dict:
        return {
            "title": self.title, "action": self.action,
            "confidence": round(self.confidence, 2),
            "confidence_pct": self.confidence_pct(),
            "confidence_band": self.confidence_band(),
            "reasoning": self.reasoning,
            "impact": self.impact, "alternatives": self.alternatives,
        }
