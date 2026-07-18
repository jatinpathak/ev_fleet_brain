"""Cross-cutting services shared by every engine.

This package is the ONE real quality signal implemented once and reused
everywhere: structured logging, measurable KPIs, explainability and
uncertainty. Engines depend on core; core never depends on engines.
"""
