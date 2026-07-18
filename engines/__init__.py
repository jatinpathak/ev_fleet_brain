"""Domain engines. Each is independently testable, logs through core, and
exposes a ``kpis()`` function plus a structured result other layers (the
orchestrator, the copilot, the dashboard) can consume.
"""
