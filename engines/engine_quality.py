"""Engine 6 (NEW, Tier 3 MVP) - Manufacturing Quality (light QMS view).

A deliberately light quality-management view of the manufacturing half of the
problem, clearly labelled an MVP:

* incoming-material quality trend from supplier defect rates;
* a simple SPC (statistical process control) chart on a synthetic process
  parameter, with Western-Electric-style out-of-control flags;
* a cell → pack → vehicle traceability link (reuses the supply-chain engine).

This is NOT a full QMS or MES integration — it is an honest, scoped demonstrator
of manufacturing-quality intelligence.

Public API
----------
* spc_series()        -> DataFrame of the control chart with limits + flags
* incoming_quality()  -> per-material incoming defect-rate summary
* kpis()              -> list[KPI]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from core.kpis import KPI, tone_for
from core.logging_config import get_logger

log = get_logger(__name__)


def spc_series(n_samples: int = 60) -> pd.DataFrame:
    """Synthetic in-line process parameter with SPC control limits and flags.

    The parameter drifts slightly late in the run so the chart has something to
    detect — a realistic demo of process monitoring, clearly synthetic.
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 5)
    target, sigma, k = config.SPC_TARGET, config.SPC_SIGMA, config.SPC_CONTROL_K

    base = rng.normal(target, sigma, n_samples)
    # Inject an upward drift in the last third (special-cause variation) that
    # grows past the 3-sigma control limit so the chart has a real excursion to
    # detect — a realistic "process going out of control" demo.
    drift = np.concatenate([
        np.zeros(2 * n_samples // 3),
        np.linspace(0, (k + 1.0) * sigma, n_samples - 2 * n_samples // 3),
    ])
    values = base + drift

    ucl, lcl = target + k * sigma, target - k * sigma
    df = pd.DataFrame({
        "sample": np.arange(1, n_samples + 1),
        "value": np.round(values, 3),
        "target": target,
        "ucl": round(ucl, 3),
        "lcl": round(lcl, 3),
    })
    # Rule 1: any point beyond 3-sigma control limits.
    df["out_of_control"] = (df["value"] > ucl) | (df["value"] < lcl)
    return df


def incoming_quality() -> pd.DataFrame:
    """Per-material incoming quality from supplier defect / on-time rates."""
    if not config.SUPPLIERS_CSV.exists():
        raise FileNotFoundError("suppliers CSV missing. Run generate_data.py first.")
    s = pd.read_csv(config.SUPPLIERS_CSV)
    g = (s.groupby("material")
           .agg(avg_defect_rate=("quality_defect_rate", "mean"),
                avg_on_time=("on_time_delivery_rate", "mean"),
                suppliers=("supplier_id", "count"))
           .reset_index())
    g["avg_defect_rate"] = g["avg_defect_rate"].round(4)
    g["avg_on_time"] = g["avg_on_time"].round(3)
    g["ppm"] = (g["avg_defect_rate"] * 1e6).round(0)   # defects per million
    return g.sort_values("avg_defect_rate", ascending=False).reset_index(drop=True)


def root_cause_hints() -> list[dict]:
    """Correlate quality signals to likely causes (suppliers / process drift).

    A light root-cause-analysis layer: it ranks incoming materials by defect
    rate, points at the worst supplier for each, and flags SPC drift. Heuristic
    correlations, clearly a decision-support aid — not a validated RCA engine.
    """
    hints = []
    iq = incoming_quality()
    if not config.SUPPLIERS_CSV.exists():
        return hints
    suppliers = pd.read_csv(config.SUPPLIERS_CSV)

    # 1) Worst incoming material -> its worst supplier.
    if len(iq):
        worst = iq.iloc[0]
        subs = suppliers[suppliers["material"] == worst["material"]]
        if len(subs):
            culprit = subs.sort_values("quality_defect_rate", ascending=False).iloc[0]
            hints.append({
                "signal": f"High incoming defect rate on {worst['material']} "
                          f"({worst['ppm']:,.0f} ppm)",
                "likely_cause": f"Supplier {culprit['supplier_name']} "
                                f"({culprit['country']}), defect rate "
                                f"{culprit['quality_defect_rate']*100:.2f}%",
                "action": "Audit this supplier's incoming lots; tighten acceptance sampling.",
                "severity": "high" if worst["ppm"] > 30000 else "medium",
            })

    # 2) SPC drift -> process special cause.
    spc = spc_series()
    n_ooc = int(spc["out_of_control"].sum())
    if n_ooc:
        hints.append({
            "signal": f"{n_ooc} SPC points beyond 3σ control limits (late-run drift)",
            "likely_cause": "Process special cause — tool wear or setpoint drift in the "
                            "final third of the run.",
            "action": "Trigger a line stop + re-centre the process; investigate the drift.",
            "severity": "high",
        })

    # 3) Low on-time delivery -> schedule risk feeding quality shortcuts.
    late = suppliers.sort_values("on_time_delivery_rate").head(1)
    if len(late) and late.iloc[0]["on_time_delivery_rate"] < 0.85:
        s = late.iloc[0]
        hints.append({
            "signal": f"Low on-time delivery from {s['supplier_name']} "
                      f"({s['on_time_delivery_rate']*100:.0f}%)",
            "likely_cause": "Expediting/late lots correlate with rushed incoming inspection.",
            "action": "Add buffer stock; de-risk the line from this supplier's slips.",
            "severity": "medium",
        })
    return hints


def kpis() -> list[KPI]:
    spc = spc_series()
    iq = incoming_quality()
    n_ooc = int(spc["out_of_control"].sum())
    worst_ppm = float(iq["ppm"].max()) if len(iq) else 0.0
    avg_defect = float(iq["avg_defect_rate"].mean()) if len(iq) else 0.0
    return [
        KPI("SPC out-of-control points", f"{n_ooc}", f"/ {len(spc)}",
            "In-line samples breaching 3σ control limits — investigate the drift.",
            "warn" if n_ooc else "good"),
        KPI("Worst incoming defect rate", f"{worst_ppm:,.0f}", "ppm",
            "Highest defect rate among incoming critical materials.",
            tone_for(worst_ppm, 20000, 40000)),
        KPI("Avg incoming defect rate", f"{avg_defect*100:.2f}", "%",
            "Mean defect rate across incoming battery materials.",
            tone_for(avg_defect, 0.02, 0.04)),
    ]


if __name__ == "__main__":
    print(spc_series().tail().to_string(index=False))
    print(incoming_quality().to_string(index=False))
