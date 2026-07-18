"""Engine 4 (NEW, Tier 1) - Supply-Chain Risk & Traceability.

The problem statement explicitly demands battery supply-chain / asset
intelligence and v1 lacked it entirely. This engine quantifies and ranks the
exposure a fleet operator carries in its battery-material supply chain.

Logic
-----
* supplier_concentration_risk() — Herfindahl (HHI) index per critical material;
  flags single-source dependencies.
* geopolitical_exposure_score() — volume-weighted country-risk exposure using a
  static, ILLUSTRATIVE country-risk table.
* material_traceability(cell_id) — trace a finished cell back through pack and
  supplier tiers (cell → pack → vehicle lineage + upstream suppliers).
* supply_risk_summary() — overall risk score + top-3 vulnerabilities with a
  business-impact (₹ exposure) figure.
* build_graph() — a networkx supplier network exposed as nodes/edges for the UI.

Every number here is illustrative and labelled as such in the UI.
"""
from __future__ import annotations

import pandas as pd

import config
from core.kpis import KPI, rupees, tone_for
from core.logging_config import get_logger, timed

log = get_logger(__name__)


def load_suppliers() -> pd.DataFrame:
    if not config.SUPPLIERS_CSV.exists():
        raise FileNotFoundError(
            f"{config.SUPPLIERS_CSV} not found. Run generate_data.py first."
        )
    df = pd.read_csv(config.SUPPLIERS_CSV)
    df["supplies_to"] = df["supplies_to"].fillna("").astype(str)
    return df


def country_risk(country: str) -> float:
    return config.COUNTRY_RISK.get(country, config.DEFAULT_COUNTRY_RISK)


def supplier_concentration_risk(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """HHI concentration per critical material (0..1; 1 = single source)."""
    df = df if df is not None else load_suppliers()
    rows = []
    for material in config.CRITICAL_MATERIALS:
        sub = df[df["material"] == material]
        if sub.empty:
            continue
        vol = sub["annual_volume"].to_numpy(dtype=float)
        shares = vol / vol.sum() if vol.sum() > 0 else vol
        hhi = float((shares ** 2).sum())
        band = ("high" if hhi >= config.HHI_HIGH
                else "moderate" if hhi >= config.HHI_MODERATE else "low")
        rows.append({
            "material": material,
            "n_suppliers": int(len(sub)),
            "hhi": round(hhi, 3),
            "concentration": band,
            "single_source": bool(len(sub) == 1),
            "countries": ", ".join(sorted(sub["country"].unique())),
        })
    return pd.DataFrame(rows).sort_values("hhi", ascending=False).reset_index(drop=True)


def geopolitical_exposure_score(df: pd.DataFrame | None = None) -> dict:
    """Volume-weighted country-risk exposure across the supply base."""
    df = df if df is not None else load_suppliers()
    df = df.copy()
    df["risk"] = df["country"].map(country_risk)
    w = df["annual_volume"].to_numpy(dtype=float)
    weighted = float((df["risk"].to_numpy() * w).sum() / w.sum()) if w.sum() > 0 else 0.0

    by_country = (
        df.assign(risk=df["risk"])
          .groupby("country")
          .agg(volume=("annual_volume", "sum"), risk=("risk", "first"))
          .reset_index()
          .sort_values("risk", ascending=False)
    )
    high_risk = by_country[by_country["risk"] >= 0.6]["country"].tolist()
    return {
        "weighted_exposure": round(weighted, 3),
        "high_risk_countries": high_risk,
        "by_country": by_country,
    }


def material_traceability(cell_id: str, suppliers: pd.DataFrame | None = None,
                          fleet_df: pd.DataFrame | None = None) -> dict:
    """Trace a finished cell back through its pack, vehicle and supplier tiers."""
    df = suppliers if suppliers is not None else load_suppliers()

    # Deterministic cell -> Tier-1 maker -> pack -> vehicle lineage.
    idx = abs(hash(str(cell_id))) % (10 ** 8)
    tier1 = df[df["tier"] == "Tier-1"].reset_index(drop=True)
    maker = tier1.iloc[idx % len(tier1)] if len(tier1) else None
    pack_id = f"PACK_{idx % 1000:03d}"

    vehicle_id = None
    if fleet_df is None and config.FLEET_DATA_CSV.exists():
        fleet_df = pd.read_csv(config.FLEET_DATA_CSV, usecols=["vehicle_id"])
    if fleet_df is not None and len(fleet_df):
        vehicle_id = str(fleet_df.iloc[idx % len(fleet_df)]["vehicle_id"])

    # Walk upstream: Tier-2 that supply this maker, then Tier-3 that supply them.
    upstream = []
    if maker is not None:
        tier2 = df[df["supplies_to"] == maker["supplier_id"]]
        for _, s2 in tier2.iterrows():
            tier3 = df[df["supplies_to"] == s2["supplier_id"]]
            upstream.append({
                "tier": s2["tier"], "supplier": s2["supplier_name"],
                "material": s2["material"], "country": s2["country"],
                "country_risk": country_risk(s2["country"]),
                "sub_suppliers": [
                    {"tier": s3["tier"], "supplier": s3["supplier_name"],
                     "material": s3["material"], "country": s3["country"],
                     "country_risk": country_risk(s3["country"])}
                    for _, s3 in tier3.iterrows()
                ],
            })

    flat_countries = []
    for u in upstream:
        flat_countries.append((u["country"], u["country_risk"]))
        flat_countries += [(s["country"], s["country_risk"]) for s in u["sub_suppliers"]]
    riskiest = max(flat_countries, key=lambda c: c[1], default=(None, 0.0))

    return {
        "cell_id": cell_id,
        "pack_id": pack_id,
        "vehicle_id": vehicle_id,
        "cell_maker": None if maker is None else maker["supplier_name"],
        "cell_maker_country": None if maker is None else maker["country"],
        "upstream": upstream,
        "riskiest_country": riskiest[0],
        "riskiest_country_risk": round(float(riskiest[1]), 2),
    }


def build_graph(df: pd.DataFrame | None = None) -> dict:
    """Return {nodes, edges} for the supplier network (networkx-backed)."""
    import networkx as nx

    df = df if df is not None else load_suppliers()
    g = nx.DiGraph()
    for _, s in df.iterrows():
        g.add_node(s["supplier_id"], name=s["supplier_name"], tier=s["tier"],
                   material=s["material"], country=s["country"],
                   risk=country_risk(s["country"]), volume=int(s["annual_volume"]))
    for _, s in df.iterrows():
        if s["supplies_to"]:
            g.add_edge(s["supplier_id"], s["supplies_to"])

    nodes = [{"id": n, **g.nodes[n]} for n in g.nodes]
    edges = [{"source": u, "target": v} for u, v in g.edges]
    return {"nodes": nodes, "edges": edges,
            "n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges()}


def supply_risk_summary(df: pd.DataFrame | None = None) -> dict:
    """Overall risk score + top-3 vulnerabilities with ₹ exposure."""
    df = df if df is not None else load_suppliers()
    with timed(log, "supply_risk_summary", n_suppliers=len(df)):
        conc = supplier_concentration_risk(df)
        geo = geopolitical_exposure_score(df)

        single_sourced = conc[conc["single_source"]]["material"].tolist()
        pct_single = 100.0 * len(single_sourced) / max(len(conc), 1)
        mean_hhi = float(conc["hhi"].mean()) if len(conc) else 0.0
        weighted_geo = geo["weighted_exposure"]

        # Blend into a 0..100 overall risk score.
        overall = 100.0 * (0.45 * mean_hhi + 0.40 * weighted_geo + 0.15 * (pct_single / 100))
        val = config.PRODUCTION_VALUE_AT_RISK_INR

        vulns = []
        for m in single_sourced:
            row = conc[conc["material"] == m].iloc[0]
            vulns.append({
                "issue": f"Single-source dependency: {m}",
                "detail": f"Only one qualified supplier ({row['countries']}).",
                "exposure_inr": round(val * 0.4),
                "severity": "high",
            })
        # Highest-concentration non-single material.
        multi = conc[~conc["single_source"]]
        if len(multi):
            top = multi.iloc[0]
            vulns.append({
                "issue": f"High concentration: {top['material']}",
                "detail": f"HHI {top['hhi']} across {top['n_suppliers']} suppliers.",
                "exposure_inr": round(val * 0.25 * top["hhi"]),
                "severity": "medium" if top["hhi"] < config.HHI_HIGH else "high",
            })
        for c in geo["high_risk_countries"][:1]:
            vulns.append({
                "issue": f"Geopolitical exposure: {c}",
                "detail": f"Country-risk {country_risk(c):.2f} on critical materials.",
                "exposure_inr": round(val * 0.2 * country_risk(c)),
                "severity": "high" if country_risk(c) >= 0.8 else "medium",
            })

        vulns = sorted(vulns, key=lambda v: v["exposure_inr"], reverse=True)[:3]

    return {
        "overall_risk_score": round(overall, 1),
        "mean_hhi": round(mean_hhi, 3),
        "pct_single_sourced": round(pct_single, 1),
        "weighted_geopolitical_exposure": round(weighted_geo, 3),
        "single_sourced_materials": single_sourced,
        "value_at_risk_inr": round(sum(v["exposure_inr"] for v in vulns)),
        "top_vulnerabilities": vulns,
    }


def kpis(df: pd.DataFrame | None = None) -> list[KPI]:
    df = df if df is not None else load_suppliers()
    s = supply_risk_summary(df)
    return [
        KPI("Overall supply risk", f"{s['overall_risk_score']:.0f}", "/ 100",
            "Blend of concentration, geopolitics and single-sourcing.",
            tone_for(s["overall_risk_score"], 40, 60)),
        KPI("Single-sourced materials", f"{s['pct_single_sourced']:.0f}", "%",
            "Critical materials with no qualified backup supplier.",
            tone_for(s["pct_single_sourced"], 20, 40)),
        KPI("Geopolitical exposure", f"{s['weighted_geopolitical_exposure']*100:.0f}", "/ 100",
            "Volume-weighted country risk across the supply base.",
            tone_for(s["weighted_geopolitical_exposure"], 0.4, 0.6)),
        KPI("Value at risk", rupees(s["value_at_risk_inr"]), "",
            "Illustrative ₹ production exposure from the top-3 vulnerabilities.", "warn"),
    ]


if __name__ == "__main__":
    print(supplier_concentration_risk().to_string(index=False))
    print("\nSummary:", supply_risk_summary())
