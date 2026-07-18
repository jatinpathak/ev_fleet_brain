"""EV Fleet Intelligence Brain - Streamlit dashboard (v3, final polish).

An executive-grade, multi-agent decision platform over seven domain engines,
a bounded planner, and a scenario simulator. Colour semantics:
green=good/savings, amber=attention, red=critical, blue=informational,
purple=predictive/AI. Every KPI card shows the number AND a "why it matters";
every MVP / simulated / synthetic component carries an honest badge.

Run:  streamlit run app.py
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import copilot
from core import monitoring
from core.kpis import KPI, rupees, tone_for
from engines import (engine_battery as eb, engine_carbon as ec,
                     engine_maintenance as em, engine_quality as eq,
                     engine_readiness as er, engine_scenario as es,
                     engine_supply_chain as sc)
import generate_data

st.set_page_config(page_title="EV Fleet Intelligence Brain", page_icon="🔋", layout="wide")

GREEN, AMBER, RED, GREY = "#1a9850", "#f6a800", "#d73027", "#8a8a8a"
BLUE, PURPLE = "#2c7fb8", "#7b3fa0"
TONE_COLOUR = {"good": GREEN, "warn": AMBER, "bad": RED, "neutral": GREY,
               "info": BLUE, "ai": PURPLE}

# Approx country centroids for the supplier map (illustrative).
COUNTRY_COORDS = {
    "India": (22.0, 79.0), "China": (35.0, 105.0), "Australia": (-25.0, 133.0),
    "Chile": (-33.0, -71.0), "Indonesia": (-2.0, 118.0), "DR Congo": (-4.0, 22.0),
    "South Korea": (37.0, 128.0), "Japan": (36.0, 138.0), "Argentina": (-38.0, -63.0),
    "Russia": (61.0, 90.0),
}


# ---------------------------------------------------------------------------
# Bootstrap + secrets
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing data and models (first run only)…")
def ensure_ready(n_vehicles: int) -> bool:
    need = not all(p.exists() for p in (
        config.BATTERY_DATA_CSV, config.FLEET_DATA_CSV, config.SUPPLIERS_CSV,
        config.MAINTENANCE_CSV, config.EMISSION_FACTORS_JSON))
    if need or _current_fleet_size() != n_vehicles:
        generate_data.main(n_vehicles)
    if not config.BATTERY_MODEL_PKL.exists():
        eb.train_model(verbose=False)
    return True


def _current_fleet_size() -> int:
    try:
        return sum(1 for _ in open(config.FLEET_DATA_CSV)) - 1
    except OSError:
        return -1


def _bridge_secret_api_key() -> None:
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            import os
            os.environ.setdefault("ANTHROPIC_API_KEY", str(st.secrets["ANTHROPIC_API_KEY"]))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cached loaders (keyed on fleet size)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_battery_data() -> pd.DataFrame:
    return pd.read_csv(config.BATTERY_DATA_CSV)


@st.cache_data(show_spinner=False)
def load_fleet_data(n: int) -> pd.DataFrame:
    return pd.read_csv(config.FLEET_DATA_CSV)


@st.cache_data(show_spinner=False)
def scored_fleet(n: int) -> pd.DataFrame:
    return er.score_fleet(load_fleet_data(n))


@st.cache_data(show_spinner=False)
def carbon_fleet(n: int) -> pd.DataFrame:
    return ec.score_carbon(load_fleet_data(n))


@st.cache_data(show_spinner=False)
def battery_anomalies() -> pd.DataFrame:
    return eb.detect_anomalies(load_battery_data())


@st.cache_data(show_spinner=False)
def battery_snapshot() -> pd.DataFrame:
    return eb.operational_snapshot(load_battery_data())


@st.cache_resource(show_spinner=False)
def battery_metrics() -> dict:
    return eb.load_model()["metrics"]


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------
def badge(label: str, colour: str = AMBER) -> str:
    return (f"<span style='background:{colour};color:white;padding:2px 8px;"
            f"border-radius:10px;font-size:0.7em;font-weight:600'>{label}</span>")


def render_kpis(kpis, cols_per_row: int = 4):
    for i in range(0, len(kpis), cols_per_row):
        row = kpis[i:i + cols_per_row]
        cols = st.columns(len(row))
        for col, k in zip(cols, row):
            colour = TONE_COLOUR.get(k.tone, GREY)
            col.markdown(
                f"<div style='border-left:5px solid {colour};padding:0.4em 0.8em;"
                f"background:rgba(140,140,140,0.06);border-radius:4px;min-height:120px'>"
                f"<div style='font-size:0.82em;color:{GREY}'>{k.label}</div>"
                f"<div style='font-size:1.7em;font-weight:700;color:{colour}'>{k.display_value()}</div>"
                f"<div style='font-size:0.74em;color:{GREY};line-height:1.2'>{k.why}</div>"
                f"</div>", unsafe_allow_html=True)


def render_recommendation(rec):
    """Render a core.recommend.Recommendation: title, confidence, reasoning, impact, alternatives."""
    d = rec.as_dict()
    band_colour = {"high": GREEN, "medium": AMBER, "low": RED}[d["confidence_band"]]
    impact = " · ".join(f"**{k}:** {v}" for k, v in d["impact"].items())
    conf_badge = badge(f"confidence {d['confidence_pct']}%", band_colour)
    st.markdown(
        f"<div style='border:1px solid {PURPLE};border-radius:8px;padding:12px;"
        f"background:rgba(123,63,160,0.05)'>"
        f"<span style='color:{PURPLE};font-weight:700'>🤖 AI recommendation</span> &nbsp;"
        f"{conf_badge}<br>"
        f"<span style='font-size:1.1em;font-weight:600'>{d['title']}</span><br>"
        f"<span style='color:{GREY}'>{d['action']}</span></div>", unsafe_allow_html=True)
    st.caption(f"💡 {d['reasoning']}")
    if impact:
        st.markdown(f"📊 {impact}")
    if d["alternatives"]:
        with st.expander("Alternative options"):
            for a in d["alternatives"]:
                st.markdown(f"- **{a['option']}** — {a['note']}")


def render_plan(plan: dict):
    """Show the planner's reasoning steps (the visible multi-agent trace)."""
    st.markdown(f"<span style='color:{PURPLE};font-weight:600'>🧠 Planner trace</span> "
                f"({len(plan['steps'])} steps, max {plan['max_steps']}, loop-guarded)",
                unsafe_allow_html=True)
    for s in plan["steps"]:
        icon = "✅" if s["status"] == "ok" else "⚠️"
        st.markdown(f"{icon} **Step {s['step']} · {s['agent']}** — {s['task']}  \n"
                    f"&nbsp;&nbsp;↳ _{s['result']}_")


def copilot_box(default_query: str, key: str):
    with st.expander("💬 Ask the Copilot (bounded multi-agent planner)", expanded=False):
        st.caption("Deterministic planner: decomposes your question, calls the relevant "
                   "agents (loop-guarded), then explains. Not an autonomous agent.")
        q = st.text_input("Your question", value=default_query, key=f"cp_{key}")
        if st.button("Ask", key=f"btn_{key}") and q.strip():
            with st.spinner("Planning + routing across agents…"):
                res = copilot.answer(q)
            render_plan(res["routed"]["plan"])
            st.divider()
            st.markdown(res["answer"])


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home(n: int):
    st.title("🏠 Executive Command Center")
    st.caption("One view of value, risk and action across the whole platform.")

    scored = scored_fleet(n)
    fsum = er.fleet_summary(scored)
    csum = ec.fleet_carbon_summary(carbon_fleet(n))
    ssum = sc.supply_risk_summary()
    snap = battery_snapshot()
    anomalies = battery_anomalies()
    msum = em.schedule_maintenance()

    render_kpis([
        KPI("5-yr savings potential", rupees(fsum["total_five_year_savings_inr"]), "",
            "Diesel-vs-EV running-cost saving across the fleet.", "good"),
        KPI("Fleet readiness", f"{100*fsum['ready_now']/max(fsum['total_vehicles'],1):.0f}", "%",
            f"{fsum['ready_now']}/{fsum['total_vehicles']} vehicles ready to electrify now.", "good"),
        KPI("Mean battery health", f"{snap['state_of_health'].mean()*100:.0f}", "%",
            "Average state of health across the operational pack fleet.",
            tone_for(snap["state_of_health"].mean(), 0.85, 0.8, higher_is_worse=False)),
        KPI("CO₂ reduction", f"{csum['savings_pct']:.0f}", "%",
            f"{csum['total_savings_co2_tonnes']:,.0f} t/yr avoided, net of upstream.", "good"),
        KPI("Top supplier risk", f"{ssum['overall_risk_score']:.0f}", "/ 100",
            f"{len(ssum['single_sourced_materials'])} single-sourced material(s).",
            tone_for(ssum["overall_risk_score"], 40, 60)),
    ], cols_per_row=5)

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("⭐ Today's Highlights")
        n_anom = int(anomalies["anomaly"].sum())
        top = scored.iloc[0]
        st.markdown(
            f"1. **Electrify {top['vehicle_id']} first** → {top['ev_match']}, "
            f"payback {top['payback_years']} yrs, saves {rupees(top['five_year_savings_inr'])}/5yr.")
        st.markdown(
            f"2. **De-risk supply** → resolve *{', '.join(ssum['single_sourced_materials']) or 'no'}* "
            f"single-sourcing ({rupees(ssum['value_at_risk_inr'])} exposed).")
        st.markdown(
            f"3. **Inspect batteries** → {n_anom} pack(s) degrading abnormally fast; "
            f"prioritise before failure.")
    with right:
        st.subheader("🚨 Alert Centre")
        n_anom = int(anomalies["anomaly"].sum())
        overdue = int((msum["schedule"]["priority"] == "high").sum())
        st.markdown(f"- {'🔴' if n_anom else '🟢'} **{n_anom}** battery anomaly flag(s)")
        st.markdown(f"- {'🟠' if ssum['single_sourced_materials'] else '🟢'} "
                    f"**{len(ssum['single_sourced_materials'])}** single-sourced material(s)")
        st.markdown(f"- {'🟠' if overdue else '🟢'} **{overdue}** high-priority maintenance job(s)")
        st.markdown(f"- 🔵 Battery model live: **RMSE {battery_metrics()['rmse_cycles']:.0f} cy**")

    st.info("Business value first: use the sidebar to drill into any domain, run a "
            "**Scenario** what-if, or ask the **Copilot**. Battery accuracy is from "
            "real-data-shaped Severson cycling; all other data is clearly-labelled synthetic.")
    copilot_box("Give me an executive summary across savings, risk and carbon.", "home")


def page_battery(n: int):
    st.title("🔋 Battery Health, RUL & Passport")
    render_kpis(eb.kpis(load_battery_data()), cols_per_row=5)
    st.divider()

    df = load_battery_data()
    anomalies = battery_anomalies()
    cells = sorted(df["cell_id"].unique())
    anom_ids = set(anomalies[anomalies.anomaly]["cell_id"])
    default_idx = cells.index(anomalies.iloc[0]["cell_id"]) if len(anomalies) else 0
    cell_id = st.selectbox("Choose a battery cell (⚠️ = anomaly)", cells, index=default_idx,
                           format_func=lambda c: f"⚠️ {c}" if c in anom_ids else c)

    hist = df[df["cell_id"] == cell_id]
    life = int(hist["cycle_life"].iloc[0])
    obs_pct = st.slider("Observe pack at what % of its life?", 10, 95, 50,
                        help="Real fleet packs are mid-life; the lab cell was cycled to death.")
    obs_cycle = max(int(life * obs_pct / 100), 5)
    result = eb.predict_health(hist[hist["cycle"] <= obs_cycle])
    pi = result["predicted_cycle_life_pi"]
    tone = {"healthy": "good", "degraded": "warn", "critical": "bad"}.get(result["status"], "neutral")
    colour = TONE_COLOUR[tone]

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<h3 style='color:{colour}'>SoH<br>{result['state_of_health']*100:.0f}%</h3>",
                unsafe_allow_html=True)
    c2.metric("Predicted cycle life", f"{result['predicted_cycle_life']:,}",
              help=f"{int(result['ci_coverage']*100)}% PI {pi[0]:,}–{pi[1]:,} ({result['ci_method']})")
    c3.metric("Remaining useful life", f"{result['remaining_useful_life']:,} cy",
              help=f"Observed at cycle {obs_cycle:,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["discharge_capacity_ah"], mode="lines",
                             name="Capacity", line=dict(color=GREEN)))
    fig.add_hline(y=config.END_OF_LIFE_CAPACITY_AH, line_dash="dash", line_color=RED,
                  annotation_text="End of life (0.88 Ah)")
    fig.add_vline(x=obs_cycle, line_dash="dot", line_color=BLUE, annotation_text="Observed now")
    fig.add_vrect(x0=pi[0], x1=pi[1], fillcolor=PURPLE, opacity=0.12, line_width=0,
                  annotation_text="90% life interval")
    fig.update_layout(title=f"Capacity fade — {cell_id}", xaxis_title="Cycle",
                      yaxis_title="Discharge capacity (Ah)", height=380)
    st.plotly_chart(fig, use_container_width=True)

    rc1, rc2 = st.columns([3, 2])
    with rc1:
        render_recommendation(eb.recommendation(result))
        st.caption(f"Top drivers ({result['top_drivers']['method']}): "
                   + ", ".join(f"{f} {v:+.2f}" for f, v in result["top_drivers"]["drivers"]))
    with rc2:
        st.subheader("🪪 Battery Passport")
        pp = eb.battery_passport(cell_id, df)
        st.markdown(f"**Chemistry:** {pp['chemistry']} · **Maker:** {pp['manufacturer']} "
                    f"({pp['origin_country']})")
        st.markdown(f"**Lifecycle:** {pp['lifecycle_stage']}")
        st.markdown(f"**Warranty:** {pp['warranty_status']} "
                    f"({pp['warranty_remaining_cycles']:,} cy left of {pp['warranty_cycles']:,})")
        st.markdown(f"**SoH:** {pp['state_of_health_pct']}% · used {pp['cycles_used']:,} cy")
    copilot_box(f"Explain the health and passport of {cell_id}.", "battery")


def page_readiness(n: int):
    st.title("🚚 Fleet Electrification Readiness")
    scored = scored_fleet(n)
    render_kpis(er.kpis(scored))
    st.divider()
    render_recommendation(er.recommendation(scored))
    st.divider()

    colf1, colf2 = st.columns([1, 2])
    ev_filter = colf1.selectbox("Filter by matched EV", ["All"] + sorted(scored["ev_match"].unique()))
    min_score = colf2.slider("Minimum readiness score", 0, 100, 0)
    view = scored.copy()
    if ev_filter != "All":
        view = view[view["ev_match"] == ev_filter]
    view = view[view["readiness_score"] >= min_score]

    st.caption(f"Showing {len(view):,} of {len(scored):,} vehicles, ranked by readiness.")
    st.dataframe(
        view[["vehicle_id", "vehicle_type", "duty_cycle", "depot", "readiness_score",
              "confidence", "ev_match", "payback_years", "five_year_savings_inr"]].head(500),
        use_container_width=True, hide_index=True)

    # Readiness heatmap: depot x duty cycle.
    if "depot" in scored:
        pivot = (scored.pivot_table(index="depot", columns="duty_cycle",
                                    values="readiness_score", aggfunc="mean").round(0))
        fig = go.Figure(go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                                   colorscale="RdYlGn", zmin=0, zmax=100,
                                   colorbar=dict(title="Readiness")))
        fig.update_layout(title="Mean readiness by depot × duty cycle", height=320)
        st.plotly_chart(fig, use_container_width=True)
    copilot_box("Which vehicles should we electrify first and why?", "readiness")


def _network_figure(graph: dict, highlight: set | None = None) -> go.Figure:
    import networkx as nx
    g = nx.DiGraph()
    for nd in graph["nodes"]:
        g.add_node(nd["id"], **nd)
    for e in graph["edges"]:
        g.add_edge(e["source"], e["target"])
    tier_x = {"Tier-3": 0, "Tier-2": 1, "Tier-1": 2}
    pos, counts = {}, {}
    for nd in graph["nodes"]:
        t = nd["tier"]; counts[t] = counts.get(t, 0) + 1
        pos[nd["id"]] = (tier_x.get(t, 1), -counts[t])
    edge_x, edge_y = [], []
    for u, v in g.edges:
        edge_x += [pos[u][0], pos[v][0], None]; edge_y += [pos[u][1], pos[v][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="rgba(150,150,150,0.5)"), hoverinfo="none"))
    sizes = [30 if (highlight and n["id"] in highlight) else 20 for n in graph["nodes"]]
    fig.add_trace(go.Scatter(
        x=[pos[n["id"]][0] for n in graph["nodes"]], y=[pos[n["id"]][1] for n in graph["nodes"]],
        mode="markers+text", text=[n["name"] for n in graph["nodes"]], textposition="middle right",
        marker=dict(size=sizes, color=[n["risk"] for n in graph["nodes"]], colorscale="RdYlGn_r",
                    cmin=0, cmax=1, showscale=True, colorbar=dict(title="Country<br>risk"),
                    line=dict(width=[3 if (highlight and n["id"] in highlight) else 0
                                     for n in graph["nodes"]], color=RED)),
        hovertext=[f"{n['name']}<br>{n['tier']} · {n['material']}<br>{n['country']} "
                   f"(risk {n['risk']:.2f})" for n in graph["nodes"]], hoverinfo="text"))
    fig.update_layout(showlegend=False, height=420, xaxis=dict(showticklabels=False, title="Tier-3 → Tier-1"),
                      yaxis=dict(showticklabels=False), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def page_supply(n: int):
    st.title("🔗 Supply-Chain Risk, Knowledge Graph & Traceability")
    st.markdown(badge("illustrative data", BLUE), unsafe_allow_html=True)
    suppliers = sc.load_suppliers()
    render_kpis(sc.kpis(suppliers))
    st.divider()
    render_recommendation(sc.recommendation())
    st.divider()

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Supplier knowledge graph")
        st.caption("Disrupt a node to propagate risk downstream →")
        disrupt = st.selectbox("Simulate disruption at",
                               ["(none)"] + suppliers["supplier_id"].tolist(),
                               format_func=lambda s: s if s == "(none)"
                               else f"{s} · {suppliers.set_index('supplier_id').loc[s,'supplier_name']}")
        highlight = None
        if disrupt != "(none)":
            prop = sc.propagate_risk(disrupt, suppliers)
            highlight = {a["id"] for a in prop["affected_nodes"]}
            st.warning(f"🔴 Disrupting **{prop['disrupted_name']}** ({prop['material']}) hits "
                       f"**{prop['n_affected']}** nodes; cell makers affected: "
                       f"{', '.join(prop['affected_cell_makers']) or 'none'}. "
                       f"Propagated risk {prop['propagated_risk']:.2f}.")
        st.plotly_chart(_network_figure(sc.build_graph(suppliers), highlight), use_container_width=True)
    with right:
        st.subheader("Supplier locations")
        rows = []
        for _, s in suppliers.iterrows():
            lat, lon = COUNTRY_COORDS.get(s["country"], (0, 0))
            rows.append({"lat": lat, "lon": lon, "name": s["supplier_name"],
                         "risk": sc.country_risk(s["country"]), "material": s["material"]})
        mdf = pd.DataFrame(rows)
        fig = go.Figure(go.Scattergeo(
            lat=mdf["lat"], lon=mdf["lon"], text=mdf["name"] + " · " + mdf["material"],
            marker=dict(size=12, color=mdf["risk"], colorscale="RdYlGn_r", cmin=0, cmax=1,
                        showscale=True, colorbar=dict(title="Risk"))))
        fig.update_layout(height=420, geo=dict(showland=True, landcolor="rgb(240,240,240)"),
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Material concentration (Herfindahl index)")
    st.dataframe(sc.supplier_concentration_risk(suppliers), hide_index=True, use_container_width=True)

    st.subheader("🔍 Trace a cell to its source")
    cell_id = st.selectbox("Cell", sorted(load_battery_data()["cell_id"].unique()), key="trace_cell")
    tr = sc.material_traceability(cell_id, suppliers)
    st.markdown(f"**{tr['cell_id']}** → pack **{tr['pack_id']}** → vehicle **{tr['vehicle_id']}** · "
                f"made by **{tr['cell_maker']}** ({tr['cell_maker_country']})")
    for u in tr["upstream"]:
        subs = ", ".join(f"{s['supplier']} ({s['country']})" for s in u["sub_suppliers"])
        flag = "🔴" if u["country_risk"] >= 0.6 else "🟢"
        st.markdown(f"- {flag} **{u['material']}** via {u['supplier']} ({u['country']}, "
                    f"risk {u['country_risk']:.2f}) ← {subs or 'raw'}")
    copilot_box(f"What supply-chain risks affect {cell_id}?", "supply")


def page_maintenance(n: int):
    st.title("🛠️ Maintenance & Charging Optimiser")
    st.markdown(badge("Tier-2 MVP", AMBER), unsafe_allow_html=True)
    render_kpis(em.kpis(fleet_df=load_fleet_data(n)))
    st.divider()
    render_recommendation(em.recommendation(fleet_df=load_fleet_data(n)))
    st.divider()

    msum = em.schedule_maintenance()
    util = em.resource_utilisation(fleet_df=load_fleet_data(n))
    st.subheader(f"Optimised maintenance calendar · {msum['method']}")
    st.caption(f"{msum['n_active']} active jobs · downtime -{msum['downtime_reduction_pct']}% "
               f"· bay utilisation {util['bay_utilisation_pct']}% · {util['technician_count']} technicians")
    calendar = msum["schedule"].groupby(["assigned_day", "priority"]).size().reset_index(name="jobs")
    fig = go.Figure()
    for prio, colour in [("high", RED), ("medium", AMBER), ("low", GREEN)]:
        sub = calendar[calendar["priority"] == prio]
        fig.add_trace(go.Bar(x=sub["assigned_day"], y=sub["jobs"], name=prio, marker_color=colour))
    fig.update_layout(barmode="stack", title="Jobs per day", xaxis_title="Day",
                      yaxis_title="Jobs", height=340)
    st.plotly_chart(fig, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Charging optimisation")
        c = em.optimise_charging(load_fleet_data(n))
        st.metric("Cost saved", f"₹{c['cost_saved_inr']:,.0f}/day", f"{c['cost_saved_pct']}%")
        st.metric("Charger utilisation", f"{c['charger_utilisation_pct']}%")
    with cc2:
        st.subheader("Charging depots")
        fleet = load_fleet_data(n)
        dep = fleet.groupby("depot").agg(lat=("lat", "mean"), lon=("lon", "mean"),
                                         vehicles=("vehicle_id", "count")).reset_index()
        fig2 = go.Figure(go.Scattermap(lat=dep["lat"], lon=dep["lon"], mode="markers+text",
                                       text=dep["depot"], textposition="top center",
                                       marker=dict(size=dep["vehicles"] / dep["vehicles"].max() * 30 + 8,
                                                   color=BLUE)))
        fig2.update_layout(map_style="carto-positron", map=dict(center=dict(lat=22, lon=79), zoom=3),
                           height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    copilot_box("How much maintenance downtime and charging cost can we save?", "maint")


def page_carbon(n: int):
    st.title("🌱 Carbon Impact (Scope 1 / 2 / 3) & Smart Charging")
    carbon = carbon_fleet(n)
    render_kpis(ec.kpis(carbon))
    st.divider()
    csum = ec.fleet_carbon_summary(carbon)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = go.Figure(go.Bar(
            x=["Scope 1<br>diesel", "Scope 3<br>diesel up.", "Scope 2<br>grid EV", "Scope 3<br>EV up."],
            y=[csum["scope1_tonnes"], csum["scope3_current_tonnes"], csum["scope2_tonnes"], csum["scope3_ev_tonnes"]],
            marker_color=[RED, "#e8836b", AMBER, "#c9b037"]))
        fig1.update_layout(title="Emissions by scope (t/yr)", yaxis_title="t CO₂/yr", height=340)
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        by_class = csum["savings_by_class_tonnes"]
        fig2 = go.Figure(go.Treemap(labels=[f"{k} class" for k in by_class],
                                    parents=[""] * len(by_class),
                                    values=list(by_class.values()),
                                    marker=dict(colors=[GREEN, "#66bd63", "#a6d96a"][:len(by_class)])))
        fig2.update_layout(title="CO₂ avoided by vehicle class (treemap)", height=340,
                           margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("⏱️ Hourly grid intensity & smart charging")
    grid = ec.hourly_grid_intensity()
    smart = ec.smart_charging_carbon(load_fleet_data(n))
    fig3 = go.Figure(go.Scatter(x=grid["hour"], y=grid["kg_co2_per_kwh"], mode="lines+markers",
                                line=dict(color=PURPLE)))
    fig3.add_vrect(x0=0, x1=5, fillcolor=GREEN, opacity=0.12, line_width=0, annotation_text="off-peak (clean)")
    fig3.add_vrect(x0=18, x1=21, fillcolor=RED, opacity=0.12, line_width=0, annotation_text="peak (dirty)")
    fig3.update_layout(title="Grid carbon intensity over 24h (illustrative)",
                       xaxis_title="Hour", yaxis_title="kgCO₂/kWh", height=320)
    st.plotly_chart(fig3, use_container_width=True)
    st.success(f"Charging off-peak instead of at the evening peak cuts grid CO₂ by "
               f"**{smart['co2_saved_pct']}%** (~{smart['annual_co2_saved_tonnes']:,.0f} t/yr).")
    st.metric("Carbon-credit value", f"₹{csum['carbon_credit_value_inr']:,.0f}/yr")
    st.caption("Scope 1 = diesel tailpipe; Scope 2 = grid charging; Scope 3 = diesel well-to-tank + "
               "battery-embodied. Factors illustrative (grid ~0.7 kgCO₂/kWh, CEA India — verify before submission).")
    copilot_box("Summarise our carbon story and smart-charging opportunity.", "carbon")


def page_scenario(n: int):
    st.title("🧪 Scenario Lab — What-If Simulation")
    st.markdown(badge("decision-support simulation", PURPLE), unsafe_allow_html=True)
    st.caption("Pick a scenario and magnitude; every downstream KPI is recomputed from the "
               "engines. Illustrative — a simulation, not a forecast.")

    label_to_key = {v[0]: k for k, v in es.SCENARIOS.items()}
    choice = st.selectbox("Scenario", list(label_to_key))
    key = label_to_key[choice]

    params = {}
    if key == "supplier_disruption":
        params["material"] = st.selectbox("Material", config.CRITICAL_MATERIALS)
        params["severity"] = st.slider("Severity (fraction offline)", 0.1, 1.0, 0.6, 0.1)
    elif key == "accelerated_degradation":
        params["fade_multiplier"] = st.slider("Fade-rate multiplier", 1.0, 3.0, 1.5, 0.1)
    elif key == "tariff_change":
        params["pct"] = st.slider("Tariff change (%)", -30.0, 50.0, 20.0, 5.0)
    elif key == "fleet_expansion":
        params["add_vehicles"] = st.slider("Add vehicles", 50, 5000, 500, 50)

    with st.spinner("Simulating…"):
        res = es.run(key, **params)

    st.subheader("Impact — before → after")
    deltas = res["deltas"]
    cols = st.columns(len(deltas))
    for col, (label, d) in zip(cols, deltas.items()):
        arrow = "▲" if d["direction"] == "up" else "▼" if d["direction"] == "down" else "▬"
        good = d["favourable"]
        colour = GREEN if good else RED if good is False else GREY
        col.markdown(
            f"<div style='border-left:5px solid {colour};padding:0.4em 0.8em;"
            f"background:rgba(140,140,140,0.06);border-radius:4px;min-height:130px'>"
            f"<div style='font-size:0.8em;color:{GREY}'>{label}</div>"
            f"<div style='font-size:1.3em;font-weight:700;color:{colour}'>{arrow} {d['pct']:+.1f}%</div>"
            f"<div style='font-size:0.78em;color:{GREY}'>{d['before']:,} → {d['after']:,}</div>"
            f"</div>", unsafe_allow_html=True)

    fig = go.Figure()
    labels = list(deltas)
    fig.add_trace(go.Bar(name="Before", x=labels, y=[deltas[l]["before"] for l in labels], marker_color=GREY))
    fig.add_trace(go.Bar(name="After", x=labels, y=[deltas[l]["after"] for l in labels], marker_color=PURPLE))
    fig.update_layout(barmode="group", title="Before vs after", height=360)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💬 Copilot: narrate this scenario", expanded=True):
        facts = res["narrative_facts"]
        text = copilot.explain("fleet", {"scenario": choice, **{k: v for k, v in facts.items()},
                                         **{k: d["after"] for k, d in deltas.items()}})
        st.markdown(text)


def page_twin(n: int):
    st.title("🛰️ Digital Twin — Unified Live Fleet State")
    st.markdown(badge("operational digital twin (MVP)", AMBER), unsafe_allow_html=True)
    st.caption("A unified asset view — vehicles, batteries, depots — NOT a physics-level simulation.")

    fleet = load_fleet_data(n)
    scored = scored_fleet(n).set_index("vehicle_id")
    sample = fleet.head(500).copy()
    sample["readiness"] = sample["vehicle_id"].map(scored["readiness_score"])
    fig = go.Figure(go.Scattermap(
        lat=sample["lat"], lon=sample["lon"], mode="markers",
        marker=dict(size=8, color=sample["readiness"], colorscale="RdYlGn", cmin=0, cmax=100,
                    showscale=True, colorbar=dict(title="Readiness")),
        text=sample["vehicle_id"] + " · " + sample["depot"]))
    fig.update_layout(map_style="carto-positron", map=dict(center=dict(lat=22, lon=79), zoom=3.3),
                      height=420, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    snap = battery_snapshot()
    a1.metric("Vehicles", f"{len(fleet):,}")
    a2.metric("Packs healthy", f"{(snap['status']=='healthy').mean()*100:.0f}%")
    a3.metric("Depots", f"{fleet['depot'].nunique()}")

    st.subheader("🏭 Manufacturing Quality — SPC + Root-Cause Hints")
    st.markdown(badge("light QMS (MVP)", AMBER), unsafe_allow_html=True)
    spc = eq.spc_series()
    figs = go.Figure()
    figs.add_trace(go.Scatter(x=spc["sample"], y=spc["value"], mode="lines+markers",
                              name="Value", line=dict(color=GREEN)))
    figs.add_hline(y=spc["ucl"].iloc[0], line_dash="dash", line_color=RED, annotation_text="UCL")
    figs.add_hline(y=spc["lcl"].iloc[0], line_dash="dash", line_color=RED, annotation_text="LCL")
    ooc = spc[spc["out_of_control"]]
    figs.add_trace(go.Scatter(x=ooc["sample"], y=ooc["value"], mode="markers", name="Out of control",
                              marker=dict(color=RED, size=11, symbol="x")))
    figs.update_layout(title="SPC control chart (3σ)", height=320, xaxis_title="Sample", yaxis_title="Value")
    st.plotly_chart(figs, use_container_width=True)
    for h in eq.root_cause_hints():
        colour = {"high": RED, "medium": AMBER}.get(h["severity"], GREY)
        st.markdown(f"<div style='border-left:4px solid {colour};padding-left:8px;margin-bottom:6px'>"
                    f"<b>{h['signal']}</b><br><span style='font-size:0.85em'>Likely cause: "
                    f"{h['likely_cause']}</span><br><span style='font-size:0.85em;color:{colour}'>"
                    f"→ {h['action']}</span></div>", unsafe_allow_html=True)
    copilot_box("Any manufacturing-quality issues we should act on?", "twin")


def page_system(n: int):
    st.title("⚙️ System, Monitoring & Integration")
    st.caption("The engineering story: model-drift monitoring, a feature store, and a REST API.")

    st.subheader("📈 Model monitoring (drift) — stub")
    st.markdown(badge("monitoring stub, not a retraining pipeline", BLUE), unsafe_allow_html=True)
    metric = st.selectbox("Monitored prediction", ["readiness_score", "remaining_useful_life"])
    shift = st.slider("Simulate world-shift in the live stream", 0.0, 0.4, 0.0, 0.05,
                      help="Injects drift so you can watch PSI react.")
    rep = monitoring.drift_report(metric, drift_shift=shift)
    m1, m2, m3 = st.columns(3)
    m1.metric("PSI", f"{rep['psi']:.3f}", rep["status"])
    m2.metric("Reference mean", f"{rep['reference_mean']}")
    m3.metric("Current mean", f"{rep['current_mean']}")
    dist = rep["distribution"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=dist["reference"], name="reference", opacity=0.6, marker_color=BLUE))
    fig.add_trace(go.Histogram(x=dist["current"], name="current", opacity=0.6, marker_color=PURPLE))
    fig.update_layout(barmode="overlay", title=f"Prediction distribution — {metric}", height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("PSI < 0.1 stable · 0.1–0.25 moderate · > 0.25 significant. A real deployment "
               "would trigger investigation/retraining above threshold.")

    st.subheader("🗄️ Feature store (SQLite)")
    try:
        from core import feature_store as fs
        counts = {t: len(fs.read(t)) for t in fs.tables()}
        st.write({t: f"{c:,} rows" for t, c in counts.items()})
    except Exception as exc:
        st.info(f"Feature store builds on first use. ({exc})")

    st.subheader("🔌 REST API (FastAPI)")
    st.code("uvicorn api:app --port 8000    # docs at /docs\n"
            "GET  /fleet/summary\n"
            "POST /battery/predict  {\"cell_id\": \"CELL_010\"}\n"
            "POST /scenario/run     {\"name\": \"tariff_change\", \"params\": {\"pct\": 15}}",
            language="bash")
    st.subheader("🐳 Containerisation")
    st.code("docker compose up   # dashboard :8501 + API :8000", language="bash")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Command Center": page_home,
    "🔋 Battery Health": page_battery,
    "🚚 Fleet Readiness": page_readiness,
    "🔗 Supply-Chain Risk": page_supply,
    "🛠️ Maintenance & Charging": page_maintenance,
    "🌱 Carbon Impact": page_carbon,
    "🧪 Scenario Lab": page_scenario,
    "🛰️ Digital Twin": page_twin,
    "⚙️ System & Monitoring": page_system,
}


def main():
    _bridge_secret_api_key()
    st.sidebar.title("🔋 EV Fleet Brain")
    st.sidebar.caption("v3 · multi-agent decision platform")
    n = int(st.sidebar.select_slider("Fleet size (scalability demo)",
                                     options=[300, 1000, 10000], value=300))
    ensure_ready(n)
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption("⚠️ Synthetic demo data · battery model real-data-shaped · "
                       "ET AI Hackathon 2026 · Problem 3")
    if n >= 10000:
        st.sidebar.success(f"Running on {n:,} vehicles — proves the pipeline scales.")

    t0 = time.perf_counter()
    PAGES[choice](n)
    st.sidebar.caption(f"Page rendered in {(time.perf_counter()-t0)*1000:.0f} ms")
    st.markdown(
        "<hr><div style='text-align:center;color:gray;font-size:0.8em'>"
        "Synthetic demo data · real Indian EVs · Severson-shaped battery model · "
        "all risk/carbon/tariff factors illustrative</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
