"""EV Fleet Intelligence Brain - Streamlit dashboard (v2).

One executive-grade multi-page app over six domain engines and an orchestrating
copilot. Colour semantics: green = good/savings, amber = attention, red =
critical. Every KPI card shows the number AND a one-line "why it matters".
MVP / simulated features carry an honest badge.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import copilot
from core import orchestrator
from engines import (engine_battery as eb, engine_carbon as ec,
                     engine_maintenance as em, engine_quality as eq,
                     engine_readiness as er, engine_supply_chain as sc)
import generate_data

st.set_page_config(page_title="EV Fleet Intelligence Brain", page_icon="🔋", layout="wide")

GREEN, AMBER, RED, GREY = "#1a9850", "#f6a800", "#d73027", "#8a8a8a"
TONE_COLOUR = {"good": GREEN, "warn": AMBER, "bad": RED, "neutral": GREY}


# ---------------------------------------------------------------------------
# Bootstrap (self-heals on a fresh deploy) + secrets bridge
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparing data and models (first run only)…")
def ensure_ready(n_vehicles: int) -> bool:
    need = not (config.BATTERY_DATA_CSV.exists()
                and config.FLEET_DATA_CSV.exists()
                and config.SUPPLIERS_CSV.exists()
                and config.MAINTENANCE_CSV.exists()
                and config.EMISSION_FACTORS_JSON.exists())
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
# Cached loaders (keyed on fleet size so the scale toggle busts them cleanly)
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


@st.cache_resource(show_spinner=False)
def battery_metrics() -> dict:
    return eb.load_model()["metrics"]


# ---------------------------------------------------------------------------
# Reusable UI components
# ---------------------------------------------------------------------------
def mvp_badge(label: str = "MVP") -> str:
    return (f"<span style='background:{AMBER};color:white;padding:2px 8px;"
            f"border-radius:10px;font-size:0.7em;font-weight:600'>{label}</span>")


def render_kpis(kpis, cols_per_row: int = 4):
    """Render a list of core.kpis.KPI as coloured cards with 'why it matters'."""
    for i in range(0, len(kpis), cols_per_row):
        row = kpis[i:i + cols_per_row]
        cols = st.columns(len(row))
        for col, k in zip(cols, row):
            colour = TONE_COLOUR.get(k.tone, GREY)
            col.markdown(
                f"<div style='border-left:5px solid {colour};padding:0.4em 0.8em;"
                f"background:rgba(140,140,140,0.06);border-radius:4px;min-height:118px'>"
                f"<div style='font-size:0.82em;color:{GREY}'>{k.label}</div>"
                f"<div style='font-size:1.7em;font-weight:700;color:{colour}'>{k.display_value()}</div>"
                f"<div style='font-size:0.74em;color:{GREY};line-height:1.2'>{k.why}</div>"
                f"</div>", unsafe_allow_html=True)


def copilot_box(default_query: str, key: str):
    """The orchestrating copilot, available on every page."""
    with st.expander("💬 Ask the Copilot (routes your question across the engines)", expanded=False):
        st.caption("Router + explainer — deterministic intent routing, then one "
                   "LLM call. Not an autonomous agent.")
        q = st.text_input("Your question", value=default_query, key=f"cp_{key}")
        if st.button("Ask", key=f"btn_{key}") and q.strip():
            with st.spinner("Routing across engines…"):
                res = copilot.answer(q)
            st.markdown(res["answer"])
            st.caption(f"Engines consulted: {', '.join(res['routed']['engines_called'])}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home(n: int):
    st.title("🔋 EV Fleet Intelligence Brain")
    st.caption("Executive view — battery health, electrification readiness, "
               "supply-chain risk, maintenance & charging, and carbon, unified.")

    scored = scored_fleet(n)
    fsum = er.fleet_summary(scored)
    csum = ec.fleet_carbon_summary(carbon_fleet(n))
    ssum = sc.supply_risk_summary()
    bmetrics = battery_metrics()
    anomalies = battery_anomalies()

    # Cross-engine headline KPIs.
    from core.kpis import KPI, rupees, tone_for
    render_kpis([
        KPI("Ready to electrify", f"{fsum['ready_now']}", f"/ {fsum['total_vehicles']}",
            "Immediate EV switch candidates (score ≥60).", "good"),
        KPI("5-yr fleet savings", rupees(fsum["total_five_year_savings_inr"]), "",
            "Diesel-vs-EV running-cost saving.", "good"),
        KPI("CO₂ avoided", f"{csum['total_savings_co2_tonnes']:,.0f}", "t/yr",
            f"{csum['savings_pct']}% net of upstream & embodied carbon.", "good"),
        KPI("Supply risk", f"{ssum['overall_risk_score']:.0f}", "/ 100",
            "Concentration + geopolitics + single-sourcing.",
            tone_for(ssum["overall_risk_score"], 40, 60)),
    ])

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("🎯 Biggest opportunities")
        top = scored.head(3)
        for _, v in top.iterrows():
            st.markdown(
                f"**{v['vehicle_id']}** → {v['ev_match']} · readiness "
                f"{v['readiness_score']}/100 · saves "
                f"₹{v['five_year_savings_inr']/1e5:.1f}L over 5 yrs "
                f"(payback {v['payback_years']} yrs)")
        st.markdown(f"**Supply chain** → resolve *{', '.join(ssum['single_sourced_materials']) or 'no'}* "
                    f"single-source exposure (₹{ssum['value_at_risk_inr']/1e7:.1f} Cr at risk)")

    with right:
        st.subheader("🚨 Alert centre")
        n_anom = int(anomalies["anomaly"].sum())
        st.markdown(f"- {'🔴' if n_anom else '🟢'} **{n_anom}** battery cell(s) degrading abnormally fast")
        st.markdown(f"- {'🟠' if ssum['single_sourced_materials'] else '🟢'} "
                    f"**{len(ssum['single_sourced_materials'])}** single-sourced critical material(s)")
        msum = em.schedule_maintenance()
        overdue = int((msum["schedule"]["priority"] == "high").sum())
        st.markdown(f"- {'🟠' if overdue else '🟢'} **{overdue}** high-priority maintenance job(s) due")

    st.info(f"Battery model accuracy on held-out cells: **RMSE "
            f"{bmetrics['rmse_cycles']:.0f} cycles · MAPE {bmetrics['mape_pct']:.1f}%** "
            f"(trained on real-data-shaped Severson-style cycling).")
    st.caption("Open any page from the sidebar. Every page has a Copilot box.")
    copilot_box("Give me a one-paragraph executive summary of the fleet.", "home")


def page_battery(n: int):
    st.title("🔋 Battery Health & Remaining Useful Life")
    render_kpis(eb.kpis(load_battery_data()))
    st.divider()

    df = load_battery_data()
    anomalies = battery_anomalies()
    cells = sorted(df["cell_id"].unique())
    default_idx = cells.index(anomalies.iloc[0]["cell_id"]) if len(anomalies) else 0
    cell_id = st.selectbox("Choose a battery cell (⚠️ = anomaly)", cells, index=default_idx,
                           format_func=lambda c: f"⚠️ {c}" if c in set(
                               anomalies[anomalies.anomaly]["cell_id"]) else c)

    hist = df[df["cell_id"] == cell_id]
    result = eb.predict_health(hist)
    colour = TONE_COLOUR.get({"healthy": "good", "degraded": "warn", "critical": "bad"}
                             .get(result["status"], "neutral"))
    pi = result["predicted_cycle_life_pi"]

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<h3 style='color:{colour}'>State of Health<br>{result['state_of_health']*100:.0f}%</h3>",
                unsafe_allow_html=True)
    c2.metric("Predicted cycle life", f"{result['predicted_cycle_life']:,}",
              help=f"{int(result['ci_coverage']*100)}% interval: {pi[0]:,}–{pi[1]:,} ({result['ci_method']})")
    c3.metric("Remaining useful life", f"{result['remaining_useful_life']:,} cycles",
              help=f"Interval: {result['remaining_useful_life_pi'][0]:,}–{result['remaining_useful_life_pi'][1]:,}")
    st.caption(f"Predicted life **{result['predicted_cycle_life']:,}** cycles "
               f"({int(result['ci_coverage']*100)}% PI **{pi[0]:,}–{pi[1]:,}**, {result['ci_method']}) · "
               f"status **{result['status'].upper()}** · at cycle {result['current_cycle']:,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["discharge_capacity_ah"],
                             mode="lines", name="Measured capacity", line=dict(color=GREEN)))
    fig.add_hline(y=config.END_OF_LIFE_CAPACITY_AH, line_dash="dash", line_color=RED,
                  annotation_text="End of life (0.88 Ah)")
    fig.add_vrect(x0=pi[0], x1=pi[1], fillcolor=AMBER, opacity=0.15, line_width=0,
                  annotation_text="90% life interval")
    fig.add_vline(x=result["predicted_cycle_life"], line_dash="dot", line_color=AMBER,
                  annotation_text="Predicted life")
    fig.update_layout(title=f"Capacity fade — {cell_id}", xaxis_title="Cycle",
                      yaxis_title="Discharge capacity (Ah)", height=420)
    st.plotly_chart(fig, use_container_width=True)

    dc1, dc2 = st.columns(2)
    with dc1:
        st.subheader("Top drivers of this prediction")
        drv = result["top_drivers"]
        st.caption(f"Method: {drv['method']}")
        for feat, val in drv["drivers"]:
            st.markdown(f"- **{feat}**: {val:+.3f}")
    with dc2:
        st.subheader("Fast-degrading cells")
        st.dataframe(anomalies[anomalies.anomaly][["cell_id", "anomaly_score"]].head(8),
                     hide_index=True, use_container_width=True)

    copilot_box(f"Explain the health of {cell_id}.", "battery")


def page_readiness(n: int):
    st.title("🚚 Fleet Electrification Readiness")
    scored = scored_fleet(n)
    render_kpis(er.kpis(scored))
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
        view[["vehicle_id", "vehicle_type", "duty_cycle", "readiness_score", "confidence",
              "ev_match", "payback_years", "five_year_savings_inr", "tco_savings_5yr_inr"]].head(500),
        use_container_width=True, hide_index=True)

    st.divider()
    vid = st.selectbox("Inspect a vehicle",
                       view["vehicle_id"].tolist() or scored["vehicle_id"].tolist())
    rec = er.vehicle_recommendation(vid, load_fleet_data(n))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Readiness", f"{rec['readiness_score']}/100")
    c2.metric("Confidence", f"{int(rec['confidence']*100)}%")
    c3.metric("Best EV", rec["ev_match"])
    c4.metric("Payback", f"{rec['payback_years']} yrs")

    tco = rec["tco_breakdown"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Stay diesel (5-yr TCO)", x=["Energy", "Maintenance", "Insurance"],
                         y=[tco["diesel"]["energy"], tco["diesel"]["maintenance"], tco["diesel"]["insurance"]],
                         marker_color=RED))
    fig.add_trace(go.Bar(name=f"Switch to {rec['ev_match']}",
                         x=["Purchase", "Energy", "Maintenance", "Insurance", "Residual credit"],
                         y=[tco["ev"]["purchase"], tco["ev"]["energy"], tco["ev"]["maintenance"],
                            tco["ev"]["insurance"], tco["ev"]["residual_credit"]], marker_color=GREEN))
    fig.update_layout(title=f"5-year Total Cost of Ownership — {vid} "
                      f"(net saving ₹{rec['tco_savings_5yr_inr']:,.0f})",
                      barmode="group", yaxis_title="₹", height=420)
    st.plotly_chart(fig, use_container_width=True)
    copilot_box(f"Why should we electrify {vid} first?", "readiness")


def page_supply(n: int):
    st.title("🔗 Supply-Chain Risk & Traceability")
    st.markdown(f"Battery-material supply intelligence &nbsp; {mvp_badge('illustrative data')}",
                unsafe_allow_html=True)
    suppliers = sc.load_suppliers()
    render_kpis(sc.kpis(suppliers))
    st.divider()

    summary = sc.supply_risk_summary(suppliers)
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Supplier network")
        graph = sc.build_graph(suppliers)
        st.plotly_chart(_network_figure(graph), use_container_width=True)
    with right:
        st.subheader("Top vulnerabilities")
        for v in summary["top_vulnerabilities"]:
            colour = {"high": RED, "medium": AMBER}.get(v["severity"], GREY)
            st.markdown(f"<div style='border-left:4px solid {colour};padding-left:8px;margin-bottom:8px'>"
                        f"<b>{v['issue']}</b><br><span style='font-size:0.85em'>{v['detail']}</span><br>"
                        f"<span style='color:{colour}'>₹{v['exposure_inr']/1e7:.2f} Cr exposure</span></div>",
                        unsafe_allow_html=True)

    st.divider()
    st.subheader("Material concentration (Herfindahl index)")
    st.dataframe(sc.supplier_concentration_risk(suppliers), hide_index=True, use_container_width=True)

    st.subheader("🔍 Trace a cell to its source")
    cells = sorted(load_battery_data()["cell_id"].unique())
    cell_id = st.selectbox("Cell", cells, key="trace_cell")
    tr = sc.material_traceability(cell_id, suppliers)
    st.markdown(f"**{tr['cell_id']}** → pack **{tr['pack_id']}** → vehicle **{tr['vehicle_id']}** · "
                f"made by **{tr['cell_maker']}** ({tr['cell_maker_country']})")
    for u in tr["upstream"]:
        subs = ", ".join(f"{s['supplier']} ({s['country']})" for s in u["sub_suppliers"])
        flag = "🔴" if u["country_risk"] >= 0.6 else "🟢"
        st.markdown(f"- {flag} **{u['material']}** via {u['supplier']} ({u['country']}, "
                    f"risk {u['country_risk']:.2f}) ← {subs or 'raw'}")
    copilot_box(f"What supply-chain risks affect {cell_id}?", "supply")


def _network_figure(graph: dict) -> go.Figure:
    """Layered layout of the supplier DiGraph coloured by country risk."""
    import networkx as nx
    g = nx.DiGraph()
    for nd in graph["nodes"]:
        g.add_node(nd["id"], **nd)
    for e in graph["edges"]:
        g.add_edge(e["source"], e["target"])
    tier_x = {"Tier-3": 0, "Tier-2": 1, "Tier-1": 2}
    pos, counts = {}, {}
    for nd in graph["nodes"]:
        t = nd["tier"]
        counts[t] = counts.get(t, 0) + 1
        pos[nd["id"]] = (tier_x.get(t, 1), -counts[t])

    edge_x, edge_y = [], []
    for u, v in g.edges:
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(color="rgba(150,150,150,0.5)"), hoverinfo="none"))
    fig.add_trace(go.Scatter(
        x=[pos[n["id"]][0] for n in graph["nodes"]],
        y=[pos[n["id"]][1] for n in graph["nodes"]],
        mode="markers+text",
        text=[n["name"] for n in graph["nodes"]], textposition="middle right",
        marker=dict(size=22, color=[n["risk"] for n in graph["nodes"]],
                    colorscale="RdYlGn_r", cmin=0, cmax=1, showscale=True,
                    colorbar=dict(title="Country<br>risk")),
        hovertext=[f"{n['name']}<br>{n['tier']} · {n['material']}<br>{n['country']} "
                   f"(risk {n['risk']:.2f})" for n in graph["nodes"]], hoverinfo="text"))
    fig.update_layout(showlegend=False, height=430, xaxis=dict(showticklabels=False, title="Tier-3 → Tier-1"),
                      yaxis=dict(showticklabels=False), margin=dict(l=10, r=10, t=10, b=10))
    return fig


def page_maintenance(n: int):
    st.title("🛠️ Maintenance & Charging Optimiser")
    st.markdown(mvp_badge("Tier-2 MVP"), unsafe_allow_html=True)
    render_kpis(em.kpis(fleet_df=load_fleet_data(n)))
    st.divider()

    msum = em.schedule_maintenance()
    st.subheader(f"Optimised maintenance calendar · method: {msum['method']}")
    st.caption(f"{msum['n_jobs']:,} jobs · downtime reduced {msum['downtime_reduction_pct']}% "
               f"vs unoptimised · avg delay {msum['avg_delay_days']} days")
    calendar = (msum["schedule"].groupby(["assigned_day", "priority"]).size()
                .reset_index(name="jobs"))
    fig = go.Figure()
    for prio, colour in [("high", RED), ("medium", AMBER), ("low", GREEN)]:
        sub = calendar[calendar["priority"] == prio]
        fig.add_trace(go.Bar(x=sub["assigned_day"], y=sub["jobs"], name=prio, marker_color=colour))
    fig.update_layout(barmode="stack", title="Jobs scheduled per day",
                      xaxis_title="Day", yaxis_title="Jobs", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Overnight charging optimisation")
    csum = em.optimise_charging(load_fleet_data(n))
    c1, c2, c3 = st.columns(3)
    c1.metric("Energy / night", f"{csum['daily_energy_kwh']:,.0f} kWh")
    c2.metric("Cost saved", f"₹{csum['cost_saved_inr']:,.0f}/day", f"{csum['cost_saved_pct']}%")
    c3.metric("Charger utilisation", f"{csum['charger_utilisation_pct']}%")
    st.caption("Cheapest-hours-first heuristic against a time-of-use tariff; "
               "baseline = charging everything at peak.")
    copilot_box("How much maintenance downtime and charging cost can we save?", "maint")


def page_carbon(n: int):
    st.title("🌱 Carbon Impact (Scope 1 / 2 / 3)")
    carbon = carbon_fleet(n)
    render_kpis(ec.kpis(carbon))
    st.divider()
    csum = ec.fleet_carbon_summary(carbon)

    fig1 = go.Figure(go.Bar(
        x=["Scope 1<br>diesel", "Scope 3<br>diesel upstream", "Scope 2<br>grid (EV)", "Scope 3<br>EV upstream"],
        y=[csum["scope1_tonnes"], csum["scope3_current_tonnes"], csum["scope2_tonnes"], csum["scope3_ev_tonnes"]],
        marker_color=[RED, "#e8836b", AMBER, "#c9b037"]))
    fig1.update_layout(title="Emissions by scope (t/yr)", yaxis_title="Tonnes CO₂/yr", height=380)

    by_class = csum["savings_by_class_tonnes"]
    fig2 = go.Figure(go.Bar(x=list(by_class.keys()), y=list(by_class.values()), marker_color=GREEN))
    fig2.update_layout(title="CO₂ avoided by vehicle class", xaxis_title="Class",
                       yaxis_title="Tonnes CO₂/yr", height=380)
    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(fig1, use_container_width=True)
    cc2.plotly_chart(fig2, use_container_width=True)

    st.metric("Carbon-credit value of avoided CO₂",
              f"₹{csum['carbon_credit_value_inr']:,.0f}/yr")
    st.caption("Scope 1 = diesel tailpipe; Scope 2 = grid electricity for charging; "
               "Scope 3 = diesel well-to-tank + amortised battery-embodied carbon. "
               "Factors illustrative (grid ~0.7 kgCO₂/kWh, CEA India — verify latest before submission).")
    copilot_box("Summarise our carbon story across all scopes.", "carbon")


def page_twin(n: int):
    st.title("🛰️ Digital Twin & Manufacturing Quality")
    st.markdown(f"{mvp_badge('operational digital twin (MVP)')} &nbsp; "
                f"{mvp_badge('light QMS (MVP)')}", unsafe_allow_html=True)
    st.caption("A live fleet-state view and a light quality view — NOT a "
               "physics-level simulation or a full MES/QMS integration.")

    fleet = load_fleet_data(n)
    scored = scored_fleet(n).set_index("vehicle_id")
    sample = fleet.head(400).copy()
    sample["readiness"] = sample["vehicle_id"].map(scored["readiness_score"])
    fig = go.Figure(go.Scattermap(
        lat=sample["lat"], lon=sample["lon"], mode="markers",
        marker=dict(size=9, color=sample["readiness"], colorscale="RdYlGn", cmin=0, cmax=100,
                    showscale=True, colorbar=dict(title="Readiness")),
        text=sample["vehicle_id"] + " · " + sample["depot"]))
    fig.update_layout(map_style="carto-positron",
                      map=dict(center=dict(lat=22.0, lon=79.0), zoom=3.3),
                      height=440, margin=dict(l=0, r=0, t=0, b=0))
    st.subheader("Live fleet grid (colour = readiness)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Manufacturing quality — SPC control chart")
    spc = eq.spc_series()
    figs = go.Figure()
    figs.add_trace(go.Scatter(x=spc["sample"], y=spc["value"], mode="lines+markers", name="Value",
                              line=dict(color=GREEN)))
    figs.add_hline(y=spc["ucl"].iloc[0], line_dash="dash", line_color=RED, annotation_text="UCL")
    figs.add_hline(y=spc["lcl"].iloc[0], line_dash="dash", line_color=RED, annotation_text="LCL")
    figs.add_hline(y=spc["target"].iloc[0], line_dash="dot", line_color=GREY, annotation_text="Target")
    ooc = spc[spc["out_of_control"]]
    figs.add_trace(go.Scatter(x=ooc["sample"], y=ooc["value"], mode="markers", name="Out of control",
                              marker=dict(color=RED, size=11, symbol="x")))
    figs.update_layout(title="Synthetic process parameter (3σ control limits)", height=360,
                       xaxis_title="Sample", yaxis_title="Value")
    st.plotly_chart(figs, use_container_width=True)
    st.dataframe(eq.incoming_quality(), hide_index=True, use_container_width=True)
    copilot_box("Any manufacturing-quality issues we should act on?", "twin")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Executive Home": page_home,
    "🔋 Battery Health": page_battery,
    "🚚 Fleet Readiness": page_readiness,
    "🔗 Supply-Chain Risk": page_supply,
    "🛠️ Maintenance & Charging": page_maintenance,
    "🌱 Carbon Impact": page_carbon,
    "🛰️ Digital Twin / Quality": page_twin,
}


def main():
    _bridge_secret_api_key()
    st.sidebar.title("EV Fleet Brain")
    n = int(st.sidebar.select_slider("Fleet size (scalability demo)",
                                     options=[300, 1000, 10000], value=300))
    ensure_ready(n)
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption("⚠️ Synthetic demo data · battery model is real-data-shaped · "
                       "ET AI Hackathon 2026 · Problem 3")
    if n >= 10000:
        st.sidebar.success(f"Running on {n:,} vehicles — proves the pipeline scales.")

    t0 = time.perf_counter()
    PAGES[choice](n)
    st.sidebar.caption(f"Page rendered in {(time.perf_counter()-t0)*1000:.0f} ms")
    st.markdown(
        "<hr><div style='text-align:center;color:gray;font-size:0.8em'>"
        "Synthetic demo data · anchored to real Indian EVs · battery model trained "
        "on Severson-style cycling · all risk/carbon factors illustrative</div>",
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
