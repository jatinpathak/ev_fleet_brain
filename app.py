"""EV Fleet Intelligence Brain - Streamlit dashboard.

One clean multi-page app tying the three engines and the copilot together.
Colour code: green = healthy / savings, amber = attention, red = critical.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import copilot
import engine_battery as eb
import engine_carbon as ec
import engine_readiness as er

st.set_page_config(page_title="EV Fleet Intelligence Brain", page_icon="🔋", layout="wide")

GREEN, AMBER, RED = "#1a9850", "#f6a800", "#d73027"


# ---------------------------------------------------------------------------
# Cached data loaders (so pages are snappy)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_battery_data() -> pd.DataFrame:
    return pd.read_csv(config.BATTERY_DATA_CSV)


@st.cache_data(show_spinner=False)
def load_fleet_data() -> pd.DataFrame:
    return pd.read_csv(config.FLEET_DATA_CSV)


@st.cache_data(show_spinner=False)
def scored_fleet() -> pd.DataFrame:
    return er.score_fleet(load_fleet_data())


@st.cache_data(show_spinner=False)
def carbon_fleet() -> pd.DataFrame:
    return ec.score_carbon(load_fleet_data())


@st.cache_resource(show_spinner=False)
def battery_metrics() -> dict:
    return eb.load_model()["metrics"]


def status_colour(status: str) -> str:
    return {"healthy": GREEN, "degraded": AMBER, "critical": RED}.get(status, AMBER)


def metric_card(col, label: str, value: str, help_text: str = ""):
    col.metric(label, value, help=help_text)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_home():
    st.title("🔋 EV Fleet Intelligence Brain")
    st.caption(
        "One assistant, three engines: battery health, electrification "
        "readiness, and carbon savings — explained in plain English."
    )

    fsum = er.fleet_summary(scored_fleet())
    csum = ec.fleet_carbon_summary(carbon_fleet())
    bmetrics = battery_metrics()

    c1, c2, c3 = st.columns(3)
    metric_card(c1, "Vehicles ready to electrify",
                f"{fsum['ready_now']} / {fsum['total_vehicles']}",
                "Readiness score >= 60")
    metric_card(c2, "5-year fleet savings",
                f"₹{fsum['total_five_year_savings_inr']/1e7:.2f} Cr",
                "Total diesel-vs-EV running-cost saving")
    metric_card(c3, "Annual CO₂ avoided",
                f"{csum['total_savings_co2_tonnes']:,.0f} t",
                f"{csum['savings_pct']}% lower than diesel")

    st.divider()
    st.subheader("What is this?")
    st.write(
        "A delivery company with 300 diesel/petrol vehicles wants to go "
        "electric — but not all at once. This tool answers three questions: "
        "**which vehicles switch first**, **how healthy the batteries are**, "
        "and **how much money and CO₂ we save**."
    )
    st.info(
        f"Battery model accuracy on held-out cells: "
        f"**RMSE {bmetrics['rmse_cycles']:.0f} cycles · "
        f"MAPE {bmetrics['mape_pct']:.1f}%**"
    )

    st.write("Use the sidebar to open **Battery Health**, **Fleet Readiness**, "
             "or **Carbon Savings**.")


def page_battery():
    st.title("🔋 Battery Health & Remaining Useful Life")
    df = load_battery_data()
    cells = sorted(df["cell_id"].unique())
    cell_id = st.selectbox("Choose a battery cell", cells)

    hist = df[df["cell_id"] == cell_id]
    result = eb.predict_health(hist)
    colour = status_colour(result["status"])

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"<h3 style='color:{colour}'>State of Health<br>"
        f"{result['state_of_health']*100:.0f}%</h3>", unsafe_allow_html=True)
    c2.metric("Predicted cycle life", f"{result['predicted_cycle_life']:,}")
    c3.metric("Remaining useful life", f"{result['remaining_useful_life']:,} cycles")
    st.caption(f"Status: **{result['status'].upper()}** · "
               f"currently at cycle {result['current_cycle']:,}")

    # Capacity-fade curve: history + end-of-life threshold.
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["cycle"], y=hist["discharge_capacity_ah"],
        mode="lines", name="Measured capacity", line=dict(color=GREEN)))
    fig.add_hline(y=config.END_OF_LIFE_CAPACITY_AH, line_dash="dash",
                  line_color=RED, annotation_text="End of life (0.88 Ah)")
    fig.add_vline(x=result["predicted_cycle_life"], line_dash="dot",
                  line_color=AMBER, annotation_text="Predicted life")
    fig.update_layout(title=f"Capacity fade — {cell_id}",
                      xaxis_title="Cycle", yaxis_title="Discharge capacity (Ah)",
                      height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💬 Explain this battery", expanded=True):
        st.write(copilot.explain("battery", result))


def page_readiness():
    st.title("🚚 Fleet Electrification Readiness")
    scored = scored_fleet()

    ev_options = ["All"] + sorted(scored["ev_match"].unique())
    colf1, colf2 = st.columns([1, 2])
    ev_filter = colf1.selectbox("Filter by matched EV", ev_options)
    min_score = colf2.slider("Minimum readiness score", 0, 100, 0)

    view = scored.copy()
    if ev_filter != "All":
        view = view[view["ev_match"] == ev_filter]
    view = view[view["readiness_score"] >= min_score]

    st.caption(f"Showing {len(view)} of {len(scored)} vehicles, ranked by readiness.")
    st.dataframe(
        view[["vehicle_id", "vehicle_type", "duty_cycle", "readiness_score",
              "ev_match", "payback_years", "annual_savings_inr",
              "five_year_savings_inr"]],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    vid = st.selectbox("Inspect a vehicle", view["vehicle_id"].tolist() or scored["vehicle_id"].tolist())
    rec = er.vehicle_recommendation(vid, load_fleet_data())

    c1, c2, c3 = st.columns(3)
    c1.metric("Readiness score", f"{rec['readiness_score']}/100")
    c2.metric("Best EV match", rec["ev_match"])
    c3.metric("Payback", f"{rec['payback_years']} yrs")

    # 5-year cumulative diesel-vs-EV cost comparison.
    years = list(range(0, config.ROI_HORIZON_YEARS + 1))
    ev = config.EV_CATALOG[rec["ev_match"]]
    diesel_cum = [rec["annual_km"] * config.DIESEL_COST_PER_KM * y for y in years]
    ev_cum = [ev["price_inr"] + rec["annual_km"] * ev["cost_per_km"] * y for y in years]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=diesel_cum, mode="lines+markers",
                             name="Stay diesel", line=dict(color=RED)))
    fig.add_trace(go.Scatter(x=years, y=ev_cum, mode="lines+markers",
                             name=f"Switch to {rec['ev_match']}", line=dict(color=GREEN)))
    fig.update_layout(title=f"5-year total cost — {vid}",
                      xaxis_title="Year", yaxis_title="Cumulative cost (₹)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("💬 Why this recommendation?", expanded=True):
        st.write(copilot.explain("vehicle", rec))


def page_carbon():
    st.title("🌱 Carbon Savings")
    carbon = carbon_fleet()
    csum = ec.fleet_carbon_summary(carbon)

    c1, c2, c3 = st.columns(3)
    c1.metric("Current CO₂ (diesel)", f"{csum['total_current_co2_tonnes']:,.0f} t/yr")
    c2.metric("Electrified CO₂ (grid)", f"{csum['total_electrified_co2_tonnes']:,.0f} t/yr")
    c3.metric("CO₂ avoided", f"{csum['total_savings_co2_tonnes']:,.0f} t/yr",
              f"{csum['savings_pct']}%")

    # CO2 saved by vehicle type.
    by_type = csum["savings_by_vehicle_type_kg"]
    fig1 = go.Figure(go.Bar(
        x=list(by_type.keys()),
        y=[v / 1000 for v in by_type.values()],
        marker_color=GREEN))
    fig1.update_layout(title="CO₂ avoided by vehicle type",
                       xaxis_title="Vehicle type", yaxis_title="Tonnes CO₂ / yr",
                       height=380)

    # 5-year cumulative savings line.
    years = list(range(0, config.ROI_HORIZON_YEARS + 1))
    annual = csum["total_savings_co2_tonnes"]
    fig2 = go.Figure(go.Scatter(
        x=years, y=[annual * y for y in years],
        mode="lines+markers", line=dict(color=GREEN)))
    fig2.update_layout(title="Cumulative CO₂ avoided (5 years)",
                       xaxis_title="Year", yaxis_title="Tonnes CO₂", height=380)

    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(fig1, use_container_width=True)
    cc2.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Assumptions: Scope 1 tailpipe diesel vs Scope 2/3 grid electricity at "
        "0.7 kgCO₂/kWh (CEA India estimate). Operational emissions only."
    )

    with st.expander("💬 What's our carbon story?", expanded=True):
        st.write(copilot.explain("fleet", {**er.fleet_summary(scored_fleet()), **csum}))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGES = {
    "🏠 Home": page_home,
    "🔋 Battery Health": page_battery,
    "🚚 Fleet Readiness": page_readiness,
    "🌱 Carbon Savings": page_carbon,
}


def main():
    st.sidebar.title("EV Fleet Brain")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption("⚠️ Synthetic demo data · ET AI Hackathon 2026 · Problem 3")
    PAGES[choice]()
    st.markdown(
        "<hr><div style='text-align:center;color:gray;font-size:0.8em'>"
        "Synthetic demo data · anchored to real Indian EVs · "
        "battery model trained on Severson-style cycling data</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
