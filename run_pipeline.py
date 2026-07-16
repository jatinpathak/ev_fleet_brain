"""One-command pipeline: build everything from scratch.

    python run_pipeline.py

Generates the synthetic datasets and trains both models, so a fresh machine is
demo-ready before `streamlit run app.py`. Safe to re-run at any time.
"""
from __future__ import annotations

import generate_data
import engine_battery as eb
import engine_readiness as er
import engine_carbon as ec


def main() -> None:
    print("=" * 60)
    print("EV Fleet Intelligence Brain — full pipeline")
    print("=" * 60)

    print("\n[1/4] Generating synthetic data ...")
    generate_data.main()

    print("\n[2/4] Training battery RUL model ...")
    _, metrics = eb.train_model(verbose=True)

    print("\n[3/4] Scoring fleet readiness ...")
    scored = er.score_fleet()
    fsum = er.fleet_summary(scored)
    print(f"  {fsum['ready_now']}/{fsum['total_vehicles']} vehicles ready; "
          f"5-yr savings ₹{fsum['total_five_year_savings_inr']/1e7:.2f} Cr")

    print("\n[4/4] Computing carbon savings ...")
    csum = ec.fleet_carbon_summary()
    print(f"  {csum['total_savings_co2_tonnes']:,.0f} tonnes CO₂ avoided/yr "
          f"({csum['savings_pct']}% reduction)")

    print("\n" + "=" * 60)
    print("Pipeline complete. Launch the dashboard with:")
    print("    streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
