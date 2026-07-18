"""One-command pipeline: build everything from scratch.

    python run_pipeline.py [--n-vehicles N]

Generates the synthetic datasets and trains the battery model, then prints a
one-line summary from every engine so a fresh machine is demo-ready before
`streamlit run app.py`. Safe to re-run at any time.
"""
from __future__ import annotations

import argparse

import config
import generate_data
from engines import (engine_battery as eb, engine_carbon as ec,
                     engine_maintenance as em, engine_readiness as er,
                     engine_supply_chain as sc)


def main(n_vehicles: int = config.DEFAULT_N_VEHICLES) -> None:
    print("=" * 64)
    print("EV Fleet Intelligence Brain — full pipeline")
    print("=" * 64)

    print(f"\n[1/6] Generating synthetic data (n_vehicles={n_vehicles}) ...")
    generate_data.main(n_vehicles)

    print("\n[2/6] Training battery RUL model ...")
    _, metrics = eb.train_model(verbose=True)

    print("\n[3/6] Scoring fleet readiness ...")
    fsum = er.fleet_summary(er.score_fleet())
    print(f"  {fsum['ready_now']}/{fsum['total_vehicles']} ready; "
          f"5-yr savings ₹{fsum['total_five_year_savings_inr']/1e7:.2f} Cr")

    print("\n[4/6] Supply-chain risk ...")
    ssum = sc.supply_risk_summary()
    print(f"  overall risk {ssum['overall_risk_score']}/100; "
          f"single-sourced: {ssum['single_sourced_materials']}")

    print("\n[5/6] Maintenance & charging optimiser ...")
    msum = em.schedule_maintenance()
    csum = em.optimise_charging()
    print(f"  downtime -{msum['downtime_reduction_pct']}% ({msum['method']}); "
          f"charging saved ₹{csum['cost_saved_inr']:,.0f}/day")

    print("\n[6/6] Carbon savings (Scope 1/2/3) ...")
    carb = ec.fleet_carbon_summary()
    print(f"  {carb['total_savings_co2_tonnes']:,.0f} t CO₂ avoided/yr "
          f"({carb['savings_pct']}% net)")

    print("\n" + "=" * 64)
    print("Pipeline complete. Launch the dashboard with:")
    print("    streamlit run app.py")
    print("=" * 64)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-vehicles", type=int, default=config.DEFAULT_N_VEHICLES)
    main(p.parse_args().n_vehicles)
