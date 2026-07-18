"""Generate the synthetic (but realistic) datasets used by every engine.

Everything is driven by a single fixed seed so the same command always produces
byte-identical files -- the reproducibility guarantee the tests rely on. The
fleet size is parametrized (``--n-vehicles``) so we can prove the app scales to
10,000 vehicles.

Datasets produced
-----------------
* battery_data_synthetic.csv      : 150 LFP cells, per-cycle rows (Severson-style).
* fleet_telematics_synthetic.csv  : N diesel/petrol commercial vehicles.
* suppliers_synthetic.csv         : battery-material supply-chain network.
* maintenance_events_synthetic.csv: per-vehicle maintenance-due backlog.
* emission_factors.json           : India grid / diesel emission factors.

NOTE: every file here is SYNTHETIC, clearly labelled as such in the README and
UI. The battery module additionally supports REAL Severson/MATR data when the
optional BatteryML dataset is present (see README).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config


def _rng(offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(config.RANDOM_SEED + offset)


# ---------------------------------------------------------------------------
# Battery (unchanged: 150 cells, the real-data-shaped headline dataset)
# ---------------------------------------------------------------------------
def generate_battery_data() -> pd.DataFrame:
    """150 LFP cells with per-cycle discharge capacity and curve summaries."""
    rng = _rng()
    n_cells = 150
    rows = []

    log_lives = rng.uniform(np.log(150), np.log(2300), size=n_cells)
    cycle_lives = np.sort(np.exp(log_lives).astype(int))

    for cell_idx in range(n_cells):
        cell_id = f"CELL_{cell_idx:03d}"
        life = int(cycle_lives[cell_idx])

        fade_rate = (config.NOMINAL_CAPACITY_AH - config.END_OF_LIFE_CAPACITY_AH) / life
        roughness = 0.004 + 0.02 * (1.0 - (life - 150) / (2300 - 150))
        roughness = float(np.clip(roughness, 0.002, 0.03))

        for cycle in range(1, life + 1):
            frac = cycle / life
            cap = (
                config.NOMINAL_CAPACITY_AH
                - fade_rate * cycle
                - 0.01 * np.sqrt(frac)
                + rng.normal(0, 0.0015)
            )
            cap = float(max(cap, 0.5))

            dq_var = roughness * (1.0 + 1.5 * frac) + abs(rng.normal(0, 0.0008))
            avg_voltage = 3.30 - 0.05 * frac + rng.normal(0, 0.005)
            # Thermal + internal-resistance proxies for anomaly detection.
            temp_c = 30.0 + 8.0 * frac + rng.normal(0, 0.6)
            int_resistance = 0.015 + 0.010 * frac + rng.normal(0, 0.0006)

            rows.append(
                {
                    "cell_id": cell_id,
                    "cycle": cycle,
                    "discharge_capacity_ah": round(cap, 5),
                    "qv_curve_variance": round(float(dq_var), 6),
                    "avg_discharge_voltage": round(float(avg_voltage), 4),
                    "avg_cell_temp_c": round(float(temp_c), 3),
                    "internal_resistance_ohm": round(float(int_resistance), 6),
                    "cycle_life": life,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fleet (parametrized to N vehicles)
# ---------------------------------------------------------------------------
def generate_fleet_data(n_vehicles: int = config.DEFAULT_N_VEHICLES) -> pd.DataFrame:
    """N commercial diesel/petrol vehicles with telematics attributes."""
    rng = _rng()
    n = int(n_vehicles)
    vehicle_types = ["diesel", "petrol"]
    duty_cycles = ["urban", "highway", "mixed"]
    depots = ["Pune", "Delhi", "Chennai", "Kolkata", "Ahmedabad"]

    annual_km = rng.lognormal(mean=np.log(35000), sigma=0.45, size=n).astype(int)
    annual_km = np.clip(annual_km, 8000, 120000)

    duty = rng.choice(duty_cycles, size=n, p=[0.5, 0.2, 0.3])

    base_daily = annual_km / 300.0
    daily_range = base_daily * np.where(
        duty == "highway", 1.4, np.where(duty == "urban", 0.8, 1.0)
    )
    daily_range = np.clip(daily_range.astype(int), 20, 500)

    payload = rng.integers(200, 1200, size=n)
    vtype = rng.choice(vehicle_types, size=n, p=[0.7, 0.3])
    depot = rng.choice(depots, size=n)
    # A synthetic lat/lon cloud per depot for the geospatial views.
    depot_coords = {
        "Pune": (18.52, 73.86), "Delhi": (28.61, 77.21), "Chennai": (13.08, 80.27),
        "Kolkata": (22.57, 88.36), "Ahmedabad": (23.03, 72.58),
    }
    lat = np.array([depot_coords[d][0] for d in depot]) + rng.normal(0, 0.15, n)
    lon = np.array([depot_coords[d][1] for d in depot]) + rng.normal(0, 0.15, n)

    df = pd.DataFrame(
        {
            "vehicle_id": [f"VEH_{i:05d}" for i in range(n)],
            "vehicle_type": vtype,
            "annual_km": annual_km,
            "avg_daily_range_km": daily_range,
            "payload_kg": payload,
            "duty_cycle": duty,
            "depot": depot,
            "lat": np.round(lat, 4),
            "lon": np.round(lon, 4),
        }
    )
    return df


# ---------------------------------------------------------------------------
# Maintenance backlog (derived from the fleet)
# ---------------------------------------------------------------------------
def generate_maintenance_events(fleet_df: pd.DataFrame) -> pd.DataFrame:
    """One upcoming maintenance job per vehicle within the planning horizon."""
    rng = _rng(offset=1)
    n = len(fleet_df)
    job_types = ["brake", "tyre", "battery-check", "general-service", "inspection"]
    # Higher-mileage vehicles fall due sooner. Due dates are spread across TWICE
    # the planning horizon, so only part of the fleet is due inside the current
    # workshop window (the rest is genuinely future work) — a realistic backlog
    # the optimiser can actually clear rather than a hopeless pile-up.
    km = fleet_df["annual_km"].to_numpy()
    urgency = (km - km.min()) / max(np.ptp(km), 1)
    due_day = np.clip(
        (config.WORKSHOP_DAYS * (1 - urgency) + rng.normal(0, 2, n)).astype(int),
        0, config.WORKSHOP_DAYS,
    )
    service_hours = rng.choice([2, 4, 6], size=n, p=[0.4, 0.4, 0.2])
    priority = np.where(due_day <= 3, "high", np.where(due_day <= 8, "medium", "low"))

    return pd.DataFrame(
        {
            "vehicle_id": fleet_df["vehicle_id"].to_numpy(),
            "depot": fleet_df["depot"].to_numpy(),
            "job_type": rng.choice(job_types, size=n),
            "due_day": due_day,
            "service_hours": service_hours,
            "priority": priority,
        }
    )


# ---------------------------------------------------------------------------
# Supply chain (curated skeleton + seeded numeric fields)
# ---------------------------------------------------------------------------
def generate_suppliers() -> pd.DataFrame:
    """A 3-tier battery-material supply network with parent linkage.

    Tier-3 = raw-material miners, Tier-2 = refiners/cathode, Tier-1 = cell makers
    that feed our packs. ``supplies_to`` encodes the edge to the downstream
    supplier so the UI can render a real network graph and trace lineage.
    """
    rng = _rng(offset=2)
    # (id, name, tier, material, country, parent_id)
    skeleton = [
        # Tier-1 cell manufacturers (feed our packs directly).
        ("S01", "Amara Cells",      "Tier-1", "LFP cell", "India",       None),
        ("S02", "GigaPack Co",      "Tier-1", "LFP cell", "China",       None),
        ("S03", "NovaCell NMC",     "Tier-1", "NMC cell", "South Korea", None),
        # Tier-2 refiners / cathode.
        ("S04", "RefinLi Ltd",      "Tier-2", "Lithium",  "China",       "S01"),
        ("S05", "AusLith Refining", "Tier-2", "Lithium",  "Australia",   "S02"),
        ("S06", "CobaltWorks",      "Tier-2", "Cobalt",   "China",       "S03"),
        ("S07", "NickelPure",       "Tier-2", "Nickel",   "Indonesia",   "S03"),
        # Tier-3 miners.
        ("S08", "Salar Mining",     "Tier-3", "Lithium",  "Chile",       "S04"),
        ("S09", "Pilbara Ore",      "Tier-3", "Lithium",  "Australia",   "S05"),
        ("S10", "Katanga Mines",    "Tier-3", "Cobalt",   "DR Congo",    "S06"),
        ("S11", "Sulawesi Nickel",  "Tier-3", "Nickel",   "Indonesia",   "S07"),
        ("S12", "Argentine Li",     "Tier-3", "Lithium",  "Argentina",   "S04"),
    ]
    rows = []
    for sid, name, tier, material, country, parent in skeleton:
        rows.append(
            {
                "supplier_id": sid,
                "supplier_name": name,
                "tier": tier,
                "material": material,
                "country": country,
                "supplies_to": parent if parent is not None else "",
                "annual_volume": int(rng.integers(2000, 20000)),
                "quality_defect_rate": round(float(rng.uniform(0.005, 0.05)), 4),
                "on_time_delivery_rate": round(float(rng.uniform(0.80, 0.99)), 3),
            }
        )
    df = pd.DataFrame(rows)

    # single_source_flag: material with exactly one supplier in the table.
    counts = df["material"].value_counts()
    df["single_source_flag"] = df["material"].map(lambda m: counts[m] == 1)
    return df


def generate_emission_factors() -> dict:
    return {
        "india_grid_co2_per_kwh": 0.7,
        "diesel_co2_per_liter": 2.68,
        "ev_efficiency_km_per_kwh": 4.5,
        "diesel_km_per_liter": 8,
        "_source": "Illustrative: CEA India grid factor ~0.7 kgCO2/kWh; verify latest before submission.",
    }


def main(n_vehicles: int = config.DEFAULT_N_VEHICLES) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    battery = generate_battery_data()
    battery.to_csv(config.BATTERY_DATA_CSV, index=False)
    print(f"Wrote {config.BATTERY_DATA_CSV} "
          f"({battery['cell_id'].nunique()} cells, {len(battery):,} rows)")

    fleet = generate_fleet_data(n_vehicles)
    fleet.to_csv(config.FLEET_DATA_CSV, index=False)
    print(f"Wrote {config.FLEET_DATA_CSV} ({len(fleet):,} vehicles)")

    maint = generate_maintenance_events(fleet)
    maint.to_csv(config.MAINTENANCE_CSV, index=False)
    print(f"Wrote {config.MAINTENANCE_CSV} ({len(maint):,} events)")

    suppliers = generate_suppliers()
    suppliers.to_csv(config.SUPPLIERS_CSV, index=False)
    print(f"Wrote {config.SUPPLIERS_CSV} ({len(suppliers)} suppliers)")

    factors = generate_emission_factors()
    with open(config.EMISSION_FACTORS_JSON, "w") as f:
        json.dump(factors, f, indent=2)
    print(f"Wrote {config.EMISSION_FACTORS_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate EV Fleet Brain datasets.")
    parser.add_argument("--n-vehicles", type=int, default=config.DEFAULT_N_VEHICLES,
                        help="fleet size (use 10000 for the scalability demo)")
    args = parser.parse_args()
    main(args.n_vehicles)
