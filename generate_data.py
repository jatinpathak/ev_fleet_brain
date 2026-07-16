"""Generate the synthetic (but realistic) datasets used by every engine.

Everything is driven by a single fixed seed so the same command always produces
byte-identical files -- the reproducibility guarantee the tests rely on.

Datasets produced
------------------
* battery_data_synthetic.csv : 150 LFP cells, per-cycle rows mimicking Severson.
* fleet_telematics_synthetic.csv : 300 diesel/petrol commercial vehicles.
* emission_factors.json : India grid / diesel emission factors.

NOTE: this is SYNTHETIC data, clearly labelled as such in the README and UI.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config


def _rng() -> np.random.Generator:
    return np.random.default_rng(config.RANDOM_SEED)


def generate_battery_data() -> pd.DataFrame:
    """150 LFP cells with per-cycle discharge capacity and curve summaries.

    Cycle lives are spread from ~150 to ~2300. Each cell fades from nominal
    capacity down through the 0.88 Ah end-of-life threshold following a smooth,
    slightly noisy curve. We also record per-cycle summaries of the
    capacity-vs-voltage (Q-V) curve so the battery engine can extract the key
    Severson "variance of delta-Q" feature from early cycles.
    """
    rng = _rng()
    n_cells = 150
    rows = []

    # Spread cycle lives log-uniformly between 150 and 2300 for realism.
    log_lives = rng.uniform(np.log(150), np.log(2300), size=n_cells)
    cycle_lives = np.sort(np.exp(log_lives).astype(int))

    for cell_idx in range(n_cells):
        cell_id = f"CELL_{cell_idx:03d}"
        life = int(cycle_lives[cell_idx])

        # Fade rate is tied to lifetime: short-lived cells fade faster. This
        # gives early-cycle features genuine predictive signal.
        fade_rate = (config.NOMINAL_CAPACITY_AH - config.END_OF_LIFE_CAPACITY_AH) / life

        # A cell-specific "roughness" that drives the early Q-V curve variance,
        # correlated (inversely) with lifetime -- the core Severson insight.
        roughness = 0.004 + 0.02 * (1.0 - (life - 150) / (2300 - 150))
        roughness = float(np.clip(roughness, 0.002, 0.03))

        for cycle in range(1, life + 1):
            # Capacity fade: mostly linear with mild square-root knee + noise.
            frac = cycle / life
            cap = (
                config.NOMINAL_CAPACITY_AH
                - fade_rate * cycle
                - 0.01 * np.sqrt(frac)
                + rng.normal(0, 0.0015)
            )
            cap = float(max(cap, 0.5))

            # Summary of the discharge Q-V curve for this cycle. dq_var is the
            # variance of the capacity-voltage delta; it grows as the cell ages
            # and is systematically larger for short-lived (rough) cells.
            dq_var = roughness * (1.0 + 1.5 * frac) + abs(rng.normal(0, 0.0008))
            avg_voltage = 3.30 - 0.05 * frac + rng.normal(0, 0.005)

            rows.append(
                {
                    "cell_id": cell_id,
                    "cycle": cycle,
                    "discharge_capacity_ah": round(cap, 5),
                    "qv_curve_variance": round(float(dq_var), 6),
                    "avg_discharge_voltage": round(float(avg_voltage), 4),
                    "cycle_life": life,
                }
            )

    return pd.DataFrame(rows)


def generate_fleet_data() -> pd.DataFrame:
    """300 commercial diesel/petrol vehicles with telematics attributes."""
    rng = _rng()
    n = 300
    vehicle_types = ["diesel", "petrol"]
    duty_cycles = ["urban", "highway", "mixed"]

    # Annual km log-normally distributed (most vehicles moderate, a few heavy).
    annual_km = rng.lognormal(mean=np.log(35000), sigma=0.45, size=n).astype(int)
    annual_km = np.clip(annual_km, 8000, 120000)

    duty = rng.choice(duty_cycles, size=n, p=[0.5, 0.2, 0.3])

    # Daily range depends on duty cycle; highway routes are longer per day.
    base_daily = annual_km / 300.0  # ~300 operating days
    daily_range = base_daily * np.where(
        duty == "highway", 1.4, np.where(duty == "urban", 0.8, 1.0)
    )
    daily_range = np.clip(daily_range.astype(int), 20, 500)

    payload = rng.integers(200, 1200, size=n)
    vtype = rng.choice(vehicle_types, size=n, p=[0.7, 0.3])

    df = pd.DataFrame(
        {
            "vehicle_id": [f"VEH_{i:03d}" for i in range(n)],
            "vehicle_type": vtype,
            "annual_km": annual_km,
            "avg_daily_range_km": daily_range,
            "payload_kg": payload,
            "duty_cycle": duty,
        }
    )
    return df


def generate_emission_factors() -> dict:
    return {
        "india_grid_co2_per_kwh": 0.7,
        "diesel_co2_per_liter": 2.68,
        "ev_efficiency_km_per_kwh": 4.5,
        "diesel_km_per_liter": 8,
    }


def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    battery = generate_battery_data()
    battery.to_csv(config.BATTERY_DATA_CSV, index=False)
    print(f"Wrote {config.BATTERY_DATA_CSV} "
          f"({battery['cell_id'].nunique()} cells, {len(battery):,} rows)")

    fleet = generate_fleet_data()
    fleet.to_csv(config.FLEET_DATA_CSV, index=False)
    print(f"Wrote {config.FLEET_DATA_CSV} ({len(fleet)} vehicles)")

    factors = generate_emission_factors()
    with open(config.EMISSION_FACTORS_JSON, "w") as f:
        json.dump(factors, f, indent=2)
    print(f"Wrote {config.EMISSION_FACTORS_JSON}")


if __name__ == "__main__":
    main()
