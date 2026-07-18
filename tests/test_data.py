"""ROUND 1 - Data generation tests: reproducibility and integrity."""
import hashlib
import json

import pandas as pd

import config
import generate_data


def _md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_files_exist():
    generate_data.main()
    for path in (config.BATTERY_DATA_CSV, config.FLEET_DATA_CSV,
                 config.SUPPLIERS_CSV, config.MAINTENANCE_CSV,
                 config.EMISSION_FACTORS_JSON):
        assert path.exists(), path


def test_reproducible_same_seed():
    """Same seed -> byte-identical files across two runs."""
    generate_data.main()
    hashes = {p: _md5(p) for p in (config.BATTERY_DATA_CSV, config.FLEET_DATA_CSV,
                                   config.SUPPLIERS_CSV, config.MAINTENANCE_CSV)}
    generate_data.main()
    for p, h in hashes.items():
        assert _md5(p) == h, p


def test_battery_shape_and_no_nan():
    generate_data.main()
    df = pd.read_csv(config.BATTERY_DATA_CSV)
    assert df["cell_id"].nunique() == 150
    assert not df.isna().any().any()
    # New anomaly-proxy columns are present.
    for col in ("avg_cell_temp_c", "internal_resistance_ohm"):
        assert col in df.columns
    lives = df.groupby("cell_id")["cycle_life"].first()
    assert lives.min() >= 150 and lives.max() <= 2300


def test_fleet_shape_and_scaling():
    generate_data.main()  # default 300
    df = pd.read_csv(config.FLEET_DATA_CSV)
    assert len(df) == config.DEFAULT_N_VEHICLES
    assert not df.isna().any().any()
    assert set(df["duty_cycle"].unique()).issubset({"urban", "highway", "mixed"})
    assert (df["annual_km"] > 0).all()
    # Parametrised scaling works.
    generate_data.main(n_vehicles=1000)
    assert len(pd.read_csv(config.FLEET_DATA_CSV)) == 1000
    generate_data.main()  # restore default for later tests


def test_suppliers_have_required_columns():
    generate_data.main()
    df = pd.read_csv(config.SUPPLIERS_CSV)
    for col in ("supplier_id", "tier", "material", "country", "annual_volume",
                "quality_defect_rate", "on_time_delivery_rate", "single_source_flag"):
        assert col in df.columns
    assert set(df["tier"].unique()).issubset({"Tier-1", "Tier-2", "Tier-3"})


def test_emission_factors():
    with open(config.EMISSION_FACTORS_JSON) as f:
        factors = json.load(f)
    for key in ("india_grid_co2_per_kwh", "diesel_co2_per_liter",
                "ev_efficiency_km_per_kwh", "diesel_km_per_liter"):
        assert key in factors
    assert factors["india_grid_co2_per_kwh"] == 0.7
