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
    assert config.BATTERY_DATA_CSV.exists()
    assert config.FLEET_DATA_CSV.exists()
    assert config.EMISSION_FACTORS_JSON.exists()


def test_reproducible_same_seed():
    """Same seed -> byte-identical files across two runs."""
    generate_data.main()
    h1_batt = _md5(config.BATTERY_DATA_CSV)
    h1_fleet = _md5(config.FLEET_DATA_CSV)
    generate_data.main()
    assert _md5(config.BATTERY_DATA_CSV) == h1_batt
    assert _md5(config.FLEET_DATA_CSV) == h1_fleet


def test_battery_shape_and_no_nan():
    df = pd.read_csv(config.BATTERY_DATA_CSV)
    assert df["cell_id"].nunique() == 150
    assert not df.isna().any().any()
    # Cycle lives spread across the expected range.
    lives = df.groupby("cell_id")["cycle_life"].first()
    assert lives.min() >= 150
    assert lives.max() <= 2300


def test_fleet_shape_and_no_nan():
    df = pd.read_csv(config.FLEET_DATA_CSV)
    assert len(df) == 300
    assert not df.isna().any().any()
    assert set(df["duty_cycle"].unique()).issubset({"urban", "highway", "mixed"})
    assert set(df["vehicle_type"].unique()).issubset({"diesel", "petrol"})
    assert (df["annual_km"] > 0).all()


def test_emission_factors():
    with open(config.EMISSION_FACTORS_JSON) as f:
        factors = json.load(f)
    for key in ("india_grid_co2_per_kwh", "diesel_co2_per_liter",
                "ev_efficiency_km_per_kwh", "diesel_km_per_liter"):
        assert key in factors
    assert factors["india_grid_co2_per_kwh"] == 0.7
