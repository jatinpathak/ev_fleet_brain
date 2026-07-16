"""Engine 1 - Battery Health & Remaining Useful Life (RUL).

Predicts a cell's total cycle life from its EARLY-cycle behaviour (first 100
cycles), following the Severson et al. approach: the log-variance of the change
in the capacity-vs-voltage curve between cycle 100 and cycle 10 is the single
most predictive feature. We add a few complementary early-cycle features and
train an XGBoost regressor on log10(cycle_life).

Public API
----------
* train_model()                 -> (model, metrics dict)  and pickles the model
* load_model()                  -> trained model (trains + caches if missing)
* predict_health(cell_history)  -> {state_of_health, predicted_cycle_life, remaining_useful_life}
* health_status(soh)            -> "healthy" | "degraded" | "critical"
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor

import config

FEATURE_COLUMNS = [
    "log_var_delta_q",     # the key Severson feature
    "capacity_fade_slope",
    "capacity_at_100",
    "capacity_range",
]


def _load_battery_df() -> pd.DataFrame:
    if not config.BATTERY_DATA_CSV.exists():
        raise FileNotFoundError(
            f"{config.BATTERY_DATA_CSV} not found. Run generate_data.py first."
        )
    return pd.read_csv(config.BATTERY_DATA_CSV)


def extract_features(cell_history: pd.DataFrame) -> dict:
    """Extract early-cycle features from one cell's per-cycle history.

    Uses only cycles up to EARLY_CYCLE_WINDOW (100). Robust to cells that have
    not yet reached 100 cycles -- it uses whatever early data is available.
    """
    hist = cell_history.sort_values("cycle")
    early = hist[hist["cycle"] <= config.EARLY_CYCLE_WINDOW]
    if len(early) < 2:
        early = hist  # brand-new cell: fall back to all we have

    # Key Severson feature: log-variance of the delta-Q(V) curve between an
    # early and a later early cycle. We approximate delta-Q via the recorded
    # per-cycle Q-V curve variance, differenced across the early window.
    q_var = early["qv_curve_variance"].to_numpy()
    delta_q = np.diff(q_var)
    var_delta_q = float(np.var(delta_q)) if len(delta_q) > 0 else 1e-9
    log_var_delta_q = float(np.log10(max(var_delta_q, 1e-12)))

    caps = early["discharge_capacity_ah"].to_numpy()
    cycles = early["cycle"].to_numpy()

    # Capacity fade slope over the early window (Ah per cycle, negative).
    if len(cycles) >= 2 and np.ptp(cycles) > 0:
        slope = float(np.polyfit(cycles, caps, 1)[0])
    else:
        slope = 0.0

    capacity_at_100 = float(caps[-1])
    capacity_range = float(caps.max() - caps.min())

    return {
        "log_var_delta_q": log_var_delta_q,
        "capacity_fade_slope": slope,
        "capacity_at_100": capacity_at_100,
        "capacity_range": capacity_range,
    }


def _build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell_id, hist in df.groupby("cell_id"):
        feats = extract_features(hist)
        feats["cell_id"] = cell_id
        feats["cycle_life"] = int(hist["cycle_life"].iloc[0])
        rows.append(feats)
    return pd.DataFrame(rows)


def train_model(verbose: bool = True):
    """Train the XGBoost RUL model, report metrics, and pickle it."""
    df = _load_battery_df()
    table = _build_feature_table(df)

    X = table[FEATURE_COLUMNS].to_numpy()
    y = np.log10(table["cycle_life"].to_numpy())

    # 60/40 train/test split (deterministic).
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.40, random_state=config.RANDOM_SEED
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=config.RANDOM_SEED,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    # Metrics in real cycle units (undo the log10).
    pred_cycles = np.power(10, model.predict(X_test))
    true_cycles = np.power(10, y_test)
    rmse = float(np.sqrt(mean_squared_error(true_cycles, pred_cycles)))
    mape = float(mean_absolute_percentage_error(true_cycles, pred_cycles) * 100)

    metrics = {"rmse_cycles": rmse, "mape_pct": mape,
               "n_train": len(X_train), "n_test": len(X_test)}

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BATTERY_MODEL_PKL, "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_COLUMNS, "metrics": metrics}, f)

    if verbose:
        print(f"Battery RUL model trained on {len(X_train)} cells, "
              f"tested on {len(X_test)}.")
        print(f"  RMSE = {rmse:.1f} cycles | MAPE = {mape:.1f}%")

    return model, metrics


def load_model():
    """Return the trained battery model, training it on demand if missing."""
    if not config.BATTERY_MODEL_PKL.exists():
        train_model(verbose=False)
    with open(config.BATTERY_MODEL_PKL, "rb") as f:
        return pickle.load(f)


def health_status(soh: float) -> str:
    """Map a state-of-health fraction to a colour-coding status."""
    if soh >= config.SOH_HEALTHY_MIN:
        return "healthy"
    if soh < config.SOH_CRITICAL_MAX:
        return "critical"
    return "degraded"


def predict_health(cell_history: pd.DataFrame) -> dict:
    """Predict SoH, total cycle life and remaining useful life for one cell.

    Parameters
    ----------
    cell_history : DataFrame of per-cycle rows for a SINGLE cell (must include
        cycle, discharge_capacity_ah, qv_curve_variance).
    """
    bundle = load_model()
    model = bundle["model"]
    feature_cols = bundle["features"]

    feats = extract_features(cell_history)
    X = np.array([[feats[c] for c in feature_cols]])
    predicted_cycle_life = float(np.power(10, model.predict(X))[0])

    hist = cell_history.sort_values("cycle")
    current_cycle = int(hist["cycle"].max())
    current_capacity = float(hist["discharge_capacity_ah"].iloc[-1])

    # State of health as fraction of nominal capacity, clamped to [0, 1].
    soh = float(np.clip(current_capacity / config.NOMINAL_CAPACITY_AH, 0.0, 1.0))

    # RUL: predicted total life minus cycles already used (never negative).
    remaining_useful_life = int(max(predicted_cycle_life - current_cycle, 0))

    return {
        "state_of_health": round(soh, 4),
        "predicted_cycle_life": int(round(predicted_cycle_life)),
        "remaining_useful_life": remaining_useful_life,
        "current_cycle": current_cycle,
        "status": health_status(soh),
    }


if __name__ == "__main__":
    _, m = train_model()
    print("Metrics:", m)
