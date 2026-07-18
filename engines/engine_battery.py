"""Engine 1 - Battery Health, RUL, anomaly detection & confidence intervals.

Predicts a cell's total cycle life from its EARLY-cycle behaviour (first 100
cycles), following Severson et al.: the log-variance of the change in the
capacity-vs-voltage curve is the single most predictive feature. We train an
XGBoost regressor on log10(cycle_life), then add three audit-closing layers:

* Confidence intervals — a split-conformal (or mapie) prediction interval on RUL.
* Anomaly detection — IsolationForest over per-cell degradation signals flags
  cells fading abnormally fast vs their cohort.
* Explainability — SHAP / feature-importance drivers per prediction.

Public API
----------
* train_model()                 -> (model, metrics)  and pickles the bundle
* load_model()                  -> trained bundle (trains + caches if missing)
* predict_health(cell_history)  -> dict incl. RUL prediction interval + drivers
* detect_anomalies(df)          -> DataFrame of per-cell anomaly flags
* kpis(df)                       -> list[KPI] for the dashboard
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import config
from core import explain, uncertainty
from core.kpis import KPI, tone_for
from core.logging_config import get_logger, timed

log = get_logger(__name__)

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
    """Extract early-cycle features from one cell's per-cycle history."""
    hist = cell_history.sort_values("cycle")
    early = hist[hist["cycle"] <= config.EARLY_CYCLE_WINDOW]
    if len(early) < 2:
        early = hist

    q_var = early["qv_curve_variance"].to_numpy()
    delta_q = np.diff(q_var)
    var_delta_q = float(np.var(delta_q)) if len(delta_q) > 0 else 1e-9
    log_var_delta_q = float(np.log10(max(var_delta_q, 1e-12)))

    caps = early["discharge_capacity_ah"].to_numpy()
    cycles = early["cycle"].to_numpy()

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


def _anomaly_features(cell_history: pd.DataFrame) -> dict:
    """Degradation signals used for cohort anomaly detection."""
    hist = cell_history.sort_values("cycle")
    early = hist[hist["cycle"] <= config.EARLY_CYCLE_WINDOW]
    if len(early) < 2:
        early = hist
    cyc = early["cycle"].to_numpy()

    def _slope(col: str) -> float:
        if col not in early or len(cyc) < 2 or np.ptp(cyc) == 0:
            return 0.0
        return float(np.polyfit(cyc, early[col].to_numpy(), 1)[0])

    return {
        "fade_rate": -_slope("discharge_capacity_ah"),   # positive = fading fast
        "temp_slope": _slope("avg_cell_temp_c"),
        "resistance_slope": _slope("internal_resistance_ohm"),
        "dq_var_mean": float(early["qv_curve_variance"].mean()),
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
    """Train the XGBoost RUL model, report metrics, calibrate CIs, and pickle it."""
    df = _load_battery_df()
    with timed(log, "train_battery", n_cells=df["cell_id"].nunique()):
        table = _build_feature_table(df)

        X = table[FEATURE_COLUMNS].to_numpy()
        y = np.log10(table["cycle_life"].to_numpy())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.40, random_state=config.RANDOM_SEED
        )

        model = XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            random_state=config.RANDOM_SEED, objective="reg:squarederror",
        )
        model.fit(X_train, y_train)

        pred_cycles = np.power(10, model.predict(X_test))
        true_cycles = np.power(10, y_test)
        rmse = float(np.sqrt(mean_squared_error(true_cycles, pred_cycles)))
        mape = float(mean_absolute_percentage_error(true_cycles, pred_cycles) * 100)

        # Calibrate a split-conformal half-width (in log space) on the test set.
        ci_halfwidth_log = uncertainty.conformal_halfwidth(
            y_test - model.predict(X_test), config.RUL_PI_COVERAGE
        )

    metrics = {
        "rmse_cycles": rmse, "mape_pct": mape,
        "n_train": len(X_train), "n_test": len(X_test),
        "ci_coverage": config.RUL_PI_COVERAGE,
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BATTERY_MODEL_PKL, "wb") as f:
        pickle.dump({
            "model": model, "features": FEATURE_COLUMNS, "metrics": metrics,
            "ci_halfwidth_log": float(ci_halfwidth_log),
            "cal_X": X_test, "cal_y": y_test,
        }, f)

    if verbose:
        print(f"Battery RUL model trained on {len(X_train)} cells, tested on {len(X_test)}.")
        print(f"  RMSE = {rmse:.1f} cycles | MAPE = {mape:.1f}% | "
              f"{int(config.RUL_PI_COVERAGE*100)}% PI half-width "
              f"= 10^{ci_halfwidth_log:.3f}x")

    return model, metrics


def load_model():
    """Return the trained battery bundle, training it on demand if missing."""
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
    """Predict SoH, cycle life, RUL (with a prediction interval) and drivers."""
    bundle = load_model()
    model = bundle["model"]
    feature_cols = bundle["features"]

    feats = extract_features(cell_history)
    X = np.array([[feats[c] for c in feature_cols]])
    pred_log = float(model.predict(X)[0])
    predicted_cycle_life = float(np.power(10, pred_log))

    # Prediction interval: prefer a live conformal interval on the stored
    # calibration set; fall back to the pickled split-conformal half-width.
    try:
        low_log, high_log, ci_method = uncertainty.prediction_interval(
            model, bundle["cal_X"], bundle["cal_y"], X, config.RUL_PI_COVERAGE
        )
        pi_low = int(round(float(np.power(10, low_log[0]))))
        pi_high = int(round(float(np.power(10, high_log[0]))))
    except Exception:
        hw = bundle.get("ci_halfwidth_log", 0.0)
        pi_low = int(round(np.power(10, pred_log - hw)))
        pi_high = int(round(np.power(10, pred_log + hw)))
        ci_method = "split-conformal"

    hist = cell_history.sort_values("cycle")
    current_cycle = int(hist["cycle"].max())
    current_capacity = float(hist["discharge_capacity_ah"].iloc[-1])
    soh = float(np.clip(current_capacity / config.NOMINAL_CAPACITY_AH, 0.0, 1.0))

    remaining_useful_life = int(max(predicted_cycle_life - current_cycle, 0))
    rul_low = int(max(pi_low - current_cycle, 0))
    rul_high = int(max(pi_high - current_cycle, 0))

    drivers = explain.top_drivers(model, X, feature_cols, k=3)

    return {
        "cell_id": str(hist["cell_id"].iloc[0]) if "cell_id" in hist else None,
        "state_of_health": round(soh, 4),
        "predicted_cycle_life": int(round(predicted_cycle_life)),
        "predicted_cycle_life_pi": [pi_low, pi_high],
        "remaining_useful_life": remaining_useful_life,
        "remaining_useful_life_pi": [rul_low, rul_high],
        "ci_coverage": config.RUL_PI_COVERAGE,
        "ci_method": ci_method,
        "current_cycle": current_cycle,
        "status": health_status(soh),
        "top_drivers": drivers,
    }


def soh_trend(cell_history: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle state-of-health trend for plotting."""
    hist = cell_history.sort_values("cycle").copy()
    hist["soh"] = (hist["discharge_capacity_ah"] / config.NOMINAL_CAPACITY_AH).clip(0, 1)
    return hist[["cycle", "discharge_capacity_ah", "soh"]]


def detect_anomalies(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Flag cells degrading abnormally fast vs their cohort (IsolationForest)."""
    df = df if df is not None else _load_battery_df()
    feats = pd.DataFrame(
        {"cell_id": cid, **_anomaly_features(hist)}
        for cid, hist in df.groupby("cell_id")
    )
    feat_cols = ["fade_rate", "temp_slope", "resistance_slope", "dq_var_mean"]
    X = feats[feat_cols].to_numpy()

    iso = IsolationForest(
        contamination=config.ANOMALY_CONTAMINATION,
        random_state=config.RANDOM_SEED,
    )
    labels = iso.fit_predict(X)             # -1 = anomaly
    scores = -iso.score_samples(X)          # higher = more anomalous
    feats["anomaly"] = labels == -1
    feats["anomaly_score"] = np.round(scores, 4)
    return feats.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def kpis(df: pd.DataFrame | None = None) -> list[KPI]:
    """Headline battery KPIs for the dashboard."""
    df = df if df is not None else _load_battery_df()
    bundle = load_model()
    metrics = bundle["metrics"]
    anomalies = detect_anomalies(df)

    # Mean SoH across cells at their latest observed cycle.
    latest = df.sort_values("cycle").groupby("cell_id").tail(1)
    mean_soh = float((latest["discharge_capacity_ah"] / config.NOMINAL_CAPACITY_AH)
                     .clip(0, 1).mean())
    n_anom = int(anomalies["anomaly"].sum())

    return [
        KPI("Model accuracy (RMSE)", f"{metrics['rmse_cycles']:.0f}", "cycles",
            f"Held-out error on {metrics['n_test']} unseen cells — a real, not synthetic, number.",
            tone_for(metrics["rmse_cycles"], 80, 150)),
        KPI("Model error (MAPE)", f"{metrics['mape_pct']:.1f}", "%",
            "Typical percentage error of the life prediction.",
            tone_for(metrics["mape_pct"], 8, 15)),
        KPI("Fast-degrading cells", f"{n_anom}", f"/ {len(anomalies)}",
            "Cells fading abnormally fast vs their cohort — prioritise for inspection.",
            "warn" if n_anom else "good"),
        KPI("Mean state of health", f"{mean_soh*100:.0f}", "%",
            "Average remaining capacity across the pack fleet.",
            tone_for(mean_soh, 0.85, 0.80, higher_is_worse=False)),
    ]


if __name__ == "__main__":
    _, m = train_model()
    print("Metrics:", m)
    print(detect_anomalies().head().to_string(index=False))
