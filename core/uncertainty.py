"""Confidence / prediction intervals exposed on every model prediction.

Strategy (in preference order), so the app degrades gracefully:
    1. Conformal prediction via ``mapie`` if it is installed.
    2. Split-conformal residual quantiles (stdlib + numpy) — always available.

Split-conformal is itself a valid conformal method: we hold out a calibration
set, measure the absolute residuals, and take the ``coverage`` quantile as the
half-width of a distribution-free prediction interval. This means the app has a
principled interval even when ``mapie`` is absent, satisfying the audit's "no
confidence intervals" gap without a hard dependency.
"""
from __future__ import annotations

import numpy as np

from core.logging_config import get_logger

log = get_logger(__name__)


def mapie_available() -> bool:
    try:
        import mapie  # noqa: F401
        return True
    except Exception:
        return False


def conformal_halfwidth(residuals: np.ndarray, coverage: float = 0.90) -> float:
    """Half-width of a split-conformal interval from calibration residuals.

    Uses the finite-sample-adjusted quantile of the absolute residuals so the
    interval has (approximately) the requested marginal coverage.
    """
    resid = np.abs(np.asarray(residuals, dtype=float))
    n = len(resid)
    if n == 0:
        return 0.0
    # Conformal quantile level with the standard (n+1) finite-sample correction.
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return float(np.quantile(resid, level, method="higher"))


def interval_from_halfwidth(point: float, halfwidth: float) -> tuple[float, float]:
    """Symmetric interval around a point prediction."""
    return point - halfwidth, point + halfwidth


def prediction_interval(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_new: np.ndarray,
    coverage: float = 0.90,
):
    """Return (lower, upper, method) prediction intervals for ``X_new``.

    Tries mapie's conformal regressor; falls back to split-conformal residuals.
    All maths happens in whatever space ``model`` predicts in (the battery model
    predicts log10(cycle_life)); callers convert back as needed.
    """
    X_cal = np.asarray(X_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    X_new = np.asarray(X_new, dtype=float)

    if mapie_available():
        try:
            from mapie.regression import MapieRegressor

            mapie = MapieRegressor(estimator=model, method="base", cv="prefit")
            mapie.fit(X_cal, y_cal)
            _, intervals = mapie.predict(X_new, alpha=1 - coverage)
            low = intervals[:, 0, 0]
            high = intervals[:, 1, 0]
            return low, high, "conformal (mapie)"
        except Exception as exc:  # pragma: no cover - defensive
            log.info("mapie_failed_fallback", extra={"kv": {"error": str(exc)}})

    # Split-conformal fallback.
    point_cal = np.asarray(model.predict(X_cal), dtype=float)
    hw = conformal_halfwidth(y_cal - point_cal, coverage)
    point_new = np.asarray(model.predict(X_new), dtype=float)
    return point_new - hw, point_new + hw, "split-conformal"


def bootstrap_interval(values: np.ndarray, coverage: float = 0.90, n_boot: int = 1000,
                       seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap interval for the MEAN of a sample.

    Used where we summarise a distribution (e.g. mean SoH) rather than wrap a
    regressor. Kept simple and deterministic.
    """
    vals = np.asarray(values, dtype=float)
    if len(vals) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)
    ])
    lo = float(np.quantile(means, (1 - coverage) / 2))
    hi = float(np.quantile(means, 1 - (1 - coverage) / 2))
    return lo, hi
