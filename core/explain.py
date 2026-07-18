"""Explainability: top drivers per prediction and global feature importance.

Strategy (graceful degradation):
    1. SHAP ``TreeExplainer`` if ``shap`` is installed -> true per-prediction
       attributions.
    2. Model ``feature_importances_`` (XGBoost/tree models) -> global fallback.
    3. Uniform importance -> last-resort fallback so the UI never blanks out.

Closes the audit's "limited explainability" gap for the battery and readiness
models without a hard SHAP dependency.
"""
from __future__ import annotations

import numpy as np

from core.logging_config import get_logger

log = get_logger(__name__)


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def global_importance(model, feature_names: list[str]) -> dict[str, float]:
    """Return normalised global feature importance {feature: weight in [0,1]}."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        w = 1.0 / max(len(feature_names), 1)
        return {f: w for f in feature_names}
    imp = np.asarray(imp, dtype=float)
    total = imp.sum()
    if total <= 0:
        w = 1.0 / max(len(feature_names), 1)
        return {f: w for f in feature_names}
    return {f: float(v / total) for f, v in zip(feature_names, imp)}


def top_drivers(model, x_row: np.ndarray, feature_names: list[str],
                k: int = 3) -> dict:
    """Top-k drivers of a single prediction.

    Returns ``{"method": ..., "drivers": [(feature, signed_contribution), ...]}``
    ranked by absolute contribution. With SHAP the contributions are signed
    (push the prediction up/down); in the fallback they are global-importance
    weights (always non-negative) and clearly labelled.
    """
    x_row = np.asarray(x_row, dtype=float).reshape(1, -1)

    if shap_available():
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            values = np.asarray(explainer.shap_values(x_row)).reshape(-1)
            pairs = sorted(
                zip(feature_names, values.tolist()),
                key=lambda p: abs(p[1]), reverse=True,
            )
            return {"method": "shap", "drivers": pairs[:k]}
        except Exception as exc:  # pragma: no cover - defensive
            log.info("shap_failed_fallback", extra={"kv": {"error": str(exc)}})

    imp = global_importance(model, feature_names)
    pairs = sorted(imp.items(), key=lambda p: p[1], reverse=True)
    return {"method": "feature_importance (global)", "drivers": pairs[:k]}
