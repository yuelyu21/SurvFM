"""Metric helpers for SurvFM-RMST reproducibility."""

from __future__ import annotations

import numpy as np

from .pseudo_rmst import km_survival_at, realized_restricted_time


def _finite_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not np.any(mask):
        raise ValueError("No finite y_true/y_pred pairs are available.")
    return y_true_arr[mask], y_pred_arr[mask]


def rmse(y_true, y_pred) -> float:
    """Root mean squared error over finite pairs."""

    y_true_arr, y_pred_arr = _finite_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def normalized_rmse(y_true, y_pred, tau: float) -> float:
    """RMSE divided by the restriction horizon ``tau``."""

    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite value.")
    return rmse(y_true, y_pred) / float(tau)


def predicted_rmst_to_risk(predicted_rmst) -> np.ndarray:
    """Orient predicted RMST for concordance metrics.

    Larger RMST means longer predicted event-free time, so risk is ``-RMST``.
    """

    return -np.asarray(predicted_rmst, dtype=float)


def uno_c_from_predicted_rmst(train_time, train_event, test_time, test_event, predicted_rmst) -> float:
    """Compute Uno's C from predicted RMST using scikit-survival.

    The dependency is imported lazily so the lightweight package remains
    importable in environments without scikit-survival.
    """

    try:
        from sksurv.metrics import concordance_index_ipcw
        from sksurv.util import Surv
    except Exception as exc:  # pragma: no cover - dependency dependent
        raise ImportError("uno_c_from_predicted_rmst requires scikit-survival.") from exc

    y_train = Surv.from_arrays(np.asarray(train_event).astype(bool), np.asarray(train_time, dtype=float))
    y_test = Surv.from_arrays(np.asarray(test_event).astype(bool), np.asarray(test_time, dtype=float))
    risk = predicted_rmst_to_risk(predicted_rmst)
    return float(concordance_index_ipcw(y_train, y_test, risk)[0])


def ipcw_rmst_rmse(
    train_time,
    train_event,
    test_time,
    test_event,
    predicted_rmst,
    tau: float,
    *,
    support_threshold: float = 1e-6,
) -> dict[str, float]:
    """Compute test-set-denominator IPCW restricted-time prediction error.

    The observable restricted outcome is ``min(T, tau)``. It is known when an
    event occurs on or before ``tau`` or when observed follow-up reaches
    ``tau``. Censored-before-horizon rows contribute zero to the Horvitz-
    Thompson numerator but remain in the full test-set denominator. Event rows
    use ``G(Y-)`` and horizon-reaching rows use ``G(tau)``, where ``G`` is the
    training-fold Kaplan-Meier estimate of censoring survival.

    A block is returned as non-evaluable (``rmse`` and ``normalized_rmse`` are
    NaN) if any otherwise observable row has an unsupported censoring weight or
    nonfinite prediction. Weights are never silently floored or truncated.
    """

    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite value.")
    if not np.isfinite(support_threshold) or support_threshold <= 0:
        raise ValueError("support_threshold must be a positive finite value.")

    train_time = np.asarray(train_time, dtype=float).reshape(-1)
    train_event = np.asarray(train_event).astype(int).reshape(-1)
    test_time = np.asarray(test_time, dtype=float).reshape(-1)
    test_event = np.asarray(test_event).astype(int).reshape(-1)
    pred = np.asarray(predicted_rmst, dtype=float).reshape(-1)

    if len(train_time) != len(train_event):
        raise ValueError("train_time and train_event must have the same length.")
    if len(test_time) != len(test_event) or len(test_time) != len(pred):
        raise ValueError("test_time, test_event and predicted_rmst must have the same length.")
    if len(test_time) == 0:
        raise ValueError("The test set must contain at least one observation.")

    event_observed = (test_event == 1) & (test_time <= tau)
    # Keep the branches mutually exclusive so an event exactly at tau is not
    # counted twice.
    horizon_observed = (~event_observed) & (test_time >= tau)
    evaluable = event_observed | horizon_observed
    censor_event = 1 - train_event
    required_g = np.full(len(test_time), np.nan, dtype=float)
    for idx in np.flatnonzero(event_observed):
        required_g[idx] = km_survival_at(
            train_time,
            censor_event,
            test_time[idx],
            left_limit=True,
        )
    g_tau = km_survival_at(train_time, censor_event, float(tau))
    required_g[horizon_observed] = g_tau

    unstable = evaluable & (~np.isfinite(required_g) | (required_g <= support_threshold))
    nonfinite_prediction = evaluable & ~np.isfinite(pred)
    valid = evaluable & ~unstable & ~nonfinite_prediction

    weights = np.zeros(len(test_time), dtype=float)
    weights[valid] = 1.0 / required_g[valid]
    positive_weights = weights[valid]
    weight_sum = float(np.sum(positive_weights))
    weight_sq_sum = float(np.sum(positive_weights**2))
    effective_sample_size = (
        weight_sum**2 / weight_sq_sum if weight_sq_sum > 0 else float("nan")
    )

    evaluation_success = bool(np.any(evaluable) and not np.any(unstable) and not np.any(nonfinite_prediction))
    if evaluation_success:
        restricted_time = realized_restricted_time(test_time, tau)
        normalized_err2 = ((restricted_time - pred) / float(tau)) ** 2
        weighted_normalized_sse = float(np.sum(weights * normalized_err2))
        normalized_rmse_value = float(np.sqrt(weighted_normalized_sse / len(test_time)))
        rmse_value = normalized_rmse_value * float(tau)
    else:
        weighted_normalized_sse = float("nan")
        normalized_rmse_value = float("nan")
        rmse_value = float("nan")

    finite_required_g = required_g[valid]
    return {
        "rmse": rmse_value,
        "normalized_rmse": normalized_rmse_value,
        "n_test": float(len(test_time)),
        "n_evaluable": float(np.sum(evaluable)),
        "n_valid_ipcw": float(np.sum(valid)),
        "n_unevaluable": float(np.sum(~evaluable)),
        "n_unstable_ipcw": float(np.sum(unstable)),
        "n_nonfinite_prediction": float(np.sum(nonfinite_prediction)),
        "evaluation_success": float(evaluation_success),
        "weighted_normalized_squared_error_sum": weighted_normalized_sse,
        "mean_ipcw_weight": float(np.mean(positive_weights)) if len(positive_weights) else float("nan"),
        "ipcw_weight_effective_sample_size": float(effective_sample_size),
        "g_tau": float(g_tau),
        "min_required_g": float(np.min(finite_required_g)) if len(finite_required_g) else float("nan"),
    }
