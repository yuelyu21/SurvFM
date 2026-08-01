"""RMST and pseudo-observation utilities for SurvFM.

These functions intentionally avoid heavy survival-package dependencies so that
the core target-construction logic remains easy to audit. Event indicators use
``1`` for observed events and ``0`` for censoring.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd


def _as_1d_float(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _as_1d_event(values, name: str = "event") -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.isin(arr, [0, 1, False, True]).all():
        raise ValueError(f"{name} must be binary with 1=event and 0=censored.")
    return arr.astype(int)


def _validate_time_event(time, event) -> tuple[np.ndarray, np.ndarray]:
    time_arr = _as_1d_float(time, "time")
    event_arr = _as_1d_event(event)
    if len(time_arr) != len(event_arr):
        raise ValueError("time and event must have the same length.")
    if np.any(time_arr < 0):
        raise ValueError("time values must be non-negative.")
    return time_arr, event_arr


def _extract_time_event(time_or_frame, event=None) -> tuple[np.ndarray, np.ndarray]:
    if event is None:
        if not isinstance(time_or_frame, pd.DataFrame):
            raise TypeError("Provide either (time, event) arrays or a DataFrame with time/event columns.")
        if "time" not in time_or_frame or "event" not in time_or_frame:
            raise ValueError("DataFrame input must contain 'time' and 'event' columns.")
        return _validate_time_event(time_or_frame["time"].to_numpy(), time_or_frame["event"].to_numpy())
    return _validate_time_event(time_or_frame, event)


def _validated_fast_exact_inputs(time, event, tau: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate inputs while preserving the submitted ``event == 1`` rule."""

    time_arr = np.asarray(time, dtype=np.float64)
    event_arr = np.asarray(event)
    if time_arr.ndim != 1 or event_arr.ndim != 1:
        raise ValueError("time and event must be one-dimensional")
    if len(time_arr) != len(event_arr):
        raise ValueError("time and event must have equal length")
    if len(time_arr) == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(time_arr)):
        raise ValueError("time contains non-finite values")
    if np.any(time_arr < 0):
        raise ValueError("time must be non-negative")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")

    event_float = np.asarray(event_arr, dtype=np.float64)
    if not np.all(np.isfinite(event_float)):
        raise ValueError("event contains non-finite values")

    # The submitted benchmark identifies an observed event with event == 1.
    # Keep additional finite endpoint codes as non-events for this target.
    event_indicator = (event_float == 1.0).astype(np.int8)
    return time_arr, event_indicator, float(tau)


def jackknife_pseudo_rmst_fast_exact(
    time,
    event,
    tau: float,
    *,
    clip: bool = True,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute exact, memory-bounded leave-one-out pseudo-RMST targets.

    This evaluates the same full jackknife definition used by the submitted
    benchmark, but computes the full event-time grid once and evaluates
    delete-one curves in bounded patient batches. It is not an influence-
    function, subsampled, grouped, split, or cross-fitted approximation.

    The submitted Kaplan-Meier convention keeps observations with
    ``time >= event_time`` in the risk set, including censoring tied with an
    event. Events at ``tau`` do not alter the integral over ``[0, tau]``.
    Final clipping to ``[0, tau]`` defaults to the submitted behavior.
    """

    time_arr, event_arr, tau = _validated_fast_exact_inputs(time, event, tau)
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    n = len(time_arr)
    all_event_times, all_event_counts = np.unique(
        time_arr[event_arr == 1], return_counts=True
    )
    before_tau = all_event_times < tau
    event_times = all_event_times[before_tau].astype(np.float64, copy=False)
    event_counts = all_event_counts[before_tau].astype(np.float64, copy=False)

    if len(event_times) == 0:
        raw = np.full(n, tau, dtype=np.float64)
        return np.clip(raw, 0.0, tau) if clip else raw

    risk_counts = np.array(
        [np.count_nonzero(time_arr >= event_time) for event_time in event_times],
        dtype=np.float64,
    )
    survival_full = np.cumprod(1.0 - event_counts / risk_counts)
    interval_widths = np.diff(np.concatenate((event_times, [tau])))
    rmst_full = event_times[0] + float(interval_widths @ survival_full)

    raw = np.empty(n, dtype=np.float64)
    event_times_row = event_times[None, :]
    risk_counts_row = risk_counts[None, :]
    event_counts_row = event_counts[None, :]

    for start in range(0, n, int(batch_size)):
        stop = min(start + int(batch_size), n)
        batch_time = time_arr[start:stop, None]
        batch_event = event_arr[start:stop, None]

        deleted_risk = risk_counts_row - (batch_time >= event_times_row)
        deleted_events = event_counts_row - (
            (batch_event == 1) & (batch_time == event_times_row)
        )

        # A full-grid time can disappear after deletion. Its neutral factor is 1.
        hazard = np.zeros_like(deleted_risk, dtype=np.float64)
        np.divide(deleted_events, deleted_risk, out=hazard, where=deleted_risk > 0)
        survival_deleted = np.cumprod(1.0 - hazard, axis=1)
        rmst_deleted = event_times[0] + survival_deleted @ interval_widths
        raw[start:stop] = n * rmst_full - (n - 1) * rmst_deleted

    return np.clip(raw, 0.0, tau) if clip else raw


def kaplan_meier_curve(time, event) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the Kaplan-Meier survival step function.

    Returns arrays of event times and survival probabilities with an initial
    point at time 0 and survival 1.0. Event indicators use 1 for observed events.
    """

    time_arr, event_arr = _validate_time_event(time, event)
    order = np.argsort(time_arr, kind="mergesort")
    time_sorted = time_arr[order]
    event_sorted = event_arr[order]
    event_times = np.unique(time_sorted[event_sorted == 1])
    if len(event_times) == 0:
        return np.array([0.0], dtype=float), np.array([1.0], dtype=float)

    surv_values = []
    survival = 1.0
    for t in event_times:
        at_risk = np.sum(time_sorted >= t)
        if at_risk <= 0:
            continue
        events_at_t = np.sum((time_sorted == t) & (event_sorted == 1))
        survival *= 1.0 - events_at_t / at_risk
        surv_values.append(survival)

    return (
        np.concatenate([[0.0], event_times.astype(float)]),
        np.concatenate([[1.0], np.asarray(surv_values, dtype=float)]),
    )


def km_survival_at(time, event, t_eval: float, *, left_limit: bool = False) -> float:
    """Evaluate the Kaplan-Meier step function at ``t_eval``.

    Set ``left_limit=True`` to evaluate immediately before ``t_eval``. This is
    required for IPCW event contributions, which use the censoring survival
    probability :math:`G(t-)` rather than the value after censoring events tied
    at ``t``.
    """

    if not np.isfinite(t_eval):
        raise ValueError("t_eval must be finite.")
    times, survival = kaplan_meier_curve(time, event)
    side = "left" if left_limit else "right"
    idx = np.searchsorted(times, float(t_eval), side=side) - 1
    idx = max(idx, 0)
    return float(survival[idx])


def km_rmst(time, event, tau: float) -> float:
    """Compute Kaplan-Meier restricted mean survival time up to ``tau``."""

    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite value.")
    time_arr, event_arr = _validate_time_event(time, event)
    times, survival = kaplan_meier_curve(time_arr, event_arr)
    grid = np.unique(np.concatenate([times[times < tau], [float(tau)]]))
    if grid[0] != 0.0:
        grid = np.insert(grid, 0, 0.0)

    rmst = 0.0
    for left, right in zip(grid[:-1], grid[1:]):
        if right <= left:
            continue
        idx = np.searchsorted(times, left, side="right") - 1
        idx = max(idx, 0)
        rmst += (right - left) * survival[idx]
    return float(np.clip(rmst, 0.0, tau))


def jackknife_pseudo_rmst(time_or_frame, event=None, tau: float | None = None, *, clip: bool = True) -> np.ndarray:
    """Construct jackknife pseudo-observation RMST targets.

    Parameters
    ----------
    time_or_frame:
        Either a sequence of observed times or a DataFrame with ``time`` and
        ``event`` columns.
    event:
        Binary event indicators when ``time_or_frame`` is an array.
    tau:
        Restriction horizon. If the first argument is a DataFrame, ``tau`` must
        be passed by keyword or as the second positional argument using
        ``jackknife_pseudo_rmst(df, tau=...)``.
    clip:
        If true, clip pseudo-targets to ``[0, tau]``. This matches the manuscript
        benchmark implementation.
    """

    if tau is None:
        if event is None:
            raise ValueError("tau must be provided.")
        if np.isscalar(event):
            tau = float(event)
            event = None
        else:
            raise ValueError("tau must be provided when event indicators are passed.")

    time_arr, event_arr = _extract_time_event(time_or_frame, event)
    tau = float(tau)
    n = len(time_arr)
    if n < 2:
        raise ValueError("At least two observations are required for jackknife pseudo-RMST.")

    rmst_all = km_rmst(time_arr, event_arr, tau)
    pseudo = np.empty(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rmst_minus_i = km_rmst(time_arr[mask], event_arr[mask], tau)
        pseudo[i] = n * rmst_all - (n - 1) * rmst_minus_i

    if clip:
        pseudo = np.clip(pseudo, 0.0, tau)
    return pseudo


def realized_restricted_time(true_or_observed_time, tau: float) -> np.ndarray:
    """Return ``min(time, tau)`` for realized restricted-time evaluation."""

    time_arr = _as_1d_float(true_or_observed_time, "time")
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite value.")
    return np.minimum(time_arr, float(tau))


def survival_functions_to_rmst(
    survival_functions: Iterable[Callable[[float], float]],
    tau: float,
    *,
    n_grid: int = 400,
) -> np.ndarray:
    """Integrate predicted survival functions into RMST predictions.

    This helper mirrors the benchmark convention for scikit-survival style
    survival-function callables.
    """

    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be a positive finite value.")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2.")

    grid = np.linspace(0.0, float(tau), int(n_grid))
    out = []
    for fn in survival_functions:
        surv = np.asarray([fn(float(t)) for t in grid], dtype=float)
        if not np.all(np.isfinite(surv)):
            raise ValueError("A survival function returned non-finite values.")
        out.append(float(np.trapezoid(surv, x=grid)))
    return np.asarray(out, dtype=float)
