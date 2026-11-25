"""Experimental survival curves from several separately fitted RMST models.

This is an extension of SurvFM, not its native single-horizon prediction API.
It assumes S(0)=1 and a piecewise-linear survival curve on a fixed grid. A
valid decreasing curve does not, by itself, demonstrate probability calibration.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, nnls


class ReconstructionFailure(RuntimeError):
    """The constrained reconstruction did not pass its numerical checks."""


def _grid_array(grid):
    grid = np.asarray(grid, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 2
        or not np.isfinite(grid).all()
        or grid[0] != 0.0
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("grid must be finite, start at zero, and increase strictly.")
    return grid


def integral_matrix(grid):
    """Map survival probabilities at knots to their trapezoidal integrals."""
    grid = _grid_array(grid)
    matrix = np.zeros((len(grid) - 1, len(grid)))
    for j, width in enumerate(np.diff(grid)):
        matrix[j:, j] += width / 2.0
        matrix[j:, j + 1] += width / 2.0
    return matrix


def reconstruct(rmst, grid, maxiter=1000):
    """Fit a monotone curve to RMST integrals; return curve, integrals, diagnostics.

    The objective is equal-weight squared integral error normalized by the
    maximum horizon. Incompatible RMST predictions need not be matched exactly.
    An unsuccessful solver or numerical check raises; no fallback is substituted.
    """
    grid = _grid_array(grid)
    rmst = np.asarray(rmst, dtype=float)
    if rmst.shape != (len(grid) - 1,) or not np.isfinite(rmst).all():
        raise ValueError("rmst must contain one finite value per positive grid point.")
    if isinstance(maxiter, (bool, np.bool_)) or not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")

    full = integral_matrix(grid) / grid[-1]
    a, b = full[:, 1:], full[:, 0] - rmst / grid[-1]
    k = len(rmst)
    # C s + d >= 0 encodes 1 >= s1 >= ... >= sk >= 0; s0 is fixed to one.
    c = np.zeros((k + 1, k))
    d = np.zeros(k + 1)
    c[0, 0], d[0] = -1.0, 1.0
    for j in range(1, k):
        c[j, j - 1], c[j, j] = 1.0, -1.0
    c[k, -1] = 1.0

    def objective(s):
        return float(np.sum((a @ s + b) ** 2))

    def jacobian(s):
        return 2.0 * a.T @ (a @ s + b)

    result = minimize(
        objective,
        np.linspace(0.9, 0.1, k),
        jac=jacobian,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda s: c @ s + d, "jac": lambda s: c}],
        options={"ftol": 1e-13, "maxiter": int(maxiter)},
    )
    if not result.success:
        raise ReconstructionFailure(f"SLSQP status {result.status}: {result.message}")
    s = np.asarray(result.x, dtype=float)
    if not np.isfinite(s).all() or not np.isfinite(result.fun):
        raise ReconstructionFailure("The solver returned non-finite values.")
    slack = c @ s + d
    active = slack < 1e-7
    residual = np.linalg.norm(jacobian(s))
    if active.any():
        _, residual = nnls(c[active].T, jacobian(s))
    if slack.min() < -1e-9 or not np.isfinite(residual) or residual > 2e-6:
        raise ReconstructionFailure(
            f"Constraint/KKT verification failed: slack={slack.min()}, residual={residual}"
        )

    # Only remove sub-tolerance numerical boundary errors after the checks pass.
    survival = np.minimum.accumulate(np.clip(np.r_[1.0, s], 0.0, 1.0))
    integrals = np.cumsum(np.diff(grid) * (survival[:-1] + survival[1:]) / 2.0)
    checked_objective = float(np.sum(((integrals - rmst) / grid[-1]) ** 2))
    if not np.isfinite(checked_objective) or abs(checked_objective - result.fun) > 1e-10:
        raise ReconstructionFailure("Independent integral/objective check failed.")
    diagnostics = {
        "SolverSuccess": True,
        "Status": int(result.status),
        "Iterations": int(result.nit),
        "Objective": checked_objective,
        "KKTResidual": float(residual),
        "MinimumSlack": float(slack.min()),
        "NumericalCleanupMax": float(np.max(np.abs(s - survival[1:]))),
    }
    return survival, integrals, diagnostics


def _query_times(times, grid):
    times = np.asarray(times, dtype=float)
    if (
        times.ndim != 1
        or times.size == 0
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("times must be a non-empty finite increasing one-dimensional array.")
    if times[0] < grid[0] or times[-1] > grid[-1]:
        raise ValueError("Extrapolation outside the reconstruction horizon is forbidden.")
    return times


def interpolate(grid, survival, times):
    """Interpolate one survival curve at one or more in-range times."""
    grid = _grid_array(grid)
    survival = np.asarray(survival, dtype=float)
    if (
        survival.shape != grid.shape
        or not np.isfinite(survival).all()
        or np.any((survival < 0.0) | (survival > 1.0))
        or np.any(np.diff(survival) > 0.0)
        or survival[0] != 1.0
    ):
        raise ValueError("survival must match grid, start at one, and decrease within [0, 1].")
    return np.interp(_query_times(times, grid), grid, survival)


class MultiHorizonSurvivalExtension:
    """Fit one independent RMST model per positive horizon, then reconstruct.

    ``model_factory(tau)`` must return a fresh SurvFMRMSTRegressor (or another
    object with the same ``fit(X, time, event)`` and ``predict(X)`` interface).
    It must construct training targets and fit preprocessing using training
    data only. This adapter projects each native prediction to [0, its horizon].
    Eight horizons require eight base-model fits; there is no fitted Cox mapper.
    """

    def __init__(self, model_factory, tau, n_horizons=8):
        if not callable(model_factory):
            raise TypeError("model_factory must be callable.")
        if np.ndim(tau) != 0 or not np.isfinite(float(tau)) or float(tau) <= 0:
            raise ValueError("tau must be a positive finite scalar.")
        if (
            isinstance(n_horizons, (bool, np.bool_))
            or not isinstance(n_horizons, (int, np.integer))
            or n_horizons < 1
        ):
            raise ValueError("n_horizons must be a positive integer.")
        self.model_factory = model_factory
        self.tau = float(tau)
        self.n_horizons = int(n_horizons)

    @staticmethod
    def _check_features(X):
        if not hasattr(X, "shape") or len(X.shape) != 2 or len(X) == 0 or X.shape[1] == 0:
            raise ValueError("X must be a non-empty two-dimensional feature table.")

    def fit(self, X, time, event):
        self.__dict__.pop("models_", None)
        self.n_base_fits_ = 0
        self._check_features(X)
        time = np.asarray(time, dtype=float)
        event = np.asarray(event)
        if time.shape != (len(X),) or not np.isfinite(time).all() or np.any(time < 0):
            raise ValueError("time must contain one finite nonnegative value per training row.")
        if event.shape != time.shape or not np.isin(event, [0, 1, False, True]).all():
            raise ValueError("event must be binary with one value per training row.")
        if np.any((time == 0.0) & (event == 1)):
            raise ValueError("Observed events at time zero conflict with the S(0)=1 reconstruction assumption.")
        if self.tau > float(np.max(time)):
            raise ValueError("tau must not exceed the maximum observed training time.")
        self.grid_ = _grid_array(np.linspace(0.0, self.tau, self.n_horizons + 1))
        models = []
        for horizon in self.grid_[1:]:
            model = self.model_factory(float(horizon))
            if any(model is previous for previous in models):
                raise ValueError("model_factory must return a fresh model for every horizon.")
            if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
                raise TypeError("Every model must implement fit and predict.")
            # Detect fresh wrappers that nonetheless share the same custom estimator.
            estimator = getattr(model, "estimator", None)
            if estimator is not None and any(estimator is getattr(previous, "estimator", None) for previous in models):
                raise ValueError("model_factory must also create a fresh underlying estimator.")
            model.fit(X, time, event)
            self.n_base_fits_ += 1
            if hasattr(model, "tau_") and model.tau_ != float(horizon):
                raise ValueError("A fitted model did not use its requested numeric horizon.")
            models.append(model)
        self.models_ = models
        return self

    def _check_prediction(self, X):
        if not hasattr(self, "models_"):
            raise RuntimeError("Call fit successfully before prediction.")
        self._check_features(X)

    @staticmethod
    def _predict_model(model, X, horizon):
        prediction = np.asarray(model.predict(X), dtype=float)
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError("Each model must return one finite RMST prediction per row.")
        return np.clip(prediction, 0.0, horizon)

    def predict_rmst(self, X):
        """Return the native clipped RMST at tau, not the reconstructed integral."""
        self._check_prediction(X)
        return self._predict_model(self.models_[-1], X, self.grid_[-1])

    def predict_rmst_by_horizon(self, X):
        """Return native clipped RMST columns at the positive grid points."""
        self._check_prediction(X)
        return np.column_stack([
            self._predict_model(model, X, horizon)
            for model, horizon in zip(self.models_, self.grid_[1:])
        ])

    def reconstruct_predictions(self, X):
        """Return native RMST, curve knots, curve integrals and per-row diagnostics."""
        rmst = self.predict_rmst_by_horizon(X)
        rows = [reconstruct(prediction, self.grid_) for prediction in rmst]
        return {
            "rmst": rmst,
            "survival": np.stack([row[0] for row in rows]),
            "integrated_rmst": np.stack([row[1] for row in rows]),
            "diagnostics": [row[2] for row in rows],
        }

    def predict_survival(self, X, times):
        """Return (n_patients, n_times) probabilities; never extrapolate beyond tau."""
        self._check_prediction(X)
        times = _query_times(times, self.grid_)
        curves = self.reconstruct_predictions(X)["survival"]
        return np.stack([interpolate(self.grid_, curve, times) for curve in curves])
