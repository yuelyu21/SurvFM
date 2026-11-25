"""Experimental OOF RMST-to-Cox survival-curve adapter.

This is an additional model, not a survival-curve output of native SurvFM.
The Cox mapping assumes proportional hazards in a standardized negative RMST
score. Its out-of-fold training scores and full-refit prediction scores may
have different distributions. Neither numerical validity nor this example
establishes probability calibration.
"""

from __future__ import annotations

import numpy as np


def _rows(X, indices):
    return X.iloc[indices] if hasattr(X, "iloc") else np.asarray(X)[indices]


class OOFCoxSurvivalExtension:
    """Fit a training-only Cox mapping on out-of-fold RMST predictions.

    ``model_factory(tau)`` must return a fresh, unfitted SurvFM regressor,
    including fresh preprocessing and a fresh backbone, on every call. Each
    inner model constructs its own pseudo-RMST targets using only its training
    fold. ``tau`` must be chosen from outer training data or fixed in advance;
    it stays fixed across inner folds. No test outcomes are accepted by this
    adapter.

    With three inner folds this uses four backbone fits: three OOF fits plus
    one full-training fit. ``predict_rmst`` retains the full-training model's
    native RMST prediction; integrating a mapped curve need not reproduce it.

    Parameters
    ----------
    model_factory : callable
        Function accepting one float ``tau`` and returning a new estimator
        with ``fit(X, time, event)`` and ``predict(X)`` methods.
    tau : float
        Positive horizon in the same units as the observed times.
    n_splits : int, default=3
        Number of stratified inner folds.
    random_state : int, default=20261006
        Seed for the inner split, not a substitute for backbone seeds.
    alpha : float, default=1e-4
        Fixed positive ridge penalty for the Cox mapping.
    """

    def __init__(
        self, model_factory, tau, n_splits=3, random_state=20261006, alpha=1e-4
    ):
        self.model_factory = model_factory
        self.tau = tau
        self.n_splits = n_splits
        self.random_state = random_state
        self.alpha = alpha
        self._fitted = False

    def fit(self, X, time, event):
        """Fit using outer-training covariates and outcomes only."""
        self._fitted = False
        time = np.asarray(time, dtype=float)
        event = np.asarray(event)
        if time.ndim != 1 or event.ndim != 1:
            raise ValueError("time and event must be one-dimensional.")
        if not len(time) or len(time) != len(event) or len(X) != len(time):
            raise ValueError("X, time and event must have equal, nonzero row counts.")
        if np.ndim(X) != 2:
            raise ValueError("X must be a two-dimensional feature table.")
        if not np.isfinite(time).all() or np.any(time < 0):
            raise ValueError("time must contain finite nonnegative values.")
        if not np.isin(event, [0, 1, False, True]).all():
            raise ValueError("event must be binary: 1=event, 0=censored.")
        event = event.astype(bool)
        tau = float(self.tau)
        if not np.isfinite(tau) or tau <= 0 or tau > time.max():
            raise ValueError("tau must be positive and within training follow-up.")
        if (
            isinstance(self.n_splits, (bool, np.bool_))
            or not isinstance(self.n_splits, (int, np.integer))
            or self.n_splits < 2
        ):
            raise ValueError("n_splits must be an integer of at least two.")
        counts = np.bincount(event.astype(int), minlength=2)
        if counts[1] < self.n_splits or (0 < counts[0] < self.n_splits):
            raise ValueError("Each present event class needs at least n_splits rows.")
        if not callable(self.model_factory):
            raise ValueError("model_factory must return a fresh model for each call.")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive.")

        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv

        splits = list(
            StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            ).split(np.zeros(len(time)), event)
        )
        # Fail explicitly instead of extrapolating a fold's KM tail to tau.
        if any(tau > time[train].max() for train, _ in splits):
            raise ValueError("tau exceeds follow-up support in an inner training fold.")

        self.tau_ = tau
        self.n_base_fits_ = 0
        oof = np.full(len(time), np.nan)
        fold_ids = np.full(len(time), -1, dtype=int)
        for fold_id, (train, valid) in enumerate(splits):
            model = self.model_factory(tau)
            model.fit(_rows(X, train), time[train], event[train])
            self.n_base_fits_ += 1
            oof[valid] = self._rmst_prediction(model, _rows(X, valid))
            fold_ids[valid] = fold_id
            del model

        score = -oof[:, None] / tau
        if not np.isfinite(score).all() or np.std(score) <= 1e-12:
            raise ValueError("OOF RMST scores are nonfinite or constant; mapping is undefined.")
        scaler = StandardScaler().fit(score)
        mapping = CoxPHSurvivalAnalysis(alpha=self.alpha)
        mapping.fit(scaler.transform(score), Surv.from_arrays(event, time))

        full_model = self.model_factory(tau)
        full_model.fit(X, time, event)
        self.n_base_fits_ += 1
        self.full_model_ = full_model
        self.scaler_ = scaler
        self.mapping_ = mapping
        self.oof_rmst_ = oof
        self.fold_ids_ = fold_ids
        self._fitted = True
        return self

    def _rmst_prediction(self, model, X):
        if hasattr(model, "tau_") and float(model.tau_) != self.tau_:
            raise ValueError("The model factory must preserve the supplied tau.")
        pred = np.asarray(model.predict(X), dtype=float)
        if pred.shape != (len(X),) or not np.isfinite(pred).all():
            raise ValueError("The base model must return one finite RMST per row.")
        if np.any((pred < 0) | (pred > self.tau_)):
            raise ValueError("This example requires RMST predictions within [0, tau].")
        return pred

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit successfully before prediction.")

    def predict_rmst(self, X):
        """Return native RMST, without replacing it by a mapped-curve integral."""
        self._check_fitted()
        return self._rmst_prediction(self.full_model_, X)

    def predict_survival(self, X, times):
        """Return probabilities with shape (n_patients, n_times), without extrapolation."""
        self._check_fitted()
        times = np.asarray(times, dtype=float)
        if (
            times.ndim != 1
            or not len(times)
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or times[0] < 0
            or times[-1] > self.tau_
        ):
            raise ValueError("times must be finite, strictly increasing and within [0, tau].")
        rmst = self.predict_rmst(X)
        score = self.scaler_.transform(-rmst[:, None] / self.tau_)
        functions = self.mapping_.predict_survival_function(score)
        survival = np.asarray([fn(times) for fn in functions])
        # sksurv's StepFunction extends its first value to earlier query
        # times. For survival, the Cox baseline hazard is zero before the
        # first observed training time, so use S(t)=1 on that interval.
        survival[:, times < self.mapping_.unique_times_[0]] = 1.0
        if (
            survival.shape != (len(X), len(times))
            or not np.isfinite(survival).all()
            or np.any((survival < 0) | (survival > 1))
            or np.any(np.diff(survival, axis=1) > 1e-12)
        ):
            raise RuntimeError("The Cox mapping returned invalid survival probabilities.")
        return survival
