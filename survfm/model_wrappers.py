"""User-facing estimator for the SurvFM-RMST target interface."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .backbones import create_backbone, get_backbone_info
from .horizons import HorizonSelection, select_rmst_horizon
from .preprocessing import build_static_tabular_preprocessor
from .pseudo_rmst import jackknife_pseudo_rmst_fast_exact


@dataclass
class PseudoRMSTRegressor:
    """Regress exact jackknife pseudo-RMST targets on baseline covariates.

    Parameters
    ----------
    estimator:
        Custom object implementing ``fit(X, y)`` and ``predict(X)``. When
        supplied, it takes precedence over ``backbone``.
    backbone:
        Registered backbone name. Defaults to ``"tabpfn"``.
    backbone_kwargs:
        Constructor arguments passed to the named backbone.
    tau:
        Explicit RMST restriction horizon. If omitted, a training-fold event-
        time quantile is selected according to ``tau_quantile``.
    preprocess:
        ``"auto"`` fits mixed-data preprocessing for pandas DataFrames and
        passes numeric arrays through. Use ``None`` to disable preprocessing or
        pass a transformer with ``fit_transform`` and ``transform``.
    """

    estimator: object | None = None
    backbone: str = "tabpfn"
    backbone_kwargs: dict = field(default_factory=dict)
    tau: float | None = None
    tau_quantile: float = 0.8
    min_events_for_tau: int = 5
    preprocess: str | object | None = "auto"
    clip_targets: bool = True
    clip_predictions: bool = True
    pseudo_batch_size: int = 256

    def _fit_preprocessor(self, X):
        self.feature_columns_ = None
        if self.preprocess is None:
            return self._numeric_array(X)
        if isinstance(self.preprocess, str) and self.preprocess == "auto":
            if isinstance(X, pd.DataFrame):
                self.feature_columns_ = list(X.columns)
                self.preprocessor_ = build_static_tabular_preprocessor(
                    X, list(X.columns)
                )
                return self._numeric_array(self.preprocessor_.fit_transform(X))
            return self._numeric_array(X)
        if isinstance(self.preprocess, str):
            raise ValueError("preprocess must be 'auto', None, or a transformer.")
        self.preprocessor_ = self.preprocess
        return self._numeric_array(self.preprocessor_.fit_transform(X))

    @staticmethod
    def _numeric_array(X) -> np.ndarray:
        array = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        try:
            array = np.asarray(array, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Features must be numeric, or pass a pandas DataFrame with preprocess='auto'."
            ) from exc
        if array.ndim != 2 or len(array) == 0:
            raise ValueError("X must be a non-empty two-dimensional table.")
        if not np.all(np.isfinite(array)):
            raise ValueError("Features contain non-finite values after preprocessing.")
        return array

    def _transform(self, X) -> np.ndarray:
        if self.feature_columns_ is not None:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("This fitted model expects a pandas DataFrame.")
            missing = [column for column in self.feature_columns_ if column not in X.columns]
            if missing:
                raise ValueError(f"Prediction data are missing columns: {missing}.")
            X = X[self.feature_columns_]
        if hasattr(self, "preprocessor_"):
            X = self.preprocessor_.transform(X)
        return self._numeric_array(X)

    def fit(self, X, time, event):
        time_arr = np.asarray(time, dtype=float).reshape(-1)
        event_arr = np.asarray(event).reshape(-1)
        if len(time_arr) != len(event_arr) or len(time_arr) != len(X):
            raise ValueError("X, time and event must contain the same number of rows.")
        if not np.isin(event_arr, [0, 1, False, True]).all():
            raise ValueError("event must be binary with 1=event and 0=censored.")

        if self.tau is None:
            selection = select_rmst_horizon(
                time_arr,
                event_arr,
                quantile=self.tau_quantile,
                min_events=self.min_events_for_tau,
            )
            self.horizon_selection_ = selection
            self.tau_ = selection.tau
        else:
            self.tau_ = float(self.tau)
            if not np.isfinite(self.tau_) or self.tau_ <= 0:
                raise ValueError("tau must be positive and finite.")
            self.horizon_selection_ = None

        X_model = self._fit_preprocessor(X)
        self.pseudo_targets_ = jackknife_pseudo_rmst_fast_exact(
            time_arr,
            event_arr,
            tau=self.tau_,
            clip=self.clip_targets,
            batch_size=self.pseudo_batch_size,
        )
        if self.estimator is not None:
            self.estimator_ = self.estimator
            self.backbone_name_ = "custom"
        else:
            self.backbone_name_ = str(self.backbone).strip().lower()
            self.estimator_ = create_backbone(self.backbone_name_, **self.backbone_kwargs)
        self.estimator_.fit(X_model, self.pseudo_targets_)
        self.n_features_in_ = int(X_model.shape[1])
        self.n_training_observations_ = int(len(X_model))
        return self

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "estimator_"):
            raise RuntimeError("Call fit before predict.")
        pred = np.asarray(self.estimator_.predict(self._transform(X)), dtype=float).reshape(-1)
        if not np.all(np.isfinite(pred)):
            raise RuntimeError("The backbone returned non-finite predictions.")
        if self.clip_predictions:
            pred = np.clip(pred, 0.0, self.tau_)
        return pred

    def predict_risk(self, X) -> np.ndarray:
        """Return risk-oriented scores; larger scores indicate shorter RMST."""

        return -self.predict(X)

    def get_metadata(self) -> dict:
        """Return serializable fit metadata without credentials or patient data."""

        if not hasattr(self, "estimator_"):
            raise RuntimeError("Call fit before requesting metadata.")
        horizon = (
            self.horizon_selection_.to_dict()
            if isinstance(self.horizon_selection_, HorizonSelection)
            else None
        )
        info = (
            get_backbone_info(self.backbone_name_).to_dict()
            if self.backbone_name_ != "custom"
            else {"name": "custom", "display_name": type(self.estimator_).__name__}
        )
        return {
            "format_version": 1,
            "backbone": info,
            "tau": float(self.tau_),
            "horizon_selection": horizon,
            "clip_targets": bool(self.clip_targets),
            "clip_predictions": bool(self.clip_predictions),
            "n_training_observations": int(self.n_training_observations_),
            "n_processed_features": int(self.n_features_in_),
            "feature_columns": (
                [str(column) for column in self.feature_columns_]
                if self.feature_columns_ is not None
                else None
            ),
        }

    def save(self, path: str | Path) -> Path:
        """Persist a fitted model and a credential-free metadata sidecar.

        Pickle files must only be loaded from trusted sources. MITRA artifacts
        are managed by AutoGluon and are intentionally not serialized here.
        """

        if not hasattr(self, "estimator_"):
            raise RuntimeError("Call fit before save.")
        if self.backbone_name_ != "custom" and not get_backbone_info(
            self.backbone_name_
        ).persistable:
            raise RuntimeError(
                f"The {self.backbone_name_!r} backbone manages external artifacts; "
                "use its native persistence workflow."
            )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(self.get_metadata(), indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path):
        """Load a trusted model artifact produced by :meth:`save`."""

        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("The artifact is not a SurvFM-RMST estimator.")
        return model


class SurvFMRMSTRegressor(PseudoRMSTRegressor):
    """Named user-facing alias for the SurvFM-RMST estimator."""


def fit_survfm_rmst(
    X_train,
    time,
    event,
    tau: float | None = None,
    *,
    estimator: object | None = None,
    backbone: str = "tabpfn",
    backbone_kwargs: dict | None = None,
    preprocess: str | object | None = "auto",
    tau_quantile: float = 0.8,
    min_events_for_tau: int = 5,
    clip_targets: bool = True,
    clip_predictions: bool = True,
) -> SurvFMRMSTRegressor:
    """Fit SurvFM-RMST in one call."""

    model = SurvFMRMSTRegressor(
        estimator=estimator,
        backbone=backbone,
        backbone_kwargs={} if backbone_kwargs is None else dict(backbone_kwargs),
        tau=tau,
        tau_quantile=tau_quantile,
        min_events_for_tau=min_events_for_tau,
        preprocess=preprocess,
        clip_targets=clip_targets,
        clip_predictions=clip_predictions,
    )
    return model.fit(X_train, time, event)
