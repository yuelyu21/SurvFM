"""Lazy backbone registry for the SurvFM framework.

Optional dependencies are imported only when their corresponding backbone is
constructed. API credentials are deliberately read at request time and are
never stored on estimator instances.
"""

from __future__ import annotations

import inspect
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BackboneInfo:
    name: str
    display_name: str
    access: str
    install_extra: str
    description: str
    persistable: bool = True

    def to_dict(self) -> dict[str, str | bool]:
        return asdict(self)


_BACKBONES = {
    "tabpfn": BackboneInfo(
        "tabpfn",
        "SurvFM-TabPFN",
        "local",
        "tabpfn",
        "Reference tabular foundation-model backbone.",
    ),
    "tabicl": BackboneInfo(
        "tabicl",
        "SurvFM-TabICL",
        "local",
        "tabicl",
        "TabICL regression backbone.",
    ),
    "tabdpt": BackboneInfo(
        "tabdpt",
        "SurvFM-TabDPT",
        "local",
        "tabdpt",
        "TabDPT regression backbone with deterministic prediction settings.",
    ),
    "tabh2o": BackboneInfo(
        "tabh2o",
        "SurvFM-TabH2O",
        "hosted API",
        "tabh2o",
        "Hosted TabH2O regression API; inputs are sent to a third-party service.",
    ),
    "mitra": BackboneInfo(
        "mitra",
        "SurvFM-MITRA",
        "local",
        "mitra",
        "AutoGluon MITRA base-model predictions without ensemble substitution.",
        persistable=False,
    ),
    "linear": BackboneInfo(
        "linear", "Linear-RMST", "local", "sklearn", "Linear regression baseline."
    ),
    "random_forest": BackboneInfo(
        "random_forest",
        "RFReg-RMST",
        "local",
        "sklearn",
        "Random-forest regression on pseudo-RMST targets.",
    ),
    "gradient_boosting": BackboneInfo(
        "gradient_boosting",
        "GBReg-RMST",
        "local",
        "sklearn",
        "Gradient-boosting regression on pseudo-RMST targets.",
    ),
}


def list_backbones() -> list[BackboneInfo]:
    """Return the registered backbones in stable display order."""

    return list(_BACKBONES.values())


def get_backbone_info(name: str) -> BackboneInfo:
    key = str(name).strip().lower()
    if key not in _BACKBONES:
        choices = ", ".join(_BACKBONES)
        raise ValueError(f"Unknown backbone {name!r}. Available backbones: {choices}.")
    return _BACKBONES[key]


def _construct_with_supported_kwargs(cls, kwargs: dict[str, Any]):
    try:
        supported = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return cls(**kwargs)
    return cls(**{key: value for key, value in kwargs.items() if key in supported})


class _TabDPTAdapter:
    def __init__(self, *, n_ensembles: int = 8, seed: int = 20260715, **kwargs):
        try:
            from tabdpt import TabDPTRegressor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'tabdpt' backbone requires TabDPT. Install the tabdpt extra."
            ) from exc
        self.model = TabDPTRegressor(**kwargs)
        self.n_ensembles = int(n_ensembles)
        self.seed = int(seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        kwargs = {"n_ensembles": self.n_ensembles, "seed": self.seed}
        try:
            supported = inspect.signature(self.model.predict).parameters
            kwargs = {key: value for key, value in kwargs.items() if key in supported}
        except (TypeError, ValueError):
            pass
        return self.model.predict(X, **kwargs)


class _TabH2OAdapter:
    """Minimal hosted-API adapter that never retains a credential value."""

    def __init__(
        self,
        *,
        api_key_env: str = "TABH2O_API_KEY",
        endpoint: str = "https://tabh2o.h2oai.com/api/v1/predict",
        timeout: float = 240.0,
        maximum_features: int = 99,
    ):
        self.api_key_env = str(api_key_env)
        self.endpoint = str(endpoint)
        self.timeout = float(timeout)
        self.maximum_features = int(maximum_features)

    def fit(self, X, y):
        x = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if x.ndim != 2 or len(x) != len(target):
            raise ValueError("X must be two-dimensional and match y.")
        if x.shape[1] > self.maximum_features:
            raise ValueError(
                f"TabH2O accepts at most {self.maximum_features} processed features; "
                f"received {x.shape[1]}."
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(target)):
            raise ValueError("TabH2O inputs must be finite after preprocessing.")
        self._x_train = x.copy()
        self._y_train = target.copy()
        return self

    def predict(self, X):
        try:
            import requests
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'tabh2o' backbone requires requests. Install the tabh2o extra."
            ) from exc

        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"TabH2O credential missing. Set {self.api_key_env} only in the runtime environment."
            )
        x_test = np.asarray(X, dtype=float)
        if x_test.ndim != 2 or x_test.shape[1] != self._x_train.shape[1]:
            raise ValueError("TabH2O test features must match the fitted training features.")
        if not np.all(np.isfinite(x_test)):
            raise ValueError("TabH2O test inputs must be finite after preprocessing.")

        columns = [f"x{i}" for i in range(self._x_train.shape[1])]
        train = np.column_stack([self._x_train, self._y_train])
        payload = {
            "train": {"data": train.tolist(), "columns": columns + ["target"]},
            "test": {"data": x_test.tolist(), "columns": columns},
            "target_column": "target",
            "task": "regression",
        }
        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"TabH2O HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        pred = np.asarray(data.get("predictions", []), dtype=float)
        if len(pred) != len(x_test):
            raise RuntimeError(
                f"TabH2O returned {len(pred)} predictions for {len(x_test)} rows."
            )
        return pred


class _MITRAAdapter:
    def __init__(
        self,
        *,
        path: str | None = None,
        time_limit: float | None = None,
        memory_ratio: float = 3.0,
        **fit_kwargs,
    ):
        try:
            from autogluon.tabular import TabularPredictor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'mitra' backbone requires AutoGluon with MITRA support. "
                "Install the mitra extra in a separate environment."
            ) from exc
        self._predictor_cls = TabularPredictor
        self.path = path
        self.time_limit = time_limit
        self.memory_ratio = float(memory_ratio)
        self.fit_kwargs = dict(fit_kwargs)

    def fit(self, X, y):
        import pandas as pd

        frame = pd.DataFrame(np.asarray(X, dtype=float))
        frame["target"] = np.asarray(y, dtype=float)
        path = self.path or tempfile.mkdtemp(prefix="survfm_mitra_")
        self.predictor_ = self._predictor_cls(
            label="target", problem_type="regression", path=str(Path(path))
        )
        kwargs = {
            "train_data": frame,
            "hyperparameters": {"MITRA": {"fine_tune": False}},
            "ag_args_fit": {"ag.max_memory_usage_ratio": self.memory_ratio},
            **self.fit_kwargs,
        }
        if self.time_limit is not None:
            kwargs["time_limit"] = self.time_limit
        self.predictor_.fit(**kwargs)
        model_names = set(self.predictor_.model_names())
        if "Mitra" not in model_names:
            raise RuntimeError(
                "AutoGluon did not produce the direct 'Mitra' base model; "
                "no ensemble substitution was made."
            )
        return self

    def predict(self, X):
        import pandas as pd

        return self.predictor_.predict(pd.DataFrame(np.asarray(X, dtype=float)), model="Mitra")


def create_backbone(name: str = "tabpfn", **kwargs):
    """Construct a registered regression backbone using lazy imports."""

    key = get_backbone_info(name).name
    if key == "tabpfn":
        try:
            from tabpfn import TabPFNRegressor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The default 'tabpfn' backbone requires TabPFN. Install the tabpfn extra."
            ) from exc
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        defaults = {
            "ignore_pretraining_limits": True,
            "n_estimators": 1,
            "show_progress_bar": False,
            "random_state": 20260709,
        }
        return _construct_with_supported_kwargs(TabPFNRegressor, {**defaults, **kwargs})
    if key == "tabicl":
        try:
            from tabicl import TabICLRegressor
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "The 'tabicl' backbone requires TabICL. Install the tabicl extra."
            ) from exc
        defaults = {
            "n_estimators": 8,
            "batch_size": 1,
            "allow_auto_download": True,
            "verbose": False,
            "random_state": 20260615,
        }
        return _construct_with_supported_kwargs(TabICLRegressor, {**defaults, **kwargs})
    if key == "tabdpt":
        return _TabDPTAdapter(**kwargs)
    if key == "tabh2o":
        return _TabH2OAdapter(**kwargs)
    if key == "mitra":
        return _MITRAAdapter(**kwargs)

    try:
        if key == "linear":
            from sklearn.linear_model import LinearRegression

            return LinearRegression(**kwargs)
        if key == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            defaults = {"n_estimators": 500, "min_samples_leaf": 3, "random_state": 42, "n_jobs": -1}
            return RandomForestRegressor(**{**defaults, **kwargs})
        if key == "gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingRegressor

            defaults = {"random_state": 42}
            return HistGradientBoostingRegressor(**{**defaults, **kwargs})
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            f"The {key!r} backbone requires scikit-learn. Install the sklearn extra."
        ) from exc
    raise AssertionError(f"Unhandled backbone registry entry: {key}")
