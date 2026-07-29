import numpy as np
import pandas as pd
import pytest

from survfm import SurvFMRMSTRegressor, fit_survfm_rmst


class MeanEstimator:
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_, dtype=float)


def test_fit_survfm_rmst_returns_rmst_predictions_and_risk():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 1, 0])

    model = fit_survfm_rmst(X, time, event, tau=3.0, estimator=MeanEstimator())
    pred = model.predict(X[:2])
    risk = model.predict_risk(X[:2])

    assert isinstance(model, SurvFMRMSTRegressor)
    assert pred.shape == (2,)
    assert np.all((0.0 <= pred) & (pred <= 3.0))
    assert np.allclose(risk, -pred)
    assert model.pseudo_targets_.shape == (4,)


def test_survfm_rmst_can_leave_predictions_unclipped():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 1, 0])

    model = fit_survfm_rmst(
        X,
        time,
        event,
        tau=3.0,
        estimator=MeanEstimator(),
        clip_predictions=False,
    )

    assert np.all(np.isfinite(model.predict(X)))


def test_explicit_horizon_rejects_nonbinary_event_codes():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    time = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1, 0, 2, 1])

    with pytest.raises(ValueError, match="event must be binary"):
        fit_survfm_rmst(
            X,
            time,
            event,
            tau=3.0,
            estimator=MeanEstimator(),
        )


def test_mixed_dataframe_preprocessing_and_persistence(tmp_path):
    X = pd.DataFrame(
        {
            "age": [40.0, 55.0, np.nan, 70.0, 62.0, 48.0],
            "stage": ["I", "II", "I", None, "III", "II"],
        }
    )
    time = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    event = np.array([1, 1, 0, 1, 1, 1])
    model = fit_survfm_rmst(
        X,
        time,
        event,
        estimator=MeanEstimator(),
        tau_quantile=0.8,
    )

    pred = model.predict(pd.DataFrame({"age": [50.0], "stage": ["unseen"]}))
    assert pred.shape == (1,)
    assert model.horizon_selection_.n_events == 5
    destination = model.save(tmp_path / "model.pkl")
    restored = SurvFMRMSTRegressor.load(destination)
    assert np.allclose(restored.predict(X.iloc[:2]), model.predict(X.iloc[:2]))
    assert destination.with_suffix(".pkl.metadata.json").exists()
