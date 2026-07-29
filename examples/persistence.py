"""Persist a fitted local estimator and its credential-free metadata."""

from pathlib import Path

from survfm import SurvFMRMSTRegressor


def save_and_reload(model: SurvFMRMSTRegressor, X_test):
    destination = model.save(Path("artifacts") / "survfm_rmst.pkl")
    restored = SurvFMRMSTRegressor.load(destination)
    return restored.predict(X_test)
