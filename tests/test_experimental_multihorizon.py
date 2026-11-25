import numpy as np
import pytest

pytest.importorskip("scipy")

from examples.experimental_multihorizon import (
    MultiHorizonSurvivalExtension,
    ReconstructionFailure,
    integral_matrix,
    interpolate,
    reconstruct,
)


def test_known_piecewise_linear_curve_is_recovered_from_its_integrals():
    grid = np.linspace(0.0, 4.0, 9)
    expected = np.array([1.0, 0.9, 0.85, 0.7, 0.65, 0.5, 0.4, 0.2, 0.1])
    integrals = np.cumsum(np.diff(grid) * (expected[:-1] + expected[1:]) / 2)
    actual, recovered_integrals, audit = reconstruct(integrals.tolist(), grid.tolist())
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-5)
    np.testing.assert_allclose(recovered_integrals, integrals, rtol=0, atol=2e-6)
    assert audit["Objective"] < 1e-12
    assert audit["KKTResidual"] <= 2e-6


def test_inconsistent_rmst_is_adjusted_without_changing_original_values():
    grid = np.linspace(0.0, 4.0, 9)
    original = np.array([0.5, 0.2, 1.4, 0.4, 2.4, 1.0, 3.4, 0.3])
    saved = original.copy()
    curve, integrals, audit = reconstruct(original, grid)
    assert curve[0] == 1.0
    assert np.all((curve >= 0) & (curve <= 1))
    assert np.all(np.diff(curve) <= 0)
    assert audit["Objective"] > 0
    np.testing.assert_array_equal(original, saved)
    np.testing.assert_allclose(integrals, integral_matrix(grid) @ curve, atol=1e-12)


def test_boundary_cases_show_the_interpolation_assumption():
    grid = np.linspace(0.0, 4.0, 9)
    survival, integrals, _ = reconstruct(grid[1:], grid)
    np.testing.assert_allclose(survival, 1.0, atol=1e-6)
    np.testing.assert_allclose(integrals, grid[1:], atol=1e-6)
    survival, integrals, _ = reconstruct(np.zeros(8), grid)
    assert survival[0] == 1.0
    assert integrals[0] > 0  # A continuous first segment cannot represent immediate death.


def test_solver_failure_has_no_fallback():
    with pytest.raises(ReconstructionFailure, match="SLSQP"):
        reconstruct(np.zeros(8), np.linspace(0, 4, 9), maxiter=1)


@pytest.mark.parametrize("grid", [[1, 2], [0, 1, 1], [0, np.inf], [], [[0, 1]]])
def test_invalid_reconstruction_grids_are_rejected(grid):
    with pytest.raises(ValueError, match="grid"):
        reconstruct([0.2], grid)


def test_single_query_time_and_no_extrapolation_or_empty_query():
    np.testing.assert_allclose(interpolate([0, 1, 2], [1, 0.6, 0.2], [0.5]), [0.8])
    for times in ([], [np.nan], [-0.1], [2.1], [1.0, 0.5], [1.0, 1.0]):
        with pytest.raises(ValueError):
            interpolate([0, 1, 2], [1, 0.6, 0.2], times)
    with pytest.raises(ValueError, match="survival"):
        interpolate([0, 1, 2], [1, 0.5, 0.7], [1])


class KnownRMSTModel:
    """An analytic curve S(t)=1-0.1t for a transparent adapter integration test."""

    def __init__(self, tau):
        self.tau = tau
        self.fit_calls = 0

    def fit(self, X, time, event):
        self.fit_calls += 1
        self.training_rows = np.array(X, copy=True)
        self.training_time = np.array(time, copy=True)
        self.training_event = np.array(event, copy=True)
        return self

    def predict(self, X):
        return np.full(len(X), self.tau - 0.05 * self.tau**2)


def test_adapter_uses_fresh_models_at_numeric_horizons_and_preserves_native_rmst():
    made = []

    def factory(tau):
        model = KnownRMSTModel(tau)
        made.append(model)
        return model

    X = np.arange(12).reshape(6, 2)
    time = np.arange(1, 7, dtype=float)
    event = np.array([1, 1, 0, 1, 0, 1])
    model = MultiHorizonSurvivalExtension(factory, tau=4).fit(X, time, event)
    assert model.n_base_fits_ == 8
    assert len({id(base) for base in made}) == 8
    np.testing.assert_allclose([base.tau for base in made], np.arange(1, 9) / 2)
    for base in made:
        assert base.fit_calls == 1
        np.testing.assert_array_equal(base.training_rows, X)
        np.testing.assert_array_equal(base.training_time, time)
        np.testing.assert_array_equal(base.training_event, event)
    result = model.reconstruct_predictions(X[:2])
    assert result["survival"].shape == (2, 9)
    assert result["rmst"].shape == result["integrated_rmst"].shape == (2, 8)
    assert len(result["diagnostics"]) == 2
    np.testing.assert_allclose(result["integrated_rmst"], result["rmst"], atol=2e-6)
    np.testing.assert_array_equal(model.predict_rmst(X[:2]), result["rmst"][:, -1])
    np.testing.assert_allclose(model.predict_survival(X[:2], [1.25]), [[0.875], [0.875]], atol=2e-5)


def test_reusing_one_model_or_custom_estimator_is_rejected():
    X = np.ones((3, 2))
    shared = KnownRMSTModel(4)
    with pytest.raises(ValueError, match="fresh model"):
        MultiHorizonSurvivalExtension(lambda tau: shared, 4).fit(X, [1, 2, 5], [1, 0, 1])

    def factory(tau):
        model = KnownRMSTModel(tau)
        model.estimator = shared
        return model

    with pytest.raises(ValueError, match="underlying estimator"):
        MultiHorizonSurvivalExtension(factory, 4).fit(X, [1, 2, 5], [1, 0, 1])


def test_invalid_fit_inputs_and_prediction_shape_are_rejected():
    model = MultiHorizonSurvivalExtension(KnownRMSTModel, 4)
    X = np.ones((3, 2))
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_rmst(X)
    with pytest.raises(ValueError, match="binary"):
        model.fit(X, [1, 2, 5], [1, 2, 0])
    with pytest.raises(ValueError, match="time"):
        model.fit(X, [1, np.nan, 5], [1, 0, 0])
    model.fit(X, [1, 2, 5], [1, 0, 0])
    with pytest.raises(ValueError, match="non-empty"):
        model.predict_survival(X[:0], [1])
    model.models_[-1].predict = lambda X: np.ones((len(X), 1))
    with pytest.raises(ValueError, match="one finite RMST"):
        model.predict_rmst(X)


def test_factory_cannot_ignore_the_requested_numeric_horizon():
    def factory(tau):
        model = KnownRMSTModel(tau)
        model.tau_ = 4.0  # A factory that silently uses the same horizon for all fits.
        return model

    model = MultiHorizonSurvivalExtension(factory, 4)
    with pytest.raises(ValueError, match="requested numeric horizon"):
        model.fit(np.ones((3, 2)), [1, 2, 5], [1, 0, 1])
    assert model.n_base_fits_ == 1
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_rmst(np.ones((1, 2)))


def test_training_support_and_time_zero_event_assumption_are_explicit():
    X = np.ones((3, 2))
    model = MultiHorizonSurvivalExtension(KnownRMSTModel, 4)
    with pytest.raises(ValueError, match="maximum observed training time"):
        model.fit(X, [1, 2, 3], [1, 0, 1])
    assert model.n_base_fits_ == 0
    with pytest.raises(ValueError, match="events at time zero"):
        model.fit(X, [0, 2, 5], [1, 0, 1])
    assert model.n_base_fits_ == 0
    # Censoring at zero is not an observed jump in survival at the boundary.
    model.fit(X, [0, 2, 5], [0, 0, 1])
    assert model.n_base_fits_ == 8


@pytest.mark.parametrize("tau,n_horizons", [(0, 8), (np.nan, 8), (4, 0), (4, 2.5), (4, True)])
def test_invalid_horizon_configuration(tau, n_horizons):
    with pytest.raises(ValueError):
        MultiHorizonSurvivalExtension(KnownRMSTModel, tau, n_horizons)
