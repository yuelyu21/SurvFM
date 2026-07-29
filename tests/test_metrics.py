import numpy as np

from survfm.metrics import ipcw_rmst_rmse, normalized_rmse, predicted_rmst_to_risk, rmse


def test_rmse_uses_finite_pairs():
    assert np.isclose(rmse([1.0, 2.0, np.nan], [1.5, 1.5, 0.0]), 0.5)


def test_normalized_rmse_divides_by_tau():
    assert np.isclose(normalized_rmse([1.0, 3.0], [2.0, 2.0], tau=4.0), 0.25)


def test_predicted_rmst_to_risk_reverses_direction():
    risk = predicted_rmst_to_risk([10.0, 5.0])
    assert risk[1] > risk[0]


def test_ipcw_rmst_rmse_equals_normalized_rmse_without_censoring():
    result = ipcw_rmst_rmse(
        train_time=[1.0, 2.0, 3.0, 4.0],
        train_event=[1, 1, 1, 1],
        test_time=[1.0, 2.0, 4.0],
        test_event=[1, 1, 0],
        predicted_rmst=[1.5, 1.5, 2.5],
        tau=3.0,
    )
    expected = normalized_rmse([1.0, 2.0, 3.0], [1.5, 1.5, 2.5], tau=3.0)
    assert np.isclose(result["normalized_rmse"], expected)
    assert result["n_test"] == 3
    assert result["n_evaluable"] == 3


def test_ipcw_rmst_rmse_uses_full_test_denominator():
    result = ipcw_rmst_rmse(
        train_time=[1.0, 2.0, 3.0, 4.0],
        train_event=[1, 0, 1, 0],
        test_time=[1.0, 2.0, 4.0],
        test_event=[1, 0, 0],
        predicted_rmst=[1.0, 2.0, 2.5],
        tau=3.0,
    )
    # The event contributes zero error. The horizon-reaching row has G(3)=2/3,
    # weight 3/2 and squared normalized error (0.5/3)^2. The censored-before-
    # horizon row contributes zero, while all three rows remain in the denominator.
    expected = np.sqrt((1.5 * (0.5 / 3.0) ** 2) / 3.0)
    assert np.isclose(result["normalized_rmse"], expected)
    assert result["n_evaluable"] == 2
    assert result["n_valid_ipcw"] == 2


def test_ipcw_event_exactly_at_tau_is_not_double_counted():
    result = ipcw_rmst_rmse(
        train_time=[1.0, 2.0, 4.0],
        train_event=[1, 1, 1],
        test_time=[3.0],
        test_event=[1],
        predicted_rmst=[2.0],
        tau=3.0,
    )
    assert np.isclose(result["normalized_rmse"], 1.0 / 3.0)
    assert np.isclose(result["mean_ipcw_weight"], 1.0)


def test_ipcw_censoring_exactly_at_tau_is_observable():
    result = ipcw_rmst_rmse(
        train_time=[1.0, 2.0, 4.0],
        train_event=[1, 1, 1],
        test_time=[3.0],
        test_event=[0],
        predicted_rmst=[2.0],
        tau=3.0,
    )
    assert np.isclose(result["normalized_rmse"], 1.0 / 3.0)
    assert result["n_evaluable"] == 1


def test_ipcw_support_failure_marks_block_nonevaluable_without_flooring():
    result = ipcw_rmst_rmse(
        train_time=[1.0, 2.0],
        train_event=[0, 0],
        test_time=[3.0],
        test_event=[0],
        predicted_rmst=[2.0],
        tau=3.0,
    )
    assert np.isnan(result["normalized_rmse"])
    assert result["evaluation_success"] == 0
    assert result["n_unstable_ipcw"] == 1
