"""Scientific checks for the example's IBS protocol, without foundation weights."""
import numpy as np
import pytest

pytest.importorskip("sksurv")
from sksurv.util import Surv
from examples.experimental_survival_curves import evaluate_ibs


def test_uncensored_ibs_matches_direct_squared_error_integral():
    train = Surv.from_arrays([True] * 4, [1., 3., 6., 9.])
    test = Surv.from_arrays([True] * 3, [1., 3., 5.])
    times = np.array([2., 4.])
    probability = np.array([[.2, .1], [.8, .3], [.9, .7]])
    truth = test["time"][:, None] > times
    expected = np.mean(np.square(probability - truth), axis=0).mean()
    actual = evaluate_ibs(train, test, probability, times)
    assert actual["status"] == "ok"
    assert actual["ibs"] == pytest.approx(expected)


def test_followup_support_failure_remains_visible_without_mutating_inputs():
    train = Surv.from_arrays([True, False, True], [1., 3., 5.])
    test = Surv.from_arrays([True, True, False], [1., 4., 7.])
    original = test.copy()
    times = np.array([2., 4.])
    probability = np.tile([.7, .4], (3, 1))
    actual = evaluate_ibs(train, test, probability, times)
    assert actual["status"] == "not_evaluable"
    assert actual["ibs"] is None and actual["error"]
    np.testing.assert_array_equal(test, original)
    np.testing.assert_array_equal(times, [2., 4.])


def test_reject_risk_scores_in_place_of_probability_matrix():
    y = Surv.from_arrays([True, True, True], [1., 3., 5.])
    with pytest.raises(ValueError, match="survival probabilities"):
        evaluate_ibs(y, y, np.array([-1., -2., -3.]), [2., 4.])
