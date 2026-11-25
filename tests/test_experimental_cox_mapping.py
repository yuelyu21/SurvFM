"""Checks for OOF isolation and actual survival-probability evaluation."""

import importlib.util
import unittest

import numpy as np
import pandas as pd

from examples.experimental_cox_mapping import OOFCoxSurvivalExtension


HAS_SK_SURV = importlib.util.find_spec("sksurv") is not None


@unittest.skipUnless(HAS_SK_SURV, "Optional scikit-survival dependency is absent")
class TestOOFCoxSurvivalExtension(unittest.TestCase):
    def cohort(self):
        rng = np.random.default_rng(16)
        X = pd.DataFrame({"row": np.arange(60), "x": rng.normal(size=60)})
        time = rng.uniform(2, 12, size=60)
        event = np.tile([True, True, False], 20)
        return X, time, event

    def test_oof_rows_and_outcomes_stay_inside_each_training_fold(self):
        X, time, event = self.cohort()
        event[np.argmin(time)] = True
        records = []

        class RecordingModel:
            def __init__(self, tau):
                self.tau_ = tau

            def fit(self, data, times, events):
                self.record = {"train": data.row.to_numpy(), "time": times.copy(),
                               "event": events.copy(), "query": []}
                records.append(self.record)
                return self

            def predict(self, data):
                self.record["query"].extend(data.row.tolist())
                return self.tau_ / (1 + np.exp(data.x.to_numpy()))

        fitted = OOFCoxSurvivalExtension(RecordingModel, tau=1.5).fit(X, time, event)
        self.assertEqual(fitted.n_base_fits_, 4)
        self.assertEqual(len(records), 4)
        seen = []
        for fold_id, record in enumerate(records[:3]):
            train, query = record["train"], record["query"]
            self.assertFalse(set(train) & set(query))
            np.testing.assert_array_equal(record["time"], time[train])
            np.testing.assert_array_equal(record["event"], event[train])
            np.testing.assert_array_equal(np.flatnonzero(fitted.fold_ids_ == fold_id), query)
            seen.extend(query)
        self.assertEqual(sorted(seen), list(range(len(X))))
        np.testing.assert_array_equal(records[-1]["train"], X.row)
        np.testing.assert_allclose(fitted.oof_rmst_, 1.5 / (1 + np.exp(X.x)))
        # All observed times exceed 2; the Cox baseline has no event yet.
        np.testing.assert_array_equal(
            fitted.predict_survival(X.iloc[:2], [0, 0.5, 1.5]), np.ones((2, 3))
        )

    def test_real_rmst_factory_produces_curves_and_ibs(self):
        from sksurv.metrics import integrated_brier_score
        from sksurv.util import Surv
        from survfm import SurvFMRMSTRegressor

        rng = np.random.default_rng(208)
        X = pd.DataFrame({"x": rng.normal(size=120), "group": np.tile(["a", "b"], 60)})
        latent = 0.2 + rng.exponential(np.exp(-0.5 * X.x.to_numpy()) * 3)
        # Common administrative follow-up makes this fixed fixture's IPCW
        # support explicit; no outcome-dependent test filtering is performed.
        time = np.minimum(latent, 6.0)
        event = latent <= 6.0

        def factory(tau):
            return SurvFMRMSTRegressor(backbone="linear", tau=tau)

        extension = OOFCoxSurvivalExtension(factory, tau=2.0).fit(
            X.iloc[:90], time[:90], event[:90]
        )
        grid = np.array([0.5, 1.0, 1.5, 2.0])
        curve = extension.predict_survival(X.iloc[90:], grid)
        self.assertEqual(curve.shape, (30, 4))
        self.assertTrue(np.all(np.diff(curve, axis=1) <= 0))
        y_train = Surv.from_arrays(event[:90], time[:90])
        y_test = Surv.from_arrays(event[90:], time[90:])
        score = integrated_brier_score(y_train, y_test, curve, grid)
        self.assertTrue(np.isfinite(score))
        self.assertGreaterEqual(score, 0)
        np.testing.assert_array_equal(
            extension.predict_rmst(X.iloc[90:]), extension.full_model_.predict(X.iloc[90:])
        )
        for bad in [[1, 1], [2, 1], [-0.1, 1], [1, 2.1], [np.nan]]:
            with self.assertRaises(ValueError):
                extension.predict_survival(X.iloc[90:], bad)

    def test_constant_oof_and_unsupported_tau_fail_explicitly(self):
        X, time, event = self.cohort()

        class ConstantModel:
            def __init__(self, tau):
                self.tau_ = tau

            def fit(self, *args):
                return self

            def predict(self, data):
                return np.full(len(data), self.tau_ / 2)

        model = OOFCoxSurvivalExtension(ConstantModel, tau=1.5)
        with self.assertRaisesRegex(ValueError, "constant"):
            model.fit(X, time, event)
        with self.assertRaises(RuntimeError):
            model.predict_survival(X, [0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "follow-up"):
            OOFCoxSurvivalExtension(ConstantModel, tau=100).fit(X, time, event)


if __name__ == "__main__":
    unittest.main()
