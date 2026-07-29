from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from survfm.pseudo_rmst import jackknife_pseudo_rmst, jackknife_pseudo_rmst_fast_exact


class FastExactPseudoRMSTTests(unittest.TestCase):
    def assert_reference_parity(self, time, event, tau, atol=1e-12):
        frame = pd.DataFrame(
            {"time": time, "event": (np.asarray(event, dtype=float) == 1.0).astype(int)}
        )
        reference = jackknife_pseudo_rmst(frame, tau=tau)
        accelerated = jackknife_pseudo_rmst_fast_exact(time, event, tau)
        np.testing.assert_allclose(accelerated, reference, rtol=0.0, atol=atol)

    def test_ordinary_censoring(self):
        self.assert_reference_parity(
            [1.0, 2.2, 3.1, 4.8, 6.0, 7.5], [1, 0, 1, 0, 1, 0], 6.5
        )

    def test_tied_event_times(self):
        self.assert_reference_parity(
            [1.0, 2.0, 2.0, 2.0, 4.0, 5.0], [0, 1, 1, 1, 1, 0], 4.5
        )

    def test_censoring_tied_with_event_time(self):
        self.assert_reference_parity(
            [1.0, 2.0, 2.0, 3.0, 3.0, 5.0], [1, 1, 0, 1, 0, 0], 4.0
        )

    def test_all_censored(self):
        time = np.array([1.0, 2.0, 3.0, 5.0])
        event = np.zeros(4, dtype=int)
        self.assert_reference_parity(time, event, 4.0)
        np.testing.assert_array_equal(
            jackknife_pseudo_rmst_fast_exact(time, event, 4.0), np.full(4, 4.0)
        )

    def test_additional_endpoint_code_preserves_submitted_event_equals_one_convention(self):
        self.assert_reference_parity(
            [1.0, 2.0, 2.5, 3.0, 4.0, 6.0], [1, 2, 0, 1, 2, 0], 5.0
        )

    def test_small_n(self):
        np.testing.assert_allclose(
            jackknife_pseudo_rmst_fast_exact([2.0], [1], 4.0),
            np.array([2.0]),
            rtol=0.0,
            atol=0.0,
        )
        self.assert_reference_parity([2.0, 5.0], [1, 0], 4.0)

    def test_clipping_behavior(self):
        time = np.array([1.0, 2.0, 3.0, 8.0, 9.0])
        event = np.array([1, 0, 0, 1, 0])
        tau = 7.0
        raw = jackknife_pseudo_rmst_fast_exact(time, event, tau, clip=False)
        clipped = jackknife_pseudo_rmst_fast_exact(time, event, tau, clip=True)
        np.testing.assert_allclose(clipped, np.clip(raw, 0.0, tau), rtol=0.0, atol=0.0)
        self.assert_reference_parity(time, event, tau)

    def test_batch_size_does_not_change_results(self):
        time = np.array([1, 2, 2, 3, 4, 6, 7, 8], dtype=float)
        event = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=int)
        expected = jackknife_pseudo_rmst_fast_exact(time, event, 7.5, batch_size=1)
        for batch_size in [2, 3, 64]:
            actual = jackknife_pseudo_rmst_fast_exact(
                time, event, 7.5, batch_size=batch_size
            )
            # Batch shape can change floating-point reduction order without
            # changing the exact jackknife estimator being evaluated.
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
