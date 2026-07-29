import numpy as np
import pandas as pd

from survfm import jackknife_pseudo_rmst, kaplan_meier_curve, km_rmst, realized_restricted_time


def test_kaplan_meier_curve_starts_at_one():
    times, survival = kaplan_meier_curve([1, 2, 3], [1, 0, 1])
    assert times[0] == 0.0
    assert survival[0] == 1.0
    assert np.all(np.diff(survival) <= 1e-12)


def test_km_rmst_all_events_simple_case():
    # S(t)=1 on [0,1), 0.5 on [1,2), 0 on [2,3].
    assert np.isclose(km_rmst([1, 2], [1, 1], tau=3), 1.5)


def test_km_rmst_no_events_equals_tau():
    assert np.isclose(km_rmst([1, 2, 3], [0, 0, 0], tau=4), 4.0)


def test_jackknife_pseudo_rmst_dataframe_input_is_bounded():
    df = pd.DataFrame({"time": [1.0, 2.0, 3.0, 4.0], "event": [1, 0, 1, 0]})
    pseudo = jackknife_pseudo_rmst(df, tau=3.0)
    assert pseudo.shape == (4,)
    assert np.all(np.isfinite(pseudo))
    assert np.all((0 <= pseudo) & (pseudo <= 3.0))


def test_realized_restricted_time():
    out = realized_restricted_time([1.0, 5.0, 10.0], tau=6.0)
    assert np.allclose(out, [1.0, 5.0, 6.0])
