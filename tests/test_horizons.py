import numpy as np
import pytest

from survfm import select_rmst_horizon


def test_training_event_quantile_horizon_has_audit_metadata():
    result = select_rmst_horizon(
        time=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 8.0]),
        event=np.array([1, 1, 0, 1, 1, 1]),
        quantile=0.8,
        min_events=5,
    )
    assert result.tau == pytest.approx(np.quantile([1.0, 2.0, 4.0, 5.0, 8.0], 0.8))
    assert result.n_events == 5
    assert result.to_dict()["quantile"] == 0.8


def test_horizon_selection_rejects_too_few_events():
    with pytest.raises(ValueError, match="observed training events"):
        select_rmst_horizon([1, 2, 3], [1, 0, 0], min_events=2)
