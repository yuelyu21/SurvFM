"""Training-fold utilities for selecting an RMST restriction horizon."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HorizonSelection:
    """Audit record for a training-derived RMST horizon."""

    tau: float
    quantile: float
    n_observations: int
    n_events: int
    minimum_event_time: float
    maximum_event_time: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def select_rmst_horizon(
    time,
    event,
    *,
    quantile: float = 0.8,
    min_events: int = 5,
) -> HorizonSelection:
    """Select ``tau`` from a quantile of observed training-fold event times.

    Only the training outcomes supplied to this function are used. This helper
    implements the benchmark convention and is intended for generic benchmark
    use. In a disease-specific application, ``tau`` should preferably be
    prespecified from the clinical decision horizon and follow-up support.
    """

    time_arr = np.asarray(time, dtype=float).reshape(-1)
    event_arr = np.asarray(event).reshape(-1)
    if len(time_arr) != len(event_arr) or len(time_arr) == 0:
        raise ValueError("time and event must be non-empty and have equal length.")
    if not np.all(np.isfinite(time_arr)) or np.any(time_arr < 0):
        raise ValueError("time must contain finite, non-negative values.")
    if not np.isin(event_arr, [0, 1, False, True]).all():
        raise ValueError("event must be binary with 1=event and 0=censored.")
    if not 0 < float(quantile) <= 1:
        raise ValueError("quantile must lie in (0, 1].")
    if int(min_events) < 1:
        raise ValueError("min_events must be at least 1.")

    event_times = time_arr[np.asarray(event_arr, dtype=bool)]
    if len(event_times) < int(min_events):
        raise ValueError(
            f"At least {int(min_events)} observed training events are required; "
            f"received {len(event_times)}."
        )

    tau = float(np.quantile(event_times, float(quantile)))
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("The selected horizon is not positive and finite.")
    return HorizonSelection(
        tau=tau,
        quantile=float(quantile),
        n_observations=int(len(time_arr)),
        n_events=int(len(event_times)),
        minimum_event_time=float(np.min(event_times)),
        maximum_event_time=float(np.max(event_times)),
    )
