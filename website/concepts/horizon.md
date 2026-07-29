# Choosing the RMST horizon

The restriction horizon is part of the prediction question. It should be chosen
before fitting when a clinically meaningful decision window is available.

## Preferred clinical workflow

Choose τ from the endpoint, intended decision and observed follow-up
support. Document the choice before evaluating model performance.

```python
model = SurvFMRMSTRegressor(backbone="tabpfn", tau=365.0)
```

## Generic benchmark workflow

When no external horizon is available, the package can derive a quantile from
observed event times in the training fold only:

```python
model = SurvFMRMSTRegressor(
    backbone="tabpfn",
    tau=None,
    tau_quantile=0.8,
    min_events_for_tau=5,
)
model.fit(X_train, time_train, event_train)
print(model.horizon_selection_.to_dict())
```

This avoids test-outcome leakage but makes the estimand split-specific. It is a
benchmark convenience, not a substitute for a prespecified clinical horizon.
