# Save and load

Local and compatible hosted estimators can be saved with a credential-free JSON
metadata sidecar.

```python
model.save("artifacts/survfm_rmst.pkl")

restored = SurvFMRMSTRegressor.load("artifacts/survfm_rmst.pkl")
predicted_rmst = restored.predict(X_test)
```

The sidecar records the backbone name, horizon, clipping rules, training-row
count, processed-feature count and training-derived horizon audit. It does not
contain outcomes, pseudo-targets or credential values.

The binary estimator artifact is different from the sidecar: depending on the
backbone, it can contain fitted preprocessing, pseudo-targets and training
context. Treat it as sensitive study data, store it in an access-controlled
location and review the third-party backbone's native persistence behavior.

!!! warning "Trusted artifacts only"
    The current release uses Python pickle for estimator persistence. Never
    load a model file from an untrusted source.

MITRA manages a directory of external AutoGluon artifacts and is intentionally
excluded from this generic persistence helper.
