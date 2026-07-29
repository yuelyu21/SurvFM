# Mixed-data preprocessing

With `preprocess="auto"`, DataFrame dtypes determine the processing path.

- Numeric columns: median imputation, then standardization.
- Other columns: most-frequent imputation, then dense one-hot encoding.
- Unknown test categories: ignored rather than treated as errors.
- All preprocessing: fitted on the training fold only.

```python
model = SurvFMRMSTRegressor(
    backbone="tabpfn",
    tau=365,
    preprocess="auto",
)
model.fit(X_train_dataframe, time_train, event_train)
```

For an already encoded numeric matrix, automatic preprocessing passes values
through after dimensionality and finiteness checks.

## Custom preprocessing

Pass any transformer that implements `fit_transform(X)` and `transform(X)`:

```python
model = SurvFMRMSTRegressor(
    backbone="linear",
    tau=365,
    preprocess=my_pipeline,
)
```

Preprocessing may change the processed feature count. This matters for hosted
services with column limits; inspect `model.get_metadata()` after fitting.
