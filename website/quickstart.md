# Quickstart

## Prepare a static table

`X_train` may be a numeric array or a pandas DataFrame containing numeric and
categorical baseline covariates. Outcomes use observed follow-up time and a
binary event indicator (`1=event`, `0=censored`).

```python
import pandas as pd

X_train = pd.DataFrame(
    {
        "age": [42, 55, 63, 71, 48, 60],
        "biomarker": [0.2, -0.1, 1.2, 0.5, -0.4, 0.8],
        "stage": ["I", "II", "III", "II", "I", "III"],
    }
)
time_train = [120, 340, 280, 500, 190, 430]
event_train = [1, 0, 1, 1, 0, 1]
```

## Fit with an explicit horizon

```python
from survfm import SurvFMRMSTRegressor

model = SurvFMRMSTRegressor(
    backbone="random_forest",
    tau=365,
    preprocess="auto",
)
model.fit(X_train, time_train, event_train)
```

The preprocessor is fitted on the training table only. Numeric columns are
median-imputed and standardized; categorical columns are most-frequent-imputed
and one-hot encoded with unseen levels ignored at prediction time.

## Predict RMST and risk ordering

```python
X_test = pd.DataFrame(
    {
        "age": [51, 68],
        "biomarker": [0.1, 1.0],
        "stage": ["II", "III"],
    }
)

predicted_rmst = model.predict(X_test)
risk_score = model.predict_risk(X_test)
```

`predict()` returns restricted event-free time in `[0, tau]`. `predict_risk()`
returns its sign reversal, so larger scores indicate shorter predicted RMST.

## Use the default backbone

```python
model = SurvFMRMSTRegressor(
    backbone="tabpfn",
    backbone_kwargs={"device": "cuda"},
    tau=365,
)
```

The installed TabPFN package controls its default checkpoint. Supply and record
an explicit checkpoint argument when reproducing a locked analysis.
