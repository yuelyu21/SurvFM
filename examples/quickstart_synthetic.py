"""Minimal local SurvFM example using a synthetic censored cohort."""

import numpy as np
import pandas as pd

from survfm import SurvFMRMSTRegressor


rng = np.random.default_rng(20260722)
n = 180
X = pd.DataFrame(
    {
        "age": rng.normal(60, 9, n),
        "biomarker": rng.normal(size=n),
        "group": rng.choice(["A", "B", "C"], size=n),
    }
)
linear_predictor = 0.35 * X["biomarker"].to_numpy() + 0.2 * (X["group"] == "C")
event_time = rng.exponential(scale=np.exp(-linear_predictor) * 18)
censor_time = rng.exponential(scale=24, size=n)
time = np.minimum(event_time, censor_time)
event = (event_time <= censor_time).astype(int)

train = np.arange(0, 130)
test = np.arange(130, n)
model = SurvFMRMSTRegressor(
    backbone="random_forest",
    tau_quantile=0.8,
)
model.fit(X.iloc[train], time[train], event[train])

print(f"Selected tau: {model.tau_:.2f}")
print("Predicted RMST:", model.predict(X.iloc[test])[:5])
print("Risk scores:", model.predict_risk(X.iloc[test])[:5])
