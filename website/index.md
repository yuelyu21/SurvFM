# SurvFM

<p class="product-kicker">RMST target interface</p>

<div class="research-notice">
<strong>Research software.</strong> SurvFM is intended for methodological
research and retrospective evaluation. It has not been validated for clinical
decision-making.
</div>

SurvFM is the broader project for target-side adaptation of tabular models to
survival outcomes. **SurvFM-RMST** is the current validated method. It connects
right-censored outcomes to standard tabular regression by constructing
jackknife pseudo-observation targets for restricted mean survival time (RMST).
The same target interface can be paired with multiple unchanged tabular
backbones.

[Install SurvFM](installation.md){ .md-button .md-button--primary }
[View the source](https://github.com/yuelyu21/SurvFM){ .md-button }

![SurvFM-RMST framework](assets/images/survfm-rmst-framework.png){ .hero-figure }

<div class="workflow-grid">
  <div class="workflow-step"><strong>1. Define the horizon</strong><br>Use a clinically relevant restriction time with adequate follow-up support, or a training-fold event-time quantile for benchmarking.</div>
  <div class="workflow-step"><strong>2. Construct targets</strong><br>Convert censored training outcomes into exact jackknife pseudo-RMST labels.</div>
  <div class="workflow-step"><strong>3. Fit a backbone</strong><br>Use TabPFN by default, select another registered backbone, or supply a custom regressor.</div>
</div>

```python
from survfm import SurvFMRMSTRegressor

model = SurvFMRMSTRegressor(backbone="tabpfn", tau=365.0)
model.fit(X_train, time_train, event_train)

predicted_rmst = model.predict(X_test)
risk_score = model.predict_risk(X_test)
```

## Scope

- Static baseline covariates and a binary first-event endpoint.
- Horizon-specific predictions targeting covariate-conditional RMST.
- Exact, clipped jackknife pseudo-observations matching the manuscript workflow.
- Train-only mixed-data preprocessing for pandas DataFrames.
- Local and hosted backbones through one estimator API.

## Naming

- **SurvFM** names the project and extensible method family.
- **SurvFM-RMST** names the RMST target interface evaluated in the current work.
- **SurvFM-TabPFN**, **SurvFM-TabICL** and related labels identify
  SurvFM-RMST paired with a particular regression backbone.
