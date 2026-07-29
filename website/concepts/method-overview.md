# Method overview

SurvFM is the broader target-interface project. Its current validated method,
SurvFM-RMST, separates the survival-specific part of the problem from the
tabular prediction engine. One outer training fold and one restriction horizon
define the complete workflow.

1. **Observed data.** Supply baseline covariates, observed follow-up and a binary
   first-event indicator.
2. **Horizon.** Use a prespecified decision horizon, or derive a training-fold
   event-time quantile for a generic benchmark.
3. **Target construction.** Estimate marginal Kaplan–Meier RMST and construct
   exact delete-one pseudo-observations.
4. **Preprocessing.** Fit numeric and categorical transformations only on the
   training fold.
5. **Regression.** Supply the same pseudo-RMST context to an unchanged tabular
   backbone.
6. **Prediction.** Return bounded RMST at the selected horizon; use its negative
   only when a risk-oriented score is required.

The method is estimand-focused: changing the backbone does not change the target
definition. Conversely, changing τ changes the prediction question and usually
requires constructing a new target and fitting a horizon-specific model.

## Minimal mathematical map

<div class="method-map" aria-label="SurvFM RMST method map">
  <span>Right-censored training fold</span>
  <span aria-hidden="true">→</span>
  <span>Kaplan–Meier RMST at τ</span>
  <span aria-hidden="true">→</span>
  <span>Jackknife pseudo-RMST labels</span>
  <span aria-hidden="true">→</span>
  <span>Tabular regression backbone</span>
  <span aria-hidden="true">→</span>
  <span>Held-out RMST prediction</span>
</div>

The [pseudo-RMST derivation](pseudo-rmst.md) gives the statistical construction;
the [shared regression interface](backbone-interface.md) shows how it is supplied
to different backbones.
