# Pseudo-RMST targets

## From censored observations to an estimand

For observation *i*, let *X*<sub>*i*</sub> denote baseline covariates,
*T*<sub>*i*</sub> the event time and *C*<sub>*i*</sub> the censoring time. Only
the observed follow-up and event indicator are available:

<div class="math-display" role="math" aria-label="Observed data consist of X i, Y i and Delta i, where Y i is the minimum of T i and C i and Delta i indicates T i less than or equal to C i.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>O</mi><mi>i</mi></msub><mo>=</mo>
  <mo>(</mo><msub><mi>X</mi><mi>i</mi></msub><mo>,</mo><msub><mi>Y</mi><mi>i</mi></msub><mo>,</mo><msub><mi>Δ</mi><mi>i</mi></msub><mo>)</mo><mo>,</mo>
  <mspace width="1em"/>
  <msub><mi>Y</mi><mi>i</mi></msub><mo>=</mo><mi>min</mi><mo>(</mo><msub><mi>T</mi><mi>i</mi></msub><mo>,</mo><msub><mi>C</mi><mi>i</mi></msub><mo>)</mo><mo>,</mo>
  <mspace width="1em"/>
  <msub><mi>Δ</mi><mi>i</mi></msub><mo>=</mo><mi>𝕀</mi><mo>(</mo><msub><mi>T</mi><mi>i</mi></msub><mo>≤</mo><msub><mi>C</mi><mi>i</mi></msub><mo>)</mo><mo>.</mo>
</math>
</div>

At restriction horizon τ, the prediction target is covariate-conditional RMST:

<div class="math-display" role="math" aria-label="Conditional RMST equals the expected minimum of T and tau conditional on X equals x, and equals the integral of conditional survival from zero to tau.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>μ</mi><mi>τ</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo>
  <mo>=</mo><mi>𝔼</mi><mo>{</mo><mi>min</mi><mo>(</mo><mi>T</mi><mo>,</mo><mi>τ</mi><mo>)</mo><mo>|</mo><mi>X</mi><mo>=</mo><mi>x</mi><mo>}</mo>
  <mo>=</mo><msubsup><mo>∫</mo><mn>0</mn><mi>τ</mi></msubsup>
  <mi>S</mi><mo>(</mo><mi>t</mi><mo>|</mo><mi>X</mi><mo>=</mo><mi>x</mi><mo>)</mo><mi>d</mi><mi>t</mi><mo>.</mo>
</math>
</div>

This is a mean restricted event-free time for observations with baseline profile
*x*. It is not an individually observed survival time. Under squared-error loss
with complete event times, this conditional mean is the optimal horizon-specific
regression function.

## Kaplan–Meier RMST functional

Within one outer training fold 𝒟, let *d*<sub>𝒟</sub>(*u*) and
*r*<sub>𝒟</sub>(*u*) be the numbers of events and observations at risk at event
time *u*. The marginal Kaplan–Meier estimator is

<div class="math-display" role="math" aria-label="Kaplan Meier survival estimator is the product over event times u less than or equal to t of one minus events divided by at risk.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mover><mi>S</mi><mo>^</mo></mover><mi>𝒟</mi></msub><mo>(</mo><mi>t</mi><mo>)</mo>
  <mo>=</mo><munder><mo>∏</mo><mrow><mi>u</mi><mo>≤</mo><mi>t</mi></mrow></munder>
  <mrow><mo>{</mo><mn>1</mn><mo>−</mo>
  <mfrac><mrow><msub><mi>d</mi><mi>𝒟</mi></msub><mo>(</mo><mi>u</mi><mo>)</mo></mrow>
  <mrow><msub><mi>r</mi><mi>𝒟</mi></msub><mo>(</mo><mi>u</mi><mo>)</mo></mrow></mfrac><mo>}</mo></mrow><mo>,</mo>
</math>
</div>

and its restricted integral is

<div class="math-display" role="math" aria-label="Estimated theta tau of D equals the integral from zero to tau of estimated survival for D.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mover><mi>θ</mi><mo>^</mo></mover><mi>τ</mi></msub><mo>(</mo><mi>𝒟</mi><mo>)</mo>
  <mo>=</mo><msubsup><mo>∫</mo><mn>0</mn><mi>τ</mi></msubsup>
  <msub><mover><mi>S</mi><mo>^</mo></mover><mi>𝒟</mi></msub><mo>(</mo><mi>t</mi><mo>)</mo><mi>d</mi><mi>t</mi><mo>.</mo>
</math>
</div>

## Delete-one pseudo-observation

Let 𝒟<sub>−*i*</sub> be the same training fold with observation *i* removed. Its
jackknife pseudo-RMST target is

<div class="math-display" role="math" aria-label="Z i at tau equals n D times estimated theta on D minus n D minus one times estimated theta on D minus i.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>Z</mi><mi>i</mi></msub><mo>(</mo><mi>τ</mi><mo>;</mo><mi>𝒟</mi><mo>)</mo>
  <mo>=</mo><msub><mi>n</mi><mi>𝒟</mi></msub>
  <msub><mover><mi>θ</mi><mo>^</mo></mover><mi>τ</mi></msub><mo>(</mo><mi>𝒟</mi><mo>)</mo>
  <mo>−</mo><mo>(</mo><msub><mi>n</mi><mi>𝒟</mi></msub><mo>−</mo><mn>1</mn><mo>)</mo>
  <msub><mover><mi>θ</mi><mo>^</mo></mover><mi>τ</mi></msub><mo>(</mo><msub><mi>𝒟</mi><mrow><mo>−</mo><mi>i</mi></mrow></msub><mo>)</mo><mo>.</mo>
</math>
</div>

The jackknife maps a censored fold-level functional to one quantitative label
per training observation. Under the marginal independent-censoring and
regularity conditions described below, its conditional expectation motivates
regression toward conditional RMST asymptotically rather than as a finite-sample
identity.

## Exact implementation

The package implements the same delete-one estimator exactly. It computes the
training event-time grid once, then evaluates leave-one-out curves in bounded
observation batches. This changes computational organization, not the target:
it is not an influence-function, subsampled, grouped or cross-fitted
approximation. The accelerated implementation is tested against the direct
leave-one-out reference.

```python
from survfm import jackknife_pseudo_rmst_fast_exact

targets = jackknife_pseudo_rmst_fast_exact(
    time_train,
    event_train,
    tau=365.0,
    clip=True,
    batch_size=256,
)
```

## Target and prediction clipping

Finite-sample pseudo-observations may fall outside the natural RMST range. The
manuscript workflow bounds targets with the projection

<div class="math-display" role="math" aria-label="Clipped Z i at tau is the projection of Z i at tau onto zero to tau, defined as the minimum of tau and the maximum of zero and u.">
<math display="block" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mover><mi>Z</mi><mo>~</mo></mover><mi>i</mi></msub><mo>(</mo><mi>τ</mi><mo>)</mo>
  <mo>=</mo><msub><mi>Π</mi><mrow><mo>[</mo><mn>0</mn><mo>,</mo><mi>τ</mi><mo>]</mo></mrow></msub>
  <mo>{</mo><msub><mi>Z</mi><mi>i</mi></msub><mo>(</mo><mi>τ</mi><mo>)</mo><mo>}</mo><mo>,</mo>
  <mspace width="1em"/>
  <msub><mi>Π</mi><mrow><mo>[</mo><mn>0</mn><mo>,</mo><mi>τ</mi><mo>]</mo></mrow></msub><mo>(</mo><mi>u</mi><mo>)</mo>
  <mo>=</mo><mi>min</mi><mo>{</mo><mi>τ</mi><mo>,</mo><mi>max</mi><mo>(</mo><mn>0</mn><mo>,</mo><mi>u</mi><mo>)</mo><mo>}</mo><mo>.</mo>
</math>
</div>

The same projection is applied separately to final predictions. These are
finite-sample stabilizations, not theoretical properties of classical
pseudo-observations.

## Assumptions and known boundary

The implemented target uses an unstratified marginal Kaplan–Meier estimator.
Its standard regression justification assumes censoring is independent of both
event time and the covariates entering the regression, together with positivity
and regularity conditions. Conditional independence of event and censoring
given covariates does **not** by itself justify this marginal construction when
censoring varies with prognostic covariates. Conditional, stratified or weighted
targets are future method development rather than hidden behavior in this API.
