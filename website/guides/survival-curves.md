# Experimental survival curves and IBS

These examples let benchmark authors obtain survival probabilities from an
RMST-based regressor. They implement two **alternative experimental extensions**:

| Extension | Additional step | Base-model fits | Added assumptions |
| --- | --- | --- | --- |
| OOF Cox mapping | Fit a one-variable Cox model to out-of-fold predicted RMST | 3 OOF fits + 1 full-training fit | Proportional hazards for the score; OOF scores must transfer to a full-training refit |
| Multi-horizon reconstruction | Fit RMST at 8 horizons, then fit a constrained piecewise-linear survival curve to their integrals | 8 fits | Fixed finite time grid, piecewise-linear survival between knots, S(0)=1 |

The default SurvFM estimator still predicts one RMST value per patient. A single
RMST value is an area under a survival curve and does not uniquely determine
that curve. These examples add the modeling steps needed to obtain a curve;
they are not a claim that the unchanged estimator supplies survival probabilities.
The marginal Kaplan–Meier pseudo-target construction also retains its original
censoring assumptions. Neither extension resolves informative censoring.

## Run a complete example

Run from a clone of this repository, in a separate Python environment:

```bash
python -m pip install -e ".[curves]"
python examples/experimental_survival_curves.py --method oof-cox --output runs/cox-demo
python examples/experimental_survival_curves.py --method multihorizon --output runs/multihorizon-demo
```

Both commands use `RFReg-RMST` with synthetic data, so no clinical dataset,
foundation-model weights or hosted service is needed. These are software
demonstrations, not SurvFM-TabPFN benchmark results.
The synthetic censoring time is independently drawn from Uniform(2, 14), with
a prespecified administrative study end at 12; censoring can occur within the
evaluation interval.

For the TabPFN backbone:

```bash
python -m pip install -e ".[curves,tabpfn]"
python examples/experimental_survival_curves.py --method oof-cox --backbone tabpfn --output runs/tabpfn-cox
python examples/experimental_survival_curves.py --method multihorizon --backbone tabpfn --output runs/tabpfn-multihorizon
```

TabPFN's own package and checkpoint access requirements apply. Set `--device`
as needed. For a reproducible benchmark, pin the package version and supply
`--checkpoint /path/to/local-regressor.ckpt`; its SHA256 is saved. Without this
option, the backbone selects its default checkpoint and the example explicitly
records that its identity was not pinned. Default checkpoints can change.
The demonstration uses one TabPFN ensemble member; this is not a recommended
performance-tuned configuration or a reproduction of manuscript results.

## Use your own training and test data

The adapters live in `examples/`, separate from the stable package API. The
following code runs from the repository root; copy the example modules or make
the checkout importable when using them from another project.

```python
import numpy as np
from survfm import SurvFMRMSTRegressor
from survfm.horizons import select_rmst_horizon
from examples.experimental_cox_mapping import OOFCoxSurvivalExtension
from examples.experimental_multihorizon import MultiHorizonSurvivalExtension

# tau can instead be a clinically prespecified horizon with adequate support.
tau = select_rmst_horizon(time_train, event_train, quantile=0.8).tau

def model_factory(horizon):
    # A NEW estimator (and preprocessor) for every fold or horizon.
    return SurvFMRMSTRegressor(
        backbone="tabpfn",
        backbone_kwargs={"device": "cpu", "n_estimators": 1, "random_state": 42},
        tau=horizon,
        clip_targets=True,
        clip_predictions=True,
    )

extension = OOFCoxSurvivalExtension(model_factory, tau, n_splits=3)
# Alternative: replace the preceding line, rather than chaining the extensions.
# extension = MultiHorizonSurvivalExtension(model_factory, tau, n_horizons=8)
extension.fit(X_train, time_train, event_train)

times = np.arange(1, 9) * tau / 8
survival = extension.predict_survival(X_test, times)  # (n_test, 8)
native_rmst = extension.predict_rmst(X_test)         # (n_test,)
native_rmst_risk = -native_rmst                     # higher = shorter RMST
```

`event=1` means an observed event and `event=0` means right censoring. Use the
same time units for observed follow-up, tau and evaluation times. Pass raw
covariates as a DataFrame for fold-specific imputation and encoding. If you
preprocess outside the factory, refit that preprocessing inside each OOF fold.
The factory must return a fresh object, not a previously fitted model.

For OOF Cox mapping, each training patient receives a prediction from a model
that did not fit that patient's outcome. Targets and preprocessing are
recomputed within each inner training fold. The outer-training tau remains
fixed across inner folds. A scaler and ridge Cox model (`alpha=1e-4`) are fitted
to `-OOF_RMST / tau`; the final full-training RMST model supplies test scores.
Test outcomes are not accepted by either adapter's fitting interface. A constant
OOF score or failed fit raises an error rather than substituting another model.

For reconstruction, the horizons are `j * tau / 8`, not eight event-time
quantiles. The solver minimizes squared discrepancies between predicted RMST
and the integrals of a curve constrained to start at 1, stay in [0, 1], and
decrease with time. It uses linear interpolation between knots. Inconsistent
RMST predictions may need adjustment. `reconstruct_predictions(X_test)` returns
both the original RMST predictions and the reconstructed integrals, plus solver
diagnostics. Solver failures are explicit. There is no extrapolation past tau.
The reconstruction example rejects observed events at time zero because of its
S(0)=1 assumption, and both adapters require tau within observed training support.
Finite-grid reconstruction does not uniquely identify unrestricted survival
distributions, and monotonicity does not demonstrate calibration.

## Evaluate IBS on the same patients and time grid

```python
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv

y_train = Surv.from_arrays(event_train.astype(bool), time_train)
y_test = Surv.from_arrays(event_test.astype(bool), time_test)
ibs = integrated_brier_score(y_train, y_test, survival, times)
```

IBS requires survival probabilities at each evaluation time, not RMST values
or risk scores. Lower values are better. See the
[scikit-survival IBS documentation](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.integrated_brier_score.html).

The example uses **eight-point IBS over [tau/8, tau]**. This is a coarse
trapezoidal calculation, not a dense-grid integral beginning at zero. Another
benchmark can prespecify a different supported grid within [0, tau] and ask
either adapter for probabilities on it. Use exactly the same interval, test
patients, censoring reference and evaluator settings for every comparator;
IBS numbers from different protocols should not be compared directly.

The runnable example estimates the IPCW censoring distribution from the
training outcomes and records a score as `not_evaluable` when the library's
follow-up support requirement fails or a required censoring probability is
at most 1e-6. It saves the error and preserves the predictions. It does not
drop test patients, shorten follow-up, change the time grid or replace a failed
score with zero. This support policy is part of the demonstration protocol,
not a universal definition of IBS. Keep support failures in benchmark summaries.

For a scalar-risk C-index, use the negative native RMST and report the chosen
C-index definition and horizon. Time-dependent concordance definitions that
require curves need the probability matrix instead. If an evaluator expects
time-by-patient format, transpose `survival`; do not change its row/time labels.

## Saved outputs and benchmark reporting

Each command requires a new output directory and saves:

- `protocol.json`: method label, training-derived tau, exact evaluation times,
  seed, backbone configuration, versions, example-source hashes and supplied
  checkpoint hash.
- `synthetic_features.csv` and `split_and_outcomes.npz`: data and split IDs for
  reproducing this synthetic demonstration.
- `predictions.npz`: test IDs, evaluation times, survival probabilities, native
  RMST predictions and their negative risk scores.
- `oof_predictions.npz` for Cox mapping, or `reconstruction.npz` and solver
  diagnostics for multi-horizon reconstruction.
- `result.json`: fit status, number of base fits, fit/prediction runtime, IBS
  status and any support error.

In a benchmark, name the complete method: **SurvFM-TabPFN + OOF Cox mapping**
or **SurvFM-TabPFN + multi-horizon reconstruction**. Record the repository commit,
package/checkpoint versions, preprocessing, splits, horizon/grid, censoring
policy, runtime and all fits. Keep native RMST separate from curve-integrated
RMST: the mapping need not preserve the original predicted area.

Choose an extension and its settings using training-only validation or a
separate development set. Do not pick whichever extension wins on each final
test dataset. The earlier small feasibility pilot showed that both routes can
produce evaluable curves; it did not establish a universally preferable route
or calibrated survival probabilities. These examples support further evaluation
without changing the paper's RMST-focused contribution.
