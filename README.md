# SurvFM

[![Tests](https://github.com/yuelyu21/SurvFM/actions/workflows/test.yml/badge.svg)](https://github.com/yuelyu21/SurvFM/actions/workflows/test.yml)
[![Documentation](https://github.com/yuelyu21/SurvFM/actions/workflows/docs.yml/badge.svg)](https://yuelyu21.github.io/SurvFM/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21735310.svg)](https://doi.org/10.5281/zenodo.21735310)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Reusable tabular prediction for right-censored survival outcomes

SurvFM is a model-agnostic framework that adapts tabular prediction models to
right-censored survival outcomes. It constructs exact jackknife
pseudo-observations for restricted mean survival time (RMST) and supplies the
same horizon-specific regression target to interchangeable tabular backbones.
Backbone-specific configurations are named SurvFM-TabPFN, SurvFM-TabICL,
SurvFM-TabDPT, SurvFM-TabH2O and SurvFM-MITRA.

> **Research software.** SurvFM is intended for methodological research and
> retrospective evaluation. It has not been validated for clinical
> decision-making.

## Installation

Clone the repository and install the core package with a lightweight local
backbone:

```bash
git clone https://github.com/yuelyu21/SurvFM.git
cd SurvFM
python -m pip install -e ".[sklearn]"
```

See the [installation guide](https://yuelyu21.github.io/SurvFM/installation/)
for isolated environments and optional backbone dependencies.

## Quick start

```python
from survfm import fit_survfm

model = fit_survfm(
    X_train,
    time_train,
    event_train,
    backbone="random_forest",
    tau=365.0,
)

predicted_rmst = model.predict(X_test)
risk_score = model.predict_risk(X_test)  # negative predicted RMST
```

The default backbone is TabPFN:

```python
model = fit_survfm(
    X_train,
    time_train,
    event_train,
    backbone="tabpfn",
    tau=365.0,
)
```

Install the corresponding optional dependency first with
`python -m pip install -e ".[tabpfn]"`. The package also exposes `tabicl`,
`tabdpt`, `tabh2o`, `mitra`, `linear`, `random_forest` and
`gradient_boosting`. List the complete registry with:

```bash
survfm-rmst backbones
survfm-rmst diagnose
```

## What the estimator does

1. Validates right-censored training outcomes (`event=1`, `censored=0`).
2. Uses an explicit RMST horizon or selects a training-fold event-time
   quantile when `tau=None`.
3. Fits train-only numeric and categorical preprocessing for pandas inputs.
4. Constructs exact, memory-bounded jackknife pseudo-RMST targets.
5. Fits the selected regression backbone and returns predictions in `[0, tau]`.

The output is a horizon-specific prediction targeting covariate-conditional
RMST. It is not a full survival curve, treatment effect or deployment-ready
clinical decision rule.

## Documentation

Read the complete documentation at
[yuelyu21.github.io/SurvFM](https://yuelyu21.github.io/SurvFM/).

To build and serve it locally:

```bash
python -m pip install -e ".[docs]"
mkdocs serve --dev-addr 127.0.0.1:8000
```

## Development

```bash
python -m pip install -e ".[sklearn,dev]"
python -m pytest
python -m build
```

## Citation

The archived `v0.2.0` software release is available at
[doi:10.5281/zenodo.21735311](https://doi.org/10.5281/zenodo.21735311).
The [all-versions DOI](https://doi.org/10.5281/zenodo.21735310) resolves to the
latest archived release. Citation metadata are provided in
[`CITATION.cff`](CITATION.cff); the record will be updated with the article DOI
when available.

## License

SurvFM is released under the [MIT License](LICENSE).
