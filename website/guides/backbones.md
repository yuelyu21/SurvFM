# Backbone setup

All registered backbones solve the same pseudo-RMST regression problem. Their
dependencies, access requirements and persistence behavior differ.

| Name | Execution | Install | Important setup |
|---|---|---|---|
| `tabpfn` | Local | `.[tabpfn]` | Accept the model license and configure checkpoint access. |
| `tabicl` | Local | `.[tabicl]` | First use may download a checkpoint. |
| `tabdpt` | Local | `.[tabdpt]` | Prediction uses `n_ensembles=8` and seed `20260715` by default. |
| `tabh2o` | Hosted API | `.[tabh2o]` | Set `TABH2O_API_KEY`; data leave the local machine. |
| `mitra` | Local | `.[mitra]` | Separate AutoGluon environment recommended; direct `Mitra` output only. |
| `linear` | Local | `.[sklearn]` | Lightweight diagnostic baseline. |
| `random_forest` | Local | `.[sklearn]` | General local starting point without model access. |
| `gradient_boosting` | Local | `.[sklearn]` | General local baseline. |

## TabPFN

Follow the [official TabPFN quickstart](https://docs.priorlabs.ai/quickstart)
and [model-access guide](https://docs.priorlabs.ai/how-to-access-gated-models).
Do not place access tokens in source files or notebooks committed to version
control.

```python
model = SurvFMRMSTRegressor(
    backbone="tabpfn",
    backbone_kwargs={"device": "cuda", "model_path": "/path/to/checkpoint"},
    tau=365,
)
```

## TabICL and TabDPT

Use the [official TabICL repository](https://github.com/soda-inria/tabicl) and
[official TabDPT repository](https://github.com/layer6ai-labs/TabDPT) for
installation and checkpoint instructions.

## TabH2O

```bash
export TABH2O_API_KEY="..."
```

```python
model = SurvFMRMSTRegressor(backbone="tabh2o", tau=365)
```

The adapter reads the key only when `predict()` sends a request. The credential
value is not stored on the estimator, in its metadata or in saved model files.
Feature names are anonymized in the payload. The hosted service receives the
processed training table, pseudo-targets and test features; do not use it for
protected or restricted data without institutional approval.

## MITRA

The adapter follows the
[official AutoGluon foundation-model workflow](https://auto.gluon.ai/stable/tutorials/tabular/tabular-foundational-models.html).
It requires the direct `Mitra` base model and does not silently substitute an
AutoGluon weighted ensemble. Use AutoGluon's native artifact management.
