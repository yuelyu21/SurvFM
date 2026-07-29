# Reproducibility

## Inspect the environment

```bash
survfm-rmst diagnose
survfm-rmst backbones
python -m pytest
```

`diagnose` reports dependency availability and whether a hosted credential is
configured, but never prints the credential value.

## Pin what matters

For a reproducible analysis, record:

- SurvFM project commit and SurvFM-RMST package version;
- Python and dependency versions;
- backbone package and checkpoint identifiers;
- device, random seed and model constructor arguments;
- feature columns and train-only preprocessing;
- event definition, split identifiers and RMST horizon rule.

`model.get_metadata()` captures the estimator-side subset of this record. It is
not a replacement for an analysis manifest.

## Manuscript versus package defaults

The manuscript freezes explicit model versions and benchmark settings. Package
defaults provide a practical interface but cannot freeze a third-party
package's future default checkpoint. Pass explicit checkpoint and version
arguments when exact manuscript reproduction is required.
