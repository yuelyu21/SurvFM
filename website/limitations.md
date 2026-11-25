# Limitations and responsible use

- The implementation targets static, baseline-covariate prediction.
- It returns one RMST prediction per chosen horizon, not a coherent full curve.
- The [experimental curve examples](guides/survival-curves.md) add a separate
  Cox mapping or multi-horizon reconstruction. These require additional
  assumptions and computation; their numerical validity does not establish
  probability calibration or general comparative performance.
- The marginal Kaplan-Meier pseudo-observation construction requires stronger
  censoring assumptions than conditional independence given covariates.
- Predictions are prognostic, not causal. They do not estimate treatment
  effects or justify treatment selection.
- Grouped agreement and discrimination do not establish individual calibration
  or clinical deployment utility.
- Hosted backbones transfer processed data to third-party infrastructure.
- Foundation-model packages and checkpoints evolve; reproducibility requires
  explicit version and checkpoint pinning.

Use independent validation, censoring-support diagnostics and a prespecified
horizon before any disease-specific application. Prospective validation and
institutional review are required before clinical use.
