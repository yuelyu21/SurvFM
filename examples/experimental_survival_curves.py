"""Run an experimental RMST-to-survival extension on synthetic censored data.

From the repository root, after installing .[curves]:
    python examples/experimental_survival_curves.py --method oof-cox --output runs/cox

Use --backbone tabpfn (and install .[curves,tabpfn]) for SurvFM-TabPFN.
The default random_forest is a lightweight RFReg-RMST demonstration.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time as timer

import numpy as np
import pandas as pd
from sksurv.metrics import integrated_brier_score
from sksurv.nonparametric import CensoringDistributionEstimator
from sksurv.util import Surv

from survfm import SurvFMRMSTRegressor
from survfm.horizons import select_rmst_horizon

# Both direct execution and import from a repository checkout are supported.
if __package__:
    from .experimental_cox_mapping import OOFCoxSurvivalExtension
    from .experimental_multihorizon import MultiHorizonSurvivalExtension
else:
    from experimental_cox_mapping import OOFCoxSurvivalExtension
    from experimental_multihorizon import MultiHorizonSurvivalExtension


def evaluate_ibs(y_train, y_test, probabilities, times):
    """Report IPCW support failures explicitly; never modify patients or times.

    The additional required-G threshold is a fixed demonstration policy. A
    benchmark must choose one policy for every comparator before evaluation.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or times[0] < 0):
        raise ValueError("IBS requires at least two finite, increasing nonnegative times.")
    if (probabilities.shape != (len(y_test), len(times))
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or np.any(np.diff(probabilities, axis=1) > 1e-9)):
        raise ValueError("Expected finite, decreasing survival probabilities of shape (n_test, n_times).")
    result = {"status": "not_evaluable", "ibs": None, "error": None,
              "interval": [float(times[0]), float(times[-1])],
              "n_times": len(times), "minimum_required_g": None,
              "required_g_threshold": 1e-6}
    try:
        # Preserve the library's full-test-follow-up support requirement.
        censor = CensoringDistributionEstimator().fit(y_train)
        gy = censor.predict_proba(y_test["time"])
        gt = censor.predict_proba(times)
        required = []
        for j, t in enumerate(times):
            cases = y_test["event"] & (y_test["time"] <= t)
            required.extend(gy[cases])
            if np.any(y_test["time"] > t):
                required.append(gt[j])
        required = np.asarray(required)
        if not len(required) or not np.isfinite(required).all():
            raise ValueError("Required IPCW censoring probabilities are unavailable.")
        result["minimum_required_g"] = float(required.min())
        if required.min() <= result["required_g_threshold"]:
            raise ValueError("Required IPCW censoring probability is <= 1e-6.")
        score = float(integrated_brier_score(y_train, y_test, probabilities, times))
        if not np.isfinite(score):
            raise ValueError("The evaluator returned a nonfinite IBS.")
        result.update(status="ok", ibs=score)
    except ValueError as exc:
        result["error"] = str(exc)
    return result


def synthetic_cohort(seed=20261007, n=360):
    """Baseline covariates; independent censoring and a fixed study end at 12.

    Study-end censoring is part of data generation, never a post-hoc metric repair.
    Only observed times/event indicators are returned to the training interface.
    """
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"biomarker": rng.normal(size=n),
                      "age": rng.normal(60, 10, n),
                      "group": rng.choice(["A", "B"], size=n)})
    eta = (0.5 * x.biomarker.to_numpy() + 0.015 * (x.age.to_numpy() - 60)
           + 0.2 * (x.group.to_numpy() == "B"))
    latent = 7 * np.exp(-eta) * rng.weibull(1.5, n)
    censor = np.minimum(rng.uniform(2, 14, n), 12.0)
    return x, np.minimum(latent, censor), latent <= censor


def checkpoint_record(path):
    if path is None:
        return {"source": "backbone default", "sha256": None,
                "note": "Pin a local checkpoint for a reproducible TabPFN benchmark."}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"source": path.name, "sha256": digest.hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["oof-cox", "multihorizon"], default="oof-cox")
    parser.add_argument("--backbone", choices=["random_forest", "tabpfn"], default="random_forest")
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing directories are never overwritten.")
    parser.add_argument("--checkpoint", type=Path, help="Optional local TabPFN checkpoint, recorded by SHA256.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args(argv)
    if args.checkpoint and (args.backbone != "tabpfn" or not args.checkpoint.is_file()):
        parser.error("--checkpoint requires --backbone tabpfn and an existing local file.")
    if args.output.exists():
        parser.error("Output directory already exists; choose a new path.")

    x, time, event = synthetic_cohort(args.seed)
    # Fixed index split; data were generated iid. No test outcomes select tau.
    train, test = np.arange(240), np.arange(240, len(x))
    tau = select_rmst_horizon(time[train], event[train], quantile=0.8).tau
    times = np.arange(1, 9, dtype=float) * tau / 8
    if args.backbone == "tabpfn":
        kwargs = {"device": args.device, "n_estimators": 1, "random_state": args.seed}
        if args.checkpoint:
            kwargs["model_path"] = str(args.checkpoint.resolve())
        name = "SurvFM-TabPFN"
    else:
        kwargs = {"n_estimators": 100, "min_samples_leaf": 3,
                  "random_state": args.seed, "n_jobs": 1}
        name = "RFReg-RMST"

    def factory(horizon):
        return SurvFMRMSTRegressor(backbone=args.backbone, backbone_kwargs=dict(kwargs),
                                  tau=horizon, clip_targets=True, clip_predictions=True)

    label = name + (" + OOF Cox mapping" if args.method == "oof-cox"
                    else " + multi-horizon reconstruction")
    versions = {}
    for package in ["survfm-rmst", "numpy", "pandas", "scipy", "scikit-learn", "scikit-survival", "tabpfn"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    protocol = {"label": label, "experimental": True, "method": args.method,
                "backbone": args.backbone, "backbone_kwargs": kwargs,
                "seed": args.seed, "tau": tau, "tau_rule": "training event-time quantile 0.8",
                "times": times.tolist(), "training_rows": len(train), "test_rows": len(test),
                "clip_targets": True, "clip_predictions": True,
                "n_oof_folds": 3 if args.method == "oof-cox" else None,
                "cox_alpha": 1e-4 if args.method == "oof-cox" else None,
                "n_horizons": 8 if args.method == "multihorizon" else 1,
                "checkpoint": checkpoint_record(args.checkpoint) if args.backbone == "tabpfn" else None,
                "versions": versions, "python": platform.python_version(),
                "data": "synthetic; independent Uniform(2,14) censoring and study end at 12",
                "example_source_sha256": {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [Path(__file__),
                                 Path(__file__).with_name("experimental_cox_mapping.py"),
                                 Path(__file__).with_name("experimental_multihorizon.py")]},
                "metric_policy": "sksurv full-follow-up support; required G > 1e-6; no patient/time changes"}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    x.to_csv(args.output / "synthetic_features.csv", index_label="row_id")
    np.savez_compressed(args.output / "split_and_outcomes.npz", train_ids=train, test_ids=test,
                        time_train=time[train], event_train=event[train],
                        time_test=time[test], event_test=event[test])
    if args.method == "oof-cox":
        extension = OOFCoxSurvivalExtension(factory, tau, random_state=args.seed)
    else:
        extension = MultiHorizonSurvivalExtension(factory, tau)

    start = timer.perf_counter()
    try:
        extension.fit(x.iloc[train], time[train], event[train])
        if args.method == "multihorizon":
            reconstruction = extension.reconstruct_predictions(x.iloc[test])
            survival = reconstruction["survival"][:, 1:]
            rmst = reconstruction["rmst"][:, -1]
            np.savez_compressed(args.output / "reconstruction.npz", grid=extension.grid_,
                                native_rmst=reconstruction["rmst"],
                                survival_at_knots=reconstruction["survival"],
                                integrated_rmst=reconstruction["integrated_rmst"])
            (args.output / "solver_diagnostics.json").write_text(
                json.dumps(reconstruction["diagnostics"], indent=2) + "\n")
        else:
            survival = extension.predict_survival(x.iloc[test], times)
            rmst = extension.predict_rmst(x.iloc[test])
            np.savez_compressed(args.output / "oof_predictions.npz", train_ids=train,
                                rmst=extension.oof_rmst_, fold_ids=extension.fold_ids_)
        elapsed = timer.perf_counter() - start
        np.savez_compressed(args.output / "predictions.npz", test_ids=test, times=times,
                            survival=survival, native_rmst_at_tau=rmst, native_rmst_risk=-rmst)
        y_train = Surv.from_arrays(event[train], time[train])
        y_test = Surv.from_arrays(event[test], time[test])
        result = {"label": label, "fit_status": "ok", "n_base_fits": extension.n_base_fits_,
                  "fit_and_predict_seconds": elapsed,
                  "metrics": evaluate_ibs(y_train, y_test, survival, times)}
    except Exception as exc:
        (args.output / "result.json").write_text(json.dumps(
            {"label": label, "fit_status": "failed", "error": f"{type(exc).__name__}: {exc}"}, indent=2) + "\n")
        raise
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    print(f"Saved protocol, split, predictions and metric status to {args.output}")


if __name__ == "__main__":
    main()
