"""Public SurvFM research package.

The package exposes the stable pieces needed to inspect and reuse the method:
RMST functionals, exact jackknife pseudo-RMST targets, RMST-scale metrics and
interchangeable regression backbones.
"""

from .metrics import (
    ipcw_rmst_rmse,
    normalized_rmse,
    predicted_rmst_to_risk,
    rmse,
    uno_c_from_predicted_rmst,
)
from .backbones import BackboneInfo, create_backbone, get_backbone_info, list_backbones
from .horizons import HorizonSelection, select_rmst_horizon
from .model_wrappers import (
    PseudoRMSTRegressor,
    SurvFMRMSTRegressor,
    fit_survfm,
    fit_survfm_rmst,
)
from .pseudo_rmst import (
    jackknife_pseudo_rmst,
    jackknife_pseudo_rmst_fast_exact,
    kaplan_meier_curve,
    km_rmst,
    km_survival_at,
    realized_restricted_time,
    survival_functions_to_rmst,
)

__all__ = [
    "PseudoRMSTRegressor",
    "SurvFMRMSTRegressor",
    "BackboneInfo",
    "HorizonSelection",
    "create_backbone",
    "fit_survfm",
    "fit_survfm_rmst",
    "get_backbone_info",
    "ipcw_rmst_rmse",
    "jackknife_pseudo_rmst",
    "jackknife_pseudo_rmst_fast_exact",
    "kaplan_meier_curve",
    "km_rmst",
    "km_survival_at",
    "list_backbones",
    "normalized_rmse",
    "predicted_rmst_to_risk",
    "realized_restricted_time",
    "rmse",
    "select_rmst_horizon",
    "survival_functions_to_rmst",
    "uno_c_from_predicted_rmst",
]

__version__ = "0.2.0"
