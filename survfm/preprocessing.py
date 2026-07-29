"""Small preprocessing helper used by SurvFM-RMST benchmark scripts."""

from __future__ import annotations

import pandas as pd


def build_static_tabular_preprocessor(df_train: pd.DataFrame, feature_cols: list[str]):
    """Build the train-fold preprocessing pipeline used for static tabular data.

    Numeric covariates are median-imputed and standardized. Categorical
    covariates are most-frequent-imputed and one-hot encoded with unknown levels
    ignored at test time.
    """

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:  # pragma: no cover - dependency dependent
        raise ImportError("build_static_tabular_preprocessor requires scikit-learn.") from exc

    x_train = df_train[feature_cols].copy()
    num_cols = [c for c in feature_cols if c.startswith("num_")]
    cat_cols = [c for c in feature_cols if c.startswith("fac_")]
    for c in [c for c in feature_cols if c not in num_cols + cat_cols]:
        if pd.api.types.is_numeric_dtype(x_train[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot),
        ]
    )
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    if not transformers:
        raise ValueError("feature_cols must contain at least one usable column.")
    return ColumnTransformer(transformers, remainder="drop")
