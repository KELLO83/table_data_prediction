from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ml.src.data import feature_registry
from ml.src.experiments.tabular_config import ColumnConfig


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def infer_columns(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    exclude: list[str],
    categorical: list[str],
) -> ColumnConfig:
    if target not in df.columns:
        raise ValueError(f"Target column not found: {target!r}")

    auto_exclude = [column for column in df.columns if column.startswith("Unnamed:")]
    excluded = sorted(set(exclude + auto_exclude + [target]))
    missing_exclude = sorted(set(exclude) - set(df.columns))
    if missing_exclude:
        raise ValueError(f"Excluded columns not found: {missing_exclude}")
    missing_categorical = sorted(set(categorical) - set(df.columns))
    if missing_categorical:
        raise ValueError(f"Categorical columns not found: {missing_categorical}")

    if features:
        missing_features = sorted(set(features) - set(df.columns))
        if missing_features:
            raise ValueError(f"Feature columns not found: {missing_features}")
        overlap = sorted(set(features) & set(excluded))
        if overlap:
            raise ValueError(f"Columns cannot be both features and excluded: {overlap}")
        feature_columns = list(features)
    else:
        feature_columns = [column for column in df.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No feature columns remain after exclusions.")

    explicit_categorical = set(categorical)
    categorical_columns = [
        column
        for column in feature_columns
        if column in explicit_categorical or not pd.api.types.is_numeric_dtype(df[column])
    ]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    return ColumnConfig(
        target=target,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        excluded_columns=excluded,
    )


def clean_training_frame(df: pd.DataFrame, config: ColumnConfig) -> pd.DataFrame:
    frame = df[config.feature_columns + [config.target]].copy()
    frame[config.target] = pd.to_numeric(frame[config.target], errors="coerce")
    for column in config.numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in config.categorical_columns:
        frame[column] = pd.Series(frame[column], index=frame.index).astype("string").fillna("__MISSING__")
    return frame.loc[pd.notna(frame[config.target])].reset_index(drop=True)


def register_runtime_feature_set(args: argparse.Namespace, config: ColumnConfig) -> str:
    feature_set_name = f"csv_{Path(args.csv).stem}_{args.target}_{args.seed}"
    feature_registry.register_feature_set(
        feature_set_name,
        feature_columns=config.feature_columns,
        categorical_columns=config.categorical_columns,
    )
    return feature_set_name
