"""Smoke-test model wrappers on synthetic generic tabular data."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.src.data import feature_registry
from ml.src.models.candidates import EXPERIMENT_MODEL_NAMES
from ml.src.models.registry import create_model


DEFAULT_MODELS = list(EXPERIMENT_MODEL_NAMES)
SmokeParam: TypeAlias = bool | int | str | tuple[int, ...]

SMOKE_FEATURE_SET = "smoke_numeric"
SMOKE_FEATURES = ["f0", "f1", "f2", "f3", "category"]
SMOKE_CATEGORICAL = ["category"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--rows", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    feature_registry.register_feature_set(SMOKE_FEATURE_SET, SMOKE_FEATURES, SMOKE_CATEGORICAL)
    X, y = make_dummy(args.rows)
    valid_start = max(4, int(args.rows * 0.75))
    for model_name in args.models:
        params = smoke_params(model_name)
        try:
            model = create_model(model_name, feature_set=SMOKE_FEATURE_SET, params=params)
            model.fit(X.iloc[:valid_start], y.iloc[:valid_start], X.iloc[valid_start:], y.iloc[valid_start:])
            pred = np.asarray(model.predict(X.iloc[valid_start : valid_start + 4]), dtype=float)
            ok = pred.shape == (min(4, len(X) - valid_start),) and np.isfinite(pred).all()
            status = "PASS" if ok else "FAIL"
            detail = f"pred_shape={pred.shape}"
        except Exception as exc:  # noqa: BLE001 - smoke script should report every wrapper failure.
            status = "FAIL"
            detail = f"{type(exc).__name__}: {exc}"
            traceback.print_exc(limit=1)
        print(f"{model_name}\t{status}\t{detail}")


def make_dummy(n_rows: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "f0": rng.normal(0.0, 1.0, size=n_rows),
            "f1": rng.normal(1.0, 0.5, size=n_rows),
            "f2": rng.integers(0, 10, size=n_rows),
            "f3": np.arange(n_rows),
            "category": [f"C{idx % 3}" for idx in range(n_rows)],
        }
    )
    y = pd.Series(0.5 * X["f0"] - 0.2 * X["f1"] + 0.01 * X["f3"] + rng.normal(0.0, 0.01, size=n_rows), name="target")
    return X, y


def smoke_params(model_name: str) -> dict[str, SmokeParam]:
    params_by_model: dict[str, dict[str, SmokeParam]] = {
        "dummy_mean": {},
        "dummy_median": {},
        "ridge": {},
        "lightgbm": {"n_estimators": 3, "num_leaves": 7, "min_child_samples": 2, "n_jobs": 2},
        "catboost": {"iterations": 3, "task_type": "CPU", "thread_count": 2, "verbose": False},
        "realmlp": {"device": "cpu", "n_epochs": 1, "batch_size": 16, "verbosity": 0, "n_threads": 2},
        "tabm": {"device": "cpu", "n_epochs": 1, "batch_size": 16, "verbosity": 0, "n_threads": 2},
        "tabr": {
            "device": "cpu",
            "n_epochs": 1,
            "batch_size": 16,
            "eval_batch_size": 16,
            "context_size": 4,
            "candidate_encoding_batch_size": 16,
            "verbosity": 0,
            "n_threads": 2,
        },
        "dcnv2": {
            "device": "cpu",
            "epochs": 1,
            "batch_size": 16,
            "verbose": 0,
            "embedding_dim": 4,
            "dnn_hidden_units": (8,),
            "cross_num": 1,
        },
        "node": {
            "device": "cpu",
            "max_epochs": 1,
            "batch_size": 16,
            "num_layers": 1,
            "num_trees": 8,
            "depth": 2,
            "additional_tree_output_dim": 1,
            "progress_bar": "none",
            "early_stopping_patience": 1,
            "num_workers": 0,
            "pin_memory": False,
        },
        "ft_transformer": {"device": "cpu", "epochs": 1, "batch_size": 16, "dim": 8, "depth": 1, "heads": 2},
        "tab_transformer": {"device": "cpu", "epochs": 1, "batch_size": 16, "dim": 8, "depth": 1, "heads": 2},
        "tabnet": {
            "device_name": "cpu",
            "max_epochs": 1,
            "batch_size": 16,
            "virtual_batch_size": 4,
            "n_d": 4,
            "n_a": 4,
            "n_steps": 2,
        },
        "tabpfn": {"device": "cpu"},
        "tabiclv2": {"device": "cpu", "verbose": False, "allow_auto_download": False},
    }
    return params_by_model.get(model_name, {})


if __name__ == "__main__":
    main()
