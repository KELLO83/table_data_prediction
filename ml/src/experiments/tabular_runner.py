from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.src.experiments.tabular_config import parse_csv_list, parse_key_value, resolve_config
from ml.src.experiments.tabular_data import clean_training_frame, infer_columns, read_csv, register_runtime_feature_set
from ml.src.experiments.tabular_models import build_model
from ml.src.experiments.tabular_results import append_result, regression_metrics, save_predictions
from ml.src.models.candidates import EXPERIMENT_MODEL_NAMES

DEFAULT_OUTPUT = Path("results/tabular_regression_experiments.csv")
DEFAULT_PREDICTION_DIR = Path("results/predictions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON experiment config.")
    parser.add_argument("--csv", type=Path, default=None, help="Input CSV file.")
    parser.add_argument("--target", default=None, help="Regression target column.")
    parser.add_argument("--model", choices=list(EXPERIMENT_MODEL_NAMES), default="lightgbm")
    parser.add_argument("--features", default="", help="Comma-separated feature columns. Defaults to all usable columns.")
    parser.add_argument("--exclude", default="", help="Comma-separated columns to exclude from features.")
    parser.add_argument("--categorical", default="", help="Comma-separated categorical feature columns.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--prediction-dir", type=Path, default=DEFAULT_PREDICTION_DIR)
    parser.add_argument("--device", default=None, help="Optional device for TabPFN, e.g. cuda or cpu.")
    parser.add_argument(
        "--model-param",
        action="append",
        type=parse_key_value,
        default=[],
        metavar="KEY=VALUE",
        help="Override model config. Example: --model-param n_estimators=100",
    )
    return parser


def main() -> None:
    row = run_experiment(build_parser().parse_args())
    print(
        f"{row['experiment_id']}: "
        f"rmse={row['rmse']:.6f}, "
        f"mae={row['mae']:.6f}, "
        f"wape={row['wape']:.6f}, "
        f"features={row['feature_count']}, "
        f"train={row['train_rows']}, valid={row['valid_rows']}"
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    args = resolve_config(args)
    params = dict(args.model_param)
    if args.device is not None:
        params["device"] = args.device

    df = read_csv(args.csv)
    config = infer_columns(
        df=df,
        target=args.target,
        features=parse_csv_list(args.features),
        exclude=parse_csv_list(args.exclude),
        categorical=parse_csv_list(args.categorical),
    )
    frame = clean_training_frame(df, config)
    train_part, valid_part = train_test_split(frame, test_size=args.test_size, random_state=args.seed)
    train_df = pd.DataFrame(train_part)
    valid_df = pd.DataFrame(valid_part)

    x_train = pd.DataFrame(train_df.loc[:, config.feature_columns]).copy()
    y_train = pd.Series(train_df.loc[:, config.target], name=config.target).copy()
    x_valid = pd.DataFrame(valid_df.loc[:, config.feature_columns]).copy()
    y_valid = pd.Series(valid_df.loc[:, config.target], name=config.target).copy()

    feature_set_name = register_runtime_feature_set(args, config)
    model = build_model(
        model_name=args.model,
        params=params,
        feature_set_name=feature_set_name,
    )

    start_train = time.perf_counter()
    model.fit(x_train, y_train, x_valid, y_valid)
    train_time = time.perf_counter() - start_train

    start_predict = time.perf_counter()
    pred = np.asarray(model.predict(x_valid), dtype=np.float64)
    predict_time = time.perf_counter() - start_predict

    metrics = regression_metrics(y_valid, pred)
    experiment_id = f"{args.csv.stem}_{args.target}_{args.model}_{len(train_df)}_seed{args.seed}"
    row: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment_id": experiment_id,
        "csv": str(args.csv),
        "target": args.target,
        "model": args.model,
        "rows": len(frame),
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "feature_count": len(config.feature_columns),
        "numeric_count": len(config.numeric_columns),
        "categorical_count": len(config.categorical_columns),
        "excluded_columns": json.dumps(config.excluded_columns, ensure_ascii=False),
        "feature_columns": json.dumps(config.feature_columns, ensure_ascii=False),
        "test_size": args.test_size,
        "seed": args.seed,
        "train_time_sec": round(train_time, 6),
        "predict_time_sec": round(predict_time, 6),
        **metrics,
        "model_params": json.dumps(params, ensure_ascii=False),
    }
    append_result(args.output, row)
    if args.save_predictions:
        save_predictions(args.prediction_dir, experiment_id, valid_df, y_valid, pred)
    return row
