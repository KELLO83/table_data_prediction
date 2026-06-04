from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


def regression_metrics(y_true: pd.Series, y_pred: npt.NDArray[np.float64]) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    abs_error = np.abs(true - y_pred)
    denom = np.maximum(np.abs(true), 1e-8)
    sum_target = np.sum(np.abs(true))
    return {
        "mae": float(np.mean(abs_error)),
        "rmse": float(math.sqrt(np.mean(np.square(true - y_pred)))),
        "mape": float(np.mean(abs_error / denom) * 100.0),
        "wape": float(np.sum(abs_error) / sum_target * 100.0) if sum_target else float("nan"),
    }


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_predictions(
    prediction_dir: Path,
    experiment_id: str,
    valid_df: pd.DataFrame,
    y_valid: pd.Series,
    pred: npt.NDArray[np.float64],
) -> None:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output = prediction_dir / f"{experiment_id}_predictions.csv"
    frame = valid_df.copy()
    frame["y_true"] = y_valid.to_numpy()
    frame["y_pred"] = pred
    frame["abs_error"] = np.abs(frame["y_true"] - frame["y_pred"])
    frame.to_csv(output, index=False, encoding="utf-8-sig")
