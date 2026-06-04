"""CSV/JSON experiment logging."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


DEFAULT_FIELDNAMES = [
    "experiment_id",
    "model_name",
    "model_family",
    "training_mode",
    "pretrained",
    "checkpoint",
    "weight_source",
    "access_mode",
    "license_checked",
    "python_executable",
    "python_version",
    "cpu_workers",
    "model_params",
    "feature_set",
    "data_size",
    "sample_size",
    "split_type",
    "split_seed",
    "group_key",
    "train_time_sec",
    "predict_time_sec",
    "valid_mape",
    "valid_mae",
    "valid_rmse",
    "valid_wape",
    "valid_smape",
    "valid_filtered_mape",
    "valid_filtered_mape_threshold",
    "valid_filtered_mape_n_rows",
    "test_mape",
    "test_mae",
    "test_rmse",
    "test_wape",
    "test_smape",
    "test_filtered_mape",
    "test_filtered_mape_threshold",
    "test_filtered_mape_n_rows",
    "source_family_metrics",
    "notes",
]


def _serialize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=_serialize)
    return value


def append_experiment_result(path: Path, row: dict[str, Any]) -> None:
    """Append a row to experiments.csv, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    normalized = {field: _serialize(row.get(field, "")) for field in DEFAULT_FIELDNAMES}
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(normalized)
