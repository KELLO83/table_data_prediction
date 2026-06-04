from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ColumnConfig:
    target: str
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    excluded_columns: list[str]


def parse_key_value(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {raw}")
    key, value = raw.split("=", 1)
    value = value.strip()
    lowered = value.lower()
    if lowered in {"true", "false"}:
        parsed: Any = lowered == "true"
    elif lowered in {"none", "null"}:
        parsed = None
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
    return key.strip(), parsed


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        if args.csv is None or args.target is None:
            raise ValueError("--csv and --target are required unless --config is provided.")
        return args

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    for key in [
        "csv",
        "target",
        "model",
        "features",
        "exclude",
        "categorical",
        "test_size",
        "seed",
        "output",
        "device",
    ]:
        if key not in config:
            continue
        value = config[key]
        if key in {"csv", "output"} and value is not None:
            value = Path(value)
        if key in {"features", "exclude", "categorical"} and isinstance(value, list):
            value = ",".join(value)
        setattr(args, key, value)
    if "model_params" in config:
        args.model_param.extend(config["model_params"].items())
    if args.csv is None or args.target is None:
        raise ValueError("Config must provide csv and target, or CLI must override them.")
    return args
