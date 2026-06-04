"""Train one generic CSV-backed tabular regression experiment."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import sys
from pathlib import Path
from typing import Any

from ml.src.experiments.tabular_config import parse_key_value, resolve_config
from ml.src.experiments.tabular_runner import run_experiment
from ml.src.models.candidates import (
    EXPERIMENT_MODEL_NAMES,
    FOUNDATION_MODEL_NAMES,
    GBDT_MODEL_NAMES,
    NEURAL_MODEL_NAMES,
    SANITY_MODEL_NAMES,
    TRANSFORMER_MODEL_NAMES,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
stdout_reconfigure = getattr(sys.stdout, "reconfigure", None)
if callable(stdout_reconfigure):
    stdout_reconfigure(encoding="utf-8", errors="replace")
stderr_reconfigure = getattr(sys.stderr, "reconfigure", None)
if callable(stderr_reconfigure):
    stderr_reconfigure(encoding="utf-8", errors="replace")

LOGGER = logging.getLogger(__name__)

MODEL_CHOICES = list(EXPERIMENT_MODEL_NAMES)
DEFAULT_CSV = ROOT / "superconductivity" / "openml_44964_superconductivity.csv"
DEFAULT_TARGET = "critical_temp"
DEFAULT_MODEL = "lightgbm"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Tabular regression JSON config.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Input CSV path. Default: {DEFAULT_CSV}")
    parser.add_argument("--target", default=DEFAULT_TARGET, help=f"Regression target column. Default: {DEFAULT_TARGET}")
    parser.add_argument("--features", default="", help="Comma-separated feature columns.")
    parser.add_argument("--exclude", default="", help="Comma-separated columns to exclude from automatic features.")
    parser.add_argument("--categorical", default="", help="Comma-separated categorical feature columns.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/tabular_regression_experiments.csv"))
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--prediction-dir", type=Path, default=Path("results/predictions"))
    parser.add_argument("--device", default=None, help="Optional model device, e.g. cuda or cpu.")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default=None, help="CatBoost task type.")
    parser.add_argument("--devices", default="0", help="CatBoost GPU device id string.")
    parser.add_argument("--gpu-ram-part", type=float, default=0.90)
    parser.add_argument(
        "--model-param",
        action="append",
        type=parse_key_value,
        default=[],
        metavar="KEY=VALUE",
        help="Override model config. Example: --model-param n_estimators=200",
    )
    parser.add_argument("--list-models", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.list_models:
        print("candidate_gbdt:", ", ".join(GBDT_MODEL_NAMES))
        print("candidate_neural:", ", ".join(NEURAL_MODEL_NAMES))
        print("candidate_transformer:", ", ".join(TRANSFORMER_MODEL_NAMES))
        print("candidate_foundation:", ", ".join(FOUNDATION_MODEL_NAMES))
        print("supported_sanity_not_candidates:", ", ".join(SANITY_MODEL_NAMES))
        print("rule: one train.py run = one concrete model only")
        return

    _setup_logging(args)
    _validate_args(args)
    result = _run_generic_tabular_mode(args)
    LOGGER.info(
        "Finished experiment=%s valid_rmse=%.6f valid_mae=%.6f valid_wape=%.6f",
        result["experiment_id"],
        result["rmse"],
        result["mae"],
        result["wape"],
    )
    print(
        f"{result['experiment_id']}: "
        f"valid_rmse={result['rmse']:.4f}, "
        f"valid_mae={result['mae']:.4f}, "
        f"valid_wape={result['wape']:.4f}"
    )


def _validate_args(args: argparse.Namespace) -> None:
    if "," in args.model:
        raise ValueError("One train.py run must train exactly one model.")
    if args.model != "catboost" and args.task_type is not None:
        raise ValueError("--task-type is only supported for catboost.")
    if args.config is None and (args.csv is None or args.target is None):
        raise ValueError("Provide --config, or provide --csv and --target.")


def _build_model_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = dict(args.model_param)
    if args.device is not None:
        params["device"] = args.device
    if args.max_epochs is not None:
        params["max_epochs"] = args.max_epochs
        params["n_epochs"] = args.max_epochs
        if args.model in {"ft_transformer", "tab_transformer", "dcnv2"}:
            params["epochs"] = args.max_epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.model == "catboost":
        task_type = args.task_type or "GPU"
        params["task_type"] = task_type
        if task_type == "GPU":
            params.setdefault("devices", args.devices)
            params.setdefault("gpu_ram_part", args.gpu_ram_part)
    return params


def _run_generic_tabular_mode(args: argparse.Namespace) -> dict[str, Any]:
    args = resolve_config(args)
    params = _build_model_params(args)
    args.model_param = list(params.items())
    args.config = None
    return run_experiment(args)


def _setup_logging(args: argparse.Namespace) -> None:
    log_file = args.log_file
    if log_file is None:
        log_model = args.model
        log_target = args.target or "config"
        if args.config is not None and args.config.exists():
            import json

            config = json.loads(args.config.read_text(encoding="utf-8-sig"))
            log_model = str(config.get("model", log_model))
            log_target = str(config.get("target", log_target))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("results/logs") / f"{log_model}_{log_target}_seed{args.seed}_{stamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    LOGGER.info("Persistent log file: %s", log_file)


if __name__ == "__main__":
    main()
