"""TabPFN wrapper using the official pretrained TabPFNRegressor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.src.data.preprocessing import build_sklearn_preprocessor
from ml.src.models.base import BaseModel
from ml.src.models.torch_sdpa import SdpaPatchReport, maybe_apply_sdpa


PROJECT_TABPFN_TOKEN_PATH = Path(__file__).resolve().parents[4] / ".secrets" / "tabpfn_token"
DEFAULT_TABPFN_VERSION = "v3"
DEFAULT_TABPFN_CHECKPOINT = "tabpfn-v3-regressor-v3_default.ckpt"


class TabPFNModel(BaseModel):
    name = "tabpfn"
    family = "foundation"

    def __init__(self, feature_set: str = "default", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "random_state": 42,
            "enable_sdpa": True,
            **(params or {}),
        }
        version = str(params.pop("version", DEFAULT_TABPFN_VERSION))
        super().__init__(
            {
                "feature_set": feature_set,
                "params": {**params, "version": version},
                "training_mode": "pretrained_inference",
                "pretrained": True,
                "checkpoint": params.get("model_path") or params.get("checkpoint", DEFAULT_TABPFN_CHECKPOINT),
                "weight_source": "official_tabpfn",
                "access_mode": "local_checkpoint_or_token",
                "license_checked": _has_tabpfn_access(params),
            }
        )
        try:
            from tabpfn import TabPFNRegressor
        except ImportError as exc:
            raise RuntimeError("TabPFN requires the official tabpfn package. Install with: pip install tabpfn") from exc

        constructor_params = {
            key: value
            for key, value in params.items()
            if key not in {"checkpoint", "enable_sdpa", "token_path"}
        }
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_sklearn_preprocessor(feature_set)),
                ("model", TabPFNRegressor(**constructor_params)),
            ]
        )
        self.sdpa_patch_report = SdpaPatchReport.disabled("model is not initialized")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        if not _has_tabpfn_access(self.config["params"]):
            raise RuntimeError(
                "TabPFN pretrained weights require Prior Labs license access before running unattended. "
                "Set TABPFN_TOKEN, create .secrets/tabpfn_token, place the cached token under "
                "~/.cache/tabpfn/auth_token or ~/.tabpfn/token, or pass a local model_path/checkpoint."
            )
        self.pipeline.fit(X_train, y_train)
        self.sdpa_patch_report = maybe_apply_sdpa(
            self.pipeline.named_steps["model"],
            enabled=bool(self.config["params"].get("enable_sdpa", True)),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.sdpa_patch_report = maybe_apply_sdpa(
            self.pipeline.named_steps["model"],
            enabled=bool(self.config["params"].get("enable_sdpa", True)),
        )
        return np.asarray(self.pipeline.predict(X), dtype=float)


def _has_tabpfn_access(params: dict[str, Any]) -> bool:
    _load_tabpfn_token(params)
    model_path = params.get("model_path") or params.get("checkpoint")
    if model_path and str(model_path) != "auto":
        return True
    if os.environ.get("TABPFN_TOKEN"):
        return True
    token_paths = [
        Path.home() / ".cache" / "tabpfn" / "auth_token",
        Path.home() / ".tabpfn" / "token",
    ]
    return any(path.exists() and path.read_text(encoding="utf-8").strip() for path in token_paths)


def _load_tabpfn_token(params: dict[str, Any]) -> None:
    if os.environ.get("TABPFN_TOKEN"):
        return
    token_file = params.get("token_path") or os.environ.get("TABPFN_TOKEN_FILE") or PROJECT_TABPFN_TOKEN_PATH
    path = Path(token_file).expanduser()
    if not path.exists():
        return
    token = path.read_text(encoding="utf-8-sig").strip().lstrip("\ufeff")
    if token:
        os.environ["TABPFN_TOKEN"] = token

