"""TabICLv2 wrapper using the official pretrained TabICLRegressor."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.src.models.base import BaseModel
from ml.src.models.torch_sdpa import SdpaPatchReport, maybe_apply_sdpa


class TabICLv2Model(BaseModel):
    name = "tabiclv2"
    family = "foundation"

    def __init__(self, feature_set: str = "default", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "checkpoint_version": "tabicl-regressor-v2-20260212.ckpt",
            "allow_auto_download": True,
            "kv_cache": False,
            "random_state": 42,
            "verbose": True,
            "enable_sdpa": True,
            **(params or {}),
        }
        super().__init__(
            {
                "feature_set": feature_set,
                "params": params,
                "training_mode": "in_context",
                "pretrained": True,
                "checkpoint": params["checkpoint_version"],
                "weight_source": "official_tabicl",
                "access_mode": "local_auto_download",
                "license_checked": False,
            }
        )
        try:
            from tabicl import TabICLRegressor
        except ImportError as exc:
            raise RuntimeError("TabICLv2 requires the official tabicl package. Install with: pip install tabicl") from exc
        constructor_params = {key: value for key, value in params.items() if key != "enable_sdpa"}
        self.model = TabICLRegressor(**constructor_params)
        self.sdpa_patch_report = SdpaPatchReport.disabled("model is not initialized")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        self.model.fit(X_train, y_train)
        self.sdpa_patch_report = maybe_apply_sdpa(
            self.model,
            enabled=bool(self.config["params"].get("enable_sdpa", True)),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.sdpa_patch_report = maybe_apply_sdpa(
            self.model,
            enabled=bool(self.config["params"].get("enable_sdpa", True)),
        )
        return np.asarray(self.model.predict(X), dtype=float)

