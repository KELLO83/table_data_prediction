"""TabNet wrapper using the official pytorch-tabnet package."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.src.data.preprocessing import build_sklearn_preprocessor
from ml.src.models.base import BaseModel
from ml.src.models.torch_sdpa import SdpaPatchReport, maybe_apply_sdpa


class TabNetModel(BaseModel):
    name = "tabnet"
    family = "transformer"

    def __init__(self, feature_set: str = "default", params: dict[str, Any] | None = None) -> None:
        params = {
            "device_name": "cuda",
            "seed": 42,
            "n_d": 32,
            "n_a": 32,
            "n_steps": 5,
            "gamma": 1.5,
            "max_epochs": 50,
            "batch_size": 2048,
            "virtual_batch_size": 256,
            "enable_sdpa": True,
            **(params or {}),
        }
        super().__init__({"feature_set": feature_set, "params": params, "training_mode": "from_scratch"})
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError as exc:
            raise RuntimeError("TabNet requires the official pytorch-tabnet package.") from exc

        model_params = {
            key: params[key]
            for key in ("device_name", "seed", "n_d", "n_a", "n_steps", "gamma")
            if key in params
        }
        self.params = params
        self.preprocessor = build_sklearn_preprocessor(feature_set)
        self.model = TabNetRegressor(**model_params)
        self.sdpa_patch_report = SdpaPatchReport.disabled("model is not initialized")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        X_train_np = _to_dense(self.preprocessor.fit_transform(X_train))
        y_train_np = y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
        eval_set = None
        if X_valid is not None and y_valid is not None:
            X_valid_np = _to_dense(self.preprocessor.transform(X_valid))
            y_valid_np = y_valid.to_numpy(dtype=np.float32).reshape(-1, 1)
            eval_set = [(X_valid_np, y_valid_np)]
        self.model.fit(
            X_train=X_train_np,
            y_train=y_train_np,
            eval_set=eval_set,
            max_epochs=int(self.params["max_epochs"]),
            batch_size=int(self.params["batch_size"]),
            virtual_batch_size=int(self.params["virtual_batch_size"]),
        )
        self.sdpa_patch_report = maybe_apply_sdpa(self.model, enabled=bool(self.params["enable_sdpa"]))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.sdpa_patch_report = maybe_apply_sdpa(self.model, enabled=bool(self.params["enable_sdpa"]))
        return np.asarray(self.model.predict(_to_dense(self.preprocessor.transform(X))), dtype=float).reshape(-1)


def _to_dense(array: Any) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype=np.float32)

