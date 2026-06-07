"""FT-Transformer wrapper using the tabular-transformers package."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ml.src.data import feature_registry
from ml.src.models.base import BaseModel
from ml.src.models.torch_sdpa import SdpaBackendLogSpec, SdpaPatchReport, log_sdpa_backend_for_model, maybe_apply_sdpa
from ml.src.models.torch_runtime import (
    CompileRequest,
    CompileReport,
    DEFAULT_TORCH_ACCELERATION_PARAMS,
    MatmulPrecisionRequest,
    PredictionRun,
    TabularTensors,
    TorchRuntime,
    TrainingRun,
    configure_matmul_precision,
    log_amp_settings,
    maybe_compile_model,
    predict_tabular_regressor,
    resolve_amp_dtype,
    should_enable_amp,
    train_tabular_regressor,
    total_train_steps,
)


LOGGER = logging.getLogger(__name__)


class FTTransformerModel(BaseModel):
    name = "ft_transformer"
    family = "transformer"

    def __init__(self, feature_set: str = "default", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "dim": 32,
            "depth": 4,
            "heads": 4,
            "epochs": 50,
            "batch_size": 2048,
            "learning_rate": 1e-3,
            **DEFAULT_TORCH_ACCELERATION_PARAMS,
            **(params or {}),
        }
        super().__init__({"feature_set": feature_set, "params": params, "training_mode": "from_scratch"})
        try:
            from fttransformer.model import FTTransformer
            import torch
        except ImportError as exc:
            raise RuntimeError("FT-Transformer requires tabular-transformers and torch.") from exc
        self.feature_set = feature_set
        self.params = params
        self.torch = torch
        self.model_cls = FTTransformer
        self.model: Any | None = None
        self.sdpa_patch_report = SdpaPatchReport.disabled("model is not initialized")
        self.compile_report = CompileReport(enabled=False, reason="model is not initialized")
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_stds: dict[str, float] = {}

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame | None = None, y_valid: pd.Series | None = None) -> None:
        x_cat, x_num = self._fit_transform(X_train)
        categories = [len(self.category_maps[col]) + 1 for col in feature_registry.get_categorical_columns(self.feature_set)]
        self.model = self.model_cls(
            categories=categories,
            num_continuous=len(feature_registry.get_numeric_columns(self.feature_set)),
            dim=int(self.params["dim"]),
            dim_out=1,
            depth=int(self.params["depth"]),
            heads=int(self.params["heads"]),
        ).to(self.params["device"])
        configure_matmul_precision(
            MatmulPrecisionRequest(
                torch=self.torch,
                precision=str(self.params["matmul_precision"]),
                device=str(self.params["device"]),
                label=self.name,
            )
        )
        self.sdpa_patch_report = maybe_apply_sdpa(self.model, enabled=bool(self.params["enable_sdpa"]))
        LOGGER.info("%s SDPA patch: %s", self.name, self.sdpa_patch_report)
        sdpa_dtype = resolve_amp_dtype(self.torch, str(self.params["amp_dtype"])) if should_enable_amp(self.torch, self.params) else self.torch.float32
        log_sdpa_backend_for_model(
            SdpaBackendLogSpec(
                module=self.model,
                batch_size=min(int(self.params["batch_size"]), len(X_train)),
                tokens=len(categories) + len(feature_registry.get_numeric_columns(self.feature_set)) + 1,
                dtype=sdpa_dtype,
                device=str(self.params["device"]),
                training=True,
                label=self.name,
            )
        )
        total_steps = total_train_steps(len(X_train), int(self.params["batch_size"]), int(self.params["epochs"]))
        self.model, self.compile_report = maybe_compile_model(
            CompileRequest(
                torch=self.torch,
                model=self.model,
                params=self.params,
                total_steps=total_steps,
                label=self.name,
            )
        )
        log_amp_settings(self.torch, self.params, self.name)
        self._train_torch_model(x_cat, x_num, y_train.to_numpy(dtype=np.float32))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FTTransformerModel is not fitted")
        x_cat, x_num = self._transform(X)
        return predict_tabular_regressor(
            PredictionRun(
                runtime=TorchRuntime(self.torch, self.model, self.params),
                tensors=TabularTensors(categorical=x_cat, numeric=x_num),
            )
        )

    def _train_torch_model(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> None:
        assert self.model is not None
        train_tabular_regressor(
            TrainingRun(
                runtime=TorchRuntime(self.torch, self.model, self.params),
                tensors=TabularTensors(categorical=x_cat, numeric=x_num, target=y),
                progress_label="ft_transformer epochs",
            )
        )

    def _fit_transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        for col in feature_registry.get_categorical_columns(self.feature_set):
            values = X[col].astype("string").fillna("__MISSING__").astype(str)
            self.category_maps[col] = {value: idx + 1 for idx, value in enumerate(sorted(values.unique()))}
        for col in feature_registry.get_numeric_columns(self.feature_set):
            values = pd.to_numeric(X[col], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std()) or 1.0
            self.numeric_medians[col] = median
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std
        return self._transform(X)

    def _transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        cat_cols = feature_registry.get_categorical_columns(self.feature_set)
        num_cols = feature_registry.get_numeric_columns(self.feature_set)
        x_cat = (
            np.column_stack([
                X[col].astype("string").fillna("__MISSING__").astype(str).map(self.category_maps[col]).fillna(0).to_numpy(dtype=np.int64)
                for col in cat_cols
            ])
            if cat_cols
            else np.zeros((len(X), 0), dtype=np.int64)
        )
        x_num = (
            np.column_stack([
                ((pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_medians[col]) - self.numeric_means[col]) / self.numeric_stds[col]).to_numpy(dtype=np.float32)
                for col in num_cols
            ])
            if num_cols
            else np.zeros((len(X), 0), dtype=np.float32)
        )
        return x_cat, x_num

