"""DCN-V2 wrapper using the external deepctr-torch implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.src.data import feature_registry
from ml.src.models.base import BaseModel
from ml.src.models.torch_sdpa import SdpaPatchReport, maybe_apply_sdpa


class DCNV2Model(BaseModel):
    name = "dcnv2"
    family = "neural"

    def __init__(self, feature_set: str = "default", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "random_state": 42,
            "batch_size": 4096,
            "epochs": 100,
            "cross_num": 3,
            "dnn_hidden_units": (256, 128),
            "embedding_dim": 8,
            "learning_rate": 1e-3,
            "verbose": 1,
            "enable_sdpa": True,
            **(params or {}),
        }
        if "max_epochs" in params:
            params["epochs"] = params.pop("max_epochs")
        super().__init__({"feature_set": feature_set, "params": params, "training_mode": "from_scratch"})
        try:
            from deepctr_torch.inputs import DenseFeat, SparseFeat, get_feature_names
            from deepctr_torch.models import DCNMix
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "DCN-V2 requires deepctr-torch. Install with: pip install deepctr-torch"
            ) from exc
        self.feature_set = feature_set
        self.params = params
        self.DenseFeat = DenseFeat
        self.SparseFeat = SparseFeat
        self.get_feature_names = get_feature_names
        self.model_cls = DCNMix
        self.torch = torch
        self.model: Any | None = None
        self.sdpa_patch_report = SdpaPatchReport.disabled("model is not initialized")
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numeric_medians: dict[str, float] = {}
        self.numeric_means: dict[str, float] = {}
        self.numeric_stds: dict[str, float] = {}
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        train_input = self._fit_transform(X_train)
        cat_cols = feature_registry.get_categorical_columns(self.feature_set)
        num_cols = feature_registry.get_numeric_columns(self.feature_set)
        sparse_features = [
            self.SparseFeat(col, vocabulary_size=len(self.category_maps[col]) + 1, embedding_dim=int(self.params["embedding_dim"]))
            for col in cat_cols
        ]
        dense_features = [self.DenseFeat(col, 1) for col in num_cols]
        feature_columns = sparse_features + dense_features
        self.feature_names = self.get_feature_names(feature_columns)
        self.model = self.model_cls(
            linear_feature_columns=feature_columns,
            dnn_feature_columns=feature_columns,
            cross_num=int(self.params["cross_num"]),
            dnn_hidden_units=tuple(self.params["dnn_hidden_units"]),
            task="regression",
            device=self.params["device"],
            seed=int(self.params["random_state"]),
        )
        self.sdpa_patch_report = maybe_apply_sdpa(self.model, enabled=bool(self.params["enable_sdpa"]))
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=float(self.params["learning_rate"]))
        self.model.compile(optimizer, "mse", metrics=["mse"])
        validation_data = None
        if X_valid is not None and y_valid is not None:
            validation_data = (self._transform(X_valid), y_valid.to_numpy(dtype=np.float32))
        self.model.fit(
            train_input,
            y_train.to_numpy(dtype=np.float32),
            batch_size=int(self.params["batch_size"]),
            epochs=int(self.params["epochs"]),
            verbose=int(self.params["verbose"]),
            validation_data=validation_data,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DCNV2Model is not fitted")
        self.sdpa_patch_report = maybe_apply_sdpa(self.model, enabled=bool(self.params["enable_sdpa"]))
        return np.asarray(
            self.model.predict(self._transform(X), batch_size=int(self.params["batch_size"])),
            dtype=float,
        ).reshape(-1)

    def _fit_transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
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

    def _transform(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for col in feature_registry.get_categorical_columns(self.feature_set):
            result[col] = (
                X[col]
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
                .map(self.category_maps[col])
                .fillna(0)
                .to_numpy(dtype=np.int64)
            )
        for col in feature_registry.get_numeric_columns(self.feature_set):
            values = pd.to_numeric(X[col], errors="coerce").fillna(self.numeric_medians[col])
            result[col] = ((values - self.numeric_means[col]) / self.numeric_stds[col]).to_numpy(dtype=np.float32)
        return result

