"""FT-Transformer wrapper using the tabular-transformers package."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ml.src.data import feature_registry
from ml.src.models.base import BaseModel
from ml.src.models.torch_sdpa import SdpaPatchReport, log_sdpa_backend_for_model, maybe_apply_sdpa
from ml.src.models.torch_runtime import (
    CompileReport,
    autocast_context,
    configure_matmul_precision,
    log_amp_settings,
    make_grad_scaler,
    maybe_compile_model,
    resolve_amp_dtype,
    should_enable_amp,
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
            "enable_sdpa": True,
            "enable_amp": True,
            "amp_dtype": "float16",
            "enable_compile": "auto",
            "compile_mode": "reduce-overhead",
            "compile_min_steps": 200,
            "compile_fullgraph": None,
            "compile_dynamic": None,
            "compile_strict": False,
            "matmul_precision": "high",
            "pin_memory": True,
            "non_blocking": True,
            "num_workers": 0,
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
        configure_matmul_precision(self.torch, str(self.params["matmul_precision"]), str(self.params["device"]), self.name)
        self.sdpa_patch_report = maybe_apply_sdpa(self.model, enabled=bool(self.params["enable_sdpa"]))
        LOGGER.info("%s SDPA patch: %s", self.name, self.sdpa_patch_report)
        sdpa_dtype = resolve_amp_dtype(self.torch, str(self.params["amp_dtype"])) if should_enable_amp(self.torch, self.params) else self.torch.float32
        log_sdpa_backend_for_model(
            self.model,
            batch_size=min(int(self.params["batch_size"]), len(X_train)),
            tokens=len(categories) + len(feature_registry.get_numeric_columns(self.feature_set)) + 1,
            dtype=sdpa_dtype,
            device=str(self.params["device"]),
            training=True,
            label=self.name,
        )
        total_steps = total_train_steps(len(X_train), int(self.params["batch_size"]), int(self.params["epochs"]))
        self.model, self.compile_report = maybe_compile_model(self.torch, self.model, self.params, total_steps, self.name)
        log_amp_settings(self.torch, self.params, self.name)
        self._train_torch_model(x_cat, x_num, y_train.to_numpy(dtype=np.float32))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("FTTransformerModel is not fitted")
        x_cat, x_num = self._transform(X)
        self.model.eval()
        preds = []
        batch_size = int(self.params["batch_size"])
        with self.torch.inference_mode():
            for start in range(0, len(X), batch_size):
                with autocast_context(self.torch, self.params):
                    out = self.model(
                        self.torch.as_tensor(x_cat[start : start + batch_size], dtype=self.torch.long, device=self.params["device"]),
                        self.torch.as_tensor(x_num[start : start + batch_size], dtype=self.torch.float32, device=self.params["device"]),
                    )
                preds.append(out.detach().cpu().numpy().reshape(-1))
        return np.concatenate(preds)

    def _train_torch_model(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray) -> None:
        assert self.model is not None
        dataset = self.torch.utils.data.TensorDataset(
            self.torch.as_tensor(x_cat, dtype=self.torch.long),
            self.torch.as_tensor(x_num, dtype=self.torch.float32),
            self.torch.as_tensor(y.reshape(-1, 1), dtype=self.torch.float32),
        )
        pin_memory = bool(self.params["pin_memory"]) and str(self.params["device"]).lower().startswith("cuda")
        non_blocking = bool(self.params["non_blocking"]) and pin_memory
        loader = self.torch.utils.data.DataLoader(
            dataset,
            batch_size=int(self.params["batch_size"]),
            shuffle=True,
            num_workers=int(self.params["num_workers"]),
            pin_memory=pin_memory,
        )
        optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=float(self.params["learning_rate"]))
        loss_fn = self.torch.nn.MSELoss()
        scaler = make_grad_scaler(self.torch, self.params)
        self.model.train()
        for _ in tqdm(range(int(self.params["epochs"])), desc="ft_transformer epochs"):
            for batch_cat, batch_num, batch_y in loader:
                batch_cat = batch_cat.to(self.params["device"], non_blocking=non_blocking)
                batch_num = batch_num.to(self.params["device"], non_blocking=non_blocking)
                batch_y = batch_y.to(self.params["device"], non_blocking=non_blocking)
                optimizer.zero_grad(set_to_none=True)
                with autocast_context(self.torch, self.params):
                    loss = loss_fn(self.model(batch_cat, batch_num), batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

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

