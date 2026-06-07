"""NODE wrapper using the PyTorch Tabular implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.src.data import feature_registry
from ml.src.models.base import BaseModel
from ml.src.models.torch_sdpa import SdpaPatchReport, maybe_apply_sdpa


class NodeModel(BaseModel):
    name = "node"
    family = "neural"

    def __init__(self, feature_set: str, params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "random_state": 42,
            "max_epochs": 100,
            "batch_size": 4096,
            "learning_rate": 1e-3,
            "num_layers": 2,
            "num_trees": 1024,
            "depth": 6,
            "additional_tree_output_dim": 3,
            "choice_function": "entmax15",
            "bin_function": "entmoid15",
            "early_stopping_patience": 5,
            "progress_bar": "simple",
            "num_workers": 0,
            "pin_memory": True,
            "enable_sdpa": True,
            **(params or {}),
        }
        super().__init__({"feature_set": feature_set, "params": params, "training_mode": "from_scratch"})
        try:
            from pytorch_tabular import TabularModel
            from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
            from pytorch_tabular.models import NodeConfig
        except ImportError as exc:
            raise RuntimeError(
                "NODE requires pytorch-tabular. Install with: pip install pytorch-tabular"
            ) from exc

        self.feature_set = feature_set
        self.params = params
        self.TabularModel = TabularModel
        self.DataConfig = DataConfig
        self.OptimizerConfig = OptimizerConfig
        self.TrainerConfig = TrainerConfig
        self.NodeConfig = NodeConfig
        self.model: Any | None = None
        self.sdpa_patch_report = SdpaPatchReport.disabled("model is not initialized")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> None:
        target = y_train.name or "target"
        train_df = X_train.copy()
        train_df[target] = y_train.to_numpy(dtype=np.float32)
        valid_df = None
        if X_valid is not None and y_valid is not None:
            valid_df = X_valid.copy()
            valid_df[target] = y_valid.to_numpy(dtype=np.float32)

        data_config = self.DataConfig(
            target=[target],
            continuous_cols=feature_registry.get_numeric_columns(self.feature_set),
            categorical_cols=feature_registry.get_categorical_columns(self.feature_set),
            num_workers=int(self.params["num_workers"]),
            pin_memory=bool(self.params["pin_memory"]),
            handle_unknown_categories=True,
            handle_missing_values=True,
        )
        model_config = self.NodeConfig(
            task="regression",
            seed=int(self.params["random_state"]),
            learning_rate=float(self.params["learning_rate"]),
            num_layers=int(self.params["num_layers"]),
            num_trees=int(self.params["num_trees"]),
            depth=int(self.params["depth"]),
            additional_tree_output_dim=int(self.params["additional_tree_output_dim"]),
            choice_function=str(self.params["choice_function"]),
            bin_function=str(self.params["bin_function"]),
        )
        trainer_config = self.TrainerConfig(
            batch_size=int(self.params["batch_size"]),
            max_epochs=int(self.params["max_epochs"]),
            accelerator=_accelerator(self.params["device"]),
            devices=1,
            progress_bar=str(self.params["progress_bar"]),
            early_stopping="valid_loss" if valid_df is not None else None,
            early_stopping_patience=int(self.params["early_stopping_patience"]),
            checkpoints=None,
            seed=int(self.params["random_state"]),
            trainer_kwargs={"enable_checkpointing": False, "log_every_n_steps": 20},
        )
        optimizer_config = self.OptimizerConfig(optimizer="Adam")
        self.model = self.TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        self.model.fit(train_df, validation=valid_df)
        self.sdpa_patch_report = maybe_apply_sdpa(self.model, enabled=bool(self.params["enable_sdpa"]))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("NodeModel is not fitted")
        self.sdpa_patch_report = maybe_apply_sdpa(self.model, enabled=bool(self.params["enable_sdpa"]))
        predictions = self.model.predict(X, progress_bar=str(self.params["progress_bar"]))
        column = next((col for col in predictions.columns if col.endswith("_prediction")), "")
        if column not in predictions.columns:
            numeric_columns = predictions.select_dtypes(include=["number"]).columns
            if len(numeric_columns) == 0:
                raise RuntimeError("NODE prediction output did not contain numeric predictions")
            column = str(numeric_columns[-1])
        return predictions[column].to_numpy(dtype=float)


def _accelerator(device: str) -> str:
    lowered = device.lower()
    if lowered.startswith("cuda") or lowered.startswith("gpu"):
        return "gpu"
    return "cpu"

