"""TabR wrapper using the pytabkit TabR sklearn interface."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.src.data.preprocessing import build_sklearn_preprocessor
from ml.src.models.base import BaseModel
from ml.src.models.torch_sdpa import SdpaPatchReport, maybe_apply_sdpa


class TabRModel(BaseModel):
    name = "tabr"
    family = "neural_retrieval"

    def __init__(self, feature_set: str = "default", params: dict[str, Any] | None = None) -> None:
        params = {
            "device": "cuda",
            "n_epochs": 100,
            "batch_size": 2048,
            "eval_batch_size": 4096,
            "context_size": 96,
            "candidate_encoding_batch_size": 16384,
            "memory_efficient": True,
            "random_state": 42,
            "n_threads": 14,
            "verbosity": 1,
            "enable_sdpa": True,
            **(params or {}),
        }
        if "max_epochs" in params:
            params["n_epochs"] = params.pop("max_epochs")
        allowed_params = {
            "num_embeddings",
            "d_main",
            "d_multiplier",
            "encoder_n_blocks",
            "predictor_n_blocks",
            "mixer_normalization",
            "context_dropout",
            "dropout0",
            "dropout1",
            "normalization",
            "activation",
            "memory_efficient",
            "candidate_encoding_batch_size",
            "n_epochs",
            "batch_size",
            "eval_batch_size",
            "context_size",
            "freeze_contexts_after_n_epochs",
            "optimizer",
            "patience",
            "transformed_target",
            "tfms",
            "quantile_output_distribution",
            "val_metric_name",
            "add_scaling_layer",
            "scale_lr_factor",
            "use_ntp_linear",
            "linear_init_type",
            "use_ntp_encoder",
            "ls_eps",
            "device",
            "random_state",
            "n_cv",
            "n_refit",
            "n_repeats",
            "val_fraction",
            "n_threads",
            "tmp_folder",
            "verbosity",
            "calibration_method",
        }
        model_params = {key: value for key, value in params.items() if key in allowed_params}
        super().__init__({"feature_set": feature_set, "params": params, "training_mode": "from_scratch"})
        try:
            from pytabkit.models.sklearn.sklearn_interfaces import RealTabR_D_Regressor
        except ImportError as exc:
            raise RuntimeError(
                "TabR requires pytabkit with the RealTabR sklearn interface. Install with: pip install pytabkit"
            ) from exc
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_sklearn_preprocessor(feature_set)),
                ("model", RealTabR_D_Regressor(**model_params)),
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
        self.pipeline.fit(X_train, y_train)
        self.sdpa_patch_report = maybe_apply_sdpa(self.pipeline, enabled=bool(self.config["params"]["enable_sdpa"]))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.sdpa_patch_report = maybe_apply_sdpa(self.pipeline, enabled=bool(self.config["params"]["enable_sdpa"]))
        return np.asarray(self.pipeline.predict(X), dtype=float)

