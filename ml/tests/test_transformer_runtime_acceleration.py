from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pandas as pd
import torch

from ml.src.data import feature_registry
from ml.src.models.transformer.FTTransformer import FTTransformerModel


class FakeFTTransformer(torch.nn.Module):
    def __init__(
        self,
        categories: list[int],
        num_continuous: int,
        dim: int = 4,
        dim_out: int = 1,
        depth: int = 1,
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.cat_embeds = torch.nn.ModuleList([torch.nn.Embedding(category + 1, dim) for category in categories])
        self.num_continuous = num_continuous
        self.num_proj = torch.nn.Linear(num_continuous, dim) if num_continuous else None
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = torch.nn.Linear(dim, dim_out)

    def forward(self, x_categ: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        tokens = [embed(x_categ[:, index]) for index, embed in enumerate(self.cat_embeds)]
        if self.num_proj is not None:
            tokens.append(self.num_proj(x_cont))
        x = torch.stack(tokens, dim=1)
        attended, _ = self.attn(x, x, x)
        return self.head(attended.mean(dim=1))


def test_ft_transformer_runtime_acceleration_options_train_and_log_sdpa(monkeypatch, caplog) -> None:
    module = types.ModuleType("fttransformer.model")
    module.FTTransformer = FakeFTTransformer
    package = types.ModuleType("fttransformer")
    package.model = module
    monkeypatch.setitem(sys.modules, "fttransformer", package)
    monkeypatch.setitem(sys.modules, "fttransformer.model", module)

    feature_registry.register_feature_set(
        "runtime_accel_test",
        feature_columns=["cat", "num"],
        categorical_columns=["cat"],
    )
    X = pd.DataFrame({"cat": ["a", "b", "a", "c", "b", "a"], "num": [1, 2, 3, 4, 5, 6]})
    y = pd.Series(np.arange(len(X), dtype=np.float32), name="target")

    model = FTTransformerModel(
        feature_set="runtime_accel_test",
        params={
            "device": "cpu",
            "epochs": 1,
            "batch_size": 3,
            "dim": 4,
            "heads": 2,
            "enable_compile": "auto",
        },
    )

    with caplog.at_level(logging.INFO):
        model.fit(X, y)
        predictions = model.predict(X)

    assert predictions.shape == (len(X),)
    assert model.sdpa_patch_report.enabled
    assert not model.compile_report.enabled
    assert "SDPA backend: expected=math" in caplog.text
