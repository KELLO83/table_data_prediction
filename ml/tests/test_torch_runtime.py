from __future__ import annotations

import torch

from ml.src.models.torch_runtime import (
    CompileRequest,
    PredictionRun,
    TabularTensors,
    TorchRuntime,
    maybe_compile_model,
    predict_tabular_regressor,
    resolve_amp_dtype,
    should_enable_amp,
    total_train_steps,
)


def test_total_train_steps_rounds_partial_batches() -> None:
    assert total_train_steps(rows=10, batch_size=4, epochs=3) == 9


def test_should_enable_amp_requires_cuda_device() -> None:
    assert not should_enable_amp(torch, {"enable_amp": True, "device": "cpu"})


def test_resolve_amp_dtype_accepts_common_names() -> None:
    assert resolve_amp_dtype(torch, "fp16") is torch.float16
    assert resolve_amp_dtype(torch, "bfloat16") is torch.bfloat16


def test_compile_auto_stays_disabled_for_short_cpu_runs() -> None:
    model = torch.nn.Linear(2, 1)
    compiled, report = maybe_compile_model(
        CompileRequest(
            torch=torch,
            model=model,
            params={"enable_compile": "auto", "device": "cpu", "compile_min_steps": 200},
            total_steps=100,
            label="test",
        )
    )

    assert compiled is model
    assert not report.enabled


def test_predict_tabular_regressor_handles_empty_input() -> None:
    class TwoInputModel(torch.nn.Module):
        def forward(self, categorical: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
            return numeric.sum(dim=1, keepdim=True)

    predictions = predict_tabular_regressor(
        PredictionRun(
            runtime=TorchRuntime(
                torch=torch,
                model=TwoInputModel(),
                params={"device": "cpu", "batch_size": 4, "enable_amp": False},
            ),
            tensors=TabularTensors(
                categorical=torch.empty((0, 0), dtype=torch.long).numpy(),
                numeric=torch.empty((0, 2), dtype=torch.float32).numpy(),
            ),
        )
    )

    assert predictions.shape == (0,)
