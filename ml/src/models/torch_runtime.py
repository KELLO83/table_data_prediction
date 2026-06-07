"""Runtime acceleration helpers for PyTorch tabular models."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import math
from typing import Any

import numpy as np
from tqdm.auto import tqdm


LOGGER = logging.getLogger(__name__)

DEFAULT_TORCH_ACCELERATION_PARAMS: dict[str, Any] = {
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
}


@dataclass(frozen=True)
class CompileReport:
    enabled: bool
    attempted: bool = False
    reason: str = ""
    mode: str | None = None


@dataclass(frozen=True)
class MatmulPrecisionRequest:
    torch: Any
    precision: str | None
    device: str
    label: str


@dataclass(frozen=True)
class CompileRequest:
    torch: Any
    model: Any
    params: dict[str, Any]
    total_steps: int
    label: str


@dataclass(frozen=True)
class TorchRuntime:
    torch: Any
    model: Any
    params: dict[str, Any]


@dataclass(frozen=True)
class TabularTensors:
    categorical: np.ndarray
    numeric: np.ndarray
    target: np.ndarray | None = None


@dataclass(frozen=True)
class TrainingRun:
    runtime: TorchRuntime
    tensors: TabularTensors
    progress_label: str


@dataclass(frozen=True)
class PredictionRun:
    runtime: TorchRuntime
    tensors: TabularTensors


@dataclass(frozen=True)
class TrainingStep:
    runtime: TorchRuntime
    batch: tuple[Any, Any, Any]
    loss_fn: Any
    optimizer: Any
    scaler: Any


def is_cuda_device(torch: Any, device: str) -> bool:
    return str(device).lower().startswith("cuda") and torch.cuda.is_available()


def configure_matmul_precision(request: MatmulPrecisionRequest) -> None:
    if (
        not request.precision
        or not is_cuda_device(request.torch, request.device)
        or not hasattr(request.torch, "set_float32_matmul_precision")
    ):
        return
    previous = request.torch.get_float32_matmul_precision()
    request.torch.set_float32_matmul_precision(str(request.precision))
    LOGGER.info("%s torch matmul precision: %s -> %s", request.label, previous, request.torch.get_float32_matmul_precision())


def resolve_amp_dtype(torch: Any, dtype_name: str) -> Any:
    normalized = dtype_name.lower()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype: {dtype_name}")


def should_enable_amp(torch: Any, params: dict[str, Any]) -> bool:
    return bool(params.get("enable_amp", True)) and is_cuda_device(torch, str(params["device"]))


def autocast_context(torch: Any, params: dict[str, Any]) -> Any:
    enabled = should_enable_amp(torch, params)
    if not enabled:
        return nullcontext()
    dtype = resolve_amp_dtype(torch, str(params.get("amp_dtype", "float16")))
    return torch.autocast("cuda", dtype=dtype, enabled=True)


def make_grad_scaler(torch: Any, params: dict[str, Any]) -> Any:
    enabled = should_enable_amp(torch, params) and str(params.get("amp_dtype", "float16")).lower() not in {"bf16", "bfloat16"}
    return torch.amp.GradScaler("cuda", enabled=enabled)


def maybe_compile_model(request: CompileRequest) -> tuple[Any, CompileReport]:
    setting = request.params.get("enable_compile", "auto")
    enabled = _compile_setting_enabled(request.torch, request.params, request.total_steps)
    if not enabled:
        return request.model, CompileReport(enabled=False, reason=f"enable_compile={setting!r}, total_steps={request.total_steps}")
    if not hasattr(request.torch, "compile"):
        return request.model, CompileReport(enabled=False, attempted=False, reason="torch.compile is unavailable")

    mode = str(request.params.get("compile_mode", "reduce-overhead"))
    kwargs: dict[str, Any] = {"mode": mode}
    if request.params.get("compile_fullgraph") is not None:
        kwargs["fullgraph"] = bool(request.params["compile_fullgraph"])
    if request.params.get("compile_dynamic") is not None:
        kwargs["dynamic"] = bool(request.params["compile_dynamic"])

    try:
        compiled = request.torch.compile(request.model, **kwargs)
    except Exception:
        if bool(request.params.get("compile_strict", False)):
            raise
        LOGGER.exception("%s torch.compile failed; falling back to eager.", request.label)
        return request.model, CompileReport(enabled=False, attempted=True, reason="torch.compile failed", mode=mode)

    LOGGER.info("%s torch.compile enabled: mode=%s total_steps=%s", request.label, mode, request.total_steps)
    return compiled, CompileReport(enabled=True, attempted=True, mode=mode)


def log_amp_settings(torch: Any, params: dict[str, Any], label: str) -> None:
    enabled = should_enable_amp(torch, params)
    dtype = str(params.get("amp_dtype", "float16"))
    LOGGER.info("%s AMP mixed precision: enabled=%s dtype=%s device=%s", label, enabled, dtype, params["device"])


def total_train_steps(rows: int, batch_size: int, epochs: int) -> int:
    if rows <= 0 or batch_size <= 0 or epochs <= 0:
        return 0
    return math.ceil(rows / batch_size) * epochs


def train_tabular_regressor(run: TrainingRun) -> None:
    torch = run.runtime.torch
    model = run.runtime.model
    params = run.runtime.params
    if run.tensors.target is None:
        raise ValueError("Training target is required.")

    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(run.tensors.categorical, dtype=torch.long),
        torch.as_tensor(run.tensors.numeric, dtype=torch.float32),
        torch.as_tensor(run.tensors.target.reshape(-1, 1), dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        num_workers=int(params["num_workers"]),
        pin_memory=uses_pinned_memory(params),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]))
    loss_fn = torch.nn.MSELoss()
    scaler = make_grad_scaler(torch, params)

    model.train()
    for _ in tqdm(range(int(params["epochs"])), desc=run.progress_label):
        for batch in loader:
            _train_batch(TrainingStep(run.runtime, batch, loss_fn, optimizer, scaler))


def predict_tabular_regressor(run: PredictionRun) -> np.ndarray:
    torch = run.runtime.torch
    model = run.runtime.model
    params = run.runtime.params
    model.eval()
    batch_size = int(params["batch_size"])
    row_count = len(run.tensors.categorical)
    if row_count == 0:
        return np.array([], dtype=float)

    predictions = []

    with torch.inference_mode():
        for start in range(0, row_count, batch_size):
            batch = TabularTensors(
                categorical=run.tensors.categorical[start : start + batch_size],
                numeric=run.tensors.numeric[start : start + batch_size],
            )
            predictions.append(_predict_batch(run.runtime, batch))
    return np.concatenate(predictions)


def uses_pinned_memory(params: dict[str, Any]) -> bool:
    return bool(params["pin_memory"]) and str(params["device"]).lower().startswith("cuda")


def uses_non_blocking_transfer(params: dict[str, Any]) -> bool:
    return bool(params["non_blocking"]) and uses_pinned_memory(params)


def _compile_setting_enabled(torch: Any, params: dict[str, Any], total_steps: int) -> bool:
    setting = params.get("enable_compile", "auto")
    if isinstance(setting, str) and setting.lower() == "auto":
        min_steps = int(params.get("compile_min_steps", 200))
        return is_cuda_device(torch, str(params["device"])) and total_steps >= min_steps
    return bool(setting) and is_cuda_device(torch, str(params["device"]))


def _train_batch(step: TrainingStep) -> None:
    batch_cat, batch_num, batch_y = step.batch
    batch_tensors = TabularTensors(categorical=batch_cat, numeric=batch_num, target=batch_y)
    moved = _move_tensors(step.runtime.params, batch_tensors)

    step.optimizer.zero_grad(set_to_none=True)
    with autocast_context(step.runtime.torch, step.runtime.params):
        loss = step.loss_fn(step.runtime.model(moved.categorical, moved.numeric), moved.target)
    step.scaler.scale(loss).backward()
    step.scaler.step(step.optimizer)
    step.scaler.update()


def _predict_batch(runtime: TorchRuntime, tensors: TabularTensors) -> np.ndarray:
    torch = runtime.torch
    batch = TabularTensors(
        categorical=torch.as_tensor(tensors.categorical, dtype=torch.long, device=runtime.params["device"]),
        numeric=torch.as_tensor(tensors.numeric, dtype=torch.float32, device=runtime.params["device"]),
    )
    with autocast_context(torch, runtime.params):
        output = runtime.model(batch.categorical, batch.numeric)
    return output.detach().cpu().numpy().reshape(-1)


def _move_tensors(params: dict[str, Any], tensors: TabularTensors) -> TabularTensors:
    non_blocking = uses_non_blocking_transfer(params)
    return TabularTensors(
        categorical=tensors.categorical.to(params["device"], non_blocking=non_blocking),
        numeric=tensors.numeric.to(params["device"], non_blocking=non_blocking),
        target=tensors.target.to(params["device"], non_blocking=non_blocking) if tensors.target is not None else None,
    )
