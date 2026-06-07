"""Runtime acceleration helpers for PyTorch tabular models."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import logging
import math
from typing import Any


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileReport:
    enabled: bool
    attempted: bool = False
    reason: str = ""
    mode: str | None = None


def is_cuda_device(torch: Any, device: str) -> bool:
    return str(device).lower().startswith("cuda") and torch.cuda.is_available()


def configure_matmul_precision(torch: Any, precision: str | None, device: str, label: str) -> None:
    if not precision or not is_cuda_device(torch, device) or not hasattr(torch, "set_float32_matmul_precision"):
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(str(precision))
    LOGGER.info("%s torch matmul precision: %s -> %s", label, previous, torch.get_float32_matmul_precision())


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


def maybe_compile_model(torch: Any, model: Any, params: dict[str, Any], total_steps: int, label: str) -> tuple[Any, CompileReport]:
    setting = params.get("enable_compile", "auto")
    enabled = _compile_setting_enabled(torch, params, total_steps)
    if not enabled:
        return model, CompileReport(enabled=False, reason=f"enable_compile={setting!r}, total_steps={total_steps}")
    if not hasattr(torch, "compile"):
        return model, CompileReport(enabled=False, attempted=False, reason="torch.compile is unavailable")

    mode = str(params.get("compile_mode", "reduce-overhead"))
    kwargs: dict[str, Any] = {"mode": mode}
    if params.get("compile_fullgraph") is not None:
        kwargs["fullgraph"] = bool(params["compile_fullgraph"])
    if params.get("compile_dynamic") is not None:
        kwargs["dynamic"] = bool(params["compile_dynamic"])

    try:
        compiled = torch.compile(model, **kwargs)
    except Exception:
        if bool(params.get("compile_strict", False)):
            raise
        LOGGER.exception("%s torch.compile failed; falling back to eager.", label)
        return model, CompileReport(enabled=False, attempted=True, reason="torch.compile failed", mode=mode)

    LOGGER.info("%s torch.compile enabled: mode=%s total_steps=%s", label, mode, total_steps)
    return compiled, CompileReport(enabled=True, attempted=True, mode=mode)


def log_amp_settings(torch: Any, params: dict[str, Any], label: str) -> None:
    enabled = should_enable_amp(torch, params)
    dtype = str(params.get("amp_dtype", "float16"))
    LOGGER.info("%s AMP mixed precision: enabled=%s dtype=%s device=%s", label, enabled, dtype, params["device"])


def total_train_steps(rows: int, batch_size: int, epochs: int) -> int:
    if rows <= 0 or batch_size <= 0 or epochs <= 0:
        return 0
    return math.ceil(rows / batch_size) * epochs


def _compile_setting_enabled(torch: Any, params: dict[str, Any], total_steps: int) -> bool:
    setting = params.get("enable_compile", "auto")
    if isinstance(setting, str) and setting.lower() == "auto":
        min_steps = int(params.get("compile_min_steps", 200))
        return is_cuda_device(torch, str(params["device"])) and total_steps >= min_steps
    return bool(setting) and is_cuda_device(torch, str(params["device"]))
